//! Image loading, preprocessing, tensor conversion, and EXIF extraction.

use anyhow::{Context, Result};
use burn::prelude::*;
use image::{GenericImageView, ImageFormat, RgbImage};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// Result of loading and preparing an image for the codec.
pub struct PreparedImage<B: Backend> {
    /// Image tensor [1, 3, H, W] in [0, 1], padded to multiple of 64.
    pub tensor: Tensor<B, 4>,
    /// Original image width before any resizing.
    pub orig_width: u32,
    /// Original image height before any resizing.
    pub orig_height: u32,
    /// Pixels of padding added to the bottom.
    pub pad_h: u32,
    /// Pixels of padding added to the right.
    pub pad_w: u32,
    /// Image format string (e.g., "JPEG", "PNG").
    pub format: String,
    /// Original file size in bytes.
    pub file_size: u64,
}

/// Load an image, optionally downscale, and convert to a padded [1,3,H,W] tensor.
pub fn load_and_prepare<B: Backend>(
    path: &Path,
    max_dim: u32,
    device: &B::Device,
) -> Result<PreparedImage<B>> {
    let file_size = fs::metadata(path)
        .with_context(|| format!("cannot stat {}", path.display()))?
        .len();

    let img = image::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let (orig_width, orig_height) = img.dimensions();

    let format_str = ImageFormat::from_path(path)
        .map(|f| format!("{f:?}"))
        .unwrap_or_else(|_| "UNKNOWN".into());

    // Convert to RGB8
    let mut rgb = img.to_rgb8();

    // Downscale if either dimension exceeds max_dim
    let max_side = orig_width.max(orig_height);
    if max_side > max_dim {
        let scale = max_dim as f64 / max_side as f64;
        let new_w = (orig_width as f64 * scale) as u32;
        let new_h = (orig_height as f64 * scale) as u32;
        rgb = image::imageops::resize(&rgb, new_w, new_h, image::imageops::FilterType::Lanczos3);
    }

    let (w, h) = (rgb.width(), rgb.height());

    // Pad to multiple of 64 using edge replication
    let pad_h = (64 - (h % 64)) % 64;
    let pad_w = (64 - (w % 64)) % 64;
    let padded_h = h + pad_h;
    let padded_w = w + pad_w;

    // Build f32 buffer [C, H, W] with edge-replicated padding
    let mut data = vec![0.0f32; 3 * (padded_h as usize) * (padded_w as usize)];
    for c in 0..3usize {
        for py in 0..padded_h {
            for px in 0..padded_w {
                // Clamp to image bounds (edge replication)
                let sy = py.min(h - 1);
                let sx = px.min(w - 1);
                let pixel = rgb.get_pixel(sx, sy)[c];
                let idx = c * (padded_h as usize * padded_w as usize)
                    + py as usize * padded_w as usize
                    + px as usize;
                data[idx] = pixel as f32 / 255.0;
            }
        }
    }

    let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), device)
        .reshape([1, 3, padded_h as usize, padded_w as usize]);

    Ok(PreparedImage {
        tensor,
        orig_width,
        orig_height,
        pad_h,
        pad_w,
        format: format_str,
        file_size,
    })
}

/// Convert a [1, 3, H, W] tensor in [0, 1] back to an `RgbImage`.
pub fn tensor_to_image<B: Backend>(tensor: &Tensor<B, 4>) -> Result<RgbImage> {
    let [_, _, h, w] = tensor.dims();

    // Clamp to [0, 1]
    let clamped = tensor.clone().clamp(0.0, 1.0);
    let data: Vec<f32> = clamped
        .reshape([3, h * w])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("tensor conversion failed: {e:?}"))?;

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let hw = h * w;
            let r = (data[y * w + x] * 255.0).round() as u8;
            let g = (data[hw + y * w + x] * 255.0).round() as u8;
            let b = (data[2 * hw + y * w + x] * 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    Ok(img)
}

/// Remove padding from a [1, 3, H, W] tensor.
pub fn unpad<B: Backend>(tensor: &Tensor<B, 4>, pad_h: u32, pad_w: u32) -> Tensor<B, 4> {
    let [_, _, h, w] = tensor.dims();
    let new_h = h - pad_h as usize;
    let new_w = w - pad_w as usize;
    tensor.clone().slice([0..1, 0..3, 0..new_h, 0..new_w])
}

/// Compute SHA-256 hash of a file.
pub fn file_sha256(path: &Path) -> Result<String> {
    let data = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let hash = Sha256::digest(&data);
    Ok(format!("{hash:x}"))
}

/// Extract EXIF metadata as a JSON value.
pub fn extract_exif(path: &Path) -> Option<serde_json::Value> {
    let file = std::fs::File::open(path).ok()?;
    let mut bufreader = std::io::BufReader::new(&file);
    let exif = exif::Reader::new().read_from_container(&mut bufreader).ok()?;

    let mut map = serde_json::Map::new();
    for field in exif.fields() {
        let tag_name = field.tag.to_string();
        let value_str = field.display_value().with_unit(&exif).to_string();
        map.insert(tag_name, serde_json::Value::String(value_str));
    }

    if map.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(map))
    }
}

/// Recognized photo file extensions.
pub fn is_photo_extension(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "tiff" | "tif" | "bmp" | "webp")
    )
}

/// Collect all photo files in a directory (recursively).
pub fn collect_photos(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut photos = Vec::new();
    collect_photos_recursive(dir, &mut photos)?;
    photos.sort();
    Ok(photos)
}

fn collect_photos_recursive(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_photos_recursive(&path, out)?;
        } else if is_photo_extension(&path) {
            out.push(path);
        }
    }
    Ok(())
}
