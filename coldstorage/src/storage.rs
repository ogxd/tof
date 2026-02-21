//! Cold storage engine — orchestrates compression, storage, and retrieval.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::blob::CompressedBlob;
use crate::codec;
use crate::codec::fast_conv::ConvDispatch;
use crate::codec::hyperprior::Hyperprior;
use crate::config::ColdStorageConfig;
use crate::db::{Database, NewPhotoRecord, PhotoRecord};
use crate::image_utils;
use crate::metrics;

/// The main cold storage engine, generic over the Burn compute backend.
pub struct ColdStorage<B: Backend + ConvDispatch> {
    pub config: ColdStorageConfig,
    db: Database,
    model: Hyperprior<B>,
    device: B::Device,
}

impl<B: Backend + ConvDispatch> ColdStorage<B> {
    /// Initialize the storage engine: create directories, open DB, load model.
    pub fn new(config: ColdStorageConfig, device: B::Device) -> Result<Self> {
        config.validate()?;

        // Create directory structure
        fs::create_dir_all(config.latents_dir())
            .context("failed to create latents directory")?;
        fs::create_dir_all(config.models_dir())
            .context("failed to create models directory")?;

        // Open database
        let db = Database::open(&config.db_path())?;

        // Load model
        eprintln!(
            "Loading model: {} (quality={})",
            config.model,
            config.quality
        );
        let model = codec::create_model::<B>(config.model, config.quality, &device);
        eprintln!("Model loaded");

        // Save model info
        let model_info = serde_json::json!({
            "name": config.model.as_str(),
            "quality": config.quality,
            "loaded_at": chrono::Utc::now().to_rfc3339(),
        });
        let info_path = config.models_dir().join("model_info.json");
        fs::write(&info_path, serde_json::to_string_pretty(&model_info)?)
            .context("failed to write model info")?;

        Ok(Self {
            config,
            db,
            model,
            device,
        })
    }

    /// Ingest a single photo file.
    ///
    /// Returns the `PhotoRecord` on success, `None` if the photo is a duplicate.
    pub fn ingest_photo(&self, path: &Path) -> Result<Option<PhotoRecord>> {
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        // Check for duplicates by SHA-256
        let sha256 = image_utils::file_sha256(&path)?;
        if self.db.exists_by_sha256(&sha256)? {
            return Ok(None);
        }

        // Load and prepare image
        let t0 = std::time::Instant::now();
        let prepared = image_utils::load_and_prepare::<B>(&path, self.config.max_dim, &self.device)?;
        let [_, _, img_h, img_w] = prepared.tensor.dims();
        eprintln!(
            "  loaded {}x{} (padded to {}x{}) in {:.1}s",
            prepared.orig_width, prepared.orig_height,
            img_w, img_h,
            t0.elapsed().as_secs_f64()
        );

        // Compress
        eprintln!("  encoding...");
        let t1 = std::time::Instant::now();
        let blob = codec::compress(&self.model, prepared.tensor.clone())?;
        eprintln!("  compressed in {:.1}s", t1.elapsed().as_secs_f64());
        let blob_bytes = blob.to_bytes()?;
        let compressed_size = blob_bytes.len() as u64;

        // Write blob to disk
        let latent_filename = format!("{}.ncl", &sha256[..16]);
        let latent_path = self.config.latents_dir().join(&latent_filename);
        fs::write(&latent_path, &blob_bytes)
            .with_context(|| format!("failed to write latent blob to {}", latent_path.display()))?;

        let compression_ratio = if compressed_size > 0 {
            prepared.file_size as f64 / compressed_size as f64
        } else {
            f64::INFINITY
        };

        // Compute quality metrics if requested
        let psnr = if self.config.compute_metrics {
            eprintln!("  decoding for metrics...");
            let t_dec = std::time::Instant::now();
            let reconstructed = codec::decompress(&self.model, &blob, &self.device)?;
            eprintln!("  decoded in {:.1}s", t_dec.elapsed().as_secs_f64());
            // Unpad for fair comparison
            let orig_unpadded =
                image_utils::unpad(&prepared.tensor, prepared.pad_h, prepared.pad_w);
            let recon_unpadded =
                image_utils::unpad(&reconstructed, prepared.pad_h, prepared.pad_w);
            let recon_clamped = recon_unpadded.clamp(0.0, 1.0);
            Some(metrics::psnr(&orig_unpadded, &recon_clamped))
        } else {
            None
        };

        // Extract EXIF
        let exif_json = image_utils::extract_exif(&path).map(|v| v.to_string());

        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let record = NewPhotoRecord {
            original_path: path.to_string_lossy().to_string(),
            filename,
            sha256,
            width: prepared.orig_width,
            height: prepared.orig_height,
            original_size_bytes: prepared.file_size,
            format: prepared.format,
            latent_path: latent_path.to_string_lossy().to_string(),
            compressed_size_bytes: compressed_size,
            compression_ratio,
            psnr,
            ms_ssim: None, // TODO: implement MS-SSIM
            exif_json,
            model_name: self.config.model.as_str().to_string(),
            quality_level: self.config.quality,
            pad_h: prepared.pad_h,
            pad_w: prepared.pad_w,
        };

        let id = self.db.insert_photo(&record)?;

        // Re-fetch the full record to return
        self.db
            .get_photo(id)?
            .ok_or_else(|| anyhow::anyhow!("failed to retrieve just-inserted photo"))
            .map(Some)
    }

    /// Ingest all photos from a directory (recursively).
    pub fn ingest_directory(&self, dir: &Path) -> Result<IngestStats> {
        let photos = image_utils::collect_photos(dir)?;
        println!("Found {} photos to ingest", photos.len());

        let pb = ProgressBar::new(photos.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        let mut stats = IngestStats::default();

        for photo_path in &photos {
            pb.set_message(
                photo_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
            );

            match self.ingest_photo(photo_path) {
                Ok(Some(rec)) => {
                    stats.ingested += 1;
                    stats.total_original += rec.original_size_bytes;
                    stats.total_compressed += rec.compressed_size_bytes;
                }
                Ok(None) => {
                    stats.skipped += 1;
                }
                Err(e) => {
                    log::warn!("Error ingesting {}: {e:#}", photo_path.display());
                    stats.errors += 1;
                }
            }
            pb.inc(1);
        }

        pb.finish_with_message("done");
        Ok(stats)
    }

    /// Retrieve and reconstruct a photo by ID.
    pub fn retrieve_photo(&self, id: i64, output: &Path) -> Result<()> {
        let record = self
            .db
            .get_photo(id)?
            .ok_or_else(|| anyhow::anyhow!("photo ID {id} not found"))?;

        println!(
            "Retrieving: {} ({})",
            record.filename,
            format_bytes(record.original_size_bytes)
        );

        // Load compressed blob
        let blob_bytes = fs::read(&record.latent_path)
            .with_context(|| format!("failed to read {}", record.latent_path))?;
        let blob = CompressedBlob::from_bytes(&blob_bytes)?;

        // Decompress
        let reconstructed = codec::decompress(&self.model, &blob, &self.device)?;

        // Remove padding
        let unpadded = image_utils::unpad(&reconstructed, record.pad_h, record.pad_w);
        let clamped = unpadded.clamp(0.0, 1.0);

        // Convert to image and save
        let img = image_utils::tensor_to_image(&clamped)?;
        img.save(output)
            .with_context(|| format!("failed to save image to {}", output.display()))?;

        println!("Restored to: {}", output.display());
        println!("  Original:  {} bytes", record.original_size_bytes);
        println!("  Stored as: {} bytes", record.compressed_size_bytes);
        println!("  Ratio:     {:.1}x", record.compression_ratio);
        if let Some(psnr) = record.psnr {
            println!("  PSNR:      {psnr:.2} dB");
        }
        if let Some(ms_ssim) = record.ms_ssim {
            println!("  MS-SSIM:   {ms_ssim:.4}");
        }

        Ok(())
    }

    /// Show aggregated storage statistics.
    pub fn show_stats(&self) -> Result<()> {
        let records = self.db.get_all_photos()?;
        if records.is_empty() {
            println!("No photos in storage.");
            return Ok(());
        }

        let total_orig: u64 = records.iter().map(|r| r.original_size_bytes).sum();
        let total_comp: u64 = records.iter().map(|r| r.compressed_size_bytes).sum();
        let ratios: Vec<f64> = records.iter().map(|r| r.compression_ratio).collect();
        let psnr_vals: Vec<f64> = records.iter().filter_map(|r| r.psnr).collect();
        let ssim_vals: Vec<f64> = records.iter().filter_map(|r| r.ms_ssim).collect();

        println!();
        println!("{}", "=".repeat(60));
        println!("Neural Cold Storage Statistics");
        println!("{}", "=".repeat(60));
        println!("  Total photos:       {:>10}", records.len());
        println!("  Original size:      {:>10}", format_bytes(total_orig));
        println!("  Compressed size:    {:>10}", format_bytes(total_comp));
        if total_comp > 0 {
            println!(
                "  Overall ratio:      {:>10.1}x",
                total_orig as f64 / total_comp as f64
            );
            println!(
                "  Space saved:        {:>9.1}%",
                (1.0 - total_comp as f64 / total_orig as f64) * 100.0
            );
        }
        println!();
        println!(
            "  Avg compression:    {:>10.1}x",
            ratios.iter().sum::<f64>() / ratios.len() as f64
        );
        println!(
            "  Min compression:    {:>10.1}x",
            ratios.iter().cloned().fold(f64::INFINITY, f64::min)
        );
        println!(
            "  Max compression:    {:>10.1}x",
            ratios.iter().cloned().fold(0.0, f64::max)
        );

        if !psnr_vals.is_empty() {
            println!();
            println!(
                "  Avg PSNR:           {:>10.2} dB",
                psnr_vals.iter().sum::<f64>() / psnr_vals.len() as f64
            );
            println!(
                "  Min PSNR:           {:>10.2} dB",
                psnr_vals.iter().cloned().fold(f64::INFINITY, f64::min)
            );
        }

        if !ssim_vals.is_empty() {
            println!(
                "  Avg MS-SSIM:        {:>10.4}",
                ssim_vals.iter().sum::<f64>() / ssim_vals.len() as f64
            );
            println!(
                "  Min MS-SSIM:        {:>10.4}",
                ssim_vals.iter().cloned().fold(f64::INFINITY, f64::min)
            );
        }

        // Model info
        let info_path = self.config.models_dir().join("model_info.json");
        if info_path.exists() {
            if let Ok(text) = fs::read_to_string(&info_path) {
                if let Ok(info) = serde_json::from_str::<serde_json::Value>(&text) {
                    println!();
                    if let Some(name) = info.get("name").and_then(|v| v.as_str()) {
                        println!("  Model:              {name}");
                    }
                    if let Some(q) = info.get("quality").and_then(|v| v.as_u64()) {
                        println!("  Quality level:      {q}");
                    }
                }
            }
        }

        Ok(())
    }

    /// List stored photos in a table.
    pub fn list_photos(&self, limit: u32) -> Result<()> {
        let records = self.db.list_photos(limit)?;
        if records.is_empty() {
            println!("No photos in storage.");
            return Ok(());
        }

        println!(
            "{:>5} {:<30} {:>10} {:>10} {:>7} {:>8}",
            "ID", "Filename", "Orig", "Stored", "Ratio", "PSNR"
        );
        println!("{}", "-".repeat(75));

        for r in &records {
            let psnr_str = r
                .psnr
                .map(|p| format!("{p:.1} dB"))
                .unwrap_or_else(|| "N/A".into());

            // Truncate long filenames
            let name = if r.filename.len() > 28 {
                format!("{}...", &r.filename[..25])
            } else {
                r.filename.clone()
            };

            println!(
                "{:>5} {:<30} {:>8.1}MB {:>8.1}MB {:>6.1}x {:>8}",
                r.id,
                name,
                r.original_size_bytes as f64 / 1e6,
                r.compressed_size_bytes as f64 / 1e6,
                r.compression_ratio,
                psnr_str,
            );
        }

        Ok(())
    }
}

/// Statistics from an ingest operation.
#[derive(Debug, Default)]
pub struct IngestStats {
    pub ingested: u64,
    pub skipped: u64,
    pub errors: u64,
    pub total_original: u64,
    pub total_compressed: u64,
}

impl std::fmt::Display for IngestStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "Ingestion Complete")?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "  Photos ingested:   {}", self.ingested)?;
        writeln!(f, "  Skipped (dupes):   {}", self.skipped)?;
        if self.errors > 0 {
            writeln!(f, "  Errors:            {}", self.errors)?;
        }
        if self.total_original > 0 {
            writeln!(
                f,
                "  Original total:    {}",
                format_bytes(self.total_original)
            )?;
            writeln!(
                f,
                "  Compressed total:  {}",
                format_bytes(self.total_compressed)
            )?;
            let ratio = self.total_original as f64 / self.total_compressed.max(1) as f64;
            writeln!(f, "  Overall ratio:     {ratio:.1}x")?;
            let saved = (1.0 - self.total_compressed as f64 / self.total_original as f64) * 100.0;
            writeln!(f, "  Space saved:       {saved:.1}%")?;
        }
        Ok(())
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes} B")
    }
}
