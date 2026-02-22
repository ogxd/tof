//! Neural image codec implementations.
//!
//! Provides the `ImageCodec` trait and the bmshj2018 hyperprior model.

pub mod entropy;
pub mod fast_conv;
pub mod gdn;
pub mod hyperprior;
pub mod weights;

use std::path::Path;

use anyhow::Result;
use burn::prelude::*;

use crate::blob::CompressedBlob;
use crate::config::ModelArch;

use self::fast_conv::ConvDispatch;
use self::hyperprior::{Hyperprior, HyperpriorConfig};

/// Compress an image tensor using the hyperprior model.
///
/// Takes an image tensor [1, 3, H, W] in [0, 1] (H, W must be multiples of 64)
/// and returns a `CompressedBlob` containing entropy-coded latent streams.
///
/// `z_pmf_tables` and `z_offsets` define the factorized prior for hyper-latents z.
pub fn compress<B: Backend + ConvDispatch>(
    model: &Hyperprior<B>,
    image: Tensor<B, 4>,
    z_pmf_tables: &[Vec<f64>],
    z_offsets: &[i32],
) -> Result<CompressedBlob> {
    let (y_hat, z_hat, _) = model.encode(image);

    let [_, y_c, y_h, y_w] = y_hat.dims();
    let [_, z_c, z_h, z_w] = z_hat.dims();

    // Step 1: Entropy-encode z with factorized prior
    let z_data: Vec<f32> = z_hat
        .reshape([z_c * z_h * z_w])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("z_hat conversion failed: {e:?}"))?;
    let z_symbols: Vec<i32> = z_data.iter().map(|&v| v.round() as i32).collect();

    let channel_indices: Vec<usize> = (0..z_c)
        .flat_map(|c| std::iter::repeat_n(c, z_h * z_w))
        .collect();

    let z_stream = entropy::factorized_encode(&z_symbols, &channel_indices, z_pmf_tables, z_offsets)?;

    // Step 2: Decode z back to get the CLAMPED z values (matching what the decoder will see)
    let z_clamped_symbols =
        entropy::factorized_decode(&z_stream, z_symbols.len(), &channel_indices, z_pmf_tables, z_offsets)?;
    let z_clamped_data: Vec<f32> = z_clamped_symbols.iter().map(|&s| s as f32).collect();
    let z_clamped = Tensor::<B, 1>::from_floats(z_clamped_data.as_slice(), &y_hat.device())
        .reshape([1, z_c, z_h, z_w]);

    // Step 3: Compute scales from CLAMPED z (this matches what decoder will compute)
    let gaussian_params = model.h_s.forward(z_clamped);
    let (means, scales) = model.split_gaussian_params(&gaussian_params);
    let [_, means_c, _, _] = means.dims();
    let [_, scales_c, _, _] = scales.dims();

    // Step 4: Entropy-encode y using the scales derived from clamped z
    let y_flat_len = y_c * y_h * y_w;
    let y_data: Vec<f32> = y_hat
        .reshape([y_flat_len])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("y_hat conversion failed: {e:?}"))?;
    let y_symbols: Vec<i32> = y_data.iter().map(|&v| v.round() as i32).collect();

    let means_flat_len = means_c * y_h * y_w;
    let means_data: Vec<f32> = means
        .reshape([means_flat_len])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("means conversion failed: {e:?}"))?;
    let means_f64: Vec<f64> = means_data.iter().map(|&v| v as f64).collect();

    let scales_flat_len = scales_c * y_h * y_w;
    let scales_data: Vec<f32> = scales
        .reshape([scales_flat_len])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("scales conversion failed: {e:?}"))?;
    let scales_f64: Vec<f64> = scales_data.iter().map(|&v| v as f64).collect();

    let y_stream = entropy::gaussian_encode(&y_symbols, &means_f64, &scales_f64)?;

    Ok(CompressedBlob {
        latent_shape: (y_h as u32, y_w as u32),
        streams: vec![y_stream, z_stream],
    })
}

/// Decompress a `CompressedBlob` back to an image tensor [1, 3, H, W].
///
/// `z_pmf_tables` and `z_offsets` define the factorized prior for hyper-latents z.
pub fn decompress<B: Backend + ConvDispatch>(
    model: &Hyperprior<B>,
    blob: &CompressedBlob,
    device: &B::Device,
    z_pmf_tables: &[Vec<f64>],
    z_offsets: &[i32],
) -> Result<Tensor<B, 4>> {
    let (y_h, y_w) = (blob.latent_shape.0 as usize, blob.latent_shape.1 as usize);

    // Decode z first
    let z_h = y_h / 4; // Hyper-latents are 4x downsampled from latents
    let z_w = y_w / 4;

    let z_stream = blob
        .streams
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("missing z stream in blob"))?;

    let num_z_symbols = model.n * z_h * z_w;
    let channel_indices: Vec<usize> = (0..model.n)
        .flat_map(|c| std::iter::repeat_n(c, z_h * z_w))
        .collect();

    let z_symbols =
        entropy::factorized_decode(z_stream, num_z_symbols, &channel_indices, z_pmf_tables, z_offsets)?;

    // Reconstruct z tensor
    let z_data: Vec<f32> = z_symbols.iter().map(|&s| s as f32).collect();
    let z_hat = Tensor::<B, 1>::from_floats(z_data.as_slice(), device)
        .reshape([1, model.n, z_h, z_w]);

    // Get Gaussian parameters from hyper-synthesis
    let gaussian_params = model.h_s.forward(z_hat);
    let (means, scales) = model.split_gaussian_params(&gaussian_params);

    // Decode y
    let y_stream = blob
        .streams
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing y stream in blob"))?;

    let num_y_symbols = model.m * y_h * y_w;

    let means_data: Vec<f32> = means
        .reshape([model.m * y_h * y_w])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("means conversion failed: {e:?}"))?;
    let means_f64: Vec<f64> = means_data.iter().map(|&v| v as f64).collect();

    let scales_data: Vec<f32> = scales
        .reshape([model.m * y_h * y_w])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("scales conversion failed: {e:?}"))?;
    let scales_f64: Vec<f64> = scales_data.iter().map(|&v| v as f64).collect();

    let y_symbols = entropy::gaussian_decode(y_stream, num_y_symbols, &means_f64, &scales_f64)?;

    // Reconstruct y tensor
    let y_data: Vec<f32> = y_symbols.iter().map(|&s| s as f32).collect();
    let y_hat = Tensor::<B, 1>::from_floats(y_data.as_slice(), device)
        .reshape([1, model.m, y_h, y_w]);

    // Synthesis transform: y_hat → reconstructed image
    Ok(model.decode(y_hat))
}

/// Create a hyperprior model, optionally loading pretrained weights.
///
/// Returns the model along with entropy bottleneck PMF tables and offsets for z coding.
/// If `weights_dir` is None, uses random init with a Laplacian fallback prior.
pub fn create_model<B: Backend + ConvDispatch>(
    _arch: ModelArch,
    quality: u8,
    weights_dir: Option<&Path>,
    device: &B::Device,
) -> Result<(Hyperprior<B>, Vec<Vec<f64>>, Vec<i32>)> {
    if let Some(dir) = weights_dir {
        weights::load_pretrained(dir, quality, device)
    } else {
        let model = HyperpriorConfig::new(quality).init(device);
        let n = model.n;
        // Fallback: Laplacian PMF tables for random init
        let pmf_tables: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let mut pmf = vec![0.0f64; 256];
                for (i, p) in pmf.iter_mut().enumerate() {
                    let val = i as f64 - 128.0;
                    *p = (-val.abs() / 2.0).exp();
                }
                let sum: f64 = pmf.iter().sum();
                for p in &mut pmf {
                    *p /= sum;
                    *p = p.max(1e-10);
                }
                pmf
            })
            .collect();
        let offsets: Vec<i32> = vec![-128; n];
        Ok((model, pmf_tables, offsets))
    }
}
