//! Neural image codec implementations.
//!
//! Provides the `ImageCodec` trait and the bmshj2018 hyperprior model.

pub mod entropy;
pub mod fast_conv;
pub mod gdn;
pub mod hyperprior;

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
pub fn compress<B: Backend + ConvDispatch>(
    model: &Hyperprior<B>,
    image: Tensor<B, 4>,
) -> Result<CompressedBlob> {
    let (y_hat, z_hat, gaussian_params) = model.encode(image);

    let [_, y_c, y_h, y_w] = y_hat.dims();
    let (means, scales) = model.split_gaussian_params(&gaussian_params);
    let [_, means_c, _, _] = means.dims();
    let [_, scales_c, _, _] = scales.dims();

    // Flatten tensors to vectors for entropy coding
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

    // Entropy-encode y with Gaussian conditional
    let y_stream = entropy::gaussian_encode(&y_symbols, &means_f64, &scales_f64)?;

    // Entropy-encode z with factorized prior
    let [_, z_c, z_h, z_w] = z_hat.dims();
    let z_data: Vec<f32> = z_hat
        .reshape([z_c * z_h * z_w])
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("z_hat conversion failed: {e:?}"))?;
    let z_symbols: Vec<i32> = z_data.iter().map(|&v| v.round() as i32).collect();

    let channel_indices: Vec<usize> = (0..z_c)
        .flat_map(|c| std::iter::repeat_n(c, z_h * z_w))
        .collect();

    // Build simple factorized prior PMF tables (Laplacian approximation as default)
    let cdf_tables: Vec<Vec<f64>> = (0..z_c)
        .map(|_| {
            let mut pmf = vec![0.0f64; 256];
            // Peaked Laplacian centered at 0 (index 128)
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

    let z_stream = entropy::factorized_encode(&z_symbols, &channel_indices, &cdf_tables)?;

    Ok(CompressedBlob {
        latent_shape: (y_h as u32, y_w as u32),
        streams: vec![y_stream, z_stream],
    })
}

/// Decompress a `CompressedBlob` back to an image tensor [1, 3, H, W].
pub fn decompress<B: Backend + ConvDispatch>(
    model: &Hyperprior<B>,
    blob: &CompressedBlob,
    device: &B::Device,
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

    let cdf_tables: Vec<Vec<f64>> = (0..model.n)
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

    let z_symbols =
        entropy::factorized_decode(z_stream, num_z_symbols, &channel_indices, &cdf_tables)?;

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

/// Create a hyperprior model for the given quality level.
pub fn create_model<B: Backend + ConvDispatch>(
    _arch: ModelArch,
    quality: u8,
    device: &B::Device,
) -> Hyperprior<B> {
    // Currently only hyperprior is implemented.
    // Other architectures can be added here.
    HyperpriorConfig::new(quality).init(device)
}
