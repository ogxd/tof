//! Fine-tuning: rate-distortion training loop for the hyperprior model.
//!
//! Uses Burn's autodiff backend to optimize the encoder/decoder transforms
//! (g_a, g_s, h_a, h_s) on a photo corpus stored in the vault.
//!
//! The loss function is:
//!   L = Distortion + λ · Rate
//!
//! where:
//!   - Distortion = MSE(x, x̂)
//!   - Rate ≈ −(1/N) · Σ log₂ p(ŷᵢ | σᵢ) − (1/N) · Σ log₂ p(ẑⱼ)
//!
//! During training, rounding is replaced with additive uniform noise (STE).

use std::path::Path;

use anyhow::{Context, Result};
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::codec;
use crate::codec::hyperprior::Hyperprior;
use crate::config::ModelArch;
use crate::image_utils;

/// The training backend: NdArray with automatic differentiation.
type TrainBackend = Autodiff<NdArray>;

/// Configuration for fine-tuning.
#[derive(Debug, Clone)]
pub struct FinetuneConfig {
    pub epochs: u32,
    pub lr: f64,
    pub lambda: f64,
    pub max_dim: u32,
    pub quality: u8,
    pub model_arch: ModelArch,
}

// ── Differentiable rate estimation ──────────────────────────────────────────

/// Approximate the standard normal CDF using the GELU-style tanh approximation.
///
///   Φ(x) ≈ 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
///
/// This is smooth, differentiable everywhere, and accurate to ~10⁻⁴.
fn normal_cdf<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let sqrt_2_over_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
    let inner =
        (x.clone() + x.clone().powf_scalar(3.0).mul_scalar(0.044715)).mul_scalar(sqrt_2_over_pi);
    (inner.tanh() + 1.0).mul_scalar(0.5)
}

/// Estimate bits per pixel for latents y under a zero-mean Gaussian conditional.
///
/// For each latent value yᵢ with predicted scale σᵢ:
///   p(yᵢ) = Φ((yᵢ + 0.5) / σᵢ) − Φ((yᵢ − 0.5) / σᵢ)
///   bits = −log₂(p(yᵢ))
///
/// Returns total rate in bits per pixel (normalized by the number of image pixels).
fn gaussian_rate<B: Backend>(
    y: Tensor<B, 4>,
    scales: Tensor<B, 4>,
    num_pixels: usize,
) -> Tensor<B, 1> {
    let scales = scales.clamp(0.01, 1e5);

    let upper = (y.clone() + 0.5) / scales.clone();
    let lower = (y - 0.5) / scales;

    let cdf_upper = normal_cdf(upper);
    let cdf_lower = normal_cdf(lower);

    // Probability of each quantization bin, clamped for numerical stability
    let prob = (cdf_upper - cdf_lower).clamp(1e-9, 1.0);

    // Bits = -log2(prob), summed and normalized by pixel count
    let bits = prob.log().mul_scalar(-1.0 / (2.0f64.ln()));
    let total_bits = bits.sum();
    total_bits.reshape([1]).div_scalar(num_pixels as f64)
}

/// Estimate bits per pixel for hyper-latents z under a Laplacian prior.
///
/// Uses the logistic function as a smooth approximation to the Laplacian CDF.
fn laplacian_rate<B: Backend>(z: Tensor<B, 4>, num_pixels: usize) -> Tensor<B, 1> {
    let upper = z.clone() + 0.5;
    let lower = z - 0.5;

    // logistic(x * 1.8) ≈ Laplacian CDF(x) with scale b=1
    let scale_factor = 1.8;
    let cdf_upper = burn::tensor::activation::sigmoid(upper.mul_scalar(scale_factor));
    let cdf_lower = burn::tensor::activation::sigmoid(lower.mul_scalar(scale_factor));

    let prob = (cdf_upper - cdf_lower).clamp(1e-9, 1.0);
    let bits = prob.log().mul_scalar(-1.0 / (2.0f64.ln()));
    let total_bits = bits.sum();
    total_bits.reshape([1]).div_scalar(num_pixels as f64)
}

/// Compute the rate-distortion loss.
///
/// Returns (differentiable_loss, distortion_mse, rate_bpp).
fn rate_distortion_loss(
    model: &Hyperprior<TrainBackend>,
    image: Tensor<TrainBackend, 4>,
    lambda: f64,
) -> (Tensor<TrainBackend, 1>, f64, f64) {
    let [_, _, h, w] = image.dims();
    let num_pixels = h * w;

    // Forward pass (training mode: uniform noise instead of rounding)
    let (x_hat, y, z, scales_hat) = model.forward(image.clone());

    // Distortion: MSE
    let diff = image - x_hat;
    let mse = (diff.clone() * diff).mean();

    // Rate estimation
    let (_, scales) = model.split_gaussian_params(&scales_hat);
    let rate_y = gaussian_rate(y, scales, num_pixels);
    let rate_z = laplacian_rate(z, num_pixels);
    let rate = rate_y + rate_z;

    // Combined R-D loss: D + λ·R
    let loss = mse.clone() + rate.clone().mul_scalar(lambda);

    let mse_val: f64 = mse.clone().into_scalar().elem();
    let rate_val: f64 = rate.into_scalar().elem();

    (loss.reshape([1]), mse_val, rate_val)
}

/// Load all photos from the vault as training tensors on the NdArray backend.
fn load_training_data(
    storage_dir: &Path,
    max_dim: u32,
    device: &<NdArray as Backend>::Device,
) -> Result<Vec<Tensor<NdArray, 4>>> {
    // Find all original photos referenced in the catalog
    let db_path = storage_dir.join("catalog.db");
    let db = crate::db::Database::open(&db_path)?;
    let records = db.get_all_photos()?;

    if records.is_empty() {
        anyhow::bail!("no photos in vault — ingest some photos first");
    }

    let mut tensors = Vec::new();
    let pb = ProgressBar::new(records.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} Loading [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    for record in &records {
        let path = std::path::Path::new(&record.original_path);
        if !path.exists() {
            pb.set_message(format!("SKIP (missing): {}", record.filename));
            pb.inc(1);
            continue;
        }

        pb.set_message(record.filename.clone());
        match image_utils::load_and_prepare::<NdArray>(path, max_dim, device) {
            Ok(prepared) => tensors.push(prepared.tensor),
            Err(e) => {
                log::warn!("Failed to load {}: {e:#}", record.filename);
            }
        }
        pb.inc(1);
    }
    pb.finish_with_message("done");

    if tensors.is_empty() {
        anyhow::bail!(
            "no loadable photos found — original files may have been moved or deleted"
        );
    }

    eprintln!("Loaded {} photos for training", tensors.len());
    Ok(tensors)
}

/// Fine-tune the hyperprior model on the photo corpus in the vault.
///
/// Loads the pretrained model directly onto the `Autodiff<NdArray>` training
/// backend, trains it, and returns the fine-tuned model on plain `NdArray`.
///
/// The encoder/decoder transforms (g_a, g_s, h_a, h_s) are optimized
/// while the entropy model parameters (PMF tables) stay frozen.
pub fn finetune(
    config: &FinetuneConfig,
    weights_dir: Option<&Path>,
    storage_dir: &Path,
) -> Result<Hyperprior<NdArray>> {
    let nd_device: <NdArray as Backend>::Device = Default::default();
    let ad_device: <TrainBackend as Backend>::Device = Default::default();

    // Load training data on NdArray backend (lighter memory footprint)
    let images = load_training_data(storage_dir, config.max_dim, &nd_device)
        .context("failed to load training corpus")?;

    // Load pretrained model directly onto the training backend (Autodiff<NdArray>)
    eprintln!("Loading model onto training backend...");
    let (mut ad_model, _pmf_tables, _offsets) = codec::create_model::<TrainBackend>(
        config.model_arch,
        config.quality,
        weights_dir,
        &ad_device,
    )?;

    // Initialize Adam optimizer
    let mut optim = AdamConfig::new()
        .with_epsilon(1e-8)
        .init::<TrainBackend, Hyperprior<TrainBackend>>();

    eprintln!();
    eprintln!(
        "Fine-tuning: {} epochs, lr={}, λ={}",
        config.epochs, config.lr, config.lambda
    );
    eprintln!("  Corpus: {} images", images.len());
    eprintln!("  Note: training uses tensor im2col + BLAS matmul (autodiff-compatible).");
    eprintln!();

    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0u32;

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_distortion = 0.0;
        let mut epoch_rate = 0.0;
        let mut count = 0;

        let pb = ProgressBar::new(images.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "  Epoch {}/{} [{{bar:30.cyan/blue}}] {{pos}}/{{len}} {{msg}}",
                    epoch + 1,
                    config.epochs
                ))
                .unwrap()
                .progress_chars("=>-"),
        );

        for image in &images {
            // Convert NdArray tensor → Autodiff<NdArray> tensor
            let ad_image: Tensor<TrainBackend, 4> =
                Tensor::from_data(image.to_data(), &ad_device);

            // Forward + loss
            let (loss, mse_val, rate_val) =
                rate_distortion_loss(&ad_model, ad_image, config.lambda);

            // Backward
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &ad_model);

            // Optimizer step (consumes model, returns updated model)
            ad_model = optim.step(config.lr, ad_model, grads);

            let psnr = if mse_val > 0.0 {
                10.0 * (1.0 / mse_val).log10()
            } else {
                f64::INFINITY
            };

            epoch_loss += mse_val + config.lambda * rate_val;
            epoch_distortion += mse_val;
            epoch_rate += rate_val;
            count += 1;

            pb.set_message(format!(
                "loss={:.4} PSNR={:.1}dB rate={:.2}bpp",
                mse_val + config.lambda * rate_val,
                psnr,
                rate_val,
            ));
            pb.inc(1);
        }

        pb.finish_and_clear();

        if count > 0 {
            let avg_loss = epoch_loss / count as f64;
            let avg_dist = epoch_distortion / count as f64;
            let avg_rate = epoch_rate / count as f64;
            let avg_psnr = if avg_dist > 0.0 {
                10.0 * (1.0 / avg_dist).log10()
            } else {
                f64::INFINITY
            };

            let marker = if avg_loss < best_loss {
                best_loss = avg_loss;
                best_epoch = epoch;
                " ★"
            } else {
                ""
            };

            eprintln!(
                "  Epoch {:>3}/{}: loss={:.6} PSNR={:.2}dB rate={:.3}bpp{}",
                epoch + 1,
                config.epochs,
                avg_loss,
                avg_psnr,
                avg_rate,
                marker,
            );
        }
    }

    eprintln!();
    eprintln!(
        "Training complete. Best loss at epoch {} ({:.6})",
        best_epoch + 1,
        best_loss,
    );

    // Convert back to inference backend: Autodiff<NdArray> → NdArray
    let trained_model: Hyperprior<NdArray> = ad_model.valid();
    Ok(trained_model)
}
