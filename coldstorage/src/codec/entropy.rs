//! Entropy coding for learned image compression.
//!
//! Provides encoding and decoding of quantized latent tensors using
//! the `constriction` crate for range/ANS coding.
//!
//! Two entropy models are used:
//!
//! 1. **Factorized prior** — for hyper-latents `z`. Each channel has a
//!    pre-computed CDF table derived from the learned density model.
//!
//! 2. **Gaussian conditional** — for latents `y`. The hyper-decoder
//!    predicts per-element `(mean, scale)` parameters defining a Gaussian
//!    distribution, whose quantized CDF is used for coding.

use std::collections::HashMap;

use anyhow::{bail, Result};
use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::stack::DefaultAnsCoder;
use constriction::stream::{Decode, Encode};

/// Quantize a (mean, scale) pair into a cache key.
///
/// We bucket mean to the nearest 0.5 and scale to the nearest 0.25.
/// This lets us reuse PMF tables across symbols with similar parameters,
/// reducing millions of PMF computations to thousands.
fn cache_key(mean: f64, scale: f64) -> (i32, i32) {
    let m = (mean * 2.0).round() as i32; // bucket to 0.5
    let s = (scale * 4.0).round().max(1.0) as i32; // bucket to 0.25, min 0.25
    (m, s)
}

/// Build or retrieve a cached entropy model for a given (mean, scale).
fn get_or_build_model(
    cache: &mut HashMap<(i32, i32), DefaultContiguousCategoricalEntropyModel>,
    mean: f64,
    scale: f64,
) -> Result<DefaultContiguousCategoricalEntropyModel> {
    let key = cache_key(mean, scale);
    if let Some(model) = cache.get(&key) {
        return Ok(model.clone());
    }
    // Reconstruct the quantized mean/scale from the key
    let q_mean = key.0 as f64 / 2.0;
    let q_scale = (key.1 as f64 / 4.0).max(0.01);
    let pmf = quantized_gaussian_pmf(q_mean, q_scale, -128, 127);
    let model = DefaultContiguousCategoricalEntropyModel
        ::from_floating_point_probabilities_fast(&pmf, None)
        .map_err(|_| anyhow::anyhow!("failed to build entropy model for mean={q_mean}, scale={q_scale}"))?;
    cache.insert(key, model.clone());
    Ok(model)
}

/// Entropy-encode a vector of integer symbols using per-symbol Gaussian parameters.
///
/// Each symbol `symbols[i]` is coded under a quantized Gaussian(means[i], scales[i]).
/// Returns the compressed byte stream.
pub fn gaussian_encode(symbols: &[i32], means: &[f64], scales: &[f64]) -> Result<Vec<u8>> {
    assert_eq!(symbols.len(), means.len());
    assert_eq!(symbols.len(), scales.len());

    let mut coder = DefaultAnsCoder::new();
    let mut model_cache = HashMap::new();

    // Encode in reverse order (ANS is a stack / LIFO)
    for i in (0..symbols.len()).rev() {
        let model = get_or_build_model(&mut model_cache, means[i], scales[i])?;
        let sym = (symbols[i] + 128).clamp(0, 255) as usize;
        coder
            .encode_symbol(sym, model)
            .map_err(|e| anyhow::anyhow!("failed to encode symbol {i}: {e:?}"))?;
    }

    log::debug!("gaussian_encode: {} symbols, {} cached models", symbols.len(), model_cache.len());

    let compressed = coder.into_compressed().unwrap_or_default();
    Ok(compressed
        .into_iter()
        .flat_map(|w| w.to_le_bytes())
        .collect())
}

/// Entropy-decode symbols using per-symbol Gaussian parameters.
///
/// Returns the decoded integer symbols.
pub fn gaussian_decode(
    data: &[u8],
    num_symbols: usize,
    means: &[f64],
    scales: &[f64],
) -> Result<Vec<i32>> {
    assert_eq!(num_symbols, means.len());
    assert_eq!(num_symbols, scales.len());

    if !data.len().is_multiple_of(4) {
        bail!("ANS data length {} is not a multiple of 4", data.len());
    }

    let words: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut coder = DefaultAnsCoder::from_compressed(words)
        .map_err(|e| anyhow::anyhow!("failed to reconstruct ANS coder: {e:?}"))?;

    let mut symbols = Vec::with_capacity(num_symbols);
    let mut model_cache = HashMap::new();

    for i in 0..num_symbols {
        let model = get_or_build_model(&mut model_cache, means[i], scales[i])?;
        let sym = coder
            .decode_symbol(model)
            .map_err(|e| anyhow::anyhow!("failed to decode symbol {i}: {e:?}"))?;
        symbols.push(sym as i32 - 128);
    }

    Ok(symbols)
}

/// Entropy-encode symbols using a factorized (non-parametric) prior.
///
/// `cdf_tables[c]` contains the PMF for channel `c`.
/// `symbols` are the quantized latent values, flattened in C,H,W order.
/// `channel_indices[i]` indicates which channel symbol `i` belongs to.
pub fn factorized_encode(
    symbols: &[i32],
    channel_indices: &[usize],
    cdf_tables: &[Vec<f64>],
) -> Result<Vec<u8>> {
    let mut coder = DefaultAnsCoder::new();

    // Pre-build models for each channel (there are only C of them)
    let models: Vec<DefaultContiguousCategoricalEntropyModel> = cdf_tables
        .iter()
        .enumerate()
        .map(|(ch, pmf)| {
            DefaultContiguousCategoricalEntropyModel
                ::from_floating_point_probabilities_fast(pmf, None)
                .map_err(|_| anyhow::anyhow!("failed to build factorized model for channel {ch}"))
        })
        .collect::<Result<Vec<_>>>()?;

    for i in (0..symbols.len()).rev() {
        let ch = channel_indices[i];
        let sym = (symbols[i] + 128).clamp(0, 255) as usize;
        coder
            .encode_symbol(sym, models[ch].clone())
            .map_err(|e| anyhow::anyhow!("failed to encode factorized symbol {i}: {e:?}"))?;
    }

    let compressed = coder.into_compressed().unwrap_or_default();
    Ok(compressed
        .into_iter()
        .flat_map(|w| w.to_le_bytes())
        .collect())
}

/// Entropy-decode symbols using a factorized prior.
pub fn factorized_decode(
    data: &[u8],
    num_symbols: usize,
    channel_indices: &[usize],
    cdf_tables: &[Vec<f64>],
) -> Result<Vec<i32>> {
    if !data.len().is_multiple_of(4) {
        bail!("ANS data length {} is not a multiple of 4", data.len());
    }

    let words: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut coder = DefaultAnsCoder::from_compressed(words)
        .map_err(|e| anyhow::anyhow!("failed to reconstruct ANS coder: {e:?}"))?;

    // Pre-build models for each channel
    let models: Vec<DefaultContiguousCategoricalEntropyModel> = cdf_tables
        .iter()
        .enumerate()
        .map(|(ch, pmf)| {
            DefaultContiguousCategoricalEntropyModel
                ::from_floating_point_probabilities_fast(pmf, None)
                .map_err(|_| anyhow::anyhow!("failed to build factorized decode model for channel {ch}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut symbols = Vec::with_capacity(num_symbols);

    for (i, &ch) in channel_indices.iter().enumerate().take(num_symbols) {
        let sym = coder
            .decode_symbol(models[ch].clone())
            .map_err(|e| anyhow::anyhow!("failed to decode factorized symbol {i}: {e:?}"))?;
        symbols.push(sym as i32 - 128);
    }

    Ok(symbols)
}

/// Compute a quantized Gaussian PMF over [min_val, max_val].
///
/// For each integer k in [min_val, max_val], compute
///   P(k) = Phi((k + 0.5 - mean) / std) - Phi((k - 0.5 - mean) / std)
/// where Phi is the standard normal CDF.
fn quantized_gaussian_pmf(mean: f64, std_dev: f64, min_val: i32, max_val: i32) -> Vec<f64> {
    let n = (max_val - min_val + 1) as usize;
    let mut pmf = Vec::with_capacity(n);
    let inv_std = 1.0 / std_dev;

    for k in min_val..=max_val {
        let upper = normal_cdf((k as f64 + 0.5 - mean) * inv_std);
        let lower = normal_cdf((k as f64 - 0.5 - mean) * inv_std);
        let p = (upper - lower).max(1e-10);
        pmf.push(p);
    }

    // Normalize
    let sum: f64 = pmf.iter().sum();
    if sum > 0.0 {
        for p in &mut pmf {
            *p /= sum;
        }
    }

    pmf
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (maximum error: 1.5e-7).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_pmf_sums_to_one() {
        let pmf = quantized_gaussian_pmf(0.0, 1.0, -128, 127);
        let sum: f64 = pmf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gaussian_pmf_peaked_at_mean() {
        let pmf = quantized_gaussian_pmf(5.0, 2.0, -128, 127);
        let peak_idx = pmf
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_idx, 133); // mean=5 maps to index 133
    }

    #[test]
    fn normal_cdf_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(10.0) - 1.0).abs() < 1e-4);
        assert!(normal_cdf(-10.0) < 1e-4);
    }
}
