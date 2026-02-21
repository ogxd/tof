//! Quality metrics for comparing original and reconstructed images.

use burn::prelude::*;

/// Compute Peak Signal-to-Noise Ratio between two image tensors.
///
/// Both tensors should be [1, C, H, W] with values in [0, 1].
/// Returns PSNR in decibels.
pub fn psnr<B: Backend>(original: &Tensor<B, 4>, reconstructed: &Tensor<B, 4>) -> f64 {
    let diff = original.clone() - reconstructed.clone();
    let mse_tensor = (diff.clone() * diff).mean();
    let mse: f64 = mse_tensor.into_scalar().elem();

    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (1.0 / mse).log10()
}
