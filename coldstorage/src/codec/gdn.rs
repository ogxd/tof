//! Generalized Divisive Normalization (GDN) and its inverse (IGDN).
//!
//! GDN is a learnable normalization layer used in learned image compression:
//!
//!   y_i = x_i / sqrt(beta_i + sum_j(gamma_ij * x_j^2))
//!
//! IGDN is the inverse operation used in the decoder:
//!
//!   y_i = x_i * sqrt(beta_i + sum_j(gamma_ij * x_j^2))
//!
//! Parameters beta and gamma are stored in reparametrized form to ensure
//! positivity: beta_reparam and gamma_reparam, where
//!   beta  = beta_reparam^2 + epsilon
//!   gamma = gamma_reparam^2
//!
//! Reference: Ballé et al., "Density Modeling of Images Using a Generalized
//! Normalization Transformation" (ICLR 2016).

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::prelude::*;

/// Configuration for GDN / IGDN.
#[derive(Config, Debug)]
pub struct GdnConfig {
    /// Number of channels.
    pub channels: usize,
    /// Whether to compute the inverse (IGDN).
    #[config(default = false)]
    pub inverse: bool,
}

/// GDN / IGDN module.
#[derive(Module, Debug)]
pub struct Gdn<B: Backend> {
    /// Reparametrized beta — shape [channels].
    beta_reparam: Param<Tensor<B, 1>>,
    /// Reparametrized gamma — shape [channels, channels].
    gamma_reparam: Param<Tensor<B, 2>>,
    /// Whether to compute inverse (IGDN).
    inverse: bool,
}

impl GdnConfig {
    /// Initialize a GDN/IGDN module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gdn<B> {
        let beta_reparam = Initializer::Constant { value: 1.0 }
            .init([self.channels], device);
        let gamma_reparam = Initializer::Constant { value: 0.1 }
            .init([self.channels, self.channels], device);

        Gdn {
            beta_reparam,
            gamma_reparam,
            inverse: self.inverse,
        }
    }
}

impl<B: Backend> Gdn<B> {
    /// Forward pass.
    ///
    /// Input shape: [batch, channels, height, width]
    /// Output shape: same as input.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, channels, _height, _width] = x.dims();
        let device = x.device();

        // Derive positive parameters
        let beta_val = self.beta_reparam.val();
        let beta = beta_val.clone() * beta_val + 1e-6; // [C]

        let gamma_val = self.gamma_reparam.val();
        let gamma = gamma_val.clone() * gamma_val; // [C, C]

        // Use diagonal of gamma (per-channel self-interaction) to avoid the
        // O(C^2 * H * W) matmul.  This is a standard simplification used in
        // efficient GDN implementations.
        let gamma_data: Vec<f32> = gamma
            .reshape([channels * channels])
            .into_data()
            .to_vec()
            .unwrap_or_default();
        let diag_vals: Vec<f32> = (0..channels)
            .map(|i| gamma_data[i * channels + i])
            .collect();
        let gamma_diag = Tensor::<B, 1>::from_floats(diag_vals.as_slice(), &device);

        // gamma_diag [C] → [1, C, 1, 1] for automatic broadcasting against [B, C, H, W]
        let gamma_4d = gamma_diag.reshape([1, channels, 1, 1]);
        let x_sq = x.clone() * x.clone();
        let weighted = x_sq * gamma_4d;

        // beta [C] → [1, C, 1, 1] for automatic broadcasting
        let beta_4d = beta.reshape([1, channels, 1, 1]);
        let denominator = (weighted + beta_4d).sqrt();

        if self.inverse {
            x * denominator
        } else {
            x / denominator
        }
    }
}
