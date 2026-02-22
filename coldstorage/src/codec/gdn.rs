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
    pub beta_reparam: Param<Tensor<B, 1>>,
    /// Reparametrized gamma — shape [channels, channels].
    pub gamma_reparam: Param<Tensor<B, 2>>,
    /// Whether to compute inverse (IGDN).
    pub inverse: bool,
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
    ///
    /// Computes: y_i = x_i / sqrt(beta_i + sum_j(gamma_ij * x_j²))  (GDN)
    ///       or: y_i = x_i * sqrt(beta_i + sum_j(gamma_ij * x_j²))  (IGDN)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();

        // Derive positive parameters
        let beta_val = self.beta_reparam.val();
        let beta = beta_val.clone() * beta_val + 1e-6; // [C]

        let gamma_val = self.gamma_reparam.val();
        let gamma = gamma_val.clone() * gamma_val; // [C, C]

        // Full gamma matmul: norm_i = sum_j(gamma_ij * x_j²)
        // x_sq: [B, C, H, W] → [B, C, H*W]
        let x_sq = x.clone() * x.clone();
        let x_sq_flat = x_sq.reshape([batch, channels, height * width]);

        // gamma: [C, C] → [1, C, C] for broadcasting over batch
        let gamma_3d = gamma.unsqueeze::<3>(); // [1, C, C]

        // matmul: [1, C, C] × [B, C, H*W] → [B, C, H*W]
        let weighted_flat = gamma_3d.matmul(x_sq_flat);
        let weighted = weighted_flat.reshape([batch, channels, height, width]);

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
