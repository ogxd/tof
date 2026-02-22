//! bmshj2018 Hyperprior model — Ballé, Minnen, Singh, Hwang, Johnston (2018).
//!
//! Architecture (ScaleHyperprior):
//!
//!   Encoder  (g_a): image → latents y
//!   Decoder  (g_s): quantized latents ŷ → reconstructed image
//!   Hyper-encoder (h_a): y → hyper-latents z
//!   Hyper-decoder (h_s): quantized z̃ → scale parameters for y
//!
//! The entropy model:
//!   - z is coded with a learned factorized prior (per-channel CDF tables).
//!   - y is coded with a zero-mean Gaussian conditional whose scale comes from h_s(z̃).

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;

use super::fast_conv::{conv2d_forward, conv_transpose2d_forward, ConvDispatch};
use super::gdn::{Gdn, GdnConfig};

// ─── Analysis Transform (Encoder) ───────────────────────────────────────────

#[derive(Config, Debug)]
pub struct AnalysisTransformConfig {
    pub n: usize,
    pub m: usize,
}

#[derive(Module, Debug)]
pub struct AnalysisTransform<B: Backend> {
    pub conv1: Conv2d<B>,
    pub gdn1: Gdn<B>,
    pub conv2: Conv2d<B>,
    pub gdn2: Gdn<B>,
    pub conv3: Conv2d<B>,
    pub gdn3: Gdn<B>,
    pub conv4: Conv2d<B>,
}

impl AnalysisTransformConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AnalysisTransform<B> {
        AnalysisTransform {
            conv1: Conv2dConfig::new([3, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .init(device),
            gdn1: GdnConfig::new(self.n).init::<B>(device),
            conv2: Conv2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .init(device),
            gdn2: GdnConfig::new(self.n).init::<B>(device),
            conv3: Conv2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .init(device),
            gdn3: GdnConfig::new(self.n).init::<B>(device),
            conv4: Conv2dConfig::new([self.n, self.m], [5, 5])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .init(device),
        }
    }
}

impl<B: Backend + ConvDispatch> AnalysisTransform<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.gdn1.forward(conv2d_forward(&self.conv1, x));
        let x = self.gdn2.forward(conv2d_forward(&self.conv2, x));
        let x = self.gdn3.forward(conv2d_forward(&self.conv3, x));
        conv2d_forward(&self.conv4, x)
    }
}

// ─── Synthesis Transform (Decoder) ──────────────────────────────────────────

#[derive(Config, Debug)]
pub struct SynthesisTransformConfig {
    pub n: usize,
    pub m: usize,
}

#[derive(Module, Debug)]
pub struct SynthesisTransform<B: Backend> {
    pub deconv1: ConvTranspose2d<B>,
    pub igdn1: Gdn<B>,
    pub deconv2: ConvTranspose2d<B>,
    pub igdn2: Gdn<B>,
    pub deconv3: ConvTranspose2d<B>,
    pub igdn3: Gdn<B>,
    pub deconv4: ConvTranspose2d<B>,
}

impl SynthesisTransformConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SynthesisTransform<B> {
        SynthesisTransform {
            deconv1: ConvTranspose2dConfig::new([self.m, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding([2, 2])
                .with_padding_out([1, 1])
                .init(device),
            igdn1: GdnConfig::new(self.n).with_inverse(true).init::<B>(device),
            deconv2: ConvTranspose2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding([2, 2])
                .with_padding_out([1, 1])
                .init(device),
            igdn2: GdnConfig::new(self.n).with_inverse(true).init::<B>(device),
            deconv3: ConvTranspose2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding([2, 2])
                .with_padding_out([1, 1])
                .init(device),
            igdn3: GdnConfig::new(self.n).with_inverse(true).init::<B>(device),
            deconv4: ConvTranspose2dConfig::new([self.n, 3], [5, 5])
                .with_stride([2, 2])
                .with_padding([2, 2])
                .with_padding_out([1, 1])
                .init(device),
        }
    }
}

impl<B: Backend + ConvDispatch> SynthesisTransform<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.igdn1.forward(conv_transpose2d_forward(&self.deconv1, x));
        let x = self.igdn2.forward(conv_transpose2d_forward(&self.deconv2, x));
        let x = self.igdn3.forward(conv_transpose2d_forward(&self.deconv3, x));
        conv_transpose2d_forward(&self.deconv4, x)
    }
}

// ─── Hyper-Analysis Transform ───────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct HyperAnalysisConfig {
    pub n: usize,
    pub m: usize,
}

#[derive(Module, Debug)]
pub struct HyperAnalysis<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
}

impl HyperAnalysisConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HyperAnalysis<B> {
        HyperAnalysis {
            conv1: Conv2dConfig::new([self.m, self.n], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .init(device),
            conv3: Conv2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .init(device),
        }
    }
}

impl<B: Backend + ConvDispatch> HyperAnalysis<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = burn::tensor::activation::relu(conv2d_forward(&self.conv1, x));
        let x = burn::tensor::activation::relu(conv2d_forward(&self.conv2, x));
        conv2d_forward(&self.conv3, x)
    }
}

// ─── Hyper-Synthesis Transform ──────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct HyperSynthesisConfig {
    pub n: usize,
    pub m: usize,
}

#[derive(Module, Debug)]
pub struct HyperSynthesis<B: Backend> {
    pub deconv1: ConvTranspose2d<B>,
    pub deconv2: ConvTranspose2d<B>,
    pub conv3: Conv2d<B>,
}

impl HyperSynthesisConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HyperSynthesis<B> {
        // h_s: N → N → N → M (ScaleHyperprior: last layer is Conv2d, outputs scales only)
        HyperSynthesis {
            deconv1: ConvTranspose2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding([2, 2])
                .with_padding_out([1, 1])
                .init(device),
            deconv2: ConvTranspose2dConfig::new([self.n, self.n], [5, 5])
                .with_stride([2, 2])
                .with_padding([2, 2])
                .with_padding_out([1, 1])
                .init(device),
            conv3: Conv2dConfig::new([self.n, self.m], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }
}

impl<B: Backend + ConvDispatch> HyperSynthesis<B> {
    /// Forward pass. Returns a tensor of shape [B, M, H, W] — predicted scales for y.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = burn::tensor::activation::relu(conv_transpose2d_forward(&self.deconv1, x));
        let x = burn::tensor::activation::relu(conv_transpose2d_forward(&self.deconv2, x));
        burn::tensor::activation::relu(conv2d_forward(&self.conv3, x))
    }
}

// ─── Full Hyperprior Model ──────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct HyperpriorConfig {
    /// Quality level (1–8).
    pub quality: u8,
}

impl HyperpriorConfig {
    fn channel_dims(&self) -> (usize, usize) {
        if self.quality <= 5 {
            (128, 192)
        } else {
            (192, 320)
        }
    }
}

#[derive(Module, Debug)]
pub struct Hyperprior<B: Backend> {
    pub g_a: AnalysisTransform<B>,
    pub g_s: SynthesisTransform<B>,
    pub h_a: HyperAnalysis<B>,
    pub h_s: HyperSynthesis<B>,
    /// Number of latent channels (M).
    pub m: usize,
    /// Number of hyper channels (N).
    pub n: usize,
}

impl HyperpriorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Hyperprior<B> {
        let (n, m) = self.channel_dims();
        Hyperprior {
            g_a: AnalysisTransformConfig::new(n, m).init(device),
            g_s: SynthesisTransformConfig::new(n, m).init(device),
            h_a: HyperAnalysisConfig::new(n, m).init(device),
            h_s: HyperSynthesisConfig::new(n, m).init(device),
            m,
            n,
        }
    }
}

impl<B: Backend + ConvDispatch> Hyperprior<B> {
    /// Encode an image tensor into quantized latents and hyper-latents.
    ///
    /// Input: [1, 3, H, W] image tensor (H, W must be multiples of 64).
    /// Returns: (y_quantized, z_quantized, scales)
    ///   - y_quantized: [1, M, H/16, W/16] — quantized latents
    ///   - z_quantized: [1, N, H/64, W/64] — quantized hyper-latents
    ///   - scales: [1, M, H/16, W/16] — predicted scales from h_s
    pub fn encode(
        &self,
        x: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let y = self.g_a.forward(x);
        let z = self.h_a.forward(y.clone());
        let z_hat = quantize(&z);
        let scales_hat = self.h_s.forward(z_hat.clone());
        let y_hat = quantize(&y);
        (y_hat, z_hat, scales_hat)
    }

    /// Decode quantized latents back to an image.
    ///
    /// Input: y_hat [1, M, H/16, W/16] — quantized latents.
    /// Returns: [1, 3, H, W] reconstructed image tensor.
    pub fn decode(&self, y_hat: Tensor<B, 4>) -> Tensor<B, 4> {
        self.g_s.forward(y_hat)
    }

    /// Full forward pass (for training): image → reconstructed image + likelihoods.
    ///
    /// Returns (x_hat, y, z, scales_hat) for computing rate-distortion loss.
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let y = self.g_a.forward(x);
        let z = self.h_a.forward(y.clone());
        let z_hat = quantize_with_noise(&z);
        let scales_hat = self.h_s.forward(z_hat.clone());
        let y_hat = quantize_with_noise(&y);
        let x_hat = self.g_s.forward(y_hat);
        (x_hat, y, z, scales_hat)
    }

    /// Extract Gaussian (mean, scale) from the hyper-synthesis output.
    ///
    /// For the ScaleHyperprior, h_s outputs [B, M, H, W] = scales only.
    /// Means are zero (zero-mean Gaussian conditional).
    /// Scales are clamped to [0.01, 100.0].
    pub fn split_gaussian_params(
        &self,
        scales_hat: &Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let means = Tensor::zeros(scales_hat.dims(), &scales_hat.device());
        let scales = scales_hat.clone().clamp(0.01, 100.0);
        (means, scales)
    }
}

/// Quantize by rounding to nearest integer (inference path).
fn quantize<B: Backend>(x: &Tensor<B, 4>) -> Tensor<B, 4> {
    x.clone().round()
}

/// Quantize with uniform noise (training path — straight-through estimator).
///
/// During training, we add U(-0.5, 0.5) noise instead of rounding,
/// which provides a differentiable approximation.
fn quantize_with_noise<B: Backend>(x: &Tensor<B, 4>) -> Tensor<B, 4> {
    let noise = Tensor::random(x.dims(), burn::tensor::Distribution::Uniform(-0.5, 0.5), &x.device());
    x.clone() + noise
}
