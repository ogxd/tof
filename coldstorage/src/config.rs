use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Supported compression model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArch {
    /// Ballé et al. 2018 — hyperprior with factorized entropy model
    Hyperprior,
    /// Minnen et al. 2018 — mean-scale hyperprior
    Mbt2018Mean,
    /// Minnen et al. 2018 — joint autoregressive + hyperprior
    Mbt2018,
    /// Cheng et al. 2020 — anchor (attention-based)
    Cheng2020,
}

impl ModelArch {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hyperprior => "hyperprior",
            Self::Mbt2018Mean => "mbt2018_mean",
            Self::Mbt2018 => "mbt2018",
            Self::Cheng2020 => "cheng2020",
        }
    }

    /// Returns (N, M) channel dimensions for a given quality level.
    ///
    /// These match the CompressAI pretrained model configurations.
    pub fn channel_dims(&self, quality: u8) -> (usize, usize) {
        match self {
            Self::Hyperprior => {
                if quality <= 5 {
                    (128, 192)
                } else {
                    (192, 320)
                }
            }
            // Other architectures share the same pattern for now
            Self::Mbt2018Mean | Self::Mbt2018 => {
                if quality <= 5 {
                    (128, 192)
                } else {
                    (192, 320)
                }
            }
            Self::Cheng2020 => {
                if quality <= 5 {
                    (128, 192)
                } else {
                    (192, 320)
                }
            }
        }
    }
}

impl std::fmt::Display for ModelArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Compute backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// Top-level configuration for the cold storage engine.
#[derive(Debug, Clone)]
pub struct ColdStorageConfig {
    /// Root directory for the vault (latents, models, database).
    pub storage_dir: PathBuf,
    /// Compression model architecture.
    pub model: ModelArch,
    /// Quality level 1–8 (higher = better quality, larger latents).
    pub quality: u8,
    /// Maximum image dimension — larger images are downscaled.
    pub max_dim: u32,
    /// Whether to compute quality metrics (PSNR) during ingest.
    pub compute_metrics: bool,
    /// Compute device.
    pub device: DeviceKind,
}

impl Default for ColdStorageConfig {
    fn default() -> Self {
        Self {
            storage_dir: PathBuf::from("./vault"),
            model: ModelArch::Hyperprior,
            quality: 6,
            max_dim: 2048,
            compute_metrics: false,
            device: DeviceKind::Cpu,
        }
    }
}

impl ColdStorageConfig {
    /// Path to the SQLite catalog database.
    pub fn db_path(&self) -> PathBuf {
        self.storage_dir.join("catalog.db")
    }

    /// Path to the latents directory.
    pub fn latents_dir(&self) -> PathBuf {
        self.storage_dir.join("latents")
    }

    /// Path to the models directory.
    pub fn models_dir(&self) -> PathBuf {
        self.storage_dir.join("models")
    }

    /// Validate configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (1..=8).contains(&self.quality),
            "quality must be between 1 and 8, got {}",
            self.quality
        );
        anyhow::ensure!(
            self.max_dim >= 64,
            "max_dim must be at least 64, got {}",
            self.max_dim
        );
        Ok(())
    }
}
