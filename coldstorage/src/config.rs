use std::path::{Path, PathBuf};
use std::process::Command;

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
    /// Path to pretrained weights directory (exported .npy files).
    pub weights_dir: Option<PathBuf>,
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
            weights_dir: None,
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

    /// Resolve the weights directory, searching standard locations if not explicitly set.
    ///
    /// Search order:
    /// 1. Explicit `--weights` path (if provided and contains manifest.json)
    /// 2. `./weights/{model}_q{quality}/`
    /// 3. `{storage_dir}/weights/{model}_q{quality}/`
    /// 4. `{exe_dir}/../weights/{model}_q{quality}/` (relative to binary)
    /// 5. `{exe_dir}/weights/{model}_q{quality}/`
    pub fn resolve_weights_dir(&self) -> Option<std::path::PathBuf> {
        let subdir = format!("{}_q{}", self.model.as_str(), self.quality);

        let canonicalize = |p: PathBuf| -> PathBuf {
            p.canonicalize().unwrap_or(p)
        };

        // If explicitly set, check it has a manifest
        if let Some(ref dir) = self.weights_dir {
            if dir.join("manifest.json").exists() {
                return Some(canonicalize(dir.clone()));
            }
            // Maybe they pointed at the parent dir, try appending the subdir
            let with_sub = dir.join(&subdir);
            if with_sub.join("manifest.json").exists() {
                return Some(canonicalize(with_sub));
            }
        }

        // Search standard locations
        let candidates = [
            // CWD-relative
            std::path::PathBuf::from("weights").join(&subdir),
            // Inside the vault
            self.storage_dir.join("weights").join(&subdir),
        ];

        for candidate in &candidates {
            if candidate.join("manifest.json").exists() {
                return Some(canonicalize(candidate.clone()));
            }
        }

        // Relative to the executable (binary is typically at target/release/coldstorage)
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                for rel in &["../weights", "../../weights", "weights"] {
                    let candidate = exe_dir.join(rel).join(&subdir);
                    if candidate.join("manifest.json").exists() {
                        return Some(canonicalize(candidate));
                    }
                }
            }
        }

        None
    }

    /// Find the `export_weights.py` script by searching standard locations.
    fn find_export_script() -> Option<PathBuf> {
        // CWD-relative
        let cwd = Path::new("export_weights.py");
        if cwd.exists() {
            return Some(cwd.to_path_buf());
        }

        // Relative to the executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                for rel in &["../../export_weights.py", "../export_weights.py", "export_weights.py"] {
                    let candidate = exe_dir.join(rel);
                    if candidate.exists() {
                        return Some(candidate.canonicalize().unwrap_or(candidate));
                    }
                }
            }
        }

        None
    }

    /// Determine the default weights root directory (where exported weights are stored).
    ///
    /// Prefers the directory next to `export_weights.py`, falls back to CWD.
    fn default_weights_root() -> PathBuf {
        if let Some(script) = Self::find_export_script() {
            if let Some(parent) = script.parent() {
                return parent.join("weights");
            }
        }
        PathBuf::from("weights")
    }

    /// Auto-export pretrained weights by invoking `export_weights.py`.
    ///
    /// Downloads the CompressAI pretrained model and exports .npy weights.
    /// Returns the path to the exported weights directory on success.
    pub fn auto_export_weights(&self) -> anyhow::Result<PathBuf> {
        let script = Self::find_export_script()
            .ok_or_else(|| anyhow::anyhow!(
                "cannot find export_weights.py — needed to download pretrained weights"
            ))?;

        let weights_root = Self::default_weights_root();
        let subdir = format!("{}_q{}", self.model.as_str(), self.quality);
        let output_dir = weights_root.join(&subdir);

        eprintln!(
            "  Downloading pretrained weights ({} quality={})...",
            self.model, self.quality
        );
        eprintln!("  Running: python3 {} --model {} --quality {} --output {}",
            script.display(), self.model.as_str(), self.quality, output_dir.display()
        );

        let status = Command::new("python3")
            .arg(&script)
            .arg("--model")
            .arg(self.model.as_str())
            .arg("--quality")
            .arg(self.quality.to_string())
            .arg("--output")
            .arg(&output_dir)
            .status()
            .map_err(|e| anyhow::anyhow!(
                "failed to run python3 (is Python installed?): {e}"
            ))?;

        anyhow::ensure!(
            status.success(),
            "export_weights.py failed (exit code: {:?}). \
             Make sure compressai, torch, and numpy are installed:\n  \
             pip install compressai torch numpy",
            status.code()
        );

        // Verify the export produced a manifest
        anyhow::ensure!(
            output_dir.join("manifest.json").exists(),
            "export completed but manifest.json not found in {}",
            output_dir.display()
        );

        let canonical = output_dir.canonicalize().unwrap_or(output_dir);
        eprintln!("  Weights exported to: {}", canonical.display());
        Ok(canonical)
    }
}
