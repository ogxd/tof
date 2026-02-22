use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::backend::NdArray;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::prelude::*;
use clap::{Parser, Subcommand};

use coldstorage::codec::fast_conv::ConvDispatch;
use coldstorage::config::{ColdStorageConfig, DeviceKind, ModelArch};
use coldstorage::storage::ColdStorage;

/// Neural Photo Cold Storage — learned image compression CLI.
///
/// Compresses photos into compact latent representations using a learned
/// neural image codec. The model weights are stored once; each photo becomes
/// a small binary blob of entropy-coded latents.
///
/// Quality levels (1–8):
///   4–5: good quality, high compression (~10–20x)
///   6:   near-transparent (recommended, ~8–12x)
///   7–8: virtually lossless (~5–8x)
#[derive(Parser)]
#[command(name = "coldstorage", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest photos into cold storage.
    Ingest {
        /// Path to a photo file or directory.
        path: PathBuf,

        /// Storage vault directory.
        #[arg(long, default_value = "./vault")]
        storage: PathBuf,

        /// Quality level (1–8, higher = better quality, larger blobs).
        #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=8))]
        quality: u8,

        /// Compression model architecture.
        #[arg(long, default_value = "hyperprior", value_enum)]
        model: ModelArch,

        /// Compute quality metrics (PSNR) during ingest. Doubles processing time.
        #[arg(long)]
        metrics: bool,

        /// Maximum image dimension in pixels. Larger images are downscaled.
        #[arg(long, default_value_t = 2048)]
        max_dim: u32,

        /// Compute device (cpu or gpu). GPU uses Metal on macOS.
        #[arg(long, default_value = "cpu", value_enum)]
        device: DeviceKind,

        /// Path to pretrained weights directory (exported .npy files).
        #[arg(long)]
        weights: Option<PathBuf>,
    },

    /// Reconstruct a photo from cold storage.
    Retrieve {
        /// Photo ID to retrieve.
        photo_id: i64,

        /// Output file path.
        #[arg(short, long, default_value = "./restored.png")]
        output: PathBuf,

        /// Storage vault directory.
        #[arg(long, default_value = "./vault")]
        storage: PathBuf,

        /// Quality level (must match the ingest quality for correct model loading).
        #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=8))]
        quality: u8,

        /// Compression model architecture.
        #[arg(long, default_value = "hyperprior", value_enum)]
        model: ModelArch,

        /// Compute device (cpu or gpu).
        #[arg(long, default_value = "cpu", value_enum)]
        device: DeviceKind,

        /// Path to pretrained weights directory (exported .npy files).
        #[arg(long)]
        weights: Option<PathBuf>,
    },

    /// Show compression statistics for the vault.
    Stats {
        /// Storage vault directory.
        #[arg(long, default_value = "./vault")]
        storage: PathBuf,

        /// Quality level.
        #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=8))]
        quality: u8,

        /// Compression model architecture.
        #[arg(long, default_value = "hyperprior", value_enum)]
        model: ModelArch,
    },

    /// List stored photos.
    List {
        /// Storage vault directory.
        #[arg(long, default_value = "./vault")]
        storage: PathBuf,

        /// Maximum number of photos to list.
        #[arg(long, default_value_t = 20)]
        limit: u32,

        /// Quality level.
        #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=8))]
        quality: u8,

        /// Compression model architecture.
        #[arg(long, default_value = "hyperprior", value_enum)]
        model: ModelArch,
    },

    /// Fine-tune the compression model on your photo corpus.
    Finetune {
        /// Storage vault directory.
        #[arg(long, default_value = "./vault")]
        storage: PathBuf,

        /// Number of training epochs.
        #[arg(long, default_value_t = 10)]
        epochs: u32,

        /// Learning rate.
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,

        /// Rate-distortion trade-off (lambda).
        #[arg(long, default_value_t = 0.01)]
        lambda: f64,

        /// Quality level.
        #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=8))]
        quality: u8,

        /// Compression model architecture.
        #[arg(long, default_value = "hyperprior", value_enum)]
        model: ModelArch,
    },
}

/// Run the ingest command with a specific backend.
fn run_ingest<B: Backend + ConvDispatch>(config: ColdStorageConfig, path: &Path, device: B::Device) -> Result<()> {
    let engine = ColdStorage::<B>::new(config, device)?;

    if path.is_dir() {
        let stats = engine.ingest_directory(path)?;
        println!("{stats}");
    } else if path.is_file() {
        match engine.ingest_photo(path)? {
            Some(rec) => {
                println!("Ingested: {}", rec.filename);
                println!("  Ratio: {:.1}x", rec.compression_ratio);
                if let Some(psnr) = rec.psnr {
                    println!("  PSNR:  {psnr:.2} dB");
                }
            }
            None => {
                println!("Photo already exists (duplicate SHA-256).");
            }
        }
    } else {
        anyhow::bail!("path not found: {}", path.display());
    }
    Ok(())
}

/// Run the retrieve command with a specific backend.
fn run_retrieve<B: Backend + ConvDispatch>(config: ColdStorageConfig, photo_id: i64, output: &Path, device: B::Device) -> Result<()> {
    let engine = ColdStorage::<B>::new(config, device)?;
    engine.retrieve_photo(photo_id, output)
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Ingest {
            path,
            storage,
            quality,
            model,
            metrics,
            max_dim,
            device,
            weights,
        } => {
            let config = ColdStorageConfig {
                storage_dir: storage,
                model,
                quality,
                max_dim,
                compute_metrics: metrics,
                device,
                weights_dir: weights,
            };
            match device {
                DeviceKind::Cpu => {
                    run_ingest::<NdArray>(config, &path, Default::default())?;
                }
                DeviceKind::Gpu => {
                    eprintln!("Using GPU (WGPU/Metal)");
                    run_ingest::<Wgpu>(config, &path, WgpuDevice::default())?;
                }
            }
        }

        Commands::Retrieve {
            photo_id,
            output,
            storage,
            quality,
            model,
            device,
            weights,
        } => {
            let config = ColdStorageConfig {
                storage_dir: storage,
                model,
                quality,
                weights_dir: weights,
                ..Default::default()
            };
            match device {
                DeviceKind::Cpu => {
                    run_retrieve::<NdArray>(config, photo_id, &output, Default::default())?;
                }
                DeviceKind::Gpu => {
                    eprintln!("Using GPU (WGPU/Metal)");
                    run_retrieve::<Wgpu>(config, photo_id, &output, WgpuDevice::default())?;
                }
            }
        }

        Commands::Stats {
            storage,
            quality,
            model,
        } => {
            // Stats/List don't need the model, use CPU backend (lightweight)
            let config = ColdStorageConfig {
                storage_dir: storage,
                model,
                quality,
                ..Default::default()
            };
            let engine = ColdStorage::<NdArray>::new(config, Default::default())?;
            engine.show_stats()?;
        }

        Commands::List {
            storage,
            limit,
            quality,
            model,
        } => {
            let config = ColdStorageConfig {
                storage_dir: storage,
                model,
                quality,
                ..Default::default()
            };
            let engine = ColdStorage::<NdArray>::new(config, Default::default())?;
            engine.list_photos(limit)?;
        }

        Commands::Finetune {
            storage,
            epochs,
            lr,
            lambda,
            quality,
            model,
        } => {
            let _ = (epochs, lr, lambda);
            let config = ColdStorageConfig {
                storage_dir: storage,
                model,
                quality,
                ..Default::default()
            };
            let _engine = ColdStorage::<NdArray>::new(config, Default::default())?;
            // TODO: implement fine-tuning with Burn's autodiff backend
            eprintln!(
                "Fine-tuning not yet implemented. \
                 (epochs={epochs}, lr={lr}, lambda={lambda})"
            );
            eprintln!("This will use Burn's Autodiff<NdArray> backend for training.");
        }
    }

    Ok(())
}
