use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lumen-quant")]
#[command(version = "0.1.0")]
#[command(about = "High-performance 4-bit model quantization engine", long_about = None)]
pub struct Cli { // Added pub
    #[command(subcommand)]
    pub command: Commands, // Added pub
}

#[derive(Subcommand)]
pub enum Commands { // Added pub
    /// Run performance and accuracy benchmarks on a model file
    Bench {
        /// Path to the model file (.safetensors or .bin)
        #[arg(short, long)]
        input: PathBuf,
    },
    /// Quantize a model to .lumen format
    Encode {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
}