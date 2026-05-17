//! # Quantization Kernels
//! 
//! This module contains the mathematical logic for transforming f32 weights
//! into 4-bit compressed representations.


pub mod block;  // looks for quant/block.rs
pub mod q4_0;   // looks for quant/q4_0.rs

pub use q4_0::BlockQ4_0;