
// To find ModelLoader, we need to look into the loader module, we do that by using "crate" which essentially asks the main.rs file 
// to look into the loader.rs file and find the ModelLoader struct and its associated methods and bring it into local scope.
use std::io::{Result};
use std::io::Write;
use rayon::prelude::*; // for parallel processing in decode

pub mod encoder;
pub mod block_buffer;
use crate::loader::ModelLoader;
use crate::quant::{ QuantizedBlock }; // to make generic encoder/decoder work we need this trait in scope


const MAGIC: &[u8; 4] = b"LUMN";
const VERSION: u32 = 2056; // Arbitrary version number for our format


pub fn encode<T: QuantizedBlock + Send + Sync>(loader: &ModelLoader, output_path: &str) -> Result<usize> {
    let mut file = std::fs::File::create(output_path)?;
    
    // 1. Write the "Formal LUMN" Header directly to the file first
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;

    #[cfg(target_arch = "x86_64")]
    {
        // Use more than 2 threads for parallel to justify the channel overhead
        if is_x86_feature_detected!("avx2") && rayon::current_num_threads() > 2 {
            println!("[ENCODER] RUNNING AVX2 PARALLEL");
            return encoder::encode_parallel::<T>(loader, file);
        }
        
        if is_x86_feature_detected!("avx2") {
            println!("[ENCODER] RUNNING AVX2 SCALAR");
            return encoder::encode_scalar::<T>(loader, file);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Assuming neon_serial is also refactored to take ownership of file
            return encoder::encode_neon_serial::<T>(loader, file);
        }
    }

    // Scalar fallback — works on everything
    println!("[ENCODER] FALLBACK TO SCALAR");
    encoder::encode_scalar::<T>(loader, file)
}

/// DECODE: Takes a .lumen loader and returns a flat Vec of original floats
pub fn decode<T: QuantizedBlock>(loader: &ModelLoader) -> Result<Vec<f32>> {
    let data = loader.get_data();
    let weight_data = &data[8..];
    let num_blocks = weight_data.len() / T::PACKED_SIZE;
    let mut all_floats = vec![0.0f32; num_blocks * T::CHUNK_SIZE];
    
    let blocks_per_task = 512;
    let floats_per_task = blocks_per_task * T::CHUNK_SIZE;
    let arch = pulp::Arch::new(); // Detect CPU architecture for optimized dequantization

    // Use par_chunks_mut (not exact) to catch the remainder
    arch.dispatch(|| {
        all_floats.par_chunks_mut(floats_per_task)
            .enumerate()
            .for_each(|(task_idx, float_chunk)| {
                let start_block_idx = task_idx * blocks_per_task;
                
                // We iterate based on the actual size of the current float_chunk
                // This naturally handles the last (smaller) chunk
                for (block_in_task_idx, f_sub_chunk) in float_chunk.chunks_exact_mut(T::CHUNK_SIZE).enumerate() {
                    let block_idx = start_block_idx + block_in_task_idx;
                    
                    if block_idx < num_blocks {
                        let b_start = block_idx * T::PACKED_SIZE;
                        let block_bytes = &weight_data[b_start..b_start + T::PACKED_SIZE];
                        let block = T::from_bytes(block_bytes);
                        block.dequantize(f_sub_chunk);
                    }
                }
            });
    });

    Ok(all_floats)
}

