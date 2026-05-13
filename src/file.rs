
// To find ModelLoader, we need to look into the loader module, we do that by using "crate" which essentially asks the main.rs file 
// to look into the loader.rs file and find the ModelLoader struct and its associated methods and bring it into local scope.
use std::io::{Seek, Write, Result};
use rayon::prelude::*; // for parallel processing in decode

pub mod encoder;
pub mod block_buffer;
use crate::loader::ModelLoader;
use crate::quant::{ QuantizedBlock }; // to make generic encoder/decoder work we need this trait in scope


const MAGIC: &[u8; 4] = b"LUMN";
const VERSION: u32 = 2056; // Arbitrary version number for our format


pub fn encode<T: QuantizedBlock + Send + Sync>(loader: &ModelLoader, output_path: &str) -> Result<usize> {
    let mut file = std::fs::File::create(output_path)?;
    
    // 1. Write Header (Magic + Version + Tensor Count)
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(loader.tensor_ranges.len() as u32).to_le_bytes())?;

    // 2. Write the Index Table
    // Entry size: ID(4) + Start(8) + End(8) + LumnSize(8) = 28 bytes per entry
    for (i, (src_start, src_end)) in loader.tensor_ranges.iter().enumerate() {
        let elements = (src_end - src_start) / 4; 
        let num_blocks = elements / T::CHUNK_SIZE;
        let lumn_size = num_blocks * T::PACKED_SIZE;

        file.write_all(&(i as u32).to_le_bytes())?;
        file.write_all(&(*src_start as u64).to_le_bytes())?;
        file.write_all(&(*src_end as u64).to_le_bytes())?;
        file.write_all(&(lumn_size as u64).to_le_bytes())?;
    }

    // 3. Padding to 32-byte boundary (GGUF standard alignment)
    // We use a simple bitwise alignment for the stream position
    let pos = file.stream_position()?;
    let aligned_pos = (pos + 31) & !31;
    let padding_needed = (aligned_pos - pos) as usize;
    
    if padding_needed > 0 {
        let padding = [0u8; 32];
        file.write_all(&padding[..padding_needed])?;
    }

    // 4. Hardware Dispatch
    encoder_dispatch::<T>(loader, file)
}

/// Helper to handle the target_arch logic so encode() stays readable
fn encoder_dispatch<T: QuantizedBlock + Send + Sync>(loader: &ModelLoader, file: std::fs::File) -> Result<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            if rayon::current_num_threads() > 2 {
                println!("[ENCODER] RUNNING AVX2 PARALLEL");
                return encoder::encode_parallel::<T>(loader, file);
            }
            println!("[ENCODER] RUNNING AVX2 SCALAR");
            return encoder::encode_scalar::<T>(loader, file);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            println!("[ENCODER] RUNNING NEON SCALAR");
            return encoder::encode_neon_serial::<T>(loader, file);
        }
    }

    println!("[ENCODER] FALLBACK TO SCALAR");
    encoder::encode_scalar::<T>(loader, file)
}

/// DECODE: Takes a .lumen loader and returns a flat Vec of original floats
pub fn decode<T: QuantizedBlock>(loader: &ModelLoader) -> Result<Vec<f32>> {
    let data = loader.get_data();
    
    // 1. Calculate total elements across all tensor ranges
    let total_elements: usize = loader.tensor_ranges.iter()
        .map(|(start, end)| {
            let bytes = end - start;
            let num_blocks = bytes / T::PACKED_SIZE;
            num_blocks * T::CHUNK_SIZE
        })
        .sum();

    let mut all_floats = vec![0.0f32; total_elements];
    let mut current_float_offset = 0;

    // 2. Process each tensor range individually
    for &(start, end) in &loader.tensor_ranges {
        let weight_data = &data[start..end];
        let num_blocks = weight_data.len() / T::PACKED_SIZE;
        let num_floats = num_blocks * T::CHUNK_SIZE;

        // Slice out the part of all_floats that belongs to this tensor
        let float_slice = &mut all_floats[current_float_offset..current_float_offset + num_floats];

        let blocks_per_task = 512;
        let floats_per_task = blocks_per_task * T::CHUNK_SIZE;
        let arch = pulp::Arch::new();

        arch.dispatch(|| {
            float_slice.par_chunks_mut(floats_per_task)
                .enumerate()
                .for_each(|(task_idx, float_chunk)| {
                    let start_block_idx = task_idx * blocks_per_task;
                    
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

        current_float_offset += num_floats;
    }

    Ok(all_floats)
}

