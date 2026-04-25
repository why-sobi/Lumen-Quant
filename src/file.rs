pub mod lumen {
    // To find ModelLoader, we need to look into the loader module, we do that by using "crate" which essentially asks the main.rs file 
    // to look into the loader.rs file and find the ModelLoader struct and its associated methods and bring it into local scope.

    use crate::loader::ModelLoader;
    use crate::quant::{ QuantizedBlock }; // to make generic encoder/decoder work we need this trait in scope
    use std::io::{BufWriter, Write, Result};
    use rayon::prelude::*; // for parallel processing in decode

    const MAGIC: &[u8; 4] = b"LUMN";
    const VERSION: u32 = 2056; // Arbitrary version number for our format

    /// ENCODE: Takes a raw weights loader and spits out a .lumen file
    // pub fn encode<T: QuantizedBlock>(loader: &ModelLoader, output_path: &str) -> Result<usize> {
    //     let file = std::fs::File::create(output_path)?;
    //     let mut writer = BufWriter::new(file);

    //     writer.write_all(MAGIC)?;
    //     writer.write_all(&VERSION.to_le_bytes())?;

    //     let mut bytes_written = 8;
    //     let mut block_buffer = vec![0u8; T::PACKED_SIZE];

    //     for chunk in loader.chunk_iterator(T::CHUNK_SIZE) { // *4 since f32 is 4 bytes and CHUNK_SIZE is in terms of number of floats (this is already handled in the chunk_iterator method, so we just pass T::CHUNK_SIZE here)
    //         let (_, mid, _) = unsafe { chunk.align_to::<f32>() };
    //         if mid.len() == T::CHUNK_SIZE {
    //             let block = T::quantize(mid);
                
    //             block.write_bytes(&mut block_buffer);

    //             writer.write_all(&block_buffer)?;
    //             bytes_written += block_buffer.len();
    //         }
    //     }
    //     writer.flush()?;
    //     Ok(bytes_written)
    // }

    pub fn encode<T: QuantizedBlock>(loader: &ModelLoader, output_path: &str) -> Result<usize> {
        let file = std::fs::File::create(output_path)?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        let mut bytes_written = 8;

        let data = loader.get_data(); // &[u8] — shared read-only, fine across threads

        for (start, end) in &loader.tensor_ranges {
            let tensor_data = &data[*start..*end];
            let n_blocks = tensor_data.len() / (T::CHUNK_SIZE * 4);
            let mut tensor_output = vec![0u8; n_blocks * T::PACKED_SIZE];

            // par_chunks_mut splits tensor_output into non-overlapping mutable slices
            // each thread gets its own out_slice — no locking, no contention
            tensor_output
                .as_mut_slice()
                .par_chunks_mut(T::PACKED_SIZE)
                .enumerate()
                .for_each(|(i, out_slice)| {
                    let chunk_start = i * T::CHUNK_SIZE * 4;
                    let chunk = &tensor_data[chunk_start..chunk_start + T::CHUNK_SIZE * 4];
                    let mid = unsafe {
                        std::slice::from_raw_parts(chunk.as_ptr() as *const f32, T::CHUNK_SIZE)
                    };
                    T::quantize(mid).write_bytes(out_slice);
                });

            writer.write_all(&tensor_output)?;
            bytes_written += tensor_output.len();
        }

        writer.flush()?;
        Ok(bytes_written)
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
}
