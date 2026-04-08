pub mod lumen {
    // To find ModelLoader, we need to look into the loader module, we do that by using "crate" which essentially asks the main.rs file 
    // to look into the loader.rs file and find the ModelLoader struct and its associated methods and bring it into local scope.

    use crate::loader::ModelLoader;
    use crate::quant::{ QuantizedBlock }; // to make generic encoder/decoder work we need this trait in scope
    use std::io::{BufWriter, Write, Result, Error, ErrorKind};

    const MAGIC: &[u8; 4] = b"LUMN";
    const VERSION: u32 = 2056; // Arbitrary version number for our format

    /// ENCODE: Takes a raw weights loader and spits out a .lumen file
    pub fn encode<T: QuantizedBlock>(loader: &ModelLoader, output_path: &str) -> Result<usize> {
        let file = std::fs::File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;

        let mut bytes_written = 8;

        for chunk in loader.chunk_iterator(T::CHUNK_SIZE) { // *4 since f32 is 4 bytes and CHUNK_SIZE is in terms of number of floats (this is already handled in the chunk_iterator method, so we just pass T::CHUNK_SIZE here)
            let (_, mid, _) = unsafe { chunk.align_to::<f32>() };
            if mid.len() == T::CHUNK_SIZE {
                let block_bytes = T::quantize(mid).as_bytes();
                
                writer.write_all(&block_bytes)?;
                bytes_written += block_bytes.len();
            }
        }
        writer.flush()?;
        Ok(bytes_written)
    }

    /// DECODE: Takes a .lumen loader and returns a flat Vec of original floats
    pub fn decode<T: QuantizedBlock>(loader: &ModelLoader) -> Result<Vec<f32>> {
        let data = loader.get_data();
        
        // 1. Validate Header
        if data.len() < 8 || &data[0..4] != MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid LUMEN file"));
        }
        
        if data[4..8] != VERSION.to_le_bytes() {
            return Err(Error::new(ErrorKind::InvalidData, "Unsupported version"));
        }

        let weight_data = &data[8..];
        
        // Use T::PACKED_SIZE (e.g., 20) instead of hardcoded 20 to determine how many bytes to read for each block, and T::CHUNK_SIZE (e.g., 32) to determine how many floats to output for each block.
        let num_blocks = weight_data.len() / T::PACKED_SIZE;
        let mut all_floats = vec![0.0f32; num_blocks * T::CHUNK_SIZE]; // have to use vector here because we don't know the size at compile time, but we can calculate it based on the number of blocks and the chunk size.
        // We are pre-allocating a vector of floats that will hold all the dequantized values. The total number of floats is the number of blocks multiplied by the chunk size (number of floats per block).

        // 3. The Loop
        for (i, raw_block) in weight_data.chunks_exact(T::PACKED_SIZE).enumerate() {
            let block = T::from_bytes(raw_block);
            
            // Get the specific "window" for this block
            let start = i * T::CHUNK_SIZE;
            let end = start + T::CHUNK_SIZE;
            
            // Pass the slice directly! No block_buffer needed.
            block.dequantize(&mut all_floats[start..end]);
        }

        Ok(all_floats)
        
    }
}
