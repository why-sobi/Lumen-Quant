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

        for chunk in loader.chunk_iterator(T::CHUNK_SIZE * 4) { // *4 since f32 is 4 bytes and CHUNK_SIZE is in terms of number of floats
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
        
        // Use T::PACKED_SIZE (e.g., 20) instead of hardcoded 20
        let num_blocks = weight_data.len() / T::PACKED_SIZE;
        let mut all_floats = Vec::with_capacity(num_blocks * T::CHUNK_SIZE);

        for raw_block in weight_data.chunks_exact(T::PACKED_SIZE) {
            let block = T::from_bytes(raw_block); // You'll add this to the trait!
            all_floats.extend_from_slice(&block.dequantize());
        }

        Ok(all_floats)

    }
}
