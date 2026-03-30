pub mod lumen {
    // To find ModelLoader, we need to look into the loader module, we do that by using "crate" which essentially asks the main.rs file 
    // to look into the loader.rs file and find the ModelLoader struct and its associated methods and bring it into local scope.

    use crate::loader::ModelLoader;
    use crate::quant::{BlockQ4_0, QuantizedBlock};
    use std::fs::File;
    use std::io::{BufWriter, Write, Result, Error, ErrorKind};

    const MAGIC: &[u8; 4] = b"LUMN";
    const VERSION: u32 = 2056; // Arbitrary version number for our format

    /// ENCODE: Takes a raw weights loader and spits out a .lumen file
    pub fn encode(loader: &ModelLoader, output_path: &str) -> Result<usize> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // 1. Write Header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;

        let mut bytes_written = 8;

        // 2. Process Blocks
        for chunk in loader.chunk_iterator(128) {
            let (_, mid, _) = unsafe { chunk.align_to::<f32>() };
            if mid.len() == 32 {
                let block_bytes = BlockQ4_0::quantize(mid).as_bytes();

                writer.write_all(&block_bytes)?;
                bytes_written += block_bytes.len();
            }
        }
        
        writer.flush()?;
        Ok(bytes_written)
    }

    /// DECODE: Takes a .lumen loader and returns a flat Vec of original floats
    pub fn decode(loader: &ModelLoader) -> Result<Vec<f32>> {
        let data = loader.get_data();
        
        // 1. Validate Header
        if data.len() < 8 || &data[0..4] != MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid LUMEN file"));
        }
        
        if data[4..8] != VERSION.to_le_bytes() {
            return Err(Error::new(ErrorKind::InvalidData, "Unsupported version"));
        }

        
        // chunk_iterator(20) because Q4_0 blocks are 20 bytes
        // We skip the first 8 bytes of the file (the header)
        let weight_data = &data[8..];
        
        // 2. Process Blocks (Starting from offset 8)
        let num_blocks = weight_data.len() / 20; // each block is 20 bytes
        let mut all_floats = Vec::with_capacity(num_blocks * 32); // pre-allocate space for all the floats we will recover (32 floats per block)


        for raw_block in weight_data.chunks_exact(20) {
            let block = BlockQ4_0::from_bytes(raw_block);

            all_floats.extend_from_slice(&block.dequantize());
        }

        Ok(all_floats)
    }
}
