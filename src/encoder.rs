// To find ModelLoader, we need to look into the loader module, we do that by using "crate" which essentially asks the main.rs file 
// to look into the loader.rs file and find the ModelLoader struct and its associated methods and bring it into local scope.

use crate::loader::ModelLoader; 
use crate::quant::{QuantizedBlock, BlockQ4_0};
use std::fs::File;
use std::io::{BufWriter, Write}; // BufWrite is a buffer to store ~8MB in RAM then write to disk in one go.

pub struct ModelEncoder {
    writer: BufWriter<File>,
}

impl ModelEncoder {
    pub fn new(output_path: &str) -> std::io::Result<Self> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        
        // Header: Magic Number + Version (4 bytes each)
        // !! IMP !! This needs to be consistent for all version of LUMEN otherwise the file format breaks and we won't be able to read it back in the future.
        writer.write_all(b"LUMN")?;
        writer.write_all(&1u32.to_le_bytes())?; 
        
        Ok(Self { writer })
    }

    pub fn encode_file(&mut self, loader: &ModelLoader) -> std::io::Result<usize> {
        let mut total_written = 8; // Header size (4 bytes for magic + 4 bytes for version)
        
        // If we were to send QuantizationType in this function, we can get rid of the magic 128 in chunk iterator
        // 128 represents the number of bytes we want to read at a time from the memory-mapped file.
        // 128 bytes => 32 f32 weights (128 / 4 = 32) which is exactly the size of one BlockQ4_0.

        // High-level streaming loop
        for chunk in loader.chunk_iterator(128) {
            let (_, mid, _) = unsafe { chunk.align_to::<f32>() };
            
            if mid.len() == 32 {
                let block = BlockQ4_0::quantize(mid);
                let bytes = block.as_bytes();
                self.writer.write_all(&bytes)?;
                total_written += bytes.len();
            }
        }
        
        self.writer.flush()?;
        Ok(total_written)
    }
}