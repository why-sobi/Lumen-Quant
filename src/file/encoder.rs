use crate::loader::ModelLoader;
use crate::quant::{ QuantizedBlock }; // to make generic encoder/decoder work we need this trait in scope

use std::fs::File;
use std::io::{BufWriter, Write, Result};
use std::thread;
use std::sync::{mpsc::sync_channel};


/// ENCODE: Takes a raw weights loader and spits out a .lumen file
pub fn encode_scalar<T: QuantizedBlock>(loader: &ModelLoader, writer: &mut BufWriter<File>) -> Result<usize> {
    
    let mut bytes_written = 8;
    let mut block_buffer = vec![0u8; T::PACKED_SIZE];

    for chunk in loader.chunk_iterator(T::CHUNK_SIZE) { // *4 since f32 is 4 bytes and CHUNK_SIZE is in terms of number of floats (this is already handled in the chunk_iterator method, so we just pass T::CHUNK_SIZE here)
        let (_, mid, _) = unsafe { chunk.align_to::<f32>() };
        if mid.len() == T::CHUNK_SIZE {
            let block = T::quantize(mid);
            
            block.write_bytes(&mut block_buffer);

            writer.write_all(&block_buffer)?;
            bytes_written += block_buffer.len();
        }
    }
    writer.flush()?;
    Ok(bytes_written)
}

pub fn encode_parallel<T>(loader: &ModelLoader, writer: &mut BufWriter<File>) -> Result<usize> 
where 
    T: QuantizedBlock + Send
{
    let (tx, rx) = sync_channel::<T>(10000); // Increased buffer

    // Scoped thread doesn't need 'static!
    thread::scope(|s| { 
        // --- PRODUCER ---
        s.spawn(|| {
            for chunk in loader.chunk_iterator(T::CHUNK_SIZE) {
                let (_, mid, _) = unsafe { chunk.align_to::<f32>() };
                if mid.len() == T::CHUNK_SIZE {
                    let block = T::quantize(mid);
                    if tx.send(block).is_err() { break; }
                }
            }
            drop(tx); // Ensure sender drops so rx loop terminates
        });

        // --- CONSUMER (Main Thread logic inside scope) ---
        let mut bytes_written = 8;
        let mut block_buffer = vec![0u8; T::PACKED_SIZE];

        for block in rx {
            block.write_bytes(&mut block_buffer);
            writer.write_all(&block_buffer)?;
            bytes_written += block_buffer.len();
        }
        
        Ok(bytes_written)
    })
}