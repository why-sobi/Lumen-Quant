use std::fs::File;
use std::io::Result;
use std::sync::mpsc::sync_channel;
use std::thread;

use crate::loader::ModelLoader;
use crate::quant::block::QuantizedBlock;
use super::block_buffer::BlockBuffer; // Assuming the import we discussed

pub fn encode_scalar<T: QuantizedBlock>(loader: &ModelLoader, writer: File) -> Result<usize> {
    let mut bytes_written = 8; // MAGIC + VERSION
    
    // Initialize your new BlockBuffer
    // block_count 1024 is a good middle ground for L3 cache/RAM staging
    let mut buffer = BlockBuffer::new(writer, 1024, T::PACKED_SIZE);

    for chunk in loader.chunk_iterator(T::CHUNK_SIZE) {
        let (_, mid, _) = unsafe { chunk.align_to::<f32>() };
        if mid.len() == T::CHUNK_SIZE {
            let block = T::quantize(mid);
            
            // Abstracted logic: fills internal Vec, dispatches when full
            buffer.fill_and_dispatch(block)?;
            bytes_written += T::PACKED_SIZE;
        }
    }

    buffer.finalize()?;
    Ok(bytes_written)
}

pub fn encode_parallel<T>(loader: &ModelLoader, writer: File) -> Result<usize> 
where 
    T: QuantizedBlock + Send
{
    let (tx, rx) = sync_channel::<T>(10000);

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
            drop(tx); 
        });

        // --- CONSUMER ---
        let mut bytes_written = 8;
        let mut buffer = BlockBuffer::new(writer, 1024, T::PACKED_SIZE);

        for block in rx {
            // Logic remains identical to scalar, but consuming from the channel
            buffer.fill_and_dispatch(block)?;
            bytes_written += T::PACKED_SIZE;
        }
        
        buffer.finalize()?;
        Ok(bytes_written)
    })
}