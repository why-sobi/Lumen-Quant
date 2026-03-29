//! ## Lumen-Quant Loader
//! This module handles the mechanical-sympathy layer of the engine,
//! specifically managing MMap and Cache-Line iteration.

use memmap2::Mmap;
use std::fs::File;

/// A high-performance loader that memory-maps model weights from disk.
/// 
/// This struct ensures that we stay within RAM constraints by streaming data.
pub struct ModelLoader {
    mmap: Mmap,
}

impl ModelLoader {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: We assume the file is not being modified externally.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    pub fn get_data(&self) -> &[u8] {
        &self.mmap
    }

    // Creates an iterator that yields 'size' bytes at a time
    pub fn chunk_iterator(&self, size: usize) -> ChunkIterator<'_> {
        ChunkIterator {
            data: &self.mmap,
            pos: 0,
            size,
        }
    }
}

pub struct ChunkIterator<'a> { 
    // The 'a tells the compiler that this struct has reference to some other 
    // data hence do not delete this until the data is deleted solving the dangling reference problem
    data: &'a [u8],
    pos: usize,
    size: usize,
}

impl<'a> Iterator for ChunkIterator<'a> { 
    // by this definition of impl we tell to compiler that we are implementing the Iterator trait for the struct ChunkIterator
    // giving us the ability to use all the methods and functionalities of the Iterator trait for our struct

    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            None
        } else {
            let end = std::cmp::min(self.pos + self.size, self.data.len());
            let chunk = &self.data[self.pos..end];
            self.pos = end;
            Some(chunk)
        }
    }
}


// Some => The next method returns Some(chunk) if there is more data to read, and None when we have reached the end of the data.
// Ok   => The function succeeded
// Err  => The function failed with an error


// Important Notes about ModelLoader:
// It uses u8 instead of f32 to represent the data because it is designed to be a low-level loader, reading raw bytes from disk 
// and then convert them to appropriate data types (like f32) in main processing loop to keep it generic and flexible for different types of data.