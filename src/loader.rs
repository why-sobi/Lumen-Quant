//! ## Lumen-Quant Loader
//! This module handles the mechanical-sympathy layer of the engine,
//! specifically managing MMap and Cache-Line iteration.

use memmap2::Mmap;
use safetensors::SafeTensors; // for .safetensor model files
use gguf_rs; // for .gguf model files
pub struct ChunkIterator<'a> {
    // The 'a tells the compiler that this struct has reference to some other 
    // data hence do not delete this until the data is deleted solving the dangling reference problem
    mmap: &'a [u8],
    ranges: Vec<(usize, usize)>,
    current_range_idx: usize,
    current_offset: usize,
    chunk_size_bytes: usize,
}

impl<'a> Iterator for ChunkIterator<'a> {
    // by this definition of impl we tell to compiler that we are implementing the Iterator trait for the struct ChunkIterator
    // giving us the ability to use all the methods and functionalities of the Iterator trait for our struct
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_range_idx >= self.ranges.len() {
            return None;
        }

        let (range_start, range_end) = self.ranges[self.current_range_idx];
        let start = range_start + self.current_offset;
        let end = start + self.chunk_size_bytes;

        if end <= range_end {
            // We have enough data in the current tensor for a full chunk
            let byte_slice = &self.mmap[start..end];
            self.current_offset += self.chunk_size_bytes;
            
            // Safety: We trust the Safetensor Dtype was F32
            let (_, floats, _) = unsafe { byte_slice.align_to::<f32>() };
            Some(floats)
        } else {
            // Current tensor is exhausted, move to the next one
            self.current_range_idx += 1;
            self.current_offset = 0;
            self.next() // Recursive call to try the next range
        }
    }
}

/// A high-performance loader that memory-maps model weights from disk.
/// 
/// This struct ensures that we stay within RAM constraints by streaming data.
pub struct ModelLoader {
    mmap: Mmap,
    pub tensor_ranges: Vec<(usize, usize)> // (start, end) byte offsets for each tensor in safetensor (mainly) file
}

impl ModelLoader {
    pub fn load(path: &str) -> std::io::Result<Self> {
        // Platform-specific file open with sequential hint
        #[cfg(windows)]
        let file = {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x08000000;
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
                .open(path)?
        };
        #[cfg(not(windows))]
        let file = File::open(path)?;

        let mmap = unsafe { Mmap::map(&file)? };

        #[cfg(unix)]
        {
            use memmap2::Advice;
            let _ = mmap.advise(Advice::Sequential);
        }

        let extension = path.split('.').last().unwrap_or("");
        let mut tensor_ranges = Vec::new();

        match extension {
            "safetensor" | "safetensors" => {
                let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{:?}", e))
                })?;
                for (_, view) in st.tensors() {
                    let slice_start_ptr = view.data().as_ptr() as usize;
                    let mmap_start_ptr = mmap.as_ptr() as usize;
                    let absolute_start = slice_start_ptr - mmap_start_ptr;
                    let absolute_end = absolute_start + view.data().len();
                    tensor_ranges.push((absolute_start, absolute_end));
                }
            }
            "gguf" => {
                // GGUF has a specific header format: [Magic][Version][TensorCount][KVCount]
                // Then KVs, then Tensor Infos (which contain the offsets).
                
                // Use a GGUF parser here. Manual parsing is tricky due to 
                // variable-length strings in the header.
                let container = gguf_rs::get_gguf_container(path).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("GGUF Error: {}", e))
                })?;

                // In GGUF, tensors are stored at an offset relative to the end of the header
                for tensor in container.tensors {
                    // Using the struct fields from your prompt: offset and size
                    let absolute_start = tensor.offset as usize;
                    let absolute_end = absolute_start + tensor.size as usize;
                    
                    tensor_ranges.push((absolute_start, absolute_end));
                }
            }
            "lumen" => { tensor_ranges.push((8, mmap.len())); }
            "bin"   => { tensor_ranges.push((0, mmap.len())); }
            _       => { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Unsupported file format")); }
        }

        Ok(Self { mmap, tensor_ranges })
    }

    pub fn get_data(&self) -> &[u8] {
        &self.mmap
    }

    // Creates an iterator that yields 'size' bytes at a time
    pub fn chunk_iterator(&self, chunk_size: usize) -> ChunkIterator<'_> {
        ChunkIterator {
            mmap: &self.mmap,
            ranges: self.tensor_ranges.clone(), // We clone the Vec of offsets
            current_range_idx: 0,               // need in .next() implementation to keep track of which tensor we are currently reading from
            current_offset: 0,                  // this is the offset within the current tensor range, we will increment this by chunk_size_bytes after each chunk is read
            chunk_size_bytes: chunk_size * 4,   // Convert floats to bytes (f32 = 4 bytes)
        }
    }
}



// Some => The next method returns Some(chunk) if there is more data to read, and None when we have reached the end of the data.
// Ok   => The function succeeded
// Err  => The function failed with an error


// Important Notes about ModelLoader:
// It uses u8 instead of f32 to represent the data because it is designed to be a low-level loader, reading raw bytes from disk 
// and then convert them to appropriate data types (like f32) in main processing loop to keep it generic and flexible for different types of data.