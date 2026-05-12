use std::io::{BufWriter, Write};
use crate::quant::QuantizedBlock;

pub struct BlockBuffer<W: Write> {
    writer: BufWriter<W>,
    data: Vec<u8>,
    cursor: usize,
    capacity: usize,
}

impl<W: Write> BlockBuffer<W> {
    pub fn new(writer: W, block_count: usize, max_block_size: usize) -> Self {
        let staging_capacity = block_count * max_block_size;

        debug_assert!(staging_capacity < 64 * 1024 * 1024, "Staging buffer exceeds 64MB - check block_count");
        
        // We ensure the BufWriter is at least as big as our staging area.
        // A good rule of thumb is 2x to 4x the staging size, or a minimum of 8MB.
        let io_buffer_size = std::cmp::max(staging_capacity * 2, 8 * 1024 * 1024);

        Self {
            writer: BufWriter::with_capacity(io_buffer_size, writer),
            data: vec![0u8; staging_capacity],
            cursor: 0,
            capacity: staging_capacity,
        }
    }

    #[inline(always)]
    pub fn fill_and_dispatch<T: QuantizedBlock>(&mut self, block: T) -> std::io::Result<()> {
        if self.cursor + T::PACKED_SIZE > self.capacity {
            // This is now a "clean" move: the data fits perfectly inside the BufWriter's memory
            self.writer.write_all(&self.data[..self.cursor])?;
            self.cursor = 0;
        }

        block.write_bytes(&mut self.data[self.cursor..self.cursor + T::PACKED_SIZE]);
        self.cursor += T::PACKED_SIZE;

        Ok(())
    }

    pub fn finalize(mut self) -> std::io::Result<()> {
        if self.cursor > 0 {
            self.writer.write_all(&self.data[..self.cursor])?;
        }
        self.writer.flush()
    }
}