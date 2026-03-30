//! # Quantization Kernels
//! 
//! This module contains the mathematical logic for transforming f32 weights
//! into 4-bit compressed representations.

pub trait QuantizedBlock { 
    // This gives us a structure how we'll define all our quantization schemes (Q4_0, Q4_1, etc). 
    // It ensures consistency and makes it easy to add new schemes in the future.
    // By defining this trait, we can write generic code that operates on any quantized block type,
    // as long as it implements the QuantizedBlock trait. This is a powerful feature of Rust's type system that promotes code reuse and abstraction.
    
    // 1. Associated Constants: 
    // These are "baked in" at compile time. 
    // Every block type must define how many floats it eats (32) 
    // and how many bytes it spits out (20).
    const CHUNK_SIZE: usize; 
    const PACKED_SIZE: usize;

    // 2. Required Methods:
    // Any struct that "implements" this trait MUST have these functions.
    fn quantize(input: &[f32]) -> Self;
    fn dequantize(&self) -> [f32; 32];
    fn as_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;
}


/// A quantized block of 32 weights.
/// Total size: 4 bytes (f32 scale) + 16 bytes (16 * 1 byte) = 20 bytes.
pub struct BlockQ4_0 {
    pub scale: f32,       // The scaling factor for this block
    pub weights: [u8; 16], // 32 weights packed into 16 bytes

    // [u8; 16] means an array of u8 (unsigned 8-bit integers = 1 byte) with a fixed length of 16. 
    // we are using 1 byte to store 2 weights (4 bits each) so 16 bytes can store 32 weights.
}

impl QuantizedBlock for BlockQ4_0 {
    const CHUNK_SIZE: usize = 32;
    const PACKED_SIZE: usize = 20; // 4 (scale) + 16 (weights)
    
    /// Quantizes a slice of 32 f32s into a single Q4_0 block.
    fn quantize(input: &[f32]) -> Self {
        // Since we are quantizing 32 weights into 16 bytes, we need to ensure the input slice has exactly 32 f32 values.
        assert_eq!(input.len(), 32, "Block size must be exactly 32"); 

        // 1. Find Max Absolute Value
        let mut max_abs = 0.0f32; // do not use f32::MIN because we're comparing the abs values.
        for &val in input { // couldve used iter().copied().fold() but this is more straightforward for the 32 values.
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }

        // 2. Calculate Scale
        let scale = max_abs / 8.0; 
        // !! IMPORTANT !!
        // We want to map the range [-max_abs, max_abs] to [-8, 7] (4 bits) 
        // This way a single weight (f32 => using 32 bits) can be represented as a 4-bit value (half a byte or u8/2) with a scale factor to recover the original range.
        // This way we are saving approx 8x space (32 bits to 4 bits) at the cost of some precision, which is the essence of quantization.

        // to avoid division by zero later on, we check here and use inv_scale later for quantization.
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 }; 
        // 3. Quantize and Pack

        let mut packed_weights  = [0u8; 16]; // an array of 16 u8 values intialized to 0.
        
        for i in 0..16 {
            // We take two floats and pack them into one byte
            // first as i8 because we need to handle negative values (and it is the smallest signed int type) and then we will clamp it to the 4-bit range.
            let v0 = (input[i * 2] * inv_scale).round() as i8; 
            let v1 = (input[i * 2 + 1] * inv_scale).round() as i8;

            // Clamp to 4-bit range (-8 to 7) & the 0x0F masks prevents any garbage bits in higher 4-bits from affecting our quantized value.
            // -8 to 7 because we want to represent both negative and positive values, and with 4 bits we can represent 16 values total (from -8 to 7).
            let q0 = v0.clamp(-8, 7) as u8 & 0x0F; // 0x0F = 00001111 in binary, this masks the lower 4 bits to ensure we only keep the quantized value for q0.
            let q1 = v1.clamp(-8, 7) as u8 & 0x0F; // 0x0F = 00001111 in binary, this masks the lower 4 bits to ensure we only keep the quantized value for q1.

            // Pack: q0 in low bits, q1 in high bits
            packed_weights[i] = q0 | (q1 << 4); // this is simply pushing q0 to the lower 4 bits and q1 to the higher 4 bits of the same byte.

            // u8: 8 bits = 1 byte      => 00000000
            // q0: 4 bits = 1/2 byte    =>     q0
            // q1: 4 bits = 1/2 byte    =>     q1
            // q1 << 4: shifts q1 to the left by 4 bits => q10000
            // q0 | (q1 << 4): combines q0 and q1 into a single byte => q1q0
        }

        BlockQ4_0 {
            scale,
            weights: packed_weights,
        }
    }
    
    /// De-quantizes a Q4_0 block back into 32 f32 weights.
    fn dequantize(&self) -> [f32; 32] {
        let mut output = [0.0f32; 32];
        let scale = self.scale;

        for i in 0..16 {
            let byte = self.weights[i];

            // 1. Extract the low 4 bits (q0) and high 4 bits (q1)
            let mut q0 = (byte & 0x0F) as i8;
            let mut q1 = (byte >> 4) as i8;

            // 2. Sign Extension Trick:
            // If the 4th bit is 1, the number is negative. 
            // We must "stretch" that sign to the 8-bit i8 level.
            if q0 > 7 { q0 -= 16; } // negative values in 4 bits are represented as 8 to 15, so we subtract 16 to get the correct negative value in i8. (15 = -1 ...)
            if q1 > 7 { q1 -= 16; }

            // 3. Scale back to f32
            output[i * 2] = q0 as f32 * scale;
            output[i * 2 + 1] = q1 as f32 * scale;
        }
        output
    }

    fn as_bytes(&self) -> Vec<u8> { // returns a vector of bytes representing the quantized block, which can be written to disk or transmitted over a network.
        let mut buf = Vec::with_capacity(Self::PACKED_SIZE);
        
        // extend_from_slice simply appends two arrays of bytes to the buffer. 
        buf.extend_from_slice(&self.scale.to_le_bytes()); // le => little endian
        buf.extend_from_slice(&self.weights);
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::PACKED_SIZE, "Invalid byte length for BlockQ4_0");

        let scale = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let mut weights = [0u8; 16];
        weights.copy_from_slice(&bytes[4..20]);
        
        BlockQ4_0 { scale, weights }
    }

}
