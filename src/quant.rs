//! # Quantization Kernels
//! 
//! This module contains the mathematical logic for transforming f32 weights
//! into 4-bit compressed representations.

use half::f16;
use std::mem;

pub trait QuantizedBlock { 
    // This gives us a structure how we'll define all our quantization schemes (Q4_0, Q4_1, etc). 
    // It ensures consistency and makes it easy to add new schemes in the future.
    // By defining this trait, we can write generic code that operates on any quantized block type,
    // as long as it implements the QuantizedBlock trait. This is a powerful feature of Rust's type system that promotes code reuse and abstraction.
    
    // These are "baked in" at compile time. 
    // Every block type must define how many floats it eats 
    // and how many bytes it spits out.
    const CHUNK_SIZE: usize; // is in terms of number of floats
    const PACKED_SIZE: usize; // is in terms of number of bytes after quantization (e.g., 20 for Q4_0)

    // Any struct that "implements" this trait MUST have these functions.
    fn quantize(input: &[f32]) -> Self;
    fn dequantize(&self, output: &mut [f32]);
    // #[warn(dead_code)]
    // fn as_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn write_bytes(&self, dest: &mut [u8]);
}

/// A quantized block of 32 weights.
/// Total size: 2 bytes (f16 scale) + 16 bytes (16 * 1 byte) = 18 bytes.
pub struct BlockQ4_0 {
    pub scale: f16,        // The scaling factor for this block
    pub weights: [u8; 32], // 32 weights packed into 16 bytes

    // [u8; 16] means an array of u8 (unsigned 8-bit integers = 1 byte) with a fixed length of 16. 
    // we are using 1 byte to store 2 weights (4 bits each) so 16 bytes can store 32 weights.
}

impl QuantizedBlock for BlockQ4_0 {
    const CHUNK_SIZE: usize = 64;
    const PACKED_SIZE: usize = 34; // 2 (scale) + 16 (weights)

    // const CHUNK_SIZE: usize = 32;
    // const PACKED_SIZE: usize = 18; // 2 (scale) + 16 (weights)
    // also change pub weights: [u8; 16] in the struct definition to match the new packed size.
    
    /// Quantizes a slice of 32 f32s into a single Q4_0 block.
    #[inline(always)] // This is a hot function, we want to inline it for performance.
    fn quantize(input: &[f32]) -> Self {
        // Since we are quantizing 32 weights into 16 bytes, we need to ensure the input slice has exactly 32 f32 values.
        assert_eq!(input.len(), Self::CHUNK_SIZE, "Block size must be exactly {}", Self::CHUNK_SIZE);

        // 1. Find Max Absolute Value
        let max_abs = input.iter()
                .map(|&val| val.abs())
                .fold(0.0f32, f32::max);

        // 2. Calculate Scale
        let scale = max_abs / 8.0; // We divide by 8 because we want to map the range [-max_abs, max_abs] to [-8, 7] (4 bits)
        // !! IMPORTANT !!
        // We want to map the range [-max_abs, max_abs] to [-8, 7] (4 bits) 
        // This way a single weight (f32 => using 32 bits) can be represented as a 4-bit value (half a byte or u8/2) with a scale factor to recover the original range.
        // This way we are saving approx 8x space (32 bits to 4 bits) at the cost of some precision, which is the essence of quantization.

        // to avoid division by zero later on, we check here and use inv_scale later for quantization.
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 }; 
        // 3. Quantize and Pack

        let mut packed_weights  = [0u8; Self::CHUNK_SIZE / 2]; // an array of 16 u8 values intialized to 0.
        
        for i in 0..Self::CHUNK_SIZE / 2 {
            // We take two floats and pack them into one byte
            // first as i8 because we need to handle negative values (and it is the smallest signed int type) and then we will clamp it to the 4-bit range.
            let v0 = (input[i * 2]     * inv_scale).round().clamp(-8.0, 7.0) as i8; 
            let v1 = (input[i * 2 + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8;

            // Clamp to 4-bit range (-8 to 7) & the 0x0F masks prevents any garbage bits in higher 4-bits from affecting our quantized value.
            // -8 to 7 because we want to represent both negative and positive values, and with 4 bits we can represent 16 values total (from -8 to 7).
            let q0 = v0 as u8 & 0x0F; // 0x0F = 00001111 in binary, this masks the lower 4 bits to ensure we only keep the quantized value for q0.
            let q1 = v1 as u8 & 0x0F; // 0x0F = 00001111 in binary, this masks the lower 4 bits to ensure we only keep the quantized value for q1.

            // Pack: q0 in low bits, q1 in high bits
            packed_weights[i] = q0 | (q1 << 4); // this is simply pushing q0 to the lower 4 bits and q1 to the higher 4 bits of the same byte.

            // u8: 8 bits = 1 byte      => 00000000
            // q0: 4 bits = 1/2 byte    =>     q0
            // q1: 4 bits = 1/2 byte    =>     q1
            // q1 << 4: shifts q1 to the left by 4 bits => q10000
            // q0 | (q1 << 4): combines q0 and q1 into a single byte => q1q0
        }

        BlockQ4_0 {
            scale: f16::from_f32(scale),
            weights: packed_weights,
        }
    }

    #[inline(always)]
    fn dequantize(&self, output: &mut [f32]) {
        let scale = self.scale;
        for i in 0..Self::CHUNK_SIZE / 2 {
            let byte = self.weights[i];
            let q0 = ((byte & 0x0F) as i8 ^ 8) - 8;
            let q1 = ((byte >> 4) as i8 ^ 8) - 8;
            
            output[i * 2]     = q0 as f32 * scale.to_f32();
            output[i * 2 + 1] = q1 as f32 * scale.to_f32();
        }
    }
    
    #[inline(always)]
    fn write_bytes(&self, dest: &mut [u8]) {
        assert!(dest.len() >= Self::PACKED_SIZE, "Destination buffer is too small for BlockQ4_0");
        
        dest[0..mem::size_of_val(&self.scale)].copy_from_slice(&self.scale.to_le_bytes());
        dest[mem::size_of_val(&self.scale)..Self::PACKED_SIZE].copy_from_slice(&self.weights);
    }

    /// Converts the quantized block into a byte array for storage or transmission.
    // // fn as_bytes(&self) -> Vec<u8> {
    //     let mut buf = Vec::with_capacity(Self::PACKED_SIZE);   
    //     // extend_from_slice simply appends two arrays of bytes to the buffer. 
    //     buf.extend_from_slice(&self.scale.to_le_bytes()); // le => little endian
    //     buf.extend_from_slice(&self.weights);
    //     buf
    // }

    /// Converts the byte array back into a quantized block.
    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::PACKED_SIZE, "Invalid byte length for BlockQ4_0");

        let scale = f16::from_le_bytes(bytes[0..2].try_into().unwrap());
        let mut weights = [0u8; Self::CHUNK_SIZE / 2]; // 16 bytes for 32 weights
        weights.copy_from_slice(&bytes[2..Self::PACKED_SIZE]);
        
        BlockQ4_0 { scale, weights }
    }

}
