//! # Quantization Kernels
//! 
//! This module contains the mathematical logic for transforming f32 weights
//! into 4-bit compressed representations.

/// A quantized block of 32 weights.
/// Total size: 2 bytes (f16 scale) + 16 bytes (32 * 4-bits) = 18 bytes.
pub struct BlockQ4_0 {
    pub scale: f32,       // The scaling factor for this block
    pub weights: [u8; 16], // 32 weights packed into 16 bytes

    // [u8; 16] means an array of u8 (unsigned 8-bit integers = 1 byte) with a fixed length of 16. 
    // we are using 1 byte to store 2 weights (4 bits each) so 16 bytes can store 32 weights.
}

/// Quantizes a slice of 32 f32s into a single Q4_0 block.
pub fn quantize_block_32(input: &[f32]) -> BlockQ4_0 {
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