use half::f16;
use std::mem;
use std::arch::x86_64::*;
use super::block::QuantizedBlock;
/// A quantized block of 32 weights.
/// Total size: 2 bytes (f16 scale) + 16 bytes (16 * 1 byte) = 18 bytes.
pub struct BlockQ4_0 {
    pub scale: f16,        // The scaling factor for this block
    pub weights: [u8; 16], // 32 weights packed into 16 bytes

    // [u8; 16] means an array of u8 (unsigned 8-bit integers = 1 byte) with a fixed length of 16. 
    // we are using 1 byte to store 2 weights (4 bits each) so 16 bytes can store 32 weights.
}

impl QuantizedBlock for BlockQ4_0 {
    const CHUNK_SIZE: usize = 32;
    const PACKED_SIZE: usize = 18; // 2 (scale) + 16 (weights)

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
    fn write_bytes(&self, dest: &mut [u8]) {
        assert!(dest.len() >= Self::PACKED_SIZE, "Destination buffer is too small for BlockQ4_0");
        
        dest[0..mem::size_of_val(&self.scale)].copy_from_slice(&self.scale.to_le_bytes());
        dest[mem::size_of_val(&self.scale)..Self::PACKED_SIZE].copy_from_slice(&self.weights);
    }
    
    /// Converts the byte array back into a quantized block.
    #[inline(always)]
    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::PACKED_SIZE, "Invalid byte length for BlockQ4_0");
        
        let scale = f16::from_le_bytes(bytes[0..2].try_into().unwrap());
        let mut weights = [0u8; Self::CHUNK_SIZE / 2]; // 16 bytes for 32 weights
        weights.copy_from_slice(&bytes[2..Self::PACKED_SIZE]);
        
        BlockQ4_0 { scale, weights }
    }
    
    #[inline(always)]
    unsafe fn dequantize(bytes: &[u8], output: &mut [f32]) {
        let block = BlockQ4_0::from_bytes(bytes);
        let scale = block.scale.to_f32(); // since rust doesnt natively support f16 :/
        
        for i in 0..Self::CHUNK_SIZE / 2 {
            let byte = block.weights[i];
            let q0 = ((byte & 0x0F) as i8 ^ 8) - 8;
            let q1 = ((byte >> 4) as i8 ^ 8) - 8;
            
            output[i * 2]     = q0 as f32 * scale;
            output[i * 2 + 1] = q1 as f32 * scale;
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_avx2(bytes: &[u8], output: &mut [f32]) {
        use std::arch::x86_64::*;

        let scale = f16::from_le_bytes(bytes[0..2].try_into().unwrap()).to_f32();
        let v_scale = _mm256_set1_ps(scale);
        let xor8 = _mm_set1_epi8(8u8 as i8);
        let sub8 = _mm_set1_epi8(8u8 as i8); // for signed clamping from [0..15] to [-8..7]
        let low_mask = _mm_set1_epi8(0x0Fu8 as i8); // mask to extract low nibbles

        // load 16 raw bytes into 128-bit register
        let raw = unsafe { _mm_loadu_si128(bytes[2..].as_ptr() as *const __m128i) };
        let raw_low  = _mm_and_si128(raw, low_mask);
        let raw_high = _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask);

        // XOR then subtract — matches scalar's (nibble ^ 8) - 8
        let low  = _mm_sub_epi8(_mm_xor_si128(raw_low,  xor8), sub8);
        let high = _mm_sub_epi8(_mm_xor_si128(raw_high, xor8), sub8);

        // _mm_bsrli_si128 shifts register right by N bytes
        // so cvtepi8_epi32 (which reads lowest 8 bytes) gets bytes 8..16
        let low_hi  = _mm_bsrli_si128(low,  8);
        let high_hi = _mm_bsrli_si128(high, 8);

        // helper: widen 8xi8 → 8xi32 → 8xf32 → multiply scale → store at output[base], [base+2], ...
        // we store interleaved: low nibbles at even indices, high nibbles at odd indices
        #[inline(always)]
        unsafe fn process_and_store(src: __m128i, v_scale: __m256, output: *mut f32, base: usize) {
            let i32s = unsafe { _mm256_cvtepi8_epi32(src)};           // 8xi8 → 8xi32
            let f32s = unsafe { _mm256_cvtepi32_ps(i32s) };             // 8xi32 → 8xf32
            let res  = unsafe { _mm256_mul_ps(f32s, v_scale) };         // multiply scale

            // store to temp then scatter to interleaved positions
            // AVX2 has no native interleaved scatter so we use a temp array
            let mut tmp = [0f32; 8];
            unsafe {_mm256_storeu_ps(tmp.as_mut_ptr(), res) };
            for j in 0..8 {
                unsafe { *output.add(base + j * 2)  = tmp[j] };
            }
        }

        let out_ptr = output.as_mut_ptr();
        unsafe { process_and_store(low,     v_scale, out_ptr, 0) };  // even 0..14
        unsafe { process_and_store(high,    v_scale, out_ptr, 1) };  // odd  1..15
        unsafe { process_and_store(low_hi,  v_scale, out_ptr, 16) }; // even 16..30
        unsafe { process_and_store(high_hi, v_scale, out_ptr, 17) }; // odd  17..31
    }

    fn dispatch_decode_fn() -> unsafe fn(&[u8], &mut [f32]) {
        // This is where we will return the appropriate decode function based on the CPU features.
        // For example, if AVX2 is available, we can return a function that uses AVX2 intrinsics for faster decoding.
        // If not, we can return a scalar fallback function.

        // #[cfg(target_arch = "x86_64")]
        // {
        //     if is_x86_feature_detected!("avx2") {
        //         println!("[DECODER] DISPATCHING AVX2 DECODE");
        //         return Self::dequantize_avx2;
        //     }
        // }
        println!("[DECODER] DISPATCHING SCALAR DECODE");
        Self::dequantize as unsafe fn(&[u8], &mut [f32]) // default to scalar decode if no special features are detected
        
    } 
}

// at the bottom of quant.rs
#[cfg(test)]
mod tests {
    use super::*;  // brings BlockQ4_0 and everything else into scope

    #[test]
    fn test_avx2_matches_scalar() {
        // 1. create a known input
        let input: Vec<f32> = (0..BlockQ4_0::CHUNK_SIZE).map(|i| (i as f32 - 32.0) * 0.1).collect();
        
        // 2. quantize it
        let block = BlockQ4_0::quantize(&input);
        
        // 3. get raw bytes
        let mut bytes = vec![0u8; BlockQ4_0::PACKED_SIZE];
        block.write_bytes(&mut bytes);
        
        // 4. decode both ways
        let mut scalar_out = vec![0f32; BlockQ4_0::CHUNK_SIZE];
        let mut avx2_out   = vec![0f32; BlockQ4_0::CHUNK_SIZE];
        
        unsafe { BlockQ4_0::dequantize(&bytes,&mut scalar_out) };
        
        unsafe { BlockQ4_0::dequantize_avx2(&bytes, &mut avx2_out) };
        
        // 5. compare
        println!("scalar: {:?}", &scalar_out[..8]);
        println!("avx2:   {:?}", &avx2_out[..8]);
        println!("bytes[0..8]: {:?}", &bytes[..8]);
        println!("scale from bytes f32: {}", f32::from_le_bytes(bytes[0..4].try_into().unwrap()));
        println!("scale from bytes f16: {}", half::f16::from_le_bytes(bytes[0..2].try_into().unwrap()).to_f32());
        println!("weights start at byte 4: {:?}", &bytes[4..8]);
        
        for i in 0..BlockQ4_0::CHUNK_SIZE {
            assert!(
                (scalar_out[i] - avx2_out[i]).abs() < 1e-5,
                "mismatch at index {}: scalar={} avx2={}", 
                i, scalar_out[i], avx2_out[i]
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_avx2_vs_scalar() {
        use std::time::Instant;
        use std::hint::black_box;

        let input: Vec<f32> = (0..BlockQ4_0::CHUNK_SIZE).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let block = BlockQ4_0::quantize(&input);
        let mut bytes = vec![0u8; BlockQ4_0::PACKED_SIZE];
        block.write_bytes(&mut bytes);

        let iterations = 1_000_000;

        // scalar
        let mut out = vec![0f32; BlockQ4_0::CHUNK_SIZE];
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe { BlockQ4_0::dequantize(black_box(&bytes), black_box(&mut out)) };
        }
        let scalar_time = start.elapsed();
        black_box(&out);
        println!("scalar: {:?}", scalar_time);

        // avx2
        let mut out2 = vec![0f32; BlockQ4_0::CHUNK_SIZE];
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe { BlockQ4_0::dequantize_avx2(black_box(&bytes), black_box(&mut out2)) };
        }
        let avx2_time = start.elapsed();
        black_box(&out2);
        println!("avx2:   {:?}", avx2_time);

        println!("speedup: {:.2}x", scalar_time.as_secs_f64() / avx2_time.as_secs_f64());
    }
}