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