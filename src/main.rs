mod loader;
mod quant; // Include our new math module

use loader::ModelLoader;
use quant::quantize_block_32;

fn main() -> std::io::Result<()> {
    let path = "weights.bin";
    let loader = ModelLoader::open(path)?;

    // 128 bytes = 32 floats (f32)
    let block_bytes = 128;
    let mut compressed_size = 0;

    println!(">>> Quantizing {} ...", path);

    for chunk in loader.chunk_iterator(block_bytes) {
        // Convert the raw u8 bytes into f32 values
        // Safe because our mock generator wrote f32s
        let floats: Vec<f32> = chunk
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        if floats.len() == 32 {
            let _block = quantize_block_32(&floats);
            compressed_size += 18; // 18 bytes per BlockQ4_0
        }
    }

    println!("Original Size:   {} MB", loader.get_data().len() / 1024 / 1024);
    println!("Quantized Size:  {} MB", compressed_size / 1024 / 1024);
    println!("Compression:     8x (roughly)");

    Ok(())
}