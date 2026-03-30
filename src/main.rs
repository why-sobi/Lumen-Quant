use crate::{ quant::BlockQ4_0 };

mod loader;
mod quant; // Include our new math module
mod file;
mod benchmark;

use file::lumen;
use benchmark::test;

// fn generate_mock_weights(path: &str) -> std::io::Result<()> {
//     println!("Generating 100MB mock weights at {}...", path);
//     let file = File::create(path)?;
//     let mut writer = BufWriter::new(file);

//     for i in 0..25_000_000 {
//         // Generate a pseudo-random float between -1.0 and 1.0
//         let val = (i as f32).sin(); 
//         writer.write_all(&val.to_le_bytes())?;
//     }
//     writer.flush()?;
//     println!("Mock weights generated successfully.");
//     Ok(())
// }

fn main() -> std::io::Result<()> {
    println!("--- REAL MODEL TEST: all-MiniLM-L6-v2 ---");

    // 1. Load from official Safetensors
    let original_floats = loader::ModelLoader::load_from_safetensor("models/all-MiniLM-L6-v2.safetensors")?;
    
    // 2. Create a temporary raw file for our Lumen encoder
    // (Our current encoder expects a ModelLoader pointing at a raw file)
    let raw_path = "temp_raw_weights.bin";
    std::fs::write(raw_path, unsafe {
        std::slice::from_raw_parts(
            original_floats.as_ptr() as *const u8,
            original_floats.len() * 4,
        )
    })?;

    // let loader = loader::ModelLoader::open(raw_path)?;

    // 3. Benchmark with CHUNK_SIZE = 64
    let report = test::run_benchmark::<BlockQ4_0>(&raw_path, "lumen-models/all-MiniLM-L6-v2_q4.lumen")?;

    println!("\nFinal Report:");
    println!("MSE Loss:        {:.8}", report.mse);
    println!("Compression:     {:.2}x", report.compression_ratio);
    println!("Encoding Time:   {}ms", report.encoding_time_ms);
    println!("Decoding Time:   {}ms", report.decoding_time_ms);


    // Clean up
    let _ = std::fs::remove_file(raw_path);

    Ok(())
}