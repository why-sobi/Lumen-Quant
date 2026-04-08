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
    let report = test::run_benchmark::<BlockQ4_0>("models/all-MiniLM-L6-v2.safetensors", "lumen-models/all-MiniLM-L6-v2_q4.lumen")?;

    println!("\nFinal Report:");
    println!("MSE Loss:        {:.8}", report.mse);
    println!("Compression:     {:.2}x", report.compression_ratio);
    println!("Encoding Time:   {}ms", report.encoding_time_ms);
    println!("Decoding Time:   {}ms", report.decoding_time_ms);


    Ok(())
}