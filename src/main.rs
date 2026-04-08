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
    let mut name: String = "Nandi-Mini-150M".to_string();
    let report = test::run_benchmark::<BlockQ4_0>(&name, &format!("models/{}.safetensors", name), &format!("lumen-models/{}_q4_0.lumen", name))?;

    name = "all-MiniLM-L6-v2".to_string();
    let report2 = test::run_benchmark::<BlockQ4_0>(&name, &format!("models/{}.safetensors", name), &format!("lumen-models/{}_q4_0.lumen", name))?;

    println!("{}\n", report);
    println!("{}", report2);

    Ok(())
}