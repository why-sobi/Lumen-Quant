use crate::{ quant::BlockQ4_0 };

mod loader;
mod quant;
mod file;
mod benchmark;
mod cli;

use file::lumen;
use benchmark::test;

fn main() -> std::io::Result<()> {
    let mut name: String = "Nandi-Mini-150M".to_string();
    let report = test::run_benchmark::<BlockQ4_0>(&name, &format!("models/{}.safetensors", name), &format!("lumen-models/{}_q4_0.lumen", name))?;
    
    name = "all-MiniLM-L6-v2".to_string();
    let report2 = test::run_benchmark::<BlockQ4_0>(&name, &format!("models/{}.safetensors", name), &format!("lumen-models/{}_q4_0.lumen", name))?;
    
    println!("{}\n", report);
    println!("{}", report2);
    
    Ok(())
}

// use cli::{Cli, Commands};
// use clap::Parser;
// use std::io::{Error, ErrorKind};
// 
// fn main() -> std::io::Result<()> {
//     let args = Cli::parse();
// 
//     match args.command {
//         Commands::Bench { input } => {
//             let model_name = input
//                 .file_stem()
//                 .and_then(|s| s.to_str())
//                 .unwrap_or("UnknownModel")
//                 .to_string();
// 
//             let mut output_path = input.clone();
//             output_path.set_extension("lumen");
// 
//             println!("--- LUMEN-QUANT BENCHMARK ---");
//             // Call your benchmark
//             let report = test::run_benchmark::<BlockQ4_0>(
//                 &model_name,
//                 &input.to_string_lossy(),
//                 &output_path.to_string_lossy(),
//             )?;
// 
//             println!("\n{}", report);
//         }
// 
//         Commands::Encode { input, output } => {
//             println!("--- ENCODING PROCESS ---");
//             println!("Input:  {:?}", input);
//             println!("Output: {:?}", output);
// 
//             // 1. Initialize the ModelLoader
//             let loader = loader::ModelLoader::load(&input.to_string_lossy())
//                 .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to load model: {}", e)))?;
// 
//             // 2. Pass the loader and the output path string to encode
//             println!("Quantizing weights to Q4_0...");
//             lumen::encode::<BlockQ4_0>(&loader, &output.to_string_lossy())
//                 .map_err(|e| Error::new(ErrorKind::Other, format!("Encoding failed: {}", e)))?;
// 
//             println!("Successfully saved quantized model to {:?}", output);
//         }
//     }
// 
//     Ok(())
// }
