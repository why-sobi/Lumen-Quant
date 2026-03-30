use crate::loader::ModelLoader;

mod loader;
mod quant; // Include our new math module
mod file;

use file::lumen;

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
    // let path = "weights.bin";
    // generate_mock_weights(path)

    let mut loader: ModelLoader = ModelLoader::open("weights.bin")?;
    lumen::encode(&loader, "model_Q4_0.lumen")?;
    
    loader = ModelLoader::open("model_Q4_0.lumen")?;
    let decoded_weights = lumen::decode(&loader)?;
    println!("Decoded {} weights successfully!", decoded_weights.len());
    
    Ok(())
}