use crate::loader::ModelLoader;

mod loader;
mod quant; // Include our new math module
mod encoder;

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

    let loader: ModelLoader = ModelLoader::open("weights.bin")?;
    let mut encoder = encoder::ModelEncoder::new("model_Q4_0.lumen")?;
    encoder.encode_file(&loader)?;

    Ok(())
}