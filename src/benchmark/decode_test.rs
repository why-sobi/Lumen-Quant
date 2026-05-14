use std::time::Instant;
use crate::file;
use crate::loader::ModelLoader;
use crate::quant::block::QuantizedBlock;

pub fn test_decode<T: QuantizedBlock + Send + Sync>(model_path: &str) -> std::io::Result<()> {
    let loader = ModelLoader::load(model_path)?;

    let start_dec = Instant::now();
    let decoded_floats = file::decode::<T>(&loader)?;
    let duration = start_dec.elapsed().as_millis();

    println!("Decoded {} floats in {} ms | {} GB/s", decoded_floats.len(), duration, (std::mem::size_of_val(&decoded_floats[0]) * decoded_floats.len()) as f64 / (duration as f64 * 1e6));
    Ok(())
}