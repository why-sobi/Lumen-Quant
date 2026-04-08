use crate::lumen;
use crate::loader::ModelLoader;
use crate::quant::QuantizedBlock;
use std::time::Instant;

pub struct QuantReport {
    pub mse: f32,
    pub compression_ratio: f32,
    pub encoding_time_ms: u128,
    pub decoding_time_ms: u128
}

pub fn run_benchmark<T: QuantizedBlock>(input_path: &str, output_path: &str) -> std::io::Result<QuantReport> {
    // 1. Setup
    let loader = ModelLoader::load(input_path)?;

    let mut original_floats = Vec::new();
    for chunk in loader.chunk_iterator(T::CHUNK_SIZE) {
        original_floats.extend_from_slice(chunk);
    }

    // 2. Encode
    let mut start_time = Instant::now();
    let bytes_written = lumen::encode::<T>(&loader, output_path)?;
    let encoding_time = start_time.elapsed().as_millis();

    // 3. Decode
    start_time = Instant::now();
    let lumen_loader = ModelLoader::load(output_path)?;
    let decoded_floats = lumen::decode::<T>(&lumen_loader)?;
    let decoding_time = start_time.elapsed().as_millis();

    // 4. Calculate Accuracy (Now lengths will match!)
    let mse = original_floats.iter()
        .zip(decoded_floats.iter())
        .map(|(o, d)| (o - d).powi(2))
        .sum::<f32>() / original_floats.len() as f32;

    // 5. Calculate Compression
    let compression_ratio = (original_floats.len() as f32 * 4.0) / (bytes_written as f32);

    Ok(QuantReport {
        mse,
        compression_ratio,
        encoding_time_ms: encoding_time,
        decoding_time_ms: decoding_time
    })
}