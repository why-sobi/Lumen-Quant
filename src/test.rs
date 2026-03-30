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
    let loader = ModelLoader::open(input_path)?;
    let original_bytes = loader.get_data();
    let (_, original_floats, _) = unsafe { original_bytes.align_to::<f32>() };
    
    // 2. Encode (Timed)
    let mut start_time = Instant::now();
    let bytes_written = lumen::encode::<T>(&loader, output_path)?;
    let encoding_time = start_time.elapsed().as_millis();

    // 3. Decode
    let lumen_loader = ModelLoader::open(output_path)?;
    start_time = Instant::now();
    let decoded_floats = lumen::decode::<T>(&lumen_loader)?;
    let decoding_time = start_time.elapsed().as_millis();

    // 4. Calculate Accuracy (MSE)
    let mut sum_sq_error = 0.0;
    for (orig, deco) in original_floats.iter().zip(decoded_floats.iter()) {
        let diff = orig - deco;
        sum_sq_error += diff * diff;
    }
    let mse = sum_sq_error / original_floats.len() as f32;

    // 5. Calculate Compression
    let compression_ratio = (original_bytes.len() as f32) / (bytes_written as f32);

    Ok(QuantReport {
        mse,
        compression_ratio,
        encoding_time_ms: encoding_time,
        decoding_time_ms: decoding_time
    })
}