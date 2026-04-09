use crate::lumen;
use crate::loader::ModelLoader;
use crate::quant::QuantizedBlock;
use std::time::Instant;
use std::fmt;

pub struct QuantReport {
    pub name: String,
    pub mse: f32,
    pub compression_ratio: f32,
    pub encoding_time_ms: u128,
    pub decoding_time_ms: u128,
    pub throughput_gbps: f64,
}

// This "overloads" the println! behavior for the report
impl fmt::Display for QuantReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "--- MODEL BENCHMARK---\n\
             Model:           {}\n\
             MSE Loss:        {:.8}\n\
             Compression:     {:.2}x\n\
             Encoding Time:   {}ms\n\
             Decoding Time:   {}ms\n\
             Throughput:      {:.2} GB/s",
            self.name, self.mse, self.compression_ratio, 
            self.encoding_time_ms, self.decoding_time_ms, self.throughput_gbps
        )
    }
}

pub fn run_benchmark<T: QuantizedBlock>(name: &str, input_path: &str, output_path: &str) -> std::io::Result<QuantReport> {
    let loader = ModelLoader::load(input_path)?;

    // 1. Gather Original Data
    let mut original_floats = Vec::new();
    for chunk in loader.chunk_iterator(T::CHUNK_SIZE) {
        original_floats.extend_from_slice(chunk);
    }

    // 2. Encode
    let start_enc = Instant::now();
    let bytes_written = lumen::encode::<T>(&loader, output_path)?;
    let encoding_time = start_enc.elapsed().as_millis();

    // 3. Decode (Force the work)
    let lumen_loader = ModelLoader::load(output_path)?;
    let start_dec = Instant::now();
    let decoded_floats = lumen::decode::<T>(&lumen_loader)?;
    let duration = start_dec.elapsed();
    let decoding_time_ms = duration.as_millis();

    // 4. Calculate Stats
    let mse = original_floats.iter()
        .zip(decoded_floats.iter())
        .map(|(o, d)| (o - d).powi(2))
        .sum::<f32>() / original_floats.len() as f32;

    let compression_ratio = (original_floats.len() as f32 * 4.0) / (bytes_written as f32);
    
    // Throughput based on the compressed data read from "disk/mmap"
    let throughput = (bytes_written as f64 / 1_000_000_000.0) / duration.as_secs_f64();

    Ok(QuantReport {
        name: name.to_string(),
        mse,
        compression_ratio,
        encoding_time_ms: encoding_time,
        decoding_time_ms: decoding_time_ms,
        throughput_gbps: throughput,
    })
}