use std::fs;
use std::time::Instant;
use std::fmt;
use rayon::prelude::*;

use crate::file;
use crate::loader::ModelLoader;
use crate::quant::block::QuantizedBlock;

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

pub fn run_benchmark<T: QuantizedBlock + Send + Sync>(name: &str, input_path: &str, output_path: &str) -> std::io::Result<QuantReport> {
    let loader = ModelLoader::load(input_path)?;

    // 1. Gather Original Data
    let mut original_floats = Vec::new();
    for chunk in loader.chunk_iterator(T::CHUNK_SIZE) {
        original_floats.extend_from_slice(chunk);
    }

    // 2. Encode
    let start_enc = Instant::now();
    let bytes_written = file::encode::<T>(&loader, output_path)?;
    let encoding_time = start_enc.elapsed().as_millis();

    let _sum: f32 = original_floats.iter().sum();
    
    // 3. Decode (Force the work)
    let lumen_loader = ModelLoader::load(output_path)?;
 
    let _warmup = std::hint::black_box(lumen_loader.get_data().iter().sum::<u8>());
 
    let start_dec = Instant::now();
    let decoded_floats = file::decode::<T>(&lumen_loader)?;
    let duration = start_dec.elapsed();
    let decoding_time_ms = duration.as_millis();

    // 4. Calculate Stats
    let mse = original_floats.iter()
        .zip(decoded_floats.iter())
        .map(|(o, d)| (o - d).powi(2))
        .sum::<f32>() / original_floats.len() as f32;

    let compression_ratio = fs::metadata(input_path)?.len() as f32 / fs::metadata(output_path)?.len() as f32;

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

pub fn run_benchmark_streamed<T: QuantizedBlock + Send + Sync>(
    name: &str, 
    input_path: &str
) -> std::io::Result<QuantReport> {
    let loader = ModelLoader::load(input_path)?;
    
    // 1. PRE-ENCODE (Timed)
    // We'll collect into a Vec<T> to simulate the data that would be on disk.
    let start_enc = Instant::now();
    let quantized_data: Vec<T> = loader.chunk_iterator(T::CHUNK_SIZE)
        .map(|chunk| T::quantize(chunk))
        .collect();
    let encoding_time = start_enc.elapsed();

    // 2. PRE-WARM (The "SSD-Killer")
    // We touch the quantized data to ensure it's in the L3 cache/RAM
    let _ = std::hint::black_box(quantized_data.iter().count());

    // 3. DECODE (Timed & Parallel)
    // This is the core "Inference Speed" test.
    let mut output_buffer = vec![0.0f32; quantized_data.len() * T::CHUNK_SIZE];
    
    let start_dec = Instant::now();
    
    output_buffer.par_chunks_mut(T::CHUNK_SIZE)
        .zip(quantized_data.par_iter())
        .for_each(|(out_chunk, block): (&mut [f32], &T)| { // Added explicit types here
            block.dequantize(out_chunk);
    });
        
    let decoding_time = start_dec.elapsed();

    // 4. MSE Calculation (Post-Timer)
    // We stream the original loader again to compare against our output_buffer
    let mut total_mse: f64 = 0.0;
    let mut idx = 0;
    for chunk in loader.chunk_iterator(T::CHUNK_SIZE) {
        let chunk: &[f32] = chunk;
        let output_chunk = &output_buffer[idx..idx + T::CHUNK_SIZE];
        
        // Explicitly annotate o and d
        for (o, d) in chunk.iter().zip(output_chunk.iter()) {
            total_mse += (*o as f64 - *d as f64).powi(2);
        }
        idx += T::CHUNK_SIZE;
    }

    // 5. Final Stats
    let total_elements = output_buffer.len();
    let final_mse = (total_mse / total_elements as f64) as f32;
    let f32_size = total_elements * 4;
    let quantized_size = quantized_data.len() * std::mem::size_of::<T>();
    
    // Throughput based on Decompressed f32 data
    let throughput = (f32_size as f64 / 1_000_000_000.0) / decoding_time.as_secs_f64();

    Ok(QuantReport {
        name: name.to_string(),
        mse: final_mse,
        compression_ratio: f32_size as f32 / quantized_size as f32,
        encoding_time_ms: encoding_time.as_millis() as u128,
        decoding_time_ms: decoding_time.as_millis() as u128,
        throughput_gbps: throughput,
    })
}