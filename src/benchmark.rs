use crate::{quant::block::QuantizedBlock};

pub mod test; // looks for benchmark/test.rs
pub mod decode_test;

pub fn test_model<T: QuantizedBlock + Send + Sync>(name: &str, base_path: &str) -> std::io::Result<()> {
    println!("{}\n", test::run_benchmark::<T>(&name, &format!("{}/models/{}.safetensors", base_path, name), &format!("{}/lumen-models/{}_q4_0.lumen", base_path, name))?);
    Ok(())
}

pub fn test_model_streamed<T: QuantizedBlock + Send + Sync>(name: &str, base_path: &str) -> std::io::Result<()> {
    println!("{}\n", test::run_benchmark_streamed::<T>(&name, &format!("{}/models/{}.safetensors", base_path, name))?);
    Ok(())
}