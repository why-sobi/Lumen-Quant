struct ModelWeights {
    weights: Vec<f32>,
}

impl ModelWeights {
    fn new(weights: Vec<f32>) -> Self {
        Self { weights }
    }

    fn max_weight(&self) -> f32 {
        self.weights.iter().copied().fold(f32::MIN, |a, b| a.max(b))
    }

    fn min_weight(&self) -> f32 {
        self.weights.iter().copied().fold(f32::MAX, |a, b| a.min(b))
    }

    fn calculate_scale(&self) -> f32 {
        let max = self.max_weight();
        let min = self.min_weight();
        if max == min { 1.0 } else { (max - min) / 255.0 }
    }

    // The core "Transformation" logic
    fn quantize(&self) -> Vec<u8> {
        let scale: f32 = self.calculate_scale();
        let min:   f32 = self.min_weight();

        self.weights
            .iter()
            .map(|&w| {
                // Formula: (weight - min) / scale
                let q = (w - min) / scale;
                q.round() as u8 // Cast to unsigned 8-bit int
            })
            .collect() // This gathers the results back into a Vec<u8>
    }
}

fn main() {
    println!(">>>\t\tLUMEN-QUANT PROTOTYPE\t\t<<<");

    // Mock weights (e.g., from a neural network layer)
    let raw_data: Vec<f32>      = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let m_weights: ModelWeights = ModelWeights::new(raw_data);

    let scale: f32 = m_weights.calculate_scale();
    let quantized: Vec<u8> = m_weights.quantize();

    println!("Scale Factor: {:.4}", scale);
    println!("Original:  {:?}", m_weights.weights);
    println!("Quantized: {:?}", quantized);
    
    // Quick verification: The first element should be 0, last should be 255
    if let (Some(&first), Some(&last)) = (quantized.first(), quantized.last()) {
        println!("Range Check: Min={} (expected 0), Max={} (expected 255)", first, last);
    }
}