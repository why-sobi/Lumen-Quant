pub struct ModelWeights {
    weights: Vec<f32>,
}

impl ModelWeights {
    // ==================================== PRIVATE METHODS ===================================

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

    // ==================================== PUBLIC METHODS ====================================

    pub fn new(weights: Vec<f32>) -> Self {
        Self { weights }
    }

    // The core "Transformation" logic
    pub fn quantize(&self) -> Vec<u8> {
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