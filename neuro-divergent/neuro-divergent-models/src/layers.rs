//! Neural network layers for forecasting models
//!
//! This module provides various layer types used in neural forecasting models.

use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Trait for neural network layers
pub trait LayerTrait {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, input: &Array2<f64>, grad_output: &Array2<f64>) -> Array2<f64>;
    fn get_params(&self) -> Vec<Array2<f64>>;
    fn get_gradients(&self) -> Vec<Array2<f64>>;
}

/// Dense (fully connected) layer
#[derive(Clone)]
pub struct Dense {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub grad_weights: Option<Array2<f64>>,
    pub grad_bias: Option<Array1<f64>>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let scale = (2.0 / input_size as f64).sqrt();

        let weights =
            Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-scale..scale));
        let bias = Array1::zeros(output_size);

        Self {
            weights,
            bias,
            grad_weights: None,
            grad_bias: None,
        }
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.bias
    }

    pub fn backward(&mut self, input: &Array2<f64>, grad_output: &Array2<f64>) -> Array2<f64> {
        // Compute weight gradients
        self.grad_weights = Some(input.t().dot(grad_output));

        // Compute bias gradients
        self.grad_bias = Some(grad_output.sum_axis(Axis(0)));

        // Compute input gradients
        grad_output.dot(&self.weights.t())
    }
}

/// Activation layer wrapper
pub struct Activation<A> {
    activation: A,
}

impl<A> Activation<A> {
    pub fn new(activation: A) -> Self {
        Self { activation }
    }
}

/// Layer normalization
pub struct LayerNorm {
    gamma: Array1<f64>,
    beta: Array1<f64>,
    eps: f64,
    grad_gamma: Option<Array1<f64>>,
    grad_beta: Option<Array1<f64>>,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Self {
        assert!(
            normalized_shape > 0 && eps.is_finite() && eps > 0.0,
            "LayerNorm requires a positive feature count and finite positive epsilon"
        );
        Self {
            gamma: Array1::ones(normalized_shape),
            beta: Array1::zeros(normalized_shape),
            eps,
            grad_gamma: None,
            grad_beta: None,
        }
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            input.ncols(),
            self.gamma.len(),
            "LayerNorm feature count mismatch"
        );
        assert!(
            input.iter().all(|value| value.is_finite()),
            "LayerNorm input must be finite"
        );
        let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let centered = input - &mean;
        // Center before squaring: subtracting E[x]^2 from E[x^2] loses the
        // entire variance for a large offset with a small, meaningful spread.
        let variance = centered
            .mapv(|x| x * x)
            .mean_axis(Axis(1))
            .unwrap()
            .insert_axis(Axis(1));

        let normalized = centered / (variance + self.eps).mapv(f64::sqrt);
        normalized * &self.gamma + &self.beta
    }
}

/// Batch normalization
pub struct BatchNorm {
    gamma: Array1<f64>,
    beta: Array1<f64>,
    running_mean: Array1<f64>,
    running_var: Array1<f64>,
    momentum: f64,
    eps: f64,
    grad_gamma: Option<Array1<f64>>,
    grad_beta: Option<Array1<f64>>,
}

impl BatchNorm {
    pub fn new(num_features: usize, momentum: f64, eps: f64) -> Self {
        assert!(
            num_features > 0
                && momentum.is_finite()
                && (0.0..=1.0).contains(&momentum)
                && eps.is_finite()
                && eps > 0.0,
            "BatchNorm requires positive features/epsilon and momentum in [0, 1]"
        );
        Self {
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            momentum,
            eps,
            grad_gamma: None,
            grad_beta: None,
        }
    }

    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Array2<f64> {
        assert_eq!(
            input.ncols(),
            self.gamma.len(),
            "BatchNorm feature count mismatch"
        );
        assert!(
            input.iter().all(|value| value.is_finite()),
            "BatchNorm input must be finite"
        );
        if training {
            assert!(
                input.nrows() > 0,
                "BatchNorm requires a nonempty training batch"
            );
            // Calculate batch statistics
            let mean = input.mean_axis(Axis(0)).unwrap();
            let centered = input - &mean;
            let var = centered.mapv(|x| x * x).mean_axis(Axis(0)).unwrap();

            // Update running statistics
            self.running_mean = &self.running_mean * (1.0 - self.momentum) + &mean * self.momentum;
            self.running_var = &self.running_var * (1.0 - self.momentum) + &var * self.momentum;

            // Normalize using batch statistics
            let normalized = centered / (var + self.eps).mapv(f64::sqrt);
            normalized * &self.gamma + &self.beta
        } else {
            // Use running statistics for inference
            let normalized = (input - &self.running_mean)
                / (self.running_var.clone() + self.eps).mapv(f64::sqrt);
            normalized * &self.gamma + &self.beta
        }
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn layer_norm_preserves_variation_on_large_offsets() {
        let layer = LayerNorm::new(3, 1e-5);
        let input = array![[1e12, 1e12 + 1.0, 1e12 + 2.0], [1e12, 1e12, 1e12]];
        let result = layer.forward(&input);
        let expected = 1.0 / (2.0_f64 / 3.0 + 1e-5).sqrt();
        for (actual, expected) in result.row(0).iter().zip([-expected, 0.0, expected]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(result.row(1).iter().all(|value| *value == 0.0));
    }

    #[test]
    fn batch_norm_centered_variance_matches_training_and_running_statistics() {
        let mut layer = BatchNorm::new(2, 1.0, 1e-5);
        let input = array![
            [1e12, 5e11],
            [1e12 + 1.0, 5e11 + 2.0],
            [1e12 + 2.0, 5e11 + 4.0]
        ];
        let training = layer.forward(&input, true);
        let inference = layer.forward(&input, false);
        assert_eq!(training, inference);
        let first = 1.0 / (2.0_f64 / 3.0 + 1e-5).sqrt();
        let second = 2.0 / (8.0_f64 / 3.0 + 1e-5).sqrt();
        assert!((training[(0, 0)] + first).abs() < 1e-12);
        assert!((training[(2, 1)] - second).abs() < 1e-12);
        assert!(training.row(1).iter().all(|value| *value == 0.0));
    }
}

/// Dropout layer
pub struct Dropout {
    rate: f64,
    rng: ChaCha8Rng,
}

impl Dropout {
    pub fn new(rate: f64, seed: u64) -> Self {
        Self {
            rate,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Array2<f64> {
        if training {
            let shape = (input.nrows(), input.ncols());
            let mask = Array2::from_shape_fn(shape, |_| {
                if self.rng.gen::<f64>() > self.rate {
                    1.0 / (1.0 - self.rate)
                } else {
                    0.0
                }
            });
            input * mask
        } else {
            input.clone()
        }
    }
}
