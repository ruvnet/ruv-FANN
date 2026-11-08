use wasm_bindgen::prelude::*;

const MAX_PARAMETERS: usize = 1_000_000;
const MAX_LAYERS: usize = 64;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    SymmetricSigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Swish,
    Gaussian,
    Elliot,
    SymmetricElliot,
    Sine,
    Cosine,
    SinSymmetric,
    CosSymmetric,
    ThresholdSymmetric,
    Threshold,
    StepSymmetric,
    Step,
}

/// Dense f64 inference with checked dimensions and a bounded parameter count.
#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    layers: Vec<usize>,
    weights: Vec<f64>,
    biases: Vec<f64>,
    activation: ActivationFunction,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new(layers: &[usize], activation: ActivationFunction) -> Result<Self, JsValue> {
        Self::try_new(layers, activation).map_err(|error| JsValue::from_str(&error))
    }

    pub fn randomize_weights(&mut self, min: f64, max: f64) -> Result<(), JsValue> {
        if !min.is_finite() || !max.is_finite() || min > max || !(max - min).is_finite() {
            return Err(JsValue::from_str("Weight range must be finite and ordered"));
        }
        for value in self.weights.iter_mut().chain(self.biases.iter_mut()) {
            *value = min + (max - min) * js_sys::Math::random();
        }
        Ok(())
    }

    pub fn set_weights(&mut self, weights: &[f64]) -> Result<(), JsValue> {
        self.set_weights_checked(weights)
            .map_err(|error| JsValue::from_str(&error))
    }

    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    pub fn run(&self, inputs: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.run_checked(inputs)
            .map_err(|error| JsValue::from_str(&error))
    }
}

impl WasmNeuralNetwork {
    fn try_new(layers: &[usize], activation: ActivationFunction) -> Result<Self, String> {
        if !(2..=MAX_LAYERS).contains(&layers.len()) || layers.contains(&0) {
            return Err("A network needs 2..=64 nonzero layers".into());
        }
        let weights = layers
            .windows(2)
            .try_fold(0usize, |sum, pair| {
                sum.checked_add(pair[0].checked_mul(pair[1])?)
            })
            .ok_or("Network dimensions overflow")?;
        let biases = layers[1..]
            .iter()
            .try_fold(0usize, |sum, size| sum.checked_add(*size))
            .ok_or("Network dimensions overflow")?;
        if weights
            .checked_add(biases)
            .is_none_or(|count| count > MAX_PARAMETERS)
        {
            return Err("Network exceeds the 1,000,000 parameter limit".into());
        }
        Ok(Self {
            layers: layers.to_vec(),
            weights: vec![0.0; weights],
            biases: vec![0.0; biases],
            activation,
        })
    }

    fn set_weights_checked(&mut self, weights: &[f64]) -> Result<(), String> {
        if weights.len() != self.weights.len() || !weights.iter().all(|value| value.is_finite()) {
            return Err("Weights must have the exact network shape and be finite".into());
        }
        self.weights.copy_from_slice(weights);
        Ok(())
    }

    fn run_checked(&self, inputs: &[f64]) -> Result<Vec<f64>, String> {
        if inputs.len() != self.layers[0] || !inputs.iter().all(|value| value.is_finite()) {
            return Err("Inputs must have the exact network shape and be finite".into());
        }
        let width = *self.layers.iter().max().unwrap();
        let mut current = Vec::with_capacity(width);
        current.extend_from_slice(inputs);
        let mut next = Vec::with_capacity(width);
        let mut weight_offset = 0;
        let mut bias_offset = 0;
        for pair in self.layers.windows(2) {
            let (previous, output) = (pair[0], pair[1]);
            next.clear();
            for neuron in 0..output {
                let mut sum = self.biases[bias_offset + neuron];
                for (index, value) in current.iter().enumerate() {
                    sum += value * self.weights[weight_offset + index * output + neuron];
                }
                if !sum.is_finite() {
                    return Err("Inference overflowed; normalize inputs and weights".into());
                }
                next.push(self.apply_activation(sum));
            }
            weight_offset += previous * output;
            bias_offset += output;
            std::mem::swap(&mut current, &mut next);
        }
        Ok(current)
    }

    fn apply_activation(&self, x: f64) -> f64 {
        let sigmoid = |value: f64| {
            if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exponential = value.exp();
                exponential / (1.0 + exponential)
            }
        };
        match self.activation {
            ActivationFunction::Linear => x,
            ActivationFunction::Sigmoid => sigmoid(x),
            ActivationFunction::SymmetricSigmoid => 2.0 * sigmoid(x) - 1.0,
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            ActivationFunction::Swish => x * sigmoid(x),
            ActivationFunction::Gaussian => (-x * x).exp(),
            ActivationFunction::Elliot => 0.5 * (x / (1.0 + x.abs())) + 0.5,
            ActivationFunction::SymmetricElliot => x / (1.0 + x.abs()),
            ActivationFunction::Sine => 0.5 * x.sin() + 0.5,
            ActivationFunction::Cosine => 0.5 * x.cos() + 0.5,
            ActivationFunction::SinSymmetric => x.sin(),
            ActivationFunction::CosSymmetric => x.cos(),
            ActivationFunction::ThresholdSymmetric | ActivationFunction::StepSymmetric => {
                if x >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            ActivationFunction::Threshold | ActivationFunction::Step => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_shapes_before_allocation() {
        for shape in [
            &[][..],
            &[1][..],
            &[0, 1][..],
            &[usize::MAX, 2][..],
            &[1001, 1001][..],
        ] {
            assert!(WasmNeuralNetwork::try_new(shape, ActivationFunction::Linear).is_err());
        }
    }

    #[test]
    fn dense_inference_preserves_layout_and_f64_precision() {
        let mut network =
            WasmNeuralNetwork::try_new(&[2, 2, 1], ActivationFunction::Linear).unwrap();
        // Input-major layout: first layer outputs [x0+3*x1, 2*x0+4*x1].
        network
            .set_weights_checked(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        assert_eq!(network.run_checked(&[1.0, 2.0]).unwrap(), vec![95.0]);
        let mut precise = WasmNeuralNetwork::try_new(&[1, 1], ActivationFunction::Linear).unwrap();
        precise.set_weights_checked(&[1.000000000001]).unwrap();
        assert_eq!(precise.run_checked(&[1.0]).unwrap(), vec![1.000000000001]);
    }

    #[test]
    fn rejects_invalid_values_without_mutating_weights() {
        let mut network = WasmNeuralNetwork::try_new(&[1, 1], ActivationFunction::Linear).unwrap();
        network.set_weights_checked(&[2.0]).unwrap();
        assert!(network.set_weights_checked(&[f64::NAN]).is_err());
        assert!(network.set_weights_checked(&[]).is_err());
        assert_eq!(network.get_weights(), vec![2.0]);
        assert!(network.run_checked(&[]).is_err());
        assert!(network.run_checked(&[f64::INFINITY]).is_err());
        assert!(network.run_checked(&[f64::MAX]).is_err());
    }

    #[test]
    fn bounded_activations_match_documented_ranges() {
        let mut network = WasmNeuralNetwork::try_new(&[1, 1], ActivationFunction::Elliot).unwrap();
        assert_eq!(network.apply_activation(0.0), 0.5);
        network.activation = ActivationFunction::SymmetricElliot;
        assert!(network.apply_activation(-1000.0) >= -1.0);
        network.activation = ActivationFunction::SinSymmetric;
        assert_eq!(network.apply_activation(0.0), 0.0);
        network.activation = ActivationFunction::Threshold;
        assert_eq!(network.apply_activation(0.0), 1.0);
    }
}
