// activation.rs - Complete activation function support

use wasm_bindgen::prelude::*;
use ruv_fann::{ActivationFunction as RuvActivation, Neuron};
use std::collections::HashMap;

#[wasm_bindgen]
pub struct ActivationFunctionManager;

#[wasm_bindgen]
impl ActivationFunctionManager {
    #[wasm_bindgen]
    pub fn get_all_functions() -> JsValue {
        let functions = vec![
            ("linear", "Linear function: f(x) = x * steepness"),
            ("sigmoid", "Sigmoid: f(x) = 1/(1+e^(-2sx))"),
            ("sigmoid_symmetric", "Symmetric sigmoid: f(x) = tanh(sx)"),
            ("tanh", "Hyperbolic tangent (alias for sigmoid_symmetric)"),
            ("gaussian", "Gaussian: f(x) = e^(-x²s²)"),
            ("gaussian_symmetric", "Symmetric Gaussian"),
            ("elliot", "Elliot function (fast sigmoid approximation)"),
            ("elliot_symmetric", "Symmetric Elliot function"),
            ("relu", "Rectified Linear Unit: f(x) = max(0, x)"),
            ("relu_leaky", "Leaky ReLU: f(x) = x > 0 ? x : 0.01x"),
            ("cos", "Cosine function"),
            ("cos_symmetric", "Symmetric cosine"),
            ("sin", "Sine function"),
            ("sin_symmetric", "Symmetric sine"),
            ("threshold", "Threshold function"),
            ("threshold_symmetric", "Symmetric threshold"),
            ("linear_piece", "Bounded linear: f(x) = max(0, min(1, x * steepness))"),
            ("linear_piece_symmetric", "Symmetric bounded linear"),
        ];
        
        serde_wasm_bindgen::to_value(&functions).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn test_activation_function(name: &str, input: f32, steepness: f32) -> Result<f32, JsValue> {
        let activation = crate::neural::parse_activation_function(name)?;
        
        // Create temporary single neuron for testing
        let mut neuron = Neuron::new(1, activation, steepness);
        
        Ok(neuron.activate(&[input]))
    }
    
    #[wasm_bindgen]
    pub fn compare_functions(input: f32) -> JsValue {
        let mut results = HashMap::new();
        
        let functions = [
            "linear", "sigmoid", "sigmoid_symmetric", "gaussian",
            "elliot", "relu", "relu_leaky", "cos", "sin", "threshold"
        ];
        
        for &func_name in &functions {
            if let Ok(result) = Self::test_activation_function(func_name, input, 1.0) {
                results.insert(func_name.to_string(), result);
            }
        }
        
        serde_wasm_bindgen::to_value(&results).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_function_properties(name: &str) -> Result<JsValue, JsValue> {
        let activation = crate::neural::parse_activation_function(name)?;
        
        let (min_output, max_output) = activation.output_range();
        let properties = serde_json::json!({
            "name": activation.name(),
            "trainable": activation.is_trainable(),
            "output_range": {
                "min": min_output,
                "max": max_output
            }
        });
        
        Ok(serde_wasm_bindgen::to_value(&properties).unwrap())
    }
}