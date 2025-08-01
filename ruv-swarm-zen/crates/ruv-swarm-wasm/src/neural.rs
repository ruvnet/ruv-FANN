// neural.rs - Main neural network WASM interface

use wasm_bindgen::prelude::*;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction as RuvActivation, TrainingData};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    inner: Network<f32>,
    training_data: Option<TrainingData<f32>>,
    metrics: NetworkMetrics,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NetworkMetrics {
    pub training_error: f32,
    pub validation_error: f32,
    pub epochs_trained: u32,
    pub total_connections: usize,
    pub memory_usage: usize,
}

#[derive(Serialize, Deserialize)]
pub struct NetworkConfig {
    pub input_size: usize,
    pub hidden_layers: Vec<LayerConfig>,
    pub output_size: usize,
    pub output_activation: String,
    pub connection_rate: Option<f32>,
    pub random_seed: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct LayerConfig {
    pub size: usize,
    pub activation: String,
    pub steepness: Option<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingDataConfig {
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmNeuralNetwork, JsValue> {
        let config: NetworkConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
        
        let mut builder = NetworkBuilder::<f32>::new()
            .input_layer(config.input_size);
            
        // Add hidden layers with specified activation functions
        for layer in config.hidden_layers {
            builder = builder.hidden_layer_with_activation(
                layer.size,
                parse_activation_function(&layer.activation)?,
                layer.steepness.unwrap_or(1.0)
            );
        }
        
        let network = builder
            .output_layer_with_activation(
                config.output_size,
                parse_activation_function(&config.output_activation)?,
                1.0
            )
            .connection_rate(config.connection_rate.unwrap_or(1.0))
            .build();
            
        Ok(WasmNeuralNetwork {
            inner: network,
            training_data: None,
            metrics: NetworkMetrics {
                training_error: 0.0,
                validation_error: 0.0,
                epochs_trained: 0,
                total_connections: 0,
                memory_usage: 0,
            }
        })
    }
    
    #[wasm_bindgen]
    pub fn run(&mut self, inputs: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner.run(inputs)
            .map_err(|e| JsValue::from_str(&format!("Network run error: {}", e)))
    }
    
    #[wasm_bindgen]
    pub fn set_training_data(&mut self, data: JsValue) -> Result<(), JsValue> {
        let training_data: TrainingDataConfig = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("Invalid training data: {}", e)))?;
            
        self.training_data = Some(TrainingData {
            inputs: training_data.inputs,
            outputs: training_data.outputs,
        });
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn get_weights(&self) -> Vec<f32> {
        self.inner.get_weights()
    }
    
    #[wasm_bindgen]
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), JsValue> {
        self.inner.set_weights(weights)
            .map_err(|e| JsValue::from_str(&format!("Set weights error: {}", e)))
    }
    
    #[wasm_bindgen]
    pub fn get_network_info(&self) -> JsValue {
        let info = serde_json::json!({
            "num_layers": self.inner.num_layers(),
            "num_inputs": self.inner.num_inputs(),
            "num_outputs": self.inner.num_outputs(),
            "total_neurons": self.inner.total_neurons(),
            "total_connections": self.inner.total_connections(),
            "metrics": self.metrics
        });
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}

pub fn parse_activation_function(name: &str) -> Result<RuvActivation, JsValue> {
    match name.to_lowercase().as_str() {
        "linear" => Ok(RuvActivation::Linear),
        "sigmoid" => Ok(RuvActivation::Sigmoid),
        "sigmoid_symmetric" | "tanh" => Ok(RuvActivation::SigmoidSymmetric),
        "gaussian" => Ok(RuvActivation::Gaussian),
        "gaussian_symmetric" => Ok(RuvActivation::GaussianSymmetric),
        "elliot" => Ok(RuvActivation::Elliot),
        "elliot_symmetric" => Ok(RuvActivation::ElliotSymmetric),
        "relu" => Ok(RuvActivation::ReLU),
        "relu_leaky" | "leaky_relu" => Ok(RuvActivation::ReLULeaky),
        "cos" => Ok(RuvActivation::Cos),
        "cos_symmetric" => Ok(RuvActivation::CosSymmetric),
        "sin" => Ok(RuvActivation::Sin),
        "sin_symmetric" => Ok(RuvActivation::SinSymmetric),
        "threshold" => Ok(RuvActivation::Threshold),
        "threshold_symmetric" => Ok(RuvActivation::ThresholdSymmetric),
        "linear_piece" => Ok(RuvActivation::LinearPiece),
        "linear_piece_symmetric" => Ok(RuvActivation::LinearPieceSymmetric),
        _ => Err(JsValue::from_str(&format!("Unknown activation function: {}", name))),
    }
}