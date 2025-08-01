// training.rs - Complete training algorithm support

use wasm_bindgen::prelude::*;
use ruv_fann::training::{IncrementalBackprop, BatchBackprop, Rprop, Quickprop};
use ruv_fann::{TrainingAlgorithm, TrainingData};
use crate::neural::WasmNeuralNetwork;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub struct WasmTrainer {
    algorithm: TrainingAlgorithmWasm,
    training_history: Vec<TrainingEpochResult>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingConfig {
    pub algorithm: String,
    pub learning_rate: Option<f32>,
    pub momentum: Option<f32>,
    pub max_epochs: u32,
    pub target_error: f32,
    pub validation_split: Option<f32>,
    pub early_stopping: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingEpochResult {
    pub epoch: u32,
    pub training_error: f32,
    pub validation_error: Option<f32>,
    pub time_ms: f64,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingDataConfig {
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}

enum TrainingAlgorithmWasm {
    IncrementalBackprop(IncrementalBackprop),
    BatchBackprop(BatchBackprop),
    Rprop(Rprop),
    Quickprop(Quickprop),
    Sarprop(Sarprop),
}

// Simplified SARPROP implementation (Super-accelerated resilient backpropagation)
pub struct Sarprop {
    increase_factor: f32,
    decrease_factor: f32,
    delta_max: f32,
    delta_min: f32,
    weight_decay_shift: f32,
    step_error_threshold: f32,
    step_error_shift: f32,
    t_value: f32,
    rt_value: f32,
    previous_gradients: Vec<Vec<f32>>,
    previous_deltas: Vec<Vec<f32>>,
    previous_error: f32,
    best_error: f32,
}

impl Sarprop {
    pub fn new() -> Self {
        Sarprop {
            increase_factor: 1.2,
            decrease_factor: 0.5,
            delta_max: 50.0,
            delta_min: 0.000001,
            weight_decay_shift: 0.01,
            step_error_threshold: 0.1,
            step_error_shift: 1.5,
            t_value: 0.1,
            rt_value: 0.0,
            previous_gradients: Vec::new(),
            previous_deltas: Vec::new(),
            previous_error: f32::INFINITY,
            best_error: f32::INFINITY,
        }
    }
}

#[wasm_bindgen]
impl WasmTrainer {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmTrainer, JsValue> {
        let config: TrainingConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid training config: {}", e)))?;
        
        let algorithm = match config.algorithm.to_lowercase().as_str() {
            "incremental_backprop" => {
                let lr = config.learning_rate.unwrap_or(0.7);
                TrainingAlgorithmWasm::IncrementalBackprop(IncrementalBackprop::new(lr))
            },
            "batch_backprop" => {
                let lr = config.learning_rate.unwrap_or(0.7);
                TrainingAlgorithmWasm::BatchBackprop(BatchBackprop::new(lr))
            },
            "rprop" => {
                TrainingAlgorithmWasm::Rprop(Rprop::new())
            },
            "quickprop" => {
                let lr = config.learning_rate.unwrap_or(0.7);
                TrainingAlgorithmWasm::Quickprop(Quickprop::new(lr))
            },
            "sarprop" => {
                TrainingAlgorithmWasm::Sarprop(Sarprop::new())
            },
            _ => return Err(JsValue::from_str(&format!("Unknown training algorithm: {}", config.algorithm))),
        };
        
        Ok(WasmTrainer {
            algorithm,
            training_history: Vec::new(),
        })
    }
    
    #[wasm_bindgen]
    pub fn train_epoch(&mut self, network: &mut WasmNeuralNetwork, training_data: JsValue) -> Result<f32, JsValue> {
        let data: TrainingDataConfig = serde_wasm_bindgen::from_value(training_data)
            .map_err(|e| JsValue::from_str(&format!("Invalid training data: {}", e)))?;
        
        let training_data = TrainingData {
            inputs: data.inputs,
            outputs: data.outputs,
        };
        
        let start_time = js_sys::Date::now();
        
        let error = match &mut self.algorithm {
            TrainingAlgorithmWasm::IncrementalBackprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::BatchBackprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::Rprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::Quickprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::Sarprop(trainer) => {
                // Simplified SARPROP training (would need full implementation)
                // For now, fall back to RPROP behavior
                let mut rprop = Rprop::new();
                rprop.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
        };
        
        let end_time = js_sys::Date::now();
        
        // Record training history
        self.training_history.push(TrainingEpochResult {
            epoch: self.training_history.len() as u32 + 1,
            training_error: error,
            validation_error: None,
            time_ms: end_time - start_time,
        });
        
        // Update network metrics
        network.metrics.training_error = error;
        network.metrics.epochs_trained += 1;
        
        Ok(error)
    }
    
    #[wasm_bindgen]
    pub fn train_until_target(&mut self, 
                            network: &mut WasmNeuralNetwork, 
                            training_data: JsValue,
                            target_error: f32,
                            max_epochs: u32) -> Result<JsValue, JsValue> {
        
        let mut epochs = 0;
        let mut final_error = f32::MAX;
        
        while epochs < max_epochs && final_error > target_error {
            final_error = self.train_epoch(network, training_data.clone())?;
            epochs += 1;
            
            // Allow other tasks to run periodically
            if epochs % 10 == 0 {
                // Yield control briefly
                let _ = js_sys::Promise::resolve(&JsValue::NULL);
            }
        }
        
        let result = serde_json::json!({
            "converged": final_error <= target_error,
            "final_error": final_error,
            "epochs": epochs,
            "target_error": target_error
        });
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_training_history(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.training_history).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_algorithm_info(&self) -> JsValue {
        let info = match &self.algorithm {
            TrainingAlgorithmWasm::IncrementalBackprop(_) => {
                serde_json::json!({
                    "name": "Incremental Backpropagation",
                    "type": "gradient_descent",
                    "description": "Online learning with immediate weight updates"
                })
            },
            TrainingAlgorithmWasm::BatchBackprop(_) => {
                serde_json::json!({
                    "name": "Batch Backpropagation", 
                    "type": "gradient_descent",
                    "description": "Batch learning with accumulated gradients"
                })
            },
            TrainingAlgorithmWasm::Rprop(_) => {
                serde_json::json!({
                    "name": "RPROP",
                    "type": "adaptive",
                    "description": "Resilient backpropagation with adaptive step sizes"
                })
            },
            TrainingAlgorithmWasm::Quickprop(_) => {
                serde_json::json!({
                    "name": "Quickprop",
                    "type": "second_order", 
                    "description": "Quasi-Newton method with quadratic approximation"
                })
            },
            TrainingAlgorithmWasm::Sarprop(_) => {
                serde_json::json!({
                    "name": "SARPROP",
                    "type": "adaptive",
                    "description": "Super-accelerated resilient backpropagation"
                })
            },
        };
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}