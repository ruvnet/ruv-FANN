// cascade.rs - Cascade correlation implementation

use wasm_bindgen::prelude::*;
use ruv_fann::cascade::{CascadeTrainer, CascadeConfig};
use ruv_fann::{TrainingData, ActivationFunction as RuvActivation};
use crate::neural::{WasmNeuralNetwork, TrainingDataConfig, parse_activation_function};
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub struct WasmCascadeTrainer {
    inner: Option<CascadeTrainer<f32>>,
    config: CascadeConfig,
}

#[derive(Serialize, Deserialize)]
pub struct CascadeConfigWasm {
    pub max_hidden_neurons: usize,
    pub num_candidates: usize,
    pub output_max_epochs: usize,
    pub candidate_max_epochs: usize,
    pub output_learning_rate: f32,
    pub candidate_learning_rate: f32,
    pub output_target_error: f32,
    pub candidate_target_correlation: f32,
    pub min_correlation_improvement: f32,
    pub candidate_weight_min: f32,
    pub candidate_weight_max: f32,
    pub candidate_activations: Vec<String>,
    pub verbose: bool,
}

impl Default for CascadeConfigWasm {
    fn default() -> Self {
        CascadeConfigWasm {
            max_hidden_neurons: 30,
            num_candidates: 8,
            output_max_epochs: 150,
            candidate_max_epochs: 150,
            output_learning_rate: 0.7,
            candidate_learning_rate: 0.7,
            output_target_error: 0.0001,
            candidate_target_correlation: 0.8,
            min_correlation_improvement: 0.001,
            candidate_weight_min: -1.0,
            candidate_weight_max: 1.0,
            candidate_activations: vec![
                "sigmoid".to_string(),
                "sigmoid_symmetric".to_string(),
                "gaussian".to_string(),
                "elliot".to_string(),
                "relu".to_string(),
            ],
            verbose: false,
        }
    }
}

#[wasm_bindgen]
impl WasmCascadeTrainer {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue, network: &WasmNeuralNetwork, training_data: JsValue) -> Result<WasmCascadeTrainer, JsValue> {
        let cascade_config: CascadeConfigWasm = if config.is_undefined() || config.is_null() {
            CascadeConfigWasm::default()
        } else {
            serde_wasm_bindgen::from_value(config)
                .map_err(|e| JsValue::from_str(&format!("Invalid cascade config: {}", e)))?
        };
        
        let data: TrainingDataConfig = serde_wasm_bindgen::from_value(training_data)
            .map_err(|e| JsValue::from_str(&format!("Invalid training data: {}", e)))?;
        
        let training_data = TrainingData {
            inputs: data.inputs,
            outputs: data.outputs,
        };
        
        // Parse activation functions
        let mut activations = Vec::new();
        for activation_name in &cascade_config.candidate_activations {
            let activation = parse_activation_function(activation_name)?;
            activations.push(activation);
        }
        
        let config = CascadeConfig {
            max_hidden_neurons: cascade_config.max_hidden_neurons,
            num_candidates: cascade_config.num_candidates,
            output_max_epochs: cascade_config.output_max_epochs,
            candidate_max_epochs: cascade_config.candidate_max_epochs,
            output_learning_rate: cascade_config.output_learning_rate,
            candidate_learning_rate: cascade_config.candidate_learning_rate,
            output_target_error: cascade_config.output_target_error,
            candidate_target_correlation: cascade_config.candidate_target_correlation,
            min_correlation_improvement: cascade_config.min_correlation_improvement,
            candidate_weight_range: (cascade_config.candidate_weight_min, cascade_config.candidate_weight_max),
            candidate_activations: activations,
            verbose: cascade_config.verbose,
            ..Default::default()
        };
        
        let trainer = CascadeTrainer::new(config.clone(), network.inner.clone(), training_data)
            .map_err(|e| JsValue::from_str(&format!("Cascade trainer creation error: {}", e)))?;
        
        Ok(WasmCascadeTrainer {
            inner: Some(trainer),
            config,
        })
    }
    
    #[wasm_bindgen]
    pub fn train(&mut self) -> Result<JsValue, JsValue> {
        let trainer = self.inner.take()
            .ok_or_else(|| JsValue::from_str("Trainer already consumed"))?;
        
        let result = trainer.train()
            .map_err(|e| JsValue::from_str(&format!("Cascade training error: {}", e)))?;
        
        let result_info = serde_json::json!({
            "converged": result.converged,
            "final_error": result.final_error,
            "hidden_neurons_added": result.hidden_neurons_added,
            "epochs": result.epochs,
            "training_time_ms": result.training_time.as_millis(),
            "network_structure": {
                "total_neurons": result.final_network.total_neurons(),
                "total_connections": result.final_network.total_connections(),
                "layers": result.final_network.num_layers()
            }
        });
        
        Ok(serde_wasm_bindgen::to_value(&result_info).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_config(&self) -> JsValue {
        let config_info = serde_json::json!({
            "max_hidden_neurons": self.config.max_hidden_neurons,
            "num_candidates": self.config.num_candidates,
            "output_max_epochs": self.config.output_max_epochs,
            "candidate_max_epochs": self.config.candidate_max_epochs,
            "output_learning_rate": self.config.output_learning_rate,
            "candidate_learning_rate": self.config.candidate_learning_rate,
            "output_target_error": self.config.output_target_error,
            "candidate_target_correlation": self.config.candidate_target_correlation
        });
        
        serde_wasm_bindgen::to_value(&config_info).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn create_default_config() -> JsValue {
        let default_config = CascadeConfigWasm::default();
        serde_wasm_bindgen::to_value(&default_config).unwrap()
    }
}