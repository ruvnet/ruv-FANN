// cognitive_neural_architectures.rs - Neural architectures for cognitive patterns

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[wasm_bindgen]
pub struct CognitiveNeuralArchitectures {
    pattern_architectures: HashMap<String, NeuralArchitectureTemplate>,
    adaptation_strategies: HashMap<String, AdaptationStrategy>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralArchitectureTemplate {
    pub pattern_name: String,
    pub architecture_type: String,
    pub layer_configs: Vec<LayerConfig>,
    pub optimization_config: OptimizationConfig,
    pub specializations: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layer_type: String,
    pub units: usize,
    pub activation: String,
    pub dropout: f32,
    pub regularization: Option<RegularizationConfig>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub optimizer: String,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub gradient_clipping: Option<f32>,
    pub early_stopping_patience: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    pub l1_penalty: f32,
    pub l2_penalty: f32,
    pub activity_regularizer: Option<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AdaptationStrategy {
    pub strategy_name: String,
    pub adaptation_rate: f32,
    pub exploration_factor: f32,
    pub memory_retention: f32,
}

#[derive(Serialize, Deserialize)]
pub struct ConvergentNeuralArchitecture {
    pub encoder: EncoderConfig,
    pub processor: ProcessorConfig,
    pub decoder: DecoderConfig,
    pub training_config: TrainingConfiguration,
}

#[derive(Serialize, Deserialize)]
pub struct EncoderConfig {
    pub layers: Vec<LayerSpec>,
    pub attention_mechanism: Option<AttentionConfig>,
}

#[derive(Serialize, Deserialize)]
pub struct ProcessorConfig {
    pub recurrent_layers: Vec<RecurrentSpec>,
    pub residual_connections: bool,
    pub layer_normalization: bool,
}

#[derive(Serialize, Deserialize)]
pub struct DecoderConfig {
    pub layers: Vec<LayerSpec>,
    pub output_activation: String,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingConfiguration {
    pub optimizer: String,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub gradient_clipping: Option<f32>,
    pub early_stopping_patience: usize,
}

#[derive(Serialize, Deserialize)]
pub struct LayerSpec {
    pub neurons: usize,
    pub activation: String,
    pub dropout: f32,
}

#[derive(Serialize, Deserialize)]
pub struct RecurrentSpec {
    pub cell_type: String,
    pub units: usize,
    pub return_sequences: bool,
}

#[derive(Serialize, Deserialize)]
pub struct AttentionConfig {
    pub attention_type: String,
    pub num_heads: usize,
    pub key_dim: usize,
}

#[derive(Serialize, Deserialize)]
pub struct DivergentNeuralArchitecture {
    pub parallel_paths: Vec<PathConfig>,
    pub fusion_mechanism: FusionConfig,
    pub regularization: RegularizationConfig,
}

#[derive(Serialize, Deserialize)]
pub struct PathConfig {
    pub path_name: String,
    pub layers: Vec<LayerSpec>,
}

#[derive(Serialize, Deserialize)]
pub struct FusionConfig {
    pub fusion_type: String,
    pub fusion_layers: Vec<LayerSpec>,
}

#[derive(Serialize, Deserialize)]
pub struct HybridCognitiveArchitecture {
    pub component_patterns: Vec<String>,
    pub integration_strategy: IntegrationStrategy,
    pub shared_representations: SharedRepresentationConfig,
    pub meta_controller: MetaControllerConfig,
}

#[derive(Serialize, Deserialize)]
pub enum IntegrationStrategy {
    DynamicGating,
    AttentionWeighted,
    HierarchicalMerge,
    AdaptiveBlending,
}

#[derive(Serialize, Deserialize)]
pub struct SharedRepresentationConfig {
    pub embedding_size: usize,
    pub shared_layers: usize,
    pub pattern_specific_layers: usize,
}

#[derive(Serialize, Deserialize)]
pub struct MetaControllerConfig {
    pub controller_type: String,
    pub decision_network: Vec<LayerSpec>,
}

#[wasm_bindgen]
impl CognitiveNeuralArchitectures {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut architectures = CognitiveNeuralArchitectures {
            pattern_architectures: HashMap::new(),
            adaptation_strategies: HashMap::new(),
        };
        
        architectures.initialize_architectures();
        architectures.initialize_adaptation_strategies();
        
        architectures
    }
    
    #[wasm_bindgen]
    pub fn get_convergent_architecture(&self) -> JsValue {
        let architecture = ConvergentNeuralArchitecture {
            // Deep, focused network for analytical tasks
            encoder: EncoderConfig {
                layers: vec![
                    LayerSpec { neurons: 512, activation: "relu".to_string(), dropout: 0.1 },
                    LayerSpec { neurons: 256, activation: "relu".to_string(), dropout: 0.1 },
                    LayerSpec { neurons: 128, activation: "relu".to_string(), dropout: 0.0 },
                ],
                attention_mechanism: Some(AttentionConfig {
                    attention_type: "self".to_string(),
                    num_heads: 8,
                    key_dim: 64,
                }),
            },
            processor: ProcessorConfig {
                recurrent_layers: vec![
                    RecurrentSpec { cell_type: "LSTM".to_string(), units: 128, return_sequences: true },
                    RecurrentSpec { cell_type: "GRU".to_string(), units: 64, return_sequences: false },
                ],
                residual_connections: true,
                layer_normalization: true,
            },
            decoder: DecoderConfig {
                layers: vec![
                    LayerSpec { neurons: 64, activation: "relu".to_string(), dropout: 0.0 },
                    LayerSpec { neurons: 32, activation: "relu".to_string(), dropout: 0.0 },
                ],
                output_activation: "sigmoid".to_string(),
            },
            training_config: TrainingConfiguration {
                optimizer: "adam".to_string(),
                learning_rate: 0.001,
                batch_size: 32,
                gradient_clipping: Some(1.0),
                early_stopping_patience: 10,
            },
        };
        
        serde_wasm_bindgen::to_value(&architecture).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_divergent_architecture(&self) -> JsValue {
        let architecture = DivergentNeuralArchitecture {
            // Wide, exploratory network for creative tasks
            parallel_paths: vec![
                PathConfig {
                    path_name: "exploration".to_string(),
                    layers: vec![
                        LayerSpec { neurons: 1024, activation: "swish".to_string(), dropout: 0.2 },
                        LayerSpec { neurons: 512, activation: "gelu".to_string(), dropout: 0.2 },
                    ],
                },
                PathConfig {
                    path_name: "synthesis".to_string(),
                    layers: vec![
                        LayerSpec { neurons: 768, activation: "tanh".to_string(), dropout: 0.3 },
                        LayerSpec { neurons: 384, activation: "sigmoid".to_string(), dropout: 0.2 },
                    ],
                },
                PathConfig {
                    path_name: "innovation".to_string(),
                    layers: vec![
                        LayerSpec { neurons: 896, activation: "relu6".to_string(), dropout: 0.25 },
                        LayerSpec { neurons: 448, activation: "elu".to_string(), dropout: 0.15 },
                    ],
                },
            ],
            fusion_mechanism: FusionConfig {
                fusion_type: "attention_weighted".to_string(),
                fusion_layers: vec![
                    LayerSpec { neurons: 512, activation: "relu".to_string(), dropout: 0.1 },
                    LayerSpec { neurons: 256, activation: "relu".to_string(), dropout: 0.0 },
                ],
            },
            regularization: RegularizationConfig {
                l1_penalty: 0.00001,
                l2_penalty: 0.0001,
                activity_regularizer: Some(0.00001),
            },
        };
        
        serde_wasm_bindgen::to_value(&architecture).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_systems_architecture(&self) -> JsValue {
        let architecture = serde_json::json!({
            "architecture_name": "Systems Thinking Network",
            "modules": {
                "relationship_mapper": {
                    "layers": [
                        {"neurons": 256, "activation": "tanh", "dropout": 0.1},
                        {"neurons": 128, "activation": "relu", "dropout": 0.1}
                    ]
                },
                "holistic_processor": {
                    "layers": [
                        {"neurons": 384, "activation": "sigmoid", "dropout": 0.15},
                        {"neurons": 192, "activation": "tanh", "dropout": 0.1}
                    ]
                },
                "integration_layer": {
                    "neurons": 96,
                    "activation": "softmax",
                    "attention_heads": 4
                }
            },
            "connections": "bidirectional_with_skip",
            "memory_cells": 64,
            "training": {
                "optimizer": "adamw",
                "learning_rate": 0.005,
                "weight_decay": 0.001
            }
        });
        
        serde_wasm_bindgen::to_value(&architecture).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_critical_architecture(&self) -> JsValue {
        let architecture = serde_json::json!({
            "architecture_name": "Critical Analysis Network",
            "validation_layers": [
                {"neurons": 128, "activation": "relu", "dropout": 0.05},
                {"neurons": 64, "activation": "relu", "dropout": 0.05}
            ],
            "error_detection": {
                "sensitivity_threshold": 0.1,
                "anomaly_detection": true,
                "confidence_scoring": true
            },
            "evaluation_metrics": ["accuracy", "precision", "recall", "f1"],
            "training": {
                "optimizer": "sgd",
                "learning_rate": 0.003,
                "momentum": 0.9,
                "nesterov": true
            }
        });
        
        serde_wasm_bindgen::to_value(&architecture).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_lateral_architecture(&self) -> JsValue {
        let architecture = serde_json::json!({
            "architecture_name": "Lateral Thinking Network",
            "innovation_paths": [
                {
                    "path": "unconventional",
                    "layers": [
                        {"neurons": 512, "activation": "swish", "dropout": 0.3},
                        {"neurons": 256, "activation": "gelu", "dropout": 0.25}
                    ]
                },
                {
                    "path": "cross_domain",
                    "layers": [
                        {"neurons": 448, "activation": "tanh", "dropout": 0.35},
                        {"neurons": 224, "activation": "sigmoid", "dropout": 0.2}
                    ]
                }
            ],
            "lateral_connections": {
                "cross_path_attention": true,
                "random_projections": 32,
                "sparsity": 0.7
            },
            "training": {
                "optimizer": "rmsprop",
                "learning_rate": 0.015,
                "decay": 0.9,
                "epsilon": 1e-8
            }
        });
        
        serde_wasm_bindgen::to_value(&architecture).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn create_hybrid_cognitive_architecture(&self, patterns: Vec<String>) -> Result<JsValue, JsValue> {
        // Create hybrid architecture combining multiple cognitive patterns
        let hybrid = HybridCognitiveArchitecture {
            component_patterns: patterns.clone(),
            integration_strategy: IntegrationStrategy::DynamicGating,
            shared_representations: SharedRepresentationConfig {
                embedding_size: 256,
                shared_layers: 2,
                pattern_specific_layers: 3,
            },
            meta_controller: MetaControllerConfig {
                controller_type: "neural_gating".to_string(),
                decision_network: vec![
                    LayerSpec { neurons: 128, activation: "relu".to_string(), dropout: 0.1 },
                    LayerSpec { neurons: 64, activation: "softmax".to_string(), dropout: 0.0 },
                ],
            },
        };
        
        Ok(serde_wasm_bindgen::to_value(&hybrid).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_architecture_for_pattern(&self, pattern: &str) -> Result<JsValue, JsValue> {
        match pattern {
            "convergent" => Ok(self.get_convergent_architecture()),
            "divergent" => Ok(self.get_divergent_architecture()),
            "systems" => Ok(self.get_systems_architecture()),
            "critical" => Ok(self.get_critical_architecture()),
            "lateral" => Ok(self.get_lateral_architecture()),
            _ => Err(JsValue::from_str(&format!("Unknown cognitive pattern: {}", pattern))),
        }
    }
    
    #[wasm_bindgen]
    pub fn get_adaptation_strategy(&self, pattern: &str) -> Result<JsValue, JsValue> {
        let strategy = self.adaptation_strategies.get(pattern)
            .ok_or_else(|| JsValue::from_str(&format!("No adaptation strategy for pattern: {}", pattern)))?;
        
        Ok(serde_wasm_bindgen::to_value(strategy).unwrap())
    }
    
    fn initialize_architectures(&mut self) {
        // Convergent architecture
        self.pattern_architectures.insert("convergent".to_string(), NeuralArchitectureTemplate {
            pattern_name: "convergent".to_string(),
            architecture_type: "feedforward_attention".to_string(),
            layer_configs: vec![
                LayerConfig {
                    layer_type: "dense".to_string(),
                    units: 512,
                    activation: "relu".to_string(),
                    dropout: 0.1,
                    regularization: Some(RegularizationConfig {
                        l1_penalty: 0.0,
                        l2_penalty: 0.01,
                        activity_regularizer: None,
                    }),
                },
                LayerConfig {
                    layer_type: "attention".to_string(),
                    units: 256,
                    activation: "linear".to_string(),
                    dropout: 0.0,
                    regularization: None,
                },
                LayerConfig {
                    layer_type: "dense".to_string(),
                    units: 128,
                    activation: "relu".to_string(),
                    dropout: 0.1,
                    regularization: None,
                },
            ],
            optimization_config: OptimizationConfig {
                optimizer: "adam".to_string(),
                learning_rate: 0.001,
                batch_size: 32,
                gradient_clipping: Some(1.0),
                early_stopping_patience: 10,
            },
            specializations: vec!["optimization".to_string(), "analysis".to_string()],
        });
        
        // Add other pattern architectures...
    }
    
    fn initialize_adaptation_strategies(&mut self) {
        self.adaptation_strategies.insert("convergent".to_string(), AdaptationStrategy {
            strategy_name: "focused_refinement".to_string(),
            adaptation_rate: 0.1,
            exploration_factor: 0.1,
            memory_retention: 0.95,
        });
        
        self.adaptation_strategies.insert("divergent".to_string(), AdaptationStrategy {
            strategy_name: "exploratory_adaptation".to_string(),
            adaptation_rate: 0.3,
            exploration_factor: 0.4,
            memory_retention: 0.8,
        });
        
        self.adaptation_strategies.insert("systems".to_string(), AdaptationStrategy {
            strategy_name: "holistic_integration".to_string(),
            adaptation_rate: 0.2,
            exploration_factor: 0.25,
            memory_retention: 0.9,
        });
        
        self.adaptation_strategies.insert("critical".to_string(), AdaptationStrategy {
            strategy_name: "analytical_refinement".to_string(),
            adaptation_rate: 0.15,
            exploration_factor: 0.15,
            memory_retention: 0.92,
        });
        
        self.adaptation_strategies.insert("lateral".to_string(), AdaptationStrategy {
            strategy_name: "innovative_exploration".to_string(),
            adaptation_rate: 0.35,
            exploration_factor: 0.5,
            memory_retention: 0.75,
        });
    }
}