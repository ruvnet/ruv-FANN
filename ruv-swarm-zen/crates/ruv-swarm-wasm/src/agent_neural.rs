// agent_neural.rs - Per-agent neural network management

use wasm_bindgen::prelude::*;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction as RuvActivation, TrainingData};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::neural::{parse_activation_function, NetworkConfig, TrainingDataConfig};

#[wasm_bindgen]
pub struct AgentNeuralNetworkManager {
    agent_networks: Arc<Mutex<HashMap<String, AgentNeuralContext>>>,
    network_templates: HashMap<String, NetworkTemplate>,
    memory_pool: NeuralMemoryPool,
}

#[derive(Clone)]
pub struct AgentNeuralContext {
    pub agent_id: String,
    pub network: Network<f32>,
    pub cognitive_pattern: CognitivePattern,
    pub training_state: TrainingState,
    pub performance_metrics: NeuralPerformanceMetrics,
    pub adaptation_history: Vec<AdaptationEvent>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CognitivePattern {
    pub pattern_type: String, // convergent, divergent, lateral, systems, critical, abstract
    pub processing_preferences: ProcessingPreferences,
    pub learning_style: LearningStyle,
    pub specialization_weights: HashMap<String, f32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessingPreferences {
    pub depth_vs_breadth: f32, // 0.0 (depth) to 1.0 (breadth)
    pub exploration_vs_exploitation: f32, // 0.0 to 1.0
    pub sequential_vs_parallel: f32, // 0.0 to 1.0
    pub analytical_vs_intuitive: f32, // 0.0 to 1.0
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LearningStyle {
    pub learning_rate_preference: f32,
    pub momentum_preference: f32,
    pub regularization_strength: f32,
    pub adaptation_speed: f32,
    pub memory_retention: f32,
}

#[derive(Clone)]
pub struct TrainingState {
    pub epochs_trained: u32,
    pub current_loss: f32,
    pub best_loss: f32,
    pub is_training: bool,
    pub last_update: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeuralPerformanceMetrics {
    pub inference_time_ms: f32,
    pub memory_usage_mb: f32,
    pub accuracy: f32,
    pub efficiency_score: f32,
    pub adaptation_success_rate: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NetworkTemplate {
    pub pattern_type: String,
    pub layer_configs: Vec<LayerConfig>,
    pub output_activation: String,
    pub learning_config: LearningConfig,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerConfig {
    pub size: usize,
    pub activation: String,
    pub dropout: Option<f32>,
    pub steepness: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LearningConfig {
    pub initial_learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub gradient_clipping: Option<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct AgentNetworkConfig {
    pub agent_id: String,
    pub agent_type: String,
    pub cognitive_pattern: String,
    pub input_size: usize,
    pub output_size: usize,
    pub task_specialization: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
pub struct AdaptationEvent {
    pub timestamp: f64,
    pub event_type: String,
    pub metrics_before: NeuralPerformanceMetrics,
    pub metrics_after: NeuralPerformanceMetrics,
    pub success: bool,
}

#[derive(Serialize, Deserialize)]
pub struct ExperienceData {
    pub inputs: Vec<Vec<f32>>,
    pub expected_outputs: Vec<Vec<f32>>,
    pub actual_outputs: Vec<Vec<f32>>,
    pub rewards: Vec<f32>,
    pub context: serde_json::Value,
}

pub struct NeuralMemoryPool {
    total_capacity: usize,
    allocated_memory: usize,
}

impl NeuralMemoryPool {
    pub fn new(capacity_bytes: usize) -> Self {
        NeuralMemoryPool {
            total_capacity: capacity_bytes,
            allocated_memory: 0,
        }
    }
}

#[wasm_bindgen]
impl AgentNeuralNetworkManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AgentNeuralNetworkManager {
        let mut manager = AgentNeuralNetworkManager {
            agent_networks: Arc::new(Mutex::new(HashMap::new())),
            network_templates: HashMap::new(),
            memory_pool: NeuralMemoryPool::new(100 * 1024 * 1024), // 100MB pool
        };
        
        manager.initialize_cognitive_templates();
        manager
    }
    
    #[wasm_bindgen]
    pub fn create_agent_network(&mut self, agent_config: JsValue) -> Result<String, JsValue> {
        let config: AgentNetworkConfig = serde_wasm_bindgen::from_value(agent_config)
            .map_err(|e| JsValue::from_str(&format!("Invalid agent config: {}", e)))?;
        
        // Select appropriate network template based on cognitive pattern
        let template = self.get_network_template(&config.cognitive_pattern)?;
        
        // Build customized network for agent
        let network = self.build_agent_network(&template, &config)?;
        
        // Create neural context
        let neural_context = AgentNeuralContext {
            agent_id: config.agent_id.clone(),
            network,
            cognitive_pattern: self.create_cognitive_pattern(&config.cognitive_pattern),
            training_state: TrainingState {
                epochs_trained: 0,
                current_loss: f32::INFINITY,
                best_loss: f32::INFINITY,
                is_training: false,
                last_update: js_sys::Date::now(),
            },
            performance_metrics: NeuralPerformanceMetrics {
                inference_time_ms: 0.0,
                memory_usage_mb: 0.0,
                accuracy: 0.0,
                efficiency_score: 0.0,
                adaptation_success_rate: 0.0,
            },
            adaptation_history: Vec::new(),
        };
        
        // Store in agent networks
        self.agent_networks.lock().unwrap()
            .insert(config.agent_id.clone(), neural_context);
        
        Ok(config.agent_id)
    }
    
    #[wasm_bindgen]
    pub fn train_agent_network(&mut self, agent_id: &str, training_data: JsValue) -> Result<JsValue, JsValue> {
        let data: TrainingDataConfig = serde_wasm_bindgen::from_value(training_data)
            .map_err(|e| JsValue::from_str(&format!("Invalid training data: {}", e)))?;
        
        let mut networks = self.agent_networks.lock().unwrap();
        let context = networks.get_mut(agent_id)
            .ok_or_else(|| JsValue::from_str(&format!("Agent network not found: {}", agent_id)))?;
        
        // Set training state
        context.training_state.is_training = true;
        
        // Perform adaptive training based on cognitive pattern
        let training_result = self.adaptive_train(&mut context.network, &data, &context.cognitive_pattern)?;
        
        // Update training state
        context.training_state.epochs_trained += training_result.epochs;
        context.training_state.current_loss = training_result.final_loss;
        if training_result.final_loss < context.training_state.best_loss {
            context.training_state.best_loss = training_result.final_loss;
        }
        context.training_state.is_training = false;
        context.training_state.last_update = js_sys::Date::now();
        
        // Record adaptation event
        context.adaptation_history.push(AdaptationEvent {
            timestamp: js_sys::Date::now(),
            event_type: "training".to_string(),
            metrics_before: context.performance_metrics.clone(),
            metrics_after: self.measure_performance(&context.network),
            success: training_result.converged,
        });
        
        Ok(serde_wasm_bindgen::to_value(&training_result).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn fine_tune_during_execution(&mut self, agent_id: &str, experience_data: JsValue) -> Result<JsValue, JsValue> {
        let experience: ExperienceData = serde_wasm_bindgen::from_value(experience_data)
            .map_err(|e| JsValue::from_str(&format!("Invalid experience data: {}", e)))?;
        
        let mut networks = self.agent_networks.lock().unwrap();
        let context = networks.get_mut(agent_id)
            .ok_or_else(|| JsValue::from_str(&format!("Agent network not found: {}", agent_id)))?;
        
        // Perform online learning based on experience
        let adaptation_result = self.online_adaptation(&mut context.network, &experience, &context.cognitive_pattern)?;
        
        // Update performance metrics
        context.performance_metrics = self.measure_performance(&context.network);
        
        Ok(serde_wasm_bindgen::to_value(&adaptation_result).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_agent_inference(&self, agent_id: &str, input: &[f32]) -> Result<Vec<f32>, JsValue> {
        let networks = self.agent_networks.lock().unwrap();
        let context = networks.get(agent_id)
            .ok_or_else(|| JsValue::from_str(&format!("Agent network not found: {}", agent_id)))?;
        
        let start_time = js_sys::Date::now();
        let output = context.network.run(input)
            .map_err(|e| JsValue::from_str(&format!("Inference error: {}", e)))?;
        let inference_time = js_sys::Date::now() - start_time;
        
        // Log performance
        web_sys::console::log_1(&JsValue::from_str(&format!(
            "Agent {} inference completed in {:.2}ms", agent_id, inference_time
        )));
        
        Ok(output)
    }
    
    #[wasm_bindgen]
    pub fn get_agent_cognitive_state(&self, agent_id: &str) -> Result<JsValue, JsValue> {
        let networks = self.agent_networks.lock().unwrap();
        let context = networks.get(agent_id)
            .ok_or_else(|| JsValue::from_str(&format!("Agent network not found: {}", agent_id)))?;
        
        let cognitive_state = serde_json::json!({
            "agent_id": agent_id,
            "cognitive_pattern": context.cognitive_pattern,
            "neural_architecture": {
                "layers": context.network.num_layers(),
                "neurons": context.network.total_neurons(),
                "connections": context.network.total_connections(),
            },
            "training_progress": {
                "epochs_trained": context.training_state.epochs_trained,
                "current_loss": context.training_state.current_loss,
                "best_loss": context.training_state.best_loss,
                "is_training": context.training_state.is_training,
            },
            "performance": context.performance_metrics,
            "adaptation_history_length": context.adaptation_history.len(),
        });
        
        Ok(serde_wasm_bindgen::to_value(&cognitive_state).unwrap())
    }
    
    // Helper methods
    fn initialize_cognitive_templates(&mut self) {
        // Convergent thinking template
        self.network_templates.insert("convergent".to_string(), NetworkTemplate {
            pattern_type: "convergent".to_string(),
            layer_configs: vec![
                LayerConfig { size: 128, activation: "relu".to_string(), dropout: Some(0.1), steepness: Some(1.0) },
                LayerConfig { size: 64, activation: "relu".to_string(), dropout: Some(0.1), steepness: Some(1.0) },
                LayerConfig { size: 32, activation: "relu".to_string(), dropout: None, steepness: Some(1.0) },
            ],
            output_activation: "sigmoid".to_string(),
            learning_config: LearningConfig {
                initial_learning_rate: 0.001,
                momentum: 0.9,
                weight_decay: 0.0001,
                gradient_clipping: Some(1.0),
            },
        });
        
        // Divergent thinking template
        self.network_templates.insert("divergent".to_string(), NetworkTemplate {
            pattern_type: "divergent".to_string(),
            layer_configs: vec![
                LayerConfig { size: 256, activation: "sigmoid".to_string(), dropout: Some(0.2), steepness: Some(1.0) },
                LayerConfig { size: 128, activation: "tanh".to_string(), dropout: Some(0.2), steepness: Some(1.0) },
                LayerConfig { size: 64, activation: "sigmoid".to_string(), dropout: Some(0.1), steepness: Some(1.0) },
                LayerConfig { size: 32, activation: "relu".to_string(), dropout: None, steepness: Some(1.0) },
            ],
            output_activation: "sigmoid_symmetric".to_string(),
            learning_config: LearningConfig {
                initial_learning_rate: 0.01,
                momentum: 0.7,
                weight_decay: 0.00001,
                gradient_clipping: None,
            },
        });
        
        // Add other cognitive patterns
        self.add_lateral_template();
        self.add_systems_template();
        self.add_critical_template();
        self.add_abstract_template();
    }
    
    fn add_lateral_template(&mut self) {
        self.network_templates.insert("lateral".to_string(), NetworkTemplate {
            pattern_type: "lateral".to_string(),
            layer_configs: vec![
                LayerConfig { size: 200, activation: "elliot".to_string(), dropout: Some(0.15), steepness: Some(1.0) },
                LayerConfig { size: 100, activation: "elliot_symmetric".to_string(), dropout: Some(0.15), steepness: Some(1.0) },
                LayerConfig { size: 50, activation: "sigmoid".to_string(), dropout: None, steepness: Some(1.0) },
            ],
            output_activation: "sigmoid".to_string(),
            learning_config: LearningConfig {
                initial_learning_rate: 0.005,
                momentum: 0.8,
                weight_decay: 0.00005,
                gradient_clipping: Some(2.0),
            },
        });
    }
    
    fn add_systems_template(&mut self) {
        self.network_templates.insert("systems".to_string(), NetworkTemplate {
            pattern_type: "systems".to_string(),
            layer_configs: vec![
                LayerConfig { size: 300, activation: "relu".to_string(), dropout: Some(0.2), steepness: Some(1.0) },
                LayerConfig { size: 150, activation: "relu".to_string(), dropout: Some(0.15), steepness: Some(1.0) },
                LayerConfig { size: 75, activation: "tanh".to_string(), dropout: Some(0.1), steepness: Some(1.0) },
                LayerConfig { size: 40, activation: "sigmoid".to_string(), dropout: None, steepness: Some(1.0) },
            ],
            output_activation: "linear".to_string(),
            learning_config: LearningConfig {
                initial_learning_rate: 0.0001,
                momentum: 0.95,
                weight_decay: 0.001,
                gradient_clipping: Some(0.5),
            },
        });
    }
    
    fn add_critical_template(&mut self) {
        self.network_templates.insert("critical".to_string(), NetworkTemplate {
            pattern_type: "critical".to_string(),
            layer_configs: vec![
                LayerConfig { size: 150, activation: "sigmoid".to_string(), dropout: Some(0.1), steepness: Some(2.0) },
                LayerConfig { size: 75, activation: "sigmoid".to_string(), dropout: Some(0.1), steepness: Some(2.0) },
                LayerConfig { size: 40, activation: "relu".to_string(), dropout: None, steepness: Some(1.0) },
            ],
            output_activation: "threshold".to_string(),
            learning_config: LearningConfig {
                initial_learning_rate: 0.0005,
                momentum: 0.85,
                weight_decay: 0.0005,
                gradient_clipping: Some(1.5),
            },
        });
    }
    
    fn add_abstract_template(&mut self) {
        self.network_templates.insert("abstract".to_string(), NetworkTemplate {
            pattern_type: "abstract".to_string(),
            layer_configs: vec![
                LayerConfig { size: 400, activation: "gaussian".to_string(), dropout: Some(0.25), steepness: Some(0.5) },
                LayerConfig { size: 200, activation: "gaussian_symmetric".to_string(), dropout: Some(0.2), steepness: Some(0.5) },
                LayerConfig { size: 100, activation: "tanh".to_string(), dropout: Some(0.15), steepness: Some(1.0) },
                LayerConfig { size: 50, activation: "sigmoid".to_string(), dropout: None, steepness: Some(1.0) },
            ],
            output_activation: "sigmoid_symmetric".to_string(),
            learning_config: LearningConfig {
                initial_learning_rate: 0.01,
                momentum: 0.6,
                weight_decay: 0.00001,
                gradient_clipping: None,
            },
        });
    }
    
    fn get_network_template(&self, pattern_type: &str) -> Result<&NetworkTemplate, JsValue> {
        self.network_templates.get(pattern_type)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown cognitive pattern: {}", pattern_type)))
    }
    
    fn build_agent_network(&self, template: &NetworkTemplate, config: &AgentNetworkConfig) -> Result<Network<f32>, JsValue> {
        let mut builder = NetworkBuilder::<f32>::new()
            .input_layer(config.input_size);
            
        // Add hidden layers from template
        for layer in &template.layer_configs {
            builder = builder.hidden_layer_with_activation(
                layer.size,
                parse_activation_function(&layer.activation)?,
                layer.steepness.unwrap_or(1.0)
            );
        }
        
        // Add output layer
        let network = builder
            .output_layer_with_activation(
                config.output_size,
                parse_activation_function(&template.output_activation)?,
                1.0
            )
            .build();
            
        Ok(network)
    }
    
    fn create_cognitive_pattern(&self, pattern_type: &str) -> CognitivePattern {
        let (depth_vs_breadth, exploration_vs_exploitation, sequential_vs_parallel, analytical_vs_intuitive) = match pattern_type {
            "convergent" => (0.8, 0.2, 0.7, 0.8),
            "divergent" => (0.2, 0.8, 0.3, 0.2),
            "lateral" => (0.5, 0.6, 0.5, 0.4),
            "systems" => (0.7, 0.4, 0.8, 0.6),
            "critical" => (0.9, 0.1, 0.6, 0.9),
            "abstract" => (0.3, 0.7, 0.4, 0.1),
            _ => (0.5, 0.5, 0.5, 0.5),
        };
        
        CognitivePattern {
            pattern_type: pattern_type.to_string(),
            processing_preferences: ProcessingPreferences {
                depth_vs_breadth,
                exploration_vs_exploitation,
                sequential_vs_parallel,
                analytical_vs_intuitive,
            },
            learning_style: LearningStyle {
                learning_rate_preference: match pattern_type {
                    "convergent" => 0.001,
                    "divergent" => 0.01,
                    "lateral" => 0.005,
                    "systems" => 0.0001,
                    "critical" => 0.0005,
                    "abstract" => 0.01,
                    _ => 0.001,
                },
                momentum_preference: 0.9,
                regularization_strength: 0.0001,
                adaptation_speed: 0.1,
                memory_retention: 0.95,
            },
            specialization_weights: HashMap::new(),
        }
    }
    
    fn adaptive_train(&self, network: &mut Network<f32>, data: &TrainingDataConfig, cognitive_pattern: &CognitivePattern) -> Result<TrainingResult, JsValue> {
        // Simplified adaptive training based on cognitive pattern
        use ruv_fann::training::{IncrementalBackprop, TrainingAlgorithm, TrainingData};
        
        let training_data = TrainingData {
            inputs: data.inputs.clone(),
            outputs: data.outputs.clone(),
        };
        
        let mut trainer = IncrementalBackprop::new(cognitive_pattern.learning_style.learning_rate_preference);
        
        let mut error = f32::INFINITY;
        let mut epochs = 0;
        let max_epochs = 100;
        let target_error = 0.001;
        
        while epochs < max_epochs && error > target_error {
            error = trainer.train_epoch(network, &training_data)
                .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?;
            epochs += 1;
        }
        
        Ok(TrainingResult {
            converged: error <= target_error,
            final_loss: error,
            epochs,
        })
    }
    
    fn online_adaptation(&self, network: &mut Network<f32>, experience: &ExperienceData, cognitive_pattern: &CognitivePattern) -> Result<AdaptationResult, JsValue> {
        // Simplified online adaptation
        use ruv_fann::training::{IncrementalBackprop, TrainingAlgorithm, TrainingData};
        
        let training_data = TrainingData {
            inputs: experience.inputs.clone(),
            outputs: experience.expected_outputs.clone(),
        };
        
        let mut trainer = IncrementalBackprop::new(
            cognitive_pattern.learning_style.learning_rate_preference * cognitive_pattern.learning_style.adaptation_speed
        );
        
        let error = trainer.train_epoch(network, &training_data)
            .map_err(|e| JsValue::from_str(&format!("Adaptation error: {}", e)))?;
        
        Ok(AdaptationResult {
            adapted: true,
            final_error: error,
            improvement: 0.0, // Would calculate actual improvement
        })
    }
    
    fn measure_performance(&self, network: &Network<f32>) -> NeuralPerformanceMetrics {
        NeuralPerformanceMetrics {
            inference_time_ms: 0.0, // Would measure actual time
            memory_usage_mb: (network.total_connections() * std::mem::size_of::<f32>()) as f32 / (1024.0 * 1024.0),
            accuracy: 0.0, // Would calculate based on test data
            efficiency_score: 0.0, // Would calculate based on metrics
            adaptation_success_rate: 0.0, // Would track over time
        }
    }
}

// Result types
#[derive(Serialize, Deserialize)]
struct TrainingResult {
    converged: bool,
    final_loss: f32,
    epochs: u32,
}

#[derive(Serialize, Deserialize)]
struct AdaptationResult {
    adapted: bool,
    final_error: f32,
    improvement: f32,
}