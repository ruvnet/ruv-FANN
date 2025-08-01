//! Agent-specific forecasting model management
//!
//! This module provides per-agent forecasting model management with adaptive
//! configuration and specialized models based on agent type and workload.

use alloc::{
    boxed::Box,
    collections::BTreeMap as HashMap,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::fmt;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::models::{ForecastModel, ModelType, ModelFactory, TimeSeriesData};

/// Manages forecasting models for individual agents
pub struct AgentForecastingManager {
    agent_models: HashMap<String, AgentForecastContext>,
    resource_limit_mb: f32,
    current_memory_usage_mb: f32,
}

/// Forecasting context for a specific agent
#[derive(Clone)]
pub struct AgentForecastContext {
    pub agent_id: String,
    pub agent_type: String,
    pub primary_model: ModelType,
    pub ensemble_models: Vec<ModelType>,
    pub model_specialization: ModelSpecialization,
    pub adaptive_config: AdaptiveModelConfig,
    pub performance_history: ModelPerformanceHistory,
}

/// Model specialization based on forecast domain
#[derive(Clone, Debug)]
pub struct ModelSpecialization {
    pub forecast_domain: ForecastDomain,
    pub temporal_patterns: Vec<TemporalPattern>,
    pub optimization_objectives: Vec<OptimizationObjective>,
}

/// Forecast domain types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ForecastDomain {
    ResourceUtilization,
    TaskCompletion,
    AgentPerformance,
    SwarmDynamics,
    AnomalyDetection,
    CapacityPlanning,
}

/// Temporal pattern in time series
#[derive(Clone, Debug)]
pub struct TemporalPattern {
    pub pattern_type: String,
    pub frequency: f32,
    pub strength: f32,
    pub confidence: f32,
}

/// Optimization objective for model training
#[derive(Clone, Debug)]
pub enum OptimizationObjective {
    MinimizeLatency,
    MaximizeAccuracy,
    BalanceAccuracyLatency,
    MinimizeMemory,
}

/// Adaptive model configuration
#[derive(Clone, Debug)]
pub struct AdaptiveModelConfig {
    pub online_learning_enabled: bool,
    pub adaptation_rate: f32,
    pub model_switching_threshold: f32,
    pub ensemble_weighting_strategy: EnsembleWeightingStrategy,
    pub retraining_frequency: u32,
}

/// Strategy for weighting ensemble models
#[derive(Clone, Debug)]
pub enum EnsembleWeightingStrategy {
    Static,
    DynamicPerformance,
    Bayesian,
    StackedGeneralization,
}

/// Performance history for model tracking
#[derive(Clone, Debug)]
pub struct ModelPerformanceHistory {
    pub total_forecasts: u64,
    pub average_confidence: f32,
    pub average_latency_ms: f32,
    pub recent_accuracies: Vec<f32>,
    pub model_switches: Vec<ModelSwitchEvent>,
}

/// Model switch event record
#[derive(Clone, Debug)]
pub struct ModelSwitchEvent {
    pub timestamp: f64,
    pub from_model: String,
    pub to_model: String,
    pub reason: String,
}

impl AgentForecastingManager {
    /// Create a new agent forecasting manager
    pub fn new(resource_limit_mb: f32) -> Self {
        Self {
            agent_models: HashMap::new(),
            resource_limit_mb,
            current_memory_usage_mb: 0.0,
        }
    }

    /// Assign a forecasting model to an agent
    pub fn assign_model(
        &mut self,
        agent_id: String,
        agent_type: String,
        requirements: ForecastRequirements,
    ) -> Result<String, String> {
        // Select optimal model based on agent type
        let primary_model = self.select_optimal_model(&agent_type, &requirements)?;

        // Create model specialization
        let model_specialization = self.create_specialization(&agent_type, &requirements);

        // Create adaptive configuration
        let adaptive_config = AdaptiveModelConfig {
            online_learning_enabled: requirements.online_learning,
            adaptation_rate: 0.01,
            model_switching_threshold: 0.85,
            ensemble_weighting_strategy: EnsembleWeightingStrategy::DynamicPerformance,
            retraining_frequency: 100,
        };

        // Initialize performance history
        let performance_history = ModelPerformanceHistory {
            total_forecasts: 0,
            average_confidence: 0.0,
            average_latency_ms: 0.0,
            recent_accuracies: Vec::new(),
            model_switches: Vec::new(),
        };

        // Check memory constraints
        let estimated_memory = self.estimate_model_memory_usage(primary_model);
        if self.current_memory_usage_mb + estimated_memory > self.resource_limit_mb {
            return Err(format!(
                "Memory limit exceeded: {} + {} > {} MB",
                self.current_memory_usage_mb, estimated_memory, self.resource_limit_mb
            ));
        }

        // Create forecast context
        let context = AgentForecastContext {
            agent_id: agent_id.clone(),
            agent_type,
            primary_model,
            ensemble_models: Vec::new(),
            model_specialization,
            adaptive_config,
            performance_history,
        };

        // Store context and update memory usage
        self.agent_models.insert(agent_id.clone(), context);
        self.current_memory_usage_mb += estimated_memory;

        Ok(agent_id)
    }

    /// Get agent's current forecast state
    pub fn get_agent_state(&self, agent_id: &str) -> Option<&AgentForecastContext> {
        self.agent_models.get(agent_id)
    }

    /// Update agent model performance
    pub fn update_performance(
        &mut self,
        agent_id: &str,
        latency_ms: f32,
        accuracy: f32,
        confidence: f32,
    ) -> Result<(), String> {
        let context = self
            .agent_models
            .get_mut(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;

        // Update performance metrics
        let history = &mut context.performance_history;
        history.total_forecasts += 1;

        // Update moving averages
        let alpha = 0.1; // Exponential moving average factor
        history.average_latency_ms =
            (1.0 - alpha) * history.average_latency_ms + alpha * latency_ms;
        history.average_confidence =
            (1.0 - alpha) * history.average_confidence + alpha * confidence;

        // Track recent accuracies
        history.recent_accuracies.push(accuracy);
        if history.recent_accuracies.len() > 100 {
            history.recent_accuracies.remove(0);
        }

        // Check if model switch is needed
        if accuracy < context.adaptive_config.model_switching_threshold {
            self.evaluate_model_switches(agent_id, accuracy)?;
        }

        Ok(())
    }

    /// Select optimal model based on agent type and requirements
    fn select_optimal_model(
        &self,
        agent_type: &str,
        requirements: &ForecastRequirements,
    ) -> Result<ModelType, String> {
        let model = match agent_type {
            "researcher" => {
                if requirements.interpretability_needed {
                    ModelType::NHITS // Good for exploratory analysis
                } else {
                    ModelType::TFT // Temporal Fusion Transformer
                }
            }
            "coder" => ModelType::LSTM,       // Sequential task patterns
            "analyst" => ModelType::TFT,      // Interpretable attention mechanism
            "optimizer" => ModelType::NBEATS, // Pure neural architecture
            "coordinator" => ModelType::DeepAR, // Probabilistic forecasts
            _ => ModelType::MLP,              // Generic baseline
        };

        Ok(model)
    }

    /// Create model specialization based on agent type
    fn create_specialization(
        &self,
        agent_type: &str,
        requirements: &ForecastRequirements,
    ) -> ModelSpecialization {
        let forecast_domain = match agent_type {
            "researcher" => ForecastDomain::TaskCompletion,
            "coder" => ForecastDomain::TaskCompletion,
            "analyst" => ForecastDomain::AgentPerformance,
            "optimizer" => ForecastDomain::ResourceUtilization,
            "coordinator" => ForecastDomain::SwarmDynamics,
            _ => ForecastDomain::AgentPerformance,
        };

        let temporal_patterns = vec![
            TemporalPattern {
                pattern_type: "daily".to_string(),
                frequency: 24.0,
                strength: 0.8,
                confidence: 0.9,
            },
            TemporalPattern {
                pattern_type: "weekly".to_string(),
                frequency: 168.0,
                strength: 0.6,
                confidence: 0.85,
            },
        ];

        let optimization_objectives = if requirements.latency_requirement_ms < 100.0 {
            vec![OptimizationObjective::MinimizeLatency]
        } else {
            vec![OptimizationObjective::BalanceAccuracyLatency]
        };

        ModelSpecialization {
            forecast_domain,
            temporal_patterns,
            optimization_objectives,
        }
    }

    /// Evaluate if model switching is needed and perform the switch
    pub fn evaluate_model_switches(
        &mut self,
        agent_id: &str,
        current_accuracy: f32,
    ) -> Result<(), String> {
        // First, gather the information we need without borrowing self mutably
        let (current_model, performance_threshold, agent_type, should_switch, recent_avg) = {
            let context = self
                .agent_models
                .get(agent_id)
                .ok_or_else(|| format!("Agent {} not found", agent_id))?;

            let current_model = context.primary_model;
            let performance_threshold = context.adaptive_config.model_switching_threshold;
            let agent_type = context.agent_type.clone();

            // Only switch if performance is consistently below threshold
            let should_switch = if context.performance_history.recent_accuracies.len() >= 5 {
                let recent_avg = context.performance_history.recent_accuracies
                    .iter()
                    .rev()
                    .take(5)
                    .sum::<f32>() / 5.0;
                (recent_avg < performance_threshold, recent_avg)
            } else {
                (false, 0.0)
            };

            (current_model, performance_threshold, agent_type, should_switch.0, should_switch.1)
        };

        // Now perform the switch if needed
        if should_switch {
            // Find a better model for this agent type
            let new_model = self.find_better_model(&agent_type, current_model)?;
            
            if new_model != current_model {
                // Get current timestamp before borrowing mutably
                let timestamp = self.get_current_timestamp();
                
                // Now we can borrow mutably to make the changes
                let context = self
                    .agent_models
                    .get_mut(agent_id)
                    .ok_or_else(|| format!("Agent {} not found", agent_id))?;

                // Record the switch event
                let switch_event = ModelSwitchEvent {
                    timestamp,
                    from_model: current_model.to_string(),
                    to_model: new_model.to_string(),
                    reason: format!("Performance below threshold: {:.3}", recent_avg),
                };
                
                context.performance_history.model_switches.push(switch_event);
                context.primary_model = new_model;
                
                // Note: We don't immediately clear performance history to allow
                // for accurate evaluation of switching conditions
                // In a production system, you might want to keep some history
                // context.performance_history.recent_accuracies.clear();
            }
        }

        Ok(())
    }

    /// Find a better model for the given agent type
    fn find_better_model(
        &self,
        agent_type: &str,
        current_model: ModelType,
    ) -> Result<ModelType, String> {
        // Get available models for this agent type
        let candidate_models = self.get_candidate_models(agent_type);
        
        // For now, use a simple fallback strategy
        // In a real implementation, this would use historical performance data
        for model in candidate_models {
            if model != current_model {
                return Ok(model);
            }
        }
        
        // If no better model found, return current model
        Ok(current_model)
    }

    /// Get candidate models for an agent type
    fn get_candidate_models(&self, agent_type: &str) -> Vec<ModelType> {
        match agent_type {
            "researcher" => vec![ModelType::NHITS, ModelType::TFT, ModelType::MLP],
            "coder" => vec![ModelType::LSTM, ModelType::GRU, ModelType::TCN],
            "analyst" => vec![ModelType::TFT, ModelType::DeepAR, ModelType::NHITS],
            "optimizer" => vec![ModelType::NBEATS, ModelType::NBEATSx, ModelType::TiDE],
            "coordinator" => vec![ModelType::DeepAR, ModelType::TFT, ModelType::Informer],
            _ => vec![ModelType::MLP, ModelType::DLinear, ModelType::NLinear],
        }
    }

    /// Estimate memory usage for a model type
    fn estimate_model_memory_usage(&self, model_type: ModelType) -> f32 {
        use crate::models::ModelFactory;
        
        if let Some(info) = ModelFactory::get_model_info(model_type) {
            info.typical_memory_mb
        } else {
            5.0 // Default estimate
        }
    }

    /// Get current timestamp (mock implementation)
    fn get_current_timestamp(&self) -> f64 {
        // In a real implementation, this would use system time
        std::f64::consts::PI * 1000.0 // Mock timestamp
    }

    /// Remove agent and free memory
    pub fn remove_agent(&mut self, agent_id: &str) -> Result<(), String> {
        if let Some(context) = self.agent_models.remove(agent_id) {
            let memory_freed = self.estimate_model_memory_usage(context.primary_model);
            self.current_memory_usage_mb -= memory_freed;
            Ok(())
        } else {
            Err(format!("Agent {} not found", agent_id))
        }
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_limit_mb: self.resource_limit_mb,
            current_usage_mb: self.current_memory_usage_mb,
            available_mb: self.resource_limit_mb - self.current_memory_usage_mb,
            num_active_agents: self.agent_models.len(),
        }
    }

    /// Clean up inactive agents to free memory
    pub fn cleanup_inactive_agents(&mut self, inactive_threshold_forecasts: u64) -> usize {
        let mut removed_count = 0;
        let agent_ids: Vec<String> = self.agent_models.keys().cloned().collect();
        
        for agent_id in agent_ids {
            if let Some(context) = self.agent_models.get(&agent_id) {
                if context.performance_history.total_forecasts < inactive_threshold_forecasts {
                    if self.remove_agent(&agent_id).is_ok() {
                        removed_count += 1;
                    }
                }
            }
        }
        
        removed_count
    }

    /// Get model performance comparison
    pub fn get_model_performance_comparison(
        &self,
        agent_id: &str,
    ) -> Result<ModelPerformanceComparison, String> {
        let context = self
            .agent_models
            .get(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;

        let current_avg_accuracy = if context.performance_history.recent_accuracies.is_empty() {
            0.0
        } else {
            context.performance_history.recent_accuracies.iter().sum::<f32>()
                / context.performance_history.recent_accuracies.len() as f32
        };

        Ok(ModelPerformanceComparison {
            agent_id: agent_id.to_string(),
            current_model: context.primary_model,
            current_avg_accuracy,
            current_avg_latency: context.performance_history.average_latency_ms,
            current_avg_confidence: context.performance_history.average_confidence,
            total_forecasts: context.performance_history.total_forecasts,
            num_model_switches: context.performance_history.model_switches.len(),
        })
    }
    
    /// Create a neural network model for an agent
    pub fn create_agent_model(&self, agent_id: &str, input_size: usize, output_size: usize) -> Result<Box<dyn ForecastModel>, String> {
        let context = self
            .agent_models
            .get(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;
        
        ModelFactory::create_model(context.primary_model, input_size, output_size)
    }
    
    /// Train an agent's neural network model
    pub fn train_agent_model(
        &mut self,
        agent_id: &str,
        model: &mut Box<dyn ForecastModel>,
        training_data: &TimeSeriesData,
    ) -> Result<(), String> {
        let context = self
            .agent_models
            .get_mut(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;
        
        // Train the model
        model.fit(training_data)?;
        
        // Update training metrics
        context.performance_history.total_forecasts += 1;
        
        Ok(())
    }
    
    /// Generate forecasts using an agent's neural network model
    pub fn forecast_with_agent_model(
        &self,
        agent_id: &str,
        model: &mut Box<dyn ForecastModel>,
        horizon: usize,
    ) -> Result<Vec<f32>, String> {
        let _context = self
            .agent_models
            .get(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;
        
        model.predict(horizon)
    }
    
    /// Create ensemble of models for an agent
    pub fn create_agent_ensemble(
        &self,
        agent_id: &str,
        input_size: usize,
        output_size: usize,
    ) -> Result<Vec<Box<dyn ForecastModel>>, String> {
        let context = self
            .agent_models
            .get(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;
        
        let mut models = Vec::new();
        
        // Create primary model
        let primary_model = ModelFactory::create_model(context.primary_model, input_size, output_size)?;
        models.push(primary_model);
        
        // Create ensemble models if specified
        for &model_type in &context.ensemble_models {
            let model = ModelFactory::create_model(model_type, input_size, output_size)?;
            models.push(model);
        }
        
        // If no ensemble models, create default ensemble based on agent type
        if context.ensemble_models.is_empty() {
            let candidate_models = self.get_candidate_models(&context.agent_type);
            for model_type in candidate_models.into_iter().take(3) { // Limit to 3 additional models
                if model_type != context.primary_model {
                    let model = ModelFactory::create_model(model_type, input_size, output_size)?;
                    models.push(model);
                }
            }
        }
        
        Ok(models)
    }
    
    /// Update agent ensemble models based on performance
    pub fn update_agent_ensemble(&mut self, agent_id: &str, model_types: Vec<ModelType>) -> Result<(), String> {
        let context = self
            .agent_models
            .get_mut(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;
        
        context.ensemble_models = model_types;
        Ok(())
    }
}

/// Memory usage statistics
#[derive(Clone, Debug)]
pub struct MemoryStats {
    pub total_limit_mb: f32,
    pub current_usage_mb: f32,
    pub available_mb: f32,
    pub num_active_agents: usize,
}

/// Model performance comparison data
#[derive(Clone, Debug)]
pub struct ModelPerformanceComparison {
    pub agent_id: String,
    pub current_model: ModelType,
    pub current_avg_accuracy: f32,
    pub current_avg_latency: f32,
    pub current_avg_confidence: f32,
    pub total_forecasts: u64,
    pub num_model_switches: usize,
}

/// Forecast requirements for model selection
#[derive(Clone)]
pub struct ForecastRequirements {
    pub horizon: usize,
    pub frequency: String,
    pub accuracy_target: f32,
    pub latency_requirement_ms: f32,
    pub interpretability_needed: bool,
    pub online_learning: bool,
}

impl Default for ForecastRequirements {
    fn default() -> Self {
        Self {
            horizon: 24,
            frequency: "H".to_string(), // Hourly
            accuracy_target: 0.9,
            latency_requirement_ms: 200.0,
            interpretability_needed: false,
            online_learning: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_model_assignment() {
        let mut manager = AgentForecastingManager::new(100.0);
        let requirements = ForecastRequirements::default();

        let result = manager.assign_model(
            "agent_1".to_string(),
            "researcher".to_string(),
            requirements,
        );

        assert!(result.is_ok());
        assert!(manager.get_agent_state("agent_1").is_some());
    }

    #[test]
    fn test_memory_constraint_handling() {
        let mut manager = AgentForecastingManager::new(30.0); // Memory limit that allows NHITS + TFT
        let requirements = ForecastRequirements::default();

        // First assignment should succeed (researcher uses NHITS ~8MB)
        let result1 = manager.assign_model(
            "agent_1".to_string(),
            "researcher".to_string(),
            requirements.clone(),
        );
        assert!(result1.is_ok(), "First assignment should succeed");

        // Second assignment should succeed (coordinator uses TFT ~20MB, total = 28MB)
        let result2 = manager.assign_model(
            "agent_2".to_string(),
            "coordinator".to_string(), // TFT model uses more memory
            requirements.clone(),
        );
        assert!(result2.is_ok(), "Second assignment should succeed");

        // Third assignment should fail due to memory constraint (another TFT model would exceed limit)
        let result3 = manager.assign_model(
            "agent_3".to_string(),
            "coordinator".to_string(), // Another TFT model would exceed limit
            requirements,
        );
        assert!(result3.is_err(), "Third assignment should fail");
        assert!(result3.unwrap_err().contains("Memory limit exceeded"));
    }

    #[test]
    fn test_model_switching_logic() {
        let mut manager = AgentForecastingManager::new(100.0);
        let requirements = ForecastRequirements::default();

        // Create an agent
        manager
            .assign_model(
                "test_agent".to_string(),
                "researcher".to_string(),
                requirements,
            )
            .unwrap();

        // Update performance with consistently low accuracy to trigger switch
        for _ in 0..6 {
            manager
                .update_performance("test_agent", 100.0, 0.7, 0.8) // Below threshold
                .unwrap();
        }

        // Check that model switch occurred
        let state = manager.get_agent_state("test_agent").unwrap();
        assert!(!state.performance_history.model_switches.is_empty());
    }

    #[test]
    fn test_memory_cleanup() {
        let mut manager = AgentForecastingManager::new(100.0);
        let requirements = ForecastRequirements::default();

        // Create multiple agents
        for i in 0..5 {
            manager
                .assign_model(
                    format!("agent_{}", i),
                    "researcher".to_string(),
                    requirements.clone(),
                )
                .unwrap();
        }

        // Simulate some activity for only some agents
        for i in 0..2 {
            for _ in 0..10 {
                manager
                    .update_performance(&format!("agent_{}", i), 50.0, 0.9, 0.85)
                    .unwrap();
            }
        }

        // Clean up inactive agents (those with < 5 forecasts)
        let removed = manager.cleanup_inactive_agents(5);
        assert_eq!(removed, 3); // Should remove 3 inactive agents
        assert_eq!(manager.agent_models.len(), 2); // 2 active agents remain
    }

    #[test]
    fn test_memory_statistics() {
        let mut manager = AgentForecastingManager::new(100.0);
        let requirements = ForecastRequirements::default();

        // Initial state
        let initial_stats = manager.get_memory_stats();
        assert_eq!(initial_stats.total_limit_mb, 100.0);
        assert_eq!(initial_stats.current_usage_mb, 0.0);
        assert_eq!(initial_stats.num_active_agents, 0);

        // Add an agent
        manager
            .assign_model(
                "agent_1".to_string(),
                "researcher".to_string(),
                requirements,
            )
            .unwrap();

        let stats = manager.get_memory_stats();
        assert!(stats.current_usage_mb > 0.0);
        assert_eq!(stats.num_active_agents, 1);
        assert_eq!(stats.available_mb, stats.total_limit_mb - stats.current_usage_mb);
    }

    #[test]
    fn test_agent_removal() {
        let mut manager = AgentForecastingManager::new(100.0);
        let requirements = ForecastRequirements::default();

        // Add an agent
        manager
            .assign_model(
                "agent_1".to_string(),
                "researcher".to_string(),
                requirements,
            )
            .unwrap();

        assert!(manager.get_agent_state("agent_1").is_some());

        // Remove the agent
        let result = manager.remove_agent("agent_1");
        assert!(result.is_ok());
        assert!(manager.get_agent_state("agent_1").is_none());

        // Check memory was freed
        let stats = manager.get_memory_stats();
        assert_eq!(stats.current_usage_mb, 0.0);
    }

    #[test]
    fn test_performance_comparison() {
        let mut manager = AgentForecastingManager::new(100.0);
        let requirements = ForecastRequirements::default();

        // Create an agent
        manager
            .assign_model(
                "test_agent".to_string(),
                "researcher".to_string(),
                requirements,
            )
            .unwrap();

        // Update performance
        for i in 0..5 {
            manager
                .update_performance(
                    "test_agent",
                    50.0 + i as f32,
                    0.9 - i as f32 * 0.01,
                    0.85 + i as f32 * 0.01,
                )
                .unwrap();
        }

        let comparison = manager.get_model_performance_comparison("test_agent").unwrap();
        assert_eq!(comparison.agent_id, "test_agent");
        assert_eq!(comparison.total_forecasts, 5);
        assert!(comparison.current_avg_accuracy > 0.0);
        assert!(comparison.current_avg_latency > 0.0);
        assert!(comparison.current_avg_confidence > 0.0);
    }
}
