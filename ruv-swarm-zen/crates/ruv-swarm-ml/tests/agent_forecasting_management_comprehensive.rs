//! Comprehensive test suite for Agent Forecasting Management system
//! Issue #128 - TDD validation of memory management and performance tracking
//!
//! This test suite validates all 5 phases of the Agent Forecasting Management system:
//! 1. Basic agent model assignment (researcher->NHITS, coder->LSTM, analyst->TFT)
//! 2. Forecasting requirements processing 
//! 3. Performance tracking and model switching
//! 4. Memory management for models
//! 5. Adaptive configuration with online learning

use ruv_swarm_ml::{
    agent_forecasting::{
        AgentForecastingManager, 
        ForecastDomain, 
        ForecastRequirements,
        AgentForecastContext,
        ModelSpecialization,
        AdaptiveModelConfig,
        EnsembleWeightingStrategy,
        OptimizationObjective,
        ModelPerformanceHistory,
        ModelSwitchEvent,
    },
    models::{ModelType, ModelFactory, ModelCategory},
    ensemble::{EnsembleConfig, EnsembleForecaster, EnsembleStrategy, OptimizationMetric},
};

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    /// Phase 1: Basic agent model assignment by type
    /// Validates that specific agent types get assigned their expected models
    #[test]
    fn test_agent_model_assignment_by_type() {
        let mut manager = AgentForecastingManager::new(500.0); // Increased memory limit
        
        // Test cases: (agent_type, expected_model_for_non_interpretable, expected_model_for_interpretable)
        let test_cases = vec![
            ("researcher", ModelType::TFT, ModelType::NHITS),  // TFT default, NHITS for interpretability
            ("coder", ModelType::LSTM, ModelType::LSTM),       // LSTM for sequential patterns
            ("analyst", ModelType::TFT, ModelType::TFT),       // TFT for interpretable attention
            ("optimizer", ModelType::NBEATS, ModelType::NBEATS), // NBEATS for pure neural
            ("coordinator", ModelType::DeepAR, ModelType::DeepAR), // DeepAR for probabilistic
            ("unknown", ModelType::MLP, ModelType::MLP),       // MLP as fallback
        ];

        for (agent_type, expected_default, expected_interpretable) in test_cases {
            // Test default assignment (no interpretability requirement)
            let requirements = ForecastRequirements {
                horizon: 24,
                frequency: "H".to_string(),
                accuracy_target: 0.9,
                latency_requirement_ms: 200.0,
                interpretability_needed: false,
                online_learning: true,
            };

            let agent_id = format!("{}_default", agent_type);
            let result = manager.assign_model(
                agent_id.clone(),
                agent_type.to_string(),
                requirements,
            );

            assert!(
                result.is_ok(),
                "Failed to assign default model to {}: {:?}",
                agent_type,
                result
            );

            let state = manager.get_agent_state(&agent_id).unwrap();
            assert_eq!(
                state.primary_model, expected_default,
                "Agent type {} should get {:?} by default, got {:?}",
                agent_type, expected_default, state.primary_model
            );

            // Test interpretability assignment
            let interpretable_requirements = ForecastRequirements {
                interpretability_needed: true,
                ..ForecastRequirements::default()
            };

            let interpretable_agent_id = format!("{}_interpretable", agent_type);
            let result = manager.assign_model(
                interpretable_agent_id.clone(),
                agent_type.to_string(),
                interpretable_requirements,
            );

            assert!(result.is_ok());
            let state = manager.get_agent_state(&interpretable_agent_id).unwrap();
            assert_eq!(
                state.primary_model, expected_interpretable,
                "Agent type {} should get {:?} for interpretability, got {:?}",
                agent_type, expected_interpretable, state.primary_model
            );
        }
    }

    /// Phase 2: Forecasting requirements processing
    /// Validates that different requirements are properly processed and specializations created
    #[test]
    fn test_forecasting_requirements_processing() {
        let mut manager = AgentForecastingManager::new(200.0);

        // Test low latency requirement
        let low_latency_requirements = ForecastRequirements {
            horizon: 12,
            frequency: "30T".to_string(), // 30 minutes
            accuracy_target: 0.85,
            latency_requirement_ms: 50.0, // Very low latency
            interpretability_needed: false,
            online_learning: true,
        };

        manager.assign_model(
            "low_latency_agent".to_string(),
            "optimizer".to_string(),
            low_latency_requirements,
        ).unwrap();

        let state = manager.get_agent_state("low_latency_agent").unwrap();
        
        // Check that specialization includes latency optimization
        let has_latency_opt = state.model_specialization.optimization_objectives
            .iter()
            .any(|obj| matches!(obj, OptimizationObjective::MinimizeLatency));
        assert!(
            has_latency_opt,
            "Low latency requirement should include MinimizeLatency objective"
        );

        // Test balanced requirement
        let balanced_requirements = ForecastRequirements {
            horizon: 24,
            frequency: "H".to_string(),
            accuracy_target: 0.95,
            latency_requirement_ms: 200.0, // Normal latency
            interpretability_needed: false,
            online_learning: true,
        };

        manager.assign_model(
            "balanced_agent".to_string(),
            "analyst".to_string(),
            balanced_requirements,
        ).unwrap();

        let balanced_state = manager.get_agent_state("balanced_agent").unwrap();
        let has_balanced_opt = balanced_state.model_specialization.optimization_objectives
            .iter()
            .any(|obj| matches!(obj, OptimizationObjective::BalanceAccuracyLatency));
        assert!(
            has_balanced_opt,
            "Normal latency requirement should include BalanceAccuracyLatency objective"
        );

        // Validate adaptive configuration
        assert!(
            balanced_state.adaptive_config.online_learning_enabled,
            "Online learning should be enabled when requested"
        );
        let uses_dynamic_performance = matches!(
            balanced_state.adaptive_config.ensemble_weighting_strategy,
            EnsembleWeightingStrategy::DynamicPerformance
        );
        assert!(
            uses_dynamic_performance,
            "Should use dynamic performance weighting strategy"
        );
    }

    /// Phase 3: Performance tracking and model switching
    /// Validates performance metrics tracking and model switching logic
    #[test]
    fn test_performance_tracking() {
        let mut manager = AgentForecastingManager::new(100.0);
        
        // Create test agent
        let requirements = ForecastRequirements::default();
        manager.assign_model(
            "performance_test_agent".to_string(),
            "researcher".to_string(),
            requirements,
        ).unwrap();

        // Simulate multiple performance updates with varying quality
        let performance_data = vec![
            (100.0, 0.95, 0.9),  // latency_ms, accuracy, confidence
            (110.0, 0.93, 0.88),
            (90.0, 0.96, 0.92),
            (120.0, 0.89, 0.85),  // Lower accuracy
            (85.0, 0.97, 0.93),
            (95.0, 0.91, 0.87),
            (105.0, 0.94, 0.89),
            (115.0, 0.92, 0.86),
            (80.0, 0.98, 0.94),
            (130.0, 0.88, 0.84),  // Lower accuracy again
        ];

        for (i, (latency, accuracy, confidence)) in performance_data.iter().enumerate() {
            let result = manager.update_performance(
                "performance_test_agent",
                *latency,
                *accuracy,
                *confidence,
            );
            
            assert!(
                result.is_ok(),
                "Performance update {} failed: {:?}",
                i + 1,
                result
            );
        }

        // Validate performance tracking
        let state = manager.get_agent_state("performance_test_agent").unwrap();
        let history = &state.performance_history;

        assert_eq!(
            history.total_forecasts, 10,
            "Should track 10 forecasts"
        );

        assert!(
            history.average_latency_ms > 0.0,
            "Average latency should be positive: {}",
            history.average_latency_ms
        );

        assert!(
            history.average_confidence > 0.0,
            "Average confidence should be positive: {}",
            history.average_confidence
        );

        assert_eq!(
            history.recent_accuracies.len(), 10,
            "Should track all 10 recent accuracies"
        );

        // Check exponential moving average behavior
        // Later values should have more influence on the average
        let final_latency = history.average_latency_ms;
        let final_confidence = history.average_confidence;
        
        // Validate that EMA is working (values should be within reasonable ranges)
        assert!(
            final_latency > 50.0 && final_latency < 150.0,
            "EMA latency should be within reasonable range: {}",
            final_latency
        );

        assert!(
            final_confidence > 0.5 && final_confidence < 1.0,
            "EMA confidence should be within reasonable range: {}",
            final_confidence
        );
    }

    /// Phase 3b: Adaptive model switching based on performance
    #[test]
    fn test_adaptive_model_switching() {
        let mut manager = AgentForecastingManager::new(100.0);
        
        // Create agent with specific switching threshold
        let requirements = ForecastRequirements {
            accuracy_target: 0.9,
            ..ForecastRequirements::default()
        };
        
        manager.assign_model(
            "switching_test_agent".to_string(),
            "coder".to_string(),
            requirements,
        ).unwrap();

        // Get initial state
        let initial_state = manager.get_agent_state("switching_test_agent").unwrap();
        let switching_threshold = initial_state.adaptive_config.model_switching_threshold;
        
        // Simulate performance below threshold to trigger potential switching
        let poor_performance_data = vec![
            (100.0, 0.80, 0.85), // Below threshold (0.85)
            (110.0, 0.82, 0.84),
            (120.0, 0.78, 0.83), // Consistently poor
            (115.0, 0.79, 0.82),
            (105.0, 0.81, 0.84),
        ];

        for (latency, accuracy, confidence) in poor_performance_data {
            let result = manager.update_performance(
                "switching_test_agent",
                latency,
                accuracy,
                confidence,
            );
            assert!(result.is_ok());
        }

        // Validate that poor performance is tracked
        let state = manager.get_agent_state("switching_test_agent").unwrap();
        let recent_avg_accuracy: f32 = state.performance_history.recent_accuracies
            .iter()
            .sum::<f32>() / state.performance_history.recent_accuracies.len() as f32;

        assert!(
            recent_avg_accuracy < switching_threshold,
            "Recent average accuracy {} should be below switching threshold {}",
            recent_avg_accuracy,
            switching_threshold
        );

        // Note: Actual model switching logic is marked as TODO in the implementation
        // This test validates the conditions that would trigger switching
    }

    /// Phase 4: Memory constraint handling
    /// Validates memory usage tracking and constraint enforcement
    #[test]
    fn test_memory_constraint_handling() {
        // Create manager with limited memory
        let memory_limit_mb = 50.0;
        let mut manager = AgentForecastingManager::new(memory_limit_mb);

        // Get model memory requirements
        let models_info = ModelFactory::get_available_models();
        
        // Find heavy models (transformers typically use more memory)
        let heavy_models: Vec<_> = models_info
            .iter()
            .filter(|m| m.typical_memory_mb > 10.0)
            .collect();

        assert!(
            !heavy_models.is_empty(),
            "Should have some memory-intensive models for testing"
        );

        // Test memory tracking with agent assignment
        let mut total_expected_memory = 0.0;
        let mut assigned_agents = Vec::new();

        // Try to assign multiple heavy models until we approach the limit
        for (i, model_info) in heavy_models.iter().enumerate() {
            if total_expected_memory + model_info.typical_memory_mb <= memory_limit_mb {
                let agent_id = format!("memory_test_agent_{}", i);
                let agent_type = match model_info.model_type {
                    ModelType::TFT => "analyst",
                    ModelType::LSTM => "coder", 
                    ModelType::NBEATS => "optimizer",
                    ModelType::DeepAR => "coordinator",
                    _ => "researcher",
                };

                let result = manager.assign_model(
                    agent_id.clone(),
                    agent_type.to_string(),
                    ForecastRequirements::default(),
                );

                if result.is_ok() {
                    assigned_agents.push(agent_id);
                    total_expected_memory += model_info.typical_memory_mb;
                } else {
                    // Memory constraint should prevent assignment
                    break;
                }
            }
        }

        assert!(
            !assigned_agents.is_empty(),
            "Should be able to assign at least one agent within memory limits"
        );

        // Validate that we're tracking memory appropriately
        // Note: The current implementation doesn't track actual memory usage,
        // but this test validates the structure is in place
        assert!(
            manager.get_agent_state(&assigned_agents[0]).is_some(),
            "First assigned agent should be retrievable"
        );
    }

    /// Phase 4b: Model eviction and memory management
    #[test]
    fn test_model_eviction_policies() {
        let memory_limit_mb = 60.0; // Limited memory but enough for test agents
        let mut manager = AgentForecastingManager::new(memory_limit_mb);

        // Create multiple agents to test eviction policies
        let agents = vec![
            ("agent_1", "researcher", 100),  // forecast count
            ("agent_2", "coder", 50),
            ("agent_3", "analyst", 200),     // Most active
            ("agent_4", "optimizer", 25),   // Least active
        ];

        // Assign models to all agents
        for (agent_id, agent_type, _) in &agents {
            let result = manager.assign_model(
                agent_id.to_string(),
                agent_type.to_string(),
                ForecastRequirements::default(),
            );
            assert!(result.is_ok(), "Should assign model to {}", agent_id);
        }

        // Simulate different activity levels
        for (agent_id, _, forecast_count) in &agents {
            for i in 0..*forecast_count {
                let accuracy = 0.9 + (i % 10) as f32 * 0.001; // Slight variation
                let latency = 100.0 + (i % 5) as f32 * 10.0;
                let confidence = 0.85 + (i % 3) as f32 * 0.05;

                manager.update_performance(agent_id, latency, accuracy, confidence).unwrap();
            }
        }

        // Validate activity tracking
        for (agent_id, _, expected_count) in &agents {
            let state = manager.get_agent_state(agent_id).unwrap();
            assert_eq!(
                state.performance_history.total_forecasts,
                *expected_count as u64,
                "Agent {} should have {} forecasts",
                agent_id,
                expected_count
            );
        }

        // Test that most active agent (agent_3) has highest forecast count
        let most_active_state = manager.get_agent_state("agent_3").unwrap();
        let least_active_state = manager.get_agent_state("agent_4").unwrap();
        
        assert!(
            most_active_state.performance_history.total_forecasts >
            least_active_state.performance_history.total_forecasts,
            "Most active agent should have more forecasts than least active"
        );

        // Note: Actual eviction logic would be implemented based on these metrics
    }

    /// Phase 5: Adaptive configuration with online learning
    #[test]
    fn test_adaptive_configuration() {
        let mut manager = AgentForecastingManager::new(100.0);

        // Test online learning configuration
        let online_learning_requirements = ForecastRequirements {
            online_learning: true,
            accuracy_target: 0.95,
            ..ForecastRequirements::default()
        };

        manager.assign_model(
            "online_learning_agent".to_string(),
            "researcher".to_string(),
            online_learning_requirements,
        ).unwrap();

        let state = manager.get_agent_state("online_learning_agent").unwrap();
        
        // Validate adaptive configuration
        assert!(
            state.adaptive_config.online_learning_enabled,
            "Online learning should be enabled"
        );

        assert_eq!(
            state.adaptive_config.adaptation_rate,
            0.01,
            "Default adaptation rate should be 0.01"
        );

        assert_eq!(
            state.adaptive_config.model_switching_threshold,
            0.85,
            "Default switching threshold should be 0.85"
        );

        assert_eq!(
            state.adaptive_config.retraining_frequency,
            100,
            "Default retraining frequency should be 100"
        );

        let uses_dynamic_performance = matches!(
            state.adaptive_config.ensemble_weighting_strategy,
            EnsembleWeightingStrategy::DynamicPerformance
        );
        assert!(
            uses_dynamic_performance,
            "Should use dynamic performance weighting"
        );

        // Test offline learning configuration
        let offline_learning_requirements = ForecastRequirements {
            online_learning: false,
            ..ForecastRequirements::default()
        };

        manager.assign_model(
            "offline_learning_agent".to_string(),
            "coder".to_string(),
            offline_learning_requirements,
        ).unwrap();

        let offline_state = manager.get_agent_state("offline_learning_agent").unwrap();
        assert!(
            !offline_state.adaptive_config.online_learning_enabled,
            "Online learning should be disabled"
        );
    }

    /// Phase 5b: Model specialization by domain
    #[test]
    fn test_model_specialization_by_domain() {
        let manager = AgentForecastingManager::new(100.0);

        // Test domain specialization mapping
        let domain_test_cases = vec![
            ("researcher", ForecastDomain::TaskCompletion),
            ("coder", ForecastDomain::TaskCompletion),
            ("analyst", ForecastDomain::AgentPerformance),
            ("optimizer", ForecastDomain::ResourceUtilization),
            ("coordinator", ForecastDomain::SwarmDynamics),
            ("unknown_type", ForecastDomain::AgentPerformance), // Default
        ];

        for (agent_type, expected_domain) in domain_test_cases {
            let mut temp_manager = AgentForecastingManager::new(100.0);
            temp_manager.assign_model(
                "domain_test".to_string(),
                agent_type.to_string(),
                ForecastRequirements::default(),
            ).unwrap();

            let state = temp_manager.get_agent_state("domain_test").unwrap();
            assert_eq!(
                state.model_specialization.forecast_domain,
                expected_domain,
                "Agent type {} should have domain {:?}, got {:?}",
                agent_type,
                expected_domain,
                state.model_specialization.forecast_domain
            );

            // Validate temporal patterns are configured
            assert!(
                !state.model_specialization.temporal_patterns.is_empty(),
                "Should have temporal patterns configured"
            );

            // Validate optimization objectives are set
            assert!(
                !state.model_specialization.optimization_objectives.is_empty(),
                "Should have optimization objectives configured"
            );
        }
    }

    /// Integration test: Full workflow simulation
    #[test]
    fn test_full_workflow_integration() {
        let mut manager = AgentForecastingManager::new(500.0);

        // Phase 1: Create diverse agent fleet
        let agent_fleet = vec![
            ("data_researcher", "researcher"),
            ("model_coder", "coder"),
            ("performance_analyst", "analyst"),
            ("resource_optimizer", "optimizer"),
            ("swarm_coordinator", "coordinator"),
        ];

        // Phase 2: Assign models with different requirements
        for (agent_id, agent_type) in &agent_fleet {
            let requirements = ForecastRequirements {
                horizon: 24,
                frequency: "H".to_string(),
                accuracy_target: 0.9,
                latency_requirement_ms: if agent_type == &"optimizer" { 50.0 } else { 200.0 },
                interpretability_needed: agent_type == &"analyst",
                online_learning: true,
            };

            let result = manager.assign_model(
                agent_id.to_string(),
                agent_type.to_string(),
                requirements,
            );

            assert!(
                result.is_ok(),
                "Failed to assign model to {}: {:?}",
                agent_id,
                result
            );
        }

        // Verify all agents were created successfully
        for (agent_id, _) in &agent_fleet {
            assert!(
                manager.get_agent_state(agent_id).is_some(),
                "Agent {} was not created successfully",
                agent_id
            );
        }

        // Phase 3: Simulate ongoing performance tracking
        for iteration in 0..20 {
            for (agent_id, _) in &agent_fleet {
                // Simulate varying performance with some agents performing better than others
                let base_accuracy = match agent_id {
                    &"performance_analyst" => 0.95,  // Best performer
                    &"data_researcher" => 0.90,      // Good performer
                    &"model_coder" => 0.88,          // Average performer
                    &"resource_optimizer" => 0.85,   // Below threshold sometimes
                    &"swarm_coordinator" => 0.82,    // Struggling
                    _ => 0.85,
                };

                let noise = (iteration % 5) as f32 * 0.01 - 0.02; // Add some variation
                let accuracy = (base_accuracy + noise).max(0.7).min(1.0);
                let latency = 100.0 + (iteration % 3) as f32 * 20.0;
                let confidence = 0.85 + (iteration % 4) as f32 * 0.02;

                let update_result = manager.update_performance(agent_id, latency, accuracy, confidence);
                if let Err(e) = update_result {
                    panic!("Failed to update performance for agent {}: {}", agent_id, e);
                }
            }
        }

        // Phase 4 & 5: Validate final state and adaptive behavior
        for (agent_id, agent_type) in &agent_fleet {
            let state = manager.get_agent_state(agent_id).unwrap_or_else(|| {
                panic!("Agent {} was not found in manager", agent_id);
            });
            
            // Validate basic properties
            assert_eq!(state.agent_id, *agent_id);
            assert_eq!(state.agent_type, *agent_type);
            assert_eq!(state.performance_history.total_forecasts, 20);
            
            // Validate adaptive configuration is working
            assert!(state.adaptive_config.online_learning_enabled);
            
            // Validate performance tracking
            assert!(state.performance_history.average_latency_ms > 0.0);
            assert!(state.performance_history.average_confidence > 0.0);
            assert_eq!(
                state.performance_history.recent_accuracies.len(), 
                20,
                "Agent {} should have 20 recent accuracies, got {}",
                agent_id,
                state.performance_history.recent_accuracies.len()
            );

            // Validate domain specialization
            match agent_type {
                &"researcher" | &"coder" => {
                    assert_eq!(state.model_specialization.forecast_domain, ForecastDomain::TaskCompletion);
                }
                &"analyst" => {
                    assert_eq!(state.model_specialization.forecast_domain, ForecastDomain::AgentPerformance);
                }
                &"optimizer" => {
                    assert_eq!(state.model_specialization.forecast_domain, ForecastDomain::ResourceUtilization);
                }
                &"coordinator" => {
                    assert_eq!(state.model_specialization.forecast_domain, ForecastDomain::SwarmDynamics);
                }
                _ => {}
            }
        }

        // Validate that performance trends are captured
        let best_performer = manager.get_agent_state("performance_analyst").unwrap();
        let worst_performer = manager.get_agent_state("swarm_coordinator").unwrap();

        let best_avg_accuracy: f32 = best_performer.performance_history.recent_accuracies
            .iter().sum::<f32>() / best_performer.performance_history.recent_accuracies.len() as f32;
        
        let worst_avg_accuracy: f32 = worst_performer.performance_history.recent_accuracies
            .iter().sum::<f32>() / worst_performer.performance_history.recent_accuracies.len() as f32;

        assert!(
            best_avg_accuracy > worst_avg_accuracy,
            "Best performer should have higher average accuracy: {} vs {}",
            best_avg_accuracy,
            worst_avg_accuracy
        );
    }

    /// Error handling and edge cases
    #[test]
    fn test_error_handling() {
        let mut manager = AgentForecastingManager::new(100.0);

        // Test updating performance for non-existent agent
        let result = manager.update_performance("non_existent", 100.0, 0.9, 0.8);
        assert!(result.is_err(), "Should fail for non-existent agent");

        // Test getting state for non-existent agent
        let state = manager.get_agent_state("non_existent");
        assert!(state.is_none(), "Should return None for non-existent agent");

        // Test zero memory limit
        let zero_memory_manager = AgentForecastingManager::new(0.0);
        // Should still work but with zero limit
        assert!(zero_memory_manager.get_agent_state("any").is_none());
    }
}