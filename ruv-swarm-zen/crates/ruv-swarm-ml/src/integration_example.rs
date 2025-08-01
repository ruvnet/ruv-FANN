//! Integration example showing ruv-fann neural networks with ruv-swarm-ml
//!
//! This example demonstrates how to use the integrated ruv-fann neural networks
//! for agent forecasting and ensemble methods.

use alloc::{vec, vec::Vec, string::String};

use crate::{
    models::{ModelFactory, ModelType, TimeSeriesData},
    agent_forecasting::{AgentForecastingManager, ForecastRequirements},
    ensemble::{EnsembleForecaster, EnsembleConfig, EnsembleStrategy, OptimizationMetric},
};

/// Example: Create and train a neural network model for agent forecasting
pub fn example_agent_neural_forecasting() -> Result<(), String> {
    // Create sample time series data
    let training_data = TimeSeriesData {
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        timestamps: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        frequency: "H".to_string(),
        static_features: None,
        dynamic_features: None,
    };
    
    // Create agent forecasting manager
    let mut manager = AgentForecastingManager::new(100.0); // 100MB memory limit
    
    // Assign LSTM model to a researcher agent
    let requirements = ForecastRequirements {
        horizon: 3,
        frequency: "H".to_string(),
        accuracy_target: 0.9,
        latency_requirement_ms: 200.0,
        interpretability_needed: true,
        online_learning: true,
    };
    
    manager.assign_model(
        "researcher_001".to_string(),
        "researcher".to_string(),
        requirements,
    )?;
    
    // Create and train neural network model for the agent
    let mut model = manager.create_agent_model("researcher_001", 5, 3)?;
    manager.train_agent_model("researcher_001", &mut model, &training_data)?;
    
    // Generate predictions
    let predictions = manager.forecast_with_agent_model("researcher_001", &mut model, 3)?;
    
    println!("Agent neural network predictions: {:?}", predictions);
    Ok(())
}

/// Example: Create and train an ensemble of neural networks
pub fn example_neural_ensemble_forecasting() -> Result<(), String> {
    // Create sample training data
    let training_data = TimeSeriesData {
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        timestamps: (1..=12).map(|x| x as f64).collect(),
        frequency: "H".to_string(),
        static_features: None,
        dynamic_features: None,
    };
    
    let validation_data = TimeSeriesData {
        values: vec![13.0, 14.0, 15.0, 16.0],
        timestamps: (13..=16).map(|x| x as f64).collect(),
        frequency: "H".to_string(),
        static_features: None,
        dynamic_features: None,
    };
    
    // Create ensemble configuration
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::WeightedAverage,
        models: vec!["LSTM".to_string(), "MLP".to_string(), "NBEATS".to_string()],
        weights: None, // Will be optimized
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    // Create ensemble with neural network models
    let model_types = vec![ModelType::LSTM, ModelType::MLP, ModelType::NBEATS];
    let (mut ensemble, mut models) = EnsembleForecaster::from_neural_models(
        config,
        model_types,
        6, // input size
        3, // output size (horizon)
    )?;
    
    // Train the ensemble
    ensemble.train_ensemble(&mut models, &training_data, Some(&validation_data))?;
    
    // Generate ensemble predictions
    let ensemble_forecast = ensemble.predict_with_models(&mut models, 3)?;
    
    println!("Ensemble neural network forecast: {:?}", ensemble_forecast.point_forecast);
    println!("Models used: {}", ensemble_forecast.models_used);
    println!("Diversity score: {:.3}", ensemble_forecast.ensemble_metrics.diversity_score);
    
    Ok(())
}

/// Example: Multi-agent forecasting with different neural network architectures
pub fn example_multi_agent_neural_forecasting() -> Result<(), String> {
    let mut manager = AgentForecastingManager::new(200.0); // 200MB memory limit
    
    // Sample data
    let training_data = TimeSeriesData {
        values: (1..=20).map(|x| x as f32 * 0.5).collect(),
        timestamps: (1..=20).map(|x| x as f64).collect(),
        frequency: "H".to_string(),
        static_features: None,
        dynamic_features: None,
    };
    
    // Create different agents with specialized models
    let agents = vec![
        ("researcher_001", "researcher", ModelType::TFT),      // Transformer for interpretability
        ("coder_001", "coder", ModelType::LSTM),              // LSTM for sequential patterns
        ("analyst_001", "analyst", ModelType::TFT),           // TFT for interpretable attention
        ("optimizer_001", "optimizer", ModelType::NBEATS),    // NBEATS for pure neural architecture
    ];
    
    let mut agent_models = Vec::new();
    
    for (agent_id, agent_type, expected_model) in agents {
        // Assign model to agent
        let requirements = ForecastRequirements::default();
        manager.assign_model(agent_id.to_string(), agent_type.to_string(), requirements)?;
        
        // Verify the correct model was assigned
        let state = manager.get_agent_state(agent_id).unwrap();
        assert_eq!(state.primary_model, expected_model);
        
        // Create ensemble for each agent
        let ensemble_models = manager.create_agent_ensemble(agent_id, 8, 4)?;
        agent_models.push((agent_id.to_string(), ensemble_models));
        
        println!("Agent {} ({}) assigned {} with {} ensemble models", 
                 agent_id, agent_type, expected_model, agent_models.last().unwrap().1.len());
    }
    
    // Train models for each agent
    for (agent_id, models) in &mut agent_models {
        for model in models {
            let mut training_model = model;
            manager.train_agent_model(agent_id, &mut training_model, &training_data)?;
        }
    }
    
    // Generate predictions from each agent's ensemble
    for (agent_id, models) in &mut agent_models {
        let predictions: Result<Vec<Vec<f32>>, String> = models
            .iter_mut()
            .map(|model| model.predict(4))
            .collect();
        
        if let Ok(pred_ensemble) = predictions {
            // Simple ensemble average
            let mut ensemble_pred = vec![0.0; 4];
            for pred in &pred_ensemble {
                for (i, &val) in pred.iter().enumerate() {
                    ensemble_pred[i] += val / pred_ensemble.len() as f32;
                }
            }
            
            println!("Agent {} ensemble prediction: {:?}", agent_id, ensemble_pred);
        }
    }
    
    // Display memory usage
    let memory_stats = manager.get_memory_stats();
    println!("Memory usage: {:.1}MB / {:.1}MB ({} agents)", 
             memory_stats.current_usage_mb, 
             memory_stats.total_limit_mb, 
             memory_stats.num_active_agents);
    
    Ok(())
}

/// Demonstrate model switching based on performance
pub fn example_adaptive_model_switching() -> Result<(), String> {
    let mut manager = AgentForecastingManager::new(150.0);
    
    // Create agent with low switching threshold for demonstration
    let requirements = ForecastRequirements {
        horizon: 2,
        accuracy_target: 0.95, // High target to trigger switching
        ..ForecastRequirements::default()
    };
    
    manager.assign_model("adaptive_agent".to_string(), "researcher".to_string(), requirements)?;
    
    // Simulate poor performance to trigger model switching
    for i in 0..8 {
        let accuracy = 0.7; // Below threshold
        let latency = 50.0 + i as f32 * 5.0;
        let confidence = 0.8;
        
        manager.update_performance("adaptive_agent", latency, accuracy, confidence)?;
    }
    
    // Check if model was switched
    let comparison = manager.get_model_performance_comparison("adaptive_agent")?;
    println!("Model switches: {}", comparison.num_model_switches);
    println!("Current model: {}", comparison.current_model);
    println!("Average accuracy: {:.3}", comparison.current_avg_accuracy);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_agent_neural_forecasting_integration() {
        assert!(example_agent_neural_forecasting().is_ok());
    }
    
    #[test]
    fn test_neural_ensemble_integration() {
        // Simplified test that doesn't involve long training
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::SimpleAverage,
            models: vec!["MLP".to_string(), "LSTM".to_string()],
            weights: None,
            meta_learner: None,
            optimization_metric: OptimizationMetric::MAE,
            stacking_cv_folds: 5,
            bootstrap_samples: 100,
            quantile_levels: vec![0.1, 0.5, 0.9],
        };
        
        // Create ensemble with neural network models
        let model_types = vec![ModelType::MLP, ModelType::LSTM];
        let result = EnsembleForecaster::from_neural_models(
            config,
            model_types,
            4, // input size
            2, // output size (horizon)
        );
        
        match result {
            Ok((ensemble, mut models)) => {
                // Test prediction without training (will use default/random weights)
                match ensemble.predict_with_models(&mut models, 2) {
                    Ok(_forecast) => println!("Ensemble integration test passed"),
                    Err(e) => panic!("Ensemble prediction failed: {}", e),
                }
            }
            Err(e) => panic!("Ensemble creation failed: {}", e),
        }
    }
    
    #[test]
    fn test_multi_agent_integration() {
        assert!(example_multi_agent_neural_forecasting().is_ok());
    }
    
    #[test]
    fn test_adaptive_switching_integration() {
        assert!(example_adaptive_model_switching().is_ok());
    }
}