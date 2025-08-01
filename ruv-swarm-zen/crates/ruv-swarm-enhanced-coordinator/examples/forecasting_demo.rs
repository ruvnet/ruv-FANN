//! Forecasting demonstration example
//! 
//! This example shows how to use the neural forecasting capabilities
//! with agent-specific models and ensemble methods.

use ruv_swarm_ml::{
    agent_forecasting::{AgentForecastingManager, ForecastRequirements},
    ensemble::{EnsembleForecaster, EnsembleConfig, EnsembleStrategy},
    models::{ModelFactory, ModelType},
    time_series::{TimeSeriesData, TimeSeriesProcessor, TransformationType},
};

fn main() {
    println!("=== RUV Swarm Neural Forecasting Demo ===\n");
    
    // Initialize forecasting manager with 100MB memory limit
    let mut forecast_manager = AgentForecastingManager::new(100.0);
    
    // Demo 1: Agent-specific model assignment
    println!("1. Assigning forecasting models to different agent types:");
    
    let agent_configs = vec![
        ("agent_researcher_1", "researcher", 24, 0.95, 100.0),
        ("agent_coder_1", "coder", 12, 0.90, 200.0),
        ("agent_analyst_1", "analyst", 48, 0.98, 500.0),
        ("agent_coordinator_1", "coordinator", 168, 0.85, 300.0),
    ];
    
    for (agent_id, agent_type, horizon, accuracy, latency) in agent_configs {
        let requirements = ForecastRequirements {
            horizon,
            frequency: "H".to_string(),
            accuracy_target: accuracy,
            latency_requirement_ms: latency,
            interpretability_needed: agent_type == "analyst",
            online_learning: true,
        };
        
        match forecast_manager.assign_model(
            agent_id.to_string(),
            agent_type.to_string(),
            requirements,
        ) {
            Ok(_) => {
                if let Some(state) = forecast_manager.get_agent_state(agent_id) {
                    println!(
                        "  ✓ {} ({}): Assigned {:?} model for {:?}",
                        agent_id,
                        agent_type,
                        state.primary_model,
                        state.model_specialization.forecast_domain
                    );
                }
            }
            Err(e) => println!("  ✗ Failed to assign model to {}: {}", agent_id, e),
        }
    }
    
    println!("\n2. Available forecasting models:");
    let models = ModelFactory::get_available_models();
    
    // Group by category
    println!("  Basic Models:");
    for model in models.iter().filter(|m| m.category == crate::models::ModelCategory::Basic) {
        println!("    - {}: {}", model.model_type, model.name);
    }
    
    println!("  Recurrent Models:");
    for model in models.iter().filter(|m| m.category == crate::models::ModelCategory::Recurrent) {
        println!("    - {}: {}", model.model_type, model.name);
    }
    
    println!("  Advanced Models:");
    for model in models.iter().filter(|m| m.category == crate::models::ModelCategory::Advanced) {
        println!("    - {}: {}", model.model_type, model.name);
    }
    
    println!("  Transformer Models:");
    for model in models.iter().filter(|m| m.category == crate::models::ModelCategory::Transformer) {
        println!("    - {}: {}", model.model_type, model.name);
    }
    
    // Demo 2: Time series processing
    println!("\n3. Time series data processing:");
    
    // Generate synthetic time series
    let mut values = Vec::new();
    let mut timestamps = Vec::new();
    for i in 0..100 {
        let t = i as f32;
        let value = 100.0 + 10.0 * (t * 0.1).sin() + 5.0 * (t * 0.05).cos() + rand::random::<f32>() * 2.0;
        values.push(value);
        timestamps.push(i as f64 * 3600.0); // Hourly data
    }
    
    let time_series = TimeSeriesData {
        values: values.clone(),
        timestamps: timestamps.clone(),
        frequency: "H".to_string(),
        unique_id: "demo_series".to_string(),
    };
    
    println!("  Original data statistics:");
    println!("    - Length: {}", time_series.values.len());
    println!("    - Mean: {:.2}", time_series.values.iter().sum::<f32>() / time_series.values.len() as f32);
    println!("    - Min: {:.2}", time_series.values.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
    println!("    - Max: {:.2}", time_series.values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Apply transformations
    let mut processor = TimeSeriesProcessor::new();
    let transformations = vec![
        TransformationType::Standardize,
        TransformationType::Difference,
    ];
    
    match processor.fit_transform(time_series.clone(), transformations) {
        Ok(processed) => {
            println!("\n  After standardization and differencing:");
            println!("    - Length: {}", processed.values.len());
            println!("    - Mean: {:.6}", processed.values.iter().sum::<f32>() / processed.values.len() as f32);
            println!("    - Std Dev: ~1.0 (standardized)");
        }
        Err(e) => println!("  ✗ Processing failed: {}", e),
    }
    
    // Demo 3: Ensemble forecasting
    println!("\n4. Ensemble forecasting demonstration:");
    
    // Create ensemble with different strategies
    let ensemble_strategies = vec![
        ("Simple Average", EnsembleStrategy::SimpleAverage),
        ("Median", EnsembleStrategy::Median),
        ("Trimmed Mean (20%)", EnsembleStrategy::TrimmedMean(0.2)),
    ];
    
    for (name, strategy) in ensemble_strategies {
        let config = EnsembleConfig {
            strategy,
            models: vec!["LSTM".to_string(), "GRU".to_string(), "TCN".to_string()],
            weights: None,
            meta_learner: None,
            optimization_metric: crate::ensemble::OptimizationMetric::MAE,
        };
        
        match EnsembleForecaster::new(config) {
            Ok(forecaster) => {
                println!("  ✓ Created {} ensemble with 3 models", name);
                
                // Simulate predictions from different models
                let predictions = vec![
                    vec![101.0, 102.5, 103.0, 102.0, 101.5], // LSTM
                    vec![100.5, 103.0, 103.5, 101.5, 100.0], // GRU
                    vec![101.5, 102.0, 102.5, 102.5, 101.0], // TCN
                ];
                
                match forecaster.ensemble_predict(&predictions) {
                    Ok(result) => {
                        println!("    Point forecast: {:?}", result.point_forecast);
                        println!("    Diversity score: {:.3}", result.ensemble_metrics.diversity_score);
                    }
                    Err(e) => println!("    ✗ Ensemble prediction failed: {}", e),
                }
            }
            Err(e) => println!("  ✗ Failed to create {} ensemble: {}", name, e),
        }
    }
    
    // Demo 4: Model requirements
    println!("\n5. Model requirements for common architectures:");
    
    let example_models = vec![
        ModelType::LSTM,
        ModelType::NBEATS,
        ModelType::TFT,
        ModelType::DeepAR,
    ];
    
    for model_type in example_models {
        let requirements = ModelFactory::get_model_requirements(model_type);
        println!("\n  {}:", model_type);
        println!("    Required params: {:?}", requirements.required_params);
        println!("    Min samples: {}", requirements.min_samples);
        if let Some(max_h) = requirements.max_horizon {
            println!("    Max horizon: {}", max_h);
        }
        println!("    Supports missing values: {}", requirements.supports_missing_values);
    }
    
    // Demo 5: Performance simulation
    println!("\n6. Simulating agent performance updates:");
    
    for i in 0..5 {
        let latency = 50.0 + rand::random::<f32>() * 100.0;
        let accuracy = 0.85 + rand::random::<f32>() * 0.1;
        let confidence = 0.8 + rand::random::<f32>() * 0.15;
        
        match forecast_manager.update_performance(
            "agent_researcher_1",
            latency,
            accuracy,
            confidence,
        ) {
            Ok(_) => {
                println!(
                    "  Update {}: latency={:.1}ms, accuracy={:.3}, confidence={:.3}",
                    i + 1, latency, accuracy, confidence
                );
            }
            Err(e) => println!("  ✗ Failed to update performance: {}", e),
        }
    }
    
    // Show final state
    if let Some(state) = forecast_manager.get_agent_state("agent_researcher_1") {
        println!("\n  Final state for agent_researcher_1:");
        println!("    Total forecasts: {}", state.performance_history.total_forecasts);
        println!("    Avg latency: {:.1}ms", state.performance_history.average_latency_ms);
        println!("    Avg confidence: {:.3}", state.performance_history.average_confidence);
    }
    
    println!("\n=== Demo Complete ===");
}

// Simple random number generation for demo
mod rand {
    pub fn random<T>() -> T
    where
        T: From<f32>,
    {
        // Simple pseudo-random for demo purposes
        static mut SEED: u32 = 42;
        unsafe {
            SEED = SEED.wrapping_mul(1664525).wrapping_add(1013904223);
            T::from((SEED as f32 / u32::MAX as f32))
        }
    }
}