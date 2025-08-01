//! Simple integration test for ruv-fann neural networks
//!
//! This test verifies that ruv-fann neural networks can be used
//! as forecast models in ruv-swarm-ml.

use ruv_swarm_ml::models::{ModelFactory, ModelType};

#[test]
fn test_ruv_fann_integration() {
    // Create a simple MLP model
    let model_result = ModelFactory::create_model(ModelType::MLP, 3, 2);
    assert!(model_result.is_ok(), "Failed to create MLP model");
    
    let model = model_result.unwrap();
    assert_eq!(model.model_type(), ModelType::MLP);
    assert!(model.complexity_score() > 0.0);
}

#[test]
fn test_multiple_model_types() {
    // Test creating different model types
    let model_types = vec![
        ModelType::MLP,
        ModelType::DLinear,
        ModelType::MLPMultivariate,
    ];
    
    for model_type in model_types {
        let model = ModelFactory::create_model(model_type, 4, 2).unwrap();
        assert_eq!(model.model_type(), model_type);
        assert!(model.complexity_score() > 0.0);
    }
}

#[test]
fn test_model_memory_usage() {
    // Create models of different sizes
    let sizes = vec![(2, 1), (10, 5), (50, 10)];
    
    let mut prev_complexity = 0.0;
    for (input_size, output_size) in sizes {
        let model = ModelFactory::create_model(ModelType::MLP, input_size, output_size).unwrap();
        let complexity = model.complexity_score();
        
        // Larger models should have higher complexity
        assert!(complexity >= prev_complexity, "Complexity should increase with model size");
        prev_complexity = complexity;
    }
}

#[test]
fn test_agent_forecasting_with_neural_model() {
    use ruv_swarm_ml::agent_forecasting::{AgentForecastingManager, ForecastRequirements};
    
    // Create agent forecasting manager
    let mut manager = AgentForecastingManager::new(100.0);
    
    // Assign model to agent
    let requirements = ForecastRequirements {
        horizon: 3,
        frequency: "H".to_string(),
        accuracy_target: 0.9,
        latency_requirement_ms: 200.0,
        interpretability_needed: false,
        online_learning: true,
    };
    
    let result = manager.assign_model(
        "test_agent".to_string(),
        "researcher".to_string(),
        requirements,
    );
    
    assert!(result.is_ok(), "Failed to assign model to agent");
    
    // Check that the agent has a neural network model
    let state = manager.get_agent_state("test_agent");
    assert!(state.is_some());
    
    // For researcher, should get TFT or NHITS model
    let agent_state = state.unwrap();
    assert!(
        agent_state.primary_model == ModelType::TFT || 
        agent_state.primary_model == ModelType::NHITS
    );
}

#[test]
fn test_ensemble_with_neural_models() {
    use ruv_swarm_ml::ensemble::{EnsembleConfig, EnsembleForecaster, EnsembleStrategy, OptimizationMetric};
    
    // Create ensemble configuration
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec!["MLP".to_string(), "DLinear".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    // Create ensemble forecaster
    let forecaster = EnsembleForecaster::new(config);
    assert!(forecaster.is_ok(), "Failed to create ensemble forecaster");
    
    // Test prediction (with dummy data)
    let predictions = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.1, 2.1, 3.1],
    ];
    
    let result = forecaster.unwrap().ensemble_predict(&predictions);
    assert!(result.is_ok(), "Failed to generate ensemble prediction");
    
    let forecast = result.unwrap();
    assert_eq!(forecast.point_forecast.len(), 3);
    assert_eq!(forecast.models_used, 2);
}