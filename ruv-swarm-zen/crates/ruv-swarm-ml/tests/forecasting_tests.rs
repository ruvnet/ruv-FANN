//! Integration tests for neural forecasting functionality

use ruv_swarm_ml::{
    agent_forecasting::{AgentForecastingManager, ForecastDomain, ForecastRequirements},
    ensemble::{EnsembleConfig, EnsembleForecaster, EnsembleStrategy, OptimizationMetric},
    models::{ModelFactory, ModelType, ModelCategory},
    time_series::{TimeSeriesData, TimeSeriesProcessor, TransformationType},
};

#[test]
fn test_agent_model_assignment() {
    let mut manager = AgentForecastingManager::new(100.0);

    // Test assigning models to different agent types
    let agents = vec![
        ("agent_1", "researcher"),
        ("agent_2", "coder"),
        ("agent_3", "analyst"),
        ("agent_4", "optimizer"),
        ("agent_5", "coordinator"),
    ];

    for (agent_id, agent_type) in agents {
        let requirements = ForecastRequirements {
            horizon: 24,
            frequency: "H".to_string(),
            accuracy_target: 0.9,
            latency_requirement_ms: 200.0,
            interpretability_needed: agent_type == "analyst",
            online_learning: true,
        };

        let result =
            manager.assign_model(agent_id.to_string(), agent_type.to_string(), requirements);

        assert!(
            result.is_ok(),
            "Failed to assign model to {}: {:?}",
            agent_id,
            result
        );

        // Verify assignment
        let state = manager.get_agent_state(agent_id);
        assert!(
            state.is_some(),
            "Agent {} not found after assignment",
            agent_id
        );

        let state = state.unwrap();
        assert_eq!(state.agent_id, agent_id);
        assert_eq!(state.agent_type, agent_type);
    }
}

#[test]
fn test_model_factory() {
    // Test getting available models
    let models = ModelFactory::get_available_models();
    assert!(!models.is_empty(), "No models available");

    // Verify we have models from each category
    let has_basic = models
        .iter()
        .any(|m| m.category == ModelCategory::Basic);
    let has_recurrent = models
        .iter()
        .any(|m| m.category == ModelCategory::Recurrent);
    let has_advanced = models
        .iter()
        .any(|m| m.category == ModelCategory::Advanced);
    let has_transformer = models
        .iter()
        .any(|m| m.category == ModelCategory::Transformer);
    let has_specialized = models
        .iter()
        .any(|m| m.category == ModelCategory::Specialized);

    assert!(has_basic, "No basic models found");
    assert!(has_recurrent, "No recurrent models found");
    assert!(has_advanced, "No advanced models found");
    assert!(has_transformer, "No transformer models found");
    assert!(has_specialized, "No specialized models found");

    // Test model requirements
    let lstm_reqs = ModelFactory::get_model_requirements(ModelType::LSTM);
    assert!(lstm_reqs
        .required_params
        .contains(&"hidden_size".to_string()));
    assert!(lstm_reqs.required_params.contains(&"horizon".to_string()));
    assert_eq!(lstm_reqs.min_samples, 100);
}

#[test]
fn test_time_series_processing() {
    let mut processor = TimeSeriesProcessor::new();

    // Create test data
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let timestamps = (0..10).map(|i| i as f64 * 3600.0).collect();

    let data = TimeSeriesData {
        values,
        timestamps,
        frequency: "H".to_string(),
        unique_id: "test".to_string(),
    };

    // Test normalization
    let normalized = processor
        .fit_transform(data.clone(), vec![TransformationType::Normalize])
        .unwrap();

    assert_eq!(normalized.values.len(), data.values.len());
    assert!(
        (normalized.values[0] - 0.0).abs() < 1e-6,
        "First value should be 0"
    );
    assert!(
        (normalized.values[9] - 1.0).abs() < 1e-6,
        "Last value should be 1"
    );

    // Test standardization
    let standardized = processor
        .fit_transform(data.clone(), vec![TransformationType::Standardize])
        .unwrap();

    let mean = standardized.values.iter().sum::<f32>() / standardized.values.len() as f32;
    assert!(mean.abs() < 1e-6, "Standardized mean should be ~0");

    // Test differencing
    let differenced = processor
        .fit_transform(data.clone(), vec![TransformationType::Difference])
        .unwrap();

    assert_eq!(differenced.values.len(), data.values.len() - 1);
    assert!(
        (differenced.values[0] - 1.0).abs() < 1e-6,
        "First difference should be 1"
    );
}

#[test]
fn test_ensemble_forecasting() {
    // Test simple average ensemble
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec![
            "model1".to_string(),
            "model2".to_string(),
            "model3".to_string(),
        ],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 1000,
        quantile_levels: vec![0.1, 0.25, 0.5, 0.75, 0.9],
    };

    let forecaster = EnsembleForecaster::new(config).unwrap();

    // Create test predictions
    let predictions = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1.5, 2.5, 3.5, 4.5, 5.5],
        vec![0.5, 1.5, 2.5, 3.5, 4.5],
    ];

    let result = forecaster.ensemble_predict(&predictions).unwrap();

    assert_eq!(result.point_forecast.len(), 5);
    assert!((result.point_forecast[0] - 1.0).abs() < 1e-6); // (1.0 + 1.5 + 0.5) / 3
    assert!((result.point_forecast[1] - 2.0).abs() < 1e-6); // (2.0 + 2.5 + 1.5) / 3

    // Test weighted average ensemble
    let config_weighted = EnsembleConfig {
        strategy: EnsembleStrategy::WeightedAverage,
        models: vec!["model1".to_string(), "model2".to_string()],
        weights: Some(vec![0.7, 0.3]),
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 1000,
        quantile_levels: vec![0.1, 0.25, 0.5, 0.75, 0.9],
    };

    let forecaster_weighted = EnsembleForecaster::new(config_weighted).unwrap();

    let predictions_weighted = vec![vec![10.0, 20.0], vec![20.0, 30.0]];

    let result_weighted = forecaster_weighted
        .ensemble_predict(&predictions_weighted)
        .unwrap();

    assert!((result_weighted.point_forecast[0] - 13.0).abs() < 1e-6); // 10*0.7 + 20*0.3
    assert!((result_weighted.point_forecast[1] - 23.0).abs() < 1e-6); // 20*0.7 + 30*0.3
}

#[test]
fn test_performance_tracking() {
    let mut manager = AgentForecastingManager::new(100.0);

    // Create an agent
    let requirements = ForecastRequirements::default();
    manager
        .assign_model(
            "test_agent".to_string(),
            "researcher".to_string(),
            requirements,
        )
        .unwrap();

    // Update performance multiple times
    for i in 0..10 {
        let latency = 50.0 + i as f32 * 10.0;
        let accuracy = 0.9 - i as f32 * 0.01;
        let confidence = 0.85 + i as f32 * 0.01;

        manager
            .update_performance("test_agent", latency, accuracy, confidence)
            .unwrap();
    }

    // Check that performance was tracked
    let state = manager.get_agent_state("test_agent").unwrap();
    assert_eq!(state.performance_history.total_forecasts, 10);
    assert!(state.performance_history.average_latency_ms > 0.0);
    assert!(state.performance_history.average_confidence > 0.0);
}

#[test]
fn test_model_specialization() {
    let manager = AgentForecastingManager::new(100.0);

    // Test that different agent types get appropriate forecast domains
    let test_cases = vec![
        ("researcher", ForecastDomain::TaskCompletion),
        ("coder", ForecastDomain::TaskCompletion),
        ("analyst", ForecastDomain::AgentPerformance),
        ("optimizer", ForecastDomain::ResourceUtilization),
        ("coordinator", ForecastDomain::SwarmDynamics),
    ];

    for (agent_type, expected_domain) in test_cases {
        let requirements = ForecastRequirements::default();
        let mut temp_manager = AgentForecastingManager::new(100.0);

        temp_manager
            .assign_model("test".to_string(), agent_type.to_string(), requirements)
            .unwrap();

        let state = temp_manager.get_agent_state("test").unwrap();
        assert_eq!(
            state.model_specialization.forecast_domain, expected_domain,
            "Agent type {} should have domain {:?}",
            agent_type, expected_domain
        );
    }
}

#[test]
fn test_seasonality_detection() {
    let processor = TimeSeriesProcessor::new();

    // Create synthetic seasonal data
    let mut values = Vec::new();
    for i in 0..100 {
        let t = i as f32;
        // Daily pattern (period 24) + weekly pattern (period 168) + noise
        let value = 100.0
            + 10.0 * (t * 2.0 * std::f32::consts::PI / 24.0).sin()
            + 5.0 * (t * 2.0 * std::f32::consts::PI / 168.0).sin()
            + (i % 3) as f32 * 0.1; // Small noise
        values.push(value);
    }

    let timestamps = (0..100).map(|i| i as f64 * 3600.0).collect();

    let data = TimeSeriesData {
        values,
        timestamps,
        frequency: "H".to_string(),
        unique_id: "seasonal_test".to_string(),
    };

    let seasonality_info = processor.detect_seasonality(&data);

    assert!(
        seasonality_info.has_seasonality,
        "Should detect seasonality"
    );
    assert!(
        !seasonality_info.seasonal_periods.is_empty(),
        "Should find seasonal periods"
    );
}

#[test]
fn test_ensemble_prediction_intervals() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec![
            "model1".to_string(),
            "model2".to_string(),
            "model3".to_string(),
        ],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 1000,
        quantile_levels: vec![0.1, 0.25, 0.5, 0.75, 0.9],
    };

    let forecaster = EnsembleForecaster::new(config).unwrap();

    // Create predictions with some variance
    let predictions = vec![
        vec![95.0, 100.0, 105.0],
        vec![100.0, 105.0, 110.0],
        vec![105.0, 110.0, 115.0],
    ];

    let result = forecaster.ensemble_predict(&predictions).unwrap();

    // Check that prediction intervals are properly ordered
    for i in 0..result.point_forecast.len() {
        assert!(
            result.prediction_intervals.level_50.0[i] <= result.point_forecast[i],
            "50% lower bound should be <= point forecast"
        );
        assert!(
            result.prediction_intervals.level_50.1[i] >= result.point_forecast[i],
            "50% upper bound should be >= point forecast"
        );
        assert!(
            result.prediction_intervals.level_80.0[i] <= result.prediction_intervals.level_50.0[i],
            "80% lower bound should be <= 50% lower bound"
        );
        assert!(
            result.prediction_intervals.level_80.1[i] >= result.prediction_intervals.level_50.1[i],
            "80% upper bound should be >= 50% upper bound"
        );
        assert!(
            result.prediction_intervals.level_95.0[i] <= result.prediction_intervals.level_80.0[i],
            "95% lower bound should be <= 80% lower bound"
        );
        assert!(
            result.prediction_intervals.level_95.1[i] >= result.prediction_intervals.level_80.1[i],
            "95% upper bound should be >= 80% upper bound"
        );
    }
}
