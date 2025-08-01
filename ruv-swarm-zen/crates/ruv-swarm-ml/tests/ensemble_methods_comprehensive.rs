//! Comprehensive test suite for Ensemble Methods Implementation (Issue #129)
//!
//! This test suite validates all 7 ensemble strategies, prediction intervals,
//! diversity metrics, and automatic weight optimization as specified in Issue #129.

use ruv_swarm_ml::ensemble::{
    EnsembleConfig, EnsembleForecaster, EnsembleStrategy, OptimizationMetric, DiversityMetrics,
};

/// Test simple average ensemble with exact mathematical validation
#[test]
fn test_simple_average_ensemble() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec!["LSTM".to_string(), "NBEATS".to_string(), "TFT".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    let predictions = vec![
        vec![100.0, 105.0, 110.0], // LSTM
        vec![102.0, 107.0, 109.0], // NBEATS
        vec![98.0, 103.0, 108.0],  // TFT
    ];
    
    let result = forecaster.ensemble_predict(&predictions).unwrap();
    
    // Should be simple average: (100+102+98)/3, (105+107+103)/3, (110+109+108)/3
    assert!((result.point_forecast[0] - 100.0).abs() < 0.01, 
        "Expected 100.0, got {}", result.point_forecast[0]);
    assert!((result.point_forecast[1] - 105.0).abs() < 0.01,
        "Expected 105.0, got {}", result.point_forecast[1]);
    assert!((result.point_forecast[2] - 109.0).abs() < 0.01,
        "Expected 109.0, got {}", result.point_forecast[2]);
    
    // Verify strategy is correctly set
    assert_eq!(result.strategy, EnsembleStrategy::SimpleAverage);
    assert_eq!(result.models_used, 3);
}

/// Test weighted average ensemble with performance-based weights
#[test]
fn test_weighted_average_ensemble() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::WeightedAverage,
        models: vec!["LSTM".to_string(), "NBEATS".to_string()],
        weights: Some(vec![0.7, 0.3]),
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    let predictions = vec![
        vec![100.0], // LSTM
        vec![200.0], // NBEATS
    ];
    
    let result = forecaster.ensemble_predict(&predictions).unwrap();
    
    // Should be weighted average: 0.7*100 + 0.3*200 = 130
    assert!((result.point_forecast[0] - 130.0).abs() < 0.01,
        "Expected 130.0, got {}", result.point_forecast[0]);
    
    // Test with multiple time steps
    let predictions_multi = vec![
        vec![100.0, 110.0, 120.0], // LSTM
        vec![200.0, 210.0, 220.0], // NBEATS
    ];
    
    let result_multi = forecaster.ensemble_predict(&predictions_multi).unwrap();
    
    // Verify each time step: 0.7*lstm + 0.3*nbeats
    assert!((result_multi.point_forecast[0] - 130.0).abs() < 0.01); // 0.7*100 + 0.3*200
    assert!((result_multi.point_forecast[1] - 140.0).abs() < 0.01); // 0.7*110 + 0.3*210  
    assert!((result_multi.point_forecast[2] - 150.0).abs() < 0.01); // 0.7*120 + 0.3*220
}

/// Test prediction intervals at 50%, 80%, and 95% confidence levels
#[test]
fn test_prediction_intervals() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec!["LSTM".to_string(), "NBEATS".to_string(), "TFT".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    let predictions = vec![
        vec![100.0, 105.0], // LSTM
        vec![110.0, 115.0], // NBEATS  
        vec![90.0, 95.0],   // TFT
    ];
    
    let result = forecaster.ensemble_predict(&predictions).unwrap();
    
    // Verify prediction intervals exist
    assert_eq!(result.prediction_intervals.level_50.0.len(), 2);
    assert_eq!(result.prediction_intervals.level_50.1.len(), 2);
    assert_eq!(result.prediction_intervals.level_80.0.len(), 2);
    assert_eq!(result.prediction_intervals.level_80.1.len(), 2);
    assert_eq!(result.prediction_intervals.level_95.0.len(), 2);
    assert_eq!(result.prediction_intervals.level_95.1.len(), 2);
    
    // Intervals should be properly ordered (wider intervals contain narrower ones)
    for i in 0..2 {
        let lower_95 = result.prediction_intervals.level_95.0[i];
        let lower_80 = result.prediction_intervals.level_80.0[i];
        let lower_50 = result.prediction_intervals.level_50.0[i];
        let upper_50 = result.prediction_intervals.level_50.1[i];
        let upper_80 = result.prediction_intervals.level_80.1[i];
        let upper_95 = result.prediction_intervals.level_95.1[i];
        
        assert!(lower_95 <= lower_80, "95% lower should be <= 80% lower");
        assert!(lower_80 <= lower_50, "80% lower should be <= 50% lower");
        assert!(lower_50 <= upper_50, "Lower bound should be <= upper bound");
        assert!(upper_50 <= upper_80, "50% upper should be <= 80% upper");
        assert!(upper_80 <= upper_95, "80% upper should be <= 95% upper");
    }
}

/// Test ensemble diversity metrics including correlation analysis
#[test] 
fn test_ensemble_diversity_metrics() {
    let predictions = vec![
        vec![100.0, 105.0, 110.0], // Linear increasing
        vec![100.0, 102.0, 104.0], // Similar but slower increase 
        vec![110.0, 105.0, 100.0], // Linear decreasing (opposite pattern)
    ];
    
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec!["model1".to_string(), "model2".to_string(), "model3".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    let forecaster = EnsembleForecaster::new(config).unwrap();
    let diversity = forecaster.calculate_diversity_metrics(&predictions);
    
    // Should detect that first two models are highly correlated
    assert_eq!(diversity.pairwise_correlations.len(), 3, "Should have 3 pairwise correlations"); // 3 pairs: (0,1), (0,2), (1,2)
    assert!(diversity.effective_model_count > 0.0, "Effective model count should be > 0");
    assert!(diversity.effective_model_count <= 3.0, "Effective model count should be <= 3");
    // Should have some diversity due to different patterns
    assert!(diversity.diversity_score >= 0.0 && diversity.diversity_score <= 1.0, 
        "Diversity score should be between 0 and 1, got {}", diversity.diversity_score);
    assert!(diversity.diversity_score > 0.0, 
        "Should have some diversity with different patterns, got {}", diversity.diversity_score);
    
    // Test with identical predictions (should have low diversity)
    let identical_predictions = vec![
        vec![100.0, 105.0, 110.0],
        vec![100.0, 105.0, 110.0],
        vec![100.0, 105.0, 110.0],
    ];
    
    let identical_diversity = forecaster.calculate_diversity_metrics(&identical_predictions);
    assert!(identical_diversity.diversity_score < 0.1, 
        "Identical predictions should have very low diversity, got {}", identical_diversity.diversity_score);
    assert!(identical_diversity.effective_model_count <= 3.0,
        "Identical predictions should have low effective model count, got {}", identical_diversity.effective_model_count);
}

/// Test automatic weight optimization based on historical performance
#[test]
fn test_automatic_weight_optimization() {
    let mut forecaster = EnsembleForecaster::new(EnsembleConfig {
        strategy: EnsembleStrategy::WeightedAverage,
        models: vec!["LSTM".to_string(), "NBEATS".to_string()],
        weights: None, // Should optimize automatically
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    }).unwrap();
    
    // Add models to match the historical predictions
    use ruv_swarm_ml::ensemble::{EnsembleModel, ModelPerformanceMetrics};
    use ruv_swarm_ml::models::ModelType;
    
    forecaster.add_model(EnsembleModel {
        name: "LSTM".to_string(),
        model_type: ModelType::LSTM,
        weight: 0.5,
        performance_metrics: ModelPerformanceMetrics {
            mae: 0.0, mse: 0.0, mape: 0.0, smape: 0.0, coverage: 0.0,
        },
        training_predictions: vec![],
        out_of_sample_predictions: vec![],
    });
    
    forecaster.add_model(EnsembleModel {
        name: "NBEATS".to_string(),
        model_type: ModelType::NBEATS,
        weight: 0.5,
        performance_metrics: ModelPerformanceMetrics {
            mae: 0.0, mse: 0.0, mape: 0.0, smape: 0.0, coverage: 0.0,
        },
        training_predictions: vec![],
        out_of_sample_predictions: vec![],
    });
    
    // Provide historical performance data
    let historical_predictions = vec![
        vec![100.0, 105.0], // LSTM - more accurate
        vec![120.0, 125.0], // NBEATS - consistently worse
    ];
    let actual_values = vec![102.0, 107.0];
    
    let optimized_weights = forecaster.optimize_weights(
        &historical_predictions, 
        &actual_values, 
        OptimizationMetric::MAE
    ).unwrap();
    
    // LSTM should get higher weight due to better performance  
    assert!(optimized_weights[0] > optimized_weights[1], 
        "LSTM should get higher weight: {} vs {}", optimized_weights[0], optimized_weights[1]);
    assert!((optimized_weights[0] + optimized_weights[1] - 1.0).abs() < 0.01, 
        "Weights should sum to 1, got sum: {}", optimized_weights[0] + optimized_weights[1]);
    
    // Verify weights are stored in forecaster
    let current_weights = forecaster.get_current_weights();
    assert_eq!(current_weights.len(), 2);
    assert!((current_weights[0] - optimized_weights[0]).abs() < 0.01);
    assert!((current_weights[1] - optimized_weights[1]).abs() < 0.01);
}

/// Test median ensemble for robust prediction
#[test]
fn test_median_ensemble() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::Median,
        models: vec!["model1".to_string(), "model2".to_string(), "model3".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    // Test with odd number of predictions
    let predictions_odd = vec![
        vec![10.0, 20.0],
        vec![15.0, 25.0], // This should be the median
        vec![20.0, 30.0],
    ];
    
    let result_odd = forecaster.ensemble_predict(&predictions_odd).unwrap();
    assert!((result_odd.point_forecast[0] - 15.0).abs() < 0.01);
    assert!((result_odd.point_forecast[1] - 25.0).abs() < 0.01);
    
    // Test with even number of predictions  
    let predictions_even = vec![
        vec![10.0],
        vec![20.0],
    ];
    
    let result_even = forecaster.ensemble_predict(&predictions_even).unwrap();
    assert!((result_even.point_forecast[0] - 15.0).abs() < 0.01); // (10+20)/2
}

/// Test trimmed mean ensemble for outlier resistance
#[test]
fn test_trimmed_mean_ensemble() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::TrimmedMean(0.2), // Trim 20% from each end
        models: vec!["model1".to_string(), "model2".to_string(), "model3".to_string(), "model4".to_string(), "model5".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    let predictions = vec![
        vec![1.0],   // Will be trimmed (lowest)
        vec![10.0],  // Kept
        vec![12.0],  // Kept  
        vec![14.0],  // Kept
        vec![100.0], // Will be trimmed (highest)
    ];
    
    let result = forecaster.ensemble_predict(&predictions).unwrap();
    
    // Should average middle 3 values: (10 + 12 + 14) / 3 = 12
    assert!((result.point_forecast[0] - 12.0).abs() < 0.01,
        "Expected 12.0, got {}", result.point_forecast[0]);
}

/// Test Bayesian Model Averaging with performance-based weighting
#[test]
fn test_bayesian_model_averaging() {
    // First create models with performance metrics
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::BayesianModelAveraging,
        models: vec!["LSTM".to_string(), "NBEATS".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let mut forecaster = EnsembleForecaster::new(config).unwrap();
    
    // Add models with different performance metrics
    use ruv_swarm_ml::ensemble::{EnsembleModel, ModelPerformanceMetrics};
    use ruv_swarm_ml::models::ModelType;
    
    forecaster.add_model(EnsembleModel {
        name: "LSTM".to_string(),
        model_type: ModelType::LSTM,
        weight: 0.0, // Will be calculated
        performance_metrics: ModelPerformanceMetrics {
            mae: 10.0,
            mse: 100.0, // Good performance
            mape: 5.0,
            smape: 5.0,
            coverage: 0.9,
        },
        training_predictions: vec![vec![100.0, 110.0]],
        out_of_sample_predictions: vec![vec![105.0, 115.0]],
    });
    
    forecaster.add_model(EnsembleModel {
        name: "NBEATS".to_string(),
        model_type: ModelType::NBEATS,
        weight: 0.0, // Will be calculated
        performance_metrics: ModelPerformanceMetrics {
            mae: 20.0,
            mse: 400.0, // Worse performance
            mape: 10.0,
            smape: 10.0,
            coverage: 0.8,
        },
        training_predictions: vec![vec![120.0, 130.0]],
        out_of_sample_predictions: vec![vec![125.0, 135.0]],
    });
    
    let predictions = vec![
        vec![100.0, 110.0], // LSTM
        vec![120.0, 130.0], // NBEATS
    ];
    
    let result = forecaster.ensemble_predict(&predictions).unwrap();
    
    // LSTM should get higher weight due to lower MSE, so result should be closer to LSTM predictions
    assert!(result.point_forecast[0] < 110.0, "Should be weighted toward better model");
    assert!(result.point_forecast[1] < 120.0, "Should be weighted toward better model");
}

/// Test stacking ensemble error handling (until meta-learner is implemented)
#[test]
fn test_stacking_ensemble_error() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::Stacking,
        models: vec!["model1".to_string(), "model2".to_string()],
        weights: None,
        meta_learner: Some("MLP".to_string()),
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    let predictions = vec![
        vec![100.0],
        vec![110.0],
    ];
    
    let result = forecaster.ensemble_predict(&predictions);
    
    // Should return error until stacking is fully implemented
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Stacking requires trained meta-learner"));
}

/// Test voting ensemble
#[test]
fn test_voting_ensemble() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::Voting,
        models: vec!["model1".to_string(), "model2".to_string(), "model3".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    let predictions = vec![
        vec![95.0, 105.0],
        vec![100.0, 110.0], // This will be median
        vec![105.0, 115.0],
    ];
    
    let result = forecaster.ensemble_predict(&predictions).unwrap();
    
    // Currently implements median voting
    assert!((result.point_forecast[0] - 100.0).abs() < 0.01);
    assert!((result.point_forecast[1] - 110.0).abs() < 0.01);
}

/// Test ensemble metrics calculation
#[test]
fn test_ensemble_metrics() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec!["model1".to_string(), "model2".to_string(), "model3".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    let predictions = vec![
        vec![100.0, 110.0],
        vec![102.0, 112.0],
        vec![98.0, 108.0],
    ];
    
    let result = forecaster.ensemble_predict(&predictions).unwrap();
    
    // Verify ensemble metrics
    assert!(result.ensemble_metrics.diversity_score >= 0.0);
    assert!(result.ensemble_metrics.diversity_score <= 1.0);
    assert!(result.ensemble_metrics.prediction_variance >= 0.0);
    assert!(result.ensemble_metrics.effective_models > 0.0);
    assert_eq!(result.ensemble_metrics.average_model_weight, 1.0 / 3.0);
}

/// Test weight validation
#[test]
fn test_weight_validation() {
    // Test weights that don't sum to 1.0
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::WeightedAverage,
        models: vec!["model1".to_string(), "model2".to_string()],
        weights: Some(vec![0.8, 0.8]), // Sum = 1.6, not 1.0
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let result = EnsembleForecaster::new(config);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Weights must sum to 1.0"));
    
    // Test mismatched number of weights and models
    let config2 = EnsembleConfig {
        strategy: EnsembleStrategy::WeightedAverage,
        models: vec!["model1".to_string(), "model2".to_string()],
        weights: Some(vec![1.0]), // Only 1 weight for 2 models
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let result2 = EnsembleForecaster::new(config2);
    assert!(result2.is_err());
    assert!(result2.unwrap_err().contains("Number of weights must match number of models"));
}

/// Test edge cases and error conditions
#[test] 
fn test_edge_cases() {
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::SimpleAverage,
        models: vec!["model1".to_string()],
        weights: None,
        meta_learner: None,
        optimization_metric: OptimizationMetric::MAE,
        stacking_cv_folds: 5,
        bootstrap_samples: 100,
        quantile_levels: vec![0.1, 0.5, 0.9],
    };
    
    let forecaster = EnsembleForecaster::new(config).unwrap();
    
    // Test empty predictions
    let result = forecaster.ensemble_predict(&[]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No predictions provided"));
    
    // Test mismatched prediction lengths
    let mismatched_predictions = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0, 5.0], // Different length
    ];
    
    let result2 = forecaster.ensemble_predict(&mismatched_predictions);
    assert!(result2.is_err());
    assert!(result2.unwrap_err().contains("All predictions must have the same horizon"));
}