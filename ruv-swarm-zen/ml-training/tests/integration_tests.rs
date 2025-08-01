//! Integration tests for the ML training pipeline

use ruv_swarm_ml_training::{
    EventType, HyperparameterOptimizer, LSTMModel, ModelEvaluator, NBEATSModel,
    NeuroDivergentModel, OptimizationMethod, ParameterRange, PerformanceMetrics, PromptData,
    SearchSpace, StackType, StreamDataLoader, StreamEvent, TCNModel, TrainingConfig,
    TrainingPipeline,
};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

fn create_test_events() -> Vec<StreamEvent> {
    let base_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    (0..200)
        .map(|i| {
            StreamEvent {
                timestamp: base_time + (i * 300), // 5 minute intervals
                agent_id: format!("test_agent_{}", i % 3),
                event_type: match i % 4 {
                    0 => EventType::TaskStarted,
                    1 => EventType::TaskCompleted,
                    2 => EventType::PromptGenerated,
                    _ => EventType::ResponseReceived,
                },
                performance_metrics: PerformanceMetrics {
                    latency_ms: 30.0 + 20.0 * ((i as f64 * 0.1).sin()),
                    tokens_per_second: 120.0 - 30.0 * ((i as f64 * 0.2).cos()),
                    memory_usage_mb: 200.0 + 50.0 * ((i as f64 * 0.05).sin()),
                    cpu_usage_percent: 35.0 + 15.0 * ((i as f64 * 0.15).cos()),
                    success_rate: 0.90 + 0.08 * ((i as f64 * 0.3).sin()),
                },
                prompt_data: if i % 3 == 0 {
                    Some(PromptData {
                        prompt_text: format!("Test prompt {}", i),
                        prompt_tokens: (40 + (i % 20)) as usize,
                        response_tokens: (80 + (i % 40)) as usize,
                        quality_score: 0.75 + 0.2 * ((i as f64 * 0.25).sin()),
                    })
                } else {
                    None
                },
            }
        })
        .collect()
}

#[tokio::test]
async fn test_data_loading() {
    let loader = StreamDataLoader::new(500, 10);
    let events = create_test_events();
    let event_count = events.len();

    let dataset = loader.load_from_stream(events.into_iter()).await.unwrap();

    assert!(dataset.sequences.len() > 0);
    assert_eq!(dataset.metadata.sequence_length, 10);
    assert!(dataset.metadata.feature_count > 0);
    assert_eq!(dataset.sequences.len(), dataset.labels.len());

    // Check that sequences have the right length
    for sequence in &dataset.sequences {
        assert_eq!(sequence.timestamps.len(), 10);
        assert_eq!(sequence.features.len(), 10);
    }
}

#[tokio::test]
async fn test_model_training() {
    let loader = StreamDataLoader::new(100, 5);
    let events = create_test_events();
    let dataset = loader.load_from_stream(events.into_iter()).await.unwrap();

    let config = TrainingConfig {
        epochs: 5,
        batch_size: 16,
        validation_split: 0.2,
        early_stopping_patience: 3,
        save_checkpoints: false,
        checkpoint_dir: "/tmp/test".to_string(),
    };

    // Test LSTM training
    let mut lstm = LSTMModel::new(32, 1);
    let metrics = lstm.train(&dataset, &config).unwrap();
    assert!(metrics.best_loss < f64::INFINITY);
    assert!(metrics.epoch_losses.len() > 0);

    // Test prediction
    let prediction = lstm.predict(&dataset.sequences[0]).unwrap();
    assert_eq!(prediction.len(), 3); // Should predict 3 values
}

#[tokio::test]
async fn test_hyperparameter_optimization() {
    let loader = StreamDataLoader::new(100, 5);
    let events = create_test_events();
    let dataset = loader.load_from_stream(events.into_iter()).await.unwrap();

    let mut parameters = HashMap::new();
    parameters.insert(
        "hidden_size".to_string(),
        ParameterRange::Discrete {
            values: vec![16.0, 32.0],
        },
    );
    parameters.insert(
        "learning_rate".to_string(),
        ParameterRange::Continuous {
            min: 0.001,
            max: 0.01,
        },
    );

    let search_space = SearchSpace { parameters };
    let optimizer = HyperparameterOptimizer::new(
        search_space,
        OptimizationMethod::RandomSearch,
        3, // Just 3 trials for testing
    );

    let config = TrainingConfig {
        epochs: 2,
        batch_size: 16,
        validation_split: 0.2,
        early_stopping_patience: 3,
        save_checkpoints: false,
        checkpoint_dir: "/tmp/test".to_string(),
    };

    let result = optimizer
        .optimize(|| Box::new(LSTMModel::new(32, 1)), &dataset, &config)
        .await
        .unwrap();

    assert!(result.best_score > 0.0);
    assert_eq!(result.trial_results.len(), 3);
    assert!(result.best_parameters.contains_key("learning_rate"));
}

#[tokio::test]
async fn test_model_evaluation() {
    let loader = StreamDataLoader::new(100, 5);
    let events = create_test_events();
    let dataset = loader.load_from_stream(events.into_iter()).await.unwrap();

    let config = TrainingConfig {
        epochs: 3,
        batch_size: 16,
        validation_split: 0.2,
        early_stopping_patience: 3,
        save_checkpoints: false,
        checkpoint_dir: "/tmp/test".to_string(),
    };

    // Train models
    let mut models: Vec<Box<dyn NeuroDivergentModel>> = vec![
        Box::new(LSTMModel::new(32, 1)),
        Box::new(TCNModel::new(vec![16, 16], 3)),
    ];

    for model in &mut models {
        model.train(&dataset, &config).unwrap();
    }

    // Evaluate
    let evaluator = ModelEvaluator::new();
    let result = evaluator.evaluate_and_select(models, &dataset).unwrap();

    assert!(!result.best_model.is_empty());
    assert_eq!(result.all_scores.len(), 2);

    for score in &result.all_scores {
        assert!(score.scores.contains_key("MSE"));
        assert!(score.scores.contains_key("MAE"));
        assert!(score.scores.contains_key("R2"));
    }
}

#[tokio::test]
async fn test_full_pipeline() {
    let config = TrainingConfig {
        epochs: 2,
        batch_size: 16,
        validation_split: 0.2,
        early_stopping_patience: 3,
        save_checkpoints: false,
        checkpoint_dir: "/tmp/test_pipeline".to_string(),
    };

    let mut pipeline = TrainingPipeline::new(config);
    let events = create_test_events();

    let result = pipeline.run(events.into_iter()).await.unwrap();

    assert!(!result.best_model.is_empty());
    assert!(result.model_scores.len() >= 3); // Should have at least 3 models
    assert!(result.dataset_metadata.total_samples > 0);
}

#[test]
fn test_hyperparameter_getters_setters() {
    let mut lstm = LSTMModel::new(128, 2);

    // Test getters
    let params = lstm.get_hyperparameters();
    assert_eq!(params.get("hidden_size"), Some(&128.0));
    assert_eq!(params.get("num_layers"), Some(&2.0));

    // Test setters
    let mut new_params = HashMap::new();
    new_params.insert("hidden_size".to_string(), 256.0);
    new_params.insert("learning_rate".to_string(), 0.005);

    lstm.set_hyperparameters(new_params);

    let updated = lstm.get_hyperparameters();
    assert_eq!(updated.get("hidden_size"), Some(&256.0));
    assert_eq!(updated.get("learning_rate"), Some(&0.005));
}

#[test]
fn test_model_names() {
    let lstm = LSTMModel::new(128, 2);
    assert_eq!(lstm.name(), "LSTM");

    let tcn = TCNModel::new(vec![64, 64], 3);
    assert_eq!(tcn.name(), "TCN");

    let nbeats = NBEATSModel::new(vec![StackType::Trend], 4);
    assert_eq!(nbeats.name(), "N-BEATS");
}
