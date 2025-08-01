//! Example of hyperparameter optimization for ML models

use ruv_swarm_ml_training::{
    EventType, HyperparameterOptimizer, LSTMModel, NBEATSModel, NeuroDivergentModel,
    OptimizationMethod, ParameterRange, PerformanceMetrics, PromptData, SearchSpace, StackType,
    StreamDataLoader, StreamEvent, TCNModel, TrainingConfig,
};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

fn generate_complex_events() -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let base_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Generate more complex patterns for hyperparameter tuning
    for day in 0..30 {
        for hour in 0..24 {
            for minute in (0..60).step_by(5) {
                let timestamp = base_time + (day * 86400) + (hour * 3600) + (minute * 60);

                // Complex performance patterns
                let day_factor = (day as f64 * 0.1).sin();
                let hour_factor = ((hour as f64 * std::f64::consts::PI / 12.0).sin() + 1.0) / 2.0;
                let weekly_pattern = ((day % 7) as f64 / 3.5 - 1.0).cos();

                // Different patterns for different agents
                for agent_id in 0..3 {
                    let agent_factor = match agent_id {
                        0 => 1.0, // Standard agent
                        1 => 1.2, // Slower agent
                        2 => 0.8, // Faster agent
                        _ => 1.0,
                    };

                    events.push(StreamEvent {
                        timestamp,
                        agent_id: format!("agent_{}", agent_id),
                        event_type: match (hour + minute) % 4 {
                            0 => EventType::TaskStarted,
                            1 => EventType::TaskCompleted,
                            2 => EventType::PromptGenerated,
                            _ => EventType::ResponseReceived,
                        },
                        performance_metrics: PerformanceMetrics {
                            latency_ms: 40.0 * agent_factor
                                + 20.0 * hour_factor
                                + 10.0 * day_factor,
                            tokens_per_second: 120.0 / agent_factor - 30.0 * hour_factor,
                            memory_usage_mb: 200.0 + 100.0 * hour_factor + 50.0 * weekly_pattern,
                            cpu_usage_percent: 30.0 + 40.0 * hour_factor + 10.0 * day_factor,
                            success_rate: 0.98 - 0.1 * hour_factor * day_factor.abs(),
                        },
                        prompt_data: if minute % 10 == 0 {
                            Some(PromptData {
                                prompt_text: format!("Complex prompt for optimization"),
                                prompt_tokens: 50 + (hour * 2),
                                response_tokens: 100 + (hour * 3),
                                quality_score: 0.9 - 0.1 * hour_factor,
                            })
                        } else {
                            None
                        },
                    });
                }
            }
        }
    }

    events
}

async fn optimize_lstm() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== LSTM Hyperparameter Optimization ===");

    // Define search space for LSTM
    let mut parameters = HashMap::new();
    parameters.insert(
        "hidden_size".to_string(),
        ParameterRange::Discrete {
            values: vec![64.0, 128.0, 256.0, 512.0],
        },
    );
    parameters.insert(
        "num_layers".to_string(),
        ParameterRange::Discrete {
            values: vec![1.0, 2.0, 3.0, 4.0],
        },
    );
    parameters.insert(
        "dropout".to_string(),
        ParameterRange::Continuous { min: 0.1, max: 0.5 },
    );
    parameters.insert(
        "learning_rate".to_string(),
        ParameterRange::Continuous {
            min: 0.0001,
            max: 0.01,
        },
    );

    let search_space = SearchSpace { parameters };

    // Create optimizer
    let optimizer = HyperparameterOptimizer::new(
        search_space,
        OptimizationMethod::RandomSearch,
        10, // Number of trials
    );

    // Load data
    let data_loader = StreamDataLoader::new(1000, 50);
    let events = generate_complex_events();
    let dataset = data_loader.load_from_stream(events.into_iter()).await?;

    // Training config
    let config = TrainingConfig {
        epochs: 20,
        batch_size: 64,
        validation_split: 0.2,
        early_stopping_patience: 3,
        save_checkpoints: false,
        checkpoint_dir: "./checkpoints".to_string(),
    };

    // Run optimization
    let result = optimizer
        .optimize(|| Box::new(LSTMModel::new(128, 2)), &dataset, &config)
        .await?;

    println!("\nOptimization Complete!");
    println!("Best parameters:");
    for (param, value) in &result.best_parameters {
        println!("  {}: {:.4}", param, value);
    }
    println!("Best validation score: {:.4}", result.best_score);

    Ok(())
}

async fn optimize_tcn() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== TCN Hyperparameter Optimization ===");

    // Define search space for TCN
    let mut parameters = HashMap::new();
    parameters.insert(
        "kernel_size".to_string(),
        ParameterRange::Discrete {
            values: vec![2.0, 3.0, 4.0, 5.0],
        },
    );
    parameters.insert(
        "dropout".to_string(),
        ParameterRange::Continuous { min: 0.1, max: 0.4 },
    );
    parameters.insert(
        "learning_rate".to_string(),
        ParameterRange::Continuous {
            min: 0.0001,
            max: 0.01,
        },
    );

    let search_space = SearchSpace { parameters };

    // Create optimizer with Bayesian optimization
    let optimizer =
        HyperparameterOptimizer::new(search_space, OptimizationMethod::BayesianOptimization, 15);

    // Load data
    let data_loader = StreamDataLoader::new(1000, 50);
    let events = generate_complex_events();
    let dataset = data_loader.load_from_stream(events.into_iter()).await?;

    // Training config
    let config = TrainingConfig {
        epochs: 25,
        batch_size: 32,
        validation_split: 0.2,
        early_stopping_patience: 5,
        save_checkpoints: false,
        checkpoint_dir: "./checkpoints".to_string(),
    };

    // Run optimization
    let result = optimizer
        .optimize(
            || Box::new(TCNModel::new(vec![64, 64, 64], 3)),
            &dataset,
            &config,
        )
        .await?;

    println!("\nOptimization Complete!");
    println!("Best parameters:");
    for (param, value) in &result.best_parameters {
        println!("  {}: {:.4}", param, value);
    }
    println!("Best validation score: {:.4}", result.best_score);

    // Show trial history
    println!("\nTrial History:");
    for (i, trial) in result.trial_results.iter().enumerate().take(5) {
        println!("  Trial {}: score = {:.4}", i + 1, trial.score);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RUV Swarm ML Training - Hyperparameter Optimization");
    println!("==================================================");

    // Optimize LSTM model
    optimize_lstm().await?;

    // Optimize TCN model
    optimize_tcn().await?;

    println!("\n\nAll optimizations completed successfully!");

    Ok(())
}
