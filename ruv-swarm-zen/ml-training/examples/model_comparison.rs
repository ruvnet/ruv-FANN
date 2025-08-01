//! Example comparing different neuro-divergent models

use ruv_swarm_ml_training::{
    EventType, LSTMModel, ModelEvaluator, NBEATSModel, NeuroDivergentModel, PerformanceMetrics,
    PromptData, StackType, StreamDataLoader, StreamEvent, TCNModel, TrainingConfig,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn generate_seasonal_events() -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let base_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Generate data with strong seasonal patterns for N-BEATS
    for day in 0..90 {
        // 3 months of data
        for hour in 0..24 {
            let timestamp = base_time + (day * 86400) + (hour * 3600);

            // Daily pattern
            let daily_pattern = ((hour as f64 - 12.0) / 6.0).tanh();

            // Weekly pattern
            let day_of_week = day % 7;
            let weekly_pattern = if day_of_week < 5 { 1.0 } else { 0.7 }; // Weekday vs weekend

            // Monthly pattern
            let monthly_pattern = ((day % 30) as f64 / 15.0 - 1.0).sin();

            // Trend component
            let trend = day as f64 * 0.5; // Gradual increase over time

            // Generate events for multiple agents
            for agent_id in 0..5 {
                // Agent-specific behavior
                let agent_efficiency = 1.0 + (agent_id as f64 * 0.1);

                events.push(StreamEvent {
                    timestamp,
                    agent_id: format!("agent_{}", agent_id),
                    event_type: match hour % 4 {
                        0 => EventType::TaskStarted,
                        1 => EventType::PromptGenerated,
                        2 => EventType::ResponseReceived,
                        _ => EventType::TaskCompleted,
                    },
                    performance_metrics: PerformanceMetrics {
                        latency_ms: 50.0 + trend + 20.0 * daily_pattern + 10.0 * monthly_pattern,
                        tokens_per_second: 100.0 * agent_efficiency * weekly_pattern
                            - 20.0 * daily_pattern,
                        memory_usage_mb: 250.0 + 50.0 * daily_pattern + trend * 0.5,
                        cpu_usage_percent: 40.0 + 30.0 * daily_pattern.abs() * weekly_pattern,
                        success_rate: 0.95 * weekly_pattern + 0.03 * monthly_pattern,
                    },
                    prompt_data: if hour % 3 == 0 {
                        Some(PromptData {
                            prompt_text: format!("Seasonal prompt at day {} hour {}", day, hour),
                            prompt_tokens: 50 + (daily_pattern * 20.0) as usize,
                            response_tokens: 100 + (daily_pattern * 40.0) as usize,
                            quality_score: 0.85 + 0.1 * weekly_pattern + 0.05 * monthly_pattern,
                        })
                    } else {
                        None
                    },
                });
            }
        }
    }

    events
}

async fn train_and_compare_models() -> Result<(), Box<dyn std::error::Error>> {
    // Load data
    println!("Loading seasonal data...");
    let data_loader = StreamDataLoader::new(1000, 72); // 3-day sequences
    let events = generate_seasonal_events();
    println!("Generated {} events", events.len());

    let dataset = data_loader.load_from_stream(events.into_iter()).await?;
    println!("Created {} training sequences\n", dataset.sequences.len());

    // Training configuration
    let config = TrainingConfig {
        epochs: 30,
        batch_size: 64,
        validation_split: 0.2,
        early_stopping_patience: 5,
        save_checkpoints: false,
        checkpoint_dir: "./checkpoints".to_string(),
    };

    // Create models
    let mut models: Vec<Box<dyn NeuroDivergentModel>> = vec![
        Box::new(LSTMModel::new(256, 3)),
        Box::new(TCNModel::new(vec![128, 128, 128, 128], 4)),
        Box::new(NBEATSModel::new(
            vec![StackType::Trend, StackType::Seasonality, StackType::Generic],
            5,
        )),
    ];

    // Train each model
    println!("Training models...");
    for model in &mut models {
        println!("\nTraining {} model...", model.name());
        let start = std::time::Instant::now();

        let metrics = model.train(&dataset, &config)?;

        let duration = start.elapsed();
        println!("  Training time: {:.2}s", duration.as_secs_f64());
        println!("  Best epoch: {}", metrics.best_epoch);
        println!("  Best loss: {:.4}", metrics.best_loss);
    }

    // Evaluate and compare models
    println!("\n\nEvaluating models...");
    let evaluator = ModelEvaluator::new();
    let selection_result = evaluator.evaluate_and_select(models, &dataset)?;

    // Display detailed comparison
    println!("\n==========================================");
    println!("         MODEL COMPARISON RESULTS         ");
    println!("==========================================\n");

    println!("WINNER: {} 🏆\n", selection_result.best_model);

    println!("Detailed Scores:");
    println!("----------------");

    // Create a score table
    let metrics = vec![
        "MSE",
        "MAE",
        "R2",
        "LatencyAccuracy",
        "SuccessRatePrediction",
    ];

    // Print header
    print!("{:<15}", "Model");
    for metric in &metrics {
        print!("{:>15}", metric);
    }
    println!();

    // Print separator
    print!("{:-<15}", "");
    for _ in &metrics {
        print!("{:-<15}", "");
    }
    println!();

    // Print scores
    for score in &selection_result.all_scores {
        print!("{:<15}", score.model_name);
        for metric in &metrics {
            if let Some(value) = score.scores.get(*metric) {
                print!("{:>15.4}", value);
            } else {
                print!("{:>15}", "N/A");
            }
        }
        println!();
    }

    // Analysis
    println!("\n\nModel Analysis:");
    println!("---------------");

    for score in &selection_result.all_scores {
        println!("\n{}:", score.model_name);

        let mse = score.scores.get("MSE").unwrap_or(&0.0);
        let r2 = score.scores.get("R2").unwrap_or(&0.0);
        let latency_acc = score.scores.get("LatencyAccuracy").unwrap_or(&0.0);

        if *mse < 50.0 {
            println!("  ✓ Excellent prediction accuracy (MSE < 50)");
        } else if *mse < 100.0 {
            println!("  ✓ Good prediction accuracy (MSE < 100)");
        } else {
            println!("  ⚠ Poor prediction accuracy (MSE > 100)");
        }

        if *r2 > 0.8 {
            println!("  ✓ Strong correlation with actual values (R² > 0.8)");
        } else if *r2 > 0.6 {
            println!("  ✓ Moderate correlation with actual values (R² > 0.6)");
        } else {
            println!("  ⚠ Weak correlation with actual values (R² < 0.6)");
        }

        if *latency_acc > 0.8 {
            println!("  ✓ Excellent latency prediction (>80% within threshold)");
        } else if *latency_acc > 0.6 {
            println!("  ✓ Good latency prediction (>60% within threshold)");
        } else {
            println!("  ⚠ Poor latency prediction (<60% within threshold)");
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RUV Swarm ML Training - Model Comparison");
    println!("=======================================\n");

    train_and_compare_models().await?;

    println!("\n\nComparison complete! The best model can now be used for:");
    println!("  • Real-time performance prediction");
    println!("  • Prompt optimization");
    println!("  • Resource allocation");
    println!("  • Anomaly detection");

    Ok(())
}
