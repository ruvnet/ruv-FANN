//! Basic example of using the ML training pipeline

use ruv_swarm_ml_training::{
    EventType, PerformanceMetrics, PromptData, StreamEvent, TrainingConfig, TrainingPipeline,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn generate_mock_events() -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let base_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Generate mock training data
    for i in 0..1000 {
        let timestamp = base_time + (i * 60); // 1 minute intervals
        let hour = (timestamp / 3600) % 24;

        // Simulate performance variations based on time of day
        let latency_base = 50.0 + 30.0 * (hour as f64 / 12.0).sin();
        let tokens_base = 100.0 - 20.0 * (hour as f64 / 12.0).cos();

        // Add some noise
        let noise = ((i * 17) % 100) as f64 / 100.0 - 0.5;

        events.push(StreamEvent {
            timestamp,
            agent_id: format!("agent_{}", i % 5),
            event_type: if i % 3 == 0 {
                EventType::TaskCompleted
            } else if i % 3 == 1 {
                EventType::PromptGenerated
            } else {
                EventType::ResponseReceived
            },
            performance_metrics: PerformanceMetrics {
                latency_ms: latency_base + noise * 10.0,
                tokens_per_second: tokens_base + noise * 5.0,
                memory_usage_mb: 256.0 + noise * 50.0,
                cpu_usage_percent: 40.0 + noise * 20.0,
                success_rate: 0.95 + noise * 0.05,
            },
            prompt_data: if i % 2 == 0 {
                Some(PromptData {
                    prompt_text: format!("Prompt {}", i),
                    prompt_tokens: 50 + (i % 50),
                    response_tokens: 100 + (i % 100),
                    quality_score: 0.8 + noise * 0.1,
                })
            } else {
                None
            },
        });
    }

    events
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RUV Swarm ML Training Pipeline - Basic Example");
    println!("=============================================\n");

    // Configure training
    let config = TrainingConfig {
        epochs: 50,
        batch_size: 32,
        validation_split: 0.2,
        early_stopping_patience: 5,
        save_checkpoints: false,
        checkpoint_dir: "./model_checkpoints".to_string(),
    };

    // Create training pipeline
    let mut pipeline = TrainingPipeline::new(config);

    // Generate mock training data
    println!("Generating mock training data...");
    let events = generate_mock_events();
    println!("Generated {} events\n", events.len());

    // Run training pipeline
    println!("Starting training pipeline...\n");
    let result = pipeline.run(events.into_iter()).await?;

    // Display results
    println!("\n\nTraining Pipeline Results");
    println!("========================");
    println!("Best Model: {}", result.best_model);
    println!("\nDataset Info:");
    println!("  Total Samples: {}", result.dataset_metadata.total_samples);
    println!("  Feature Count: {}", result.dataset_metadata.feature_count);
    println!(
        "  Sequence Length: {}",
        result.dataset_metadata.sequence_length
    );
    println!(
        "  Number of Agents: {}",
        result.dataset_metadata.agents.len()
    );

    println!("\nModel Performance Scores:");
    for model_score in &result.model_scores {
        println!("\n  {}:", model_score.model_name);
        for (metric, score) in &model_score.scores {
            println!("    {}: {:.4}", metric, score);
        }
    }

    println!("\n\nPipeline completed successfully!");

    Ok(())
}
