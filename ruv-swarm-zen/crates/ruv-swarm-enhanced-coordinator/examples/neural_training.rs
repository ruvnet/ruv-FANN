//! Distributed Neural Network Training Example
//! 
//! This example demonstrates how to use the swarm for distributed
//! neural network training with data parallelism.

use ruv_swarm::{
    agent::{AgentType, CognitiveStyle},
    swarm::{Swarm, SwarmConfig},
    topology::Topology,
    neural::{NeuralTask, TrainingConfig},
    Result,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use ruv_lib::{Network, ActivationFunc, TrainAlgorithm, TrainData};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("=== Distributed Neural Training Example ===\n");

    // Create swarm configuration for neural training
    let config = SwarmConfig {
        max_agents: 8,
        topology: Topology::HierarchicalRing,
        cognitive_diversity: true,
        heartbeat_interval: Duration::from_secs(2),
        ..Default::default()
    };

    // Initialize swarm
    let mut swarm = Swarm::new(config)?;

    // Spawn coordinator agent
    println!("Spawning coordinator...");
    let coordinator = swarm.spawn_agent_with_style(
        AgentType::Coordinator,
        CognitiveStyle::Analytical
    ).await?;

    // Spawn worker agents with diverse cognitive styles
    println!("Spawning worker agents with cognitive diversity...");
    let workers = vec![
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Creative).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Practical).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Strategic).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::DetailOriented).await?,
    ];

    println!("  Coordinator: {:?}", coordinator);
    for (i, worker) in workers.iter().enumerate() {
        println!("  Worker {}: {:?}", i + 1, worker);
    }

    // Create neural network
    println!("\nCreating neural network...");
    let layers = vec![4, 8, 8, 3]; // Iris dataset: 4 inputs, 3 outputs
    let mut network = Network::new(&layers)?;
    network.set_activation_func(ActivationFunc::Sigmoid);

    // Load training data (simulated for example)
    let train_data = create_sample_training_data()?;
    
    // Configure distributed training
    let training_config = TrainingConfig {
        algorithm: TrainAlgorithm::ParallelRprop,
        max_epochs: 1000,
        desired_error: 0.001,
        batch_size: 32,
        learning_rate: 0.7,
        momentum: 0.1,
        early_stopping: true,
        validation_split: 0.2,
    };

    // Create neural training task
    let neural_task = NeuralTask {
        network: Arc::new(network),
        training_data: Arc::new(train_data),
        config: training_config,
        partition_strategy: "data_parallel".to_string(),
    };

    println!("\nStarting distributed training...");
    println!("  Algorithm: Parallel RPROP");
    println!("  Workers: {}", workers.len());
    println!("  Max epochs: {}", training_config.max_epochs);
    println!("  Target error: {}", training_config.desired_error);

    let start_time = Instant::now();

    // Execute distributed training
    let result = swarm.train_neural_network(neural_task).await?;

    let training_time = start_time.elapsed();

    // Display results
    println!("\nTraining completed in {:?}", training_time);
    println!("Final error: {}", result.final_error);
    println!("Epochs trained: {}", result.epochs);
    println!("Training history:");
    
    for (epoch, error) in result.error_history.iter().step_by(100) {
        println!("  Epoch {}: error = {:.6}", epoch, error);
    }

    // Test the trained network
    println!("\nTesting trained network...");
    let test_samples = create_test_samples()?;
    
    for (i, sample) in test_samples.iter().enumerate() {
        let output = result.network.run(&sample.input)?;
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        println!("  Sample {}: input={:?} -> class={}", 
                 i + 1, sample.input, predicted_class);
    }

    // Performance statistics
    let stats = swarm.get_neural_stats();
    println!("\nDistributed Training Statistics:");
    println!("  Total gradients computed: {}", stats.gradients_computed);
    println!("  Average gradient time: {:?}", stats.avg_gradient_time);
    println!("  Communication overhead: {:?}", stats.communication_overhead);
    println!("  Speedup factor: {:.2}x", stats.speedup_factor);

    // Shutdown
    swarm.shutdown().await?;
    println!("\nExample completed!");
    
    Ok(())
}

// Helper function to create sample training data
fn create_sample_training_data() -> Result<TrainData> {
    // Simulated Iris-like dataset
    let inputs = vec![
        vec![5.1, 3.5, 1.4, 0.2], // Setosa
        vec![4.9, 3.0, 1.4, 0.2],
        vec![7.0, 3.2, 4.7, 1.4], // Versicolor
        vec![6.4, 3.2, 4.5, 1.5],
        vec![6.3, 3.3, 6.0, 2.5], // Virginica
        vec![5.8, 2.7, 5.1, 1.9],
        // ... more samples
    ];
    
    let outputs = vec![
        vec![1.0, 0.0, 0.0], // Setosa
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0], // Versicolor
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0], // Virginica
        vec![0.0, 0.0, 1.0],
        // ... more outputs
    ];

    TrainData::new(inputs, outputs)
}

// Helper function to create test samples
fn create_test_samples() -> Result<Vec<TestSample>> {
    Ok(vec![
        TestSample { input: vec![5.0, 3.4, 1.5, 0.2] }, // Should be Setosa
        TestSample { input: vec![6.5, 3.0, 4.5, 1.3] }, // Should be Versicolor
        TestSample { input: vec![6.0, 3.0, 5.5, 2.0] }, // Should be Virginica
    ])
}

struct TestSample {
    input: Vec<f32>,
}