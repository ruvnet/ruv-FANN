//! Chaos testing for swarm resilience
//! 
//! These tests introduce random failures, load spikes, and resource constraints
//! to validate the swarm's ability to maintain stability under chaotic conditions.

use ruv_swarm_core::{
    agent::{Agent, AgentType},
    swarm::{Swarm, SwarmConfig, Topology},
    task::{Task, TaskResult},
    chaos::{ChaosEngine, ChaosEvent, ChaosScenario},
    metrics::StabilityMetrics,
};
use rand::{Rng, distributions::WeightedIndex, prelude::*};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Chaos configuration for different test scenarios
#[derive(Clone)]
struct ChaosConfig {
    agent_failure_rate: f64,
    network_failure_rate: f64,
    message_corruption_rate: f64,
    resource_spike_probability: f64,
    partition_probability: f64,
    byzantine_agent_probability: f64,
    duration: Duration,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            agent_failure_rate: 0.1,
            network_failure_rate: 0.05,
            message_corruption_rate: 0.02,
            resource_spike_probability: 0.1,
            partition_probability: 0.05,
            byzantine_agent_probability: 0.01,
            duration: Duration::from_secs(60),
        }
    }
}

/// Create a chaos-enabled swarm
async fn create_chaos_swarm(chaos_config: ChaosConfig) -> Result<(Swarm, ChaosEngine), Box<dyn std::error::Error>> {
    let config = SwarmConfig {
        topology: Topology::Mesh,
        max_agents: 50,
        heartbeat_interval: Duration::from_millis(500),
        task_timeout: Duration::from_secs(30),
        persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
        chaos_mode: true,
    };
    
    let swarm = Swarm::new(config).await?;
    let chaos_engine = ChaosEngine::new(swarm.clone(), chaos_config);
    
    Ok((swarm, chaos_engine))
}

/// Generate random workload
fn generate_chaos_workload(size: usize) -> Vec<Task> {
    let mut rng = thread_rng();
    
    (0..size).map(|i| {
        match rng.gen_range(0..4) {
            0 => Task::ComputeIntensive {
                id: format!("compute_{}", i),
                complexity: rng.gen_range(100..10000),
                data: vec![rng.gen(); rng.gen_range(10..100)],
            },
            1 => Task::DataProcessing {
                input_data: vec![rng.gen(); rng.gen_range(50..500)],
                operations: vec!["filter", "map", "reduce"],
                output_format: "binary",
            },
            2 => Task::NetworkIntensive {
                peers: rng.gen_range(2..10),
                message_size: rng.gen_range(100..10000),
                rounds: rng.gen_range(1..20),
            },
            _ => Task::MixedWorkload {
                compute_complexity: rng.gen_range(100..1000),
                data_size: rng.gen_range(100..1000),
                network_ops: rng.gen_range(10..100),
            },
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_random_agent_failures() {
        let chaos_config = ChaosConfig {
            agent_failure_rate: 0.2, // 20% failure rate
            ..Default::default()
        };
        
        let (mut swarm, chaos_engine) = create_chaos_swarm(chaos_config).await.unwrap();
        
        // Spawn agents
        let mut agents = vec![];
        for _ in 0..20 {
            agents.push(swarm.spawn(AgentType::NeuralProcessor).await.unwrap());
        }
        
        // Start chaos engine
        let chaos_handle = tokio::spawn(async move {
            chaos_engine.run().await
        });
        
        // Submit continuous workload
        let mut completed_tasks = 0;
        let mut failed_tasks = 0;
        let total_tasks = 100;
        
        for i in 0..total_tasks {
            let task = Task::SimpleComputation {
                input: vec![i as f64; 10],
            };
            
            match swarm.orchestrate(task).await {
                Ok(_) => completed_tasks += 1,
                Err(_) => failed_tasks += 1,
            }
            
            // Add some delay
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        // Stop chaos
        chaos_handle.abort();
        
        // Despite 20% failure rate, most tasks should complete
        let success_rate = completed_tasks as f64 / total_tasks as f64;
        assert!(success_rate > 0.7, "Success rate too low: {}", success_rate);
        
        // Verify swarm recovered
        let health = swarm.get_health_status().await.unwrap();
        assert!(health.operational_percentage > 0.5);
    }

    #[tokio::test]
    async fn test_load_spikes() {
        let (mut swarm, _) = create_chaos_swarm(ChaosConfig::default()).await.unwrap();
        
        // Spawn baseline agents
        for _ in 0..10 {
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        }
        
        // Monitor resource usage
        let metrics_collector = swarm.start_metrics_collection().await.unwrap();
        
        // Normal load phase
        let normal_tasks = generate_chaos_workload(20);
        for task in normal_tasks {
            swarm.submit_task(task).await.unwrap();
        }
        
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Sudden load spike
        let spike_tasks = generate_chaos_workload(200);
        let spike_start = Instant::now();
        
        let mut handles = vec![];
        for task in spike_tasks {
            handles.push(swarm.orchestrate(task));
        }
        
        // Wait for completion
        let results = futures::future::join_all(handles).await;
        let spike_duration = spike_start.elapsed();
        
        // Analyze metrics
        let metrics = metrics_collector.get_metrics().await;
        
        // Verify swarm handled the spike
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        assert!(success_count as f64 / results.len() as f64 > 0.8);
        
        // Check if swarm auto-scaled
        let final_agent_count = swarm.agent_count();
        assert!(final_agent_count > 10, "Swarm should have scaled up");
        
        println!("Load spike handled in {:?} with {} agents", spike_duration, final_agent_count);
    }

    #[tokio::test]
    async fn test_resource_constraints() {
        let config = SwarmConfig {
            topology: Topology::Star,
            max_agents: 20,
            heartbeat_interval: Duration::from_millis(500),
            task_timeout: Duration::from_secs(30),
            persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
            memory_limit: Some(50 * 1024 * 1024), // 50MB limit
            cpu_limit: Some(0.5), // 50% CPU
        };
        
        let mut swarm = Swarm::new(config).await.unwrap();
        
        // Spawn agents
        for _ in 0..10 {
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        }
        
        // Generate memory-intensive tasks
        let mut tasks = vec![];
        for i in 0..50 {
            tasks.push(Task::MemoryIntensive {
                data_size_mb: 5, // Each task uses 5MB
            });
        }
        
        // Submit tasks and track resource exhaustion
        let mut resource_errors = 0;
        let mut completed = 0;
        
        for task in tasks {
            match swarm.submit_task(task).await {
                Ok(task_id) => {
                    // Check if task completes
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if let Ok(status) = swarm.get_task_status(&task_id).await {
                        if matches!(status, TaskStatus::Completed(_)) {
                            completed += 1;
                        }
                    }
                }
                Err(SwarmError::ResourceExhausted) => resource_errors += 1,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
        
        // Should hit resource limits
        assert!(resource_errors > 0, "Should have encountered resource limits");
        assert!(completed > 0, "Some tasks should have completed");
        
        // Verify resource management
        let resource_stats = swarm.get_resource_stats().await.unwrap();
        assert!(resource_stats.memory_usage_mb <= 50);
    }

    #[tokio::test]
    async fn test_cascading_chaos() {
        let chaos_config = ChaosConfig {
            agent_failure_rate: 0.1,
            network_failure_rate: 0.1,
            message_corruption_rate: 0.05,
            partition_probability: 0.1,
            ..Default::default()
        };
        
        let (mut swarm, chaos_engine) = create_chaos_swarm(chaos_config).await.unwrap();
        
        // Create interconnected agent network
        let coordinator = swarm.spawn(AgentType::Coordinator).await.unwrap();
        let mut workers = vec![];
        
        for _ in 0..5 {
            let worker = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
            swarm.add_dependency(&coordinator, &worker).await.unwrap();
            workers.push(worker);
        }
        
        // Start chaos
        let chaos_handle = tokio::spawn(async move {
            chaos_engine.run_scenario(ChaosScenario::CascadingFailure).await
        });
        
        // Submit hierarchical tasks
        let mut task_results = HashMap::new();
        
        for i in 0..20 {
            let task = Task::HierarchicalComputation {
                coordinator: coordinator.clone(),
                subtasks: vec![
                    ("phase1", vec![i as f64; 10]),
                    ("phase2", vec![(i * 2) as f64; 10]),
                    ("phase3", vec![(i * 3) as f64; 10]),
                ],
            };
            
            let task_id = format!("hierarchical_{}", i);
            match swarm.orchestrate_with_id(task, task_id.clone()).await {
                Ok(result) => task_results.insert(task_id, (true, result)),
                Err(e) => task_results.insert(task_id, (false, e.to_string())),
            };
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        // Stop chaos
        chaos_handle.abort();
        
        // Analyze results
        let successful = task_results.values().filter(|(success, _)| *success).count();
        let failed = task_results.len() - successful;
        
        println!("Cascading chaos: {} successful, {} failed", successful, failed);
        
        // Despite cascading failures, some tasks should complete
        assert!(successful > 0, "No tasks completed under cascading chaos");
        
        // Verify system stability metrics
        let stability = swarm.calculate_stability_score().await.unwrap();
        assert!(stability > 0.3, "System too unstable: {}", stability);
    }

    #[tokio::test]
    async fn test_adaptive_chaos_response() {
        let (mut swarm, chaos_engine) = create_chaos_swarm(ChaosConfig::default()).await.unwrap();
        
        // Enable adaptive responses
        swarm.enable_adaptive_mode().await.unwrap();
        
        // Spawn initial agents
        for _ in 0..10 {
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        }
        
        // Track adaptation metrics
        let adaptation_monitor = swarm.start_adaptation_monitoring().await.unwrap();
        
        // Run progressive chaos scenarios
        let scenarios = vec![
            ChaosScenario::LightLoad,
            ChaosScenario::ModerateFailures,
            ChaosScenario::HeavyNetworkIssues,
            ChaosScenario::ExtremeChaos,
        ];
        
        for scenario in scenarios {
            println!("Running scenario: {:?}", scenario);
            
            // Apply chaos
            let scenario_handle = tokio::spawn({
                let engine = chaos_engine.clone();
                async move {
                    engine.run_scenario(scenario).await
                }
            });
            
            // Submit workload
            let tasks = generate_chaos_workload(50);
            let mut success_rate = 0.0;
            
            for task in tasks {
                if swarm.orchestrate(task).await.is_ok() {
                    success_rate += 1.0;
                }
            }
            
            success_rate /= 50.0;
            
            // Stop scenario
            scenario_handle.abort();
            
            // Check adaptations
            let adaptations = adaptation_monitor.get_recent_adaptations().await;
            assert!(!adaptations.is_empty(), "Swarm should adapt to chaos");
            
            println!("Success rate: {:.2}%, Adaptations: {}", 
                     success_rate * 100.0, adaptations.len());
            
            // Allow recovery time
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        
        // Verify swarm learned from chaos
        let learning_metrics = swarm.get_learning_metrics().await.unwrap();
        assert!(learning_metrics.adaptation_effectiveness > 0.5);
    }

    #[tokio::test]
    async fn test_chaos_recovery_time() {
        let chaos_config = ChaosConfig {
            agent_failure_rate: 0.3,
            network_failure_rate: 0.2,
            ..Default::default()
        };
        
        let (mut swarm, chaos_engine) = create_chaos_swarm(chaos_config).await.unwrap();
        
        // Spawn agents
        for _ in 0..20 {
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        }
        
        // Baseline performance
        let baseline_start = Instant::now();
        let baseline_task = Task::SimpleComputation {
            input: vec![1.0; 100],
        };
        swarm.orchestrate(baseline_task).await.unwrap();
        let baseline_time = baseline_start.elapsed();
        
        // Apply chaos for 10 seconds
        let chaos_handle = tokio::spawn(async move {
            chaos_engine.run_for_duration(Duration::from_secs(10)).await
        });
        
        // Wait for chaos to end
        chaos_handle.await.unwrap();
        
        // Measure recovery time
        let recovery_start = Instant::now();
        let mut recovered = false;
        let mut recovery_time = Duration::from_secs(0);
        
        for i in 0..30 {
            let task = Task::SimpleComputation {
                input: vec![i as f64; 100],
            };
            
            let task_start = Instant::now();
            if let Ok(_) = swarm.orchestrate(task).await {
                let task_time = task_start.elapsed();
                
                // Consider recovered when performance is within 20% of baseline
                if task_time <= baseline_time * 120 / 100 {
                    recovered = true;
                    recovery_time = recovery_start.elapsed();
                    break;
                }
            }
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        assert!(recovered, "Swarm did not recover from chaos");
        assert!(recovery_time < Duration::from_secs(15), 
                "Recovery took too long: {:?}", recovery_time);
        
        println!("Recovery time: {:?}", recovery_time);
    }

    #[tokio::test]
    async fn test_chaos_patterns() {
        let (mut swarm, chaos_engine) = create_chaos_swarm(ChaosConfig::default()).await.unwrap();
        
        // Spawn diverse agents
        for agent_type in vec![
            AgentType::NeuralProcessor,
            AgentType::DataAnalyzer,
            AgentType::Coordinator,
            AgentType::PatternRecognizer,
        ] {
            for _ in 0..5 {
                swarm.spawn(agent_type.clone()).await.unwrap();
            }
        }
        
        // Define chaos patterns
        let patterns = vec![
            // Periodic failures
            ChaosPattern::Periodic {
                event: ChaosEvent::AgentCrash,
                interval: Duration::from_secs(5),
            },
            // Burst failures
            ChaosPattern::Burst {
                events: vec![ChaosEvent::NetworkPartition; 5],
                duration: Duration::from_secs(2),
            },
            // Random walk
            ChaosPattern::RandomWalk {
                min_severity: 0.1,
                max_severity: 0.8,
                step_size: 0.1,
            },
            // Targeted attacks
            ChaosPattern::Targeted {
                target_type: AgentType::Coordinator,
                event: ChaosEvent::Byzantine,
            },
        ];
        
        // Test each pattern
        for pattern in patterns {
            println!("Testing pattern: {:?}", pattern);
            
            // Reset swarm health
            swarm.heal_all_agents().await.unwrap();
            
            // Apply pattern
            let pattern_handle = tokio::spawn({
                let engine = chaos_engine.clone();
                let pattern = pattern.clone();
                async move {
                    engine.apply_pattern(pattern, Duration::from_secs(20)).await
                }
            });
            
            // Run workload during pattern
            let mut pattern_metrics = PatternMetrics::default();
            let tasks = generate_chaos_workload(30);
            
            for task in tasks {
                let start = Instant::now();
                match swarm.orchestrate(task).await {
                    Ok(_) => {
                        pattern_metrics.successes += 1;
                        pattern_metrics.total_latency += start.elapsed();
                    }
                    Err(_) => pattern_metrics.failures += 1,
                }
            }
            
            // Stop pattern
            pattern_handle.abort();
            
            // Analyze pattern impact
            let success_rate = pattern_metrics.successes as f64 / 
                              (pattern_metrics.successes + pattern_metrics.failures) as f64;
            let avg_latency = pattern_metrics.total_latency / pattern_metrics.successes as u32;
            
            println!("Pattern impact - Success rate: {:.2}%, Avg latency: {:?}",
                     success_rate * 100.0, avg_latency);
            
            // Different patterns should have different impacts
            match pattern {
                ChaosPattern::Periodic { .. } => {
                    assert!(success_rate > 0.7, "Periodic failures too disruptive");
                }
                ChaosPattern::Burst { .. } => {
                    assert!(success_rate > 0.5, "Burst failures too severe");
                }
                ChaosPattern::RandomWalk { .. } => {
                    assert!(success_rate > 0.6, "Random walk too chaotic");
                }
                ChaosPattern::Targeted { .. } => {
                    assert!(success_rate > 0.4, "Targeted attacks too effective");
                }
            }
        }
    }
}

// Helper types for chaos testing

#[derive(Debug, Clone)]
enum ChaosPattern {
    Periodic {
        event: ChaosEvent,
        interval: Duration,
    },
    Burst {
        events: Vec<ChaosEvent>,
        duration: Duration,
    },
    RandomWalk {
        min_severity: f64,
        max_severity: f64,
        step_size: f64,
    },
    Targeted {
        target_type: AgentType,
        event: ChaosEvent,
    },
}

#[derive(Default)]
struct PatternMetrics {
    successes: usize,
    failures: usize,
    total_latency: Duration,
}

// Extension trait for ChaosEngine
impl ChaosEngine {
    async fn run_for_duration(&self, duration: Duration) {
        let start = Instant::now();
        while start.elapsed() < duration {
            self.inject_random_failure().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    
    async fn apply_pattern(&self, pattern: ChaosPattern, duration: Duration) {
        let start = Instant::now();
        
        match pattern {
            ChaosPattern::Periodic { event, interval } => {
                while start.elapsed() < duration {
                    self.inject_event(event.clone()).await;
                    tokio::time::sleep(interval).await;
                }
            }
            ChaosPattern::Burst { events, duration: burst_duration } => {
                while start.elapsed() < duration {
                    for event in &events {
                        self.inject_event(event.clone()).await;
                    }
                    tokio::time::sleep(burst_duration).await;
                    tokio::time::sleep(burst_duration * 2).await; // Rest period
                }
            }
            ChaosPattern::RandomWalk { mut min_severity, max_severity, step_size } => {
                let mut severity = (min_severity + max_severity) / 2.0;
                let mut rng = thread_rng();
                
                while start.elapsed() < duration {
                    self.set_chaos_severity(severity).await;
                    
                    // Random walk
                    if rng.gen_bool(0.5) {
                        severity = (severity + step_size).min(max_severity);
                    } else {
                        severity = (severity - step_size).max(min_severity);
                    }
                    
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
            ChaosPattern::Targeted { target_type, event } => {
                while start.elapsed() < duration {
                    self.inject_targeted_event(target_type.clone(), event.clone()).await;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }
}