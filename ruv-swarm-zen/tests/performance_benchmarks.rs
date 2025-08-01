//! Performance benchmarks and load testing for swarm behavior
//! 
//! These tests measure throughput, latency, scalability limits,
//! memory usage, and overall system performance under various loads.

use ruv_swarm_core::{
    agent::{Agent, AgentType},
    swarm::{Swarm, SwarmConfig, Topology},
    task::{Task, TaskResult},
    metrics::{PerformanceMetrics, SystemMetrics},
};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::runtime::Runtime;
use sysinfo::{System, SystemExt, ProcessExt};

/// Helper to create a benchmark swarm
async fn create_benchmark_swarm(
    topology: Topology,
    agent_count: usize,
) -> Swarm {
    let config = SwarmConfig {
        topology,
        max_agents: agent_count * 2,
        heartbeat_interval: Duration::from_secs(5),
        task_timeout: Duration::from_secs(300),
        persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
    };
    
    let mut swarm = Swarm::new(config).await.unwrap();
    
    // Pre-spawn agents
    for _ in 0..agent_count {
        swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
    }
    
    swarm
}

/// Generate synthetic workload
fn generate_workload(size: usize) -> Vec<Task> {
    (0..size).map(|i| {
        Task::ComputeIntensive {
            id: format!("task_{}", i),
            complexity: 1000 + (i % 1000),
            data: vec![i as f64; 100],
        }
    }).collect()
}

/// Benchmark throughput for different swarm sizes
fn benchmark_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("throughput");
    
    for agent_count in [1, 5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("agents", agent_count),
            agent_count,
            |b, &agent_count| {
                b.to_async(&rt).iter(|| async move {
                    let swarm = create_benchmark_swarm(Topology::Mesh, agent_count).await;
                    let tasks = generate_workload(100);
                    
                    let start = Instant::now();
                    let mut handles = vec![];
                    
                    for task in tasks {
                        let handle = swarm.orchestrate(task);
                        handles.push(handle);
                    }
                    
                    // Wait for all tasks
                    for handle in handles {
                        handle.await.unwrap();
                    }
                    
                    start.elapsed()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark latency for single task execution
fn benchmark_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("latency");
    
    for topology in [Topology::Star, Topology::Mesh, Topology::Hierarchical].iter() {
        group.bench_with_input(
            BenchmarkId::new("topology", format!("{:?}", topology)),
            topology,
            |b, topology| {
                b.to_async(&rt).iter(|| async move {
                    let swarm = create_benchmark_swarm(topology.clone(), 10).await;
                    let task = Task::ComputeIntensive {
                        id: "latency_test".to_string(),
                        complexity: 100,
                        data: vec![1.0; 50],
                    };
                    
                    let start = Instant::now();
                    swarm.orchestrate(task).await.unwrap();
                    start.elapsed()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark message passing overhead
fn benchmark_message_passing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("message_passing");
    
    for message_size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("size_bytes", message_size),
            message_size,
            |b, &message_size| {
                b.to_async(&rt).iter(|| async move {
                    let swarm = create_benchmark_swarm(Topology::FullyConnected, 10).await;
                    let agents = swarm.get_agent_ids().await;
                    
                    let data = vec![0u8; message_size];
                    let start = Instant::now();
                    
                    // Send messages between all agent pairs
                    for i in 0..agents.len() {
                        for j in 0..agents.len() {
                            if i != j {
                                swarm.send_direct_message(
                                    &agents[i],
                                    &agents[j],
                                    data.clone(),
                                ).await.unwrap();
                            }
                        }
                    }
                    
                    start.elapsed()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark agent spawning performance
fn benchmark_agent_spawning(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("agent_spawning");
    
    group.bench_function("spawn_100_agents", |b| {
        b.to_async(&rt).iter(|| async {
            let config = SwarmConfig {
                topology: Topology::Star,
                max_agents: 200,
                heartbeat_interval: Duration::from_secs(5),
                task_timeout: Duration::from_secs(300),
                persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
            };
            
            let mut swarm = Swarm::new(config).await.unwrap();
            let start = Instant::now();
            
            for _ in 0..100 {
                swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
            }
            
            start.elapsed()
        });
    });
    
    group.finish();
}

/// Load test with concurrent tasks
#[tokio::test]
async fn load_test_concurrent_tasks() {
    let concurrent_levels = vec![10, 50, 100, 500, 1000];
    let mut results = vec![];
    
    for concurrent_tasks in concurrent_levels {
        let swarm = create_benchmark_swarm(Topology::Mesh, 20).await;
        let tasks = generate_workload(concurrent_tasks);
        
        let start = Instant::now();
        let mut handles = vec![];
        
        // Submit all tasks concurrently
        for task in tasks {
            handles.push(tokio::spawn({
                let swarm = swarm.clone();
                async move {
                    swarm.orchestrate(task).await
                }
            }));
        }
        
        // Wait for completion
        let mut completed = 0;
        let mut failed = 0;
        
        for handle in handles {
            match handle.await {
                Ok(Ok(_)) => completed += 1,
                _ => failed += 1,
            }
        }
        
        let duration = start.elapsed();
        let throughput = completed as f64 / duration.as_secs_f64();
        
        results.push((concurrent_tasks, completed, failed, throughput));
        
        println!(
            "Concurrent tasks: {}, Completed: {}, Failed: {}, Throughput: {:.2} tasks/sec",
            concurrent_tasks, completed, failed, throughput
        );
    }
    
    // Verify performance scales appropriately
    let base_throughput = results[0].3;
    let max_throughput = results.iter().map(|r| r.3).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    assert!(max_throughput > base_throughput * 2.0, "Throughput should scale with concurrency");
}

/// Memory usage under load
#[tokio::test]
async fn test_memory_usage() {
    let mut system = System::new_all();
    let process_id = sysinfo::get_current_pid().unwrap();
    
    // Baseline memory
    system.refresh_process(process_id);
    let baseline_memory = system.process(process_id).unwrap().memory();
    
    // Create large swarm
    let swarm = create_benchmark_swarm(Topology::Mesh, 100).await;
    
    // Generate heavy workload
    let tasks = generate_workload(1000);
    let mut handles = vec![];
    
    for task in tasks {
        handles.push(swarm.orchestrate(task));
    }
    
    // Measure peak memory during execution
    let mut peak_memory = baseline_memory;
    let monitor_handle = tokio::spawn(async move {
        let mut max_mem = 0;
        for _ in 0..30 {
            system.refresh_process(process_id);
            if let Some(process) = system.process(process_id) {
                max_mem = max_mem.max(process.memory());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        max_mem
    });
    
    // Wait for tasks
    for handle in handles {
        handle.await.unwrap();
    }
    
    peak_memory = monitor_handle.await.unwrap();
    
    // Calculate memory overhead
    let memory_per_agent = (peak_memory - baseline_memory) / 100;
    let memory_mb = memory_per_agent / 1024;
    
    println!("Memory per agent: {} MB", memory_mb);
    assert!(memory_mb < 50, "Memory usage per agent should be reasonable");
}

/// Scalability limits test
#[tokio::test]
async fn test_scalability_limits() {
    let agent_counts = vec![10, 50, 100, 200, 500];
    let mut results = vec![];
    
    for agent_count in agent_counts {
        let start = Instant::now();
        
        match create_benchmark_swarm(Topology::Star, agent_count).await {
            swarm => {
                let spawn_time = start.elapsed();
                
                // Test task distribution time
                let task_start = Instant::now();
                let task = Task::Broadcast {
                    message: "scalability test".to_string(),
                };
                
                swarm.orchestrate(task).await.unwrap();
                let task_time = task_start.elapsed();
                
                results.push((agent_count, spawn_time, task_time));
                
                println!(
                    "Agents: {}, Spawn time: {:?}, Task distribution: {:?}",
                    agent_count, spawn_time, task_time
                );
            }
        }
    }
    
    // Verify linear or better scaling
    for i in 1..results.len() {
        let (count1, _, time1) = results[i-1];
        let (count2, _, time2) = results[i];
        
        let scale_factor = count2 as f64 / count1 as f64;
        let time_factor = time2.as_secs_f64() / time1.as_secs_f64();
        
        // Task distribution should scale sub-linearly
        assert!(
            time_factor < scale_factor * 1.5,
            "Task distribution should scale well"
        );
    }
}

/// Network topology performance comparison
#[tokio::test]
async fn test_topology_performance() {
    let topologies = vec![
        Topology::Star,
        Topology::Mesh,
        Topology::Hierarchical,
        Topology::FullyConnected,
    ];
    
    let mut results = HashMap::new();
    
    for topology in topologies {
        let swarm = create_benchmark_swarm(topology.clone(), 20).await;
        let tasks = generate_workload(100);
        
        let start = Instant::now();
        let mut handles = vec![];
        
        for task in tasks {
            handles.push(swarm.orchestrate(task));
        }
        
        for handle in handles {
            handle.await.unwrap();
        }
        
        let duration = start.elapsed();
        let metrics = swarm.get_performance_metrics().await.unwrap();
        
        results.insert(format!("{:?}", topology), (duration, metrics));
    }
    
    // Analyze results
    for (topology, (duration, metrics)) in &results {
        println!(
            "Topology: {}, Duration: {:?}, Avg latency: {:.2}ms, Message count: {}",
            topology,
            duration,
            metrics.average_latency_ms,
            metrics.total_messages
        );
    }
    
    // Star should have lowest message count
    let star_messages = results.get("Star").unwrap().1.total_messages;
    let mesh_messages = results.get("Mesh").unwrap().1.total_messages;
    assert!(star_messages < mesh_messages, "Star topology should use fewer messages");
}

/// CPU utilization test
#[tokio::test]
async fn test_cpu_utilization() {
    let swarm = create_benchmark_swarm(Topology::Mesh, 20).await;
    let num_cpus = num_cpus::get();
    
    // Monitor CPU usage
    let monitor = swarm.start_resource_monitoring().await.unwrap();
    
    // Generate CPU-intensive workload
    let tasks: Vec<_> = (0..num_cpus * 10).map(|i| {
        Task::ComputeIntensive {
            id: format!("cpu_test_{}", i),
            complexity: 10000,
            data: vec![i as f64; 1000],
        }
    }).collect();
    
    let start = Instant::now();
    let mut handles = vec![];
    
    for task in tasks {
        handles.push(swarm.orchestrate(task));
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    let duration = start.elapsed();
    let cpu_stats = monitor.stop().await.unwrap();
    
    println!(
        "CPU cores: {}, Tasks: {}, Duration: {:?}, Avg CPU: {:.1}%",
        num_cpus,
        num_cpus * 10,
        duration,
        cpu_stats.average_cpu_usage
    );
    
    // Should utilize available CPUs efficiently
    assert!(
        cpu_stats.average_cpu_usage > 70.0,
        "Should utilize most available CPU capacity"
    );
    
    assert!(
        cpu_stats.average_cpu_usage < 95.0,
        "Should not oversaturate CPUs"
    );
}

/// Benchmark checkpoint performance
#[tokio::test]
async fn test_checkpoint_performance() {
    let swarm = create_benchmark_swarm(Topology::Star, 50).await;
    
    // Add some state
    let tasks = generate_workload(100);
    for task in tasks {
        swarm.submit_task(task).await.unwrap();
    }
    
    // Benchmark checkpoint creation
    let start = Instant::now();
    let checkpoint = swarm.create_checkpoint().await.unwrap();
    let create_time = start.elapsed();
    
    // Benchmark checkpoint save
    let start = Instant::now();
    swarm.save_checkpoint(&checkpoint).await.unwrap();
    let save_time = start.elapsed();
    
    // Benchmark checkpoint restore
    let start = Instant::now();
    swarm.restore_from_checkpoint(checkpoint.id()).await.unwrap();
    let restore_time = start.elapsed();
    
    println!(
        "Checkpoint times - Create: {:?}, Save: {:?}, Restore: {:?}",
        create_time, save_time, restore_time
    );
    
    // All operations should complete in reasonable time
    assert!(create_time < Duration::from_secs(1));
    assert!(save_time < Duration::from_secs(2));
    assert!(restore_time < Duration::from_secs(2));
}

// Criterion benchmark groups
criterion_group!(
    benches,
    benchmark_throughput,
    benchmark_latency,
    benchmark_message_passing,
    benchmark_agent_spawning
);

criterion_main!(benches);