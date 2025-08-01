//! Comprehensive benchmarks for Enhanced Queen Coordinator
//! 
//! These benchmarks validate performance claims and measure real-world throughput,
//! latency, and scalability characteristics of the coordinator.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use tokio::runtime::Runtime;

use ruv_swarm_enhanced_coordinator::{
    enhanced_queen_coordinator::*,
    prelude::*,
};

/// Benchmark coordinator creation and initialization
fn bench_coordinator_creation(c: &mut Criterion) {
    c.bench_function("coordinator_creation", |b| {
        b.iter(|| {
            let config = CoordinatorConfig::default();
            black_box(EnhancedQueenCoordinator::new(config))
        })
    });
}

/// Benchmark agent registration performance
fn bench_agent_registration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("agent_registration");
    
    for agent_count in [1, 10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("sequential", agent_count),
            agent_count,
            |b, &agent_count| {
                b.to_async(&rt).iter(|| async {
                    let coordinator = EnhancedQueenCoordinator::new(CoordinatorConfig::default());
                    
                    for i in 0..agent_count {
                        let agent_id = format!("agent_{:04}", i);
                        let capabilities = vec!["compute".to_string(), "analysis".to_string()];
                        let pattern = match i % 6 {
                            0 => CognitivePattern::Convergent,
                            1 => CognitivePattern::Divergent,
                            2 => CognitivePattern::Lateral,
                            3 => CognitivePattern::Systems,
                            4 => CognitivePattern::Critical,
                            _ => CognitivePattern::Abstract,
                        };
                        
                        coordinator.register_agent(agent_id, capabilities, pattern).await.unwrap();
                    }
                    
                    black_box(coordinator)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark task submission performance
fn bench_task_submission(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("task_submission");
    group.measurement_time(Duration::from_secs(20));
    
    for task_count in [1, 10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("with_agents", task_count),
            task_count,
            |b, &task_count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        // Setup: Create coordinator with agents
                        let coordinator = EnhancedQueenCoordinator::new(CoordinatorConfig::default());
                        let rt_handle = tokio::runtime::Handle::current();
                        
                        rt_handle.block_on(async {
                            // Register agents for each cognitive pattern
                            for i in 0..20 {
                                let agent_id = format!("bench_agent_{:02}", i);
                                let capabilities = vec![
                                    "compute".to_string(),
                                    "analysis".to_string(),
                                    "io".to_string(),
                                ];
                                let pattern = match i % 6 {
                                    0 => CognitivePattern::Convergent,
                                    1 => CognitivePattern::Divergent,
                                    2 => CognitivePattern::Lateral,
                                    3 => CognitivePattern::Systems,
                                    4 => CognitivePattern::Critical,
                                    _ => CognitivePattern::Abstract,
                                };
                                
                                coordinator.register_agent(agent_id, capabilities, pattern).await.unwrap();
                            }
                        });
                        
                        coordinator
                    },
                    |coordinator| async move {
                        // Benchmark: Submit tasks
                        for i in 0..task_count {
                            let task = Task::new(format!("bench_task_{:04}", i), "compute")
                                .require_capability("compute")
                                .with_priority(if i % 10 == 0 { 
                                    TaskPriority::High 
                                } else { 
                                    TaskPriority::Normal 
                                });
                            
                            black_box(coordinator.submit_task(task).await.unwrap());
                        }
                    }
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark task completion and metrics tracking
fn bench_task_completion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("task_completion");
    group.measurement_time(Duration::from_secs(15));
    
    for task_count in [10, 50, 100, 250].iter() {
        group.bench_with_input(
            BenchmarkId::new("complete_tasks", task_count),
            task_count,
            |b, &task_count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        // Setup: Create coordinator with agents and submitted tasks
                        let coordinator = EnhancedQueenCoordinator::new(CoordinatorConfig::default());
                        let rt_handle = tokio::runtime::Handle::current();
                        
                        let assignments = rt_handle.block_on(async {
                            // Register agents
                            for i in 0..10 {
                                let agent_id = format!("completion_agent_{:02}", i);
                                let capabilities = vec!["compute".to_string()];
                                coordinator.register_agent(
                                    agent_id, 
                                    capabilities, 
                                    CognitivePattern::Convergent
                                ).await.unwrap();
                            }
                            
                            // Submit tasks
                            let mut assignments = Vec::new();
                            for i in 0..task_count {
                                let task = Task::new(format!("completion_task_{:04}", i), "compute")
                                    .require_capability("compute");
                                
                                let assignment = coordinator.submit_task(task).await.unwrap();
                                assignments.push(assignment);
                            }
                            
                            assignments
                        });
                        
                        (coordinator, assignments)
                    },
                    |(coordinator, assignments)| async move {
                        // Benchmark: Complete tasks
                        for assignment in assignments {
                            let task_result = TaskResult::success("Benchmark task completed")
                                .with_task_id(assignment.task_id.clone())
                                .with_execution_time(100); // 100ms execution time
                            
                            black_box(
                                coordinator.complete_task(assignment.task_id, task_result).await.unwrap()
                            );
                        }
                    }
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark strategic analysis performance
fn bench_strategic_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("strategic_analysis");
    
    for agent_count in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("analyze_swarm", agent_count),
            agent_count,
            |b, &agent_count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        // Setup: Create coordinator with varying numbers of agents
                        let coordinator = EnhancedQueenCoordinator::new(CoordinatorConfig::default());
                        let rt_handle = tokio::runtime::Handle::current();
                        
                        rt_handle.block_on(async {
                            for i in 0..agent_count {
                                let agent_id = format!("analysis_agent_{:04}", i);
                                let capabilities = vec![
                                    "compute".to_string(),
                                    "analysis".to_string(),
                                ];
                                let pattern = match i % 6 {
                                    0 => CognitivePattern::Convergent,
                                    1 => CognitivePattern::Divergent,
                                    2 => CognitivePattern::Lateral,
                                    3 => CognitivePattern::Systems,
                                    4 => CognitivePattern::Critical,
                                    _ => CognitivePattern::Abstract,
                                };
                                
                                coordinator.register_agent(agent_id, capabilities, pattern).await.unwrap();
                            }
                        });
                        
                        coordinator
                    },
                    |coordinator| async move {
                        black_box(coordinator.strategic_analysis().await)
                    }
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark performance report generation
fn bench_performance_report(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("performance_report");
    
    for completed_tasks in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("generate_report", completed_tasks),
            completed_tasks,
            |b, &completed_tasks| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        // Setup: Create coordinator with completed tasks
                        let coordinator = EnhancedQueenCoordinator::new(CoordinatorConfig::default());
                        let rt_handle = tokio::runtime::Handle::current();
                        
                        rt_handle.block_on(async {
                            // Register agents
                            for i in 0..20 {
                                let agent_id = format!("report_agent_{:02}", i);
                                let capabilities = vec!["compute".to_string()];
                                coordinator.register_agent(
                                    agent_id, 
                                    capabilities, 
                                    CognitivePattern::Convergent
                                ).await.unwrap();
                            }
                            
                            // Submit and complete tasks to generate performance data
                            for i in 0..completed_tasks {
                                let task = Task::new(format!("report_task_{:04}", i), "compute")
                                    .require_capability("compute");
                                
                                let assignment = coordinator.submit_task(task).await.unwrap();
                                
                                let task_result = TaskResult::success("Report benchmark task")
                                    .with_task_id(assignment.task_id.clone())
                                    .with_execution_time(50 + (i % 100) as u64); // Variable execution time
                                
                                coordinator.complete_task(assignment.task_id, task_result).await.unwrap();
                            }
                        });
                        
                        coordinator
                    },
                    |coordinator| async move {
                        black_box(coordinator.generate_performance_report().await)
                    }
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(30));
    
    for concurrency_level in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_task_submission", concurrency_level),
            concurrency_level,
            |b, &concurrency_level| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        // Setup: Create coordinator with agents
                        let coordinator = std::sync::Arc::new(
                            EnhancedQueenCoordinator::new(CoordinatorConfig::default())
                        );
                        let rt_handle = tokio::runtime::Handle::current();
                        
                        rt_handle.block_on(async {
                            // Register enough agents for concurrent operations
                            for i in 0..(concurrency_level * 2) {
                                let agent_id = format!("concurrent_agent_{:02}", i);
                                let capabilities = vec!["compute".to_string()];
                                coordinator.register_agent(
                                    agent_id, 
                                    capabilities, 
                                    CognitivePattern::Convergent
                                ).await.unwrap();
                            }
                        });
                        
                        coordinator
                    },
                    |coordinator| async move {
                        // Benchmark: Submit tasks concurrently
                        let mut handles = Vec::new();
                        
                        for i in 0..concurrency_level {
                            let coordinator_clone = coordinator.clone();
                            let handle = tokio::spawn(async move {
                                for j in 0..10 {
                                    let task = Task::new(
                                        format!("concurrent_task_{}_{:02}", i, j), 
                                        "compute"
                                    ).require_capability("compute");
                                    
                                    coordinator_clone.submit_task(task).await.unwrap();
                                }
                            });
                            handles.push(handle);
                        }
                        
                        // Wait for all concurrent operations to complete
                        for handle in handles {
                            handle.await.unwrap();
                        }
                        
                        black_box(())
                    }
                )
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage and scalability
fn bench_memory_scalability(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_scalability");
    group.measurement_time(Duration::from_secs(25));
    
    for scale_factor in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("large_scale_operations", scale_factor),
            scale_factor,
            |b, &scale_factor| {
                b.to_async(&rt).iter(|| async {
                    let coordinator = EnhancedQueenCoordinator::new(CoordinatorConfig {
                        max_agents: scale_factor as usize,
                        ..CoordinatorConfig::default()
                    });
                    
                    // Register agents up to scale factor
                    let agent_count = (scale_factor / 10).min(200); // Reasonable limit for benchmarking
                    for i in 0..agent_count {
                        let agent_id = format!("scale_agent_{:04}", i);
                        let capabilities = vec!["compute".to_string()];
                        let pattern = match i % 6 {
                            0 => CognitivePattern::Convergent,
                            1 => CognitivePattern::Divergent,
                            2 => CognitivePattern::Lateral,
                            3 => CognitivePattern::Systems,
                            4 => CognitivePattern::Critical,
                            _ => CognitivePattern::Abstract,
                        };
                        
                        coordinator.register_agent(agent_id, capabilities, pattern).await.unwrap();
                    }
                    
                    // Submit tasks
                    let task_count = scale_factor / 5;
                    for i in 0..task_count {
                        let task = Task::new(format!("scale_task_{:04}", i), "compute")
                            .require_capability("compute");
                        
                        coordinator.submit_task(task).await.unwrap();
                    }
                    
                    // Perform analysis to test memory usage with large datasets
                    let analysis = coordinator.strategic_analysis().await;
                    
                    black_box((coordinator, analysis))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark performance tracking with high throughput
fn bench_performance_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_tracking");
    
    for metric_count in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("track_metrics", metric_count),
            metric_count,
            |b, &metric_count| {
                b.iter_with_setup(
                    || {
                        // Setup: Create performance tracker
                        let mut tracker = AgentPerformanceTracker::new();
                        let agent_id = "benchmark_agent".to_string();
                        tracker.initialize_agent(&agent_id);
                        (tracker, agent_id)
                    },
                    |(mut tracker, agent_id)| {
                        // Benchmark: Track many performance metrics
                        for i in 0..metric_count {
                            let metrics = TaskMetrics {
                                response_time_ms: 100.0 + (i % 200) as f64,
                                success: i % 10 != 0, // 90% success rate
                                resource_efficiency: 0.8 + (i % 20) as f64 / 100.0,
                                cognitive_effectiveness: 0.7 + (i % 30) as f64 / 100.0,
                            };
                            
                            tracker.track_agent_performance(agent_id.clone(), metrics);
                        }
                        
                        black_box(tracker.get_performance_report())
                    }
                )
            },
        );
    }
    
    group.finish();
}

/// Comprehensive end-to-end workflow benchmark
fn bench_end_to_end_workflow(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("end_to_end_workflow", |b| {
        b.to_async(&rt).iter(|| async {
            let coordinator = EnhancedQueenCoordinator::new(CoordinatorConfig::default());
            
            // Phase 1: Agent registration
            for i in 0..25 {
                let agent_id = format!("workflow_agent_{:02}", i);
                let capabilities = vec!["compute".to_string(), "analysis".to_string()];
                let pattern = match i % 6 {
                    0 => CognitivePattern::Convergent,
                    1 => CognitivePattern::Divergent,
                    2 => CognitivePattern::Lateral,
                    3 => CognitivePattern::Systems,
                    4 => CognitivePattern::Critical,
                    _ => CognitivePattern::Abstract,
                };
                
                coordinator.register_agent(agent_id, capabilities, pattern).await.unwrap();
            }
            
            // Phase 2: Task submission
            let mut assignments = Vec::new();
            for i in 0..100 {
                let task = Task::new(format!("workflow_task_{:03}", i), "compute")
                    .require_capability("compute")
                    .with_priority(if i % 20 == 0 { 
                        TaskPriority::High 
                    } else { 
                        TaskPriority::Normal 
                    });
                
                let assignment = coordinator.submit_task(task).await.unwrap();
                assignments.push(assignment);
            }
            
            // Phase 3: Task completion
            for assignment in assignments {
                let task_result = TaskResult::success("Workflow task completed")
                    .with_task_id(assignment.task_id.clone())
                    .with_execution_time(75 + (assignment.task_id.0.len() % 50) as u64);
                
                coordinator.complete_task(assignment.task_id, task_result).await.unwrap();
            }
            
            // Phase 4: Analysis and reporting
            let analysis = coordinator.strategic_analysis().await;
            let report = coordinator.generate_performance_report().await;
            
            black_box((coordinator, analysis, report))
        })
    });
}

criterion_group!(
    coordinator_benches,
    bench_coordinator_creation,
    bench_agent_registration,
    bench_task_submission,
    bench_task_completion,
    bench_strategic_analysis,
    bench_performance_report,
    bench_concurrent_operations,
    bench_memory_scalability,
    bench_performance_tracking,
    bench_end_to_end_workflow
);

criterion_main!(coordinator_benches);