//! Task Orchestration Benchmarks
//! 
//! Measures the performance of task distribution, scheduling,
//! and completion across different swarm configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ruv_swarm::{
    agent::{AgentType, CognitiveStyle},
    swarm::{Swarm, SwarmConfig},
    task::{Task, TaskType, TaskPriority},
    topology::Topology,
};
use tokio::runtime::Runtime;
use std::time::Duration;

fn simple_task_orchestration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("orchestrate_simple_task", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                
                // Spawn workers
                for _ in 0..5 {
                    swarm.spawn_agent(AgentType::Worker).await.unwrap();
                }
                
                let task = Task::new(
                    "Simple computation",
                    TaskType::Compute,
                    vec!["Step 1", "Step 2", "Step 3"]
                ).unwrap();
                
                let result = swarm.orchestrate(black_box(task)).await.unwrap();
                black_box(result);
            });
        });
    });
}

fn parallel_task_distribution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("parallel_distribution");
    
    for worker_count in [5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*worker_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(worker_count),
            worker_count,
            |b, &workers| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            max_agents: workers + 10,
                            ..Default::default()
                        }).unwrap();
                        
                        // Spawn workers
                        for _ in 0..workers {
                            swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        }
                        
                        // Create task with many subtasks
                        let subtasks: Vec<&str> = (0..workers)
                            .map(|i| Box::leak(format!("Subtask {}", i).into_boxed_str()) as &str)
                            .collect();
                        
                        let task = Task::new(
                            "Parallel workload",
                            TaskType::Compute,
                            subtasks
                        ).unwrap();
                        
                        let result = swarm.orchestrate(task).await.unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn task_scheduling_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("scheduling");
    
    let priorities = vec![
        ("High", TaskPriority::High),
        ("Normal", TaskPriority::Normal),
        ("Low", TaskPriority::Low),
    ];
    
    for (name, priority) in priorities {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &priority,
            |b, priority| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                        
                        // Spawn workers
                        for _ in 0..10 {
                            swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        }
                        
                        // Create and schedule multiple tasks
                        for i in 0..20 {
                            let mut task = Task::new(
                                format!("Task {}", i),
                                TaskType::Compute,
                                vec!["Process"]
                            ).unwrap();
                            task.set_priority(*priority);
                            
                            swarm.schedule_task(black_box(task)).await.unwrap();
                        }
                        
                        // Execute all scheduled tasks
                        swarm.execute_scheduled().await.unwrap();
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn cognitive_task_routing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("cognitive_routing");
    
    let task_types = vec![
        ("Analysis", TaskType::Analysis),
        ("Creative", TaskType::Creative),
        ("Implementation", TaskType::Implementation),
    ];
    
    for (name, task_type) in task_types {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &task_type,
            |b, task_type| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            cognitive_diversity: true,
                            ..Default::default()
                        }).unwrap();
                        
                        // Spawn diverse agents
                        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Analytical).await.unwrap();
                        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Creative).await.unwrap();
                        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Strategic).await.unwrap();
                        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Practical).await.unwrap();
                        
                        let task = Task::new(
                            format!("{:?} task", task_type),
                            *task_type,
                            vec!["Step 1", "Step 2", "Step 3", "Step 4"]
                        ).unwrap();
                        
                        let result = swarm.orchestrate_with_diversity(black_box(task)).await.unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn hierarchical_orchestration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hierarchical");
    
    for depth in [2, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("depth_{}", depth)),
            depth,
            |b, &depth| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            topology: Topology::HierarchicalRing,
                            ..Default::default()
                        }).unwrap();
                        
                        // Create hierarchy
                        let coordinator = swarm.spawn_agent(AgentType::Coordinator).await.unwrap();
                        
                        let mut managers = Vec::new();
                        for _ in 0..3 {
                            managers.push(swarm.spawn_agent(AgentType::Coordinator).await.unwrap());
                        }
                        
                        // Workers under each manager
                        for _ in 0..(3 * depth) {
                            swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        }
                        
                        // Complex hierarchical task
                        let subtasks: Vec<&str> = (0..(3 * depth))
                            .map(|i| Box::leak(format!("Subtask {}", i).into_boxed_str()) as &str)
                            .collect();
                        
                        let task = Task::new(
                            "Hierarchical processing",
                            TaskType::Compute,
                            subtasks
                        ).unwrap();
                        
                        let result = swarm.orchestrate_hierarchical(task, coordinator).await.unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn task_recovery_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("recovery");
    
    group.bench_function("failure_detection", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig {
                    heartbeat_interval: Duration::from_millis(100),
                    ..Default::default()
                }).unwrap();
                
                // Spawn workers
                let mut workers = Vec::new();
                for _ in 0..10 {
                    workers.push(swarm.spawn_agent(AgentType::Worker).await.unwrap());
                }
                
                // Simulate failure
                swarm.simulate_agent_failure(workers[5]).await;
                
                // Measure detection and recovery
                let task = Task::new(
                    "Resilient task",
                    TaskType::Compute,
                    vec!["Step 1", "Step 2", "Step 3"]
                ).unwrap();
                
                let result = swarm.orchestrate_with_recovery(black_box(task)).await.unwrap();
                black_box(result);
            });
        });
    });
    
    group.finish();
}

fn load_balancing_effectiveness(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("load_balancing");
    
    let strategies = vec![
        ("RoundRobin", "round_robin"),
        ("LeastLoaded", "least_loaded"),
        ("Random", "random"),
        ("Adaptive", "adaptive"),
    ];
    
    for (name, strategy) in strategies {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &strategy,
            |b, strategy| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            load_balancing_strategy: strategy.to_string(),
                            ..Default::default()
                        }).unwrap();
                        
                        // Spawn workers
                        for _ in 0..20 {
                            swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        }
                        
                        // Create many small tasks
                        for i in 0..100 {
                            let task = Task::new(
                                format!("Task {}", i),
                                TaskType::Compute,
                                vec!["Process"]
                            ).unwrap();
                            
                            swarm.submit_task(task).await.unwrap();
                        }
                        
                        // Process all tasks
                        swarm.process_all_tasks().await.unwrap();
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn task_pipeline_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("pipeline_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                
                // Create pipeline stages
                let stage1_workers = vec![
                    swarm.spawn_agent(AgentType::Worker).await.unwrap(),
                    swarm.spawn_agent(AgentType::Worker).await.unwrap(),
                ];
                
                let stage2_workers = vec![
                    swarm.spawn_agent(AgentType::Worker).await.unwrap(),
                    swarm.spawn_agent(AgentType::Worker).await.unwrap(),
                ];
                
                let stage3_workers = vec![
                    swarm.spawn_agent(AgentType::Worker).await.unwrap(),
                    swarm.spawn_agent(AgentType::Worker).await.unwrap(),
                ];
                
                // Create pipeline task
                let pipeline = swarm.create_pipeline()
                    .add_stage("Extract", stage1_workers)
                    .add_stage("Transform", stage2_workers)
                    .add_stage("Load", stage3_workers)
                    .build();
                
                // Execute pipeline with data
                let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
                let result = swarm.execute_pipeline(pipeline, black_box(data)).await.unwrap();
                black_box(result);
            });
        });
    });
}

criterion_group!(
    benches,
    simple_task_orchestration,
    parallel_task_distribution,
    task_scheduling_overhead,
    cognitive_task_routing,
    hierarchical_orchestration,
    task_recovery_performance,
    load_balancing_effectiveness,
    task_pipeline_performance
);

criterion_main!(benches);