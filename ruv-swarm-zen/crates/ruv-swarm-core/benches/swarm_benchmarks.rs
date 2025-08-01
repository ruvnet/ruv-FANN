use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ruv_swarm_core::{
    agent::DynamicAgent,
    task::{Task, TaskPriority},
};

#[cfg(feature = "std")]
use ruv_swarm_core::{
    swarm::{Swarm, SwarmConfig},
    task::DistributionStrategy,
};

fn task_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_creation");

    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    let _task = Task::new(format!("task_{}", i), "compute")
                        .with_priority(TaskPriority::Normal);
                }
            });
        });
    }

    group.finish();
}

#[cfg(feature = "std")]
fn swarm_agent_registration(c: &mut Criterion) {
    use tokio::runtime::Runtime;
    let rt = Runtime::new().unwrap();

    c.bench_function("agent_registration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = SwarmConfig::default();
                let mut swarm = Swarm::new(config);

                for i in 0..100 {
                    let agent =
                        DynamicAgent::new(format!("agent_{}", i), vec!["compute".to_string()]);
                    let _ = swarm.register_agent(agent);
                }
            });
        });
    });
}

#[cfg(feature = "std")]
fn task_distribution_strategies(c: &mut Criterion) {
    use tokio::runtime::Runtime;
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("task_distribution");

    let strategies = [
        ("round_robin", DistributionStrategy::RoundRobin),
        ("least_loaded", DistributionStrategy::LeastLoaded),
        ("capability_based", DistributionStrategy::CapabilityBased),
        ("priority_based", DistributionStrategy::Priority),
    ];

    for (name, strategy) in strategies.iter() {
        group.bench_function(*name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut config = SwarmConfig::default();
                    config.distribution_strategy = *strategy;
                    let mut swarm = Swarm::new(config);

                    // Register agents
                    for i in 0..10 {
                        let agent =
                            DynamicAgent::new(format!("agent_{}", i), vec!["compute".to_string()]);
                        let _ = swarm.register_agent(agent);
                    }

                    // Submit tasks
                    for i in 0..100 {
                        let task = Task::new(format!("task_{}", i), "compute")
                            .with_priority(TaskPriority::Normal)
                            .require_capability("compute");

                        let _ = swarm.submit_task(task);
                    }
                });
            });
        });
    }

    group.finish();
}

#[cfg(feature = "std")]
criterion_group!(
    benches,
    task_creation_benchmark,
    swarm_agent_registration,
    task_distribution_strategies
);

#[cfg(not(feature = "std"))]
criterion_group!(benches, task_creation_benchmark);

criterion_main!(benches);
