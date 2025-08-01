//! Agent Spawning Benchmarks
//! 
//! Measures the performance of agent creation and initialization
//! across different configurations and scales.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ruv_swarm::{
    agent::{AgentType, CognitiveStyle},
    swarm::{Swarm, SwarmConfig},
    topology::Topology,
};
use tokio::runtime::Runtime;

fn spawn_single_agent(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("spawn_single_worker", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                let agent = swarm.spawn_agent(black_box(AgentType::Worker)).await.unwrap();
                black_box(agent);
            });
        });
    });
}

fn spawn_multiple_agents(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("spawn_multiple");
    
    for count in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            max_agents: count + 10,
                            ..Default::default()
                        }).unwrap();
                        
                        for _ in 0..count {
                            swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

fn spawn_with_cognitive_styles(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("spawn_cognitive");
    
    let styles = vec![
        CognitiveStyle::Analytical,
        CognitiveStyle::Creative,
        CognitiveStyle::Strategic,
        CognitiveStyle::Practical,
        CognitiveStyle::DetailOriented,
    ];
    
    for style in styles {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", style)),
            &style,
            |b, style| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                        let agent = swarm.spawn_agent_with_style(
                            AgentType::Worker,
                            black_box(*style)
                        ).await.unwrap();
                        black_box(agent);
                    });
                });
            },
        );
    }
    group.finish();
}

fn spawn_different_agent_types(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("spawn_types");
    
    let types = vec![
        ("Worker", AgentType::Worker),
        ("Coordinator", AgentType::Coordinator),
        ("Analyzer", AgentType::Analyzer),
    ];
    
    for (name, agent_type) in types {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &agent_type,
            |b, agent_type| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                        let agent = swarm.spawn_agent(black_box(*agent_type)).await.unwrap();
                        black_box(agent);
                    });
                });
            },
        );
    }
    group.finish();
}

fn spawn_with_different_topologies(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("spawn_topology");
    
    let topologies = vec![
        ("FullyConnected", Topology::FullyConnected),
        ("Ring", Topology::Ring),
        ("Star", Topology::Star),
        ("Mesh", Topology::Mesh),
        ("HierarchicalRing", Topology::HierarchicalRing),
        ("SmallWorld", Topology::SmallWorld),
    ];
    
    for (name, topology) in topologies {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &topology,
            |b, topology| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            topology: *topology,
                            ..Default::default()
                        }).unwrap();
                        
                        // Spawn 20 agents to see topology impact
                        for _ in 0..20 {
                            swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

fn spawn_batch_vs_sequential(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("spawn_batch");
    
    let agent_count = 100;
    
    group.bench_function("sequential", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig {
                    max_agents: agent_count + 10,
                    ..Default::default()
                }).unwrap();
                
                for _ in 0..agent_count {
                    swarm.spawn_agent(AgentType::Worker).await.unwrap();
                }
            });
        });
    });
    
    group.bench_function("batch", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig {
                    max_agents: agent_count + 10,
                    ..Default::default()
                }).unwrap();
                
                swarm.spawn_agents_batch(
                    vec![AgentType::Worker; agent_count]
                ).await.unwrap();
            });
        });
    });
    
    group.finish();
}

fn agent_initialization_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("init_overhead");
    
    // Measure just the agent creation without swarm overhead
    group.bench_function("raw_agent_creation", |b| {
        b.iter(|| {
            rt.block_on(async {
                use ruv_swarm::agent::Agent;
                let agent = Agent::new(
                    AgentType::Worker,
                    CognitiveStyle::Analytical
                );
                black_box(agent);
            });
        });
    });
    
    // Measure with swarm integration
    group.bench_function("swarm_integrated", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                let agent = swarm.spawn_agent(AgentType::Worker).await.unwrap();
                black_box(agent);
            });
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    spawn_single_agent,
    spawn_multiple_agents,
    spawn_with_cognitive_styles,
    spawn_different_agent_types,
    spawn_with_different_topologies,
    spawn_batch_vs_sequential,
    agent_initialization_overhead
);

criterion_main!(benches);