//! Message Passing Benchmarks
//! 
//! Measures the performance of inter-agent communication
//! including throughput, latency, and different message patterns.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ruv_swarm::{
    agent::{AgentType, AgentId},
    message::{Message, MessageType},
    swarm::{Swarm, SwarmConfig},
    topology::Topology,
};
use tokio::runtime::Runtime;
use std::time::Duration;

fn unicast_message_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("unicast_latency");
    
    group.bench_function("single_hop", |b| {
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                let sender = swarm.spawn_agent(AgentType::Worker).await.unwrap();
                let receiver = swarm.spawn_agent(AgentType::Worker).await.unwrap();
                
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    swarm.send_message(
                        sender,
                        receiver,
                        black_box(Message::new(MessageType::Task, "test"))
                    ).await.unwrap();
                }
                start.elapsed()
            })
        });
    });
    
    group.finish();
}

fn broadcast_message_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("broadcast");
    
    for agent_count in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*agent_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            agent_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            max_agents: count + 10,
                            ..Default::default()
                        }).unwrap();
                        
                        let sender = swarm.spawn_agent(AgentType::Coordinator).await.unwrap();
                        let mut receivers = Vec::new();
                        for _ in 0..count {
                            receivers.push(swarm.spawn_agent(AgentType::Worker).await.unwrap());
                        }
                        
                        swarm.broadcast_message(
                            sender,
                            black_box(Message::new(MessageType::Broadcast, "announcement"))
                        ).await.unwrap();
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn message_routing_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("routing_overhead");
    
    let topologies = vec![
        ("FullyConnected", Topology::FullyConnected),
        ("Ring", Topology::Ring),
        ("Star", Topology::Star),
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
                            max_agents: 50,
                            ..Default::default()
                        }).unwrap();
                        
                        // Create network
                        let mut agents = Vec::new();
                        for _ in 0..20 {
                            agents.push(swarm.spawn_agent(AgentType::Worker).await.unwrap());
                        }
                        
                        // Send message across network
                        swarm.route_message(
                            agents[0],
                            agents[agents.len() - 1],
                            black_box(Message::new(MessageType::Task, "routed"))
                        ).await.unwrap();
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn concurrent_message_handling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_messages");
    
    for concurrency in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*concurrency as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            max_agents: concurrency * 2 + 10,
                            ..Default::default()
                        }).unwrap();
                        
                        let receiver = swarm.spawn_agent(AgentType::Coordinator).await.unwrap();
                        let mut senders = Vec::new();
                        
                        for _ in 0..concurrency {
                            senders.push(swarm.spawn_agent(AgentType::Worker).await.unwrap());
                        }
                        
                        // Send concurrent messages
                        let mut handles = Vec::new();
                        for sender in senders {
                            let msg = Message::new(MessageType::Task, "concurrent");
                            handles.push(swarm.send_message_async(sender, receiver, msg));
                        }
                        
                        // Wait for all messages
                        for handle in handles {
                            handle.await.unwrap();
                        }
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn message_size_impact(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("message_size");
    
    let sizes = vec![
        ("Small", 64),
        ("Medium", 1024),
        ("Large", 16384),
        ("XLarge", 262144),
    ];
    
    for (name, size) in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                        let sender = swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        let receiver = swarm.spawn_agent(AgentType::Worker).await.unwrap();
                        
                        let payload = vec![0u8; size];
                        swarm.send_message(
                            sender,
                            receiver,
                            black_box(Message::new_with_data(MessageType::Data, payload))
                        ).await.unwrap();
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn message_queue_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("message_queue");
    
    group.bench_function("enqueue_rate", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                let agent = swarm.spawn_agent(AgentType::Worker).await.unwrap();
                
                for i in 0..1000 {
                    swarm.enqueue_message(
                        agent,
                        black_box(Message::new(MessageType::Task, format!("msg_{}", i)))
                    );
                }
            });
        });
    });
    
    group.bench_function("dequeue_rate", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
                let agent = swarm.spawn_agent(AgentType::Worker).await.unwrap();
                
                // Pre-fill queue
                for i in 0..1000 {
                    swarm.enqueue_message(
                        agent,
                        Message::new(MessageType::Task, format!("msg_{}", i))
                    );
                }
                
                // Measure dequeue
                while let Some(msg) = swarm.dequeue_message(agent).await {
                    black_box(msg);
                }
            });
        });
    });
    
    group.finish();
}

fn gossip_protocol_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("gossip");
    
    for network_size in [20, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(network_size),
            network_size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut swarm = Swarm::new(SwarmConfig {
                            max_agents: size + 10,
                            topology: Topology::SmallWorld,
                            ..Default::default()
                        }).unwrap();
                        
                        // Create network
                        let mut agents = Vec::new();
                        for _ in 0..size {
                            agents.push(swarm.spawn_agent(AgentType::Worker).await.unwrap());
                        }
                        
                        // Start gossip from random node
                        swarm.gossip_message(
                            agents[0],
                            black_box(Message::new(MessageType::Gossip, "rumor")),
                            3 // fanout
                        ).await.unwrap();
                        
                        // Wait for propagation
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    });
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    unicast_message_latency,
    broadcast_message_performance,
    message_routing_overhead,
    concurrent_message_handling,
    message_size_impact,
    message_queue_performance,
    gossip_protocol_performance
);

criterion_main!(benches);