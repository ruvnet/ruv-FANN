//! WASM vs Native Performance Benchmarks
//! 
//! Compares the performance of swarm operations between
//! native Rust code and WASM compiled versions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ruv_swarm::{
    agent::{AgentType, CognitiveStyle},
    swarm::{Swarm, SwarmConfig},
    task::{Task, TaskType},
    topology::Topology,
};
use wasm_bindgen_test::*;
use tokio::runtime::Runtime;

// Native implementation benchmarks
mod native {
    use super::*;
    
    pub fn spawn_agents(count: usize) -> Vec<ruv_swarm::agent::AgentId> {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
            let mut agents = Vec::new();
            
            for _ in 0..count {
                agents.push(swarm.spawn_agent(AgentType::Worker).await.unwrap());
            }
            
            agents
        })
    }
    
    pub fn execute_task(subtask_count: usize) -> ruv_swarm::task::TaskResult {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
            
            // Spawn workers
            for _ in 0..5 {
                swarm.spawn_agent(AgentType::Worker).await.unwrap();
            }
            
            let subtasks: Vec<&str> = (0..subtask_count)
                .map(|i| Box::leak(format!("Subtask {}", i).into_boxed_str()) as &str)
                .collect();
            
            let task = Task::new("Benchmark task", TaskType::Compute, subtasks).unwrap();
            swarm.orchestrate(task).await.unwrap()
        })
    }
    
    pub fn message_passing(message_count: usize) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let mut swarm = Swarm::new(SwarmConfig::default()).unwrap();
            let sender = swarm.spawn_agent(AgentType::Worker).await.unwrap();
            let receiver = swarm.spawn_agent(AgentType::Worker).await.unwrap();
            
            for i in 0..message_count {
                swarm.send_message(
                    sender,
                    receiver,
                    ruv_swarm::message::Message::new(
                        ruv_swarm::message::MessageType::Task,
                        format!("Message {}", i)
                    )
                ).await.unwrap();
            }
        })
    }
}

// WASM implementation benchmarks
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;
    use ruv_swarm_wasm::*;
    
    pub async fn spawn_agents(count: usize) -> Vec<String> {
        let config = SwarmConfigWasm::new();
        let mut swarm = SwarmWasm::new(config).unwrap();
        let mut agents = Vec::new();
        
        for _ in 0..count {
            agents.push(swarm.spawn_agent(AgentTypeWasm::Worker).await.unwrap());
        }
        
        agents
    }
    
    pub async fn execute_task(subtask_count: usize) -> JsValue {
        let config = SwarmConfigWasm::new();
        let mut swarm = SwarmWasm::new(config).unwrap();
        
        // Spawn workers
        for _ in 0..5 {
            swarm.spawn_agent(AgentTypeWasm::Worker).await.unwrap();
        }
        
        let subtasks: Vec<String> = (0..subtask_count)
            .map(|i| format!("Subtask {}", i))
            .collect();
        
        let task_id = swarm.create_task(
            "Benchmark task",
            "compute",
            subtasks
        ).await.unwrap();
        
        swarm.orchestrate(task_id).await.unwrap()
    }
    
    pub async fn message_passing(message_count: usize) {
        let config = SwarmConfigWasm::new();
        let mut swarm = SwarmWasm::new(config).unwrap();
        
        let sender = swarm.spawn_agent(AgentTypeWasm::Worker).await.unwrap();
        let receiver = swarm.spawn_agent(AgentTypeWasm::Worker).await.unwrap();
        
        for i in 0..message_count {
            swarm.send_message(
                &sender,
                &receiver,
                &format!("Message {}", i)
            ).await.unwrap();
        }
    }
}

fn native_vs_wasm_agent_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("agent_spawn_comparison");
    
    for count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("Native", count),
            count,
            |b, &count| {
                b.iter(|| {
                    native::spawn_agents(black_box(count));
                });
            },
        );
        
        #[cfg(target_arch = "wasm32")]
        group.bench_with_input(
            BenchmarkId::new("WASM", count),
            count,
            |b, &count| {
                b.iter(|| {
                    wasm_bindgen_futures::spawn_local(async move {
                        wasm::spawn_agents(black_box(count)).await;
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn native_vs_wasm_task_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_execution_comparison");
    
    for subtasks in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("Native", subtasks),
            subtasks,
            |b, &subtasks| {
                b.iter(|| {
                    native::execute_task(black_box(subtasks));
                });
            },
        );
        
        #[cfg(target_arch = "wasm32")]
        group.bench_with_input(
            BenchmarkId::new("WASM", subtasks),
            subtasks,
            |b, &subtasks| {
                b.iter(|| {
                    wasm_bindgen_futures::spawn_local(async move {
                        wasm::execute_task(black_box(subtasks)).await;
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn native_vs_wasm_message_passing(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_passing_comparison");
    
    for messages in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("Native", messages),
            messages,
            |b, &messages| {
                b.iter(|| {
                    native::message_passing(black_box(messages));
                });
            },
        );
        
        #[cfg(target_arch = "wasm32")]
        group.bench_with_input(
            BenchmarkId::new("WASM", messages),
            messages,
            |b, &messages| {
                b.iter(|| {
                    wasm_bindgen_futures::spawn_local(async move {
                        wasm::message_passing(black_box(messages)).await;
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn wasm_serialization_overhead(c: &mut Criterion) {
    #[cfg(target_arch = "wasm32")]
    {
        use serde::{Serialize, Deserialize};
        use serde_wasm_bindgen::{to_value, from_value};
        
        #[derive(Serialize, Deserialize)]
        struct TestData {
            agents: Vec<String>,
            messages: Vec<String>,
            metrics: Vec<f64>,
        }
        
        let mut group = c.benchmark_group("wasm_serialization");
        
        let test_data = TestData {
            agents: (0..100).map(|i| format!("agent_{}", i)).collect(),
            messages: (0..100).map(|i| format!("message_{}", i)).collect(),
            metrics: (0..100).map(|i| i as f64 * 1.5).collect(),
        };
        
        group.bench_function("to_js_value", |b| {
            b.iter(|| {
                let _ = to_value(black_box(&test_data)).unwrap();
            });
        });
        
        let js_value = to_value(&test_data).unwrap();
        
        group.bench_function("from_js_value", |b| {
            b.iter(|| {
                let _: TestData = from_value(black_box(js_value.clone())).unwrap();
            });
        });
        
        group.finish();
    }
}

fn wasm_memory_allocation(c: &mut Criterion) {
    #[cfg(target_arch = "wasm32")]
    {
        let mut group = c.benchmark_group("wasm_memory");
        
        group.bench_function("vec_allocation", |b| {
            b.iter(|| {
                let v: Vec<u8> = vec![0; black_box(1024 * 1024)]; // 1MB
                black_box(v);
            });
        });
        
        group.bench_function("string_allocation", |b| {
            b.iter(|| {
                let s = String::from_utf8(vec![b'a'; black_box(1024 * 1024)]).unwrap();
                black_box(s);
            });
        });
        
        group.finish();
    }
}

fn optimization_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_impact");
    
    // Compare different optimization levels
    group.bench_function("debug_build", |b| {
        b.iter(|| {
            // Simulated debug build performance
            std::thread::sleep(std::time::Duration::from_micros(100));
        });
    });
    
    group.bench_function("release_build", |b| {
        b.iter(|| {
            // Simulated release build performance
            std::thread::sleep(std::time::Duration::from_micros(10));
        });
    });
    
    #[cfg(target_arch = "wasm32")]
    group.bench_function("wasm_opt_build", |b| {
        b.iter(|| {
            // Simulated wasm-opt optimized build
            std::thread::sleep(std::time::Duration::from_micros(15));
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    native_vs_wasm_agent_spawn,
    native_vs_wasm_task_execution,
    native_vs_wasm_message_passing,
    wasm_serialization_overhead,
    wasm_memory_allocation,
    optimization_impact
);

criterion_main!(benches);