// Core swarm orchestration WASM interfaces
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use ruv_swarm_core::{Agent, Swarm, Task, Topology};

mod agent;
mod swarm;
mod task;
mod topology;

pub use agent::*;
pub use swarm::*;
pub use task::*;
pub use topology::*;

// Performance monitoring for WASM
#[wasm_bindgen]
pub struct SwarmPerformance {
    agent_count: u32,
    task_count: u32,
    memory_usage: u32,
    execution_time_ms: f64,
}

#[wasm_bindgen]
impl SwarmPerformance {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SwarmPerformance {
        SwarmPerformance {
            agent_count: 0,
            task_count: 0,
            memory_usage: 0,
            execution_time_ms: 0.0,
        }
    }

    #[wasm_bindgen]
    pub fn update(&mut self, agents: u32, tasks: u32) {
        self.agent_count = agents;
        self.task_count = tasks;
        self.memory_usage = crate::utils::get_current_memory_usage();
    }

    #[wasm_bindgen]
    pub fn get_metrics(&self) -> JsValue {
        let metrics = serde_json::json!({
            "agent_count": self.agent_count,
            "task_count": self.task_count,
            "memory_usage_mb": (self.memory_usage as f64) / (1024.0 * 1024.0),
            "execution_time_ms": self.execution_time_ms,
            "agents_per_mb": (self.agent_count as f64) / ((self.memory_usage as f64) / (1024.0 * 1024.0))
        });
        
        serde_wasm_bindgen::to_value(&metrics).unwrap()
    }
}