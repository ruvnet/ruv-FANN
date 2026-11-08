use wasm_bindgen::prelude::*;

mod forecasting;
mod memory_pool;
mod network;
mod orchestration;
mod simd_ops;
mod simd_tests;
mod utils;

pub use memory_pool::MemoryPool;
pub use simd_ops::{detect_simd_capabilities, SimdBenchmark, SimdMatrixOps, SimdVectorOps};
pub use simd_tests::{
    run_simd_verification_suite, simd_performance_report, validate_simd_implementation,
};
pub use utils::{set_panic_hook, RuntimeFeatures};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn init() {
    set_panic_hook();
}

pub use network::{ActivationFunction, WasmNeuralNetwork};

pub use orchestration::{WasmAgent, WasmSwarmOrchestrator, WasmTaskResult};

pub use forecasting::WasmForecastingModel;

// Export functions for use from JavaScript
#[wasm_bindgen]
pub fn create_neural_network(
    layers: &[usize],
    activation: ActivationFunction,
) -> Result<WasmNeuralNetwork, JsValue> {
    WasmNeuralNetwork::new(layers, activation)
}

#[wasm_bindgen]
pub fn create_swarm_orchestrator(topology: &str) -> Result<WasmSwarmOrchestrator, JsValue> {
    WasmSwarmOrchestrator::new(topology)
}

#[wasm_bindgen]
pub fn create_forecasting_model(model_type: &str) -> Result<WasmForecastingModel, JsValue> {
    WasmForecastingModel::new(model_type)
}

#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn get_features() -> String {
    let simd_support = detect_simd_support();
    serde_json::json!({
        "neural_networks": true,
        "forecasting": true,
        "swarm_orchestration": true,
        "cognitive_diversity": false,
        "forecast_baselines": ["linear", "mean", "naive"],
        "distributed_execution": false,
        "simd_support": simd_support,
        "simd_capabilities": detect_simd_capabilities()
    })
    .to_string()
}

/// Runtime SIMD support detection - fixed to properly detect SIMD compilation
fn detect_simd_support() -> bool {
    // For WebAssembly builds
    #[cfg(target_arch = "wasm32")]
    {
        cfg!(target_feature = "simd128")
    }

    // For non-WASM platforms, use target features
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(any(
            target_feature = "sse",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "neon"
        ))]
        {
            true
        }
        #[cfg(not(any(
            target_feature = "sse",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "neon"
        )))]
        {
            false
        }
    }
}

/// Test SIMD functionality at runtime by attempting a simple operation
fn test_simd_runtime() -> bool {
    // Try to use the SIMD operations and catch any panics
    let test_vec_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let test_vec_b = vec![2.0f32, 3.0, 4.0, 5.0];

    // Use std::panic::catch_unwind to safely test SIMD operations
    let result = std::panic::catch_unwind(|| {
        let ops = crate::simd_ops::SimdVectorOps::new();
        ops.dot_product(&test_vec_a, &test_vec_b)
    });

    match result {
        Ok(value) => {
            // Check if the result is approximately correct (should be 40.0)
            (value - 40.0).abs() < 0.1
        }
        Err(_) => false,
    }
}

// Performance monitoring for optimization targets
#[wasm_bindgen]
pub struct PerformanceMonitor {
    load_time: f64,
    spawn_times: Vec<f64>,
    memory_usage: usize,
    simd_enabled: bool,
}

#[wasm_bindgen]
impl PerformanceMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            load_time: 0.0,
            spawn_times: Vec::new(),
            memory_usage: 0,
            simd_enabled: detect_simd_support(),
        }
    }

    pub fn record_load_time(&mut self, time: f64) {
        self.load_time = time;
    }

    pub fn record_spawn_time(&mut self, time: f64) {
        self.spawn_times.push(time);
    }

    pub fn update_memory_usage(&mut self, bytes: usize) {
        self.memory_usage = bytes;
    }

    pub fn get_average_spawn_time(&self) -> f64 {
        if self.spawn_times.is_empty() {
            0.0
        } else {
            self.spawn_times.iter().sum::<f64>() / self.spawn_times.len() as f64
        }
    }

    pub fn get_memory_usage_mb(&self) -> f64 {
        (self.memory_usage as f64) / (1024.0 * 1024.0)
    }

    pub fn meets_performance_targets(&self) -> bool {
        self.load_time < 500.0
            && self.get_average_spawn_time() < 100.0
            && self.get_memory_usage_mb() < 50.0
    }

    pub fn get_report(&self) -> String {
        format!(
            "Performance Report:\n\
             - Load Time: {:.2}ms (Target: <500ms) {}\n\
             - Avg Spawn Time: {:.2}ms (Target: <100ms) {}\n\
             - Memory Usage: {:.2}MB (Target: <50MB) {}\n\
             - SIMD Enabled: {}\n\
             - All Targets Met: {}",
            self.load_time,
            if self.load_time < 500.0 { "✓" } else { "✗" },
            self.get_average_spawn_time(),
            if self.get_average_spawn_time() < 100.0 {
                "✓"
            } else {
                "✗"
            },
            self.get_memory_usage_mb(),
            if self.get_memory_usage_mb() < 50.0 {
                "✓"
            } else {
                "✗"
            },
            self.simd_enabled,
            self.meets_performance_targets()
        )
    }
}

// Optimized agent spawning with memory pooling
#[wasm_bindgen]
pub struct OptimizedAgentSpawner {
    memory_pool: memory_pool::AgentMemoryPool,
    performance_monitor: PerformanceMonitor,
    active_agents: Vec<OptimizedAgent>,
}

#[wasm_bindgen]
pub struct OptimizedAgent {
    id: String,
    agent_type: String,
    memory_size: usize,
    #[wasm_bindgen(skip)]
    memory: Vec<u8>,
}

#[wasm_bindgen]
impl OptimizedAgentSpawner {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            memory_pool: memory_pool::AgentMemoryPool::new(),
            performance_monitor: PerformanceMonitor::new(),
            active_agents: Vec::new(),
        }
    }

    pub fn spawn_agent(&mut self, agent_type: &str, complexity: &str) -> Result<String, JsValue> {
        let start = js_sys::Date::now();

        // Allocate memory from pool
        let memory = self
            .memory_pool
            .allocate_for_agent(complexity)
            .ok_or_else(|| JsValue::from_str("Memory allocation failed"))?;

        let memory_size = memory.len();
        let agent_id = format!("agent-{}-{}", agent_type, js_sys::Math::random());

        let agent = OptimizedAgent {
            id: agent_id.clone(),
            agent_type: agent_type.to_string(),
            memory_size,
            memory,
        };

        self.active_agents.push(agent);

        // Record metrics
        let spawn_time = js_sys::Date::now() - start;
        self.performance_monitor.record_spawn_time(spawn_time);
        self.performance_monitor
            .update_memory_usage(self.memory_pool.total_memory_usage_mb() as usize * 1024 * 1024);

        Ok(agent_id)
    }

    pub fn release_agent(&mut self, agent_id: &str) -> Result<(), JsValue> {
        if let Some(pos) = self.active_agents.iter().position(|a| a.id == agent_id) {
            let agent = self.active_agents.remove(pos);
            self.memory_pool.deallocate_agent_memory(agent.memory);
            self.performance_monitor.update_memory_usage(
                self.memory_pool.total_memory_usage_mb() as usize * 1024 * 1024,
            );
            Ok(())
        } else {
            Err(JsValue::from_str("Agent not found"))
        }
    }

    pub fn get_performance_report(&self) -> String {
        self.performance_monitor.get_report()
    }

    pub fn get_active_agent_count(&self) -> usize {
        self.active_agents.len()
    }

    pub fn is_within_memory_target(&self) -> bool {
        self.memory_pool.is_within_memory_target()
    }
}
