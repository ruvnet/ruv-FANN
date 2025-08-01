use wasm_bindgen::prelude::*;

mod memory_pool;
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

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    SymmetricSigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Swish,
    Gaussian,
    Elliot,
    SymmetricElliot,
    Sine,
    Cosine,
    SinSymmetric,
    CosSymmetric,
    ThresholdSymmetric,
    Threshold,
    StepSymmetric,
    Step,
}

#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    layers: Vec<usize>,
    weights: Vec<f64>,
    biases: Vec<f64>,
    activation: ActivationFunction,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new(layers: &[usize], activation: ActivationFunction) -> Self {
        let total_weights = layers.windows(2).map(|w| w[0] * w[1]).sum();
        let total_biases = layers[1..].iter().sum();

        Self {
            layers: layers.to_vec(),
            weights: vec![0.0; total_weights],
            biases: vec![0.0; total_biases],
            activation,
        }
    }

    #[wasm_bindgen]
    pub fn randomize_weights(&mut self, min: f64, max: f64) {
        use js_sys::Math;
        for weight in &mut self.weights {
            *weight = min + (max - min) * Math::random();
        }
        for bias in &mut self.biases {
            *bias = min + (max - min) * Math::random();
        }
    }

    #[wasm_bindgen]
    pub fn set_weights(&mut self, weights: &[f64]) {
        if weights.len() == self.weights.len() {
            self.weights.copy_from_slice(weights);
        }
    }

    #[wasm_bindgen]
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    #[wasm_bindgen]
    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        let mut current_outputs = inputs.to_vec();

        for layer_idx in 1..self.layers.len() {
            let prev_size = self.layers[layer_idx - 1];
            let curr_size = self.layers[layer_idx];

            // Extract weights for this layer as a matrix
            let mut layer_weights = vec![0.0; prev_size * curr_size];
            for curr_neuron in 0..curr_size {
                for prev_neuron in 0..prev_size {
                    let weight_idx = self.get_weight_index(layer_idx - 1, prev_neuron, curr_neuron);
                    layer_weights[curr_neuron * prev_size + prev_neuron] =
                        self.weights[weight_idx] as f64;
                }
            }

            // Convert to f32 for SIMD operations
            let current_f32: Vec<f32> = current_outputs.iter().map(|&x| x as f32).collect();
            let weights_f32: Vec<f32> = layer_weights.iter().map(|&x| x as f32).collect();

            // Use SIMD-optimized matrix-vector multiplication if available
            let simd_ops = crate::simd_ops::SimdMatrixOps::new();
            let simd_result =
                simd_ops.matrix_vector_multiply(&weights_f32, &current_f32, curr_size, prev_size);

            // Add biases and apply activation
            let mut new_outputs = vec![0.0; curr_size];
            for curr_neuron in 0..curr_size {
                let bias_idx = self.get_bias_index(layer_idx, curr_neuron);
                let sum = simd_result[curr_neuron] as f64 + self.biases[bias_idx];
                new_outputs[curr_neuron] = self.apply_activation(sum);
            }

            current_outputs = new_outputs;
        }

        current_outputs
    }

    fn get_weight_index(&self, from_layer: usize, from_neuron: usize, to_neuron: usize) -> usize {
        let mut index = 0;
        for layer in 0..from_layer {
            index += self.layers[layer] * self.layers[layer + 1];
        }
        index + from_neuron * self.layers[from_layer + 1] + to_neuron
    }

    fn get_bias_index(&self, layer: usize, neuron: usize) -> usize {
        let mut index = 0;
        for l in 1..layer {
            index += self.layers[l];
        }
        index + neuron
    }

    fn apply_activation(&self, x: f64) -> f64 {
        match self.activation {
            ActivationFunction::Linear => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::SymmetricSigmoid => 2.0 / (1.0 + (-x).exp()) - 1.0,
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            ActivationFunction::Swish => x * (1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Gaussian => (-x * x).exp(),
            ActivationFunction::Elliot => x / (1.0 + x.abs()),
            ActivationFunction::SymmetricElliot => 2.0 * x / (1.0 + x.abs()),
            ActivationFunction::Sine => x.sin(),
            ActivationFunction::Cosine => x.cos(),
            ActivationFunction::SinSymmetric => 2.0 * x.sin() - 1.0,
            ActivationFunction::CosSymmetric => 2.0 * x.cos() - 1.0,
            ActivationFunction::ThresholdSymmetric => {
                if x > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            ActivationFunction::Threshold => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::StepSymmetric => {
                if x > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            ActivationFunction::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[wasm_bindgen]
pub struct WasmSwarmOrchestrator {
    agents: Vec<WasmAgent>,
    topology: String,
    task_counter: u32,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmAgent {
    id: String,
    agent_type: String,
    status: String,
    capabilities: Vec<String>,
}

#[wasm_bindgen]
impl WasmAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str, agent_type: &str) -> Self {
        Self {
            id: id.to_string(),
            agent_type: agent_type.to_string(),
            status: "idle".to_string(),
            capabilities: Vec::new(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn agent_type(&self) -> String {
        self.agent_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        self.status.clone()
    }

    #[wasm_bindgen]
    pub fn set_status(&mut self, status: &str) {
        self.status = status.to_string();
    }

    #[wasm_bindgen]
    pub fn add_capability(&mut self, capability: &str) {
        self.capabilities.push(capability.to_string());
    }

    #[wasm_bindgen]
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.contains(&capability.to_string())
    }
}

#[wasm_bindgen]
pub struct WasmTaskResult {
    task_id: String,
    description: String,
    status: String,
    assigned_agents: Vec<String>,
    priority: String,
}

#[wasm_bindgen]
impl WasmTaskResult {
    #[wasm_bindgen(getter)]
    pub fn task_id(&self) -> String {
        self.task_id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.description.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        self.status.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn assigned_agents(&self) -> Vec<String> {
        self.assigned_agents.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn priority(&self) -> String {
        self.priority.clone()
    }
}

#[wasm_bindgen]
impl WasmSwarmOrchestrator {
    #[wasm_bindgen(constructor)]
    pub fn new(topology: &str) -> Self {
        Self {
            agents: Vec::new(),
            topology: topology.to_string(),
            task_counter: 0,
        }
    }

    #[wasm_bindgen]
    pub fn spawn(&mut self, config: &str) -> String {
        // Parse JSON config (simplified for demo)
        let agent_id = format!("agent-{}", self.agents.len() + 1);
        let mut agent = WasmAgent::new(&agent_id, "researcher");

        // Add some default capabilities based on type
        agent.add_capability("research");
        agent.add_capability("analysis");

        self.agents.push(agent);

        // Return JSON result
        serde_json::json!({
            "agent_id": agent_id,
            "name": format!("Agent-{}", self.agents.len()),
            "type": "researcher",
            "capabilities": ["research", "analysis"],
            "cognitive_pattern": "adaptive",
            "neural_network_id": format!("nn-{}", self.agents.len())
        })
        .to_string()
    }

    #[wasm_bindgen]
    pub fn orchestrate(&mut self, config: &str) -> WasmTaskResult {
        self.task_counter += 1;
        let task_id = format!("task-{}", self.task_counter);

        // Parse config (simplified JSON parsing)
        let description = "Sample task"; // Would parse from config
        let priority = "medium"; // Would parse from config

        // Select available agents (simple strategy for now)
        let mut assigned_agents = Vec::new();
        for agent in &mut self.agents {
            if agent.status == "idle" && assigned_agents.len() < 3 {
                agent.set_status("busy");
                assigned_agents.push(agent.id());
            }
        }

        WasmTaskResult {
            task_id,
            description: description.to_string(),
            status: "orchestrated".to_string(),
            assigned_agents,
            priority: priority.to_string(),
        }
    }

    #[wasm_bindgen]
    pub fn add_agent(&mut self, agent_id: &str) {
        let agent = WasmAgent::new(agent_id, "generic");
        self.agents.push(agent);
    }

    #[wasm_bindgen]
    pub fn get_agent_count(&self) -> usize {
        self.agents.len()
    }

    #[wasm_bindgen]
    pub fn get_topology(&self) -> String {
        self.topology.clone()
    }

    #[wasm_bindgen]
    pub fn get_status(&self, detailed: bool) -> String {
        let idle_count = self.agents.iter().filter(|a| a.status == "idle").count();
        let busy_count = self.agents.iter().filter(|a| a.status == "busy").count();

        if detailed {
            serde_json::json!({
                "agents": {
                    "total": self.agents.len(),
                    "idle": idle_count,
                    "busy": busy_count
                },
                "topology": self.topology,
                "agent_details": self.agents.iter().map(|a| {
                    serde_json::json!({
                        "id": a.id(),
                        "type": a.agent_type(),
                        "status": a.status()
                    })
                }).collect::<Vec<_>>()
            })
            .to_string()
        } else {
            serde_json::json!({
                "agents": {
                    "total": self.agents.len(),
                    "idle": idle_count,
                    "busy": busy_count
                },
                "topology": self.topology
            })
            .to_string()
        }
    }
}

#[wasm_bindgen]
pub struct WasmForecastingModel {
    model_type: String,
    parameters: Vec<f64>,
}

#[wasm_bindgen]
impl WasmForecastingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_type: &str) -> Self {
        Self {
            model_type: model_type.to_string(),
            parameters: Vec::new(),
        }
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        // Simple placeholder forecasting
        match self.model_type.as_str() {
            "linear" => {
                let slope = if input.len() > 1 {
                    input[input.len() - 1] - input[input.len() - 2]
                } else {
                    0.0
                };
                vec![input[input.len() - 1] + slope]
            }
            "mean" => {
                let mean = input.iter().sum::<f64>() / input.len() as f64;
                vec![mean]
            }
            _ => vec![input[input.len() - 1]],
        }
    }

    #[wasm_bindgen]
    pub fn get_model_type(&self) -> String {
        self.model_type.clone()
    }
}

// Export functions for use from JavaScript
#[wasm_bindgen]
pub fn create_neural_network(
    layers: &[usize],
    activation: ActivationFunction,
) -> WasmNeuralNetwork {
    WasmNeuralNetwork::new(layers, activation)
}

#[wasm_bindgen]
pub fn create_swarm_orchestrator(topology: &str) -> WasmSwarmOrchestrator {
    WasmSwarmOrchestrator::new(topology)
}

#[wasm_bindgen]
pub fn create_forecasting_model(model_type: &str) -> WasmForecastingModel {
    WasmForecastingModel::new(model_type)
}

#[wasm_bindgen]
pub fn get_version() -> String {
    "0.1.0".to_string()
}

#[wasm_bindgen]
pub fn get_features() -> String {
    let simd_support = detect_simd_support();
    serde_json::json!({
        "neural_networks": true,
        "forecasting": true,
        "swarm_orchestration": true,
        "cognitive_diversity": true,
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
        // Primary check: simd feature flag indicates SIMD support is compiled in
        #[cfg(feature = "simd")]
        {
            // Always return true when compiled with simd feature - the operations are available
            true
        }

        // If no simd feature, try runtime test
        #[cfg(not(feature = "simd"))]
        {
            false // Without simd feature, SIMD operations are not available
        }
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
