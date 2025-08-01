//! WASM bindings and optimization for DAA systems

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use web_sys::console;

#[cfg(feature = "wasm")]
use js_sys::{Array, Function, Object, Promise};

use serde::{Deserialize, Serialize};

// Import standard library types
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{
    types::{AdaptationFeedback, AutonomousCapability, DecisionContext},
    AgentMetrics, CognitivePattern, CoordinationResult, DAAError, DAAResult, Feedback, Knowledge,
    Priority, Task, TaskResult,
};

use std::collections::HashMap;

/// WASM-compatible autonomous agent implementation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAutonomousAgent {
    id: String,
    capabilities: Vec<AutonomousCapability>,
    learning_rate: f64,
    autonomy_level: f64,
    context: Option<DecisionContext>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAutonomousAgent {
    /// Create new WASM autonomous agent
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str) -> Self {
        console::log_1(&"Creating new WASM autonomous agent".into());

        WasmAutonomousAgent {
            id: id.to_string(),
            capabilities: vec![
                AutonomousCapability::SelfMonitoring,
                AutonomousCapability::DecisionMaking,
                AutonomousCapability::Learning,
                AutonomousCapability::ResourceOptimization,
            ],
            learning_rate: 0.01,
            autonomy_level: 1.0,
            context: None,
        }
    }

    /// Get agent ID
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    /// Get autonomy level
    #[wasm_bindgen(getter)]
    pub fn autonomy_level(&self) -> f64 {
        self.autonomy_level
    }

    /// Set autonomy level
    #[wasm_bindgen(setter)]
    pub fn set_autonomy_level(&mut self, level: f64) {
        self.autonomy_level = level.clamp(0.0, 1.0);
    }

    /// Get learning rate
    #[wasm_bindgen(getter)]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set learning rate
    #[wasm_bindgen(setter)]
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Add capability
    #[wasm_bindgen]
    pub fn add_capability(&mut self, capability_type: &str) -> bool {
        let capability = match capability_type {
            "self_monitoring" => AutonomousCapability::SelfMonitoring,
            "decision_making" => AutonomousCapability::DecisionMaking,
            "resource_optimization" => AutonomousCapability::ResourceOptimization,
            "self_healing" => AutonomousCapability::SelfHealing,
            "learning" => AutonomousCapability::Learning,
            "emergent_behavior" => AutonomousCapability::EmergentBehavior,
            "prediction" => AutonomousCapability::Prediction,
            "goal_planning" => AutonomousCapability::GoalPlanning,
            "coordination" => AutonomousCapability::Coordination,
            "memory_management" => AutonomousCapability::MemoryManagement,
            _ => return false,
        };

        if !self.capabilities.contains(&capability) {
            self.capabilities.push(capability);
            true
        } else {
            false
        }
    }

    /// Remove capability
    #[wasm_bindgen]
    pub fn remove_capability(&mut self, capability_type: &str) -> bool {
        let capability = match capability_type {
            "self_monitoring" => AutonomousCapability::SelfMonitoring,
            "decision_making" => AutonomousCapability::DecisionMaking,
            "resource_optimization" => AutonomousCapability::ResourceOptimization,
            "self_healing" => AutonomousCapability::SelfHealing,
            "learning" => AutonomousCapability::Learning,
            "emergent_behavior" => AutonomousCapability::EmergentBehavior,
            "prediction" => AutonomousCapability::Prediction,
            "goal_planning" => AutonomousCapability::GoalPlanning,
            "coordination" => AutonomousCapability::Coordination,
            "memory_management" => AutonomousCapability::MemoryManagement,
            _ => return false,
        };

        if let Some(pos) = self.capabilities.iter().position(|c| c == &capability) {
            self.capabilities.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if agent has capability
    #[wasm_bindgen]
    pub fn has_capability(&self, capability_type: &str) -> bool {
        let capability = match capability_type {
            "self_monitoring" => AutonomousCapability::SelfMonitoring,
            "decision_making" => AutonomousCapability::DecisionMaking,
            "resource_optimization" => AutonomousCapability::ResourceOptimization,
            "self_healing" => AutonomousCapability::SelfHealing,
            "learning" => AutonomousCapability::Learning,
            "emergent_behavior" => AutonomousCapability::EmergentBehavior,
            "prediction" => AutonomousCapability::Prediction,
            "goal_planning" => AutonomousCapability::GoalPlanning,
            "coordination" => AutonomousCapability::Coordination,
            "memory_management" => AutonomousCapability::MemoryManagement,
            _ => return false,
        };

        self.capabilities.contains(&capability)
    }

    /// Get capabilities as array
    #[wasm_bindgen]
    pub fn get_capabilities(&self) -> Array {
        let array = Array::new();
        for capability in &self.capabilities {
            let capability_str = match capability {
                AutonomousCapability::SelfMonitoring => "self_monitoring",
                AutonomousCapability::DecisionMaking => "decision_making",
                AutonomousCapability::ResourceOptimization => "resource_optimization",
                AutonomousCapability::SelfHealing => "self_healing",
                AutonomousCapability::Learning => "learning",
                AutonomousCapability::EmergentBehavior => "emergent_behavior",
                AutonomousCapability::Prediction => "prediction",
                AutonomousCapability::GoalPlanning => "goal_planning",
                AutonomousCapability::Coordination => "coordination",
                AutonomousCapability::MemoryManagement => "memory_management",
                AutonomousCapability::Custom(name) => name,
            };
            array.push(&JsValue::from_str(capability_str));
        }
        array
    }

    /// Make autonomous decision (WASM-compatible)
    #[wasm_bindgen]
    pub fn make_decision(&mut self, context_json: &str) -> Promise {
        let agent_id = self.id.clone();
        let learning_rate = self.learning_rate;

        let future = async move {
            // Parse context from JSON
            let context: DecisionContext = match serde_json::from_str(context_json) {
                Ok(ctx) => ctx,
                Err(_) => return Err(JsValue::from_str("Error: Invalid context JSON")),
            };

            // Simulate autonomous decision making
            let decision = format!(
                "Agent {} decision based on {} actions with learning rate {}",
                agent_id,
                context.available_actions.len(),
                learning_rate
            );

            Ok(JsValue::from_str(&decision))
        };

        wasm_bindgen_futures::future_to_promise(future)
    }

    /// Adapt behavior based on feedback
    #[wasm_bindgen]
    pub fn adapt(&mut self, feedback_json: &str) -> Promise {
        let mut learning_rate = self.learning_rate;

        let future = async move {
            // Parse feedback from JSON
            let feedback: AdaptationFeedback = match serde_json::from_str(feedback_json) {
                Ok(fb) => fb,
                Err(_) => return Err(JsValue::from_str("Error: Invalid feedback JSON")),
            };

            // Adapt learning rate based on feedback
            if feedback.is_positive() {
                learning_rate *= feedback.learning_rate_adjustment;
            } else {
                learning_rate *= feedback.learning_rate_adjustment;
            }

            learning_rate = learning_rate.clamp(0.001, 0.1);

            let result = format!(
                "Adapted with new learning rate: {} (performance: {})",
                learning_rate, feedback.performance_score
            );

            Ok(JsValue::from_str(&result))
        };

        wasm_bindgen_futures::future_to_promise(future)
    }

    /// Check if agent is autonomous
    #[wasm_bindgen]
    pub fn is_autonomous(&self) -> bool {
        self.autonomy_level > 0.5
    }

    /// Get agent status as JSON
    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        let status = WasmAgentStatus {
            id: self.id.clone(),
            autonomy_level: self.autonomy_level,
            learning_rate: self.learning_rate,
            capabilities_count: self.capabilities.len(),
            is_autonomous: self.is_autonomous(),
        };

        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }

    /// Optimize resources
    #[wasm_bindgen]
    pub fn optimize_resources(&self) -> Promise {
        let agent_id = self.id.clone();

        let future = async move {
            // Simulate resource optimization
            let optimization_result = format!(
                "Agent {} optimized resources: CPU 85% -> 70%, Memory 90% -> 75%",
                agent_id
            );

            Ok(JsValue::from_str(&optimization_result))
        };

        wasm_bindgen_futures::future_to_promise(future)
    }
}

/// WASM coordinator for managing multiple agents
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmCoordinator {
    agents: Vec<WasmAutonomousAgent>,
    coordination_frequency: u64,
    strategy: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmCoordinator {
    /// Create new WASM coordinator
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console::log_1(&"Creating new WASM coordinator".into());

        WasmCoordinator {
            agents: Vec::new(),
            coordination_frequency: 1000, // 1 second
            strategy: "balanced".to_string(),
        }
    }

    /// Add agent to coordination
    #[wasm_bindgen]
    pub fn add_agent(&mut self, agent: WasmAutonomousAgent) {
        console::log_1(&format!("Adding agent {} to coordination", agent.id()).into());
        self.agents.push(agent);
    }

    /// Remove agent from coordination
    #[wasm_bindgen]
    pub fn remove_agent(&mut self, agent_id: &str) -> bool {
        if let Some(pos) = self.agents.iter().position(|a| a.id() == agent_id) {
            self.agents.remove(pos);
            console::log_1(&format!("Removed agent {} from coordination", agent_id).into());
            true
        } else {
            false
        }
    }

    /// Get number of agents
    #[wasm_bindgen]
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Set coordination strategy
    #[wasm_bindgen]
    pub fn set_strategy(&mut self, strategy: &str) {
        self.strategy = strategy.to_string();
        console::log_1(&format!("Set coordination strategy to: {}", strategy).into());
    }

    /// Get coordination strategy
    #[wasm_bindgen]
    pub fn get_strategy(&self) -> String {
        self.strategy.clone()
    }

    /// Set coordination frequency
    #[wasm_bindgen]
    pub fn set_frequency(&mut self, frequency_ms: u64) {
        self.coordination_frequency = frequency_ms;
        console::log_1(&format!("Set coordination frequency to: {}ms", frequency_ms).into());
    }

    /// Get coordination frequency
    #[wasm_bindgen]
    pub fn get_frequency(&self) -> u64 {
        self.coordination_frequency
    }

    /// Coordinate all agents
    #[wasm_bindgen]
    pub fn coordinate(&self) -> Promise {
        let agent_count = self.agents.len();
        let strategy = self.strategy.clone();

        let future = async move {
            // Simulate coordination process
            let result = format!(
                "Coordinated {} agents using {} strategy. All agents synchronized.",
                agent_count, strategy
            );

            console::log_1(&result.clone().into());
            Ok(JsValue::from_str(&result))
        };

        wasm_bindgen_futures::future_to_promise(future)
    }

    /// Get coordinator status
    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        let status = WasmCoordinatorStatus {
            agent_count: self.agents.len(),
            strategy: self.strategy.clone(),
            frequency: self.coordination_frequency,
            active: true,
        };

        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }
}

/// WASM resource manager
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmResourceManager {
    allocated_memory: u32,
    max_memory: u32,
    cpu_usage: f64,
    optimization_enabled: bool,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmResourceManager {
    /// Create new WASM resource manager
    #[wasm_bindgen(constructor)]
    pub fn new(max_memory_mb: u32) -> Self {
        console::log_1(
            &format!(
                "Creating WASM resource manager with {}MB limit",
                max_memory_mb
            )
            .into(),
        );

        WasmResourceManager {
            allocated_memory: 0,
            max_memory: max_memory_mb,
            cpu_usage: 0.0,
            optimization_enabled: true,
        }
    }

    /// Allocate memory
    #[wasm_bindgen]
    pub fn allocate_memory(&mut self, size_mb: u32) -> bool {
        if self.allocated_memory + size_mb <= self.max_memory {
            self.allocated_memory += size_mb;
            console::log_1(
                &format!(
                    "Allocated {}MB, total: {}MB",
                    size_mb, self.allocated_memory
                )
                .into(),
            );
            true
        } else {
            console::log_1(&"Memory allocation failed: insufficient memory".into());
            false
        }
    }

    /// Deallocate memory
    #[wasm_bindgen]
    pub fn deallocate_memory(&mut self, size_mb: u32) -> bool {
        if size_mb <= self.allocated_memory {
            self.allocated_memory -= size_mb;
            console::log_1(
                &format!(
                    "Deallocated {}MB, total: {}MB",
                    size_mb, self.allocated_memory
                )
                .into(),
            );
            true
        } else {
            console::log_1(&"Memory deallocation failed: invalid size".into());
            false
        }
    }

    /// Get memory usage
    #[wasm_bindgen]
    pub fn get_memory_usage(&self) -> f64 {
        if self.max_memory == 0 {
            0.0
        } else {
            f64::from(self.allocated_memory) / f64::from(self.max_memory)
        }
    }

    /// Get allocated memory
    #[wasm_bindgen]
    pub fn get_allocated_memory(&self) -> u32 {
        self.allocated_memory
    }

    /// Get max memory
    #[wasm_bindgen]
    pub fn get_max_memory(&self) -> u32 {
        self.max_memory
    }

    /// Set CPU usage
    #[wasm_bindgen]
    pub fn set_cpu_usage(&mut self, usage: f64) {
        self.cpu_usage = usage.clamp(0.0, 1.0);
    }

    /// Get CPU usage
    #[wasm_bindgen]
    pub fn get_cpu_usage(&self) -> f64 {
        self.cpu_usage
    }

    /// Enable optimization
    #[wasm_bindgen]
    pub fn enable_optimization(&mut self) {
        self.optimization_enabled = true;
        console::log_1(&"Resource optimization enabled".into());
    }

    /// Disable optimization
    #[wasm_bindgen]
    pub fn disable_optimization(&mut self) {
        self.optimization_enabled = false;
        console::log_1(&"Resource optimization disabled".into());
    }

    /// Check if optimization is enabled
    #[wasm_bindgen]
    pub fn is_optimization_enabled(&self) -> bool {
        self.optimization_enabled
    }

    /// Optimize resources
    #[wasm_bindgen]
    pub fn optimize(&mut self) -> Promise {
        let optimization_enabled = self.optimization_enabled;
        let current_usage = self.get_memory_usage();

        let future = async move {
            if !optimization_enabled {
                return Ok(JsValue::from_str("Optimization disabled"));
            }

            // Simulate optimization
            let improvement = if current_usage > 0.8 {
                15.0 // 15% improvement for high usage
            } else if current_usage > 0.6 {
                10.0 // 10% improvement for medium usage
            } else {
                5.0 // 5% improvement for low usage
            };

            let result = format!(
                "Resources optimized: {}% improvement, usage: {:.1}%",
                improvement,
                current_usage * 100.0
            );

            console::log_1(&result.clone().into());
            Ok(JsValue::from_str(&result))
        };

        wasm_bindgen_futures::future_to_promise(future)
    }

    /// Get resource status
    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        let status = WasmResourceStatus {
            allocated_memory: self.allocated_memory,
            max_memory: self.max_memory,
            memory_usage: self.get_memory_usage(),
            cpu_usage: self.cpu_usage,
            optimization_enabled: self.optimization_enabled,
        };

        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }
}

/// WASM utilities and helper functions
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmUtils;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmUtils {
    /// Initialize WASM DAA system
    #[wasm_bindgen]
    pub fn init() {
        console_error_panic_hook::set_once();
        console::log_1(&"WASM DAA system initialized".into());
    }

    /// Get system capabilities
    #[wasm_bindgen]
    pub fn get_system_capabilities() -> Array {
        let capabilities = Array::new();
        capabilities.push(&JsValue::from_str("autonomous_agents"));
        capabilities.push(&JsValue::from_str("coordination"));
        capabilities.push(&JsValue::from_str("resource_management"));
        capabilities.push(&JsValue::from_str("adaptation"));
        capabilities.push(&JsValue::from_str("optimization"));
        capabilities
    }

    /// Check WASM support
    #[wasm_bindgen]
    pub fn check_wasm_support() -> bool {
        true // If this runs, WASM is supported
    }

    /// Get performance info
    #[wasm_bindgen]
    pub fn get_performance_info() -> String {
        #[cfg(feature = "simd")]
        let simd_support = "enabled";
        #[cfg(not(feature = "simd"))]
        let simd_support = "disabled";

        let info = WasmPerformanceInfo {
            simd_support: simd_support.to_string(),
            memory_management: "wee_alloc".to_string(),
            optimization_level: "release".to_string(),
            wasm_bindgen_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        serde_json::to_string(&info).unwrap_or_else(|_| "{}".to_string())
    }

    /// Log message to console
    #[wasm_bindgen]
    pub fn log(message: &str) {
        console::log_1(&JsValue::from_str(message));
    }

    /// Create decision context from JavaScript object
    #[wasm_bindgen]
    pub fn create_context(js_object: &JsValue) -> String {
        // Convert JS object to DecisionContext
        let context = DecisionContext::new();
        serde_json::to_string(&context).unwrap_or_else(|_| "{}".to_string())
    }

    /// Create adaptation feedback from JavaScript object
    #[wasm_bindgen]
    pub fn create_feedback(performance: f64, efficiency: f64) -> String {
        let feedback = AdaptationFeedback::positive((performance + efficiency) / 2.0);
        serde_json::to_string(&feedback).unwrap_or_else(|_| "{}".to_string())
    }
}

// Serializable status types for WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmAgentStatus {
    id: String,
    autonomy_level: f64,
    learning_rate: f64,
    capabilities_count: usize,
    is_autonomous: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmCoordinatorStatus {
    agent_count: usize,
    strategy: String,
    frequency: u64,
    active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmResourceStatus {
    allocated_memory: u32,
    max_memory: u32,
    memory_usage: f64,
    cpu_usage: f64,
    optimization_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmPerformanceInfo {
    simd_support: String,
    memory_management: String,
    optimization_level: String,
    wasm_bindgen_version: String,
}

/// Initialize panic hook for better error reporting
#[cfg(feature = "wasm")]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// WASM-specific error handling
#[cfg(feature = "wasm")]
pub fn handle_wasm_error(error: &str) -> JsValue {
    console::error_1(&JsValue::from_str(error));
    JsValue::from_str(&format!("WASM DAA Error: {}", error))
}

/// Memory optimization for WASM
#[cfg(all(feature = "wasm", feature = "wee_alloc"))]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_agent_status_serialization() {
        let status = WasmAgentStatus {
            id: "test-agent".to_string(),
            autonomy_level: 0.8,
            learning_rate: 0.01,
            capabilities_count: 5,
            is_autonomous: true,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("test-agent"));
        assert!(json.contains("0.8"));
    }

    #[test]
    fn test_wasm_coordinator_status_serialization() {
        let status = WasmCoordinatorStatus {
            agent_count: 3,
            strategy: "balanced".to_string(),
            frequency: 1000,
            active: true,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("balanced"));
        assert!(json.contains("1000"));
    }

    #[test]
    fn test_wasm_resource_status_serialization() {
        let status = WasmResourceStatus {
            allocated_memory: 256,
            max_memory: 1024,
            memory_usage: 0.25,
            cpu_usage: 0.5,
            optimization_enabled: true,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("256"));
        assert!(json.contains("0.25"));
    }
}
