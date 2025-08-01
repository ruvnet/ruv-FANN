//! Simplified WASM bindings for DAA systems

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use web_sys::console;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// WASM-compatible DAA agent
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmDAAAgent {
    id: String,
    autonomy_level: f64,
    learning_rate: f64,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmDAAAgent {
    /// Create new WASM DAA agent
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str) -> Self {
        console::log_1(&"Creating new WASM DAA agent".into());

        WasmDAAAgent {
            id: id.to_string(),
            autonomy_level: 1.0,
            learning_rate: 0.01,
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

    /// Make decision (simplified)
    #[wasm_bindgen]
    pub fn make_decision(&mut self, context: &str) -> String {
        console::log_1(
            &format!(
                "Agent {} making decision with context: {}",
                self.id, context
            )
            .into(),
        );

        // Simple decision logic
        let decision = format!(
            "Agent {} decision: action_{} (autonomy: {:.2})",
            self.id,
            (self.autonomy_level * 10.0) as u32,
            self.autonomy_level
        );

        decision
    }

    /// Adapt behavior (simplified)
    #[wasm_bindgen]
    pub fn adapt(&mut self, performance_score: f64) -> String {
        // Adjust learning rate based on performance
        if performance_score > 0.8 {
            self.learning_rate *= 1.1;
        } else if performance_score < 0.5 {
            self.learning_rate *= 0.9;
        }

        self.learning_rate = self.learning_rate.clamp(0.001, 0.1);

        format!(
            "Agent {} adapted: new learning rate = {:.4}",
            self.id, self.learning_rate
        )
    }

    /// Get agent status as JSON
    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        let status = AgentStatus {
            id: self.id.clone(),
            autonomy_level: self.autonomy_level,
            learning_rate: self.learning_rate,
        };

        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }
}

/// WASM DAA coordinator
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmDAACoordinator {
    agent_count: u32,
    coordination_frequency: u64,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmDAACoordinator {
    /// Create new coordinator
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console::log_1(&"Creating new WASM DAA coordinator".into());

        WasmDAACoordinator {
            agent_count: 0,
            coordination_frequency: 1000,
        }
    }

    /// Add agent to coordination
    #[wasm_bindgen]
    pub fn add_agent(&mut self) {
        self.agent_count += 1;
        console::log_1(&format!("Added agent, total: {}", self.agent_count).into());
    }

    /// Remove agent from coordination
    #[wasm_bindgen]
    pub fn remove_agent(&mut self) -> bool {
        if self.agent_count > 0 {
            self.agent_count -= 1;
            true
        } else {
            false
        }
    }

    /// Get agent count
    #[wasm_bindgen]
    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    /// Coordinate agents
    #[wasm_bindgen]
    pub fn coordinate(&self) -> String {
        console::log_1(&format!("Coordinating {} agents", self.agent_count).into());

        format!(
            "Coordinated {} agents at frequency {}ms",
            self.agent_count, self.coordination_frequency
        )
    }

    /// Get coordinator status
    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        let status = CoordinatorStatus {
            agent_count: self.agent_count,
            coordination_frequency: self.coordination_frequency,
        };

        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }
}

/// DAA utilities
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct DAAUtils;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl DAAUtils {
    /// Initialize DAA system
    #[wasm_bindgen]
    pub fn init() {
        console_error_panic_hook::set_once();
        console::log_1(&"DAA WASM system initialized".into());
    }

    /// Check SIMD support
    #[wasm_bindgen]
    pub fn check_simd_support() -> bool {
        // Check for WebAssembly SIMD support
        #[cfg(target_feature = "simd128")]
        {
            true
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            false
        }
    }

    /// Get system info
    #[wasm_bindgen]
    pub fn get_system_info() -> String {
        let info = SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            simd_support: Self::check_simd_support(),
            wasm_bindgen_version: "0.2".to_string(),
        };

        serde_json::to_string(&info).unwrap_or_else(|_| "{}".to_string())
    }
}

// Serializable types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentStatus {
    id: String,
    autonomy_level: f64,
    learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoordinatorStatus {
    agent_count: u32,
    coordination_frequency: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemInfo {
    version: String,
    simd_support: bool,
    wasm_bindgen_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_status_serialization() {
        let status = AgentStatus {
            id: "test-agent".to_string(),
            autonomy_level: 0.8,
            learning_rate: 0.01,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("test-agent"));
        assert!(json.contains("0.8"));
    }
}
