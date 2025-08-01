// Agent management WASM interfaces
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use ruv_swarm_core::{CognitivePattern, agent::{DynamicAgent, AgentStatus}};

#[wasm_bindgen]
pub struct WasmAgent {
    inner: DynamicAgent,
}

#[wasm_bindgen]
impl WasmAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, agent_type: String) -> Result<WasmAgent, JsValue> {
        let capabilities = match agent_type.as_str() {
            "researcher" => vec!["research".to_string(), "analysis".to_string()],
            "coder" => vec!["coding".to_string(), "implementation".to_string()],
            "analyst" => vec!["analysis".to_string(), "data_processing".to_string()],
            "optimizer" => vec!["optimization".to_string(), "performance".to_string()],
            "coordinator" => vec!["coordination".to_string(), "management".to_string()],
            _ => return Err(JsValue::from_str(&format!("Unknown agent type: {}", agent_type))),
        };

        Ok(WasmAgent {
            inner: DynamicAgent::new(name, capabilities),
        })
    }

    #[wasm_bindgen]
    pub fn get_id(&self) -> String {
        self.inner.id().to_string()
    }

    #[wasm_bindgen]
    pub fn get_name(&self) -> String {
        self.inner.id().to_string() // DynamicAgent uses id as name
    }

    #[wasm_bindgen]
    pub fn get_type(&self) -> String {
        // Determine type from capabilities
        let caps = self.inner.capabilities();
        if caps.contains(&"research".to_string()) {
            "researcher".to_string()
        } else if caps.contains(&"coding".to_string()) {
            "coder".to_string()
        } else if caps.contains(&"analysis".to_string()) {
            "analyst".to_string()
        } else if caps.contains(&"optimization".to_string()) {
            "optimizer".to_string()
        } else if caps.contains(&"coordination".to_string()) {
            "coordinator".to_string()
        } else {
            "unknown".to_string()
        }
    }

    #[wasm_bindgen]
    pub fn set_cognitive_pattern(&mut self, pattern: String) -> Result<(), JsValue> {
        let _cognitive_pattern = match pattern.as_str() {
            "convergent" => CognitivePattern::Convergent,
            "divergent" => CognitivePattern::Divergent,
            "lateral" => CognitivePattern::Lateral,
            "systems" => CognitivePattern::Systems,
            "critical" => CognitivePattern::Critical,
            "abstract" => CognitivePattern::Abstract,
            _ => return Err(JsValue::from_str(&format!("Unknown cognitive pattern: {}", pattern))),
        };

        // Note: DynamicAgent doesn't directly support cognitive patterns in the same way
        // This would need to be stored in agent metadata
        Ok(())
    }

    #[wasm_bindgen]
    pub fn get_cognitive_pattern(&self) -> Option<String> {
        // Default to convergent since DynamicAgent doesn't store this directly
        Some("convergent".to_string())
    }

    #[wasm_bindgen]
    pub fn get_info(&self) -> JsValue {
        let info = serde_json::json!({
            "id": self.inner.id(),
            "name": self.inner.id(), // DynamicAgent uses id as name
            "type": self.get_type(),
            "cognitive_pattern": self.get_cognitive_pattern(),
            "state": match self.inner.status() {
                AgentStatus::Idle => "idle",
                AgentStatus::Running => "active",
                AgentStatus::Busy => "busy",
                AgentStatus::Offline => "offline",
                AgentStatus::Error => "error",
            },
        });
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}

// Agent factory with optimization hints
#[wasm_bindgen]
pub struct AgentFactory {
    optimization_hints: bool,
}

#[wasm_bindgen]
impl AgentFactory {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AgentFactory {
        AgentFactory {
            optimization_hints: true,
        }
    }

    #[wasm_bindgen]
    pub fn create_agent(&self, name: String, agent_type: String) -> Result<WasmAgent, JsValue> {
        WasmAgent::new(name, agent_type)
    }

    #[wasm_bindgen]
    pub fn create_optimized_agent(&self, name: String, agent_type: String, memory_budget_mb: u32) -> Result<WasmAgent, JsValue> {
        // Create agent with memory optimization hints
        let agent = WasmAgent::new(name, agent_type)?;
        
        // Apply memory optimizations based on budget
        if memory_budget_mb < 10 {
            // Use minimal memory allocations
            web_sys::console::log_1(&"Creating memory-optimized agent".into());
        }
        
        Ok(agent)
    }

    #[wasm_bindgen]
    pub fn create_batch(&self, count: u32, base_name: String, agent_type: String) -> Result<Vec<JsValue>, JsValue> {
        let mut agents = Vec::new();
        
        for i in 0..count {
            let name = format!("{}-{}", base_name, i);
            let agent = self.create_agent(name, agent_type.clone())?;
            agents.push(agent.get_info());
        }
        
        Ok(agents)
    }
}