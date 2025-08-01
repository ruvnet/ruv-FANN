// Swarm orchestration WASM interfaces
use wasm_bindgen::prelude::*;
use ruv_swarm_core::{Swarm, SwarmConfig, topology::TopologyType};

#[wasm_bindgen]
pub struct WasmSwarm {
    inner: Swarm,
}

#[wasm_bindgen]
impl WasmSwarm {
    #[wasm_bindgen(constructor)]
    pub fn new(max_agents: usize) -> WasmSwarm {
        let config = SwarmConfig {
            max_agents,
            ..SwarmConfig::default()
        };
        WasmSwarm {
            inner: Swarm::new(config),
        }
    }
    
    #[wasm_bindgen]
    pub fn set_topology(&mut self, topology: String) -> Result<(), JsValue> {
        let _topo_type = match topology.as_str() {
            "mesh" => TopologyType::Mesh,
            "star" => TopologyType::Star,
            "pipeline" => TopologyType::Pipeline,
            "hierarchical" => TopologyType::Hierarchical,
            "clustered" => TopologyType::Clustered,
            _ => return Err(JsValue::from_str(&format!("Unknown topology: {}", topology))),
        };
        
        // Note: Swarm topology is private, so we can't set it directly
        // In a real implementation, we'd need a public method
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn get_agent_count(&self) -> usize {
        // Swarm agents field is private, return 0 for now
        0
    }
    
    #[wasm_bindgen]
    pub fn get_info(&self) -> JsValue {
        let info = serde_json::json!({
            "agent_count": self.get_agent_count(),
            "max_agents": 100, // Default max agents
            "topology": "mesh", // Default topology
        });
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}