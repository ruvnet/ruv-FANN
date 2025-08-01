// Network topology WASM interfaces
use wasm_bindgen::prelude::*;
use ruv_swarm_core::Topology;

#[wasm_bindgen]
pub struct TopologyManager;

#[wasm_bindgen]
impl TopologyManager {
    #[wasm_bindgen]
    pub fn list_topologies() -> Vec<JsValue> {
        vec![
            JsValue::from_str("mesh"),
            JsValue::from_str("star"),
            JsValue::from_str("ring"),
            JsValue::from_str("hierarchical"),
        ]
    }
    
    #[wasm_bindgen]
    pub fn get_topology_info(topology: String) -> JsValue {
        let info = match topology.as_str() {
            "mesh" => serde_json::json!({
                "name": "mesh",
                "description": "Fully connected topology where every agent can communicate with every other agent",
                "scalability": "O(n²) connections",
                "use_case": "Small to medium swarms requiring high collaboration"
            }),
            "star" => serde_json::json!({
                "name": "star",
                "description": "Central coordinator with all agents connected to it",
                "scalability": "O(n) connections",
                "use_case": "Centralized control and coordination"
            }),
            "ring" => serde_json::json!({
                "name": "ring",
                "description": "Agents connected in a circular pattern",
                "scalability": "O(n) connections",
                "use_case": "Sequential processing and pipeline architectures"
            }),
            "hierarchical" => serde_json::json!({
                "name": "hierarchical",
                "description": "Tree-like structure with multiple levels",
                "scalability": "O(n) connections",
                "use_case": "Large swarms with delegation and sub-coordination"
            }),
            _ => serde_json::json!({
                "error": format!("Unknown topology: {}", topology)
            }),
        };
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}