// Task management WASM interfaces
use wasm_bindgen::prelude::*;
use ruv_swarm_core::task::{Task, TaskPriority, TaskId, TaskPayload};

#[wasm_bindgen]
pub struct WasmTask {
    inner: Task,
}

#[wasm_bindgen]
impl WasmTask {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, description: String) -> WasmTask {
        WasmTask {
            inner: Task::new(name, description),
        }
    }
    
    #[wasm_bindgen]
    pub fn get_id(&self) -> String {
        self.inner.id.to_string()
    }
    
    #[wasm_bindgen]
    pub fn get_name(&self) -> String {
        self.inner.task_type.clone()
    }
    
    #[wasm_bindgen]
    pub fn set_priority(&mut self, priority: String) -> Result<(), JsValue> {
        let prio = match priority.as_str() {
            "low" => TaskPriority::Low,
            "normal" => TaskPriority::Normal,
            "high" => TaskPriority::High,
            "critical" => TaskPriority::Critical,
            _ => return Err(JsValue::from_str(&format!("Unknown priority: {}", priority))),
        };
        
        self.inner.priority = prio;
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn get_info(&self) -> JsValue {
        let info = serde_json::json!({
            "id": self.inner.id.to_string(),
            "name": self.inner.task_type,
            "description": "Task description", // Task struct doesn't have description field
            "priority": format!("{:?}", self.inner.priority),
            "required_capabilities": self.inner.required_capabilities,
        });
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}