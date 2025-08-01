use wasm_bindgen::prelude::*;
use web_sys::console;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
pub struct DAAAgent {
    id: String,
    autonomy_level: f64,
}

#[wasm_bindgen]
impl DAAAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str) -> Self {
        console::log_1(&format!("Creating DAA agent: {}", id).into());
        DAAAgent {
            id: id.to_string(),
            autonomy_level: 1.0,
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }
    
    #[wasm_bindgen]
    pub fn make_decision(&self, context: &str) -> String {
        format!("Agent {} decision for context: {}", self.id, context)
    }
    
    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        let status = Status {
            id: self.id.clone(),
            autonomy_level: self.autonomy_level,
        };
        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }
}

#[wasm_bindgen]
pub fn init_daa() {
    console_error_panic_hook::set_once();
    console::log_1(&"DAA WASM initialized".into());
}

#[derive(Serialize, Deserialize)]
struct Status {
    id: String,
    autonomy_level: f64,
}
