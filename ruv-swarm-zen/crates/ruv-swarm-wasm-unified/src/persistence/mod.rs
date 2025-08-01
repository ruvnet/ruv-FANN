// Persistence layer WASM interfaces
use wasm_bindgen::prelude::*;
use std::collections::HashMap;

#[wasm_bindgen]
pub struct PersistenceManager {
    storage: HashMap<String, String>,
}

#[wasm_bindgen]
impl PersistenceManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<PersistenceManager, JsValue> {
        Ok(PersistenceManager {
            storage: HashMap::new(),
        })
    }
    
    #[wasm_bindgen]
    pub fn store(&mut self, key: &str, value: JsValue) -> Result<(), JsValue> {
        let data = crate::utils::DataBridge::json_stringify(&value)?;
        self.storage.insert(key.to_string(), data);
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn retrieve(&self, key: &str) -> Result<JsValue, JsValue> {
        if let Some(data) = self.storage.get(key) {
            crate::utils::DataBridge::json_parse(data)
        } else {
            Err(JsValue::from_str(&format!("Key not found: {}", key)))
        }
    }
}