// JavaScript-WASM bridge utilities
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// Utility for converting between JS and Rust types
#[wasm_bindgen]
pub struct DataBridge;

#[wasm_bindgen]
impl DataBridge {
    #[wasm_bindgen]
    pub fn array_to_vec_f32(array: &js_sys::Float32Array) -> Vec<f32> {
        array.to_vec()
    }
    
    #[wasm_bindgen]
    pub fn vec_f32_to_array(vec: Vec<f32>) -> js_sys::Float32Array {
        js_sys::Float32Array::from(&vec[..])
    }
    
    #[wasm_bindgen]
    pub fn array_to_vec_u32(array: &js_sys::Uint32Array) -> Vec<u32> {
        array.to_vec()
    }
    
    #[wasm_bindgen]
    pub fn vec_u32_to_array(vec: Vec<u32>) -> js_sys::Uint32Array {
        js_sys::Uint32Array::from(&vec[..])
    }
    
    #[wasm_bindgen]
    pub fn json_parse(json_str: &str) -> Result<JsValue, JsValue> {
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(value) => Ok(serde_wasm_bindgen::to_value(&value)?),
            Err(e) => Err(JsValue::from_str(&format!("JSON parse error: {}", e))),
        }
    }
    
    #[wasm_bindgen]
    pub fn json_stringify(value: &JsValue) -> Result<String, JsValue> {
        let parsed: serde_json::Value = serde_wasm_bindgen::from_value(value.clone())?;
        match serde_json::to_string(&parsed) {
            Ok(json_str) => Ok(json_str),
            Err(e) => Err(JsValue::from_str(&format!("JSON stringify error: {}", e))),
        }
    }
}

// Promise utilities for async operations
#[wasm_bindgen]
pub struct PromiseUtils;

#[wasm_bindgen]
impl PromiseUtils {
    #[wasm_bindgen]
    pub fn create_resolved_promise(value: JsValue) -> js_sys::Promise {
        js_sys::Promise::resolve(&value)
    }
    
    #[wasm_bindgen]
    pub fn create_rejected_promise(error: JsValue) -> js_sys::Promise {
        js_sys::Promise::reject(&error)
    }
}

// Error handling utilities
#[wasm_bindgen]
pub struct ErrorBridge;

#[wasm_bindgen]
impl ErrorBridge {
    #[wasm_bindgen]
    pub fn create_error(message: &str) -> JsValue {
        js_sys::Error::new(message).into()
    }
    
    #[wasm_bindgen]
    pub fn create_type_error(message: &str) -> JsValue {
        js_sys::TypeError::new(message).into()
    }
    
    #[wasm_bindgen]
    pub fn create_range_error(message: &str) -> JsValue {
        js_sys::RangeError::new(message).into()
    }
}

// Performance monitoring bridge
#[wasm_bindgen]
pub struct PerformanceBridge {
    marks: std::collections::HashMap<String, f64>,
}

#[wasm_bindgen]
impl PerformanceBridge {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PerformanceBridge {
        PerformanceBridge {
            marks: std::collections::HashMap::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn mark(&mut self, name: String) {
        if let Some(window) = web_sys::window() {
            if let Some(performance) = window.performance() {
                self.marks.insert(name, performance.now());
            }
        }
    }
    
    #[wasm_bindgen]
    pub fn measure(&self, name: String, start_mark: String, end_mark: String) -> Option<f64> {
        let start = self.marks.get(&start_mark)?;
        let end = self.marks.get(&end_mark)?;
        Some(end - start)
    }
    
    #[wasm_bindgen]
    pub fn get_metrics(&self) -> JsValue {
        let metrics = serde_json::json!({
            "marks": self.marks,
            "count": self.marks.len(),
        });
        
        serde_wasm_bindgen::to_value(&metrics).unwrap()
    }
}

// Shared memory utilities for Web Workers
#[wasm_bindgen]
pub struct SharedMemoryBridge;

#[wasm_bindgen]
impl SharedMemoryBridge {
    #[wasm_bindgen]
    pub fn create_shared_buffer(size: usize) -> Result<js_sys::SharedArrayBuffer, JsValue> {
        // Check if SharedArrayBuffer is available
        if js_sys::eval("typeof SharedArrayBuffer !== 'undefined'")?.as_bool().unwrap_or(false) {
            Ok(js_sys::SharedArrayBuffer::new(size as u32))
        } else {
            Err(JsValue::from_str("SharedArrayBuffer not available"))
        }
    }
    
    #[wasm_bindgen]
    pub fn is_shared_memory_available() -> bool {
        js_sys::eval("typeof SharedArrayBuffer !== 'undefined'")
            .map(|v| v.as_bool().unwrap_or(false))
            .unwrap_or(false)
    }
}

// Type conversion utilities
#[derive(Serialize, Deserialize)]
pub struct TypedData<T> {
    pub data: T,
    pub type_name: String,
}

#[wasm_bindgen]
pub fn convert_to_typed_data(value: JsValue, type_name: String) -> Result<JsValue, JsValue> {
    let data: serde_json::Value = serde_wasm_bindgen::from_value(value)?;
    let typed = TypedData {
        data,
        type_name,
    };
    
    Ok(serde_wasm_bindgen::to_value(&typed)?)
}

// Utility for batched operations
#[wasm_bindgen]
pub struct BatchProcessor;

#[wasm_bindgen]
impl BatchProcessor {
    #[wasm_bindgen]
    pub fn process_batch(
        items: Vec<JsValue>,
        batch_size: usize,
        processor: &js_sys::Function,
    ) -> Result<Vec<JsValue>, JsValue> {
        let mut results = Vec::new();
        
        for chunk in items.chunks(batch_size) {
            let batch_array = js_sys::Array::new();
            for item in chunk {
                batch_array.push(item);
            }
            
            let result = processor.call1(&JsValue::NULL, &batch_array)?;
            results.push(result);
        }
        
        Ok(results)
    }
}

// Debug utilities
#[wasm_bindgen]
pub fn enable_debug_mode() {
    crate::utils::log("Debug mode enabled for WASM module");
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
pub fn get_debug_info() -> JsValue {
    // Convert JsValues to serializable format
    let features_js = crate::get_features();
    let capabilities_js = crate::utils::get_system_capabilities();
    
    let info = serde_json::json!({
        "version": crate::get_version(),
        "features": "see features endpoint", // JsValue not directly serializable
        "memory": {
            "pages": crate::utils::get_current_memory_usage() / 64, // Estimate pages
            "usage_kb": crate::utils::get_current_memory_usage(),
        },
        "capabilities": "see capabilities endpoint", // JsValue not directly serializable
    });
    
    serde_wasm_bindgen::to_value(&info).unwrap()
}