// Utilities for WASM optimization and system detection
use wasm_bindgen::prelude::*;
use web_sys::window;

mod memory;
mod simd;
pub mod bridge;

pub use memory::*;
pub use simd::*;
pub use bridge::*;

// Detect SIMD support at runtime
pub fn detect_simd_support() -> bool {
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

// Detect Web Workers support
pub fn detect_worker_support() -> bool {
    js_sys::eval("typeof Worker !== 'undefined'")
        .map(|v| v.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

// Calculate optimal memory allocation based on device capabilities
pub fn calculate_optimal_memory() -> u32 {
    // Device memory is not widely supported, use fallback
    let available = 4.0; // Default to 4GB
    
    // Use 1/8 of available device memory, minimum 4MB, maximum 64MB
    ((available * 1024.0f64 / 8.0).max(4.0).min(64.0) / 0.0625) as u32
}

// Get current memory usage
pub fn get_current_memory_usage() -> u32 {
    let memory = wasm_bindgen::memory();
    // Get byte length safely
    if let Ok(buffer) = js_sys::Reflect::get(&memory, &"buffer".into()) {
        if let Ok(length) = js_sys::Reflect::get(&buffer, &"byteLength".into()) {
            return (length.as_f64().unwrap_or(0.0) / 1024.0) as u32;
        }
    }
    0 // Fallback
}

// Performance timing utilities
#[wasm_bindgen]
pub struct PerformanceTimer {
    start_time: f64,
}

#[wasm_bindgen]
impl PerformanceTimer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PerformanceTimer {
        let start_time = window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
            
        PerformanceTimer { start_time }
    }

    #[wasm_bindgen]
    pub fn elapsed(&self) -> f64 {
        window()
            .and_then(|w| w.performance())
            .map(|p| p.now() - self.start_time)
            .unwrap_or(0.0)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.start_time = window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
    }
}

// Console logging utilities
#[wasm_bindgen]
pub fn log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}

#[wasm_bindgen]
pub fn error(msg: &str) {
    web_sys::console::error_1(&msg.into());
}

#[wasm_bindgen]
pub fn warn(msg: &str) {
    web_sys::console::warn_1(&msg.into());
}

// Feature detection summary
#[wasm_bindgen]
pub fn get_system_capabilities() -> JsValue {
    let caps = serde_json::json!({
        "simd": detect_simd_support(),
        "workers": detect_worker_support(),
        "memory_pages": calculate_optimal_memory(),
        "current_memory_kb": get_current_memory_usage(),
        "wasm_features": {
            "bulk_memory": cfg!(target_feature = "bulk-memory"),
            "simd128": cfg!(target_feature = "simd128"),
            "reference_types": cfg!(target_feature = "reference-types"),
            "multi_value": cfg!(target_feature = "multivalue"),
        }
    });
    
    serde_wasm_bindgen::to_value(&caps).unwrap()
}