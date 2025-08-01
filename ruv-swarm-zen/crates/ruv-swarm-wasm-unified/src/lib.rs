// ruv-swarm unified WASM module
// Exposes all capabilities from ruv-FANN ecosystem through optimized WebAssembly interfaces

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// Configure allocator for WASM
#[cfg(feature = "wee_alloc_feature")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Set panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// Re-export submodules
pub mod core;
pub mod neural;
pub mod forecasting;
pub mod persistence;
pub mod utils;

// Main configuration structure
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct WasmConfig {
    pub use_simd: bool,
    pub enable_parallel: bool,
    pub memory_pages: u32,
    pub stack_size: u32,
}

#[wasm_bindgen]
impl WasmConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmConfig {
        WasmConfig {
            use_simd: utils::detect_simd_support(),
            enable_parallel: utils::detect_worker_support(),
            memory_pages: utils::calculate_optimal_memory(),
            stack_size: 1024 * 1024, // 1MB default
        }
    }
    
    #[wasm_bindgen]
    pub fn optimize_for_neural_networks(&mut self) {
        self.memory_pages = 256; // 16MB for neural processing
        self.stack_size = 2 * 1024 * 1024; // 2MB for deep networks
    }
    
    #[wasm_bindgen]
    pub fn optimize_for_swarm(&mut self, agent_count: u32) {
        self.memory_pages = 64 + (agent_count * 4); // Base + per-agent
        self.enable_parallel = agent_count > 5;
    }

    #[wasm_bindgen]
    pub fn get_info(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self).unwrap()
    }
}

// Version information
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn get_features() -> JsValue {
    let features = serde_json::json!({
        "version": get_version(),
        "simd": cfg!(target_feature = "simd128"),
        "parallel": cfg!(feature = "parallel"),
        "optimize": cfg!(feature = "optimize"),
        "neural": cfg!(feature = "ruv-fann"),
        "forecasting": cfg!(feature = "neuro-divergent"),
        "allocator": if cfg!(feature = "wee_alloc_feature") { "wee_alloc" } else { "system" },
    });
    
    serde_wasm_bindgen::to_value(&features).unwrap()
}