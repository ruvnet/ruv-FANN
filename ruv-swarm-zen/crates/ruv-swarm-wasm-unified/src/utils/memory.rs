// Memory management utilities for efficient WASM operation
use wasm_bindgen::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Thread-safe memory pool
type MemoryPool = Arc<Mutex<HashMap<String, Vec<u8>>>>;

// Global memory manager instance
lazy_static::lazy_static! {
    static ref MEMORY_MANAGER: MemoryPool = Arc::new(Mutex::new(HashMap::new()));
}

#[wasm_bindgen]
pub struct WasmMemoryManager {
    allocation_pools: HashMap<String, Vec<u8>>,
    peak_usage: usize,
    current_usage: usize,
}

#[wasm_bindgen]
impl WasmMemoryManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmMemoryManager {
        WasmMemoryManager {
            allocation_pools: HashMap::new(),
            peak_usage: 0,
            current_usage: 0,
        }
    }
    
    // Pre-allocate memory pools for different use cases
    #[wasm_bindgen]
    pub fn initialize_pools(&mut self) {
        // Neural network weight storage
        self.create_pool("neural_weights", 10 * 1024 * 1024); // 10MB
        
        // Agent state storage
        self.create_pool("agent_states", 5 * 1024 * 1024); // 5MB
        
        // Task queue and results
        self.create_pool("task_data", 2 * 1024 * 1024); // 2MB
        
        // Time series data
        self.create_pool("timeseries", 8 * 1024 * 1024); // 8MB
        
        crate::utils::log(&format!("Memory pools initialized: {} MB total", 
            self.current_usage / (1024 * 1024)));
    }
    
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> JsValue {
        let stats = serde_json::json!({
            "current_usage_mb": self.current_usage as f64 / (1024.0 * 1024.0),
            "peak_usage_mb": self.peak_usage as f64 / (1024.0 * 1024.0),
            "pools": self.allocation_pools.keys().cloned().collect::<Vec<_>>(),
            "wasm_memory_pages": crate::utils::get_current_memory_usage() / 64,
            "efficiency": if self.peak_usage > 0 {
                (self.current_usage as f64 / self.peak_usage as f64) * 100.0
            } else {
                0.0
            }
        });
        
        serde_wasm_bindgen::to_value(&stats).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn allocate_from_pool(&mut self, pool_name: &str, size: usize) -> Result<usize, JsValue> {
        if let Some(pool) = self.allocation_pools.get_mut(pool_name) {
            if pool.capacity() >= size {
                // Return offset into pool
                Ok(0) // Simplified for now
            } else {
                Err(JsValue::from_str(&format!("Pool {} has insufficient capacity", pool_name)))
            }
        } else {
            Err(JsValue::from_str(&format!("Pool {} not found", pool_name)))
        }
    }
    
    #[wasm_bindgen]
    pub fn deallocate(&mut self, pool_name: &str, offset: usize, size: usize) -> Result<(), JsValue> {
        // Mark memory as available in pool
        // Simplified implementation - real version would track free blocks
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn compact_pools(&mut self) {
        // Compact fragmented memory pools
        for (name, pool) in &mut self.allocation_pools {
            crate::utils::log(&format!("Compacting pool: {}", name));
            // Defragmentation logic would go here
        }
    }
    
    #[wasm_bindgen]
    pub fn grow_memory(&mut self, pages: u32) -> Result<u32, JsValue> {
        // Simplified memory growth simulation
        crate::utils::log(&format!("Memory growth requested: {} pages", pages));
        Ok(pages) // Return success for simulation
    }
    
    fn create_pool(&mut self, name: &str, size: usize) {
        let pool = Vec::with_capacity(size);
        self.allocation_pools.insert(name.to_string(), pool);
        self.current_usage += size;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }
}

// Memory pressure monitoring
#[wasm_bindgen]
pub struct MemoryMonitor {
    threshold_mb: f64,
    callback: Option<js_sys::Function>,
}

#[wasm_bindgen]
impl MemoryMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new(threshold_mb: f64) -> MemoryMonitor {
        MemoryMonitor {
            threshold_mb,
            callback: None,
        }
    }
    
    #[wasm_bindgen]
    pub fn set_callback(&mut self, callback: js_sys::Function) {
        self.callback = Some(callback);
    }
    
    #[wasm_bindgen]
    pub fn check_memory_pressure(&self) -> bool {
        let current_mb = (crate::utils::get_current_memory_usage() as f64) / 1024.0;
        let pressure = current_mb > self.threshold_mb;
        
        if pressure && self.callback.is_some() {
            let _ = self.callback.as_ref().unwrap().call0(&JsValue::NULL);
        }
        
        pressure
    }
    
    #[wasm_bindgen]
    pub fn get_memory_info(&self) -> JsValue {
        let memory_kb = crate::utils::get_current_memory_usage();
        let info = serde_json::json!({
            "current_mb": (memory_kb as f64) / 1024.0,
            "threshold_mb": self.threshold_mb,
            "pressure": (memory_kb as f64) / 1024.0 > self.threshold_mb,
            "pages": memory_kb / 64,
        });
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}

// Utility functions for memory optimization
#[wasm_bindgen]
pub fn optimize_memory_layout() {
    crate::utils::log("Optimizing memory layout for WASM execution...");
    // Force garbage collection if available
    if let Ok(gc) = js_sys::eval("if (typeof gc !== 'undefined') { gc(); return true; } return false;") {
        if gc.as_bool().unwrap_or(false) {
            crate::utils::log("Garbage collection completed");
        }
    }
}

#[wasm_bindgen]
pub fn estimate_memory_for_agents(agent_count: u32) -> u32 {
    // Estimate memory requirements
    let base_memory = 4 * 1024 * 1024; // 4MB base
    let per_agent = 512 * 1024; // 512KB per agent
    let neural_overhead = if agent_count > 50 { 8 * 1024 * 1024 } else { 4 * 1024 * 1024 };
    
    (base_memory + (agent_count * per_agent) + neural_overhead) / 1024 // Return in KB
}