use wasm_bindgen::prelude::*;

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct RuntimeFeatures {
    simd_available: bool,
    threads_available: bool,
    memory_limit: u64,
}

#[wasm_bindgen]
impl RuntimeFeatures {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let simd_available = detect_simd_support();
        let threads_available = detect_threads_support();
        let memory_limit = get_memory_limit();

        RuntimeFeatures {
            simd_available,
            threads_available,
            memory_limit,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn simd_available(&self) -> bool {
        self.simd_available
    }

    #[wasm_bindgen(getter)]
    pub fn threads_available(&self) -> bool {
        self.threads_available
    }

    #[wasm_bindgen(getter)]
    pub fn memory_limit(&self) -> u64 {
        self.memory_limit
    }

    #[wasm_bindgen]
    pub fn get_features_json(&self) -> String {
        format!(
            r#"{{"simd": {}, "threads": {}, "memory_limit": {}}}"#,
            self.simd_available, self.threads_available, self.memory_limit
        )
    }
}

#[cfg(target_arch = "wasm32")]
fn detect_simd_support() -> bool {
    cfg!(target_feature = "simd128")
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_simd_support() -> bool {
    cfg!(any(
        target_feature = "sse",
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "neon"
    ))
}

fn detect_threads_support() -> bool {
    // Check if SharedArrayBuffer is available (required for threads)
    #[cfg(target_arch = "wasm32")]
    {
        if !cfg!(target_feature = "atomics") {
            return false;
        }
        let window = web_sys::window();
        if let Some(window) = window {
            let global = window.as_ref();
            let shared_array_buffer =
                js_sys::Reflect::get(global, &JsValue::from_str("SharedArrayBuffer"));
            return shared_array_buffer.is_ok() && !shared_array_buffer.unwrap().is_undefined();
        }
    }
    false
}

fn get_memory_limit() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        // WASM32 has a 4GB memory limit
        4 * 1024 * 1024 * 1024
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // For non-WASM, return system memory
        0
    }
}

// JavaScript Array to Rust Vec conversion helpers
#[wasm_bindgen]
pub fn js_array_to_vec_f32(array: &JsValue) -> Result<Vec<f32>, JsValue> {
    let array = js_sys::Array::from(array);
    let length = array.length();
    let mut vec = Vec::with_capacity(length as usize);

    for i in 0..length {
        let value = array.get(i);
        if let Some(num) = value.as_f64() {
            vec.push(num as f32);
        } else {
            return Err(JsValue::from_str(&format!(
                "Invalid value at index {}: expected number",
                i
            )));
        }
    }

    Ok(vec)
}

#[wasm_bindgen]
pub fn vec_f32_to_js_array(vec: Vec<f32>) -> js_sys::Float32Array {
    js_sys::Float32Array::from(&vec[..])
}

// Performance measurement utilities
#[wasm_bindgen]
pub struct PerformanceTimer {
    start_time: f64,
    name: String,
}

#[wasm_bindgen]
impl PerformanceTimer {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String) -> Self {
        let start_time = js_sys::Date::now();
        PerformanceTimer { start_time, name }
    }

    #[wasm_bindgen]
    pub fn elapsed(&self) -> f64 {
        js_sys::Date::now() - self.start_time
    }

    #[wasm_bindgen]
    pub fn log(&self) {
        web_sys::console::log_1(&JsValue::from_str(&format!(
            "{}: {:.2}ms",
            self.name,
            self.elapsed()
        )));
    }
}

// Memory usage monitoring
#[wasm_bindgen]
pub fn get_wasm_memory_usage() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        // Return a placeholder value since direct memory access is complex in WASM
        65536 // 64KB placeholder
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

// Console logging utilities
#[wasm_bindgen]
pub fn console_log(message: &str) {
    web_sys::console::log_1(&JsValue::from_str(message));
}

#[wasm_bindgen]
pub fn console_error(message: &str) {
    web_sys::console::error_1(&JsValue::from_str(message));
}

#[wasm_bindgen]
pub fn console_warn(message: &str) {
    web_sys::console::warn_1(&JsValue::from_str(message));
}

// Error handling utilities
#[wasm_bindgen]
pub fn format_js_error(error: JsValue) -> String {
    if let Some(error_obj) = error.dyn_ref::<js_sys::Error>() {
        format!("Error: {}", error_obj.message())
    } else {
        format!("Unknown error: {:?}", error)
    }
}
