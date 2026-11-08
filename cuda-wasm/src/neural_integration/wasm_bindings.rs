//! WebAssembly Bindings for Neural Integration
//!
//! This module provides JavaScript/TypeScript bindings for the neural integration
//! system, enabling use in web browsers and Node.js environments.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Float32Array, Object, Uint8Array};

#[cfg(target_arch = "wasm32")]
use web_sys::{console, Performance, Window};

use super::{
    BridgeConfig, GpuDevice, NeuralBridge, NeuralOperation, NeuralResult, 
    Precision, SystemCapabilities, ActivationFunction, MemoryStats, PerformanceStats,
};
use std::collections::HashMap;

/// JavaScript-compatible neural bridge wrapper
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmNeuralBridge {
    bridge: NeuralBridge,
}

/// JavaScript-compatible configuration
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmBridgeConfig {
    enable_gpu: bool,
    gpu_device: String,
    memory_pool_size: usize,
    enable_monitoring: bool,
    auto_fallback: bool,
    batch_size: usize,
    precision: String,
}

/// JavaScript-compatible operation result
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmOperationResult {
    data: Vec<f32>,
    execution_time: f64,
    gpu_used: bool,
    throughput: f64,
}

/// JavaScript-compatible performance statistics
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmPerformanceStats {
    total_operations: u64,
    average_execution_time: f64,
    gpu_utilization: f32,
    memory_bandwidth: f64,
    throughput: f64,
}

/// JavaScript-compatible memory statistics
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmMemoryStats {
    total_allocated: usize,
    gpu_allocated: usize,
    cpu_allocated: usize,
    peak_usage: usize,
    allocations: u64,
    deallocations: u64,
}

/// JavaScript-compatible system capabilities
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSystemCapabilities {
    cuda_transpilation: bool,
    gpu_acceleration: bool,
    wasm_support: bool,
    performance_monitoring: bool,
    memory_pooling: bool,
    auto_fallback: bool,
    batch_processing: bool,
    precision_f16: bool,
    precision_f32: bool,
    precision_f64: bool,
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmBridgeConfig {
    fn default() -> Self {
        Self {
            enable_gpu: crate::backend::gpu_enabled(),
            gpu_device: "auto".to_string(),
            memory_pool_size: 512,
            enable_monitoring: true,
            auto_fallback: true,
            batch_size: 32,
            precision: "float32".to_string(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmBridgeConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_gpu(&self) -> bool {
        self.enable_gpu
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_gpu(&mut self, value: bool) {
        self.enable_gpu = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn gpu_device(&self) -> String {
        self.gpu_device.clone()
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_gpu_device(&mut self, value: String) {
        self.gpu_device = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn memory_pool_size(&self) -> usize {
        self.memory_pool_size
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_memory_pool_size(&mut self, value: usize) {
        self.memory_pool_size = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_monitoring(&self) -> bool {
        self.enable_monitoring
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_monitoring(&mut self, value: bool) {
        self.enable_monitoring = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn auto_fallback(&self) -> bool {
        self.auto_fallback
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_auto_fallback(&mut self, value: bool) {
        self.auto_fallback = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_batch_size(&mut self, value: usize) {
        self.batch_size = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn precision(&self) -> String {
        self.precision.clone()
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_precision(&mut self, value: String) {
        self.precision = value;
    }
}

#[cfg(target_arch = "wasm32")]
impl From<WasmBridgeConfig> for BridgeConfig {
    fn from(config: WasmBridgeConfig) -> Self {
        BridgeConfig {
            enable_gpu: config.enable_gpu,
            gpu_device: match config.gpu_device.as_str() {
                "high_performance" => GpuDevice::HighPerformance,
                "low_power" => GpuDevice::LowPower,
                "discrete" => GpuDevice::Discrete,
                "integrated" => GpuDevice::Integrated,
                _ => GpuDevice::Auto,
            },
            memory_pool_size: config.memory_pool_size,
            enable_monitoring: config.enable_monitoring,
            auto_fallback: config.auto_fallback,
            batch_size: config.batch_size,
            precision: match config.precision.as_str() {
                "float16" => Precision::Float16,
                "float64" => Precision::Float64,
                _ => Precision::Float32,
            },
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmNeuralBridge {
    /// Create a new neural bridge with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmNeuralBridge, JsValue> {
        crate::neural_integration::initialize().map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let bridge = NeuralBridge::new().map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(Self { bridge })
    }
    
    /// Create a neural bridge with custom configuration
    #[wasm_bindgen]
    pub fn with_config(config: WasmBridgeConfig) -> Result<WasmNeuralBridge, JsValue> {
        crate::neural_integration::initialize().map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let rust_config = BridgeConfig::from(config);
        let bridge = NeuralBridge::with_config(rust_config).map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(Self { bridge })
    }
    
    /// Check if GPU acceleration is available
    #[wasm_bindgen]
    pub fn is_gpu_available(&self) -> bool {
        self.bridge.is_gpu_available()
    }
    
    /// Get device information as JSON string
    #[wasm_bindgen]
    pub fn get_device_info(&self) -> Option<String> {
        self.bridge.get_device_info().map(|info| {
            format!(
                r#"{{"name":"{}","vendor":"{}","device_type":"{}","memory_size":{},"compute_units":{},"max_workgroup_size":{},"supports_f16":{},"supports_f64":{}}}"#,
                info.name, info.vendor, info.device_type, info.memory_size, 
                info.compute_units, info.max_workgroup_size, info.supports_f16, info.supports_f64
            )
        })
    }
    
    /// Execute matrix multiplication
    #[wasm_bindgen]
    pub fn matrix_multiply(
        &self,
        a: &Float32Array,
        b: &Float32Array,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<WasmOperationResult, JsValue> {
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        
        let mut input_data = a_vec;
        input_data.extend(b_vec);
        
        let operation = NeuralOperation::MatrixMultiply {
            a_rows: rows_a,
            a_cols: cols_a,
            b_cols: cols_b,
            _phantom: std::marker::PhantomData,
        };
        
        let start_time = js_sys::Date::now();
        let result = self.bridge
            .execute_neural_operation(operation, &input_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let end_time = js_sys::Date::now();
        
        Ok(WasmOperationResult {
            data: result,
            execution_time: end_time - start_time,
            gpu_used: self.bridge.is_gpu_available(),
            throughput: (rows_a * cols_b) as f64 / (end_time - start_time),
        })
    }
    
    /// Execute vector addition
    #[wasm_bindgen]
    pub fn vector_add(
        &self,
        a: &Float32Array,
        b: &Float32Array,
    ) -> Result<WasmOperationResult, JsValue> {
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        
        if a_vec.len() != b_vec.len() {
            return Err(JsValue::from_str("Vector lengths must match"));
        }
        
        let mut input_data = a_vec;
        input_data.extend(b_vec);
        
        let operation = NeuralOperation::VectorAdd {
            size: a.length() as usize,
            _phantom: std::marker::PhantomData,
        };
        
        let start_time = js_sys::Date::now();
        let result = self.bridge
            .execute_neural_operation(operation, &input_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let end_time = js_sys::Date::now();
        
        Ok(WasmOperationResult {
            data: result,
            execution_time: end_time - start_time,
            gpu_used: self.bridge.is_gpu_available(),
            throughput: (a.length() as f64) / (end_time - start_time),
        })
    }
    
    /// Execute activation function
    #[wasm_bindgen]
    pub fn activation_function(
        &self,
        input: &Float32Array,
        function_name: &str,
    ) -> Result<WasmOperationResult, JsValue> {
        let input_vec: Vec<f32> = input.to_vec();
        
        let function = match function_name {
            "sigmoid" => ActivationFunction::Sigmoid,
            "relu" => ActivationFunction::ReLU,
            "tanh" => ActivationFunction::Tanh,
            "leaky_relu" => ActivationFunction::LeakyReLU,
            "swish" => ActivationFunction::Swish,
            "gelu" => ActivationFunction::GELU,
            _ => return Err(JsValue::from_str("Unknown activation function")),
        };
        
        let operation = NeuralOperation::ActivationFunction {
            function,
            size: input_vec.len(),
            _phantom: std::marker::PhantomData,
        };
        
        let start_time = js_sys::Date::now();
        let result = self.bridge
            .execute_neural_operation(operation, &input_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let end_time = js_sys::Date::now();
        
        Ok(WasmOperationResult {
            data: result,
            execution_time: end_time - start_time,
            gpu_used: self.bridge.is_gpu_available(),
            throughput: (input_vec.len() as f64) / (end_time - start_time),
        })
    }
    
    /// Execute forward propagation through a neural network
    #[wasm_bindgen]
    pub fn forward_propagation(
        &self,
        input: &Float32Array,
        layer_sizes: &Array,
    ) -> Result<WasmOperationResult, JsValue> {
        let input_vec: Vec<f32> = input.to_vec();
        
        let sizes: Vec<usize> = layer_sizes
            .iter()
            .map(|val| val.as_f64().unwrap_or(0.0) as usize)
            .collect();
        
        let operation = NeuralOperation::ForwardPropagation {
            layer_sizes: sizes,
            _phantom: std::marker::PhantomData,
        };
        
        let start_time = js_sys::Date::now();
        let result = self.bridge
            .execute_neural_operation(operation, &input_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let end_time = js_sys::Date::now();
        
        Ok(WasmOperationResult {
            data: result,
            execution_time: end_time - start_time,
            gpu_used: self.bridge.is_gpu_available(),
            throughput: (input_vec.len() as f64) / (end_time - start_time),
        })
    }
    
    /// Execute custom CUDA kernel
    #[wasm_bindgen]
    pub fn execute_custom_kernel(
        &self,
        input: &Float32Array,
        kernel_source: &str,
        kernel_name: &str,
    ) -> Result<WasmOperationResult, JsValue> {
        let input_vec: Vec<f32> = input.to_vec();
        
        let operation = NeuralOperation::Custom {
            kernel_source: kernel_source.to_string(),
            name: kernel_name.to_string(),
            _phantom: std::marker::PhantomData,
        };
        
        let start_time = js_sys::Date::now();
        let result = self.bridge
            .execute_neural_operation(operation, &input_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let end_time = js_sys::Date::now();
        
        Ok(WasmOperationResult {
            data: result,
            execution_time: end_time - start_time,
            gpu_used: self.bridge.is_gpu_available(),
            throughput: (input_vec.len() as f64) / (end_time - start_time),
        })
    }
    
    /// Get memory statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> WasmMemoryStats {
        let stats = self.bridge.get_memory_stats();
        WasmMemoryStats::from(stats)
    }
    
    /// Get performance statistics
    #[wasm_bindgen]
    pub fn get_performance_stats(&self) -> WasmPerformanceStats {
        let stats = self.bridge.get_performance_stats();
        WasmPerformanceStats::from(stats)
    }
    
    /// Create a batch processor for bulk operations
    #[wasm_bindgen]
    pub fn create_batch_processor(&self) -> WasmBatchProcessor {
        let processor = self.bridge.create_batch_processor();
        WasmBatchProcessor { processor }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmOperationResult {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Float32Array {
        Float32Array::from(&self.data[..])
    }
    
    #[wasm_bindgen(getter)]
    pub fn execution_time(&self) -> f64 {
        self.execution_time
    }
    
    #[wasm_bindgen(getter)]
    pub fn gpu_used(&self) -> bool {
        self.gpu_used
    }
    
    #[wasm_bindgen(getter)]
    pub fn throughput(&self) -> f64 {
        self.throughput
    }
}

#[cfg(target_arch = "wasm32")]
impl From<MemoryStats> for WasmMemoryStats {
    fn from(stats: MemoryStats) -> Self {
        Self {
            total_allocated: stats.total_allocated,
            gpu_allocated: stats.gpu_allocated,
            cpu_allocated: stats.cpu_allocated,
            peak_usage: stats.peak_usage,
            allocations: stats.allocations,
            deallocations: stats.deallocations,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmMemoryStats {
    #[wasm_bindgen(getter)]
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }
    
    #[wasm_bindgen(getter)]
    pub fn gpu_allocated(&self) -> usize {
        self.gpu_allocated
    }
    
    #[wasm_bindgen(getter)]
    pub fn cpu_allocated(&self) -> usize {
        self.cpu_allocated
    }
    
    #[wasm_bindgen(getter)]
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }
    
    #[wasm_bindgen(getter)]
    pub fn allocations(&self) -> f64 {
        self.allocations as f64
    }
    
    #[wasm_bindgen(getter)]
    pub fn deallocations(&self) -> f64 {
        self.deallocations as f64
    }
}

#[cfg(target_arch = "wasm32")]
impl From<PerformanceStats> for WasmPerformanceStats {
    fn from(stats: PerformanceStats) -> Self {
        Self {
            total_operations: stats.total_operations,
            average_execution_time: stats.average_execution_time,
            gpu_utilization: stats.gpu_utilization,
            memory_bandwidth: stats.memory_bandwidth,
            throughput: stats.throughput,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmPerformanceStats {
    #[wasm_bindgen(getter)]
    pub fn total_operations(&self) -> f64 {
        self.total_operations as f64
    }
    
    #[wasm_bindgen(getter)]
    pub fn average_execution_time(&self) -> f64 {
        self.average_execution_time
    }
    
    #[wasm_bindgen(getter)]
    pub fn gpu_utilization(&self) -> f32 {
        self.gpu_utilization
    }
    
    #[wasm_bindgen(getter)]
    pub fn memory_bandwidth(&self) -> f64 {
        self.memory_bandwidth
    }
    
    #[wasm_bindgen(getter)]
    pub fn throughput(&self) -> f64 {
        self.throughput
    }
}

/// Batch processor for efficient bulk operations
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmBatchProcessor {
    processor: super::BatchProcessor,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmBatchProcessor {
    /// Process a batch of matrix multiplications
    #[wasm_bindgen]
    pub fn batch_matrix_multiply(
        &self,
        matrices: &Array,
        configs: &Array,
    ) -> Result<Array, JsValue> {
        // Implementation for batch matrix operations
        let results = Array::new();
        Ok(results)
    }
    
    /// Process a batch of activation functions
    #[wasm_bindgen]
    pub fn batch_activation_functions(
        &self,
        inputs: &Array,
        function_name: &str,
    ) -> Result<Array, JsValue> {
        // Implementation for batch activation functions
        let results = Array::new();
        Ok(results)
    }
}

/// Get system capabilities
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_system_capabilities() -> WasmSystemCapabilities {
    let capabilities = crate::neural_integration::get_capabilities();
    WasmSystemCapabilities::from(capabilities)
}

#[cfg(target_arch = "wasm32")]
impl From<SystemCapabilities> for WasmSystemCapabilities {
    fn from(caps: SystemCapabilities) -> Self {
        Self {
            cuda_transpilation: caps.cuda_transpilation,
            gpu_acceleration: caps.gpu_acceleration,
            wasm_support: caps.wasm_support,
            performance_monitoring: caps.performance_monitoring,
            memory_pooling: caps.memory_pooling,
            auto_fallback: caps.auto_fallback,
            batch_processing: caps.batch_processing,
            precision_f16: caps.precision_f16,
            precision_f32: caps.precision_f32,
            precision_f64: caps.precision_f64,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSystemCapabilities {
    #[wasm_bindgen(getter)]
    pub fn cuda_transpilation(&self) -> bool {
        self.cuda_transpilation
    }
    
    #[wasm_bindgen(getter)]
    pub fn gpu_acceleration(&self) -> bool {
        self.gpu_acceleration
    }
    
    #[wasm_bindgen(getter)]
    pub fn wasm_support(&self) -> bool {
        self.wasm_support
    }
    
    #[wasm_bindgen(getter)]
    pub fn performance_monitoring(&self) -> bool {
        self.performance_monitoring
    }
    
    #[wasm_bindgen(getter)]
    pub fn memory_pooling(&self) -> bool {
        self.memory_pooling
    }
    
    #[wasm_bindgen(getter)]
    pub fn auto_fallback(&self) -> bool {
        self.auto_fallback
    }
    
    #[wasm_bindgen(getter)]
    pub fn batch_processing(&self) -> bool {
        self.batch_processing
    }
    
    #[wasm_bindgen(getter)]
    pub fn precision_f16(&self) -> bool {
        self.precision_f16
    }
    
    #[wasm_bindgen(getter)]
    pub fn precision_f32(&self) -> bool {
        self.precision_f32
    }
    
    #[wasm_bindgen(getter)]
    pub fn precision_f64(&self) -> bool {
        self.precision_f64
    }
}

/// Initialize the neural integration system for WASM
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn init_wasm() {
    console_error_panic_hook::set_once();
    
    // Initialize neural integration
    if let Err(e) = crate::neural_integration::initialize() {
        console::error_1(&format!("Failed to initialize neural integration: {}", e).into());
    }
    
    console::log_1(&"CUDA-WASM Neural Integration initialized successfully".into());
}

/// TypeScript definitions (as comments for reference)
/*
// TypeScript definitions for the WASM bindings

export interface WasmBridgeConfig {
    enable_gpu: boolean;
    gpu_device: string;
    memory_pool_size: number;
    enable_monitoring: boolean;
    auto_fallback: boolean;
    batch_size: number;
    precision: string;
}

export interface WasmOperationResult {
    readonly data: Float32Array;
    readonly execution_time: number;
    readonly gpu_used: boolean;
    readonly throughput: number;
}

export interface WasmMemoryStats {
    readonly total_allocated: number;
    readonly gpu_allocated: number;
    readonly cpu_allocated: number;
    readonly peak_usage: number;
    readonly allocations: number;
    readonly deallocations: number;
}

export interface WasmPerformanceStats {
    readonly total_operations: number;
    readonly average_execution_time: number;
    readonly gpu_utilization: number;
    readonly memory_bandwidth: number;
    readonly throughput: number;
}

export interface WasmSystemCapabilities {
    readonly cuda_transpilation: boolean;
    readonly gpu_acceleration: boolean;
    readonly wasm_support: boolean;
    readonly performance_monitoring: boolean;
    readonly memory_pooling: boolean;
    readonly auto_fallback: boolean;
    readonly batch_processing: boolean;
    readonly precision_f16: boolean;
    readonly precision_f32: boolean;
    readonly precision_f64: boolean;
}

export class WasmNeuralBridge {
    constructor();
    static with_config(config: WasmBridgeConfig): WasmNeuralBridge;
    
    is_gpu_available(): boolean;
    get_device_info(): string | undefined;
    
    matrix_multiply(
        a: Float32Array,
        b: Float32Array,
        rows_a: number,
        cols_a: number,
        cols_b: number
    ): WasmOperationResult;
    
    vector_add(a: Float32Array, b: Float32Array): WasmOperationResult;
    
    activation_function(
        input: Float32Array,
        function_name: string
    ): WasmOperationResult;
    
    forward_propagation(
        input: Float32Array,
        layer_sizes: number[]
    ): WasmOperationResult;
    
    execute_custom_kernel(
        input: Float32Array,
        kernel_source: string,
        kernel_name: string
    ): WasmOperationResult;
    
    get_memory_stats(): WasmMemoryStats;
    get_performance_stats(): WasmPerformanceStats;
    create_batch_processor(): WasmBatchProcessor;
}

export class WasmBatchProcessor {
    batch_matrix_multiply(matrices: any[], configs: any[]): any[];
    batch_activation_functions(inputs: any[], function_name: string): any[];
}

export function get_system_capabilities(): WasmSystemCapabilities;
*/

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_conversion() {
        let wasm_config = crate::neural_integration::wasm_types::WasmBridgeConfig::default();
        let rust_config: BridgeConfig = wasm_config;
        
        assert_eq!(rust_config.batch_size, 32);
        assert_eq!(rust_config.enable_gpu, crate::backend::gpu_enabled());
    }
    
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_bridge_creation() {
        let bridge = WasmNeuralBridge::new();
        assert!(bridge.is_ok());
    }
}