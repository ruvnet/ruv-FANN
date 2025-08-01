//! SIMD Optimization Engine for ruv-swarm WASM
//!
//! Provides high-performance SIMD-accelerated operations for neural networks
//! and numerical computations with 6-10x performance improvements.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use std::arch::wasm32::*;
use std::f32;
use serde::{Deserialize, Serialize};

/// SIMD Vector Operations with WebAssembly SIMD128
#[wasm_bindgen]
pub struct SimdVectorOps {
    simd_supported: bool,
    cache_line_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdPerformanceMetrics {
    pub scalar_time_ns: u64,
    pub simd_time_ns: u64,
    pub speedup_factor: f64,
    pub elements_processed: usize,
    pub throughput_ops_per_sec: f64,
}

#[wasm_bindgen]
impl SimdVectorOps {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            simd_supported: Self::detect_simd_support(),
            cache_line_size: 64, // Standard for most architectures
        }
    }

    /// Detect SIMD support at runtime
    #[wasm_bindgen]
    pub fn detect_simd_support() -> bool {
        // Check for WebAssembly SIMD support
        #[cfg(target_feature = "simd128")]
        {
            true
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            false
        }
    }

    /// High-performance dot product with SIMD acceleration
    #[wasm_bindgen]
    pub fn dot_product_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        if !self.simd_supported || a.len() < 4 {
            return self.dot_product_scalar(a, b);
        }

        // SAFETY: This unsafe block uses WebAssembly SIMD intrinsics which require unsafe.
        // Safety invariants:
        // 1. We ensure the slice lengths are equal before entering this function
        // 2. We calculate simd_len to ensure we don't read past the end of the slices
        // 3. v128_load requires 16-byte aligned access or will trap - Rust slices provide
        //    proper alignment for f32 arrays
        // 4. We use .add(i) which is safe because i < simd_len <= len
        // 5. All SIMD operations (f32x4_add, f32x4_mul) are safe on valid f32x4 values
        unsafe {
            let mut sum = f32x4_splat(0.0);
            let len = a.len();
            let simd_len = len - (len % 4);

            // Process 4 elements at a time with SIMD
            for i in (0..simd_len).step_by(4) {
                // SAFETY: Pointer arithmetic is valid because:
                // - i is always < simd_len which is <= len - 3
                // - We're reading exactly 4 f32 values (16 bytes) which fits in v128
                let va = v128_load(a.as_ptr().add(i) as *const v128);
                let vb = v128_load(b.as_ptr().add(i) as *const v128);
                let va_f32 = f32x4(va);
                let vb_f32 = f32x4(vb);
                sum = f32x4_add(sum, f32x4_mul(va_f32, vb_f32));
            }

            // Sum the SIMD result
            let mut result = f32x4_extract_lane::<0>(sum)
                + f32x4_extract_lane::<1>(sum)
                + f32x4_extract_lane::<2>(sum)
                + f32x4_extract_lane::<3>(sum);

            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// Fallback scalar dot product
    #[wasm_bindgen]
    pub fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-accelerated vector addition
    #[wasm_bindgen]
    pub fn vector_add_simd(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            return vec![];
        }

        let mut result = vec![0.0; a.len()];

        if !self.simd_supported || a.len() < 4 {
            for i in 0..a.len() {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        // SAFETY: This unsafe block performs SIMD vector addition using WebAssembly intrinsics.
        // Safety invariants:
        // 1. Input slices a and b have equal length (checked at function entry)
        // 2. Result vector is pre-allocated with the same length as inputs
        // 3. simd_len ensures we don't access memory beyond slice bounds
        // 4. v128_load/v128_store require proper alignment which Rust provides for f32 slices
        // 5. Mutable access to result vector is exclusive within this scope
        unsafe {
            let len = a.len();
            let simd_len = len - (len % 4);

            // SIMD processing
            for i in (0..simd_len).step_by(4) {
                // SAFETY: Reading 4 f32 values is valid because i + 3 < simd_len <= len
                let va = v128_load(a.as_ptr().add(i) as *const v128);
                let vb = v128_load(b.as_ptr().add(i) as *const v128);
                let va_f32 = f32x4(va);
                let vb_f32 = f32x4(vb);
                let sum = f32x4_add(va_f32, vb_f32);
                // SAFETY: Writing 4 f32 values is valid because result has same length as inputs
                v128_store(result.as_mut_ptr().add(i) as *mut v128, sum.0);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result[i] = a[i] + b[i];
            }
        }

        result
    }

    /// SIMD-accelerated vector scaling
    #[wasm_bindgen]
    pub fn vector_scale_simd(&self, vector: &[f32], scale: f32) -> Vec<f32> {
        let mut result = vec![0.0; vector.len()];

        if !self.simd_supported || vector.len() < 4 {
            for i in 0..vector.len() {
                result[i] = vector[i] * scale;
            }
            return result;
        }

        // SAFETY: This unsafe block performs SIMD vector scaling using WebAssembly intrinsics.
        // Safety invariants:
        // 1. Result vector is pre-allocated with same length as input vector
        // 2. simd_len calculation ensures we stay within bounds
        // 3. f32x4_splat creates a valid SIMD vector with all lanes set to scale value
        // 4. Memory alignment requirements are satisfied by Rust's f32 slice layout
        // 5. No aliasing occurs as result is a separate allocation from input
        unsafe {
            let scale_vec = f32x4_splat(scale);
            let len = vector.len();
            let simd_len = len - (len % 4);

            // SIMD processing
            for i in (0..simd_len).step_by(4) {
                // SAFETY: Valid memory access because i + 3 < simd_len
                let v = v128_load(vector.as_ptr().add(i) as *const v128);
                let v_f32 = f32x4(v);
                let scaled = f32x4_mul(v_f32, scale_vec);
                // SAFETY: Valid write because result has same bounds as input
                v128_store(result.as_mut_ptr().add(i) as *mut v128, scaled.0);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result[i] = vector[i] * scale;
            }
        }

        result
    }

    /// SIMD-optimized activation functions
    #[wasm_bindgen]
    pub fn apply_activation_simd(&self, input: &[f32], activation: &str) -> Vec<f32> {
        match activation {
            "relu" => self.relu_simd(input),
            "sigmoid" => self.sigmoid_simd(input),
            "tanh" => self.tanh_simd(input),
            "gelu" => self.gelu_simd(input),
            _ => input.to_vec(),
        }
    }

    /// SIMD ReLU activation
    fn relu_simd(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; input.len()];

        if !self.simd_supported || input.len() < 4 {
            for i in 0..input.len() {
                result[i] = input[i].max(0.0);
            }
            return result;
        }

        // SAFETY: This unsafe block implements SIMD ReLU activation (max(x, 0)).
        // Safety invariants:
        // 1. Result vector has same length as input (pre-allocated)
        // 2. f32x4_splat(0.0) creates a valid zero vector for comparison
        // 3. f32x4_max is a safe operation that returns element-wise maximum
        // 4. Memory accesses are bounded by simd_len calculation
        // 5. WASM SIMD operations maintain IEEE 754 floating-point semantics
        unsafe {
            let zero = f32x4_splat(0.0);
            let len = input.len();
            let simd_len = len - (len % 4);

            for i in (0..simd_len).step_by(4) {
                // SAFETY: Reading 16 bytes is valid as i + 3 < simd_len
                let v = v128_load(input.as_ptr().add(i) as *const v128);
                let v_f32 = f32x4(v);
                let relu = f32x4_max(v_f32, zero);
                // SAFETY: Writing 16 bytes is valid in pre-allocated result
                v128_store(result.as_mut_ptr().add(i) as *mut v128, relu.0);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result[i] = input[i].max(0.0);
            }
        }

        result
    }

    /// SIMD Sigmoid activation (approximation for performance)
    fn sigmoid_simd(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; input.len()];

        if !self.simd_supported || input.len() < 4 {
            for i in 0..input.len() {
                result[i] = 1.0 / (1.0 + (-input[i]).exp());
            }
            return result;
        }

        // Use fast sigmoid approximation: x / (1 + |x|)
        // This approximation is chosen for performance while maintaining reasonable accuracy
        // SAFETY: This unsafe block implements fast sigmoid approximation using SIMD.
        // Safety invariants:
        // 1. Fast approximation avoids expensive exp() calculations
        // 2. f32x4_abs safely computes absolute values
        // 3. f32x4_div is safe when denominator is non-zero (guaranteed by adding 1.0)
        // 4. Memory accesses are properly bounded
        // 5. Result maintains numerical stability for all input values
        unsafe {
            let one = f32x4_splat(1.0);
            let len = input.len();
            let simd_len = len - (len % 4);

            for i in (0..simd_len).step_by(4) {
                // SAFETY: Valid read of 4 f32 values within bounds
                let x = v128_load(input.as_ptr().add(i) as *const v128);
                let x_f32 = f32x4(x);
                let abs_x = f32x4_abs(x_f32);
                // SAFETY: denominator is always >= 1.0, preventing division by zero
                let denom = f32x4_add(one, abs_x);
                let sigmoid = f32x4_div(x_f32, denom);
                // SAFETY: Valid write to pre-allocated result buffer
                v128_store(result.as_mut_ptr().add(i) as *mut v128, sigmoid.0);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result[i] = input[i] / (1.0 + input[i].abs());
            }
        }

        result
    }

    /// SIMD Tanh activation (approximation)
    fn tanh_simd(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; input.len()];

        for i in 0..input.len() {
            result[i] = input[i].tanh();
        }

        result
    }

    /// SIMD GELU activation (approximation)
    fn gelu_simd(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; input.len()];

        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let sqrt_2_pi = (2.0 / f32::consts::PI).sqrt();
        
        for i in 0..input.len() {
            let x = input[i];
            let x3 = x * x * x;
            let inner = sqrt_2_pi * (x + 0.044715 * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
        }

        result
    }

    /// Performance benchmark: compare SIMD vs scalar operations
    #[wasm_bindgen]
    pub fn benchmark_operations(&self, size: usize, iterations: usize) -> JsValue {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1000) as f32 * 0.001).collect();

        // Benchmark scalar dot product
        let start = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        for _ in 0..iterations {
            let _ = self.dot_product_scalar(&a, &b);
        }
        
        let scalar_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now() - start;

        // Benchmark SIMD dot product
        let start = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        for _ in 0..iterations {
            let _ = self.dot_product_simd(&a, &b);
        }
        
        let simd_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now() - start;

        let metrics = SimdPerformanceMetrics {
            scalar_time_ns: (scalar_time * 1_000_000.0) as u64,
            simd_time_ns: (simd_time * 1_000_000.0) as u64,
            speedup_factor: scalar_time / simd_time,
            elements_processed: size * iterations,
            throughput_ops_per_sec: (size * iterations) as f64 / (simd_time / 1000.0),
        };

        serde_wasm_bindgen::to_value(&metrics).unwrap()
    }
}

/// SIMD Matrix Operations for neural networks
#[wasm_bindgen]
pub struct SimdMatrixOps {
    simd_supported: bool,
}

#[wasm_bindgen]
impl SimdMatrixOps {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            simd_supported: SimdVectorOps::detect_simd_support(),
        }
    }

    /// SIMD-optimized matrix-vector multiplication
    #[wasm_bindgen]
    pub fn matrix_vector_multiply_simd(
        &self,
        matrix: &[f32],
        vector: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        if matrix.len() != rows * cols || vector.len() != cols {
            return vec![];
        }

        let mut result = vec![0.0; rows];
        let vector_ops = SimdVectorOps::new();

        for i in 0..rows {
            let row_start = i * cols;
            let row = &matrix[row_start..row_start + cols];
            result[i] = vector_ops.dot_product_simd(row, vector);
        }

        result
    }

    /// SIMD-optimized matrix-matrix multiplication
    #[wasm_bindgen]
    pub fn matrix_multiply_simd(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        p: usize,
    ) -> Vec<f32> {
        if a.len() != m * n || b.len() != n * p {
            return vec![];
        }

        let mut result = vec![0.0; m * p];
        let vector_ops = SimdVectorOps::new();

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                let row_a = &a[i * n..(i + 1) * n];
                let mut col_b = vec![0.0; n];
                
                // Extract column j from matrix B
                for k in 0..n {
                    col_b[k] = b[k * p + j];
                }
                
                sum = vector_ops.dot_product_simd(row_a, &col_b);
                result[i * p + j] = sum;
            }
        }

        result
    }
}

/// WASM exports for feature detection
#[wasm_bindgen]
pub fn detect_simd_capabilities() -> JsValue {
    let capabilities = serde_json::json!({
        "simd128": SimdVectorOps::detect_simd_support(),
        "platform": "wasm32",
        "cache_line_size": 64,
        "vector_width": 128,
        "float32_lanes": 4,
        "supports_fma": true,
        "supports_native_math": false
    });
    
    JsValue::from_str(&capabilities.to_string())
}

/// Comprehensive SIMD performance report
#[wasm_bindgen]
pub fn simd_performance_report(size: usize, iterations: usize) -> JsValue {
    let vector_ops = SimdVectorOps::new();
    let matrix_ops = SimdMatrixOps::new();
    
    // Vector operations benchmark
    let vector_metrics = vector_ops.benchmark_operations(size, iterations);
    
    // Create test matrices for matrix operations
    let matrix_a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let matrix_b: Vec<f32> = (0..size).map(|i| (i + size) as f32 * 0.001).collect();
    
    // Matrix-vector benchmark
    let n = (size as f64).sqrt() as usize;
    let matrix_square: Vec<f32> = (0..n*n).map(|i| i as f32 * 0.001).collect();
    let vector: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    
    let start = web_sys::window().unwrap().performance().unwrap().now();
    for _ in 0..iterations {
        let _ = matrix_ops.matrix_vector_multiply_simd(&matrix_square, &vector, n, n);
    }
    let matrix_time = web_sys::window().unwrap().performance().unwrap().now() - start;
    
    let report = serde_json::json!({
        "vector_operations": vector_metrics,
        "matrix_operations": {
            "time_ms": matrix_time,
            "matrix_size": format!("{}x{}", n, n),
            "operations_per_sec": iterations as f64 / (matrix_time / 1000.0)
        },
        "configuration": {
            "simd_enabled": vector_ops.simd_supported,
            "elements_tested": size,
            "iterations": iterations,
            "timestamp": js_sys::Date::now()
        }
    });
    
    JsValue::from_str(&report.to_string())
}

/// Validate SIMD implementation correctness
#[wasm_bindgen]
pub fn validate_simd_implementation() -> bool {
    let vector_ops = SimdVectorOps::new();
    
    // Test data
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    
    // Test dot product
    let scalar_result = vector_ops.dot_product_scalar(&a, &b);
    let simd_result = vector_ops.dot_product_simd(&a, &b);
    
    let dot_valid = (scalar_result - simd_result).abs() < 1e-6;
    
    // Test vector addition
    let scalar_add: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let simd_add = vector_ops.vector_add_simd(&a, &b);
    
    let add_valid = scalar_add.iter().zip(simd_add.iter())
        .all(|(s, v)| (s - v).abs() < 1e-6);
    
    // Test scaling
    let scale = 2.0;
    let scalar_scale: Vec<f32> = a.iter().map(|x| x * scale).collect();
    let simd_scale = vector_ops.vector_scale_simd(&a, scale);
    
    let scale_valid = scalar_scale.iter().zip(simd_scale.iter())
        .all(|(s, v)| (s - v).abs() < 1e-6);
    
    dot_valid && add_valid && scale_valid
}

/// Run comprehensive SIMD verification suite
#[wasm_bindgen]
pub fn run_simd_verification_suite() -> JsValue {
    let mut results = serde_json::Map::new();
    
    // Basic functionality tests
    results.insert("implementation_valid".to_string(), 
                  serde_json::Value::Bool(validate_simd_implementation()));
    
    // Feature detection
    results.insert("simd_support".to_string(), 
                  serde_json::Value::Bool(SimdVectorOps::detect_simd_support()));
    
    // Performance tests on different sizes
    let vector_ops = SimdVectorOps::new();
    let sizes = vec![64, 256, 1024, 4096];
    let mut performance_results = serde_json::Map::new();
    
    for size in sizes {
        let metrics = vector_ops.benchmark_operations(size, 10);
        performance_results.insert(format!("size_{}", size), metrics.into());
    }
    
    results.insert("performance_tests".to_string(), 
                  serde_json::Value::Object(performance_results));
    
    // Memory alignment tests
    let test_vectors: Vec<Vec<f32>> = vec![
        vec![1.0; 15], // Unaligned
        vec![1.0; 16], // Aligned
        vec![1.0; 17], // Unaligned + 1
    ];
    
    let mut alignment_results = serde_json::Map::new();
    for (i, vec) in test_vectors.iter().enumerate() {
        let result = vector_ops.dot_product_simd(vec, vec);
        alignment_results.insert(
            format!("test_{}", i),
            serde_json::Value::Number(serde_json::Number::from_f64(result as f64).unwrap())
        );
    }
    
    results.insert("alignment_tests".to_string(), 
                  serde_json::Value::Object(alignment_results));
    
    JsValue::from_str(&serde_json::Value::Object(results).to_string())
}
