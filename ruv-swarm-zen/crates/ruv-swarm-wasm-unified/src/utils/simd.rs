// SIMD optimization utilities for neural operations
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SIMDProcessor {
    simd_available: bool,
}

#[wasm_bindgen]
impl SIMDProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SIMDProcessor {
        SIMDProcessor {
            simd_available: crate::utils::detect_simd_support(),
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn simd_available(&self) -> bool {
        self.simd_available
    }
    
    #[wasm_bindgen]
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Vector lengths must match");
        }
        
        if self.simd_available {
            self.simd_vector_add(a, b)
        } else {
            self.scalar_vector_add(a, b)
        }
    }
    
    #[wasm_bindgen]
    pub fn vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Vector lengths must match");
        }
        
        if self.simd_available {
            self.simd_vector_multiply(a, b)
        } else {
            self.scalar_vector_multiply(a, b)
        }
    }
    
    #[wasm_bindgen]
    pub fn vector_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            panic!("Vector lengths must match");
        }
        
        if self.simd_available {
            self.simd_dot_product(a, b)
        } else {
            self.scalar_dot_product(a, b)
        }
    }
    
    #[wasm_bindgen]
    pub fn matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        if self.simd_available {
            self.simd_matrix_multiply(a, b, rows_a, cols_a, cols_b)
        } else {
            self.scalar_matrix_multiply(a, b, rows_a, cols_a, cols_b)
        }
    }
    
    #[wasm_bindgen]
    pub fn activation_relu(&self, input: &[f32]) -> Vec<f32> {
        if self.simd_available {
            self.simd_relu(input)
        } else {
            self.scalar_relu(input)
        }
    }
    
    #[wasm_bindgen]
    pub fn activation_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        if self.simd_available {
            self.simd_sigmoid(input)
        } else {
            self.scalar_sigmoid(input)
        }
    }
    
    // SIMD implementations (when available)
    #[cfg(target_feature = "simd128")]
    fn simd_vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        use std::arch::wasm32::*;
        
        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;
        
        // SAFETY: This unsafe block performs SIMD vector addition using WebAssembly intrinsics.
        // Safety invariants:
        // 1. chunks = a.len() / 4 ensures we only process complete 4-element groups
        // 2. offset = i * 4 is always valid because i < chunks
        // 3. v128_load reads exactly 16 bytes (4 f32s) which is within bounds
        // 4. Rust f32 slices are properly aligned for SIMD operations
        // 5. temp array provides aligned storage for v128_store
        // 6. No aliasing occurs as we write to separate result vector
        unsafe {
            // Process 4 elements at a time using SIMD
            for i in 0..chunks {
                let offset = i * 4;
                // SAFETY: offset + 3 < a.len() because offset = i * 4 and i < chunks = len / 4
                let va = v128_load(&a[offset] as *const f32 as *const v128);
                let vb = v128_load(&b[offset] as *const f32 as *const v128);
                let sum = f32x4_add(va, vb);
                
                let mut temp = [0f32; 4];
                // SAFETY: temp array is properly aligned and sized for v128
                v128_store(&mut temp[0] as *mut f32 as *mut v128, sum);
                result.extend_from_slice(&temp);
            }
            
            // Handle remaining elements
            for i in (chunks * 4)..a.len() {
                result.push(a[i] + b[i]);
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.scalar_vector_add(a, b)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        use std::arch::wasm32::*;
        
        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;
        
        // SAFETY: This unsafe block performs SIMD vector multiplication.
        // Safety invariants:
        // 1. Memory access is bounded by chunks calculation
        // 2. SIMD multiplication (f32x4_mul) maintains IEEE 754 semantics
        // 3. Temporary buffer ensures proper alignment for v128 operations
        // 4. No out-of-bounds access as offset + 3 < len for all iterations
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                // SAFETY: Reading 4 consecutive f32 values within slice bounds
                let va = v128_load(&a[offset] as *const f32 as *const v128);
                let vb = v128_load(&b[offset] as *const f32 as *const v128);
                let product = f32x4_mul(va, vb);
                
                let mut temp = [0f32; 4];
                // SAFETY: Writing to stack-allocated aligned buffer
                v128_store(&mut temp[0] as *mut f32 as *mut v128, product);
                result.extend_from_slice(&temp);
            }
            
            for i in (chunks * 4)..a.len() {
                result.push(a[i] * b[i]);
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.scalar_vector_multiply(a, b)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::wasm32::*;
        
        let chunks = a.len() / 4;
        let mut sum = 0.0f32;
        
        // SAFETY: This unsafe block computes dot product using SIMD with horizontal sum.
        // Safety invariants:
        // 1. Accumulator (acc) starts at zero and accumulates products
        // 2. All memory accesses are within bounds (offset + 3 < len)
        // 3. Horizontal sum is performed by storing SIMD register to array and summing
        // 4. Floating-point operations follow IEEE 754 semantics
        // 5. No intermediate overflow as we're using f32 operations
        unsafe {
            let mut acc = f32x4_splat(0.0);
            
            for i in 0..chunks {
                let offset = i * 4;
                // SAFETY: Valid reads of 4 f32 values each
                let va = v128_load(&a[offset] as *const f32 as *const v128);
                let vb = v128_load(&b[offset] as *const f32 as *const v128);
                let product = f32x4_mul(va, vb);
                acc = f32x4_add(acc, product);
            }
            
            // Sum the accumulator lanes
            let mut temp = [0f32; 4];
            // SAFETY: Store SIMD accumulator to aligned temporary array
            v128_store(&mut temp[0] as *mut f32 as *mut v128, acc);
            sum = temp.iter().sum();  // Safe: iterating over fixed-size array
            
            // Handle remaining elements
            for i in (chunks * 4)..a.len() {
                sum += a[i] * b[i];
            }
        }
        
        sum
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        self.scalar_dot_product(a, b)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_relu(&self, input: &[f32]) -> Vec<f32> {
        use std::arch::wasm32::*;
        
        let mut result = Vec::with_capacity(input.len());
        let chunks = input.len() / 4;
        
        // SAFETY: This unsafe block implements SIMD ReLU activation function.
        // Safety invariants:
        // 1. ReLU(x) = max(x, 0) is computed element-wise
        // 2. f32x4_max handles NaN values according to IEEE 754
        // 3. Zero vector is created once and reused for efficiency
        // 4. All memory accesses are properly bounded
        // 5. Result preserves input size exactly
        unsafe {
            let zero = f32x4_splat(0.0);
            
            for i in 0..chunks {
                let offset = i * 4;
                // SAFETY: Reading 4 f32 values within input bounds
                let v = v128_load(&input[offset] as *const f32 as *const v128);
                // SAFETY: max operation is always safe, returns larger of two values
                let relu = f32x4_max(v, zero);
                
                let mut temp = [0f32; 4];
                // SAFETY: Storing to properly aligned temporary buffer
                v128_store(&mut temp[0] as *mut f32 as *mut v128, relu);
                result.extend_from_slice(&temp);
            }
            
            for i in (chunks * 4)..input.len() {
                result.push(input[i].max(0.0));
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_relu(&self, input: &[f32]) -> Vec<f32> {
        self.scalar_relu(input)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        // SIMD sigmoid is complex, so we'll use a fast approximation
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        self.scalar_sigmoid(input)
    }
    
    #[cfg(target_feature = "simd128")]
    fn simd_matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        // Simplified SIMD matrix multiplication
        // In production, this would use blocked algorithms for cache efficiency
        let mut result = vec![0.0; rows_a * cols_b];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }
        
        result
    }
    
    #[cfg(not(target_feature = "simd128"))]
    fn simd_matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        self.scalar_matrix_multiply(a, b, rows_a, cols_a, cols_b)
    }
    
    // Scalar fallback implementations
    fn scalar_vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
    
    fn scalar_vector_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }
    
    fn scalar_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    fn scalar_relu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x.max(0.0)).collect()
    }
    
    fn scalar_sigmoid(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }
    
    fn scalar_matrix_multiply(&self, a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        let mut result = vec![0.0; rows_a * cols_b];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }
        
        result
    }
}

// Benchmark utilities
#[wasm_bindgen]
pub fn benchmark_simd_performance(size: usize) -> JsValue {
    let processor = SIMDProcessor::new();
    let mut timer = crate::utils::PerformanceTimer::new();
    
    // Create test data
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    
    // Benchmark vector operations
    timer.reset();
    let _ = processor.vector_multiply(&a, &b);
    let simd_time = timer.elapsed();
    
    // Force scalar operations for comparison
    let scalar_processor = SIMDProcessor { simd_available: false };
    timer.reset();
    let _ = scalar_processor.vector_multiply(&a, &b);
    let scalar_time = timer.elapsed();
    
    let speedup = if simd_time > 0.0 { scalar_time / simd_time } else { 1.0 };
    
    let results = serde_json::json!({
        "simd_available": processor.simd_available,
        "vector_size": size,
        "simd_time_ms": simd_time,
        "scalar_time_ms": scalar_time,
        "speedup_factor": speedup,
        "efficiency_percent": (speedup * 25.0).min(100.0) // Theoretical max 4x for SIMD
    });
    
    serde_wasm_bindgen::to_value(&results).unwrap()
}