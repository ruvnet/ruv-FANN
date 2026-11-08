//! SIMD-optimized mathematical operations for neural networks
//!
//! This module provides high-performance SIMD implementations of common
//! neural network operations including matrix multiplication, activation
//! functions, and vector operations.

use wasm_bindgen::prelude::*;
use wide::{f32x4, CmpGe};

/// SIMD-accelerated vector operations
#[wasm_bindgen]
pub struct SimdVectorOps;

#[wasm_bindgen]
impl SimdVectorOps {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        SimdVectorOps
    }

    /// SIMD-optimized vector dot product
    #[wasm_bindgen]
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        simd_dot_product(a, b)
    }

    /// SIMD-optimized vector addition
    #[wasm_bindgen]
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            return Vec::new();
        }

        simd_vector_add(a, b)
    }

    /// SIMD-optimized vector scaling
    #[wasm_bindgen]
    pub fn vector_scale(&self, vec: &[f32], scalar: f32) -> Vec<f32> {
        simd_vector_scale(vec, scalar)
    }

    /// SIMD-optimized activation function application
    #[wasm_bindgen]
    pub fn apply_activation(&self, vec: &[f32], activation: &str) -> Vec<f32> {
        match activation {
            "relu" => simd_relu(vec),
            "sigmoid" => simd_sigmoid(vec),
            "tanh" => simd_tanh(vec),
            "linear" => vec.to_vec(),
            _ => vec.to_vec(),
        }
    }
}

/// SIMD-accelerated matrix operations
#[wasm_bindgen]
pub struct SimdMatrixOps;

#[wasm_bindgen]
impl SimdMatrixOps {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        SimdMatrixOps
    }

    /// SIMD-optimized matrix-vector multiplication
    #[wasm_bindgen]
    pub fn matrix_vector_multiply(
        &self,
        matrix: &[f32],
        vector: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        if rows.checked_mul(cols) != Some(matrix.len()) || vector.len() != cols || rows > 1_000_000
        {
            return Vec::new();
        }

        simd_matrix_vector_multiply(matrix, vector, rows, cols)
    }

    /// SIMD-optimized matrix-matrix multiplication (small matrices)
    #[wasm_bindgen]
    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        a_rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Vec<f32> {
        if a_rows.checked_mul(a_cols) != Some(a.len())
            || a_cols.checked_mul(b_cols) != Some(b.len())
            || a_rows
                .checked_mul(b_cols)
                .is_none_or(|count| count > 1_000_000)
        {
            return Vec::new();
        }

        simd_matrix_multiply(a, b, a_rows, a_cols, b_cols)
    }
}

// SIMD implementation functions
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = f32x4::splat(0.0);
    let chunks = len / 4;

    // Process 4 elements at a time using SIMD
    for i in 0..chunks {
        let start = i * 4;
        let a_vec = f32x4::new([a[start], a[start + 1], a[start + 2], a[start + 3]]);
        let b_vec = f32x4::new([b[start], b[start + 1], b[start + 2], b[start + 3]]);
        sum += a_vec * b_vec;
    }

    // Sum up the SIMD register
    let sum_array = sum.as_array_ref();
    let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result += a[i] * b[i];
    }

    result
}

fn simd_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len();
    let mut result = Vec::with_capacity(len);
    let chunks = len / 4;

    // Process 4 elements at a time using SIMD
    for i in 0..chunks {
        let start = i * 4;
        let a_vec = f32x4::new([a[start], a[start + 1], a[start + 2], a[start + 3]]);
        let b_vec = f32x4::new([b[start], b[start + 1], b[start + 2], b[start + 3]]);
        let sum_vec = a_vec + b_vec;
        let sum_array = sum_vec.as_array_ref();
        result.extend_from_slice(sum_array);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result.push(a[i] + b[i]);
    }

    result
}

fn simd_vector_scale(vec: &[f32], scalar: f32) -> Vec<f32> {
    let len = vec.len();
    let mut result = Vec::with_capacity(len);
    let chunks = len / 4;
    let scalar_vec = f32x4::splat(scalar);

    // Process 4 elements at a time using SIMD
    for i in 0..chunks {
        let start = i * 4;
        let vec_simd = f32x4::new([vec[start], vec[start + 1], vec[start + 2], vec[start + 3]]);
        let scaled_vec = vec_simd * scalar_vec;
        let scaled_array = scaled_vec.as_array_ref();
        result.extend_from_slice(scaled_array);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result.push(vec[i] * scalar);
    }

    result
}

fn simd_relu(vec: &[f32]) -> Vec<f32> {
    let len = vec.len();
    let mut result = Vec::with_capacity(len);
    let chunks = len / 4;
    let zero_vec = f32x4::splat(0.0);

    // Process 4 elements at a time using SIMD
    for i in 0..chunks {
        let start = i * 4;
        let vec_simd = f32x4::new([vec[start], vec[start + 1], vec[start + 2], vec[start + 3]]);
        let relu_vec = vec_simd.max(zero_vec);
        let relu_array = relu_vec.as_array_ref();
        result.extend_from_slice(relu_array);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result.push(vec[i].max(0.0));
    }

    result
}

fn simd_sigmoid(vec: &[f32]) -> Vec<f32> {
    let len = vec.len();
    let mut result = Vec::with_capacity(len);
    let chunks = len / 4;
    let one_vec = f32x4::splat(1.0);

    // Process elements using SIMD (sigmoid approximation for SIMD)
    for i in 0..chunks {
        let start = i * 4;
        let vec_simd = f32x4::new([vec[start], vec[start + 1], vec[start + 2], vec[start + 3]]);

        // Fast sigmoid approximation: 1 / (1 + exp(-x))
        // Using a polynomial approximation for better SIMD performance
        let sigmoid_vec = sigmoid_simd(vec_simd);
        let sigmoid_array = sigmoid_vec.as_array_ref();
        result.extend_from_slice(sigmoid_array);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result.push(1.0 / (1.0 + (-vec[i]).exp()));
    }

    result
}

fn simd_tanh(vec: &[f32]) -> Vec<f32> {
    let len = vec.len();
    let mut result = Vec::with_capacity(len);
    let chunks = len / 4;

    // Process elements using SIMD (tanh approximation for SIMD)
    for i in 0..chunks {
        let start = i * 4;
        let vec_simd = f32x4::new([vec[start], vec[start + 1], vec[start + 2], vec[start + 3]]);

        // Fast tanh approximation for SIMD
        let tanh_vec = tanh_simd(vec_simd);
        let tanh_array = tanh_vec.as_array_ref();
        result.extend_from_slice(tanh_array);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        result.push(vec[i].tanh());
    }

    result
}

fn simd_matrix_vector_multiply(
    matrix: &[f32],
    vector: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(rows);

    for row in 0..rows {
        let row_start = row * cols;
        let row_slice = &matrix[row_start..row_start + cols];
        let dot_product = simd_dot_product(row_slice, vector);
        result.push(dot_product);
    }

    result
}

fn simd_matrix_multiply(
    a: &[f32],
    b: &[f32],
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) -> Vec<f32> {
    let mut result = vec![0.0; a_rows * b_cols];

    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0.0;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            result[i * b_cols + j] = sum;
        }
    }

    result
}

// Keep vector lanes consistent with the scalar tail. exp(-abs(x)) avoids
// positive overflow, including wide's out-of-range exponential behavior.
fn sigmoid_simd(x: f32x4) -> f32x4 {
    let exponential = (-x.abs()).exp();
    let denominator = f32x4::ONE + exponential;
    let positive = f32x4::ONE / denominator;
    let negative = exponential / denominator;
    let value = x.cmp_ge(f32x4::ZERO).blend(positive, negative);
    x.is_nan().blend(x, value)
}

fn tanh_simd(x: f32x4) -> f32x4 {
    let exponential = (-f32x4::splat(2.0) * x.abs()).exp();
    let value = ((f32x4::ONE - exponential) / (f32x4::ONE + exponential)).copysign(x);
    x.is_nan().blend(x, value)
}

/// Performance benchmarking utilities
#[wasm_bindgen]
pub struct SimdBenchmark;

#[wasm_bindgen]
impl SimdBenchmark {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        SimdBenchmark
    }

    /// Benchmark SIMD vs scalar dot product
    #[wasm_bindgen]
    pub fn benchmark_dot_product(&self, size: usize, iterations: usize) -> String {
        if size
            .checked_mul(iterations)
            .is_none_or(|work| work > 50_000_000)
            || size > 1_000_000
        {
            return serde_json::json!({"error": "Benchmark work budget exceeded"}).to_string();
        }
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.2) + 1.0).collect();

        // SIMD benchmark
        let start_time = js_sys::Date::now();
        for _ in 0..iterations {
            std::hint::black_box(simd_dot_product(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
            ));
        }
        let simd_time = js_sys::Date::now() - start_time;

        // Scalar benchmark
        let start_time = js_sys::Date::now();
        for _ in 0..iterations {
            let mut sum = 0.0f32;
            let (a, b) = (std::hint::black_box(&a), std::hint::black_box(&b));
            for i in 0..size {
                sum += a[i] * b[i];
            }
            std::hint::black_box(sum);
        }
        let scalar_time = js_sys::Date::now() - start_time;

        let speedup = (simd_time > 0.0).then(|| scalar_time / simd_time);
        serde_json::json!({"simd_time": simd_time, "scalar_time": scalar_time, "speedup": speedup})
            .to_string()
    }

    /// Benchmark SIMD vs scalar activation functions
    #[wasm_bindgen]
    pub fn benchmark_activation(&self, size: usize, iterations: usize, activation: &str) -> String {
        if size
            .checked_mul(iterations)
            .is_none_or(|work| work > 50_000_000)
            || size > 1_000_000
        {
            return serde_json::json!({"error": "Benchmark work budget exceeded"}).to_string();
        }
        let vec: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1) - 5.0).collect();

        // SIMD benchmark
        let start_time = js_sys::Date::now();
        for _ in 0..iterations {
            match activation {
                "relu" => {
                    std::hint::black_box(simd_relu(std::hint::black_box(&vec)));
                }
                "sigmoid" => {
                    std::hint::black_box(simd_sigmoid(std::hint::black_box(&vec)));
                }
                "tanh" => {
                    std::hint::black_box(simd_tanh(std::hint::black_box(&vec)));
                }
                _ => {}
            }
        }
        let simd_time = js_sys::Date::now() - start_time;

        // Scalar benchmark
        let start_time = js_sys::Date::now();
        for _ in 0..iterations {
            let vec = std::hint::black_box(&vec);
            let result: Vec<f32> = match activation {
                "relu" => vec.iter().map(|&x| x.max(0.0)).collect(),
                "sigmoid" => vec.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
                "tanh" => vec.iter().map(|&x| x.tanh()).collect(),
                _ => vec.to_vec(),
            };
            std::hint::black_box(result);
        }
        let scalar_time = js_sys::Date::now() - start_time;

        let speedup = (simd_time > 0.0).then(|| scalar_time / simd_time);
        serde_json::json!({"activation": activation, "simd_time": simd_time,
            "scalar_time": scalar_time, "speedup": speedup})
        .to_string()
    }
}

/// SIMD feature detection and runtime capabilities
#[wasm_bindgen]
pub fn detect_simd_capabilities() -> String {
    let mut capabilities = Vec::new();

    // Check compilation features
    #[cfg(target_feature = "simd128")]
    capabilities.push("\"simd128\": true");

    #[cfg(not(target_feature = "simd128"))]
    capabilities.push("\"simd128\": false");

    #[cfg(feature = "simd")]
    capabilities.push("\"feature_simd\": true");

    #[cfg(not(feature = "simd"))]
    capabilities.push("\"feature_simd\": false");

    // Runtime detection would require JS integration
    capabilities.push("\"runtime_detection\": \"requires_js_integration\"");

    format!("{{{}}}", capabilities.join(", "))
}

#[cfg(test)]
mod numeric_regressions {
    use super::*;

    #[test]
    fn vector_and_scalar_activations_agree_across_all_remainders() {
        for length in 1..=103 {
            let input: Vec<f32> = (0..length).map(|index| index as f32 * 0.2 - 10.0).collect();
            for (index, value) in simd_sigmoid(&input).iter().enumerate() {
                let expected = 1.0 / (1.0 + (-input[index]).exp());
                assert!((value - expected).abs() < 2e-6, "sigmoid {length}:{index}");
            }
            for (index, value) in simd_tanh(&input).iter().enumerate() {
                assert!(
                    (value - input[index].tanh()).abs() < 2e-6,
                    "tanh {length}:{index}"
                );
            }
        }
        let extremes = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN, -0.0];
        let values = simd_tanh(&extremes);
        assert_eq!(values[0], -1.0);
        assert_eq!(values[1], 1.0);
        assert!(values[2].is_nan());
        assert!(values[3].is_sign_negative());
    }

    #[test]
    fn matrix_dimensions_reject_integer_overflow_and_output_bombs() {
        let ops = SimdMatrixOps::new();
        assert!(ops
            .matrix_vector_multiply(&[], &[], usize::MAX, 2)
            .is_empty());
        assert!(ops.matrix_multiply(&[], &[], usize::MAX, 0, 2).is_empty());
        assert_eq!(
            ops.matrix_vector_multiply(&[1.0, 2.0, 3.0, 4.0], &[2.0, 3.0], 2, 2),
            vec![8.0, 18.0]
        );
    }
}
