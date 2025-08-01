//! SIMD functionality tests and benchmarks
//!
//! This module provides comprehensive testing for SIMD operations
//! and performance benchmarking utilities.

#[cfg(test)]
mod tests {
    use super::super::simd_ops::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_simd_dot_product() {
        let ops = SimdVectorOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = ops.dot_product(&a, &b);
        let expected = 240.0; // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8 + 8*9

        assert!(
            (result - expected).abs() < 0.001,
            "SIMD dot product failed: got {}, expected {}",
            result,
            expected
        );
    }

    #[wasm_bindgen_test]
    fn test_simd_vector_add() {
        let ops = SimdVectorOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = ops.vector_add(&a, &b);
        let expected = vec![6.0, 8.0, 10.0, 12.0];

        assert_eq!(result.len(), expected.len());
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 0.001,
                "SIMD vector add failed at index {}: got {}, expected {}",
                i,
                r,
                e
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_simd_relu_activation() {
        let ops = SimdVectorOps::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 4.0];

        let result = ops.apply_activation(&input, "relu");
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0];

        assert_eq!(result.len(), expected.len());
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 0.001,
                "SIMD ReLU failed at index {}: got {}, expected {}",
                i,
                r,
                e
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_simd_matrix_vector_multiply() {
        let ops = SimdMatrixOps::new();

        // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = vec![1.0, 2.0, 3.0];

        let result = ops.matrix_vector_multiply(&matrix, &vector, 2, 3);
        let expected = vec![14.0, 32.0]; // [1*1+2*2+3*3, 4*1+5*2+6*3]

        assert_eq!(result.len(), expected.len());
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 0.001,
                "SIMD matrix-vector multiply failed at index {}: got {}, expected {}",
                i,
                r,
                e
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_simd_performance_benchmark() {
        let benchmark = SimdBenchmark::new();

        // Test small benchmark
        let result = benchmark.benchmark_dot_product(1000, 100);

        // Check that we get a valid JSON response
        assert!(result.contains("simd_time"));
        assert!(result.contains("scalar_time"));
        assert!(result.contains("speedup"));
    }
}

// JavaScript integration tests
use crate::{detect_simd_capabilities, SimdBenchmark, SimdMatrixOps, SimdVectorOps};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn run_simd_verification_suite() -> String {
    let mut results = Vec::new();

    // Test 1: Feature Detection
    let simd_caps = detect_simd_capabilities();
    results.push(format!("SIMD Capabilities: {}", simd_caps));

    // Test 2: Basic Operations
    let ops = SimdVectorOps::new();
    let test_vec_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let test_vec_b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let dot_result = ops.dot_product(&test_vec_a, &test_vec_b);
    results.push(format!(
        "Dot Product Test: {} (expected: 240.0)",
        dot_result
    ));

    // Test 3: Activation Functions
    let relu_input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let relu_result = ops.apply_activation(&relu_input, "relu");
    results.push(format!(
        "ReLU Test: {:?} (expected: [0.0, 0.0, 0.0, 1.0, 2.0])",
        relu_result
    ));

    // Test 4: Matrix Operations
    let matrix_ops = SimdMatrixOps::new();
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vector = vec![1.0, 2.0, 3.0];
    let mv_result = matrix_ops.matrix_vector_multiply(&matrix, &vector, 2, 3);
    results.push(format!(
        "Matrix-Vector Multiply: {:?} (expected: [14.0, 32.0])",
        mv_result
    ));

    // Test 5: Performance Benchmarks
    let benchmark = SimdBenchmark::new();
    let perf_dot = benchmark.benchmark_dot_product(1000, 100);
    let perf_relu = benchmark.benchmark_activation(1000, 100, "relu");

    results.push(format!("Dot Product Benchmark: {}", perf_dot));
    results.push(format!("ReLU Benchmark: {}", perf_relu));

    // Test 6: Neural Network Integration
    let neural_net = crate::WasmNeuralNetwork::new(&[3, 4, 2], crate::ActivationFunction::ReLU);
    // This would test the SIMD-optimized neural network run method

    results.join("\n")
}

#[wasm_bindgen]
pub fn simd_performance_report(size: usize, iterations: usize) -> String {
    let benchmark = SimdBenchmark::new();

    let dot_product_bench = benchmark.benchmark_dot_product(size, iterations);
    let relu_bench = benchmark.benchmark_activation(size, iterations, "relu");
    let sigmoid_bench = benchmark.benchmark_activation(size, iterations, "sigmoid");
    let tanh_bench = benchmark.benchmark_activation(size, iterations, "tanh");

    format!(
        r#"{{
            "test_config": {{
                "vector_size": {},
                "iterations": {}
            }},
            "benchmarks": {{
                "dot_product": {},
                "relu": {},
                "sigmoid": {},
                "tanh": {}
            }},
            "simd_status": "{}",
            "capabilities": {}
        }}"#,
        size,
        iterations,
        dot_product_bench,
        relu_bench,
        sigmoid_bench,
        tanh_bench,
        if crate::detect_simd_support() {
            "enabled"
        } else {
            "disabled"
        },
        detect_simd_capabilities()
    )
}

/// Comprehensive SIMD feature validation
#[wasm_bindgen]
pub fn validate_simd_implementation() -> bool {
    // Test basic SIMD operations
    let ops = SimdVectorOps::new();

    // Test 1: Dot product accuracy
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let dot_result = ops.dot_product(&a, &b);
    let expected_dot = 40.0; // 1*2 + 2*3 + 3*4 + 4*5

    if (dot_result - expected_dot).abs() > 0.001 {
        return false;
    }

    // Test 2: Vector addition accuracy
    let add_result = ops.vector_add(&a, &b);
    let expected_add = vec![3.0, 5.0, 7.0, 9.0];

    if add_result.len() != expected_add.len() {
        return false;
    }

    for (r, e) in add_result.iter().zip(expected_add.iter()) {
        if (r - e).abs() > 0.001 {
            return false;
        }
    }

    // Test 3: ReLU activation accuracy
    let relu_input = vec![-1.0, 0.0, 1.0, 2.0];
    let relu_result = ops.apply_activation(&relu_input, "relu");
    let expected_relu = vec![0.0, 0.0, 1.0, 2.0];

    if relu_result.len() != expected_relu.len() {
        return false;
    }

    for (r, e) in relu_result.iter().zip(expected_relu.iter()) {
        if (r - e).abs() > 0.001 {
            return false;
        }
    }

    // Test 4: Matrix operations
    let matrix_ops = SimdMatrixOps::new();
    let matrix = vec![1.0, 2.0, 3.0, 4.0];
    let vector = vec![1.0, 2.0];
    let mv_result = matrix_ops.matrix_vector_multiply(&matrix, &vector, 2, 2);
    let expected_mv = vec![5.0, 11.0]; // [1*1+2*2, 3*1+4*2]

    if mv_result.len() != expected_mv.len() {
        return false;
    }

    for (r, e) in mv_result.iter().zip(expected_mv.iter()) {
        if (r - e).abs() > 0.001 {
            return false;
        }
    }

    true
}
