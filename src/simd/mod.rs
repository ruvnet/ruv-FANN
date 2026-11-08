//! SIMD optimizations for neural network computations
//!
//! This module provides vectorized implementations of critical operations:
//! - Matrix multiplication with AVX2/AVX-512 support
//! - Vectorized activation functions
//! - Parallel gradient computation
//!
//! Matrix kernels use checked dimensions and runtime CPU-feature dispatch.
//! Benchmark representative shapes on the deployment CPU before choosing a path.
//! Multi-threading is available with the `parallel` feature.

pub use crate::ActivationFunction;
use num_traits::Float;
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Use AVX2 instructions if available
    pub use_avx2: bool,
    /// Use AVX-512 instructions if available
    pub use_avx512: bool,
    /// Block size for cache-friendly matrix operations
    pub block_size: usize,
    /// Number of threads for parallel operations
    pub num_threads: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            use_avx2: {
                #[cfg(target_arch = "x86_64")]
                {
                    is_x86_feature_detected!("avx2")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            },
            use_avx512: {
                #[cfg(target_arch = "x86_64")]
                {
                    is_x86_feature_detected!("avx512f")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            },
            block_size: 64, // Good balance for most L1 cache sizes
            num_threads: std::thread::available_parallelism().map_or(1, usize::from),
        }
    }
}

/// Trait for SIMD-accelerated matrix operations
pub trait SimdMatrixOps<T: Float + Send + Sync> {
    /// Perform matrix multiplication: C = A * B.
    /// Panics on overflowing dimensions or insufficient buffers before executing SIMD.
    fn matmul(&self, a: &[T], b: &[T], c: &mut [T], m: usize, n: usize, k: usize);

    /// Perform matrix-vector multiplication: y = A * x
    fn matvec(&self, a: &[T], x: &[T], y: &mut [T], m: usize, n: usize);

    /// Add bias vector to matrix rows
    fn add_bias(&self, matrix: &mut [T], bias: &[T], rows: usize, cols: usize);

    /// Apply activation function element-wise
    fn apply_activation(&self, data: &mut [T], activation: ActivationFunction);

    /// Compute activation derivatives.
    /// Sigmoid and tanh variants expect activated outputs; other functions
    /// expect pre-activation inputs, preserving the existing API convention.
    fn activation_derivatives(
        &self,
        data: &[T],
        derivatives: &mut [T],
        activation: ActivationFunction,
    );
}

/// Default leaky ReLU alpha used by SIMD kernels.
const LEAKY_RELU_ALPHA: f32 = 0.01;

/// CPU-based SIMD implementation
pub struct CpuSimdOps {
    config: SimdConfig,
}

impl CpuSimdOps {
    pub fn new(mut config: SimdConfig) -> Self {
        config.block_size = config.block_size.max(1);
        #[cfg(target_arch = "x86_64")]
        {
            // Configuration is a preference, never evidence of CPU support.
            config.use_avx2 &= is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            config.use_avx512 &= is_x86_feature_detected!("avx512f");
        }
        Self { config }
    }

    pub fn new_with_defaults() -> Self {
        Self::new(SimdConfig::default())
    }
}

impl SimdMatrixOps<f32> for CpuSimdOps {
    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert!(
            a.len() >= checked_area(m, k),
            "matmul: input A is too short"
        );
        assert!(
            b.len() >= checked_area(k, n),
            "matmul: input B is too short"
        );
        assert!(c.len() >= checked_area(m, n), "matmul: output is too short");
        if m == 0 || n == 0 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 {
                unsafe {
                    self.matmul_avx2(a, b, c, m, n, k);
                }
            } else {
                self.matmul_scalar(a, b, c, m, n, k);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.matmul_scalar(a, b, c, m, n, k);
        }
    }

    fn matvec(&self, a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
        assert!(a.len() >= checked_area(m, n), "matvec: matrix is too short");
        assert!(x.len() >= n, "matvec: input is too short");
        assert!(y.len() >= m, "matvec: output is too short");
        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 {
                unsafe {
                    self.matvec_avx2(a, x, y, m, n);
                }
            } else {
                self.matvec_scalar(a, x, y, m, n);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.matvec_scalar(a, x, y, m, n);
        }
    }

    fn add_bias(&self, matrix: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        assert!(
            matrix.len() >= checked_area(rows, cols),
            "add_bias: matrix is too short"
        );
        assert!(bias.len() >= cols, "add_bias: bias is too short");
        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 {
                unsafe {
                    self.add_bias_avx2(matrix, bias, rows, cols);
                }
            } else {
                self.add_bias_scalar(matrix, bias, rows, cols);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.add_bias_scalar(matrix, bias, rows, cols);
        }
    }

    fn apply_activation(&self, data: &mut [f32], activation: ActivationFunction) {
        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 {
                unsafe {
                    self.apply_activation_avx2(data, activation);
                }
            } else {
                self.apply_activation_scalar(data, activation);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.apply_activation_scalar(data, activation);
        }
    }

    fn activation_derivatives(
        &self,
        data: &[f32],
        derivatives: &mut [f32],
        activation: ActivationFunction,
    ) {
        assert!(
            derivatives.len() >= data.len(),
            "activation_derivatives: output is too short"
        );
        #[cfg(target_arch = "x86_64")]
        {
            if self.config.use_avx2 {
                unsafe {
                    self.activation_derivatives_avx2(data, derivatives, activation);
                }
            } else {
                self.activation_derivatives_scalar(data, derivatives, activation);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.activation_derivatives_scalar(data, derivatives, activation);
        }
    }
}

impl CpuSimdOps {
    /// Scalar fallback for matrix multiplication
    fn matmul_scalar(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        debug_assert!(
            a.len() >= m * k,
            "matmul_scalar: a.len() ({}) < m*k ({}*{}={})",
            a.len(),
            m,
            k,
            m * k
        );
        debug_assert!(
            b.len() >= k * n,
            "matmul_scalar: b.len() ({}) < k*n ({}*{}={})",
            b.len(),
            k,
            n,
            k * n
        );
        debug_assert!(
            c.len() >= m * n,
            "matmul_scalar: c.len() ({}) < m*n ({}*{}={})",
            c.len(),
            m,
            n,
            m * n
        );
        // Initialize output to zero
        c[..m * n].fill(0.0);

        // Use blocking for better cache performance
        let block_size = self.config.block_size;

        for i_block in (0..m).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                for k_block in (0..k).step_by(block_size) {
                    let i_end = i_block.saturating_add(block_size).min(m);
                    let j_end = j_block.saturating_add(block_size).min(n);
                    let k_end = k_block.saturating_add(block_size).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k_idx in k_block..k_end {
                                sum += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }

    /// AVX2 optimized matrix multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn matmul_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        debug_assert!(a.len() >= m * k, "matmul_avx2: a.len() < m*k");
        debug_assert!(b.len() >= k * n, "matmul_avx2: b.len() < k*n");
        debug_assert!(c.len() >= m * n, "matmul_avx2: c.len() < m*n");
        // Initialize output to zero
        c[..m * n].fill(0.0);

        const SIMD_WIDTH: usize = 8; // AVX2 processes 8 f32 at once
        let block_size = self.config.block_size;

        for i_block in (0..m).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                for k_block in (0..k).step_by(block_size) {
                    let i_end = i_block.saturating_add(block_size).min(m);
                    let j_end = j_block.saturating_add(block_size).min(n);
                    let k_end = k_block.saturating_add(block_size).min(k);

                    for i in i_block..i_end {
                        for j in (j_block..j_end).step_by(SIMD_WIDTH) {
                            let remaining = (j_end - j).min(SIMD_WIDTH);

                            if remaining == SIMD_WIDTH {
                                // Full SIMD vector processing
                                let mut sum_vec = _mm256_setzero_ps();

                                for k_idx in k_block..k_end {
                                    let a_val = _mm256_set1_ps(a[i * k + k_idx]);
                                    let b_ptr = b.as_ptr().add(k_idx * n + j);
                                    let b_vec = _mm256_loadu_ps(b_ptr);
                                    sum_vec = _mm256_fmadd_ps(a_val, b_vec, sum_vec);
                                }

                                // Store result
                                let c_ptr = c.as_mut_ptr().add(i * n + j);
                                let c_vec = _mm256_loadu_ps(c_ptr);
                                let result = _mm256_add_ps(c_vec, sum_vec);
                                _mm256_storeu_ps(c_ptr, result);
                            } else {
                                // Handle remaining elements with scalar code
                                for j_idx in j..(j + remaining) {
                                    let mut sum = 0.0;
                                    for k_idx in k_block..k_end {
                                        sum += a[i * k + k_idx] * b[k_idx * n + j_idx];
                                    }
                                    c[i * n + j_idx] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Scalar matrix-vector multiplication
    fn matvec_scalar(&self, a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
        debug_assert!(a.len() >= m * n, "matvec_scalar: a.len() < m*n");
        debug_assert!(x.len() >= n, "matvec_scalar: x.len() < n");
        debug_assert!(y.len() >= m, "matvec_scalar: y.len() < m");
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[i * n + j] * x[j];
            }
            y[i] = sum;
        }
    }

    /// AVX2 optimized matrix-vector multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn matvec_avx2(&self, a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
        debug_assert!(a.len() >= m * n, "matvec_avx2: a.len() < m*n");
        debug_assert!(x.len() >= n, "matvec_avx2: x.len() < n");
        debug_assert!(y.len() >= m, "matvec_avx2: y.len() < m");
        const SIMD_WIDTH: usize = 8;

        for i in 0..m {
            let mut sum_vec = _mm256_setzero_ps();

            // Process in chunks of 8
            let chunks = n / SIMD_WIDTH;
            for chunk in 0..chunks {
                let j = chunk * SIMD_WIDTH;
                let a_ptr = a.as_ptr().add(i * n + j);
                let x_ptr = x.as_ptr().add(j);

                let a_vec = _mm256_loadu_ps(a_ptr);
                let x_vec = _mm256_loadu_ps(x_ptr);

                sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);
            }

            // Horizontal sum of the vector
            let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum_vec);
            let mut sum = sum_array.iter().sum::<f32>();

            // Handle remaining elements
            for j in (chunks * SIMD_WIDTH)..n {
                sum += a[i * n + j] * x[j];
            }

            y[i] = sum;
        }
    }

    /// Scalar bias addition
    fn add_bias_scalar(&self, matrix: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        debug_assert!(
            matrix.len() >= rows * cols,
            "add_bias_scalar: matrix.len() < rows*cols"
        );
        debug_assert!(bias.len() >= cols, "add_bias_scalar: bias.len() < cols");
        for i in 0..rows {
            for j in 0..cols {
                matrix[i * cols + j] += bias[j];
            }
        }
    }

    /// AVX2 optimized bias addition
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_bias_avx2(&self, matrix: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        debug_assert!(
            matrix.len() >= rows * cols,
            "add_bias_avx2: matrix.len() < rows*cols"
        );
        debug_assert!(bias.len() >= cols, "add_bias_avx2: bias.len() < cols");
        const SIMD_WIDTH: usize = 8;

        for i in 0..rows {
            let mut j = 0;

            // Process in chunks of 8
            while j + SIMD_WIDTH <= cols {
                let matrix_ptr = matrix.as_mut_ptr().add(i * cols + j);
                let bias_ptr = bias.as_ptr().add(j);

                let matrix_vec = _mm256_loadu_ps(matrix_ptr);
                let bias_vec = _mm256_loadu_ps(bias_ptr);
                let result = _mm256_add_ps(matrix_vec, bias_vec);

                _mm256_storeu_ps(matrix_ptr, result);
                j += SIMD_WIDTH;
            }

            // Handle remaining elements
            while j < cols {
                matrix[i * cols + j] += bias[j];
                j += 1;
            }
        }
    }

    /// Scalar activation function application
    fn apply_activation_scalar(&self, data: &mut [f32], activation: ActivationFunction) {
        for value in data {
            *value = activation.apply(*value, 1.0);
        }
    }

    /// AVX2 optimized activation function application
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn apply_activation_avx2(&self, data: &mut [f32], activation: ActivationFunction) {
        const SIMD_WIDTH: usize = 8;
        let len = data.len();
        let mut i = 0;

        match activation {
            ActivationFunction::ReLU => {
                let zero = _mm256_setzero_ps();

                while i + SIMD_WIDTH <= len {
                    let ptr = data.as_mut_ptr().add(i);
                    let vec = _mm256_loadu_ps(ptr);
                    let result = _mm256_max_ps(vec, zero);
                    _mm256_storeu_ps(ptr, result);
                    i += SIMD_WIDTH;
                }
            }
            _ => {
                // For more complex functions, use scalar fallback for now
                self.apply_activation_scalar(data, activation);
                return;
            }
        }

        // Handle remaining elements
        while i < len {
            match activation {
                ActivationFunction::ReLU => {
                    data[i] = data[i].max(0.0);
                }
                _ => unreachable!(),
            }
            i += 1;
        }
    }

    /// Scalar activation derivatives
    fn activation_derivatives_scalar(
        &self,
        data: &[f32],
        derivatives: &mut [f32],
        activation: ActivationFunction,
    ) {
        debug_assert!(
            derivatives.len() >= data.len(),
            "activation_derivatives_scalar: derivatives.len() ({}) < data.len() ({})",
            derivatives.len(),
            data.len()
        );
        match activation {
            ActivationFunction::Sigmoid => {
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = x * (1.0 - x);
                }
            }
            ActivationFunction::Tanh | ActivationFunction::SigmoidSymmetric => {
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = 1.0 - x * x;
                }
            }
            ActivationFunction::ReLU => {
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = if x > 0.0 { 1.0 } else { 0.0 };
                }
            }
            ActivationFunction::LeakyRelu | ActivationFunction::ReLULeaky => {
                let alpha = LEAKY_RELU_ALPHA;
                for (i, &x) in data.iter().enumerate() {
                    derivatives[i] = if x > 0.0 { 1.0 } else { alpha };
                }
            }
            // Remaining functions use pre-activation inputs, like GELU and Swish.
            _ => {
                for (derivative, &x) in derivatives.iter_mut().zip(data) {
                    *derivative = activation.derivative(x, activation.apply(x, 1.0), 1.0);
                }
            }
        }
    }

    /// AVX2 optimized activation derivatives
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn activation_derivatives_avx2(
        &self,
        data: &[f32],
        derivatives: &mut [f32],
        activation: ActivationFunction,
    ) {
        debug_assert!(
            derivatives.len() >= data.len(),
            "activation_derivatives_avx2: derivatives.len() < data.len()"
        );
        const SIMD_WIDTH: usize = 8;
        let len = data.len();
        let mut i = 0;

        match activation {
            ActivationFunction::ReLU => {
                let zero = _mm256_setzero_ps();
                let one = _mm256_set1_ps(1.0);

                while i + SIMD_WIDTH <= len {
                    let data_ptr = data.as_ptr().add(i);
                    let deriv_ptr = derivatives.as_mut_ptr().add(i);

                    let data_vec = _mm256_loadu_ps(data_ptr);
                    let mask = _mm256_cmp_ps(data_vec, zero, _CMP_GT_OS);
                    let result = _mm256_and_ps(mask, one);

                    _mm256_storeu_ps(deriv_ptr, result);
                    i += SIMD_WIDTH;
                }
            }
            _ => {
                // For more complex functions, use scalar fallback
                self.activation_derivatives_scalar(data, derivatives, activation);
                return;
            }
        }

        // Handle remaining elements
        while i < len {
            match activation {
                ActivationFunction::ReLU => {
                    derivatives[i] = if data[i] > 0.0 { 1.0 } else { 0.0 };
                }
                _ => unreachable!(),
            }
            i += 1;
        }
    }
}

fn checked_area(rows: usize, cols: usize) -> usize {
    rows.checked_mul(cols)
        .expect("matrix dimensions overflow usize")
}

/// Parallel training operations using rayon
pub struct ParallelTraining {
    simd_ops: CpuSimdOps,
}

impl ParallelTraining {
    pub fn new() -> Self {
        Self {
            simd_ops: CpuSimdOps::new_with_defaults(),
        }
    }

    pub fn new_with_config(config: SimdConfig) -> Self {
        Self {
            simd_ops: CpuSimdOps::new(config),
        }
    }

    /// Parallel batch processing for training
    pub fn process_batch_parallel<F>(&self, inputs: &[Vec<f32>], outputs: &[Vec<f32>], processor: F)
    where
        F: Fn(&[f32], &[f32]) + Send + Sync,
    {
        assert_eq!(
            inputs.len(),
            outputs.len(),
            "batch inputs and outputs must match"
        );
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;
        #[cfg(feature = "parallel")]
        let pairs = inputs.par_iter().zip(outputs.par_iter());
        #[cfg(not(feature = "parallel"))]
        let pairs = inputs.iter().zip(outputs.iter());
        pairs.for_each(|(input, output)| {
            processor(input, output);
        });
    }

    /// Parallel gradient computation
    pub fn compute_gradients_parallel(
        &self,
        network_weights: &[Vec<f32>],
        activations: &[Vec<f32>],
        errors: &[Vec<f32>],
        gradients: &mut [Vec<f32>],
    ) {
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;
        #[cfg(feature = "parallel")]
        let layers = gradients.par_iter_mut().enumerate();
        #[cfg(not(feature = "parallel"))]
        let layers = gradients.iter_mut().enumerate();
        layers.for_each(|(layer_idx, layer_gradients)| {
            if layer_idx < network_weights.len()
                && layer_idx < activations.len()
                && layer_idx < errors.len()
            {
                self.simd_ops.matmul(
                    &errors[layer_idx],
                    &activations[layer_idx],
                    layer_gradients,
                    errors[layer_idx].len(),
                    activations[layer_idx].len(),
                    1,
                );
            }
        });
    }
}

impl Default for ParallelTraining {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_config_creation() {
        let config = SimdConfig::default();
        assert!(config.block_size > 0);
        assert!(config.num_threads > 0);
    }

    #[test]
    fn test_cpu_simd_ops_creation() {
        let ops = CpuSimdOps::new_with_defaults();
        assert!(ops.config.block_size > 0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let ops = CpuSimdOps::new_with_defaults();

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut c = vec![0.0; 4]; // 2x2 result

        ops.matmul(&a, &b, &mut c, 2, 2, 2);

        // Expected result: [19, 22, 43, 50]
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_activation() {
        let ops = CpuSimdOps::new_with_defaults();
        let mut data = vec![-1.0, 0.0, 1.0, -2.0, 3.0];

        ops.apply_activation(&mut data, ActivationFunction::ReLU);

        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_relu_derivatives() {
        let ops = CpuSimdOps::new_with_defaults();
        let data = vec![-1.0, 0.0, 1.0, -2.0, 3.0];
        let mut derivatives = vec![0.0; 5];

        ops.activation_derivatives(&data, &mut derivatives, ActivationFunction::ReLU);

        assert_eq!(derivatives, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
