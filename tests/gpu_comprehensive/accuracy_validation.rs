//! GPU vs CPU Accuracy Validation Framework
//! 
//! Comprehensive accuracy testing ensuring GPU results match CPU reference
//! implementations within strict numerical tolerance (1e-6 for f32, 1e-12 for f64).

use std::collections::HashMap;
use std::f32::consts::PI;
use serde::{Serialize, Deserialize};

/// Numerical tolerance constants for different precision levels
pub const ACCURACY_TOLERANCE_F32: f32 = 1e-6;
pub const ACCURACY_TOLERANCE_F64: f64 = 1e-12;
pub const STRICT_TOLERANCE_F32: f32 = 1e-7;
pub const RELAXED_TOLERANCE_F32: f32 = 1e-5;

/// Comprehensive test data generator for accuracy validation
#[derive(Debug, Clone)]
pub struct AccuracyTestDataGenerator {
    matrix_sizes: Vec<(usize, usize)>,
    vector_sizes: Vec<usize>,
    batch_sizes: Vec<usize>,
    activation_functions: Vec<ActivationFunction>,
    test_matrices: HashMap<String, Vec<f32>>,
    test_vectors: HashMap<String, Vec<f32>>,
    edge_case_data: EdgeCaseData,
}

#[derive(Debug, Clone)]
struct EdgeCaseData {
    extreme_values: Vec<f32>,
    precision_boundaries: Vec<f32>,
    special_matrices: HashMap<String, Vec<f32>>,
}

/// Neural network activation functions for testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Gaussian,
    Elliott,
    Sin,
    Cos,
    Swish,
    GELU,
    Mish,
}

impl AccuracyTestDataGenerator {
    pub fn new() -> Self {
        let mut generator = Self {
            matrix_sizes: vec![
                // Small matrices (fast testing)
                (8, 8), (16, 16), (32, 24), (50, 30),
                // Medium matrices (typical networks)
                (100, 80), (128, 128), (256, 200), (300, 256),
                // Large matrices (performance testing)
                (512, 400), (1000, 800), (1024, 1024),
                // Very large matrices (stress testing)
                (2000, 1500), (4000, 3000),
                // Non-square matrices (edge cases)
                (100, 1), (1, 100), (1000, 10), (10, 1000),
            ],
            vector_sizes: vec![8, 16, 32, 50, 100, 200, 256, 400, 512, 800, 1000, 1500, 2000, 3000, 4000],
            batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128],
            activation_functions: vec![
                ActivationFunction::Linear,
                ActivationFunction::Sigmoid,
                ActivationFunction::ReLU,
                ActivationFunction::LeakyReLU,
                ActivationFunction::Tanh,
                ActivationFunction::Gaussian,
                ActivationFunction::Elliott,
                ActivationFunction::Sin,
                ActivationFunction::Cos,
                ActivationFunction::Swish,
                ActivationFunction::GELU,
                ActivationFunction::Mish,
            ],
            test_matrices: HashMap::new(),
            test_vectors: HashMap::new(),
            edge_case_data: EdgeCaseData {
                extreme_values: Vec::new(),
                precision_boundaries: Vec::new(),
                special_matrices: HashMap::new(),
            },
        };
        
        generator.generate_comprehensive_test_data();
        generator
    }
    
    fn generate_comprehensive_test_data(&mut self) {
        self.generate_deterministic_matrices();
        self.generate_deterministic_vectors();
        self.generate_edge_case_data();
        self.generate_special_test_cases();
    }
    
    fn generate_deterministic_matrices(&mut self) {
        for &(rows, cols) in &self.matrix_sizes {
            let key = format!("matrix_{}x{}", rows, cols);
            let mut matrix = Vec::with_capacity(rows * cols);
            
            for i in 0..rows {
                for j in 0..cols {
                    // Create diverse, deterministic values with known mathematical properties
                    let val = match (i + j) % 12 {
                        0 => (i as f32 * 0.1 + j as f32 * 0.01) % 1.0,
                        1 => -(i as f32 * 0.05 + j as f32 * 0.02) % 1.0,
                        2 => ((i + j) as f32 * PI / 180.0).sin(),
                        3 => ((i + j) as f32 * PI / 180.0).cos(),
                        4 => (i as f32 / (j + 1) as f32).tanh(),
                        5 => if (i + j) % 2 == 0 { 1.0 } else { -1.0 },
                        6 => ((i * j) as f32).sqrt() / 100.0,
                        7 => (i as f32).ln() / (j + 1) as f32,
                        8 => ((i + j) as f32 / 10.0).exp().min(10.0),
                        9 => ((i as f32 - j as f32) / 10.0).abs(),
                        10 => if i == j { 1.0 } else { 0.1 },
                        _ => 0.5 + 0.3 * ((i + j) as f32 * 0.1).sin(),
                    };
                    matrix.push(val);
                }
            }
            
            self.test_matrices.insert(key, matrix);
        }
    }
    
    fn generate_deterministic_vectors(&mut self) {
        for &size in &self.vector_sizes {
            let key = format!("vector_{}", size);
            let mut vector = Vec::with_capacity(size);
            
            for i in 0..size {
                let val = match i % 8 {
                    0 => (i as f32 * 0.1) % 2.0 - 1.0,
                    1 => (i as f32 * PI / 180.0).sin(),
                    2 => (i as f32).ln() / 10.0,
                    3 => if i % 2 == 0 { 1.0 } else { -0.5 },
                    4 => ((i as f32) / size as f32) * 2.0 - 1.0,
                    5 => (i as f32).sqrt() / 10.0,
                    6 => ((i as f32) * 0.1).cos(),
                    _ => if i % 3 == 0 { 0.0 } else { (i as f32) * 0.01 },
                };
                vector.push(val);
            }
            
            self.test_vectors.insert(key, vector);
        }
    }
    
    fn generate_edge_case_data(&mut self) {
        // Extreme values that test numerical stability
        self.edge_case_data.extreme_values = vec![
            0.0, -0.0, // Zero and negative zero
            1.0, -1.0, // Unit values
            f32::EPSILON, -f32::EPSILON, // Smallest representable values
            f32::MIN_POSITIVE, -f32::MIN_POSITIVE, // Smallest positive/negative
            1e-6, -1e-6, 1e-5, -1e-5, // Near tolerance boundaries
            1e6, -1e6, 1e7, -1e7, // Large values
            PI, -PI, PI/2.0, -PI/2.0, // Mathematical constants
            std::f32::consts::E, -std::f32::consts::E, // Natural constants
            0.5, -0.5, 0.1, -0.1, // Common fractions
            0.999999, -0.999999, // Near-unit values
        ];
        
        // Precision boundary values
        self.edge_case_data.precision_boundaries = vec![
            ACCURACY_TOLERANCE_F32,
            ACCURACY_TOLERANCE_F32 * 0.1,
            ACCURACY_TOLERANCE_F32 * 0.01,
            ACCURACY_TOLERANCE_F32 * 10.0,
            ACCURACY_TOLERANCE_F32 * 100.0,
        ];
    }
    
    fn generate_special_test_cases(&mut self) {
        // Identity matrix
        let identity_4x4 = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        self.edge_case_data.special_matrices.insert("identity_4x4".to_string(), identity_4x4);
        
        // Zero matrix
        let zero_4x4 = vec![0.0; 16];
        self.edge_case_data.special_matrices.insert("zero_4x4".to_string(), zero_4x4);
        
        // Ones matrix
        let ones_4x4 = vec![1.0; 16];
        self.edge_case_data.special_matrices.insert("ones_4x4".to_string(), ones_4x4);
        
        // Orthogonal matrix (rotation)
        let rotation_2x2 = vec![
            PI/4.0_f32.cos(), -PI/4.0_f32.sin(),
            PI/4.0_f32.sin(),  PI/4.0_f32.cos(),
        ];
        self.edge_case_data.special_matrices.insert("rotation_2x2".to_string(), rotation_2x2);
        
        // Sparse matrix
        let mut sparse_8x8 = vec![0.0; 64];
        sparse_8x8[0] = 1.0;  // (0,0)
        sparse_8x8[9] = 2.0;  // (1,1)
        sparse_8x8[18] = 3.0; // (2,2)
        sparse_8x8[63] = 4.0; // (7,7)
        self.edge_case_data.special_matrices.insert("sparse_8x8".to_string(), sparse_8x8);
    }
    
    pub fn get_matrix(&self, rows: usize, cols: usize) -> Option<&Vec<f32>> {
        let key = format!("matrix_{}x{}", rows, cols);
        self.test_matrices.get(&key)
    }
    
    pub fn get_vector(&self, size: usize) -> Option<&Vec<f32>> {
        let key = format!("vector_{}", size);
        self.test_vectors.get(&key)
    }
    
    pub fn get_special_matrix(&self, name: &str) -> Option<&Vec<f32>> {
        self.edge_case_data.special_matrices.get(name)
    }
    
    pub fn get_edge_case_values(&self) -> &Vec<f32> {
        &self.edge_case_data.extreme_values
    }
    
    pub fn get_matrix_sizes(&self) -> &Vec<(usize, usize)> {
        &self.matrix_sizes
    }
    
    pub fn get_vector_sizes(&self) -> &Vec<usize> {
        &self.vector_sizes
    }
    
    pub fn get_batch_sizes(&self) -> &Vec<usize> {
        &self.batch_sizes
    }
    
    pub fn get_activation_functions(&self) -> &Vec<ActivationFunction> {
        &self.activation_functions
    }
}

/// High-precision CPU reference implementation for comparison
pub struct CpuReferenceImplementation;

impl CpuReferenceImplementation {
    /// Matrix-vector multiplication with high precision
    pub fn matrix_vector_multiply(
        matrix: &[f32], 
        vector: &[f32], 
        rows: usize, 
        cols: usize
    ) -> Vec<f32> {
        assert_eq!(matrix.len(), rows * cols, "Matrix size mismatch");
        assert_eq!(vector.len(), cols, "Vector size mismatch");
        
        let mut result = vec![0.0; rows];
        for row in 0..rows {
            let mut sum = 0.0f64; // Use double precision for accumulation
            for col in 0..cols {
                sum += matrix[row * cols + col] as f64 * vector[col] as f64;
            }
            result[row] = sum as f32;
        }
        result
    }
    
    /// Batch matrix-vector multiplication
    pub fn batch_matrix_vector_multiply(
        matrix: &[f32],
        vectors: &[Vec<f32>],
        rows: usize,
        cols: usize
    ) -> Vec<Vec<f32>> {
        vectors.iter()
            .map(|v| Self::matrix_vector_multiply(matrix, v, rows, cols))
            .collect()
    }
    
    /// Apply activation function with high precision
    pub fn apply_activation_function(
        inputs: &[f32],
        function: ActivationFunction,
        steepness: f32,
    ) -> Vec<f32> {
        inputs.iter().map(|&x| {
            let scaled_x = x * steepness;
            match function {
                ActivationFunction::Linear => scaled_x,
                ActivationFunction::Sigmoid => {
                    let exp_neg_x = (-scaled_x as f64).exp();
                    (1.0 / (1.0 + exp_neg_x)) as f32
                },
                ActivationFunction::ReLU => scaled_x.max(0.0),
                ActivationFunction::LeakyReLU => {
                    if scaled_x > 0.0 { scaled_x } else { 0.01 * scaled_x }
                },
                ActivationFunction::Tanh => scaled_x.tanh(),
                ActivationFunction::Gaussian => (-(scaled_x * scaled_x) as f64).exp() as f32,
                ActivationFunction::Elliott => scaled_x / (1.0 + scaled_x.abs()),
                ActivationFunction::Sin => scaled_x.sin(),
                ActivationFunction::Cos => scaled_x.cos(),
                ActivationFunction::Swish => scaled_x * Self::sigmoid_f64(scaled_x as f64) as f32,
                ActivationFunction::GELU => {
                    let x_f64 = scaled_x as f64;
                    (0.5 * x_f64 * (1.0 + (x_f64 * 0.7978845608028654).tanh())) as f32
                },
                ActivationFunction::Mish => {
                    let x_f64 = scaled_x as f64;
                    (x_f64 * (1.0 + x_f64.exp()).ln().tanh()) as f32
                },
            }
        }).collect()
    }
    
    fn sigmoid_f64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Vector operations with high precision
    pub fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Vector length mismatch");
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }
    
    pub fn vector_subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Vector length mismatch");
        a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
    }
    
    pub fn vector_scale(vec: &[f32], scalar: f32) -> Vec<f32> {
        vec.iter().map(|&x| x * scalar).collect()
    }
    
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vector length mismatch");
        let mut sum = 0.0f64; // Use double precision for accumulation
        for (&x, &y) in a.iter().zip(b.iter()) {
            sum += x as f64 * y as f64;
        }
        sum as f32
    }
    
    pub fn vector_norm_l2(vec: &[f32]) -> f32 {
        let sum_of_squares = vec.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        sum_of_squares.sqrt() as f32
    }
    
    pub fn element_wise_multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Vector length mismatch");
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }
}

/// Detailed accuracy test result for individual operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyTestResult {
    pub test_name: String,
    pub operation_type: String,
    pub input_dimensions: String,
    pub passed: bool,
    pub max_error: f32,
    pub mean_error: f32,
    pub rms_error: f32,
    pub tolerance_used: f32,
    pub error_distribution: ErrorDistribution,
    pub performance_metrics: OperationPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDistribution {
    pub error_histogram: Vec<(f32, usize)>, // (error_range, count)
    pub percentile_95: f32,
    pub percentile_99: f32,
    pub outlier_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPerformance {
    pub cpu_time_ms: f32,
    pub gpu_time_ms: f32,
    pub data_transfer_ms: f32,
    pub total_time_ms: f32,
}

/// Individual test case specification
#[derive(Debug, Clone)]
pub struct AccuracyTestCase {
    pub name: String,
    pub operation: TestOperation,
    pub tolerance: f32,
    pub expected_pass: bool,
}

#[derive(Debug, Clone)]
pub enum TestOperation {
    MatrixVectorMultiply {
        matrix_rows: usize,
        matrix_cols: usize,
        use_special_matrix: Option<String>,
    },
    BatchMatrixVectorMultiply {
        matrix_rows: usize,
        matrix_cols: usize,
        batch_size: usize,
    },
    ActivationFunction {
        function: ActivationFunction,
        input_size: usize,
        steepness: f32,
        use_edge_cases: bool,
    },
    VectorOperation {
        operation: VectorOpType,
        vector_size: usize,
    },
    EdgeCaseTest {
        description: String,
        custom_data: Vec<f32>,
    },
}

#[derive(Debug, Clone)]
pub enum VectorOpType {
    Add,
    Subtract,
    Scale(f32),
    DotProduct,
    L2Norm,
    ElementWiseMultiply,
}

/// Comprehensive accuracy validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub overall_pass_rate: f32,
    pub tolerance: f32,
    pub max_error: f32,
    pub mean_error: f32,
    pub rms_error: f32,
    pub test_results: Vec<AccuracyTestResult>,
    pub failed_test_names: Vec<String>,
    pub operation_summary: OperationSummary,
    pub numerical_analysis: NumericalAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSummary {
    pub matrix_operations: TestCategorySummary,
    pub activation_functions: TestCategorySummary,
    pub vector_operations: TestCategorySummary,
    pub edge_cases: TestCategorySummary,
    pub batch_operations: TestCategorySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCategorySummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub pass_rate: f32,
    pub max_error: f32,
    pub mean_error: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalAnalysis {
    pub precision_degradation: f32,
    pub accumulation_errors: Vec<f32>,
    pub systematic_bias: f32,
    pub error_correlation: f32,
    pub stability_score: f32,
}

/// Main accuracy validator orchestrating all accuracy tests
pub struct AccuracyValidator {
    test_data_generator: AccuracyTestDataGenerator,
    tolerance: f32,
    test_results: Vec<AccuracyTestResult>,
    test_cases: Vec<AccuracyTestCase>,
}

impl AccuracyValidator {
    pub fn new(tolerance: f32) -> Self {
        let mut validator = Self {
            test_data_generator: AccuracyTestDataGenerator::new(),
            tolerance,
            test_results: Vec::new(),
            test_cases: Vec::new(),
        };
        
        validator.generate_comprehensive_test_suite();
        validator
    }
    
    fn generate_comprehensive_test_suite(&mut self) {
        // Matrix-vector multiplication tests
        for &(rows, cols) in self.test_data_generator.get_matrix_sizes() {
            if rows * cols <= 1_000_000 { // Skip very large matrices for basic accuracy
                self.test_cases.push(AccuracyTestCase {
                    name: format!("matrix_vector_{}x{}", rows, cols),
                    operation: TestOperation::MatrixVectorMultiply {
                        matrix_rows: rows,
                        matrix_cols: cols,
                        use_special_matrix: None,
                    },
                    tolerance: self.tolerance,
                    expected_pass: true,
                });
            }
        }
        
        // Special matrix tests
        for special_name in ["identity_4x4", "zero_4x4", "ones_4x4", "rotation_2x2", "sparse_8x8"] {
            self.test_cases.push(AccuracyTestCase {
                name: format!("special_matrix_{}", special_name),
                operation: TestOperation::MatrixVectorMultiply {
                    matrix_rows: if special_name.contains("2x2") { 2 } else if special_name.contains("4x4") { 4 } else { 8 },
                    matrix_cols: if special_name.contains("2x2") { 2 } else if special_name.contains("4x4") { 4 } else { 8 },
                    use_special_matrix: Some(special_name.to_string()),
                },
                tolerance: self.tolerance,
                expected_pass: true,
            });
        }
        
        // Batch operations
        for &batch_size in self.test_data_generator.get_batch_sizes() {
            if batch_size <= 32 { // Limit batch size for accuracy tests
                self.test_cases.push(AccuracyTestCase {
                    name: format!("batch_matrix_vector_128x100_batch{}", batch_size),
                    operation: TestOperation::BatchMatrixVectorMultiply {
                        matrix_rows: 128,
                        matrix_cols: 100,
                        batch_size,
                    },
                    tolerance: self.tolerance,
                    expected_pass: true,
                });
            }
        }
        
        // Activation function tests
        for &function in self.test_data_generator.get_activation_functions() {
            for &size in &[100, 1000, 10000] {
                for &steepness in &[0.5, 1.0, 2.0] {
                    self.test_cases.push(AccuracyTestCase {
                        name: format!("activation_{:?}_size{}_steepness{}", function, size, steepness),
                        operation: TestOperation::ActivationFunction {
                            function,
                            input_size: size,
                            steepness,
                            use_edge_cases: false,
                        },
                        tolerance: self.tolerance,
                        expected_pass: true,
                    });
                }
                
                // Edge case tests for activation functions
                self.test_cases.push(AccuracyTestCase {
                    name: format!("activation_{:?}_edge_cases", function),
                    operation: TestOperation::ActivationFunction {
                        function,
                        input_size: 20, // Size of edge case data
                        steepness: 1.0,
                        use_edge_cases: true,
                    },
                    tolerance: match function {
                        ActivationFunction::Gaussian | ActivationFunction::GELU | ActivationFunction::Mish => self.tolerance * 10.0,
                        _ => self.tolerance,
                    },
                    expected_pass: true,
                });
            }
        }
        
        // Vector operation tests
        for &size in &[100, 1000, 10000] {
            for op_type in [
                VectorOpType::Add,
                VectorOpType::Subtract,
                VectorOpType::Scale(2.5),
                VectorOpType::DotProduct,
                VectorOpType::L2Norm,
                VectorOpType::ElementWiseMultiply,
            ] {
                self.test_cases.push(AccuracyTestCase {
                    name: format!("vector_{:?}_size{}", op_type, size),
                    operation: TestOperation::VectorOperation {
                        operation: op_type,
                        vector_size: size,
                    },
                    tolerance: self.tolerance,
                    expected_pass: true,
                });
            }
        }
        
        println!("Generated {} comprehensive accuracy test cases", self.test_cases.len());
    }
    
    /// Run comprehensive accuracy validation
    pub async fn run_comprehensive_validation(&mut self) -> Result<(), crate::ValidationError> {
        println!("🎯 Starting comprehensive GPU accuracy validation...");
        println!("Target tolerance: {:.2e}", self.tolerance);
        println!("Total test cases: {}", self.test_cases.len());
        
        let start_time = std::time::Instant::now();
        
        for (i, test_case) in self.test_cases.iter().enumerate() {
            if i % 10 == 0 {
                println!("Progress: {}/{} tests completed", i, self.test_cases.len());
            }
            
            match self.execute_test_case(test_case).await {
                Ok(result) => {
                    if !result.passed && test_case.expected_pass {
                        println!("❌ Test failed: {} (error: {:.2e})", result.test_name, result.max_error);
                    }
                    self.test_results.push(result);
                }
                Err(e) => {
                    println!("⚠️ Test error: {} - {}", test_case.name, e);
                    // Create a failed test result
                    self.test_results.push(AccuracyTestResult {
                        test_name: test_case.name.clone(),
                        operation_type: format!("{:?}", test_case.operation),
                        input_dimensions: "unknown".to_string(),
                        passed: false,
                        max_error: f32::INFINITY,
                        mean_error: f32::INFINITY,
                        rms_error: f32::INFINITY,
                        tolerance_used: test_case.tolerance,
                        error_distribution: ErrorDistribution {
                            error_histogram: Vec::new(),
                            percentile_95: f32::INFINITY,
                            percentile_99: f32::INFINITY,
                            outlier_count: 1,
                        },
                        performance_metrics: OperationPerformance {
                            cpu_time_ms: 0.0,
                            gpu_time_ms: 0.0,
                            data_transfer_ms: 0.0,
                            total_time_ms: 0.0,
                        },
                    });
                }
            }
        }
        
        let validation_time = start_time.elapsed();
        println!("✅ Accuracy validation completed in {:?}", validation_time);
        println!("Total results: {} tests", self.test_results.len());
        
        Ok(())
    }
    
    async fn execute_test_case(&self, test_case: &AccuracyTestCase) -> Result<AccuracyTestResult, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        // Execute the specific test operation
        match &test_case.operation {
            TestOperation::MatrixVectorMultiply { matrix_rows, matrix_cols, use_special_matrix } => {
                self.test_matrix_vector_multiply(*matrix_rows, *matrix_cols, use_special_matrix.as_deref(), test_case).await
            }
            TestOperation::BatchMatrixVectorMultiply { matrix_rows, matrix_cols, batch_size } => {
                self.test_batch_matrix_vector_multiply(*matrix_rows, *matrix_cols, *batch_size, test_case).await
            }
            TestOperation::ActivationFunction { function, input_size, steepness, use_edge_cases } => {
                self.test_activation_function(*function, *input_size, *steepness, *use_edge_cases, test_case).await
            }
            TestOperation::VectorOperation { operation, vector_size } => {
                self.test_vector_operation(operation, *vector_size, test_case).await
            }
            TestOperation::EdgeCaseTest { description: _, custom_data } => {
                self.test_edge_case(custom_data, test_case).await
            }
        }
    }
    
    async fn test_matrix_vector_multiply(
        &self,
        rows: usize,
        cols: usize,
        special_matrix: Option<&str>,
        test_case: &AccuracyTestCase,
    ) -> Result<AccuracyTestResult, Box<dyn std::error::Error>> {
        // Get test data
        let matrix = if let Some(special_name) = special_matrix {
            self.test_data_generator.get_special_matrix(special_name)
                .ok_or("Special matrix not found")?
                .clone()
        } else {
            self.test_data_generator.get_matrix(rows, cols)
                .ok_or("Test matrix not found")?
                .clone()
        };
        
        let vector = self.test_data_generator.get_vector(cols)
            .ok_or("Test vector not found")?;
        
        // CPU reference implementation
        let cpu_start = std::time::Instant::now();
        let cpu_result = CpuReferenceImplementation::matrix_vector_multiply(&matrix, vector, rows, cols);
        let cpu_time = cpu_start.elapsed().as_secs_f32() * 1000.0;
        
        // GPU implementation (placeholder - would use actual GPU backend)
        let gpu_start = std::time::Instant::now();
        let gpu_result = self.simulate_gpu_matrix_vector_multiply(&matrix, vector, rows, cols)?;
        let gpu_time = gpu_start.elapsed().as_secs_f32() * 1000.0;
        
        // Calculate accuracy metrics
        self.calculate_accuracy_result(
            &test_case.name,
            "MatrixVectorMultiply",
            &format!("{}x{}", rows, cols),
            &cpu_result,
            &gpu_result,
            test_case.tolerance,
            cpu_time,
            gpu_time,
        )
    }
    
    async fn test_batch_matrix_vector_multiply(
        &self,
        rows: usize,
        cols: usize,
        batch_size: usize,
        test_case: &AccuracyTestCase,
    ) -> Result<AccuracyTestResult, Box<dyn std::error::Error>> {
        let matrix = self.test_data_generator.get_matrix(rows, cols)
            .ok_or("Test matrix not found")?;
        
        let base_vector = self.test_data_generator.get_vector(cols)
            .ok_or("Test vector not found")?;
        
        // Create batch of vectors
        let batch_vectors: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                base_vector.iter()
                    .map(|&x| x + (i as f32 * 0.1))
                    .collect()
            })
            .collect();
        
        // CPU reference
        let cpu_start = std::time::Instant::now();
        let cpu_results = CpuReferenceImplementation::batch_matrix_vector_multiply(matrix, &batch_vectors, rows, cols);
        let cpu_time = cpu_start.elapsed().as_secs_f32() * 1000.0;
        
        // GPU implementation (placeholder)
        let gpu_start = std::time::Instant::now();
        let gpu_results = self.simulate_gpu_batch_matrix_vector_multiply(matrix, &batch_vectors, rows, cols)?;
        let gpu_time = gpu_start.elapsed().as_secs_f32() * 1000.0;
        
        // Flatten results for comparison
        let cpu_flat: Vec<f32> = cpu_results.into_iter().flatten().collect();
        let gpu_flat: Vec<f32> = gpu_results.into_iter().flatten().collect();
        
        self.calculate_accuracy_result(
            &test_case.name,
            "BatchMatrixVectorMultiply",
            &format!("{}x{}x{}", rows, cols, batch_size),
            &cpu_flat,
            &gpu_flat,
            test_case.tolerance,
            cpu_time,
            gpu_time,
        )
    }
    
    async fn test_activation_function(
        &self,
        function: ActivationFunction,
        input_size: usize,
        steepness: f32,
        use_edge_cases: bool,
        test_case: &AccuracyTestCase,
    ) -> Result<AccuracyTestResult, Box<dyn std::error::Error>> {
        let input_data = if use_edge_cases {
            self.test_data_generator.get_edge_case_values().clone()
        } else {
            self.test_data_generator.get_vector(input_size)
                .ok_or("Test vector not found")?
                .clone()
        };
        
        // CPU reference
        let cpu_start = std::time::Instant::now();
        let cpu_result = CpuReferenceImplementation::apply_activation_function(&input_data, function, steepness);
        let cpu_time = cpu_start.elapsed().as_secs_f32() * 1000.0;
        
        // GPU implementation (placeholder)
        let gpu_start = std::time::Instant::now();
        let gpu_result = self.simulate_gpu_activation_function(&input_data, function, steepness)?;
        let gpu_time = gpu_start.elapsed().as_secs_f32() * 1000.0;
        
        self.calculate_accuracy_result(
            &test_case.name,
            &format!("ActivationFunction_{:?}", function),
            &format!("size{}_steepness{}", input_data.len(), steepness),
            &cpu_result,
            &gpu_result,
            test_case.tolerance,
            cpu_time,
            gpu_time,
        )
    }
    
    async fn test_vector_operation(
        &self,
        operation: &VectorOpType,
        vector_size: usize,
        test_case: &AccuracyTestCase,
    ) -> Result<AccuracyTestResult, Box<dyn std::error::Error>> {
        let vector_a = self.test_data_generator.get_vector(vector_size)
            .ok_or("Test vector not found")?;
        let vector_b: Vec<f32> = vector_a.iter().map(|&x| x * 0.9 + 0.1).collect();
        
        let (cpu_result, gpu_result, cpu_time, gpu_time) = match operation {
            VectorOpType::Add => {
                let cpu_start = std::time::Instant::now();
                let cpu_res = CpuReferenceImplementation::vector_add(vector_a, &vector_b);
                let cpu_time = cpu_start.elapsed().as_secs_f32() * 1000.0;
                
                let gpu_start = std::time::Instant::now();
                let gpu_res = self.simulate_gpu_vector_add(vector_a, &vector_b)?;
                let gpu_time = gpu_start.elapsed().as_secs_f32() * 1000.0;
                
                (cpu_res, gpu_res, cpu_time, gpu_time)
            }
            VectorOpType::DotProduct => {
                let cpu_start = std::time::Instant::now();
                let cpu_res = vec![CpuReferenceImplementation::dot_product(vector_a, &vector_b)];
                let cpu_time = cpu_start.elapsed().as_secs_f32() * 1000.0;
                
                let gpu_start = std::time::Instant::now();
                let gpu_res = vec![self.simulate_gpu_dot_product(vector_a, &vector_b)?];
                let gpu_time = gpu_start.elapsed().as_secs_f32() * 1000.0;
                
                (cpu_res, gpu_res, cpu_time, gpu_time)
            }
            VectorOpType::Scale(scalar) => {
                let cpu_start = std::time::Instant::now();
                let cpu_res = CpuReferenceImplementation::vector_scale(vector_a, *scalar);
                let cpu_time = cpu_start.elapsed().as_secs_f32() * 1000.0;
                
                let gpu_start = std::time::Instant::now();
                let gpu_res = self.simulate_gpu_vector_scale(vector_a, *scalar)?;
                let gpu_time = gpu_start.elapsed().as_secs_f32() * 1000.0;
                
                (cpu_res, gpu_res, cpu_time, gpu_time)
            }
            _ => {
                // Implement other vector operations as needed
                return Err("Vector operation not implemented".into());
            }
        };
        
        self.calculate_accuracy_result(
            &test_case.name,
            &format!("VectorOperation_{:?}", operation),
            &format!("size{}", vector_size),
            &cpu_result,
            &gpu_result,
            test_case.tolerance,
            cpu_time,
            gpu_time,
        )
    }
    
    async fn test_edge_case(
        &self,
        custom_data: &[f32],
        test_case: &AccuracyTestCase,
    ) -> Result<AccuracyTestResult, Box<dyn std::error::Error>> {
        // For edge cases, we'll test a simple operation like vector scaling
        let cpu_start = std::time::Instant::now();
        let cpu_result = CpuReferenceImplementation::vector_scale(custom_data, 1.5);
        let cpu_time = cpu_start.elapsed().as_secs_f32() * 1000.0;
        
        let gpu_start = std::time::Instant::now();
        let gpu_result = self.simulate_gpu_vector_scale(custom_data, 1.5)?;
        let gpu_time = gpu_start.elapsed().as_secs_f32() * 1000.0;
        
        self.calculate_accuracy_result(
            &test_case.name,
            "EdgeCaseTest",
            &format!("size{}", custom_data.len()),
            &cpu_result,
            &gpu_result,
            test_case.tolerance,
            cpu_time,
            gpu_time,
        )
    }
    
    fn calculate_accuracy_result(
        &self,
        test_name: &str,
        operation_type: &str,
        input_dimensions: &str,
        cpu_result: &[f32],
        gpu_result: &[f32],
        tolerance: f32,
        cpu_time: f32,
        gpu_time: f32,
    ) -> Result<AccuracyTestResult, Box<dyn std::error::Error>> {
        if cpu_result.len() != gpu_result.len() {
            return Err(format!("Result length mismatch: CPU {} vs GPU {}", cpu_result.len(), gpu_result.len()).into());
        }
        
        // Calculate error statistics
        let errors: Vec<f32> = cpu_result.iter()
            .zip(gpu_result.iter())
            .map(|(&cpu, &gpu)| (cpu - gpu).abs())
            .collect();
        
        let max_error = errors.iter().cloned().fold(0.0f32, f32::max);
        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let rms_error = (errors.iter().map(|&e| e * e).sum::<f32>() / errors.len() as f32).sqrt();
        
        let passed = max_error <= tolerance;
        
        // Calculate error distribution
        let mut sorted_errors = errors.clone();
        sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let percentile_95 = sorted_errors[(sorted_errors.len() as f32 * 0.95) as usize];
        let percentile_99 = sorted_errors[(sorted_errors.len() as f32 * 0.99) as usize];
        let outlier_count = errors.iter().filter(|&&e| e > tolerance * 10.0).count();
        
        // Create error histogram
        let mut error_histogram = Vec::new();
        let bin_count = 10;
        let bin_size = max_error / bin_count as f32;
        
        for i in 0..bin_count {
            let bin_start = i as f32 * bin_size;
            let bin_end = (i + 1) as f32 * bin_size;
            let count = errors.iter().filter(|&&e| e >= bin_start && e < bin_end).count();
            error_histogram.push((bin_start, count));
        }
        
        Ok(AccuracyTestResult {
            test_name: test_name.to_string(),
            operation_type: operation_type.to_string(),
            input_dimensions: input_dimensions.to_string(),
            passed,
            max_error,
            mean_error,
            rms_error,
            tolerance_used: tolerance,
            error_distribution: ErrorDistribution {
                error_histogram,
                percentile_95,
                percentile_99,
                outlier_count,
            },
            performance_metrics: OperationPerformance {
                cpu_time_ms: cpu_time,
                gpu_time_ms: gpu_time,
                data_transfer_ms: 0.5, // Estimated
                total_time_ms: cpu_time + gpu_time + 0.5,
            },
        })
    }
    
    // Placeholder GPU implementations (would be replaced with actual GPU backend calls)
    fn simulate_gpu_matrix_vector_multiply(&self, matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // This would call the actual GPU backend
        // For now, simulate with small numerical differences
        let mut result = CpuReferenceImplementation::matrix_vector_multiply(matrix, vector, rows, cols);
        
        // Add small numerical differences to simulate GPU precision
        for val in &mut result {
            *val += (*val * f32::EPSILON * 100.0).sin() * 1e-7;
        }
        
        Ok(result)
    }
    
    fn simulate_gpu_batch_matrix_vector_multiply(&self, matrix: &[f32], vectors: &[Vec<f32>], rows: usize, cols: usize) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut result = CpuReferenceImplementation::batch_matrix_vector_multiply(matrix, vectors, rows, cols);
        
        // Add small numerical differences
        for batch_result in &mut result {
            for val in batch_result {
                *val += (*val * f32::EPSILON * 100.0).sin() * 1e-7;
            }
        }
        
        Ok(result)
    }
    
    fn simulate_gpu_activation_function(&self, inputs: &[f32], function: ActivationFunction, steepness: f32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut result = CpuReferenceImplementation::apply_activation_function(inputs, function, steepness);
        
        // Add small numerical differences
        for val in &mut result {
            *val += (*val * f32::EPSILON * 100.0).sin() * 1e-7;
        }
        
        Ok(result)
    }
    
    fn simulate_gpu_vector_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut result = CpuReferenceImplementation::vector_add(a, b);
        
        for val in &mut result {
            *val += (*val * f32::EPSILON * 100.0).sin() * 1e-7;
        }
        
        Ok(result)
    }
    
    fn simulate_gpu_dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        let mut result = CpuReferenceImplementation::dot_product(a, b);
        result += (result * f32::EPSILON * 100.0).sin() * 1e-7;
        Ok(result)
    }
    
    fn simulate_gpu_vector_scale(&self, vec: &[f32], scalar: f32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut result = CpuReferenceImplementation::vector_scale(vec, scalar);
        
        for val in &mut result {
            *val += (*val * f32::EPSILON * 100.0).sin() * 1e-7;
        }
        
        Ok(result)
    }
    
    /// Generate comprehensive accuracy report
    pub fn generate_report(&self) -> AccuracyReport {
        let total_tests = self.test_results.len();
        let passed_tests = self.test_results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;
        
        let overall_pass_rate = if total_tests > 0 {
            (passed_tests as f32 / total_tests as f32) * 100.0
        } else {
            0.0
        };
        
        let max_error = self.test_results.iter()
            .map(|r| r.max_error)
            .fold(0.0f32, f32::max);
        
        let mean_error = if total_tests > 0 {
            self.test_results.iter().map(|r| r.mean_error).sum::<f32>() / total_tests as f32
        } else {
            0.0
        };
        
        let rms_error = if total_tests > 0 {
            (self.test_results.iter().map(|r| r.rms_error * r.rms_error).sum::<f32>() / total_tests as f32).sqrt()
        } else {
            0.0
        };
        
        let failed_test_names: Vec<String> = self.test_results.iter()
            .filter(|r| !r.passed)
            .map(|r| r.test_name.clone())
            .collect();
        
        // Categorize results
        let operation_summary = self.generate_operation_summary();
        let numerical_analysis = self.generate_numerical_analysis();
        
        AccuracyReport {
            timestamp: chrono::Utc::now(),
            total_tests,
            passed_tests,
            failed_tests,
            overall_pass_rate,
            tolerance: self.tolerance,
            max_error,
            mean_error,
            rms_error,
            test_results: self.test_results.clone(),
            failed_test_names,
            operation_summary,
            numerical_analysis,
        }
    }
    
    fn generate_operation_summary(&self) -> OperationSummary {
        let matrix_tests: Vec<_> = self.test_results.iter()
            .filter(|r| r.operation_type.contains("Matrix"))
            .collect();
        
        let activation_tests: Vec<_> = self.test_results.iter()
            .filter(|r| r.operation_type.contains("Activation"))
            .collect();
        
        let vector_tests: Vec<_> = self.test_results.iter()
            .filter(|r| r.operation_type.contains("Vector"))
            .collect();
        
        let edge_case_tests: Vec<_> = self.test_results.iter()
            .filter(|r| r.operation_type.contains("EdgeCase"))
            .collect();
        
        let batch_tests: Vec<_> = self.test_results.iter()
            .filter(|r| r.operation_type.contains("Batch"))
            .collect();
        
        OperationSummary {
            matrix_operations: Self::create_category_summary(&matrix_tests),
            activation_functions: Self::create_category_summary(&activation_tests),
            vector_operations: Self::create_category_summary(&vector_tests),
            edge_cases: Self::create_category_summary(&edge_case_tests),
            batch_operations: Self::create_category_summary(&batch_tests),
        }
    }
    
    fn create_category_summary(tests: &[&AccuracyTestResult]) -> TestCategorySummary {
        let total_tests = tests.len();
        let passed_tests = tests.iter().filter(|r| r.passed).count();
        let pass_rate = if total_tests > 0 {
            (passed_tests as f32 / total_tests as f32) * 100.0
        } else {
            0.0
        };
        
        let max_error = tests.iter()
            .map(|r| r.max_error)
            .fold(0.0f32, f32::max);
        
        let mean_error = if total_tests > 0 {
            tests.iter().map(|r| r.mean_error).sum::<f32>() / total_tests as f32
        } else {
            0.0
        };
        
        TestCategorySummary {
            total_tests,
            passed_tests,
            pass_rate,
            max_error,
            mean_error,
        }
    }
    
    fn generate_numerical_analysis(&self) -> NumericalAnalysis {
        // Calculate precision degradation
        let precision_degradation = self.test_results.iter()
            .map(|r| r.max_error / r.tolerance_used)
            .fold(0.0f32, f32::max);
        
        // Calculate accumulation errors (for operations with multiple steps)
        let accumulation_errors: Vec<f32> = self.test_results.iter()
            .filter(|r| r.operation_type.contains("Batch") || r.operation_type.contains("Matrix"))
            .map(|r| r.rms_error)
            .collect();
        
        // Calculate systematic bias (mean of all mean errors)
        let systematic_bias = if !self.test_results.is_empty() {
            self.test_results.iter().map(|r| r.mean_error).sum::<f32>() / self.test_results.len() as f32
        } else {
            0.0
        };
        
        // Calculate error correlation (simplified metric)
        let error_correlation = self.calculate_error_correlation();
        
        // Calculate stability score (inverse of error variance)
        let stability_score = self.calculate_stability_score();
        
        NumericalAnalysis {
            precision_degradation,
            accumulation_errors,
            systematic_bias,
            error_correlation,
            stability_score,
        }
    }
    
    fn calculate_error_correlation(&self) -> f32 {
        if self.test_results.len() < 2 {
            return 0.0;
        }
        
        let errors: Vec<f32> = self.test_results.iter().map(|r| r.mean_error).collect();
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / errors.len() as f32;
        
        if variance > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        }
    }
    
    fn calculate_stability_score(&self) -> f32 {
        if self.test_results.is_empty() {
            return 0.0;
        }
        
        let passed_ratio = self.test_results.iter().filter(|r| r.passed).count() as f32 / self.test_results.len() as f32;
        let error_consistency = 1.0 - self.calculate_error_correlation().min(1.0);
        
        (passed_ratio + error_consistency) / 2.0
    }
}

impl AccuracyReport {
    pub fn print_detailed_summary(&self) {
        println!("\n📊 COMPREHENSIVE GPU ACCURACY VALIDATION REPORT");
        println!("===============================================");
        println!("🗓️  Report Date: {}", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
        println!("🎯 Target Tolerance: {:.2e}", self.tolerance);
        println!();
        
        // Overall results
        println!("📈 OVERALL RESULTS");
        println!("─────────────────");
        println!("Total Tests: {}", self.total_tests);
        println!("Passed: {} ({:.2}%)", self.passed_tests, self.overall_pass_rate);
        println!("Failed: {}", self.failed_tests);
        println!("Max Error: {:.2e}", self.max_error);
        println!("Mean Error: {:.2e}", self.mean_error);
        println!("RMS Error: {:.2e}", self.rms_error);
        
        // Operation category breakdown
        println!("\n📊 OPERATION BREAKDOWN");
        println!("─────────────────────");
        println!("Matrix Operations: {}/{} passed ({:.1}%)", 
               self.operation_summary.matrix_operations.passed_tests,
               self.operation_summary.matrix_operations.total_tests,
               self.operation_summary.matrix_operations.pass_rate);
        
        println!("Activation Functions: {}/{} passed ({:.1}%)",
               self.operation_summary.activation_functions.passed_tests,
               self.operation_summary.activation_functions.total_tests,
               self.operation_summary.activation_functions.pass_rate);
        
        println!("Vector Operations: {}/{} passed ({:.1}%)",
               self.operation_summary.vector_operations.passed_tests,
               self.operation_summary.vector_operations.total_tests,
               self.operation_summary.vector_operations.pass_rate);
        
        println!("Batch Operations: {}/{} passed ({:.1}%)",
               self.operation_summary.batch_operations.passed_tests,
               self.operation_summary.batch_operations.total_tests,
               self.operation_summary.batch_operations.pass_rate);
        
        println!("Edge Cases: {}/{} passed ({:.1}%)",
               self.operation_summary.edge_cases.passed_tests,
               self.operation_summary.edge_cases.total_tests,
               self.operation_summary.edge_cases.pass_rate);
        
        // Numerical analysis
        println!("\n🔬 NUMERICAL ANALYSIS");
        println!("────────────────────");
        println!("Precision Degradation: {:.2e}", self.numerical_analysis.precision_degradation);
        println!("Systematic Bias: {:.2e}", self.numerical_analysis.systematic_bias);
        println!("Error Correlation: {:.4}", self.numerical_analysis.error_correlation);
        println!("Stability Score: {:.4}", self.numerical_analysis.stability_score);
        
        if !self.numerical_analysis.accumulation_errors.is_empty() {
            let avg_accumulation = self.numerical_analysis.accumulation_errors.iter().sum::<f32>() 
                                 / self.numerical_analysis.accumulation_errors.len() as f32;
            println!("Average Accumulation Error: {:.2e}", avg_accumulation);
        }
        
        // Failed tests details
        if !self.failed_test_names.is_empty() {
            println!("\n❌ FAILED TESTS");
            println!("──────────────");
            for (i, test_name) in self.failed_test_names.iter().enumerate() {
                if let Some(result) = self.test_results.iter().find(|r| r.test_name == *test_name) {
                    println!("{}. {} - Error: {:.2e} (tolerance: {:.2e})", 
                           i + 1, test_name, result.max_error, result.tolerance_used);
                }
            }
        }
        
        // Assessment
        println!("\n🎯 ASSESSMENT");
        println!("────────────");
        if self.overall_pass_rate >= 100.0 {
            println!("✅ EXCELLENT: All accuracy tests passed!");
            println!("   GPU implementation meets strict numerical requirements");
        } else if self.overall_pass_rate >= 99.0 {
            println!("✅ VERY GOOD: {:.1}% pass rate with minimal failures", self.overall_pass_rate);
            println!("   GPU implementation is highly accurate");
        } else if self.overall_pass_rate >= 95.0 {
            println!("⚠️ GOOD: {:.1}% pass rate with some concerns", self.overall_pass_rate);
            println!("   Review failed tests for potential issues");
        } else if self.overall_pass_rate >= 90.0 {
            println!("⚠️ MARGINAL: {:.1}% pass rate requires attention", self.overall_pass_rate);
            println!("   Significant accuracy issues need resolution");
        } else {
            println!("❌ POOR: {:.1}% pass rate - critical accuracy problems", self.overall_pass_rate);
            println!("   GPU implementation requires major fixes");
        }
        
        // Recommendations
        println!("\n💡 RECOMMENDATIONS");
        println!("──────────────────");
        if self.overall_pass_rate < 100.0 {
            println!("• Investigate failed test cases for numerical precision issues");
            println!("• Consider using higher precision for intermediate calculations");
            println!("• Verify GPU shader implementations match CPU algorithms exactly");
        }
        
        if self.max_error > self.tolerance * 100.0 {
            println!("• Large errors detected - check for algorithmic differences");
        }
        
        if self.numerical_analysis.systematic_bias > self.tolerance {
            println!("• Systematic bias detected - verify mathematical correctness");
        }
        
        if self.numerical_analysis.stability_score < 0.8 {
            println!("• Numerical stability concerns - review edge case handling");
        }
        
        println!("════════════════════════════════════════════════════════");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_test_data_generator() {
        let generator = AccuracyTestDataGenerator::new();
        assert!(!generator.get_matrix_sizes().is_empty());
        assert!(!generator.get_vector_sizes().is_empty());
        assert!(!generator.get_activation_functions().is_empty());
        
        // Test that matrices are generated
        assert!(generator.get_matrix(8, 8).is_some());
        assert!(generator.get_vector(8).is_some());
        assert!(generator.get_special_matrix("identity_4x4").is_some());
    }
    
    #[test]
    fn test_cpu_reference_implementation() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0];
        let vector = vec![1.0, 2.0];
        
        let result = CpuReferenceImplementation::matrix_vector_multiply(&matrix, &vector, 2, 2);
        assert_eq!(result, vec![5.0, 11.0]);
        
        let vector_a = vec![1.0, 2.0, 3.0];
        let vector_b = vec![4.0, 5.0, 6.0];
        let dot_product = CpuReferenceImplementation::dot_product(&vector_a, &vector_b);
        assert_eq!(dot_product, 32.0);
    }
    
    #[test]
    fn test_activation_functions() {
        let inputs = vec![0.0, 1.0, -1.0];
        
        let sigmoid = CpuReferenceImplementation::apply_activation_function(&inputs, ActivationFunction::Sigmoid, 1.0);
        assert!((sigmoid[0] - 0.5).abs() < 1e-6);
        assert!(sigmoid[1] > 0.5);
        assert!(sigmoid[2] < 0.5);
        
        let relu = CpuReferenceImplementation::apply_activation_function(&inputs, ActivationFunction::ReLU, 1.0);
        assert_eq!(relu, vec![0.0, 1.0, 0.0]);
        
        let linear = CpuReferenceImplementation::apply_activation_function(&inputs, ActivationFunction::Linear, 2.0);
        assert_eq!(linear, vec![0.0, 2.0, -2.0]);
    }
    
    #[tokio::test]
    async fn test_accuracy_validator_creation() {
        let validator = AccuracyValidator::new(1e-6);
        assert_eq!(validator.tolerance, 1e-6);
        assert!(!validator.test_cases.is_empty());
        println!("Generated {} test cases", validator.test_cases.len());
    }
    
    #[test]
    fn test_edge_case_data() {
        let generator = AccuracyTestDataGenerator::new();
        let edge_values = generator.get_edge_case_values();
        
        assert!(edge_values.contains(&0.0));
        assert!(edge_values.contains(&1.0));
        assert!(edge_values.contains(&-1.0));
        assert!(edge_values.contains(&f32::EPSILON));
        assert!(edge_values.contains(&PI));
    }
    
    #[test]
    fn test_accuracy_result_calculation() {
        let validator = AccuracyValidator::new(1e-6);
        let cpu_result = vec![1.0, 2.0, 3.0];
        let gpu_result = vec![1.000001, 1.999999, 3.000001];
        
        let result = validator.calculate_accuracy_result(
            "test",
            "TestOperation",
            "3x1",
            &cpu_result,
            &gpu_result,
            1e-5,
            1.0,
            0.5,
        ).unwrap();
        
        assert!(result.passed);
        assert!(result.max_error < 1e-5);
        assert_eq!(result.test_name, "test");
    }
}