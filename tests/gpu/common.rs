//! Common test utilities and mock implementations
//!
//! This module provides shared test infrastructure including:
//! - Mock types for external dependencies
//! - Test stubs for missing implementations
//! - Common test utilities and helpers

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mock CognitivePattern enum for tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitivePattern {
    /// Convergent thinking - focused, logical problem solving
    Convergent,
    /// Divergent thinking - creative, exploratory approach  
    Divergent,
    /// Lateral thinking - unconventional, indirect solutions
    Lateral,
    /// Systems thinking - holistic, interconnected approach
    Systems,
    /// Critical thinking - analytical, evaluative mindset
    Critical,
    /// Adaptive thinking - flexible, responsive approach
    Adaptive,
}

impl Default for CognitivePattern {
    fn default() -> Self {
        CognitivePattern::Convergent
    }
}

/// Mock GPU DAA Agent for tests
#[derive(Debug, Clone)]
pub struct GPUDAAAgent {
    pub id: String,
    pub cognitive_pattern: CognitivePattern,
    pub active_tasks: Vec<GPUTask>,
}

impl GPUDAAAgent {
    pub fn new(id: String, pattern: CognitivePattern) -> Self {
        Self {
            id,
            cognitive_pattern: pattern,
            active_tasks: Vec::new(),
        }
    }

    pub async fn execute_task(&mut self, task: GPUTask) -> Result<(), Box<dyn std::error::Error>> {
        self.active_tasks.push(task);
        Ok(())
    }
}

/// Mock GPU Task types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUTaskType {
    Training,
    Inference,
    MemoryTransfer,
    Optimization,
    Coordination,
}

/// Mock GPU Task structure
#[derive(Debug, Clone)]
pub struct GPUTask {
    pub id: String,
    pub task_type: GPUTaskType,
    pub resource_requirements: GPUResourceRequirements,
    pub priority: u8,
}

impl GPUTask {
    pub fn new(id: String, task_type: GPUTaskType) -> Self {
        Self {
            id,
            task_type,
            resource_requirements: GPUResourceRequirements::default(),
            priority: 5,
        }
    }
}

/// Mock GPU Resource Requirements
#[derive(Debug, Clone, Default)]
pub struct GPUResourceRequirements {
    pub memory_mb: usize,
    pub compute_units: u32,
    pub bandwidth_gbps: f64,
}

/// Mock scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    FirstComeFirstServe,
    Priority,
    RoundRobin,
    Adaptive,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        SchedulingPolicy::Priority
    }
}

/// Mock priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 1,
    Medium = 5,
    High = 8,
    Critical = 10,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Medium
    }
}

/// Test result helpers
pub fn create_test_operation_result() -> TestOperationResult {
    TestOperationResult {
        operation_type: "test_operation".to_string(),
        problem_size: 1000,
        cpu_time_ms: 100.0,
        gpu_time_ms: Some(25.0),
        simd_time_ms: Some(50.0),
        speedup_gpu: Some(4.0),
        speedup_simd: Some(2.0),
        accuracy_error: 0.001,
        memory_usage_mb: 128.0,
    }
}

/// Test operation result structure
#[derive(Debug, Clone)]
pub struct TestOperationResult {
    pub operation_type: String,
    pub problem_size: usize,
    pub cpu_time_ms: f64,
    pub gpu_time_ms: Option<f64>,
    pub simd_time_ms: Option<f64>,
    pub speedup_gpu: Option<f64>,
    pub speedup_simd: Option<f64>,
    pub accuracy_error: f64,
    pub memory_usage_mb: f64,
}

/// Mock GPU benchmark suite for missing methods
pub struct MockGPUBenchmarkSuite {
    pub results: Vec<TestOperationResult>,
}

impl MockGPUBenchmarkSuite {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub async fn benchmark_nn_training(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let result = TestOperationResult {
            operation_type: "nn_training".to_string(),
            problem_size: 10000,
            cpu_time_ms: 1000.0,
            gpu_time_ms: Some(250.0),
            simd_time_ms: Some(500.0),
            speedup_gpu: Some(4.0),
            speedup_simd: Some(2.0),
            accuracy_error: 0.0001,
            memory_usage_mb: 512.0,
        };
        self.results.push(result);
        Ok(())
    }

    pub async fn benchmark_nn_inference(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let result = TestOperationResult {
            operation_type: "nn_inference".to_string(),
            problem_size: 5000,
            cpu_time_ms: 50.0,
            gpu_time_ms: Some(12.5),
            simd_time_ms: Some(25.0),
            speedup_gpu: Some(4.0),
            speedup_simd: Some(2.0),
            accuracy_error: 0.0001,
            memory_usage_mb: 256.0,
        };
        self.results.push(result);
        Ok(())
    }

    pub async fn benchmark_memory_transfer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let result = TestOperationResult {
            operation_type: "memory_transfer".to_string(),
            problem_size: 1000000,
            cpu_time_ms: 200.0,
            gpu_time_ms: Some(50.0),
            simd_time_ms: None,
            speedup_gpu: Some(4.0),
            speedup_simd: None,
            accuracy_error: 0.0,
            memory_usage_mb: 1024.0,
        };
        self.results.push(result);
        Ok(())
    }
}

/// GPU feature gates for conditional compilation
#[cfg(feature = "gpu")]
pub fn webgpu_available() -> bool {
    true
}

#[cfg(not(feature = "gpu"))]
pub fn webgpu_available() -> bool {
    false
}

/// Test helper functions (avoiding macro conflicts)
pub fn skip_if_no_gpu() -> bool {
    if !webgpu_available() {
        println!("Skipping GPU test - WebGPU feature not enabled");
        false
    } else {
        true
    }
}

pub fn create_mock_agent(id: &str, pattern: CognitivePattern) -> GPUDAAAgent {
    GPUDAAAgent::new(id.to_string(), pattern)
}

/// Error types for test mocks
#[derive(Debug, thiserror::Error)]
pub enum MockError {
    #[error("GPU not available")]
    GpuNotAvailable,
    #[error("Task execution failed: {0}")]
    TaskExecutionFailed(String),
    #[error("Resource allocation failed")]
    ResourceAllocationFailed,
}

pub type MockResult<T> = Result<T, MockError>;
