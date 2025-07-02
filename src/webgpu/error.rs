//! Error types for WebGPU compute backend

use thiserror::Error;

/// Errors that can occur during compute operations
#[derive(Error, Debug)]
pub enum ComputeError {
    #[error("Backend initialization failed: {0}")]
    InitializationError(String),

    #[error("GPU not available or compatible")]
    GpuUnavailable,

    #[error("Buffer allocation failed: {0}")]
    AllocationError(String),

    #[error("Shader compilation failed: {0}")]
    ShaderError(String),

    #[error("Memory transfer failed: {0}")]
    TransferError(String),

    #[error("Compute operation failed: {0}")]
    ComputeError(String),

    #[error("Backend-specific error: {0}")]
    BackendError(String),

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Async operation error: {0}")]
    AsyncError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("General error: {0}")]
    General(String),
}

/// Result type for compute operations
pub type ComputeResult<T> = Result<T, ComputeError>;
