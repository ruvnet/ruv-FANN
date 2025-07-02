//! Error types for WebGPU acceleration

use thiserror::Error;

/// Result type for compute operations
pub type ComputeResult<T> = Result<T, ComputeError>;

/// Errors that can occur during GPU computation
#[derive(Error, Debug, Clone)]
pub enum ComputeError {
    /// GPU device is not available or initialization failed
    #[error("GPU device unavailable: {0}")]
    DeviceUnavailable(String),

    /// Invalid matrix dimensions for the operation
    #[error("Invalid dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: String, actual: String },

    /// GPU memory allocation failed
    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocationFailed { size: u64 },

    /// Compute operation failed
    #[error("Compute operation failed: {operation}")]
    ComputeFailed { operation: String },

    /// Shader compilation or execution error
    #[error("Shader error: {0}")]
    ShaderError(String),

    /// Buffer operation error
    #[error("Buffer operation failed: {0}")]
    BufferError(String),

    /// Unsupported operation on current hardware
    #[error("Unsupported operation: {operation} on {device}")]
    UnsupportedOperation { operation: String, device: String },

    /// Timeout waiting for operation to complete
    #[error("Operation timeout: {operation} after {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },

    /// Backend initialization failed
    #[error("Backend initialization failed: {backend}")]
    BackendInitializationFailed { backend: String },

    /// Internal error (should not happen in normal operation)
    #[error("Internal error: {0}")]
    Internal(String),
}

impl ComputeError {
    /// Create a device unavailable error
    pub fn device_unavailable(reason: impl Into<String>) -> Self {
        Self::DeviceUnavailable(reason.into())
    }

    /// Create an invalid dimensions error
    pub fn invalid_dimensions(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::InvalidDimensions {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a memory allocation error
    pub fn memory_allocation_failed(size: u64) -> Self {
        Self::MemoryAllocationFailed { size }
    }

    /// Create a compute operation failed error
    pub fn compute_failed(operation: impl Into<String>) -> Self {
        Self::ComputeFailed {
            operation: operation.into(),
        }
    }

    /// Create a shader error
    pub fn shader_error(message: impl Into<String>) -> Self {
        Self::ShaderError(message.into())
    }

    /// Create a buffer error
    pub fn buffer_error(message: impl Into<String>) -> Self {
        Self::BufferError(message.into())
    }

    /// Create an unsupported operation error
    pub fn unsupported_operation(operation: impl Into<String>, device: impl Into<String>) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            device: device.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_ms,
        }
    }

    /// Create a backend initialization error
    pub fn backend_initialization_failed(backend: impl Into<String>) -> Self {
        Self::BackendInitializationFailed {
            backend: backend.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
    
    /// Create a backend error (generic backend-related error)
    pub fn backend_error(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
    
    /// Create a memory error
    pub fn memory_error(message: impl Into<String>) -> Self {
        Self::BufferError(message.into())
    }

    /// Check if this error is recoverable (can retry with different backend)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::DeviceUnavailable(_)
                | Self::MemoryAllocationFailed { .. }
                | Self::UnsupportedOperation { .. }
                | Self::Timeout { .. }
                | Self::BackendInitializationFailed { .. }
        )
    }

    /// Get suggested recovery action
    pub fn recovery_suggestion(&self) -> &'static str {
        match self {
            Self::DeviceUnavailable(_) => "Try fallback to CPU backend",
            Self::InvalidDimensions { .. } => "Check input dimensions",
            Self::MemoryAllocationFailed { .. } => "Reduce batch size or try CPU backend",
            Self::ComputeFailed { .. } => "Retry with different parameters",
            Self::ShaderError(_) => "Report as bug - shader compilation failed",
            Self::BufferError(_) => "Retry operation or use CPU backend",
            Self::UnsupportedOperation { .. } => "Use CPU backend for this operation",
            Self::Timeout { .. } => "Reduce workload size or increase timeout",
            Self::BackendInitializationFailed { .. } => "Try different GPU backend",
            Self::Internal(_) => "Report as bug - internal error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = ComputeError::device_unavailable("WebGPU not supported");
        assert!(matches!(err, ComputeError::DeviceUnavailable(_)));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_error_formatting() {
        let err = ComputeError::invalid_dimensions("3x3", "2x2");
        let error_string = format!("{}", err);
        assert!(error_string.contains("expected 3x3"));
        assert!(error_string.contains("got 2x2"));
    }

    #[test]
    fn test_recovery_suggestions() {
        let err = ComputeError::memory_allocation_failed(1024);
        assert_eq!(err.recovery_suggestion(), "Reduce batch size or try CPU backend");
    }

    #[test]
    fn test_recoverable_errors() {
        assert!(ComputeError::device_unavailable("test").is_recoverable());
        assert!(ComputeError::memory_allocation_failed(1024).is_recoverable());
        assert!(!ComputeError::invalid_dimensions("3x3", "2x2").is_recoverable());
    }
}