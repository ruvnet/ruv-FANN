//! Error handling for OpenCV operations

use thiserror::Error;

/// OpenCV error types
#[derive(Error, Debug)]
pub enum Error {
    #[error("Memory allocation failed: {0}")]
    Memory(String),
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    #[error("Operation failed: {0}")]
    Operation(String),
    
    #[error("I/O error: {0}")]
    Io(String),
    
    #[error("CUDA error: {0}")]
    Cuda(String),
    
    #[error("Type conversion error: {0}")]
    TypeConversion(String),
    
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),
    
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    
    #[error("External library error: {0}")]
    External(String),
}

/// OpenCV result type
pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string())
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::External(err.to_string())
    }
}