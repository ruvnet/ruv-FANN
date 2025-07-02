//! Pure Rust implementation of the Fast Artificial Neural Network (FANN) library
//!
//! This crate provides a modern, safe, and efficient implementation of neural networks
//! inspired by the original FANN library, with support for generic floating-point types.
//! Includes full cascade correlation support for dynamic network topology optimization.

// Re-export main types
pub use activation::ActivationFunction;
pub use connection::Connection;
pub use layer::Layer;
pub use network::{Network, NetworkBuilder, NetworkError};
pub use neuron::Neuron;

// Re-export training types
pub use training::{
    ParallelTrainingOptions, TrainingAlgorithm as TrainingAlgorithmTrait, TrainingData,
    TrainingError, TrainingState,
};

/// Enumeration of available training algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingAlgorithm {
    IncrementalBackprop,
    BatchBackprop,
    Batch,           // Alias for BatchBackprop
    Backpropagation, // Alias for IncrementalBackprop
    RProp,
    QuickProp,
}

// Re-export cascade training types
pub use cascade::{CascadeConfig, CascadeError, CascadeNetwork, CascadeTrainer};

// Re-export comprehensive error handling
pub use errors::{ErrorCategory, RuvFannError, ValidationError};

// Modules
pub mod activation;
pub mod cascade;
pub mod connection;
pub mod errors;
pub mod integration;
pub mod layer;
pub mod network;
pub mod neuron;
pub mod training;

// Optional I/O module
#[cfg(feature = "io")]
pub mod io;

// WebGPU acceleration module
pub mod webgpu;

// Test module
#[cfg(test)]
mod tests;

// Mock types for testing
pub mod mock_types;
