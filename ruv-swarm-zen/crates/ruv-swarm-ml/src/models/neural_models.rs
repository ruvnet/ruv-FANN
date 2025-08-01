//! Neural network model implementations using ruv-fann
//!
//! This module delegates to neural_bridge for actual implementations.

use super::{ForecastModel, ModelType};
use super::neural_bridge;
use alloc::boxed::Box;
use alloc::string::String;

/// Model factory for creating concrete neural network models
pub fn create_model(model_type: ModelType, input_size: usize, output_size: usize) -> Result<Box<dyn ForecastModel>, String> {
    // Delegate to neural_bridge for actual implementation
    neural_bridge::create_neural_model(model_type, input_size, output_size)
}