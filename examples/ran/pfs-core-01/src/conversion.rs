use crate::neural_service::*;
use crate::error::{ServiceError, ServiceResult};
use ruv_fann::{ActivationFunction as RuvActivationFunction};

/// Convert protobuf ActivationFunction to ruv-FANN ActivationFunction
pub fn convert_activation_function(proto_func: i32) -> ServiceResult<RuvActivationFunction> {
    match ActivationFunction::try_from(proto_func) {
        Ok(ActivationFunction::Sigmoid) => Ok(RuvActivationFunction::Sigmoid),
        Ok(ActivationFunction::Tanh) => Ok(RuvActivationFunction::Tanh),
        Ok(ActivationFunction::Relu) => Ok(RuvActivationFunction::Linear), // ReLU not directly available
        Ok(ActivationFunction::LeakyRelu) => Ok(RuvActivationFunction::Linear), // LeakyReLU not directly available
        Ok(ActivationFunction::Linear) => Ok(RuvActivationFunction::Linear),
        Ok(ActivationFunction::Softmax) => Ok(RuvActivationFunction::Linear), // Softmax not directly available
        Err(_) => Err(ServiceError::InvalidInput(format!("Unknown activation function: {}", proto_func))),
    }
}

/// Convert protobuf TrainingAlgorithm to string for identification
pub fn convert_training_algorithm(proto_algo: i32) -> ServiceResult<&'static str> {
    match TrainingAlgorithm::try_from(proto_algo) {
        Ok(TrainingAlgorithm::Backpropagation) => Ok("backprop"),
        Ok(TrainingAlgorithm::Rprop) => Ok("rprop"),
        Ok(TrainingAlgorithm::Quickprop) => Ok("quickprop"),
        Ok(TrainingAlgorithm::Batch) => Ok("batch"),
        Err(_) => Err(ServiceError::InvalidInput(format!("Unknown training algorithm: {}", proto_algo))),
    }
}

/// Convert protobuf TrainingData to ruv-FANN TrainingData
pub fn convert_training_data(proto_data: &TrainingData) -> ServiceResult<ruv_fann::TrainingData<f64>> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    
    for example in &proto_data.examples {
        if example.inputs.len() != proto_data.input_size as usize {
            return Err(ServiceError::InvalidInput(
                format!("Input size mismatch: expected {}, got {}", 
                    proto_data.input_size, example.inputs.len())
            ));
        }
        
        if example.outputs.len() != proto_data.output_size as usize {
            return Err(ServiceError::InvalidInput(
                format!("Output size mismatch: expected {}, got {}", 
                    proto_data.output_size, example.outputs.len())
            ));
        }
        
        inputs.push(example.inputs.clone());
        outputs.push(example.outputs.clone());
    }
    
    if inputs.is_empty() {
        return Err(ServiceError::InvalidInput("Training data cannot be empty".to_string()));
    }
    
    Ok(ruv_fann::TrainingData { inputs, outputs })
}

/// Validate model configuration
pub fn validate_model_config(config: &ModelConfig) -> ServiceResult<()> {
    if config.layers.is_empty() {
        return Err(ServiceError::InvalidInput("Model must have at least one layer".to_string()));
    }
    
    if config.layers.len() < 2 {
        return Err(ServiceError::InvalidInput("Model must have at least input and output layers".to_string()));
    }
    
    for (i, &layer_size) in config.layers.iter().enumerate() {
        if layer_size == 0 {
            return Err(ServiceError::InvalidInput(format!("Layer {} cannot have 0 neurons", i)));
        }
    }
    
    if config.learning_rate <= 0.0 {
        return Err(ServiceError::InvalidInput("Learning rate must be positive".to_string()));
    }
    
    if config.desired_error <= 0.0 {
        return Err(ServiceError::InvalidInput("Desired error must be positive".to_string()));
    }
    
    if config.max_epochs == 0 {
        return Err(ServiceError::InvalidInput("Max epochs must be positive".to_string()));
    }
    
    Ok(())
}

/// Convert vector of f64 to vector of f32 for compatibility
pub fn f64_to_f32(input: &[f64]) -> Vec<f32> {
    input.iter().map(|&x| x as f32).collect()
}

/// Convert vector of f32 to vector of f64 for compatibility
pub fn f32_to_f64(input: &[f32]) -> Vec<f64> {
    input.iter().map(|&x| x as f64).collect()
}

/// Validate input vector size
pub fn validate_input_size(input: &[f64], expected_size: usize) -> ServiceResult<()> {
    if input.len() != expected_size {
        return Err(ServiceError::InvalidInput(
            format!("Input size mismatch: expected {}, got {}", expected_size, input.len())
        ));
    }
    Ok(())
}

/// Validate that input vector contains valid numbers
pub fn validate_input_values(input: &[f64]) -> ServiceResult<()> {
    for (i, &value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(ServiceError::InvalidInput(
                format!("Input value at index {} is not finite: {}", i, value)
            ));
        }
    }
    Ok(())
}