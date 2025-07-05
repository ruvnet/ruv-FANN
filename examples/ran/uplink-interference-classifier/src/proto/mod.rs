//! Protocol buffer definitions for the interference classifier service
//! 
//! This module contains the auto-generated protobuf code and helper utilities.

// Include the generated protobuf code
tonic::include_proto!("interference_classifier");

// Re-export for convenience
pub use interference_classifier_client::InterferenceClassifierClient;
pub use interference_classifier_server::{InterferenceClassifier, InterferenceClassifierServer};

// Helper functions for working with protobuf messages
impl ClassifyRequest {
    /// Create a new classification request
    pub fn new(
        cell_id: String,
        measurements: Vec<NoiseFloorMeasurement>,
        cell_params: Option<CellParameters>,
    ) -> Self {
        Self {
            cell_id,
            measurements,
            cell_params,
        }
    }
}

impl NoiseFloorMeasurement {
    /// Create a new noise floor measurement
    pub fn new(
        timestamp: String,
        noise_floor_pusch: f64,
        noise_floor_pucch: f64,
        cell_ret: f64,
        rsrp: f64,
        sinr: f64,
        active_users: u32,
        prb_utilization: f64,
    ) -> Self {
        Self {
            timestamp,
            noise_floor_pusch,
            noise_floor_pucch,
            cell_ret,
            rsrp,
            sinr,
            active_users,
            prb_utilization,
        }
    }
}

impl CellParameters {
    /// Create new cell parameters
    pub fn new(
        cell_id: String,
        frequency_band: String,
        tx_power: f64,
        antenna_count: u32,
        bandwidth_mhz: f64,
        technology: String,
    ) -> Self {
        Self {
            cell_id,
            frequency_band,
            tx_power,
            antenna_count,
            bandwidth_mhz,
            technology,
        }
    }
}

impl TrainingExample {
    /// Create a new training example
    pub fn new(
        measurements: Vec<NoiseFloorMeasurement>,
        cell_params: Option<CellParameters>,
        true_interference_class: String,
    ) -> Self {
        Self {
            measurements,
            cell_params,
            true_interference_class,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(
        hidden_layers: Vec<u32>,
        learning_rate: f64,
        max_epochs: u32,
        target_accuracy: f64,
        activation_function: String,
        dropout_rate: f64,
    ) -> Self {
        Self {
            hidden_layers,
            learning_rate,
            max_epochs,
            target_accuracy,
            activation_function,
            dropout_rate,
        }
    }
    
    /// Create default configuration
    pub fn default() -> Self {
        Self {
            hidden_layers: vec![64, 32, 16],
            learning_rate: 0.001,
            max_epochs: 1000,
            target_accuracy: 0.95,
            activation_function: "relu".to_string(),
            dropout_rate: 0.2,
        }
    }
}