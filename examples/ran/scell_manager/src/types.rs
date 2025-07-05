//! Type definitions for the SCell Manager

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// User Equipment identifier
pub type UEId = String;

/// Cell identifier
pub type CellId = String;

/// Model identifier
pub type ModelId = String;

/// UE metrics for prediction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UEMetrics {
    pub ue_id: UEId,
    pub pcell_throughput_mbps: f32,
    pub buffer_status_report_bytes: i64,
    pub pcell_cqi: f32,
    pub pcell_rsrp: f32,
    pub pcell_sinr: f32,
    pub active_bearers: i32,
    pub data_rate_req_mbps: f32,
    pub timestamp_utc: DateTime<Utc>,
}

impl UEMetrics {
    pub fn new(ue_id: UEId) -> Self {
        Self {
            ue_id,
            pcell_throughput_mbps: 0.0,
            buffer_status_report_bytes: 0,
            pcell_cqi: 0.0,
            pcell_rsrp: 0.0,
            pcell_sinr: 0.0,
            active_bearers: 0,
            data_rate_req_mbps: 0.0,
            timestamp_utc: Utc::now(),
        }
    }
    
    /// Convert to feature vector for ML model
    pub fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            self.pcell_throughput_mbps,
            self.buffer_status_report_bytes as f32,
            self.pcell_cqi,
            self.pcell_rsrp,
            self.pcell_sinr,
            self.active_bearers as f32,
            self.data_rate_req_mbps,
            // Time-based features
            self.timestamp_utc.hour() as f32,
            self.timestamp_utc.minute() as f32,
            self.timestamp_utc.weekday().num_days_from_monday() as f32,
        ]
    }
}

/// SCell activation prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCellPrediction {
    pub ue_id: UEId,
    pub scell_activation_recommended: bool,
    pub confidence_score: f32,
    pub predicted_throughput_demand: f32,
    pub reasoning: String,
    pub timestamp_utc: DateTime<Utc>,
}

impl SCellPrediction {
    pub fn new(ue_id: UEId, activation_recommended: bool, confidence: f32) -> Self {
        Self {
            ue_id,
            scell_activation_recommended: activation_recommended,
            confidence_score: confidence,
            predicted_throughput_demand: 0.0,
            reasoning: String::new(),
            timestamp_utc: Utc::now(),
        }
    }
}

/// Training example for model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub input_metrics: UEMetrics,
    pub historical_sequence: Vec<UEMetrics>,
    pub actual_scell_needed: bool,
    pub actual_throughput_demand: f32,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub auc_roc: f32,
    pub mean_absolute_error: f32,
    pub total_predictions: i32,
    pub true_positives: i32,
    pub false_positives: i32,
    pub true_negatives: i32,
    pub false_negatives: i32,
}

impl ModelMetrics {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn calculate_derived_metrics(&mut self) {
        let tp = self.true_positives as f32;
        let fp = self.false_positives as f32;
        let tn = self.true_negatives as f32;
        let fn_val = self.false_negatives as f32;
        
        self.accuracy = (tp + tn) / (tp + tn + fp + fn_val);
        self.precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        self.recall = if tp + fn_val > 0.0 { tp / (tp + fn_val) } else { 0.0 };
        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * (self.precision * self.recall) / (self.precision + self.recall)
        } else {
            0.0
        };
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: i32,
    pub learning_rate: f32,
    pub batch_size: i32,
    pub validation_split: f32,
    pub sequence_length: i32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            validation_split: 0.2,
            sequence_length: 10,
        }
    }
}

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub healthy: bool,
    pub version: String,
    pub active_models: i32,
    pub total_predictions: i64,
    pub average_prediction_time_ms: f32,
    pub uptime_seconds: i64,
    pub system_info: HashMap<String, String>,
}

impl SystemStatus {
    pub fn new() -> Self {
        let mut system_info = HashMap::new();
        system_info.insert("rust_version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        system_info.insert("build_timestamp".to_string(), env!("VERGEN_BUILD_TIMESTAMP").unwrap_or("unknown").to_string());
        
        Self {
            healthy: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            active_models: 0,
            total_predictions: 0,
            average_prediction_time_ms: 0.0,
            uptime_seconds: 0,
            system_info,
        }
    }
}

/// Prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub ue_id: UEId,
    pub current_metrics: UEMetrics,
    pub historical_metrics: Vec<UEMetrics>,
    pub prediction_horizon_seconds: i32,
}

/// Streaming prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub ue_ids: Vec<UEId>,
    pub update_interval_seconds: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ue_metrics_feature_vector() {
        let mut metrics = UEMetrics::new("test_ue".to_string());
        metrics.pcell_throughput_mbps = 50.0;
        metrics.buffer_status_report_bytes = 1000;
        metrics.pcell_cqi = 15.0;
        
        let features = metrics.to_feature_vector();
        assert_eq!(features.len(), 10); // 7 base features + 3 time features
        assert_eq!(features[0], 50.0);
        assert_eq!(features[1], 1000.0);
        assert_eq!(features[2], 15.0);
    }
    
    #[test]
    fn test_model_metrics_calculation() {
        let mut metrics = ModelMetrics::new();
        metrics.true_positives = 80;
        metrics.false_positives = 10;
        metrics.true_negatives = 85;
        metrics.false_negatives = 5;
        
        metrics.calculate_derived_metrics();
        
        assert!((metrics.accuracy - 0.9167).abs() < 0.01);
        assert!((metrics.precision - 0.8889).abs() < 0.01);
        assert!((metrics.recall - 0.9412).abs() < 0.01);
    }
}