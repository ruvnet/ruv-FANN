//! Handover prediction engine and result processing
//!
//! This module provides the main prediction interface for the handover system,
//! including result processing, confidence scoring, and target cell selection.

use crate::data::{UeMetrics, NeighborCell};
use crate::features::{FeatureExtractor, FeatureVector};
use crate::model::{HandoverModel, ModelInfo};
use crate::{OptMobError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Handover prediction errors
#[derive(Error, Debug)]
pub enum PredictionError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    
    #[error("Insufficient data for prediction: {0}")]
    InsufficientData(String),
    
    #[error("Feature extraction failed: {0}")]
    FeatureExtraction(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Prediction computation failed: {0}")]
    ComputationError(String),
}

/// Handover prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub ue_id: String,
    pub prediction_timestamp: chrono::DateTime<chrono::Utc>,
    pub ho_probability: f64,
    pub confidence: f64,
    pub target_cell_id: Option<String>,
    pub target_cell_confidence: f64,
    pub candidate_cells: Vec<CandidateCell>,
    pub decision_factors: DecisionFactors,
    pub recommendation: HandoverRecommendation,
}

/// Candidate target cell with prediction score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateCell {
    pub cell_id: String,
    pub handover_probability: f64,
    pub suitability_score: f64,
    pub rsrp_gain: f64,
    pub sinr_gain: f64,
    pub load_factor: f64,
    pub distance_penalty: f64,
    pub overall_score: f64,
}

/// Factors that influenced the handover decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionFactors {
    pub serving_cell_quality: f64,
    pub mobility_factor: f64,
    pub neighbor_advantage: f64,
    pub load_balancing: f64,
    pub urgency_score: f64,
    pub time_factors: HashMap<String, f64>,
    pub feature_contributions: HashMap<String, f64>,
}

/// Handover recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandoverRecommendation {
    Immediate {
        target_cell_id: String,
        urgency: UrgencyLevel,
        expected_gain: f64,
    },
    Scheduled {
        target_cell_id: String,
        recommended_delay_ms: u64,
        conditions: Vec<String>,
    },
    Monitor {
        reasons: Vec<String>,
        next_check_ms: u64,
    },
    NoHandover {
        reasons: Vec<String>,
        confidence: f64,
    },
}

/// Urgency level for immediate handovers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Handover predictor engine
pub struct HandoverPredictor {
    model: Option<HandoverModel>,
    feature_extractor: FeatureExtractor,
    prediction_history: HashMap<String, Vec<PredictionResult>>,
    configuration: PredictorConfig,
}

/// Predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    pub handover_threshold: f64,
    pub confidence_threshold: f64,
    pub feature_window_size: usize,
    pub prediction_horizon_seconds: i64,
    pub candidate_cell_limit: usize,
    pub enable_load_balancing: bool,
    pub enable_historical_context: bool,
    pub min_rsrp_gain_db: f64,
    pub min_sinr_gain_db: f64,
    pub max_handover_rate_per_minute: f64,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            handover_threshold: 0.5,
            confidence_threshold: 0.7,
            feature_window_size: 10,
            prediction_horizon_seconds: 30,
            candidate_cell_limit: 5,
            enable_load_balancing: true,
            enable_historical_context: true,
            min_rsrp_gain_db: 3.0,
            min_sinr_gain_db: 2.0,
            max_handover_rate_per_minute: 2.0,
        }
    }
}

impl HandoverPredictor {
    /// Create a new predictor with default configuration
    pub fn new() -> Self {
        let config = PredictorConfig::default();
        Self::with_config(config)
    }
    
    /// Create a new predictor with custom configuration
    pub fn with_config(config: PredictorConfig) -> Self {
        Self {
            model: None,
            feature_extractor: FeatureExtractor::new(config.feature_window_size),
            prediction_history: HashMap::new(),
            configuration: config,
        }
    }
    
    /// Load a trained model
    pub fn load_model(&mut self, model: HandoverModel) -> Result<()> {
        self.model = Some(model);
        Ok(())
    }
    
    /// Load model from file
    pub fn load_model_from_file(&mut self, path: &str) -> Result<()> {
        let model = HandoverModel::load(path)?;
        self.load_model(model)
    }
    
    /// Add UE metrics to the feature extraction buffer
    pub fn add_metrics(&mut self, metrics: UeMetrics) -> Result<()> {
        // Validate metrics
        metrics.validate().map_err(|e| OptMobError::Data(e))?;
        
        // Add to feature extractor
        self.feature_extractor.add_metrics(metrics);
        
        Ok(())
    }
    
    /// Predict handover for the current UE state
    pub fn predict(&mut self, ue_id: &str) -> Result<PredictionResult> {
        // Check if model is loaded
        let model = self.model.as_ref()
            .ok_or(PredictionError::ModelNotLoaded)?;
        
        // Check if we have enough data
        if !self.feature_extractor.is_ready() {
            return Err(PredictionError::InsufficientData(
                "Not enough metrics in buffer".to_string()
            ).into());
        }
        
        // Extract features
        let features = self.feature_extractor.extract_features()
            .map_err(|e| PredictionError::FeatureExtraction(e.to_string()))?;
        
        // Run prediction
        let ho_probability = model.predict(&features)
            .map_err(|e| PredictionError::ComputationError(e.to_string()))?;
        
        // Calculate confidence
        let confidence = self.calculate_confidence(ho_probability, &features);
        
        // Find candidate cells and select best target
        let candidate_cells = self.find_candidate_cells(&features)?;
        let target_cell = self.select_target_cell(&candidate_cells);
        
        // Calculate decision factors
        let decision_factors = self.calculate_decision_factors(&features);
        
        // Generate recommendation
        let recommendation = self.generate_recommendation(
            ho_probability,
            confidence,
            &target_cell,
            &candidate_cells,
            &decision_factors,
        );
        
        // Create prediction result
        let result = PredictionResult {
            ue_id: ue_id.to_string(),
            prediction_timestamp: chrono::Utc::now(),
            ho_probability,
            confidence,
            target_cell_id: target_cell.as_ref().map(|c| c.cell_id.clone()),
            target_cell_confidence: target_cell.as_ref().map(|c| c.overall_score).unwrap_or(0.0),
            candidate_cells,
            decision_factors,
            recommendation,
        };
        
        // Store in history
        self.store_prediction_history(ue_id, &result);
        
        Ok(result)
    }
    
    /// Batch prediction for multiple UEs
    pub fn predict_batch(&mut self, ue_metrics: &[(String, Vec<UeMetrics>)]) -> Result<Vec<PredictionResult>> {
        let mut results = Vec::new();
        
        for (ue_id, metrics) in ue_metrics {
            // Reset feature extractor for this UE
            self.feature_extractor.reset();
            
            // Add all metrics for this UE
            for metric in metrics {
                self.add_metrics(metric.clone())?;
            }
            
            // Predict
            if self.feature_extractor.is_ready() {
                match self.predict(ue_id) {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        tracing::warn!("Prediction failed for UE {}: {}", ue_id, e);
                        // Continue with other UEs
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Get model information
    pub fn get_model_info(&self) -> Option<ModelInfo> {
        self.model.as_ref().map(|m| m.get_model_info())
    }
    
    /// Get prediction history for a UE
    pub fn get_prediction_history(&self, ue_id: &str) -> Option<&Vec<PredictionResult>> {
        self.prediction_history.get(ue_id)
    }
    
    /// Clear prediction history
    pub fn clear_history(&mut self) {
        self.prediction_history.clear();
    }
    
    /// Calculate prediction confidence
    fn calculate_confidence(&self, probability: f64, features: &FeatureVector) -> f64 {
        let mut confidence = 0.0;
        
        // Distance from decision boundary (0.5)
        let boundary_distance = (probability - 0.5).abs();
        confidence += boundary_distance * 0.4;
        
        // Signal quality factor
        if features.features.len() > 2 {
            let rsrp = features.features[0]; // Normalized RSRP
            let sinr = features.features[1]; // Normalized SINR
            
            // Higher confidence for stronger signals
            let signal_quality = (rsrp + sinr) / 2.0;
            confidence += signal_quality.max(0.0).min(1.0) * 0.3;
        }
        
        // Consistency with recent predictions (if available)
        // This would use prediction history in a full implementation
        confidence += 0.3; // Placeholder
        
        confidence.min(1.0)
    }
    
    /// Find candidate target cells
    fn find_candidate_cells(&self, features: &FeatureVector) -> Result<Vec<CandidateCell>> {
        let mut candidates = Vec::new();
        
        // Extract neighbor information from features
        // In a full implementation, this would use actual neighbor cell data
        // For now, we'll simulate some candidate cells
        
        for i in 0..self.configuration.candidate_cell_limit {
            let cell_id = format!("Cell_{}", i + 1);
            
            // Simulate cell properties
            let rsrp_gain = 3.0 + i as f64; // dB
            let sinr_gain = 2.0 + i as f64 * 0.5; // dB
            let load_factor = 0.3 + (i as f64 * 0.1); // 30-80% load
            let distance_penalty = i as f64 * 0.1; // Distance penalty
            
            // Calculate suitability score
            let suitability_score = self.calculate_suitability_score(
                rsrp_gain, sinr_gain, load_factor, distance_penalty
            );
            
            // Calculate handover probability for this cell
            let handover_probability = suitability_score * 0.8; // Simplified
            
            // Overall score combines multiple factors
            let overall_score = (suitability_score * 0.6) + 
                               ((rsrp_gain + sinr_gain) / 10.0 * 0.4);
            
            candidates.push(CandidateCell {
                cell_id,
                handover_probability,
                suitability_score,
                rsrp_gain,
                sinr_gain,
                load_factor,
                distance_penalty,
                overall_score,
            });
        }
        
        // Sort by overall score
        candidates.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());
        
        // Filter candidates that meet minimum requirements
        candidates.retain(|c| {
            c.rsrp_gain >= self.configuration.min_rsrp_gain_db &&
            c.sinr_gain >= self.configuration.min_sinr_gain_db
        });
        
        Ok(candidates)
    }
    
    /// Calculate cell suitability score
    fn calculate_suitability_score(
        &self, 
        rsrp_gain: f64, 
        sinr_gain: f64, 
        load_factor: f64, 
        distance_penalty: f64
    ) -> f64 {
        let mut score = 0.0;
        
        // Signal quality improvement (40% weight)
        let signal_improvement = (rsrp_gain / 10.0 + sinr_gain / 10.0) / 2.0;
        score += signal_improvement.min(1.0) * 0.4;
        
        // Load balancing (30% weight) - prefer less loaded cells
        if self.configuration.enable_load_balancing {
            let load_score = (1.0 - load_factor).max(0.0);
            score += load_score * 0.3;
        }
        
        // Distance factor (20% weight) - prefer closer cells
        let distance_score = (1.0 - distance_penalty).max(0.0);
        score += distance_score * 0.2;
        
        // Base suitability (10% weight)
        score += 0.1;
        
        score.min(1.0)
    }
    
    /// Select the best target cell
    fn select_target_cell(&self, candidates: &[CandidateCell]) -> Option<CandidateCell> {
        candidates.first().cloned()
    }
    
    /// Calculate decision factors that influenced the prediction
    fn calculate_decision_factors(&self, features: &FeatureVector) -> DecisionFactors {
        let mut time_factors = HashMap::new();
        let mut feature_contributions = HashMap::new();
        
        // Extract key features (simplified)
        let serving_rsrp = features.features.get(0).unwrap_or(&0.0);
        let serving_sinr = features.features.get(1).unwrap_or(&0.0);
        let ue_speed = features.features.get(3).unwrap_or(&0.0);
        let neighbor_rsrp = features.features.get(4).unwrap_or(&0.0);
        
        // Calculate factor scores
        let serving_cell_quality = (serving_rsrp + serving_sinr) / 2.0;
        let mobility_factor = ue_speed / 100.0; // Normalize speed
        let neighbor_advantage = (neighbor_rsrp - serving_rsrp).max(0.0) / 10.0;
        let load_balancing = 0.5; // Placeholder
        let urgency_score = self.calculate_urgency_from_features(features);
        
        // Time-based factors
        time_factors.insert("hour_of_day".to_string(), 0.5);
        time_factors.insert("day_of_week".to_string(), 0.3);
        time_factors.insert("is_busy_hour".to_string(), 0.7);
        
        // Feature contributions (top features)
        for (i, feature_name) in features.feature_names.iter().enumerate().take(10) {
            if let Some(&value) = features.features.get(i) {
                feature_contributions.insert(feature_name.clone(), value);
            }
        }
        
        DecisionFactors {
            serving_cell_quality,
            mobility_factor,
            neighbor_advantage,
            load_balancing,
            urgency_score,
            time_factors,
            feature_contributions,
        }
    }
    
    /// Calculate urgency score from features
    fn calculate_urgency_from_features(&self, features: &FeatureVector) -> f64 {
        let mut urgency = 0.0;
        
        // Poor serving cell quality increases urgency
        if let Some(&rsrp) = features.features.get(0) {
            if rsrp < -0.5 { // Normalized poor RSRP
                urgency += 0.3;
            }
        }
        
        // High mobility increases urgency
        if let Some(&speed) = features.features.get(3) {
            if speed > 0.6 { // Normalized high speed
                urgency += 0.2;
            }
        }
        
        // Strong neighbor available
        if let Some(&neighbor_rsrp) = features.features.get(4) {
            if let Some(&serving_rsrp) = features.features.get(0) {
                if neighbor_rsrp > serving_rsrp + 0.3 { // Significant neighbor advantage
                    urgency += 0.3;
                }
            }
        }
        
        // Time-sensitive factors
        urgency += 0.2; // Base urgency
        
        urgency.min(1.0)
    }
    
    /// Generate handover recommendation based on prediction
    fn generate_recommendation(
        &self,
        ho_probability: f64,
        confidence: f64,
        target_cell: &Option<CandidateCell>,
        candidates: &[CandidateCell],
        decision_factors: &DecisionFactors,
    ) -> HandoverRecommendation {
        // Check confidence threshold
        if confidence < self.configuration.confidence_threshold {
            return HandoverRecommendation::Monitor {
                reasons: vec!["Low prediction confidence".to_string()],
                next_check_ms: 5000, // Check again in 5 seconds
            };
        }
        
        // Check handover threshold
        if ho_probability < self.configuration.handover_threshold {
            return HandoverRecommendation::NoHandover {
                reasons: vec![
                    "Handover probability below threshold".to_string(),
                    format!("Current serving cell quality adequate: {:.2}", 
                           decision_factors.serving_cell_quality)
                ],
                confidence,
            };
        }
        
        // Check if we have a suitable target cell
        let target = match target_cell {
            Some(cell) => cell,
            None => {
                return HandoverRecommendation::Monitor {
                    reasons: vec!["No suitable target cell found".to_string()],
                    next_check_ms: 10000,
                };
            }
        };
        
        // Determine urgency level
        let urgency = if decision_factors.urgency_score > 0.8 {
            UrgencyLevel::Critical
        } else if decision_factors.urgency_score > 0.6 {
            UrgencyLevel::High
        } else if decision_factors.urgency_score > 0.4 {
            UrgencyLevel::Medium
        } else {
            UrgencyLevel::Low
        };
        
        // Determine if immediate or scheduled
        match urgency {
            UrgencyLevel::Critical | UrgencyLevel::High => {
                HandoverRecommendation::Immediate {
                    target_cell_id: target.cell_id.clone(),
                    urgency,
                    expected_gain: target.rsrp_gain + target.sinr_gain,
                }
            },
            UrgencyLevel::Medium => {
                HandoverRecommendation::Scheduled {
                    target_cell_id: target.cell_id.clone(),
                    recommended_delay_ms: 2000, // 2 second delay
                    conditions: vec![
                        "Confirm target cell availability".to_string(),
                        "Check load balancing impact".to_string(),
                    ],
                }
            },
            UrgencyLevel::Low => {
                HandoverRecommendation::Monitor {
                    reasons: vec![
                        format!("Target cell available: {}", target.cell_id),
                        "Low urgency, continue monitoring".to_string(),
                    ],
                    next_check_ms: 15000, // Check again in 15 seconds
                }
            }
        }
    }
    
    /// Store prediction result in history
    fn store_prediction_history(&mut self, ue_id: &str, result: &PredictionResult) {
        let history = self.prediction_history.entry(ue_id.to_string()).or_insert_with(Vec::new);
        history.push(result.clone());
        
        // Keep only recent predictions (last 100)
        if history.len() > 100 {
            history.remove(0);
        }
    }
    
    /// Reset feature extractor
    pub fn reset(&mut self) {
        self.feature_extractor.reset();
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &PredictorConfig {
        &self.configuration
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: PredictorConfig) {
        self.configuration = config;
    }
}

impl Default for HandoverPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::UeMetrics;
    use crate::features::FeatureExtractor;
    
    #[test]
    fn test_predictor_creation() {
        let predictor = HandoverPredictor::new();
        assert!(predictor.model.is_none());
    }
    
    #[test]
    fn test_confidence_calculation() {
        let predictor = HandoverPredictor::new();
        let features = FeatureVector {
            features: vec![0.5, 0.7, 0.3], // Sample normalized features
            feature_names: vec!["rsrp".to_string(), "sinr".to_string(), "speed".to_string()],
            timestamp: chrono::Utc::now(),
            ue_id: "test".to_string(),
        };
        
        let confidence = predictor.calculate_confidence(0.8, &features);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
    
    #[test]
    fn test_candidate_cell_finding() {
        let predictor = HandoverPredictor::new();
        let features = FeatureVector {
            features: vec![0.5; 10], // Sample features
            feature_names: (0..10).map(|i| format!("feature_{}", i)).collect(),
            timestamp: chrono::Utc::now(),
            ue_id: "test".to_string(),
        };
        
        let candidates = predictor.find_candidate_cells(&features).unwrap();
        assert!(!candidates.is_empty());
        assert!(candidates.len() <= predictor.configuration.candidate_cell_limit);
    }
}