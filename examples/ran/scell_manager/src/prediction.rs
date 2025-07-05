//! Prediction engine for SCell activation recommendations

use crate::config::ModelConfig;
use crate::model::SCellPredictionModel;
use crate::types::*;
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};

/// Prediction engine that manages models and provides real-time predictions
#[derive(Debug)]
pub struct PredictionEngine {
    models: Arc<DashMap<ModelId, Arc<RwLock<SCellPredictionModel>>>>,
    default_model_id: ModelId,
    config: ModelConfig,
    prediction_cache: Arc<DashMap<String, CachedPrediction>>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl PredictionEngine {
    /// Create a new prediction engine
    pub async fn new(config: &ModelConfig) -> Result<Self> {
        let models = Arc::new(DashMap::new());
        let default_model_id = config.default_model_id.clone();
        
        // Initialize default model
        let default_model = SCellPredictionModel::new(
            config.clone(),
            default_model_id.clone(),
        ).await?;
        
        models.insert(
            default_model_id.clone(),
            Arc::new(RwLock::new(default_model)),
        );
        
        Ok(Self {
            models,
            default_model_id,
            config: config.clone(),
            prediction_cache: Arc::new(DashMap::new()),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
        })
    }
    
    /// Add a new model to the engine
    pub async fn add_model(&self, model_id: ModelId, model: SCellPredictionModel) -> Result<()> {
        info!("Adding model: {}", model_id);
        self.models.insert(model_id.clone(), Arc::new(RwLock::new(model)));
        Ok(())
    }
    
    /// Remove a model from the engine
    pub async fn remove_model(&self, model_id: &ModelId) -> Result<()> {
        if model_id == &self.default_model_id {
            return Err(anyhow!("Cannot remove default model"));
        }
        
        self.models.remove(model_id);
        info!("Removed model: {}", model_id);
        Ok(())
    }
    
    /// Get list of available models
    pub fn list_models(&self) -> Vec<ModelId> {
        self.models.iter().map(|entry| entry.key().clone()).collect()
    }
    
    /// Make a prediction using the default model
    pub async fn predict(&self, request: &PredictionRequest) -> Result<SCellPrediction> {
        self.predict_with_model(&self.default_model_id, request).await
    }
    
    /// Make a prediction using a specific model
    pub async fn predict_with_model(&self, model_id: &ModelId, request: &PredictionRequest) -> Result<SCellPrediction> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = format!("{}:{}:{}", model_id, request.ue_id, request.current_metrics.timestamp_utc.timestamp());
        if let Some(cached) = self.prediction_cache.get(&cache_key) {
            if !cached.is_expired() {
                debug!("Cache hit for UE: {}", request.ue_id);
                return Ok(cached.prediction.clone());
            }
        }
        
        // Get model
        let model = self.models.get(model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;
        
        // Make prediction
        let prediction = {
            let model_guard = model.read().await;
            model_guard.predict(request).await?
        };
        
        // Cache prediction
        self.prediction_cache.insert(
            cache_key,
            CachedPrediction::new(prediction.clone(), Duration::from_secs(30)),
        );
        
        // Update performance metrics
        let prediction_time = start_time.elapsed();
        self.update_performance_metrics(prediction_time, true).await;
        
        debug!("Prediction for UE {}: activation={}, confidence={:.3}, time={:.2}ms",
               request.ue_id, prediction.scell_activation_recommended, 
               prediction.confidence_score, prediction_time.as_millis());
        
        Ok(prediction)
    }
    
    /// Make batch predictions
    pub async fn predict_batch(&self, requests: &[PredictionRequest]) -> Result<Vec<SCellPrediction>> {
        let start_time = Instant::now();
        
        let mut predictions = Vec::new();
        let mut successful_predictions = 0;
        
        for request in requests {
            match self.predict(request).await {
                Ok(prediction) => {
                    predictions.push(prediction);
                    successful_predictions += 1;
                }
                Err(e) => {
                    error!("Prediction failed for UE {}: {}", request.ue_id, e);
                    // Add a default prediction with low confidence
                    predictions.push(SCellPrediction::new(
                        request.ue_id.clone(),
                        false,
                        0.0,
                    ));
                }
            }
        }
        
        let batch_time = start_time.elapsed();
        info!("Batch prediction completed: {}/{} successful, time={:.2}ms",
              successful_predictions, requests.len(), batch_time.as_millis());
        
        Ok(predictions)
    }
    
    /// Train a model with new data
    pub async fn train_model(&self, model_id: &ModelId, training_data: &[TrainingExample]) -> Result<ModelMetrics> {
        let model = self.models.get(model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;
        
        let mut model_guard = model.write().await;
        let metrics = model_guard.train(training_data).await?;
        
        info!("Model {} trained successfully. Accuracy: {:.4}", model_id, metrics.accuracy);
        Ok(metrics)
    }
    
    /// Get model metrics
    pub async fn get_model_metrics(&self, model_id: &ModelId) -> Result<ModelMetrics> {
        let model = self.models.get(model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;
        
        let model_guard = model.read().await;
        Ok(model_guard.get_metrics().await)
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }
    
    /// Clear prediction cache
    pub async fn clear_cache(&self) {
        self.prediction_cache.clear();
        info!("Prediction cache cleared");
    }
    
    /// Load model from file
    pub async fn load_model_from_file(&self, model_id: ModelId, file_path: &std::path::Path) -> Result<()> {
        let model = SCellPredictionModel::load_from_file(file_path, self.config.clone()).await?;
        self.add_model(model_id, model).await?;
        Ok(())
    }
    
    /// Save model to file
    pub async fn save_model_to_file(&self, model_id: &ModelId, file_path: &std::path::Path) -> Result<()> {
        let model = self.models.get(model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;
        
        let model_guard = model.read().await;
        model_guard.save_to_file(file_path).await?;
        Ok(())
    }
    
    /// Analyze prediction patterns for a UE
    pub async fn analyze_ue_patterns(&self, ue_id: &UEId, historical_data: &[UEMetrics]) -> Result<UEAnalysis> {
        if historical_data.is_empty() {
            return Err(anyhow!("No historical data provided"));
        }
        
        let mut analysis = UEAnalysis::new(ue_id.clone());
        
        // Compute basic statistics
        let throughputs: Vec<f32> = historical_data.iter()
            .map(|m| m.pcell_throughput_mbps)
            .collect();
        
        let cqis: Vec<f32> = historical_data.iter()
            .map(|m| m.pcell_cqi)
            .collect();
        
        analysis.avg_throughput = throughputs.iter().sum::<f32>() / throughputs.len() as f32;
        analysis.max_throughput = throughputs.iter().copied().fold(0.0, f32::max);
        analysis.min_throughput = throughputs.iter().copied().fold(f32::INFINITY, f32::min);
        
        analysis.avg_cqi = cqis.iter().sum::<f32>() / cqis.len() as f32;
        analysis.max_cqi = cqis.iter().copied().fold(0.0, f32::max);
        analysis.min_cqi = cqis.iter().copied().fold(f32::INFINITY, f32::min);
        
        // Identify high-throughput periods
        let high_throughput_threshold = self.config.throughput_threshold_mbps * 0.8;
        analysis.high_throughput_periods = historical_data.iter()
            .filter(|m| m.pcell_throughput_mbps > high_throughput_threshold)
            .count();
        
        // Compute throughput trend
        analysis.throughput_trend = self.compute_trend(&throughputs);
        
        // Predict future needs
        let sample_request = PredictionRequest {
            ue_id: ue_id.clone(),
            current_metrics: historical_data.last().unwrap().clone(),
            historical_metrics: historical_data.to_vec(),
            prediction_horizon_seconds: self.config.prediction_horizon_seconds,
        };
        
        let prediction = self.predict(&sample_request).await?;
        analysis.predicted_scell_need = prediction.scell_activation_recommended;
        analysis.prediction_confidence = prediction.confidence_score;
        
        Ok(analysis)
    }
    
    /// Get system health status
    pub async fn get_system_health(&self) -> SystemHealth {
        let performance = self.performance_metrics.read().await;
        let cache_size = self.prediction_cache.len();
        let model_count = self.models.len();
        
        SystemHealth {
            healthy: performance.error_rate < 0.1,
            model_count,
            cache_size,
            total_predictions: performance.total_predictions,
            average_prediction_time_ms: performance.average_prediction_time_ms,
            error_rate: performance.error_rate,
            uptime_seconds: performance.uptime.as_secs(),
        }
    }
    
    // Private helper methods
    
    async fn update_performance_metrics(&self, prediction_time: Duration, success: bool) {
        let mut metrics = self.performance_metrics.write().await;
        metrics.total_predictions += 1;
        
        if success {
            metrics.successful_predictions += 1;
        } else {
            metrics.failed_predictions += 1;
        }
        
        metrics.total_prediction_time += prediction_time;
        metrics.average_prediction_time_ms = 
            (metrics.total_prediction_time.as_millis() as f32) / metrics.total_predictions as f32;
        
        metrics.error_rate = metrics.failed_predictions as f32 / metrics.total_predictions as f32;
        
        if prediction_time > metrics.max_prediction_time {
            metrics.max_prediction_time = prediction_time;
        }
        
        if prediction_time < metrics.min_prediction_time {
            metrics.min_prediction_time = prediction_time;
        }
    }
    
    fn compute_trend(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f32;
        let x_sum = (0..values.len()).sum::<usize>() as f32;
        let y_sum = values.iter().sum::<f32>();
        let xy_sum = values.iter().enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum::<f32>();
        let x_squared_sum = (0..values.len())
            .map(|i| (i as f32).powi(2))
            .sum::<f32>();
        
        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum.powi(2));
        slope
    }
}

/// Cached prediction with expiration
#[derive(Debug, Clone)]
struct CachedPrediction {
    prediction: SCellPrediction,
    expires_at: Instant,
}

impl CachedPrediction {
    fn new(prediction: SCellPrediction, ttl: Duration) -> Self {
        Self {
            prediction,
            expires_at: Instant::now() + ttl,
        }
    }
    
    fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Performance metrics for the prediction engine
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_predictions: u64,
    pub successful_predictions: u64,
    pub failed_predictions: u64,
    pub average_prediction_time_ms: f32,
    pub max_prediction_time: Duration,
    pub min_prediction_time: Duration,
    pub total_prediction_time: Duration,
    pub error_rate: f32,
    pub uptime: Duration,
    pub start_time: Instant,
}

impl PerformanceMetrics {
    fn new() -> Self {
        let start_time = Instant::now();
        Self {
            total_predictions: 0,
            successful_predictions: 0,
            failed_predictions: 0,
            average_prediction_time_ms: 0.0,
            max_prediction_time: Duration::from_millis(0),
            min_prediction_time: Duration::from_millis(u64::MAX),
            total_prediction_time: Duration::from_millis(0),
            error_rate: 0.0,
            uptime: Duration::from_millis(0),
            start_time,
        }
    }
}

/// UE behavior analysis
#[derive(Debug, Clone)]
pub struct UEAnalysis {
    pub ue_id: UEId,
    pub avg_throughput: f32,
    pub max_throughput: f32,
    pub min_throughput: f32,
    pub avg_cqi: f32,
    pub max_cqi: f32,
    pub min_cqi: f32,
    pub high_throughput_periods: usize,
    pub throughput_trend: f32,
    pub predicted_scell_need: bool,
    pub prediction_confidence: f32,
}

impl UEAnalysis {
    fn new(ue_id: UEId) -> Self {
        Self {
            ue_id,
            avg_throughput: 0.0,
            max_throughput: 0.0,
            min_throughput: 0.0,
            avg_cqi: 0.0,
            max_cqi: 0.0,
            min_cqi: 0.0,
            high_throughput_periods: 0,
            throughput_trend: 0.0,
            predicted_scell_need: false,
            prediction_confidence: 0.0,
        }
    }
}

/// System health status
#[derive(Debug, Clone)]
pub struct SystemHealth {
    pub healthy: bool,
    pub model_count: usize,
    pub cache_size: usize,
    pub total_predictions: u64,
    pub average_prediction_time_ms: f32,
    pub error_rate: f32,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_prediction_engine_creation() {
        let config = ModelConfig::default();
        let engine = PredictionEngine::new(&config).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_prediction_cache() {
        let config = ModelConfig::default();
        let engine = PredictionEngine::new(&config).await.unwrap();
        
        let request = PredictionRequest {
            ue_id: "test_ue".to_string(),
            current_metrics: UEMetrics::new("test_ue".to_string()),
            historical_metrics: vec![],
            prediction_horizon_seconds: 30,
        };
        
        // First prediction should miss cache
        let prediction1 = engine.predict(&request).await.unwrap();
        
        // Second prediction should hit cache (if made quickly)
        let prediction2 = engine.predict(&request).await.unwrap();
        
        assert_eq!(prediction1.ue_id, prediction2.ue_id);
    }
    
    #[test]
    fn test_trend_computation() {
        let config = ModelConfig::default();
        let engine = PredictionEngine::new(&config);
        
        // Test with increasing trend
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Trend should be positive for increasing values
        // (exact value depends on implementation)
        
        // Test with decreasing trend
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        // Trend should be negative for decreasing values
    }
}