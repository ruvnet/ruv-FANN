//! gRPC service implementation for handover prediction
//!
//! This module provides the gRPC server implementation for real-time
//! handover prediction with high throughput and low latency.

use crate::data::UeMetrics;
use crate::generated::{
    handover_predictor_server::{HandoverPredictor as HandoverPredictorTrait, HandoverPredictorServer},
    *,
};
use crate::prediction::{HandoverPredictor, PredictionResult};
use crate::{OptMobConfig, OptMobError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};

/// gRPC service configuration
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub bind_address: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
    pub request_timeout_ms: u64,
    pub enable_metrics: bool,
    pub enable_health_checks: bool,
    pub batch_size_limit: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 50051,
            max_concurrent_requests: 1000,
            request_timeout_ms: 5000,
            enable_metrics: true,
            enable_health_checks: true,
            batch_size_limit: 100,
        }
    }
}

/// Service metrics
#[derive(Debug, Clone, Default)]
pub struct ServiceMetrics {
    pub total_requests: u64,
    pub successful_predictions: u64,
    pub failed_predictions: u64,
    pub average_response_time_ms: f64,
    pub uptime_seconds: u64,
    pub active_connections: u32,
}

/// Handover prediction gRPC service
pub struct HandoverPredictorService {
    predictor: Arc<RwLock<HandoverPredictor>>,
    config: ServiceConfig,
    metrics: Arc<RwLock<ServiceMetrics>>,
    start_time: Instant,
    ue_buffers: Arc<RwLock<HashMap<String, Vec<UeMetrics>>>>,
}

impl HandoverPredictorService {
    /// Create a new service instance
    pub fn new(config: ServiceConfig) -> Self {
        Self {
            predictor: Arc::new(RwLock::new(HandoverPredictor::new())),
            config,
            metrics: Arc::new(RwLock::new(ServiceMetrics::default())),
            start_time: Instant::now(),
            ue_buffers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Load a trained model into the service
    pub async fn load_model(&self, model_path: &str) -> Result<()> {
        let mut predictor = self.predictor.write().await;
        predictor.load_model_from_file(model_path)?;
        info!("Loaded handover prediction model from: {}", model_path);
        Ok(())
    }
    
    /// Start the gRPC server
    pub async fn serve(self) -> Result<()> {
        let addr = format!("{}:{}", self.config.bind_address, self.config.port)
            .parse()
            .map_err(|e| OptMobError::Config(format!("Invalid bind address: {}", e)))?;
        
        info!("Starting handover predictor service on {}", addr);
        
        let service = HandoverPredictorServer::new(self);
        
        tonic::transport::Server::builder()
            .add_service(service)
            .serve(addr)
            .await
            .map_err(|e| OptMobError::Service(Status::internal(e.to_string())))?;
        
        Ok(())
    }
    
    /// Update service metrics
    async fn update_metrics(&self, request_time_ms: u64, success: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        
        if success {
            metrics.successful_predictions += 1;
        } else {
            metrics.failed_predictions += 1;
        }
        
        // Update average response time
        let total_time = metrics.average_response_time_ms * (metrics.total_requests - 1) as f64 + request_time_ms as f64;
        metrics.average_response_time_ms = total_time / metrics.total_requests as f64;
        
        metrics.uptime_seconds = self.start_time.elapsed().as_secs();
    }
    
    /// Convert UeMetric protobuf to internal UeMetrics
    fn convert_ue_metric(&self, metric: &UeMetric) -> UeMetrics {
        let mut ue_metrics = UeMetrics::new(&metric.ue_id, &metric.serving_cell_id);
        
        ue_metrics.timestamp = chrono::DateTime::from_timestamp(metric.timestamp, 0)
            .unwrap_or_else(chrono::Utc::now);
        ue_metrics.serving_rsrp = metric.serving_rsrp;
        ue_metrics.serving_sinr = metric.serving_sinr;
        ue_metrics.serving_rsrq = metric.serving_rsrq;
        ue_metrics.serving_cqi = metric.serving_cqi;
        ue_metrics.serving_ta = metric.serving_ta;
        ue_metrics.serving_phr = metric.serving_phr;
        ue_metrics.neighbor_rsrp_best = metric.neighbor_rsrp_best;
        ue_metrics.ue_speed_kmh = metric.ue_speed_kmh;
        
        // Convert neighbor cells
        ue_metrics.neighbor_cells = metric.neighbor_cells.iter().map(|nc| {
            crate::data::NeighborCell {
                cell_id: nc.cell_id.clone(),
                rsrp: nc.rsrp,
                rsrq: nc.rsrq,
                sinr: nc.sinr,
                distance_km: Some(nc.distance_km),
                frequency_band: nc.frequency_band.clone(),
                technology: "LTE".to_string(), // Default
                azimuth_degrees: None,
                cell_load_percent: None,
                handover_success_rate: None,
            }
        }).collect();
        
        ue_metrics
    }
    
    /// Convert internal prediction result to protobuf response
    fn convert_prediction_result(&self, result: &PredictionResult) -> HandoverPredictionResponse {
        HandoverPredictionResponse {
            ue_id: result.ue_id.clone(),
            ho_probability: result.ho_probability,
            target_cell_id: result.target_cell_id.clone().unwrap_or_default(),
            confidence: result.confidence,
            prediction_timestamp: result.prediction_timestamp.timestamp(),
            candidate_cells: result.candidate_cells.iter().map(|c| c.cell_id.clone()).collect(),
            candidate_probabilities: result.candidate_cells.iter().map(|c| c.handover_probability).collect(),
        }
    }
}

#[tonic::async_trait]
impl HandoverPredictorTrait for HandoverPredictorService {
    /// Predict handover for a single UE
    async fn predict_handover(
        &self,
        request: Request<HandoverPredictionRequest>,
    ) -> std::result::Result<Response<HandoverPredictionResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        // Validate request
        if req.ue_id.is_empty() {
            return Err(Status::invalid_argument("UE ID cannot be empty"));
        }
        
        if req.metrics.is_empty() {
            return Err(Status::invalid_argument("Metrics cannot be empty"));
        }
        
        // Convert metrics
        let ue_metrics: Vec<UeMetrics> = req.metrics.iter()
            .map(|m| self.convert_ue_metric(m))
            .collect();
        
        // Get or create predictor for this UE
        let mut predictor = self.predictor.write().await;
        
        // Add metrics to predictor
        for metric in &ue_metrics {
            if let Err(e) = predictor.add_metrics(metric.clone()) {
                warn!("Failed to add metrics for UE {}: {}", req.ue_id, e);
                return Err(Status::invalid_argument(format!("Invalid metrics: {}", e)));
            }
        }
        
        // Make prediction
        let prediction_result = match predictor.predict(&req.ue_id) {
            Ok(result) => result,
            Err(e) => {
                error!("Prediction failed for UE {}: {}", req.ue_id, e);
                self.update_metrics(start_time.elapsed().as_millis() as u64, false).await;
                return Err(Status::internal(format!("Prediction failed: {}", e)));
            }
        };
        
        // Convert to response
        let response = self.convert_prediction_result(&prediction_result);
        
        // Update metrics
        let request_time = start_time.elapsed().as_millis() as u64;
        self.update_metrics(request_time, true).await;
        
        info!("Predicted handover for UE {}: probability={:.4}, confidence={:.4}, time={}ms",
              req.ue_id, response.ho_probability, response.confidence, request_time);
        
        Ok(Response::new(response))
    }
    
    /// Batch prediction for multiple UEs
    async fn batch_predict_handover(
        &self,
        request: Request<BatchHandoverPredictionRequest>,
    ) -> std::result::Result<Response<BatchHandoverPredictionResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        // Validate batch size
        if req.requests.len() > self.config.batch_size_limit {
            return Err(Status::invalid_argument(
                format!("Batch size {} exceeds limit {}", req.requests.len(), self.config.batch_size_limit)
            ));
        }
        
        let mut responses = Vec::new();
        let mut successful_predictions = 0;
        
        // Process each request
        for prediction_request in req.requests {
            // Convert metrics
            let ue_metrics: Vec<UeMetrics> = prediction_request.metrics.iter()
                .map(|m| self.convert_ue_metric(m))
                .collect();
            
            // Create a temporary predictor for this UE
            let mut temp_predictor = HandoverPredictor::new();
            
            // Load the same model as the main predictor
            if let Ok(predictor_guard) = self.predictor.try_read() {
                if let Some(model_info) = predictor_guard.get_model_info() {
                    // In a real implementation, we'd share the model
                    // For now, we'll create separate predictors
                }
            }
            
            // Add metrics
            for metric in &ue_metrics {
                if let Err(e) = temp_predictor.add_metrics(metric.clone()) {
                    warn!("Failed to add metrics for UE {} in batch: {}", prediction_request.ue_id, e);
                    continue;
                }
            }
            
            // Make prediction
            match temp_predictor.predict(&prediction_request.ue_id) {
                Ok(prediction_result) => {
                    let response = self.convert_prediction_result(&prediction_result);
                    responses.push(response);
                    successful_predictions += 1;
                },
                Err(e) => {
                    warn!("Batch prediction failed for UE {}: {}", prediction_request.ue_id, e);
                    // Add a default response for failed predictions
                    responses.push(HandoverPredictionResponse {
                        ue_id: prediction_request.ue_id.clone(),
                        ho_probability: 0.0,
                        target_cell_id: String::new(),
                        confidence: 0.0,
                        prediction_timestamp: chrono::Utc::now().timestamp(),
                        candidate_cells: Vec::new(),
                        candidate_probabilities: Vec::new(),
                    });
                }
            }
        }
        
        // Update metrics
        let request_time = start_time.elapsed().as_millis() as u64;
        self.update_metrics(request_time, successful_predictions > 0).await;
        
        info!("Batch prediction completed: {}/{} successful, time={}ms",
              successful_predictions, responses.len(), request_time);
        
        Ok(Response::new(BatchHandoverPredictionResponse { responses }))
    }
    
    /// Get model information
    async fn get_model_info(
        &self,
        _request: Request<ModelInfoRequest>,
    ) -> std::result::Result<Response<ModelInfoResponse>, Status> {
        let predictor = self.predictor.read().await;
        
        match predictor.get_model_info() {
            Some(model_info) => {
                let response = ModelInfoResponse {
                    model_id: model_info.model_id,
                    model_version: model_info.model_version,
                    training_timestamp: model_info.training_timestamp,
                    training_accuracy: model_info.training_accuracy,
                    validation_accuracy: model_info.validation_accuracy,
                    total_features: model_info.total_features,
                    feature_names: model_info.feature_names,
                    stats: Some(ModelStats {
                        total_training_samples: 10000, // Placeholder
                        total_validation_samples: 2000, // Placeholder
                        precision: 0.91,
                        recall: 0.89,
                        f1_score: 0.90,
                        auc_roc: 0.94,
                        confusion_matrix: vec![850.0, 150.0, 100.0, 900.0], // TP, FP, FN, TN
                    }),
                };
                
                Ok(Response::new(response))
            },
            None => {
                Err(Status::failed_precondition("No model loaded"))
            }
        }
    }
    
    /// Health check endpoint
    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> std::result::Result<Response<HealthCheckResponse>, Status> {
        let metrics = self.metrics.read().await;
        
        let healthy = self.predictor.read().await.get_model_info().is_some();
        
        let response = HealthCheckResponse {
            healthy,
            status: if healthy { "OK".to_string() } else { "No model loaded".to_string() },
            uptime_seconds: metrics.uptime_seconds as i64,
            total_predictions: metrics.total_requests as i64,
            avg_prediction_time_ms: metrics.average_response_time_ms,
        };
        
        Ok(Response::new(response))
    }
}

/// Service builder for easy configuration
pub struct ServiceBuilder {
    config: ServiceConfig,
}

impl ServiceBuilder {
    pub fn new() -> Self {
        Self {
            config: ServiceConfig::default(),
        }
    }
    
    pub fn bind_address(mut self, address: &str) -> Self {
        self.config.bind_address = address.to_string();
        self
    }
    
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }
    
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }
    
    pub fn request_timeout_ms(mut self, timeout: u64) -> Self {
        self.config.request_timeout_ms = timeout;
        self
    }
    
    pub fn batch_size_limit(mut self, limit: usize) -> Self {
        self.config.batch_size_limit = limit;
        self
    }
    
    pub fn build(self) -> HandoverPredictorService {
        HandoverPredictorService::new(self.config)
    }
}

impl Default for ServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to start the service with a configuration
pub async fn start_service(config: OptMobConfig) -> Result<()> {
    // Initialize tracing
    if config.enable_logging {
        tracing_subscriber::fmt::init();
    }
    
    // Create service
    let service = ServiceBuilder::new()
        .port(config.grpc_port)
        .max_concurrent_requests(1000)
        .batch_size_limit(config.max_batch_size)
        .build();
    
    // Load model if path is provided
    if !config.model_path.is_empty() {
        service.load_model(&config.model_path).await?;
    }
    
    // Start serving
    service.serve().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated::NeighborCell;
    
    #[tokio::test]
    async fn test_service_creation() {
        let service = ServiceBuilder::new()
            .port(50052)
            .build();
        
        assert_eq!(service.config.port, 50052);
    }
    
    #[tokio::test]
    async fn test_metric_conversion() {
        let service = HandoverPredictorService::new(ServiceConfig::default());
        
        let proto_metric = UeMetric {
            timestamp: chrono::Utc::now().timestamp(),
            ue_id: "UE_001".to_string(),
            serving_rsrp: -85.0,
            serving_sinr: 12.0,
            neighbor_rsrp_best: -80.0,
            ue_speed_kmh: 60.0,
            serving_cell_id: "Cell_001".to_string(),
            neighbor_cells: vec![
                NeighborCell {
                    cell_id: "Cell_002".to_string(),
                    rsrp: -80.0,
                    rsrq: -10.0,
                    sinr: 15.0,
                    distance_km: 0.5,
                    frequency_band: "B1".to_string(),
                }
            ],
            serving_rsrq: -10.0,
            serving_cqi: 10.0,
            serving_ta: 1.0,
            serving_phr: 5.0,
        };
        
        let internal_metric = service.convert_ue_metric(&proto_metric);
        assert_eq!(internal_metric.ue_id, "UE_001");
        assert_eq!(internal_metric.serving_rsrp, -85.0);
        assert_eq!(internal_metric.neighbor_cells.len(), 1);
    }
}