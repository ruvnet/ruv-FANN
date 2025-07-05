//! gRPC service implementation for SCell Manager

use crate::config::SCellManagerConfig;
use crate::metrics::MetricsCollector;
use crate::prediction::PredictionEngine;
use crate::proto::s_cell_manager_service_server::{SCellManagerService as SCellManagerServiceTrait, SCellManagerServiceServer};
use crate::proto::*;
use crate::types::*;
use anyhow::Result;
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

/// gRPC service implementation
#[derive(Debug)]
pub struct SCellManagerService {
    prediction_engine: Arc<RwLock<PredictionEngine>>,
    metrics_collector: Arc<MetricsCollector>,
    config: SCellManagerConfig,
    start_time: Instant,
}

impl SCellManagerService {
    /// Create a new service instance
    pub fn new(
        prediction_engine: Arc<RwLock<PredictionEngine>>,
        metrics_collector: Arc<MetricsCollector>,
        config: SCellManagerConfig,
    ) -> Self {
        Self {
            prediction_engine,
            metrics_collector,
            config,
            start_time: Instant::now(),
        }
    }
    
    /// Convert protobuf UEMetrics to internal type
    fn convert_ue_metrics(&self, proto_metrics: &UeMetrics) -> crate::types::UEMetrics {
        crate::types::UEMetrics {
            ue_id: proto_metrics.ue_id.clone(),
            pcell_throughput_mbps: proto_metrics.pcell_throughput_mbps,
            buffer_status_report_bytes: proto_metrics.buffer_status_report_bytes,
            pcell_cqi: proto_metrics.pcell_cqi,
            pcell_rsrp: proto_metrics.pcell_rsrp,
            pcell_sinr: proto_metrics.pcell_sinr,
            active_bearers: proto_metrics.active_bearers,
            data_rate_req_mbps: proto_metrics.data_rate_req_mbps,
            timestamp_utc: DateTime::from_timestamp(proto_metrics.timestamp_utc, 0)
                .unwrap_or_else(|| Utc::now()),
        }
    }
    
    /// Convert internal SCellPrediction to protobuf
    fn convert_prediction_to_proto(&self, prediction: &crate::types::SCellPrediction) -> PredictScellNeedResponse {
        PredictScellNeedResponse {
            ue_id: prediction.ue_id.clone(),
            scell_activation_recommended: prediction.scell_activation_recommended,
            confidence_score: prediction.confidence_score,
            predicted_throughput_demand: prediction.predicted_throughput_demand,
            reasoning: prediction.reasoning.clone(),
            timestamp_utc: prediction.timestamp_utc.timestamp(),
        }
    }
    
    /// Convert internal ModelMetrics to protobuf
    fn convert_model_metrics_to_proto(&self, metrics: &crate::types::ModelMetrics) -> ModelMetrics {
        ModelMetrics {
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            f1_score: metrics.f1_score,
            auc_roc: metrics.auc_roc,
            mean_absolute_error: metrics.mean_absolute_error,
            total_predictions: metrics.total_predictions,
            true_positives: metrics.true_positives,
            false_positives: metrics.false_positives,
            true_negatives: metrics.true_negatives,
            false_negatives: metrics.false_negatives,
        }
    }
    
    /// Convert protobuf training config to internal type
    fn convert_training_config(&self, proto_config: &TrainingConfig) -> crate::types::TrainingConfig {
        crate::types::TrainingConfig {
            epochs: proto_config.epochs,
            learning_rate: proto_config.learning_rate,
            batch_size: proto_config.batch_size,
            validation_split: proto_config.validation_split,
            sequence_length: proto_config.sequence_length,
        }
    }
    
    /// Validate request parameters
    fn validate_prediction_request(&self, request: &PredictScellNeedRequest) -> Result<(), Status> {
        if request.ue_id.is_empty() {
            return Err(Status::invalid_argument("UE ID cannot be empty"));
        }
        
        if request.current_metrics.is_none() {
            return Err(Status::invalid_argument("Current metrics are required"));
        }
        
        if request.prediction_horizon_seconds <= 0 {
            return Err(Status::invalid_argument("Prediction horizon must be positive"));
        }
        
        let metrics = request.current_metrics.as_ref().unwrap();
        if metrics.pcell_throughput_mbps < 0.0 {
            return Err(Status::invalid_argument("Throughput cannot be negative"));
        }
        
        if metrics.pcell_cqi < 0.0 || metrics.pcell_cqi > 15.0 {
            return Err(Status::invalid_argument("CQI must be between 0 and 15"));
        }
        
        Ok(())
    }
}

#[tonic::async_trait]
impl SCellManagerServiceTrait for SCellManagerService {
    /// Predict SCell need for a UE
    async fn predict_scell_need(
        &self,
        request: Request<PredictScellNeedRequest>,
    ) -> Result<Response<PredictScellNeedResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        debug!("Received prediction request for UE: {}", req.ue_id);
        
        // Validate request
        self.validate_prediction_request(&req)?;
        
        // Convert protobuf types to internal types
        let current_metrics = self.convert_ue_metrics(req.current_metrics.as_ref().unwrap());
        let historical_metrics: Vec<crate::types::UEMetrics> = req.historical_metrics
            .iter()
            .map(|m| self.convert_ue_metrics(m))
            .collect();
        
        let prediction_request = crate::types::PredictionRequest {
            ue_id: req.ue_id.clone(),
            current_metrics,
            historical_metrics,
            prediction_horizon_seconds: req.prediction_horizon_seconds,
        };
        
        // Make prediction
        let prediction_result = {
            let engine = self.prediction_engine.read().await;
            engine.predict(&prediction_request).await
        };
        
        let duration = start_time.elapsed();
        
        match prediction_result {
            Ok(prediction) => {
                // Record metrics
                self.metrics_collector.record_prediction(
                    duration,
                    true,
                    prediction.confidence_score,
                    prediction.scell_activation_recommended,
                    prediction.predicted_throughput_demand,
                );
                
                let response = self.convert_prediction_to_proto(&prediction);
                
                info!("Prediction completed for UE {}: activation={}, confidence={:.3}, time={:.2}ms",
                      req.ue_id, prediction.scell_activation_recommended, 
                      prediction.confidence_score, duration.as_millis());
                
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Prediction failed for UE {}: {}", req.ue_id, e);
                
                // Record failed prediction
                self.metrics_collector.record_prediction(
                    duration,
                    false,
                    0.0,
                    false,
                    0.0,
                );
                
                Err(Status::internal(format!("Prediction failed: {}", e)))
            }
        }
    }
    
    /// Train model with new data
    async fn train_model(
        &self,
        request: Request<TrainModelRequest>,
    ) -> Result<Response<TrainModelResponse>, Status> {
        let req = request.into_inner();
        
        info!("Received training request for model: {}", req.model_id);
        
        if req.training_data.is_empty() {
            return Err(Status::invalid_argument("Training data cannot be empty"));
        }
        
        // Convert training data
        let mut training_examples = Vec::new();
        for example in &req.training_data {
            if let Some(ref input_metrics) = example.input_metrics {
                let historical_sequence = example.historical_sequence
                    .iter()
                    .map(|m| self.convert_ue_metrics(m))
                    .collect();
                
                training_examples.push(crate::types::TrainingExample {
                    input_metrics: self.convert_ue_metrics(input_metrics),
                    historical_sequence,
                    actual_scell_needed: example.actual_scell_needed,
                    actual_throughput_demand: example.actual_throughput_demand,
                });
            }
        }
        
        // Train model
        let training_result = {
            let engine = self.prediction_engine.read().await;
            engine.train_model(&req.model_id, &training_examples).await
        };
        
        match training_result {
            Ok(metrics) => {
                info!("Model {} training completed. Accuracy: {:.4}", req.model_id, metrics.accuracy);
                
                // Update model metrics
                self.metrics_collector.record_model_metrics(
                    metrics.accuracy,
                    metrics.precision,
                    metrics.recall,
                    1, // Assuming one model for now
                );
                
                let response = TrainModelResponse {
                    model_id: req.model_id,
                    success: true,
                    error_message: String::new(),
                    metrics: Some(self.convert_model_metrics_to_proto(&metrics)),
                };
                
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Training failed for model {}: {}", req.model_id, e);
                
                let response = TrainModelResponse {
                    model_id: req.model_id,
                    success: false,
                    error_message: e.to_string(),
                    metrics: None,
                };
                
                Ok(Response::new(response))
            }
        }
    }
    
    /// Get model performance metrics
    async fn get_model_metrics(
        &self,
        request: Request<GetModelMetricsRequest>,
    ) -> Result<Response<GetModelMetricsResponse>, Status> {
        let req = request.into_inner();
        
        debug!("Received metrics request for model: {}", req.model_id);
        
        let metrics_result = {
            let engine = self.prediction_engine.read().await;
            engine.get_model_metrics(&req.model_id).await
        };
        
        match metrics_result {
            Ok(metrics) => {
                let response = GetModelMetricsResponse {
                    model_id: req.model_id,
                    metrics: Some(self.convert_model_metrics_to_proto(&metrics)),
                    last_updated_utc: Utc::now().timestamp(),
                };
                
                Ok(Response::new(response))
            }
            Err(e) => {
                warn!("Failed to get metrics for model {}: {}", req.model_id, e);
                Err(Status::not_found(format!("Model not found: {}", req.model_id)))
            }
        }
    }
    
    /// Get system status
    async fn get_system_status(
        &self,
        _request: Request<GetSystemStatusRequest>,
    ) -> Result<Response<GetSystemStatusResponse>, Status> {
        debug!("Received system status request");
        
        let health = {
            let engine = self.prediction_engine.read().await;
            engine.get_system_health().await
        };
        
        let metrics_summary = self.metrics_collector.get_summary();
        
        let mut system_info = std::collections::HashMap::new();
        system_info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        system_info.insert("rust_version".to_string(), env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown").to_string());
        system_info.insert("total_predictions".to_string(), metrics_summary.total_predictions.to_string());
        system_info.insert("success_rate".to_string(), format!("{:.4}", metrics_summary.success_rate()));
        
        let response = GetSystemStatusResponse {
            healthy: health.healthy,
            version: env!("CARGO_PKG_VERSION").to_string(),
            active_models: health.model_count as i32,
            total_predictions: metrics_summary.total_predictions as i64,
            average_prediction_time_ms: health.average_prediction_time_ms,
            uptime_seconds: self.start_time.elapsed().as_secs() as i64,
            system_info,
        };
        
        Ok(Response::new(response))
    }
    
    /// Stream real-time predictions
    async fn stream_predictions(
        &self,
        request: Request<StreamPredictionsRequest>,
    ) -> Result<Response<Self::StreamPredictionsStream>, Status> {
        let req = request.into_inner();
        
        info!("Starting prediction stream for {} UEs", req.ue_ids.len());
        
        if req.ue_ids.is_empty() {
            return Err(Status::invalid_argument("UE IDs list cannot be empty"));
        }
        
        if req.update_interval_seconds <= 0 {
            return Err(Status::invalid_argument("Update interval must be positive"));
        }
        
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let prediction_engine = self.prediction_engine.clone();
        let ue_ids = req.ue_ids.clone();
        let update_interval = Duration::from_secs(req.update_interval_seconds as u64);
        
        // Spawn background task for streaming predictions
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(update_interval);
            
            loop {
                interval.tick().await;
                
                for ue_id in &ue_ids {
                    // Create a mock request for demonstration
                    // In a real implementation, you would get current metrics from a data source
                    let mock_metrics = crate::types::UEMetrics::new(ue_id.clone());
                    let prediction_request = crate::types::PredictionRequest {
                        ue_id: ue_id.clone(),
                        current_metrics: mock_metrics,
                        historical_metrics: vec![],
                        prediction_horizon_seconds: 30,
                    };
                    
                    let engine = prediction_engine.read().await;
                    match engine.predict(&prediction_request).await {
                        Ok(prediction) => {
                            let proto_prediction = PredictScellNeedResponse {
                                ue_id: prediction.ue_id.clone(),
                                scell_activation_recommended: prediction.scell_activation_recommended,
                                confidence_score: prediction.confidence_score,
                                predicted_throughput_demand: prediction.predicted_throughput_demand,
                                reasoning: prediction.reasoning,
                                timestamp_utc: prediction.timestamp_utc.timestamp(),
                            };
                            
                            let update = PredictionUpdate {
                                ue_id: ue_id.clone(),
                                prediction: Some(proto_prediction),
                                timestamp_utc: Utc::now().timestamp(),
                            };
                            
                            if tx.send(Ok(update)).await.is_err() {
                                warn!("Stream closed for UE: {}", ue_id);
                                return;
                            }
                        }
                        Err(e) => {
                            error!("Failed to predict for UE {}: {}", ue_id, e);
                        }
                    }
                }
            }
        });
        
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
    
    type StreamPredictionsStream = std::pin::Pin<Box<dyn tokio_stream::Stream<Item = Result<PredictionUpdate, Status>> + Send>>;
}

/// Create a gRPC server
pub fn create_server(service: SCellManagerService) -> SCellManagerServiceServer<SCellManagerService> {
    SCellManagerServiceServer::new(service)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SCellManagerConfig;
    use crate::prediction::PredictionEngine;
    use chrono::Utc;
    
    async fn create_test_service() -> SCellManagerService {
        let config = SCellManagerConfig::default();
        let prediction_engine = Arc::new(RwLock::new(
            PredictionEngine::new(&config.model_config).await.unwrap()
        ));
        let metrics_collector = Arc::new(MetricsCollector::new().unwrap());
        
        SCellManagerService::new(prediction_engine, metrics_collector, config)
    }
    
    #[tokio::test]
    async fn test_predict_scell_need() {
        let service = create_test_service().await;
        
        let request = PredictScellNeedRequest {
            ue_id: "test_ue".to_string(),
            current_metrics: Some(UeMetrics {
                ue_id: "test_ue".to_string(),
                pcell_throughput_mbps: 50.0,
                buffer_status_report_bytes: 1000,
                pcell_cqi: 10.0,
                pcell_rsrp: -80.0,
                pcell_sinr: 15.0,
                active_bearers: 2,
                data_rate_req_mbps: 100.0,
                timestamp_utc: Utc::now().timestamp(),
            }),
            historical_metrics: vec![],
            prediction_horizon_seconds: 30,
        };
        
        let response = service.predict_scell_need(Request::new(request)).await;
        assert!(response.is_ok());
        
        let resp = response.unwrap().into_inner();
        assert_eq!(resp.ue_id, "test_ue");
        assert!(resp.confidence_score >= 0.0 && resp.confidence_score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_get_system_status() {
        let service = create_test_service().await;
        
        let request = GetSystemStatusRequest {};
        let response = service.get_system_status(Request::new(request)).await;
        
        assert!(response.is_ok());
        let resp = response.unwrap().into_inner();
        assert!(resp.healthy);
        assert!(!resp.version.is_empty());
    }
    
    #[tokio::test]
    async fn test_request_validation() {
        let service = create_test_service().await;
        
        // Test empty UE ID
        let request = PredictScellNeedRequest {
            ue_id: String::new(),
            current_metrics: Some(UeMetrics {
                ue_id: "test_ue".to_string(),
                pcell_throughput_mbps: 50.0,
                buffer_status_report_bytes: 1000,
                pcell_cqi: 10.0,
                pcell_rsrp: -80.0,
                pcell_sinr: 15.0,
                active_bearers: 2,
                data_rate_req_mbps: 100.0,
                timestamp_utc: Utc::now().timestamp(),
            }),
            historical_metrics: vec![],
            prediction_horizon_seconds: 30,
        };
        
        assert!(service.validate_prediction_request(&request).is_err());
        
        // Test missing metrics
        let request = PredictScellNeedRequest {
            ue_id: "test_ue".to_string(),
            current_metrics: None,
            historical_metrics: vec![],
            prediction_horizon_seconds: 30,
        };
        
        assert!(service.validate_prediction_request(&request).is_err());
    }
}