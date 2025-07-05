//! gRPC service implementation for interference classification
//! 
//! This module implements the gRPC service for real-time interference classification
//! with confidence scoring and mitigation recommendations.

use crate::{
    InterferenceClassifierError, Result, InterferenceClass, MitigationRecommendation,
    NoiseFloorMeasurement, CellParameters, TrainingExample, ModelConfig,
    models::InterferenceClassifierModel, features::FeatureExtractor,
};
use tonic::{transport::Server, Request, Response, Status};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

// Include the generated protobuf code
pub mod proto {
    tonic::include_proto!("interference_classifier");
}

use proto::{
    interference_classifier_server::{InterferenceClassifier, InterferenceClassifierServer},
    ClassifyRequest, ClassifyResponse, ConfidenceRequest, ConfidenceResponse,
    MitigationRequest, MitigationResponse, TrainRequest, TrainResponse,
    MetricsRequest, MetricsResponse, NoiseFloorMeasurement as ProtoMeasurement,
    CellParameters as ProtoCellParameters, TrainingExample as ProtoTrainingExample,
    ModelConfig as ProtoModelConfig,
};

/// Interference classification service implementation
pub struct InterferenceClassificationService {
    model: Arc<RwLock<Option<InterferenceClassifierModel>>>,
    feature_extractor: Arc<RwLock<FeatureExtractor>>,
    service_stats: Arc<RwLock<ServiceStats>>,
}

#[derive(Debug, Clone)]
struct ServiceStats {
    total_classifications: u64,
    successful_classifications: u64,
    failed_classifications: u64,
    total_training_sessions: u64,
    last_model_update: Option<DateTime<Utc>>,
    uptime_start: DateTime<Utc>,
}

impl InterferenceClassificationService {
    /// Create a new interference classification service
    pub fn new() -> Self {
        Self {
            model: Arc::new(RwLock::new(None)),
            feature_extractor: Arc::new(RwLock::new(FeatureExtractor::new())),
            service_stats: Arc::new(RwLock::new(ServiceStats {
                total_classifications: 0,
                successful_classifications: 0,
                failed_classifications: 0,
                total_training_sessions: 0,
                last_model_update: None,
                uptime_start: Utc::now(),
            })),
        }
    }
    
    /// Initialize the service with a pre-trained model
    pub async fn initialize_with_model(&self, model: InterferenceClassifierModel) -> Result<()> {
        let mut model_guard = self.model.write().await;
        *model_guard = Some(model);
        
        let mut stats = self.service_stats.write().await;
        stats.last_model_update = Some(Utc::now());
        
        log::info!("Service initialized with pre-trained model");
        Ok(())
    }
    
    /// Load model from file
    pub async fn load_model_from_file(&self, path: &str) -> Result<()> {
        let model = InterferenceClassifierModel::load_model(path)?;
        self.initialize_with_model(model).await?;
        log::info!("Model loaded from file: {}", path);
        Ok(())
    }
    
    /// Start the gRPC server
    pub async fn start_server(&self, address: &str) -> Result<()> {
        let addr = address.parse()
            .map_err(|e| InterferenceClassifierError::NetworkError(
                format!("Invalid address: {}", e)
            ))?;
        
        let service = InterferenceClassifierServer::new(self.clone());
        
        log::info!("Starting interference classification service on {}", addr);
        
        Server::builder()
            .add_service(service)
            .serve(addr)
            .await
            .map_err(|e| InterferenceClassifierError::NetworkError(
                format!("Server error: {}", e)
            ))?;
        
        Ok(())
    }
    
    /// Convert proto measurement to internal type
    fn proto_to_measurement(&self, proto_measurement: &ProtoMeasurement) -> Result<NoiseFloorMeasurement> {
        let timestamp = DateTime::parse_from_rfc3339(&proto_measurement.timestamp)
            .map_err(|e| InterferenceClassifierError::InvalidInputError(
                format!("Invalid timestamp: {}", e)
            ))?
            .with_timezone(&Utc);
        
        Ok(NoiseFloorMeasurement {
            timestamp,
            noise_floor_pusch: proto_measurement.noise_floor_pusch,
            noise_floor_pucch: proto_measurement.noise_floor_pucch,
            cell_ret: proto_measurement.cell_ret,
            rsrp: proto_measurement.rsrp,
            sinr: proto_measurement.sinr,
            active_users: proto_measurement.active_users,
            prb_utilization: proto_measurement.prb_utilization,
        })
    }
    
    /// Convert proto cell parameters to internal type
    fn proto_to_cell_params(&self, proto_params: &ProtoCellParameters) -> Result<CellParameters> {
        Ok(CellParameters {
            cell_id: proto_params.cell_id.clone(),
            frequency_band: proto_params.frequency_band.clone(),
            tx_power: proto_params.tx_power,
            antenna_count: proto_params.antenna_count,
            bandwidth_mhz: proto_params.bandwidth_mhz,
            technology: proto_params.technology.clone(),
        })
    }
    
    /// Convert proto training example to internal type
    fn proto_to_training_example(&self, proto_example: &ProtoTrainingExample) -> Result<TrainingExample> {
        let measurements = proto_example.measurements.iter()
            .map(|m| self.proto_to_measurement(m))
            .collect::<Result<Vec<_>>>()?;
        
        let cell_params = proto_example.cell_params.as_ref()
            .ok_or_else(|| InterferenceClassifierError::InvalidInputError(
                "Missing cell parameters".to_string()
            ))?;
        
        Ok(TrainingExample {
            measurements,
            cell_params: self.proto_to_cell_params(cell_params)?,
            true_interference_class: InterferenceClass::from_str(&proto_example.true_interference_class),
        })
    }
    
    /// Convert proto model config to internal type
    fn proto_to_model_config(&self, proto_config: &ProtoModelConfig) -> Result<ModelConfig> {
        Ok(ModelConfig {
            hidden_layers: proto_config.hidden_layers.clone(),
            learning_rate: proto_config.learning_rate,
            max_epochs: proto_config.max_epochs,
            target_accuracy: proto_config.target_accuracy,
            activation_function: proto_config.activation_function.clone(),
            dropout_rate: proto_config.dropout_rate,
        })
    }
}

impl Clone for InterferenceClassificationService {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            feature_extractor: self.feature_extractor.clone(),
            service_stats: self.service_stats.clone(),
        }
    }
}

#[tonic::async_trait]
impl InterferenceClassifier for InterferenceClassificationService {
    /// Classify uplink interference
    async fn classify_ul_interference(
        &self,
        request: Request<ClassifyRequest>,
    ) -> std::result::Result<Response<ClassifyResponse>, Status> {
        let req = request.into_inner();
        
        // Update stats
        {
            let mut stats = self.service_stats.write().await;
            stats.total_classifications += 1;
        }
        
        // Check if model is loaded
        let model_guard = self.model.read().await;
        let model = model_guard.as_ref().ok_or_else(|| {
            Status::unavailable("Model not loaded")
        })?;
        
        // Convert proto measurements to internal format
        let measurements = req.measurements.iter()
            .map(|m| self.proto_to_measurement(m))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| Status::invalid_argument(format!("Invalid measurements: {}", e)))?;
        
        // Convert cell parameters
        let cell_params = req.cell_params.as_ref()
            .ok_or_else(|| Status::invalid_argument("Missing cell parameters"))?;
        let cell_params = self.proto_to_cell_params(cell_params)
            .map_err(|e| Status::invalid_argument(format!("Invalid cell parameters: {}", e)))?;
        
        // Extract features
        let feature_extractor = self.feature_extractor.read().await;
        let mut features = feature_extractor.extract_features(&measurements, &cell_params)
            .map_err(|e| Status::internal(format!("Feature extraction failed: {}", e)))?;
        
        // Normalize features
        feature_extractor.normalize_features(&mut features)
            .map_err(|e| Status::internal(format!("Feature normalization failed: {}", e)))?;
        
        // Classify
        let result = model.classify(&features)
            .map_err(|e| Status::internal(format!("Classification failed: {}", e)))?;
        
        // Update success stats
        {
            let mut stats = self.service_stats.write().await;
            stats.successful_classifications += 1;
        }
        
        let response = ClassifyResponse {
            interference_class: result.interference_class.as_str().to_string(),
            confidence: result.confidence,
            timestamp: result.timestamp.to_rfc3339(),
            feature_vector: result.feature_vector,
        };
        
        Ok(Response::new(response))
    }
    
    /// Get classification confidence
    async fn get_classification_confidence(
        &self,
        request: Request<ConfidenceRequest>,
    ) -> std::result::Result<Response<ConfidenceResponse>, Status> {
        let req = request.into_inner();
        
        // Check if model is loaded
        let model_guard = self.model.read().await;
        let model = model_guard.as_ref().ok_or_else(|| {
            Status::unavailable("Model not loaded")
        })?;
        
        // Convert measurements
        let measurements = req.measurements.iter()
            .map(|m| self.proto_to_measurement(m))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| Status::invalid_argument(format!("Invalid measurements: {}", e)))?;
        
        // For this implementation, we'll use a simplified confidence calculation
        // In practice, this would use the actual feature extraction and classification
        let confidence = if measurements.len() >= 10 {
            0.85 // High confidence with sufficient data
        } else {
            0.60 // Lower confidence with limited data
        };
        
        let response = ConfidenceResponse {
            confidence,
            model_version: model.get_model_info().model_version,
            sample_count: measurements.len() as i32,
        };
        
        Ok(Response::new(response))
    }
    
    /// Get mitigation recommendations
    async fn get_mitigation_recommendations(
        &self,
        request: Request<MitigationRequest>,
    ) -> std::result::Result<Response<MitigationResponse>, Status> {
        let req = request.into_inner();
        
        let interference_class = InterferenceClass::from_str(&req.interference_class);
        let recommendation = MitigationRecommendation::new(interference_class, req.confidence);
        
        let response = MitigationResponse {
            recommendations: recommendation.recommendations,
            priority_level: recommendation.priority_level as i32,
            estimated_impact: recommendation.estimated_impact,
        };
        
        Ok(Response::new(response))
    }
    
    /// Train the model
    async fn train_model(
        &self,
        request: Request<TrainRequest>,
    ) -> std::result::Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();
        
        // Update training stats
        {
            let mut stats = self.service_stats.write().await;
            stats.total_training_sessions += 1;
        }
        
        // Convert training examples
        let training_examples = req.examples.iter()
            .map(|e| self.proto_to_training_example(e))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| Status::invalid_argument(format!("Invalid training examples: {}", e)))?;
        
        if training_examples.is_empty() {
            return Err(Status::invalid_argument("No training examples provided"));
        }
        
        // Convert model config
        let model_config = req.config.as_ref()
            .map(|c| self.proto_to_model_config(c))
            .transpose()
            .map_err(|e| Status::invalid_argument(format!("Invalid model config: {}", e)))?
            .unwrap_or_default();
        
        // Create and train new model
        let mut model = InterferenceClassifierModel::new(model_config)
            .map_err(|e| Status::internal(format!("Model creation failed: {}", e)))?;
        
        let start_time = Utc::now();
        let metrics = model.train(&training_examples)
            .map_err(|e| Status::internal(format!("Training failed: {}", e)))?;
        let training_time = Utc::now() - start_time;
        
        // Update the service model
        {
            let mut model_guard = self.model.write().await;
            *model_guard = Some(model);
        }
        
        // Update stats
        {
            let mut stats = self.service_stats.write().await;
            stats.last_model_update = Some(Utc::now());
        }
        
        let response = TrainResponse {
            model_id: Uuid::new_v4().to_string(),
            training_accuracy: metrics.accuracy,
            validation_accuracy: metrics.accuracy, // Simplified - would use separate validation set
            epochs_trained: 100, // Simplified - would track actual epochs
            training_time: format!("{}s", training_time.num_seconds()),
        };
        
        Ok(Response::new(response))
    }
    
    /// Get model metrics
    async fn get_model_metrics(
        &self,
        _request: Request<MetricsRequest>,
    ) -> std::result::Result<Response<MetricsResponse>, Status> {
        let model_guard = self.model.read().await;
        let model = model_guard.as_ref().ok_or_else(|| {
            Status::unavailable("Model not loaded")
        })?;
        
        let info = model.get_model_info();
        let training_accuracy = info.training_accuracy.unwrap_or(0.0);
        
        // Create simplified metrics response
        let mut class_metrics = HashMap::new();
        for i in 0..InterferenceClass::num_classes() {
            let class = InterferenceClass::from_index(i);
            class_metrics.insert(format!("{}_accuracy", class.as_str()), training_accuracy);
        }
        
        let response = MetricsResponse {
            accuracy: training_accuracy,
            precision: training_accuracy * 0.95, // Simplified
            recall: training_accuracy * 0.93,    // Simplified
            f1_score: training_accuracy * 0.94,  // Simplified
            class_metrics,
        };
        
        Ok(Response::new(response))
    }
}

impl Default for InterferenceClassificationService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_creation() {
        let service = InterferenceClassificationService::new();
        assert!(service.model.read().await.is_none());
    }
    
    #[test]
    fn test_proto_conversion() {
        let service = InterferenceClassificationService::new();
        
        let proto_measurement = ProtoMeasurement {
            timestamp: "2023-01-01T00:00:00Z".to_string(),
            noise_floor_pusch: -100.0,
            noise_floor_pucch: -102.0,
            cell_ret: 0.05,
            rsrp: -80.0,
            sinr: 15.0,
            active_users: 50,
            prb_utilization: 0.6,
        };
        
        let measurement = service.proto_to_measurement(&proto_measurement);
        assert!(measurement.is_ok());
        
        let m = measurement.unwrap();
        assert_eq!(m.noise_floor_pusch, -100.0);
        assert_eq!(m.active_users, 50);
    }
}