//! gRPC server implementation for the Model Registry service

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use tonic::{transport::Server, Request, Response, Status, Streaming};
use tokio_stream::{Stream, StreamExt};
use tracing::{error, info, instrument, warn};
use chrono::{DateTime, Utc};

use crate::generated::{
    model_registry_server::{ModelRegistry, ModelRegistryServer},
    *,
};
use crate::services::registry_service::{ModelRegistryService, ModelRegistrationInfo, CreateVersionInfo, ModelListFilters};
use crate::config::RegistryConfig;
use crate::error::{RegistryError, RegistryResult};

/// gRPC server implementation
pub struct ModelRegistryServer {
    service: Arc<ModelRegistryService>,
    config: Arc<RegistryConfig>,
}

impl ModelRegistryServer {
    /// Create a new gRPC server
    pub fn new(service: ModelRegistryService, config: RegistryConfig) -> Self {
        Self {
            service: Arc::new(service),
            config: Arc::new(config),
        }
    }

    /// Start the gRPC server
    pub async fn serve(self, addr: std::net::SocketAddr) -> anyhow::Result<()> {
        let service = ModelRegistryServer::new(self.service, self.config);
        
        info!(%addr, "Starting Model Registry gRPC server");
        
        Server::builder()
            .add_service(ModelRegistryServer::new(service))
            .serve(addr)
            .await?;
            
        Ok(())
    }

    /// Convert protobuf ModelInfo to service ModelRegistrationInfo
    fn convert_to_registration_info(&self, model_info: ModelInfo) -> RegistryResult<ModelRegistrationInfo> {
        Ok(ModelRegistrationInfo {
            name: model_info.name,
            description: if model_info.description.is_empty() { None } else { Some(model_info.description) },
            category: self.convert_category_to_string(model_info.category()),
            model_type: self.convert_model_type_to_string(model_info.model_type()),
            tags: if model_info.tags.is_empty() { None } else { Some(model_info.tags) },
            capabilities: model_info.capabilities.map(|c| serde_json::to_value(c).unwrap_or_default()),
            configuration: model_info.configuration.map(|c| serde_json::to_value(c).unwrap_or_default()),
            metadata: model_info.metadata.map(|m| serde_json::to_value(m).unwrap_or_default()),
            initial_version: Some("1.0.0".to_string()),
            artifact_format: Some("ruvrann_binary".to_string()),
            performance_metrics: None,
            training_data_info: None,
        })
    }

    /// Convert ModelCategory enum to string
    fn convert_category_to_string(&self, category: ModelCategory) -> String {
        match category {
            ModelCategory::PredictiveOptimization => "predictive_optimization".to_string(),
            ModelCategory::ServiceAssurance => "service_assurance".to_string(),
            ModelCategory::NetworkIntelligence => "network_intelligence".to_string(),
            ModelCategory::AnomalyDetection => "anomaly_detection".to_string(),
            ModelCategory::Forecasting => "forecasting".to_string(),
            ModelCategory::Classification => "classification".to_string(),
            ModelCategory::Regression => "regression".to_string(),
            ModelCategory::Clustering => "clustering".to_string(),
            _ => "unspecified".to_string(),
        }
    }

    /// Convert ModelType enum to string
    fn convert_model_type_to_string(&self, model_type: ModelType) -> String {
        match model_type {
            ModelType::NeuralNetwork => "neural_network".to_string(),
            ModelType::RandomForest => "random_forest".to_string(),
            ModelType::Svm => "svm".to_string(),
            ModelType::LinearRegression => "linear_regression".to_string(),
            ModelType::LogisticRegression => "logistic_regression".to_string(),
            ModelType::Ensemble => "ensemble".to_string(),
            ModelType::DeepLearning => "deep_learning".to_string(),
            ModelType::TimeSeries => "time_series".to_string(),
            _ => "unspecified".to_string(),
        }
    }

    /// Convert service ModelInfo to protobuf ModelInfo
    fn convert_from_service_model_info(&self, model_info: crate::services::registry_service::ModelInfo) -> ModelInfo {
        ModelInfo {
            model_id: model_info.model_id,
            name: model_info.name,
            description: model_info.description.unwrap_or_default(),
            version: model_info.version_info.as_ref().map(|v| v.version_number.clone()).unwrap_or_default(),
            category: self.string_to_model_category(&model_info.category),
            model_type: self.string_to_model_type(&model_info.model_type),
            tags: model_info.tags.unwrap_or_default(),
            capabilities: model_info.capabilities.map(|c| ModelCapabilities::default()), // TODO: Proper conversion
            configuration: model_info.configuration.map(|c| ModelConfiguration::default()), // TODO: Proper conversion
            metadata: model_info.metadata.map(|m| ModelMetadata::default()), // TODO: Proper conversion
            created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::from(model_info.created_at))),
            updated_at: Some(prost_types::Timestamp::from(std::time::SystemTime::from(model_info.updated_at))),
            created_by: model_info.created_by,
            status: self.string_to_model_status(&model_info.status),
        }
    }

    /// Convert string to ModelCategory
    fn string_to_model_category(&self, category: &str) -> i32 {
        match category {
            "predictive_optimization" => ModelCategory::PredictiveOptimization as i32,
            "service_assurance" => ModelCategory::ServiceAssurance as i32,
            "network_intelligence" => ModelCategory::NetworkIntelligence as i32,
            "anomaly_detection" => ModelCategory::AnomalyDetection as i32,
            "forecasting" => ModelCategory::Forecasting as i32,
            "classification" => ModelCategory::Classification as i32,
            "regression" => ModelCategory::Regression as i32,
            "clustering" => ModelCategory::Clustering as i32,
            _ => ModelCategory::Unspecified as i32,
        }
    }

    /// Convert string to ModelType
    fn string_to_model_type(&self, model_type: &str) -> i32 {
        match model_type {
            "neural_network" => ModelType::NeuralNetwork as i32,
            "random_forest" => ModelType::RandomForest as i32,
            "svm" => ModelType::Svm as i32,
            "linear_regression" => ModelType::LinearRegression as i32,
            "logistic_regression" => ModelType::LogisticRegression as i32,
            "ensemble" => ModelType::Ensemble as i32,
            "deep_learning" => ModelType::DeepLearning as i32,
            "time_series" => ModelType::TimeSeries as i32,
            _ => ModelType::Unspecified as i32,
        }
    }

    /// Convert string to ModelStatus
    fn string_to_model_status(&self, status: &str) -> i32 {
        match status {
            "registered" => ModelStatus::Registered as i32,
            "training" => ModelStatus::Training as i32,
            "trained" => ModelStatus::Trained as i32,
            "deployed" => ModelStatus::Deployed as i32,
            "retired" => ModelStatus::Retired as i32,
            "failed" => ModelStatus::Failed as i32,
            _ => ModelStatus::Unspecified as i32,
        }
    }

    /// Extract user identity from request metadata
    fn extract_user_identity(&self, request: &Request<impl std::fmt::Debug>) -> String {
        // TODO: Implement proper authentication and extract user from JWT/API key
        request
            .metadata()
            .get("user-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("system")
            .to_string()
    }
}

#[tonic::async_trait]
impl ModelRegistry for ModelRegistryServer {
    #[instrument(skip(self, request))]
    async fn register_model(
        &self,
        request: Request<RegisterModelRequest>,
    ) -> Result<Response<RegisterModelResponse>, Status> {
        let req = request.into_inner();
        let user_id = self.extract_user_identity(&Request::new(&req));

        let model_info = req.model_info
            .ok_or_else(|| Status::invalid_argument("model_info is required"))?;
        
        let registration_info = self.convert_to_registration_info(model_info)
            .map_err(|e| Status::invalid_argument(format!("Invalid model info: {}", e)))?;

        match self.service.register_model(registration_info, req.model_artifact, user_id).await {
            Ok(result) => {
                info!(
                    model_id = %result.model_id,
                    version_id = %result.version_id,
                    "Model registered successfully"
                );
                
                Ok(Response::new(RegisterModelResponse {
                    model_id: result.model_id,
                    version_id: result.version_id,
                    success: true,
                    message: "Model registered successfully".to_string(),
                }))
            }
            Err(e) => {
                error!(error = %e, "Failed to register model");
                Err(e.into())
            }
        }
    }

    #[instrument(skip(self, request))]
    async fn unregister_model(
        &self,
        request: Request<UnregisterModelRequest>,
    ) -> Result<Response<UnregisterModelResponse>, Status> {
        // TODO: Implement model unregistration
        Err(Status::unimplemented("Model unregistration not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn update_model(
        &self,
        request: Request<UpdateModelRequest>,
    ) -> Result<Response<UpdateModelResponse>, Status> {
        // TODO: Implement model updates
        Err(Status::unimplemented("Model updates not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn get_model(
        &self,
        request: Request<GetModelRequest>,
    ) -> Result<Response<GetModelResponse>, Status> {
        let req = request.into_inner();

        match self.service.get_model(&req.model_id, req.version_id.as_deref(), req.include_artifact).await {
            Ok(Some(model_info)) => {
                let proto_model_info = self.convert_from_service_model_info(model_info.clone());
                let version_info = model_info.version_info.map(|v| ModelVersion {
                    version_id: v.version_id,
                    model_id: req.model_id.clone(),
                    version_number: v.version_number,
                    description: v.description.unwrap_or_default(),
                    artifact: Some(ModelArtifact {
                        artifact_id: v.version_id.clone(),
                        storage_path: v.artifact_path,
                        checksum: v.artifact_checksum,
                        size_bytes: v.artifact_size_bytes,
                        format: ArtifactFormat::RuvrannBinary as i32,
                        properties: HashMap::new(),
                    }),
                    performance: None, // TODO: Convert performance metrics
                    created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::from(v.created_at))),
                    created_by: v.created_by,
                    is_active: v.is_active,
                    status: VersionStatus::Active as i32, // TODO: Convert status
                });

                Ok(Response::new(GetModelResponse {
                    model_info: Some(proto_model_info),
                    version_info,
                    model_artifact: model_info.artifact_data.unwrap_or_default(),
                }))
            }
            Ok(None) => {
                Err(Status::not_found(format!("Model not found: {}", req.model_id)))
            }
            Err(e) => {
                error!(error = %e, model_id = %req.model_id, "Failed to get model");
                Err(e.into())
            }
        }
    }

    #[instrument(skip(self, request))]
    async fn list_models(
        &self,
        request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let req = request.into_inner();
        
        let filters = ModelListFilters {
            category: if req.category() == ModelCategory::Unspecified {
                None
            } else {
                Some(self.convert_category_to_string(req.category()))
            },
            status: if req.status() == ModelStatus::Unspecified {
                None
            } else {
                Some(match req.status() {
                    ModelStatus::Registered => "registered".to_string(),
                    ModelStatus::Training => "training".to_string(),
                    ModelStatus::Trained => "trained".to_string(),
                    ModelStatus::Deployed => "deployed".to_string(),
                    ModelStatus::Retired => "retired".to_string(),
                    ModelStatus::Failed => "failed".to_string(),
                    _ => "unspecified".to_string(),
                })
            },
            model_type: None,
        };

        let page_size = if req.page_size == 0 { 50 } else { req.page_size as u64 };
        let page = req.page_token.parse::<u64>().unwrap_or(0);

        match self.service.list_models(filters, page_size, page).await {
            Ok(result) => {
                let models = result.models.into_iter().map(|summary| ModelInfo {
                    model_id: summary.model_id,
                    name: summary.name,
                    description: summary.description.unwrap_or_default(),
                    version: "".to_string(), // TODO: Get active version
                    category: self.string_to_model_category(&summary.category),
                    model_type: self.string_to_model_type(&summary.model_type),
                    tags: vec![],
                    capabilities: None,
                    configuration: None,
                    metadata: None,
                    created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::from(summary.created_at))),
                    updated_at: Some(prost_types::Timestamp::from(std::time::SystemTime::from(summary.updated_at))),
                    created_by: summary.created_by,
                    status: self.string_to_model_status(&summary.status),
                }).collect();

                let next_page_token = if result.has_next_page {
                    (page + 1).to_string()
                } else {
                    String::new()
                };

                Ok(Response::new(ListModelsResponse {
                    models,
                    next_page_token,
                    total_count: result.total_count as i32,
                }))
            }
            Err(e) => {
                error!(error = %e, "Failed to list models");
                Err(e.into())
            }
        }
    }

    #[instrument(skip(self, request))]
    async fn search_models(
        &self,
        request: Request<SearchModelsRequest>,
    ) -> Result<Response<SearchModelsResponse>, Status> {
        let req = request.into_inner();
        
        let filters = ModelListFilters {
            category: if req.category() == ModelCategory::Unspecified {
                None
            } else {
                Some(self.convert_category_to_string(req.category()))
            },
            status: None,
            model_type: if req.model_type() == ModelType::Unspecified {
                None
            } else {
                Some(self.convert_model_type_to_string(req.model_type()))
            },
        };

        let page_size = if req.page_size == 0 { 50 } else { req.page_size as u64 };
        let page = req.page_token.parse::<u64>().unwrap_or(0);
        let tags = if req.tags.is_empty() { None } else { Some(req.tags) };

        match self.service.search_models(&req.query, tags, filters, page_size, page).await {
            Ok(result) => {
                let models = result.models.into_iter().map(|summary| ModelInfo {
                    model_id: summary.model_id,
                    name: summary.name,
                    description: summary.description.unwrap_or_default(),
                    version: "".to_string(),
                    category: self.string_to_model_category(&summary.category),
                    model_type: self.string_to_model_type(&summary.model_type),
                    tags: vec![],
                    capabilities: None,
                    configuration: None,
                    metadata: None,
                    created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::from(summary.created_at))),
                    updated_at: Some(prost_types::Timestamp::from(std::time::SystemTime::from(summary.updated_at))),
                    created_by: summary.created_by,
                    status: self.string_to_model_status(&summary.status),
                }).collect();

                let next_page_token = if result.has_next_page {
                    (page + 1).to_string()
                } else {
                    String::new()
                };

                Ok(Response::new(SearchModelsResponse {
                    models,
                    next_page_token,
                    total_count: result.total_count as i32,
                    suggestions: result.suggestions,
                }))
            }
            Err(e) => {
                error!(error = %e, query = %req.query, "Failed to search models");
                Err(e.into())
            }
        }
    }

    #[instrument(skip(self, request))]
    async fn create_model_version(
        &self,
        request: Request<CreateModelVersionRequest>,
    ) -> Result<Response<CreateModelVersionResponse>, Status> {
        let req = request.into_inner();
        let user_id = self.extract_user_identity(&Request::new(&req));

        let version_info = CreateVersionInfo {
            version_number: req.version_number,
            description: if req.description.is_empty() { None } else { Some(req.description) },
            artifact_format: Some("ruvrann_binary".to_string()),
            performance_metrics: req.performance.map(|p| serde_json::to_value(p).unwrap_or_default()),
            training_data_info: None,
        };

        match self.service.create_model_version(&req.model_id, version_info, req.model_artifact, user_id, req.activate).await {
            Ok(version_id) => {
                info!(
                    model_id = %req.model_id,
                    version_id = %version_id,
                    "Model version created successfully"
                );
                
                Ok(Response::new(CreateModelVersionResponse {
                    version_id,
                    success: true,
                    message: "Model version created successfully".to_string(),
                }))
            }
            Err(e) => {
                error!(error = %e, model_id = %req.model_id, "Failed to create model version");
                Err(e.into())
            }
        }
    }

    #[instrument(skip(self, request))]
    async fn get_model_versions(
        &self,
        request: Request<GetModelVersionsRequest>,
    ) -> Result<Response<GetModelVersionsResponse>, Status> {
        // TODO: Implement get model versions
        Err(Status::unimplemented("Get model versions not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn activate_model_version(
        &self,
        request: Request<ActivateModelVersionRequest>,
    ) -> Result<Response<ActivateModelVersionResponse>, Status> {
        // TODO: Implement activate model version
        Err(Status::unimplemented("Activate model version not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn deactivate_model_version(
        &self,
        request: Request<DeactivateModelVersionRequest>,
    ) -> Result<Response<DeactivateModelVersionResponse>, Status> {
        // TODO: Implement deactivate model version
        Err(Status::unimplemented("Deactivate model version not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn deploy_model(
        &self,
        request: Request<DeployModelRequest>,
    ) -> Result<Response<DeployModelResponse>, Status> {
        // TODO: Implement model deployment
        Err(Status::unimplemented("Model deployment not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn retire_model(
        &self,
        request: Request<RetireModelRequest>,
    ) -> Result<Response<RetireModelResponse>, Status> {
        // TODO: Implement model retirement
        Err(Status::unimplemented("Model retirement not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn get_deployment_status(
        &self,
        request: Request<GetDeploymentStatusRequest>,
    ) -> Result<Response<GetDeploymentStatusResponse>, Status> {
        // TODO: Implement get deployment status
        Err(Status::unimplemented("Get deployment status not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn record_model_metrics(
        &self,
        request: Request<RecordModelMetricsRequest>,
    ) -> Result<Response<RecordModelMetricsResponse>, Status> {
        // TODO: Implement record model metrics
        Err(Status::unimplemented("Record model metrics not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn get_model_metrics(
        &self,
        request: Request<GetModelMetricsRequest>,
    ) -> Result<Response<GetModelMetricsResponse>, Status> {
        // TODO: Implement get model metrics
        Err(Status::unimplemented("Get model metrics not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn get_model_health(
        &self,
        request: Request<GetModelHealthRequest>,
    ) -> Result<Response<GetModelHealthResponse>, Status> {
        // TODO: Implement get model health
        Err(Status::unimplemented("Get model health not yet implemented"))
    }

    #[instrument(skip(self, request))]
    async fn get_registry_stats(
        &self,
        request: Request<google::protobuf::Empty>,
    ) -> Result<Response<GetRegistryStatsResponse>, Status> {
        match self.service.get_registry_stats().await {
            Ok(stats) => {
                Ok(Response::new(GetRegistryStatsResponse {
                    total_models: stats.total_models as i32,
                    total_versions: stats.total_versions as i32,
                    active_deployments: stats.active_deployments as i32,
                    models_by_category: stats.models_by_category.into_iter().map(|(k, v)| (k, v as i32)).collect(),
                    models_by_status: stats.models_by_status.into_iter().map(|(k, v)| (k, v as i32)).collect(),
                    total_predictions_served: stats.total_predictions_served as i64,
                    average_model_latency_ms: stats.average_model_latency_ms,
                    last_updated: Some(prost_types::Timestamp::from(std::time::SystemTime::from(stats.last_updated))),
                }))
            }
            Err(e) => {
                error!(error = %e, "Failed to get registry stats");
                Err(e.into())
            }
        }
    }

    #[instrument(skip(self, request))]
    async fn get_model_usage_stats(
        &self,
        request: Request<GetModelUsageStatsRequest>,
    ) -> Result<Response<GetModelUsageStatsResponse>, Status> {
        // TODO: Implement get model usage stats
        Err(Status::unimplemented("Get model usage stats not yet implemented"))
    }
}