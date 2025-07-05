use crate::neural_service::*;
use crate::model_manager::ModelManager;
use crate::error::{ServiceError, ServiceResult};
use crate::conversion::*;
use ruv_fann::{Network, NetworkBuilder, training::{TrainingData, IncrementalBackprop, BatchBackprop, RProp, QuickProp, TrainingAlgorithm, MseError}};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tonic::{Request, Response, Status};
use tracing::{info, warn, error};

pub struct NeuralServiceImpl {
    model_manager: Arc<ModelManager>,
    training_semaphore: Arc<Semaphore>,
    start_time: Instant,
}

impl NeuralServiceImpl {
    pub fn new(model_manager: Arc<ModelManager>, max_concurrent_training: usize) -> Self {
        Self {
            model_manager,
            training_semaphore: Arc::new(Semaphore::new(max_concurrent_training)),
            start_time: Instant::now(),
        }
    }

    async fn train_model_internal(
        &self,
        model_id: String,
        training_data: TrainingData<f64>,
        training_config: TrainingConfig,
    ) -> ServiceResult<TrainingResults> {
        let _permit = self.training_semaphore.acquire().await
            .map_err(|e| ServiceError::Internal(format!("Failed to acquire training permit: {}", e)))?;

        info!("Starting training for model: {}", model_id);
        let start_time = Instant::now();

        // Load the network
        let mut network = self.model_manager.load_network(&model_id).await?;
        let model_info = self.model_manager.get_model_info(&model_id).await?;

        // Configure training parameters
        let max_epochs = model_info.config.max_epochs;
        let desired_error = model_info.config.desired_error;
        let learning_rate = model_info.config.learning_rate;
        let training_algorithm = convert_training_algorithm(model_info.config.training_algorithm)?;

        // Set network parameters
        network.set_learning_rate(learning_rate);
        network.set_activation_function_hidden(convert_activation_function(model_info.config.activation)?);
        network.set_activation_function_output(convert_activation_function(model_info.config.activation)?);

        // Split training data for validation if requested
        let (train_data, validation_data) = if training_config.validation_split > 0.0 {
            let split_index = (training_data.len() as f64 * (1.0 - training_config.validation_split)) as usize;
            let (train_inputs, train_outputs) = training_data.split_at(split_index);
            let (val_inputs, val_outputs) = training_data.split_from(split_index);
            
            let train_td = TrainingData::new(train_inputs.to_vec(), train_outputs.to_vec())
                .map_err(|e| ServiceError::TrainingFailed(format!("Failed to create training data: {}", e)))?;
            let val_td = TrainingData::new(val_inputs.to_vec(), val_outputs.to_vec())
                .map_err(|e| ServiceError::TrainingFailed(format!("Failed to create validation data: {}", e)))?;
            
            (train_td, Some(val_td))
        } else {
            (training_data, None)
        };

        // Train the network
        let mut error_history = Vec::new();
        let mut best_error = f64::MAX;
        let mut patience_counter = 0;
        let mut final_error = desired_error;

        for epoch in 0..max_epochs {
            // Shuffle training data if requested
            let current_train_data = if training_config.shuffle {
                train_data.shuffle()
            } else {
                train_data.clone()
            };

            // Train for one epoch
            let epoch_error = match training_algorithm {
                ruv_fann::TrainingAlgorithm::IncrementalBackprop => {
                    network.train_incremental(&current_train_data)
                        .map_err(|e| ServiceError::TrainingFailed(format!("Training failed at epoch {}: {}", epoch, e)))?
                }
                ruv_fann::TrainingAlgorithm::BatchBackprop => {
                    network.train_batch(&current_train_data)
                        .map_err(|e| ServiceError::TrainingFailed(format!("Training failed at epoch {}: {}", epoch, e)))?
                }
                ruv_fann::TrainingAlgorithm::RProp => {
                    network.train_rprop(&current_train_data)
                        .map_err(|e| ServiceError::TrainingFailed(format!("Training failed at epoch {}: {}", epoch, e)))?
                }
                ruv_fann::TrainingAlgorithm::QuickProp => {
                    network.train_quickprop(&current_train_data)
                        .map_err(|e| ServiceError::TrainingFailed(format!("Training failed at epoch {}: {}", epoch, e)))?
                }
                _ => {
                    return Err(ServiceError::TrainingFailed("Unsupported training algorithm".to_string()));
                }
            };

            error_history.push(epoch_error);
            final_error = epoch_error;

            // Check for early stopping
            if epoch_error < desired_error {
                info!("Training completed: desired error reached at epoch {}", epoch);
                break;
            }

            // Patience-based early stopping
            if training_config.patience > 0 {
                if epoch_error < best_error {
                    best_error = epoch_error;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= training_config.patience {
                        info!("Training stopped early due to patience at epoch {}", epoch);
                        break;
                    }
                }
            }

            // Log progress every 100 epochs
            if epoch % 100 == 0 {
                info!("Epoch {}: Error = {:.6}", epoch, epoch_error);
            }
        }

        let training_time = start_time.elapsed();

        // Calculate validation error if validation data is available
        let validation_error = if let Some(val_data) = validation_data {
            let val_error = network.calculate_mse(&val_data)
                .map_err(|e| ServiceError::TrainingFailed(format!("Failed to calculate validation error: {}", e)))?;
            Some(val_error)
        } else {
            None
        };

        // Update model with trained network
        self.model_manager.update_model_after_training(
            &model_id,
            &network,
            final_error,
            error_history.len() as u32,
        ).await?;

        let results = TrainingResults {
            epochs_completed: error_history.len() as u32,
            final_error,
            error_history,
            training_time_seconds: training_time.as_secs_f64(),
            validation_error: validation_error.unwrap_or(0.0),
        };

        info!("Training completed for model: {} in {:.2}s", model_id, training_time.as_secs_f64());
        Ok(results)
    }
}

#[tonic::async_trait]
impl neural_service_server::NeuralService for NeuralServiceImpl {
    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();
        
        // Validate request
        if req.model_config.is_none() {
            return Err(Status::invalid_argument("Model config is required"));
        }
        
        if req.training_data.is_none() {
            return Err(Status::invalid_argument("Training data is required"));
        }

        let model_config = req.model_config.unwrap();
        let training_data = req.training_data.unwrap();
        let training_config = req.training_config.unwrap_or_default();

        // Validate model configuration
        validate_model_config(&model_config)?;

        // Create model
        let model_id = self.model_manager.create_model(model_config).await?;

        // Convert training data
        let ruv_training_data = convert_training_data(&training_data)?;

        // Train model
        match self.train_model_internal(model_id.clone(), ruv_training_data, training_config).await {
            Ok(results) => {
                let response = TrainResponse {
                    model_id,
                    results: Some(results),
                    status: "success".to_string(),
                    message: "Model training completed successfully".to_string(),
                };
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Training failed: {}", e);
                // Clean up failed model
                if let Err(cleanup_err) = self.model_manager.delete_model(&model_id).await {
                    warn!("Failed to cleanup failed model {}: {}", model_id, cleanup_err);
                }
                
                let response = TrainResponse {
                    model_id,
                    results: None,
                    status: "error".to_string(),
                    message: format!("Training failed: {}", e),
                };
                Ok(Response::new(response))
            }
        }
    }

    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();

        // Validate input
        if req.model_id.is_empty() {
            return Err(Status::invalid_argument("Model ID is required"));
        }

        if req.input_vector.is_empty() {
            return Err(Status::invalid_argument("Input vector is required"));
        }

        // Validate input values
        validate_input_values(&req.input_vector)?;

        // Load model
        let network = self.model_manager.load_network(&req.model_id).await?;
        let model_info = self.model_manager.get_model_info(&req.model_id).await?;

        // Validate input size
        validate_input_size(&req.input_vector, model_info.input_size)?;

        // Make prediction
        let output = network.run(&req.input_vector)
            .map_err(|e| ServiceError::PredictionFailed(format!("Prediction failed: {}", e)))?;

        // Calculate confidence (simple implementation based on output variance)
        let confidence = if output.len() > 1 {
            let mean = output.iter().sum::<f64>() / output.len() as f64;
            let variance = output.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / output.len() as f64;
            1.0 - variance.min(1.0) // Simple confidence measure
        } else {
            1.0 - (output[0] - 0.5).abs() * 2.0 // For single output
        };

        let response = PredictResponse {
            output_vector: output,
            status: "success".to_string(),
            message: "Prediction completed successfully".to_string(),
            confidence,
        };

        Ok(Response::new(response))
    }

    async fn get_model_info(
        &self,
        request: Request<GetModelInfoRequest>,
    ) -> Result<Response<GetModelInfoResponse>, Status> {
        let req = request.into_inner();

        if req.model_id.is_empty() {
            return Err(Status::invalid_argument("Model ID is required"));
        }

        let model_info = self.model_manager.get_model_info(&req.model_id).await?;

        let response = GetModelInfoResponse {
            model_id: model_info.id,
            config: Some(model_info.config),
            metadata: Some(ModelMetadata {
                created_at: model_info.created_at.to_rfc3339(),
                updated_at: model_info.updated_at.to_rfc3339(),
                size_bytes: model_info.metadata.size_bytes,
                total_parameters: model_info.metadata.total_parameters,
                version: model_info.metadata.version,
            }),
            status: "success".to_string(),
            message: "Model information retrieved successfully".to_string(),
        };

        Ok(Response::new(response))
    }

    async fn list_models(
        &self,
        request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let req = request.into_inner();

        let page = req.page;
        let page_size = req.page_size.max(1).min(100); // Limit page size
        let filter = if req.filter.is_empty() { None } else { Some(req.filter) };

        let models = self.model_manager.list_models(page, page_size, filter).await?;
        let total_count = self.model_manager.get_model_count().await;

        let model_summaries: Vec<ModelSummary> = models
            .into_iter()
            .map(|model| ModelSummary {
                model_id: model.id,
                name: model.name,
                description: model.description,
                created_at: model.created_at.to_rfc3339(),
                size_bytes: model.metadata.size_bytes,
                status: if model.metadata.training_completed { "trained" } else { "created" }.to_string(),
            })
            .collect();

        let response = ListModelsResponse {
            models: model_summaries,
            total_count: total_count as u32,
            status: "success".to_string(),
            message: "Models retrieved successfully".to_string(),
        };

        Ok(Response::new(response))
    }

    async fn delete_model(
        &self,
        request: Request<DeleteModelRequest>,
    ) -> Result<Response<DeleteModelResponse>, Status> {
        let req = request.into_inner();

        if req.model_id.is_empty() {
            return Err(Status::invalid_argument("Model ID is required"));
        }

        self.model_manager.delete_model(&req.model_id).await?;

        let response = DeleteModelResponse {
            status: "success".to_string(),
            message: "Model deleted successfully".to_string(),
        };

        Ok(Response::new(response))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let active_models = self.model_manager.get_model_count().await;
        let uptime = self.start_time.elapsed();

        let response = HealthResponse {
            status: "healthy".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            active_models: active_models as u32,
            uptime_seconds: uptime.as_secs_f64(),
        };

        Ok(Response::new(response))
    }
}

// Default implementation removed to avoid conflict with protobuf