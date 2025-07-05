use crate::config::*;
use crate::engine::*;
use crate::error::*;
use crate::validation::*;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument};

// Include the generated protobuf code
pub mod feature_engineering {
    tonic::include_proto!("feature_engineering");
}

use feature_engineering::*;

/// gRPC service implementation for feature engineering
#[derive(Debug)]
pub struct FeatureEngineeringServiceImpl {
    engine: Arc<FeatureEngine>,
    validator: Arc<FeatureValidator>,
    start_time: Instant,
}

impl FeatureEngineeringServiceImpl {
    /// Create a new gRPC service implementation
    pub fn new(config: FeatureEngineConfig) -> FeatureEngineResult<Self> {
        let engine = Arc::new(FeatureEngine::new(config.clone())?);
        let validator = Arc::new(FeatureValidator::new(config));
        
        Ok(Self {
            engine,
            validator,
            start_time: Instant::now(),
        })
    }
}

#[tonic::async_trait]
impl feature_engineering_service_server::FeatureEngineeringService for FeatureEngineeringServiceImpl {
    /// Generate features for a single time-series
    #[instrument(skip(self, request))]
    async fn generate_features(
        &self,
        request: Request<GenerateFeaturesRequest>,
    ) -> Result<Response<GenerateFeaturesResponse>, Status> {
        let req = request.into_inner();
        
        info!(
            "Generating features for time-series '{}', input: {}, output: {}",
            req.time_series_id, req.input_path, req.output_path
        );

        // Convert protobuf config to internal config
        let feature_config = req.config
            .ok_or_else(|| Status::invalid_argument("Feature config is required"))?;
        let internal_config = self.convert_feature_config(feature_config)?;

        // Generate features
        let result = self.engine.generate_features(
            &req.time_series_id,
            Path::new(&req.input_path),
            Path::new(&req.output_path),
            &internal_config,
        ).await
        .map_err(|e| Status::from(e))?;

        // Convert result to protobuf response
        let response = GenerateFeaturesResponse {
            time_series_id: result.time_series_id,
            output_path: result.output_path.to_string_lossy().to_string(),
            stats: Some(self.convert_feature_generation_stats(result.stats)),
            generated_features: result.generated_features,
        };

        Ok(Response::new(response))
    }

    /// Generate features for multiple time-series in batch
    #[instrument(skip(self, request))]
    async fn generate_batch_features(
        &self,
        request: Request<GenerateBatchFeaturesRequest>,
    ) -> Result<Response<GenerateBatchFeaturesResponse>, Status> {
        let req = request.into_inner();
        
        info!(
            "Generating batch features for {} time-series",
            req.time_series_ids.len()
        );

        // Convert protobuf config to internal config
        let feature_config = req.config
            .ok_or_else(|| Status::invalid_argument("Feature config is required"))?;
        let internal_config = self.convert_feature_config(feature_config)?;

        // Generate batch features
        let result = self.engine.generate_batch_features(
            &req.time_series_ids,
            Path::new(&req.input_directory),
            Path::new(&req.output_directory),
            &internal_config,
            req.max_parallel_jobs as usize,
        ).await
        .map_err(|e| Status::from(e))?;

        // Convert results to protobuf response
        let mut responses = Vec::new();
        for result in result.results {
            responses.push(GenerateFeaturesResponse {
                time_series_id: result.time_series_id,
                output_path: result.output_path.to_string_lossy().to_string(),
                stats: Some(self.convert_feature_generation_stats(result.stats)),
                generated_features: result.generated_features,
            });
        }

        let response = GenerateBatchFeaturesResponse {
            results: responses,
            batch_stats: Some(self.convert_batch_processing_stats(result.batch_stats)),
        };

        Ok(Response::new(response))
    }

    /// Validate feature generation output
    #[instrument(skip(self, request))]
    async fn validate_features(
        &self,
        request: Request<ValidateFeaturesRequest>,
    ) -> Result<Response<ValidateFeaturesResponse>, Status> {
        let req = request.into_inner();
        
        info!("Validating features at path: {}", req.output_path);

        // Validate the sample batch
        let result = self.validator.validate_sample_batch(
            Path::new(&req.output_path),
            req.expected_series_count as usize,
        ).await
        .map_err(|e| Status::from(e))?;

        let validation_stats = ValidationStats {
            validation_time_ms: 0, // TODO: Implement timing
            total_series_validated: result.processed_series as i32,
            valid_series: if result.is_valid { result.processed_series as i32 } else { 0 },
            invalid_series: if !result.is_valid { result.processed_series as i32 } else { 0 },
            schema_errors: result.errors.clone(),
        };

        let response = ValidateFeaturesResponse {
            is_valid: result.is_valid,
            validation_errors: result.errors,
            validation_stats: Some(validation_stats),
        };

        Ok(Response::new(response))
    }

    /// Get feature generation statistics
    #[instrument(skip(self, request))]
    async fn get_feature_stats(
        &self,
        request: Request<GetFeatureStatsRequest>,
    ) -> Result<Response<GetFeatureStatsResponse>, Status> {
        let req = request.into_inner();
        
        info!("Getting feature stats for time-series: {}", req.time_series_id);

        if let Some(stats) = self.engine.get_stats(&req.time_series_id).await {
            let feature_stats = FeatureStats {
                feature_name: req.feature_name,
                time_series_id: req.time_series_id,
                mean: 0.0, // TODO: Implement actual statistics calculation
                std_dev: 0.0,
                min_value: 0.0,
                max_value: 0.0,
                median: 0.0,
                null_count: 0,
                total_count: stats.output_rows as i64,
                percentiles: vec![],
            };

            let response = GetFeatureStatsResponse {
                stats: Some(feature_stats),
            };

            Ok(Response::new(response))
        } else {
            Err(Status::not_found("Statistics not found for the given time-series"))
        }
    }

    /// Health check
    #[instrument(skip(self, request))]
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let _req = request.into_inner();
        
        match self.engine.health_check().await {
            Ok(health_status) => {
                let response = HealthCheckResponse {
                    is_healthy: health_status.is_healthy,
                    status: "OK".to_string(),
                    version: health_status.version,
                    uptime_seconds: health_status.uptime_seconds as i64,
                };
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Health check failed: {}", e);
                let response = HealthCheckResponse {
                    is_healthy: false,
                    status: format!("ERROR: {}", e),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    uptime_seconds: self.start_time.elapsed().as_secs() as i64,
                };
                Ok(Response::new(response))
            }
        }
    }
}

impl FeatureEngineeringServiceImpl {
    /// Convert protobuf FeatureConfig to internal FeatureConfig
    fn convert_feature_config(&self, config: FeatureConfig) -> Result<crate::config::FeatureConfig, Status> {
        let lag_features = config.lag_features
            .map(|lf| crate::config::LagFeatureConfig {
                enabled: lf.enabled,
                lag_periods: lf.lag_periods,
                target_columns: lf.target_columns,
            })
            .unwrap_or_default();

        let rolling_window = config.rolling_window
            .map(|rw| crate::config::RollingWindowConfig {
                enabled: rw.enabled,
                window_sizes: rw.window_sizes,
                statistics: rw.statistics,
                target_columns: rw.target_columns,
            })
            .unwrap_or_default();

        let time_features = config.time_features
            .map(|tf| crate::config::TimeBasedFeatureConfig {
                enabled: tf.enabled,
                features: tf.features,
                timestamp_column: tf.timestamp_column,
                timezone: tf.timezone,
            })
            .unwrap_or_default();

        let output = config.output
            .map(|out| crate::config::OutputConfig {
                format: out.format,
                compression: out.compression,
                include_metadata: out.include_metadata,
                validate_schema: out.validate_schema,
            })
            .unwrap_or_default();

        Ok(crate::config::FeatureConfig {
            lag_features,
            rolling_window,
            time_features,
            output,
        })
    }

    /// Convert internal FeatureGenerationStats to protobuf
    fn convert_feature_generation_stats(
        &self,
        stats: crate::FeatureGenerationStats,
    ) -> FeatureGenerationStats {
        FeatureGenerationStats {
            processing_time_ms: stats.processing_time_ms as i64,
            input_rows: stats.input_rows as i64,
            output_rows: stats.output_rows as i64,
            features_generated: stats.features_generated as i32,
            memory_usage_mb: stats.memory_usage_mb as i64,
            feature_names: stats.feature_names,
        }
    }

    /// Convert internal BatchProcessingStats to protobuf
    fn convert_batch_processing_stats(
        &self,
        stats: crate::BatchProcessingStats,
    ) -> BatchProcessingStats {
        BatchProcessingStats {
            total_processing_time_ms: stats.total_processing_time_ms as i64,
            total_time_series: stats.total_time_series as i32,
            successful_series: stats.successful_series as i32,
            failed_series: stats.failed_series as i32,
            total_input_rows: stats.total_input_rows as i64,
            total_output_rows: stats.total_output_rows as i64,
            peak_memory_usage_mb: stats.peak_memory_usage_mb as i64,
        }
    }
}

impl Default for LagFeatureConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            lag_periods: Vec::new(),
            target_columns: Vec::new(),
        }
    }
}

impl Default for RollingWindowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window_sizes: Vec::new(),
            statistics: Vec::new(),
            target_columns: Vec::new(),
        }
    }
}

impl Default for TimeBasedFeatureConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            features: Vec::new(),
            timestamp_column: String::new(),
            timezone: String::new(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: "parquet".to_string(),
            compression: "snappy".to_string(),
            include_metadata: true,
            validate_schema: true,
        }
    }
}

/// gRPC server
pub struct FeatureEngineeringServer {
    service: FeatureEngineeringServiceImpl,
    config: FeatureEngineConfig,
}

impl FeatureEngineeringServer {
    /// Create a new gRPC server
    pub fn new(config: FeatureEngineConfig) -> FeatureEngineResult<Self> {
        let service = FeatureEngineeringServiceImpl::new(config.clone())?;
        
        Ok(Self {
            service,
            config,
        })
    }

    /// Start the gRPC server
    pub async fn serve(&self) -> FeatureEngineResult<()> {
        let addr = format!("{}:{}", self.config.service.host, self.config.service.port)
            .parse()
            .map_err(|e| FeatureEngineError::config(format!("Invalid server address: {}", e)))?;

        info!("Starting Feature Engineering gRPC server on {}", addr);

        let service = feature_engineering_service_server::FeatureEngineeringServiceServer::new(
            self.service.clone()
        );

        tonic::transport::Server::builder()
            .add_service(service)
            .serve(addr)
            .await
            .map_err(|e| FeatureEngineError::grpc(e.into()))?;

        Ok(())
    }
}

/// Clone implementation for the service
impl Clone for FeatureEngineeringServiceImpl {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            validator: Arc::clone(&self.validator),
            start_time: self.start_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Request;
    
    fn create_test_config() -> FeatureEngineConfig {
        FeatureEngineConfig::default()
    }
    
    #[tokio::test]
    async fn test_service_creation() {
        let config = create_test_config();
        let service = FeatureEngineeringServiceImpl::new(config);
        assert!(service.is_ok());
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let config = create_test_config();
        let service = FeatureEngineeringServiceImpl::new(config).unwrap();
        
        let request = Request::new(HealthCheckRequest {
            service_name: "feature-engineering".to_string(),
        });
        
        let response = service.health_check(request).await;
        assert!(response.is_ok());
        
        let health_response = response.unwrap().into_inner();
        assert!(health_response.is_healthy);
        assert_eq!(health_response.status, "OK");
    }
    
    #[test]
    fn test_feature_config_conversion() {
        let config = create_test_config();
        let service = FeatureEngineeringServiceImpl::new(config).unwrap();
        
        let proto_config = FeatureConfig {
            lag_features: Some(LagFeatureConfig {
                enabled: true,
                lag_periods: vec![1, 2, 3],
                target_columns: vec!["kpi_value".to_string()],
            }),
            rolling_window: Some(RollingWindowConfig {
                enabled: true,
                window_sizes: vec![3, 6, 12],
                statistics: vec!["mean".to_string(), "std".to_string()],
                target_columns: vec!["kpi_value".to_string()],
            }),
            time_features: Some(TimeBasedFeatureConfig {
                enabled: true,
                features: vec!["hour_of_day".to_string()],
                timestamp_column: "timestamp".to_string(),
                timezone: "UTC".to_string(),
            }),
            output: Some(OutputConfig {
                format: "parquet".to_string(),
                compression: "snappy".to_string(),
                include_metadata: true,
                validate_schema: true,
            }),
        };
        
        let internal_config = service.convert_feature_config(proto_config);
        assert!(internal_config.is_ok());
        
        let config = internal_config.unwrap();
        assert!(config.lag_features.enabled);
        assert!(config.rolling_window.enabled);
        assert!(config.time_features.enabled);
        assert_eq!(config.output.format, "parquet");
    }
}