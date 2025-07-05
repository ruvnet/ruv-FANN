//! gRPC service implementation for the data ingestion service

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::Stream;
use tokio::sync::RwLock;
use tokio::time::interval;
use tokio_stream::wrappers::IntervalStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::IngestionConfig;
use crate::error::IngestionError;
use crate::ingestion::IngestionEngine;
use crate::monitoring::IngestionMonitor;
use crate::proto::data_ingestion_service_server::DataIngestionService;
use crate::proto::*;
use crate::watcher::{DirectoryWatcher, WatchConfig};

/// gRPC service implementation
pub struct DataIngestionServiceImpl {
    engine: Arc<IngestionEngine>,
    monitor: Arc<IngestionMonitor>,
    watcher: Arc<DirectoryWatcher>,
    active_jobs: Arc<RwLock<HashMap<String, ActiveJob>>>,
    config: Arc<IngestionConfig>,
}

impl DataIngestionServiceImpl {
    pub fn new(config: IngestionConfig) -> Self {
        let config = Arc::new(config);
        let engine = Arc::new(IngestionEngine::new((*config).clone()));
        let monitor = Arc::new(IngestionMonitor::new((*config).clone()));
        let watcher = Arc::new(DirectoryWatcher::new(
            Arc::clone(&config),
            Arc::clone(&engine),
        ));
        
        Self {
            engine,
            monitor,
            watcher,
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    /// Convert IngestionConfig to proto IngestionConfig
    fn config_to_proto(&self, config: &crate::config::IngestionConfig) -> crate::proto::IngestionConfig {
        crate::proto::IngestionConfig {
            file_patterns: config.file_patterns.clone(),
            recursive: config.recursive,
            schema: Some(self.schema_to_proto(&config.schema)),
            batch_size: config.batch_size as i32,
            max_concurrent_files: config.max_concurrent_files as i32,
            max_error_rate: config.max_error_rate,
            skip_malformed_rows: config.skip_malformed_rows,
            compression_codec: config.compression_codec.clone(),
            row_group_size: config.row_group_size as i32,
        }
    }
    
    /// Convert StandardSchema to proto StandardSchema
    fn schema_to_proto(&self, schema: &crate::schema::StandardSchema) -> crate::proto::StandardSchema {
        crate::proto::StandardSchema {
            timestamp_column: schema.timestamp_column.clone(),
            cell_id_column: schema.cell_id_column.clone(),
            kpi_name_column: schema.kpi_name_column.clone(),
            kpi_value_column: schema.kpi_value_column.clone(),
            ue_id_column: schema.ue_id_column.clone().unwrap_or_default(),
            sector_id_column: schema.sector_id_column.clone().unwrap_or_default(),
            column_types: schema.column_types.clone(),
        }
    }
    
    /// Convert proto IngestionConfig to IngestionConfig
    fn proto_to_config(&self, proto_config: &crate::proto::IngestionConfig) -> crate::config::IngestionConfig {
        let mut config = crate::config::IngestionConfig::default();
        
        config.file_patterns = proto_config.file_patterns.clone();
        config.recursive = proto_config.recursive;
        config.batch_size = proto_config.batch_size as usize;
        config.max_concurrent_files = proto_config.max_concurrent_files as usize;
        config.max_error_rate = proto_config.max_error_rate;
        config.skip_malformed_rows = proto_config.skip_malformed_rows;
        config.compression_codec = proto_config.compression_codec.clone();
        config.row_group_size = proto_config.row_group_size as usize;
        
        if let Some(schema) = &proto_config.schema {
            config.schema = self.proto_to_schema(schema);
        }
        
        config
    }
    
    /// Convert proto StandardSchema to StandardSchema
    fn proto_to_schema(&self, proto_schema: &crate::proto::StandardSchema) -> crate::schema::StandardSchema {
        let mut schema = crate::schema::StandardSchema::default();
        
        schema.timestamp_column = proto_schema.timestamp_column.clone();
        schema.cell_id_column = proto_schema.cell_id_column.clone();
        schema.kpi_name_column = proto_schema.kpi_name_column.clone();
        schema.kpi_value_column = proto_schema.kpi_value_column.clone();
        
        if !proto_schema.ue_id_column.is_empty() {
            schema.ue_id_column = Some(proto_schema.ue_id_column.clone());
        }
        
        if !proto_schema.sector_id_column.is_empty() {
            schema.sector_id_column = Some(proto_schema.sector_id_column.clone());
        }
        
        schema.column_types = proto_schema.column_types.clone();
        
        schema
    }
    
    /// Format timestamp as string
    fn format_timestamp(timestamp: SystemTime) -> String {
        timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string()
    }
    
    /// Create IngestionJob from ActiveJob
    fn create_ingestion_job(&self, job_id: &str, job: &ActiveJob) -> IngestionJob {
        let progress = IngestionProgress {
            total_files: job.total_files,
            processed_files: job.processed_files,
            failed_files: job.failed_files,
            total_rows: job.total_rows,
            processed_rows: job.processed_rows,
            failed_rows: job.failed_rows,
            progress_percentage: if job.total_files > 0 {
                (job.processed_files as f64 / job.total_files as f64) * 100.0
            } else {
                0.0
            },
            current_file: job.current_file.clone().unwrap_or_default(),
            estimated_completion: job.estimated_completion
                .map(Self::format_timestamp)
                .unwrap_or_default(),
        };
        
        let metrics = IngestionMetrics {
            throughput_mb_per_second: job.throughput_mb_per_second,
            rows_per_second: job.rows_per_second,
            average_file_processing_time_ms: job.average_file_processing_time_ms,
            error_rate: job.error_rate,
            total_parsing_errors: job.total_parsing_errors,
            total_validation_errors: job.total_validation_errors,
            cpu_usage_percent: 0.0, // Would be updated from system monitor
            memory_usage_mb: 0.0,    // Would be updated from system monitor
            input_size_bytes: job.input_size_bytes as i64,
            output_size_bytes: job.output_size_bytes as i64,
            compression_ratio: if job.input_size_bytes > 0 {
                job.output_size_bytes as f64 / job.input_size_bytes as f64
            } else {
                0.0
            },
        };
        
        IngestionJob {
            job_id: job_id.to_string(),
            input_directory: job.input_directory.clone(),
            output_directory: job.output_directory.clone(),
            status: job.status.clone(),
            created_at: Self::format_timestamp(job.created_at),
            updated_at: Self::format_timestamp(job.updated_at),
            config: Some(self.config_to_proto(&job.config)),
            progress: Some(progress),
            metrics: Some(metrics),
        }
    }
}

#[tonic::async_trait]
impl DataIngestionService for DataIngestionServiceImpl {
    async fn start_ingestion(
        &self,
        request: Request<StartIngestionRequest>,
    ) -> Result<Response<StartIngestionResponse>, Status> {
        let req = request.into_inner();
        let job_id = Uuid::new_v4().to_string();
        
        info!("Starting ingestion job {}: {} -> {}", job_id, req.input_directory, req.output_directory);
        
        // Validate request
        if req.input_directory.is_empty() {
            return Err(Status::invalid_argument("input_directory is required"));
        }
        
        if req.output_directory.is_empty() {
            return Err(Status::invalid_argument("output_directory is required"));
        }
        
        // Create job configuration
        let config = if let Some(proto_config) = req.config {
            self.proto_to_config(&proto_config)
        } else {
            (*self.config).clone()
        };
        
        // Validate configuration
        if let Err(e) = config.validate() {
            return Err(Status::invalid_argument(format!("invalid configuration: {}", e)));
        }
        
        // Create active job
        let active_job = ActiveJob {
            input_directory: req.input_directory.clone(),
            output_directory: req.output_directory.clone(),
            config,
            status: "running".to_string(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            total_files: 0,
            processed_files: 0,
            failed_files: 0,
            total_rows: 0,
            processed_rows: 0,
            failed_rows: 0,
            current_file: None,
            estimated_completion: None,
            throughput_mb_per_second: 0.0,
            rows_per_second: 0.0,
            average_file_processing_time_ms: 0.0,
            error_rate: 0.0,
            total_parsing_errors: 0,
            total_validation_errors: 0,
            input_size_bytes: 0,
            output_size_bytes: 0,
        };
        
        self.active_jobs.write().await.insert(job_id.clone(), active_job);
        
        // Start ingestion in background
        let engine = Arc::clone(&self.engine);
        let monitor = Arc::clone(&self.monitor);
        let active_jobs = Arc::clone(&self.active_jobs);
        let job_id_clone = job_id.clone();
        let input_dir = req.input_directory.clone();
        let output_dir = req.output_directory.clone();
        
        tokio::spawn(async move {
            let input_path = std::path::PathBuf::from(input_dir);
            let output_path = std::path::PathBuf::from(output_dir);
            
            match engine.process_directory(&input_path, &output_path).await {
                Ok(result) => {
                    // Update job with success
                    let mut jobs = active_jobs.write().await;
                    if let Some(job) = jobs.get_mut(&job_id_clone) {
                        job.status = "completed".to_string();
                        job.updated_at = SystemTime::now();
                        job.processed_files = result.files_processed as i64;
                        job.failed_files = result.files_failed as i64;
                        job.processed_rows = result.rows_processed as i64;
                        job.failed_rows = result.rows_failed as i64;
                        job.input_size_bytes = result.input_size_bytes;
                        job.output_size_bytes = result.output_size_bytes;
                        job.error_rate = result.error_rate();
                        job.throughput_mb_per_second = result.throughput_mb_per_second();
                    }
                    
                    monitor.record_job_completion(&job_id_clone, true).await;
                    info!("Ingestion job {} completed successfully", job_id_clone);
                }
                Err(e) => {
                    // Update job with failure
                    let mut jobs = active_jobs.write().await;
                    if let Some(job) = jobs.get_mut(&job_id_clone) {
                        job.status = "failed".to_string();
                        job.updated_at = SystemTime::now();
                    }
                    
                    monitor.record_job_completion(&job_id_clone, false).await;
                    error!("Ingestion job {} failed: {}", job_id_clone, e);
                }
            }
        });
        
        Ok(Response::new(StartIngestionResponse {
            job_id,
            status: "running".to_string(),
            message: "Ingestion job started successfully".to_string(),
        }))
    }
    
    async fn stop_ingestion(
        &self,
        request: Request<StopIngestionRequest>,
    ) -> Result<Response<StopIngestionResponse>, Status> {
        let req = request.into_inner();
        
        info!("Stopping ingestion job: {}", req.job_id);
        
        let mut active_jobs = self.active_jobs.write().await;
        
        if let Some(job) = active_jobs.get_mut(&req.job_id) {
            if job.status == "running" {
                job.status = "stopped".to_string();
                job.updated_at = SystemTime::now();
                
                self.monitor.record_job_completion(&req.job_id, false).await;
                
                Ok(Response::new(StopIngestionResponse {
                    job_id: req.job_id,
                    status: "stopped".to_string(),
                    message: "Ingestion job stopped successfully".to_string(),
                }))
            } else {
                Err(Status::failed_precondition(
                    format!("Job {} is not running (status: {})", req.job_id, job.status)
                ))
            }
        } else {
            Err(Status::not_found(format!("Job not found: {}", req.job_id)))
        }
    }
    
    async fn get_ingestion_status(
        &self,
        request: Request<GetIngestionStatusRequest>,
    ) -> Result<Response<GetIngestionStatusResponse>, Status> {
        let req = request.into_inner();
        
        let active_jobs = self.active_jobs.read().await;
        
        if let Some(job) = active_jobs.get(&req.job_id) {
            let progress = IngestionProgress {
                total_files: job.total_files,
                processed_files: job.processed_files,
                failed_files: job.failed_files,
                total_rows: job.total_rows,
                processed_rows: job.processed_rows,
                failed_rows: job.failed_rows,
                progress_percentage: if job.total_files > 0 {
                    (job.processed_files as f64 / job.total_files as f64) * 100.0
                } else {
                    0.0
                },
                current_file: job.current_file.clone().unwrap_or_default(),
                estimated_completion: job.estimated_completion
                    .map(Self::format_timestamp)
                    .unwrap_or_default(),
            };
            
            Ok(Response::new(GetIngestionStatusResponse {
                job_id: req.job_id,
                status: job.status.clone(),
                progress: Some(progress),
                errors: Vec::new(), // Could be populated from error tracking
            }))
        } else {
            Err(Status::not_found(format!("Job not found: {}", req.job_id)))
        }
    }
    
    async fn get_ingestion_metrics(
        &self,
        request: Request<GetIngestionMetricsRequest>,
    ) -> Result<Response<GetIngestionMetricsResponse>, Status> {
        let req = request.into_inner();
        
        let active_jobs = self.active_jobs.read().await;
        
        if let Some(job) = active_jobs.get(&req.job_id) {
            let metrics = IngestionMetrics {
                throughput_mb_per_second: job.throughput_mb_per_second,
                rows_per_second: job.rows_per_second,
                average_file_processing_time_ms: job.average_file_processing_time_ms,
                error_rate: job.error_rate,
                total_parsing_errors: job.total_parsing_errors,
                total_validation_errors: job.total_validation_errors,
                cpu_usage_percent: 0.0, // Would be updated from system monitor
                memory_usage_mb: 0.0,    // Would be updated from system monitor
                input_size_bytes: job.input_size_bytes as i64,
                output_size_bytes: job.output_size_bytes as i64,
                compression_ratio: if job.input_size_bytes > 0 {
                    job.output_size_bytes as f64 / job.input_size_bytes as f64
                } else {
                    0.0
                },
            };
            
            Ok(Response::new(GetIngestionMetricsResponse {
                job_id: req.job_id,
                metrics: Some(metrics),
            }))
        } else {
            Err(Status::not_found(format!("Job not found: {}", req.job_id)))
        }
    }
    
    async fn list_ingestion_jobs(
        &self,
        request: Request<ListIngestionJobsRequest>,
    ) -> Result<Response<ListIngestionJobsResponse>, Status> {
        let req = request.into_inner();
        
        let active_jobs = self.active_jobs.read().await;
        let mut jobs = Vec::new();
        
        for (job_id, job) in active_jobs.iter() {
            // Filter by status if specified
            if !req.status_filter.is_empty() && job.status != req.status_filter {
                continue;
            }
            
            jobs.push(self.create_ingestion_job(job_id, job));
        }
        
        Ok(Response::new(ListIngestionJobsResponse { jobs }))
    }
    
    type StreamIngestionEventsStream = Pin<Box<dyn Stream<Item = Result<IngestionEvent, Status>> + Send>>;
    
    async fn stream_ingestion_events(
        &self,
        request: Request<StreamIngestionEventsRequest>,
    ) -> Result<Response<Self::StreamIngestionEventsStream>, Status> {
        let req = request.into_inner();
        let job_id_filter = if req.job_id.is_empty() { None } else { Some(req.job_id) };
        
        info!("Starting event stream for job filter: {:?}", job_id_filter);
        
        // Create a stream that emits events periodically
        let active_jobs = Arc::clone(&self.active_jobs);
        let interval_stream = IntervalStream::new(interval(Duration::from_secs(1)));
        
        let event_stream = interval_stream.filter_map(move |_| {
            let active_jobs_clone = Arc::clone(&active_jobs);
            let job_id_filter_clone = job_id_filter.clone();
            
            async move {
                let jobs = active_jobs_clone.read().await;
                
                for (job_id, job) in jobs.iter() {
                    if let Some(ref filter) = job_id_filter_clone {
                        if job_id != filter {
                            continue;
                        }
                    }
                    
                    if job.status == "running" && job.current_file.is_some() {
                        let event = IngestionEvent {
                            job_id: job_id.clone(),
                            event_type: "file_processing".to_string(),
                            file_path: job.current_file.clone().unwrap_or_default(),
                            timestamp: Self::format_timestamp(SystemTime::now()),
                            message: format!("Processing file: {}", job.current_file.as_ref().unwrap()),
                            metadata: HashMap::new(),
                        };
                        
                        return Some(Ok(event));
                    }
                }
                
                None
            }
        });
        
        Ok(Response::new(Box::pin(event_stream)))
    }
}

/// Active job tracking
#[derive(Debug, Clone)]
struct ActiveJob {
    input_directory: String,
    output_directory: String,
    config: crate::config::IngestionConfig,
    status: String,
    created_at: SystemTime,
    updated_at: SystemTime,
    total_files: i64,
    processed_files: i64,
    failed_files: i64,
    total_rows: i64,
    processed_rows: i64,
    failed_rows: i64,
    current_file: Option<String>,
    estimated_completion: Option<SystemTime>,
    throughput_mb_per_second: f64,
    rows_per_second: f64,
    average_file_processing_time_ms: f64,
    error_rate: f64,
    total_parsing_errors: i64,
    total_validation_errors: i64,
    input_size_bytes: u64,
    output_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_creation() {
        let config = crate::config::IngestionConfig::default();
        let service = DataIngestionServiceImpl::new(config);
        
        // Test that the service was created successfully
        assert!(Arc::strong_count(&service.engine) == 1);
    }
    
    #[tokio::test]
    async fn test_config_conversion() {
        let config = crate::config::IngestionConfig::default();
        let service = DataIngestionServiceImpl::new(config.clone());
        
        let proto_config = service.config_to_proto(&config);
        let converted_back = service.proto_to_config(&proto_config);
        
        assert_eq!(config.batch_size, converted_back.batch_size);
        assert_eq!(config.max_concurrent_files, converted_back.max_concurrent_files);
        assert_eq!(config.compression_codec, converted_back.compression_codec);
    }
    
    #[tokio::test]
    async fn test_list_empty_jobs() {
        let config = crate::config::IngestionConfig::default();
        let service = DataIngestionServiceImpl::new(config);
        
        let request = Request::new(ListIngestionJobsRequest {
            status_filter: String::new(),
        });
        
        let response = service.list_ingestion_jobs(request).await.unwrap();
        assert!(response.into_inner().jobs.is_empty());
    }
}