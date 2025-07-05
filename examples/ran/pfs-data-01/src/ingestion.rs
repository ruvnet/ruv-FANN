//! Core ingestion engine for processing files and converting to Parquet format

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::*;
use arrow::datatypes::{DataType, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use csv::ReaderBuilder;
use futures::stream::{self, StreamExt};
use serde_json::Value;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncReadExt;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::IngestionConfig;
use crate::error::{ErrorStats, IngestionError, IngestionResult};
use crate::monitoring::IngestionMonitor;
use crate::schema::StandardSchema;
use crate::storage::ParquetWriter;

/// Core ingestion engine
pub struct IngestionEngine {
    config: Arc<IngestionConfig>,
    monitor: Arc<IngestionMonitor>,
    semaphore: Arc<Semaphore>,
    error_stats: Arc<RwLock<ErrorStats>>,
}

impl IngestionEngine {
    pub fn new(config: IngestionConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_files));
        let monitor = Arc::new(IngestionMonitor::new(config.clone()));
        let error_stats = Arc::new(RwLock::new(ErrorStats::new()));
        
        Self {
            config: Arc::new(config),
            monitor,
            semaphore,
            error_stats,
        }
    }
    
    /// Process a directory of files
    pub async fn process_directory(&self, input_dir: &Path, output_dir: &Path) -> IngestionResult<ProcessingResult> {
        let start_time = Instant::now();
        let job_id = Uuid::new_v4().to_string();
        
        info!("Starting ingestion job {} for directory: {:?}", job_id, input_dir);
        
        // Discover files
        let files = self.discover_files(input_dir).await?;
        if files.is_empty() {
            warn!("No files found in directory: {:?}", input_dir);
            return Ok(ProcessingResult::empty(job_id));
        }
        
        info!("Found {} files to process", files.len());
        
        // Create output directory
        tokio::fs::create_dir_all(output_dir).await?;
        
        // Process files concurrently
        let results: Vec<_> = stream::iter(files.into_iter())
            .map(|file_path| {
                let engine = self.clone();
                let output_dir = output_dir.to_path_buf();
                async move {
                    let _permit = engine.semaphore.acquire().await.unwrap();
                    engine.process_file(&file_path, &output_dir).await
                }
            })
            .buffer_unordered(self.config.max_concurrent_files)
            .collect()
            .await;
        
        // Aggregate results
        let mut total_result = ProcessingResult::empty(job_id);
        let mut error_count = 0;
        
        for result in results {
            match result {
                Ok(file_result) => {
                    total_result.merge(file_result);
                }
                Err(e) => {
                    error_count += 1;
                    error!("File processing error: {}", e);
                    
                    // Update error stats
                    let mut stats = self.error_stats.write().await;
                    stats.record_error(&e);
                    
                    // Check error rate
                    let error_rate = stats.error_rate(total_result.files_processed + error_count);
                    if error_rate > self.config.max_error_rate {
                        return Err(IngestionError::ErrorRateExceeded {
                            current_rate: error_rate,
                            max_rate: self.config.max_error_rate,
                        });
                    }
                }
            }
        }
        
        total_result.processing_time = start_time.elapsed();
        total_result.files_failed = error_count;
        
        info!(
            "Ingestion job {} completed: {} files processed, {} failed in {:?}",
            total_result.job_id,
            total_result.files_processed,
            total_result.files_failed,
            total_result.processing_time
        );
        
        Ok(total_result)
    }
    
    /// Process a single file
    pub async fn process_file(&self, input_path: &Path, output_dir: &Path) -> IngestionResult<ProcessingResult> {
        let start_time = Instant::now();
        let job_id = Uuid::new_v4().to_string();
        
        debug!("Processing file: {:?}", input_path);
        
        // Check file size
        let metadata = tokio::fs::metadata(input_path).await?;
        if metadata.len() > self.config.max_file_size_bytes() as u64 {
            return Err(IngestionError::resource_limit_exceeded(
                format!("File too large: {} bytes", metadata.len())
            ));
        }
        
        // Process with timeout
        let processing_future = self.process_file_internal(input_path, output_dir);
        let timeout_duration = Duration::from_secs(self.config.processing_timeout_seconds);
        
        let mut result = match timeout(timeout_duration, processing_future).await {
            Ok(result) => result?,
            Err(_) => return Err(IngestionError::processing_timeout(input_path.to_string_lossy())),
        };
        
        result.job_id = job_id;
        result.processing_time = start_time.elapsed();
        
        debug!("File processing completed in {:?}", result.processing_time);
        
        Ok(result)
    }
    
    async fn process_file_internal(&self, input_path: &Path, output_dir: &Path) -> IngestionResult<ProcessingResult> {
        let extension = input_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let mut result = ProcessingResult::empty(String::new());
        
        match extension.as_str() {
            "csv" => {
                result = self.process_csv_file(input_path, output_dir).await?;
            }
            "json" => {
                result = self.process_json_file(input_path, output_dir).await?;
            }
            _ => {
                return Err(IngestionError::invalid_file_format(input_path.to_string_lossy()));
            }
        }
        
        result.files_processed = 1;
        Ok(result)
    }
    
    async fn process_csv_file(&self, input_path: &Path, output_dir: &Path) -> IngestionResult<ProcessingResult> {
        let mut file = File::open(input_path).await?;
        let mut content = String::new();
        file.read_to_string(&mut content).await?;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(content.as_bytes());
        
        let headers = reader.headers()?.clone();
        let header_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        
        // Map columns to standard schema
        let column_mapping = self.config.schema.map_columns(&header_names);
        
        // Create batches
        let mut batches = Vec::new();
        let mut records = Vec::new();
        let mut row_count = 0;
        let mut error_count = 0;
        
        for result in reader.records() {
            match result {
                Ok(record) => {
                    match self.process_csv_record(&record, &header_names, &column_mapping) {
                        Ok(row_data) => {
                            records.push(row_data);
                            row_count += 1;
                            
                            // Create batch when we reach batch size
                            if records.len() >= self.config.batch_size {
                                let batch = self.create_record_batch(&records)?;
                                batches.push(batch);
                                records.clear();
                            }
                        }
                        Err(e) => {
                            error_count += 1;
                            if !self.config.skip_malformed_rows {
                                return Err(e);
                            }
                            warn!("Skipping malformed row: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error_count += 1;
                    if !self.config.skip_malformed_rows {
                        return Err(IngestionError::CsvParse(e));
                    }
                    warn!("Skipping malformed row: {}", e);
                }
            }
        }
        
        // Process remaining records
        if !records.is_empty() {
            let batch = self.create_record_batch(&records)?;
            batches.push(batch);
        }
        
        // Write to Parquet
        let output_path = self.get_output_path(input_path, output_dir, "parquet")?;
        let writer = ParquetWriter::new(self.config.clone());
        writer.write_batches(&batches, &output_path).await?;
        
        let mut result = ProcessingResult::empty(String::new());
        result.rows_processed = row_count;
        result.rows_failed = error_count;
        result.input_size_bytes = tokio::fs::metadata(input_path).await?.len();
        result.output_size_bytes = tokio::fs::metadata(&output_path).await?.len();
        
        Ok(result)
    }
    
    async fn process_json_file(&self, input_path: &Path, output_dir: &Path) -> IngestionResult<ProcessingResult> {
        let mut file = File::open(input_path).await?;
        let mut content = String::new();
        file.read_to_string(&mut content).await?;
        
        let mut records = Vec::new();
        let mut row_count = 0;
        let mut error_count = 0;
        
        // Handle different JSON formats
        if content.trim().starts_with('[') {
            // JSON array
            let json_array: Vec<Value> = serde_json::from_str(&content)?;
            for value in json_array {
                match self.process_json_record(&value) {
                    Ok(row_data) => {
                        records.push(row_data);
                        row_count += 1;
                    }
                    Err(e) => {
                        error_count += 1;
                        if !self.config.skip_malformed_rows {
                            return Err(e);
                        }
                        warn!("Skipping malformed JSON record: {}", e);
                    }
                }
            }
        } else {
            // JSON Lines format
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                
                match serde_json::from_str::<Value>(line) {
                    Ok(value) => {
                        match self.process_json_record(&value) {
                            Ok(row_data) => {
                                records.push(row_data);
                                row_count += 1;
                            }
                            Err(e) => {
                                error_count += 1;
                                if !self.config.skip_malformed_rows {
                                    return Err(e);
                                }
                                warn!("Skipping malformed JSON record: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error_count += 1;
                        if !self.config.skip_malformed_rows {
                            return Err(IngestionError::JsonParse(e));
                        }
                        warn!("Skipping malformed JSON line: {}", e);
                    }
                }
            }
        }
        
        // Create batches and write to Parquet
        let mut batches = Vec::new();
        for chunk in records.chunks(self.config.batch_size) {
            let batch = self.create_record_batch(chunk)?;
            batches.push(batch);
        }
        
        let output_path = self.get_output_path(input_path, output_dir, "parquet")?;
        let writer = ParquetWriter::new(self.config.clone());
        writer.write_batches(&batches, &output_path).await?;
        
        let mut result = ProcessingResult::empty(String::new());
        result.rows_processed = row_count;
        result.rows_failed = error_count;
        result.input_size_bytes = tokio::fs::metadata(input_path).await?.len();
        result.output_size_bytes = tokio::fs::metadata(&output_path).await?.len();
        
        Ok(result)
    }
    
    fn process_csv_record(
        &self,
        record: &csv::StringRecord,
        headers: &[String],
        column_mapping: &HashMap<String, String>,
    ) -> IngestionResult<RowData> {
        let mut row_data = RowData::new();
        
        for (i, value) in record.iter().enumerate() {
            if let Some(header) = headers.get(i) {
                if let Some(standard_column) = column_mapping.get(header) {
                    let processed_value = self.process_value(value, standard_column)?;
                    row_data.insert(standard_column.clone(), processed_value);
                }
            }
        }
        
        // Validate required columns
        self.validate_row_data(&row_data)?;
        
        Ok(row_data)
    }
    
    fn process_json_record(&self, value: &Value) -> IngestionResult<RowData> {
        let mut row_data = RowData::new();
        
        if let Value::Object(obj) = value {
            for (key, val) in obj {
                let normalized_key = key.to_lowercase().replace(" ", "_").replace("-", "_");
                let string_value = match val {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => "".to_string(),
                    _ => val.to_string(),
                };
                
                // Map to standard columns
                if let Some(standard_column) = self.map_json_key(&normalized_key) {
                    let processed_value = self.process_value(&string_value, &standard_column)?;
                    row_data.insert(standard_column, processed_value);
                }
            }
        }
        
        self.validate_row_data(&row_data)?;
        
        Ok(row_data)
    }
    
    fn map_json_key(&self, key: &str) -> Option<String> {
        // Simple mapping logic - could be made more sophisticated
        match key {
            k if k.contains("timestamp") || k.contains("time") => Some("timestamp".to_string()),
            k if k.contains("cell") && k.contains("id") => Some("cell_id".to_string()),
            k if k.contains("kpi") && k.contains("name") => Some("kpi_name".to_string()),
            k if k.contains("kpi") && k.contains("value") => Some("kpi_value".to_string()),
            k if k.contains("ue") && k.contains("id") => Some("ue_id".to_string()),
            k if k.contains("sector") && k.contains("id") => Some("sector_id".to_string()),
            _ => None,
        }
    }
    
    fn process_value(&self, value: &str, column: &str) -> IngestionResult<ProcessedValue> {
        if self.config.schema.is_null_value(value) {
            return Ok(ProcessedValue::Null);
        }
        
        let data_type = self.config.schema.column_types.get(column)
            .ok_or_else(|| IngestionError::schema_validation(
                format!("unknown column: {}", column)
            ))?;
        
        match data_type.as_str() {
            "string" => Ok(ProcessedValue::String(value.to_string())),
            "int32" => value.parse::<i32>()
                .map(ProcessedValue::Int32)
                .map_err(|_| IngestionError::data_validation(
                    format!("invalid int32 value '{}' for column '{}'", value, column)
                )),
            "int64" => value.parse::<i64>()
                .map(ProcessedValue::Int64)
                .map_err(|_| IngestionError::data_validation(
                    format!("invalid int64 value '{}' for column '{}'", value, column)
                )),
            "float32" => value.parse::<f32>()
                .map(ProcessedValue::Float32)
                .map_err(|_| IngestionError::data_validation(
                    format!("invalid float32 value '{}' for column '{}'", value, column)
                )),
            "float64" => value.parse::<f64>()
                .map(ProcessedValue::Float64)
                .map_err(|_| IngestionError::data_validation(
                    format!("invalid float64 value '{}' for column '{}'", value, column)
                )),
            "boolean" => value.parse::<bool>()
                .map(ProcessedValue::Boolean)
                .map_err(|_| IngestionError::data_validation(
                    format!("invalid boolean value '{}' for column '{}'", value, column)
                )),
            "timestamp" => {
                let timestamp = self.parse_timestamp(value)?;
                self.config.schema.validate_timestamp(&timestamp)?;
                Ok(ProcessedValue::Timestamp(timestamp))
            }
            _ => Err(IngestionError::schema_validation(
                format!("unsupported data type: {}", data_type)
            )),
        }
    }
    
    fn parse_timestamp(&self, value: &str) -> IngestionResult<DateTime<Utc>> {
        // Try multiple timestamp formats
        let formats = vec![
            "%Y-%m-%d %H:%M:%S%.3f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%.3fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%.3f",
            "%Y-%m-%dT%H:%M:%S",
        ];
        
        for format in formats {
            if let Ok(dt) = DateTime::parse_from_str(value, format) {
                return Ok(dt.with_timezone(&Utc));
            }
        }
        
        // Try parsing as Unix timestamp
        if let Ok(timestamp) = value.parse::<i64>() {
            if let Some(dt) = DateTime::from_timestamp(timestamp, 0) {
                return Ok(dt);
            }
        }
        
        Err(IngestionError::data_validation(
            format!("invalid timestamp format: {}", value)
        ))
    }
    
    fn validate_row_data(&self, row_data: &RowData) -> IngestionResult<()> {
        // Check required columns
        for required in &self.config.schema.required_columns {
            if !row_data.contains_key(required) {
                return Err(IngestionError::data_validation(
                    format!("missing required column: {}", required)
                ));
            }
        }
        
        // Validate KPI name if present
        if let Some(ProcessedValue::String(kpi_name)) = row_data.get("kpi_name") {
            self.config.schema.validate_kpi_name(kpi_name)?;
        }
        
        Ok(())
    }
    
    fn create_record_batch(&self, records: &[RowData]) -> IngestionResult<RecordBatch> {
        let schema = self.config.schema.to_arrow_schema()?;
        let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
        
        for field in schema.fields() {
            let array = self.create_array_for_column(field.name(), records, field.data_type())?;
            arrays.push(array);
        }
        
        Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
    }
    
    fn create_array_for_column(
        &self,
        column: &str,
        records: &[RowData],
        data_type: &DataType,
    ) -> IngestionResult<Arc<dyn Array>> {
        match data_type {
            DataType::Utf8 => {
                let values: Vec<Option<String>> = records.iter()
                    .map(|record| match record.get(column) {
                        Some(ProcessedValue::String(s)) => Some(s.clone()),
                        Some(ProcessedValue::Null) => None,
                        _ => None,
                    })
                    .collect();
                Ok(Arc::new(StringArray::from(values)))
            }
            DataType::Int32 => {
                let values: Vec<Option<i32>> = records.iter()
                    .map(|record| match record.get(column) {
                        Some(ProcessedValue::Int32(i)) => Some(*i),
                        Some(ProcessedValue::Null) => None,
                        _ => None,
                    })
                    .collect();
                Ok(Arc::new(Int32Array::from(values)))
            }
            DataType::Int64 => {
                let values: Vec<Option<i64>> = records.iter()
                    .map(|record| match record.get(column) {
                        Some(ProcessedValue::Int64(i)) => Some(*i),
                        Some(ProcessedValue::Null) => None,
                        _ => None,
                    })
                    .collect();
                Ok(Arc::new(Int64Array::from(values)))
            }
            DataType::Float32 => {
                let values: Vec<Option<f32>> = records.iter()
                    .map(|record| match record.get(column) {
                        Some(ProcessedValue::Float32(f)) => Some(*f),
                        Some(ProcessedValue::Null) => None,
                        _ => None,
                    })
                    .collect();
                Ok(Arc::new(Float32Array::from(values)))
            }
            DataType::Float64 => {
                let values: Vec<Option<f64>> = records.iter()
                    .map(|record| match record.get(column) {
                        Some(ProcessedValue::Float64(f)) => Some(*f),
                        Some(ProcessedValue::Null) => None,
                        _ => None,
                    })
                    .collect();
                Ok(Arc::new(Float64Array::from(values)))
            }
            DataType::Boolean => {
                let values: Vec<Option<bool>> = records.iter()
                    .map(|record| match record.get(column) {
                        Some(ProcessedValue::Boolean(b)) => Some(*b),
                        Some(ProcessedValue::Null) => None,
                        _ => None,
                    })
                    .collect();
                Ok(Arc::new(BooleanArray::from(values)))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                let values: Vec<Option<i64>> = records.iter()
                    .map(|record| match record.get(column) {
                        Some(ProcessedValue::Timestamp(dt)) => Some(dt.timestamp_millis()),
                        Some(ProcessedValue::Null) => None,
                        _ => None,
                    })
                    .collect();
                Ok(Arc::new(TimestampMillisecondArray::from(values)))
            }
            _ => Err(IngestionError::schema_validation(
                format!("unsupported data type for column '{}': {:?}", column, data_type)
            )),
        }
    }
    
    async fn discover_files(&self, dir: &Path) -> IngestionResult<Vec<PathBuf>> {
        let mut files = Vec::new();
        
        if self.config.recursive {
            self.discover_files_recursive(dir, &mut files).await?;
        } else {
            let mut entries = tokio::fs::read_dir(dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_file() && self.config.matches_pattern(&path) {
                    files.push(path);
                }
            }
        }
        
        // Sort files for consistent processing order
        files.sort();
        
        Ok(files)
    }
    
    fn discover_files_recursive<'a>(&'a self, dir: &'a Path, files: &'a mut Vec<PathBuf>) -> std::pin::Pin<Box<dyn std::future::Future<Output = IngestionResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut entries = tokio::fs::read_dir(dir).await?;
            
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_file() && self.config.matches_pattern(&path) {
                    files.push(path);
                } else if path.is_dir() {
                    self.discover_files_recursive(&path, files).await?;
                }
            }
            
            Ok(())
        })
    }
    
    fn get_output_path(&self, input_path: &Path, output_dir: &Path, extension: &str) -> IngestionResult<PathBuf> {
        let file_stem = input_path.file_stem()
            .ok_or_else(|| IngestionError::config("invalid input file name"))?;
        
        let output_name = format!("{}.{}", file_stem.to_string_lossy(), extension);
        Ok(output_dir.join(output_name))
    }
}

impl Clone for IngestionEngine {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            monitor: Arc::clone(&self.monitor),
            semaphore: Arc::clone(&self.semaphore),
            error_stats: Arc::clone(&self.error_stats),
        }
    }
}

/// Processed value types
#[derive(Debug, Clone)]
pub enum ProcessedValue {
    String(String),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Boolean(bool),
    Timestamp(DateTime<Utc>),
    Null,
}

/// Row data structure
pub type RowData = HashMap<String, ProcessedValue>;

/// Processing result
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub job_id: String,
    pub files_processed: u64,
    pub files_failed: u64,
    pub rows_processed: u64,
    pub rows_failed: u64,
    pub input_size_bytes: u64,
    pub output_size_bytes: u64,
    pub processing_time: Duration,
}

impl ProcessingResult {
    pub fn empty(job_id: String) -> Self {
        Self {
            job_id,
            files_processed: 0,
            files_failed: 0,
            rows_processed: 0,
            rows_failed: 0,
            input_size_bytes: 0,
            output_size_bytes: 0,
            processing_time: Duration::from_secs(0),
        }
    }
    
    pub fn merge(&mut self, other: ProcessingResult) {
        self.files_processed += other.files_processed;
        self.files_failed += other.files_failed;
        self.rows_processed += other.rows_processed;
        self.rows_failed += other.rows_failed;
        self.input_size_bytes += other.input_size_bytes;
        self.output_size_bytes += other.output_size_bytes;
        self.processing_time = self.processing_time.max(other.processing_time);
    }
    
    pub fn error_rate(&self) -> f64 {
        if self.rows_processed + self.rows_failed == 0 {
            0.0
        } else {
            self.rows_failed as f64 / (self.rows_processed + self.rows_failed) as f64
        }
    }
    
    pub fn throughput_mb_per_second(&self) -> f64 {
        if self.processing_time.as_secs_f64() == 0.0 {
            0.0
        } else {
            (self.input_size_bytes as f64 / 1024.0 / 1024.0) / self.processing_time.as_secs_f64()
        }
    }
    
    pub fn compression_ratio(&self) -> f64 {
        if self.input_size_bytes == 0 {
            0.0
        } else {
            self.output_size_bytes as f64 / self.input_size_bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_ingestion_engine_creation() {
        let config = IngestionConfig::default();
        let engine = IngestionEngine::new(config);
        
        assert!(engine.config.batch_size > 0);
    }
    
    #[test]
    fn test_processing_result_merge() {
        let mut result1 = ProcessingResult::empty("job1".to_string());
        result1.files_processed = 5;
        result1.rows_processed = 1000;
        
        let mut result2 = ProcessingResult::empty("job2".to_string());
        result2.files_processed = 3;
        result2.rows_processed = 500;
        
        result1.merge(result2);
        
        assert_eq!(result1.files_processed, 8);
        assert_eq!(result1.rows_processed, 1500);
    }
    
    #[test]
    fn test_error_rate_calculation() {
        let mut result = ProcessingResult::empty("job1".to_string());
        result.rows_processed = 950;
        result.rows_failed = 50;
        
        assert_eq!(result.error_rate(), 0.05); // 5% error rate
    }
}