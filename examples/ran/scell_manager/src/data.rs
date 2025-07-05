//! Data pipeline and management for SCell Manager

use crate::types::*;
use anyhow::{anyhow, Result};
use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use log::{debug, info, warn};
use parquet::arrow::{ArrowReader, ArrowWriter, ParquetFileArrowReader};
use parquet::file::reader::{FileReader, SerializedFileReader};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Data pipeline for processing UE metrics and training data
#[derive(Debug)]
pub struct DataPipeline {
    data_dir: PathBuf,
    cache: HashMap<String, CachedDataset>,
    max_cache_size: usize,
}

impl DataPipeline {
    /// Create a new data pipeline
    pub fn new(data_dir: PathBuf, max_cache_size: usize) -> Self {
        std::fs::create_dir_all(&data_dir).ok();
        
        Self {
            data_dir,
            cache: HashMap::new(),
            max_cache_size,
        }
    }
    
    /// Load UE metrics from Parquet file
    pub fn load_ue_metrics(&mut self, file_path: &Path) -> Result<Vec<UEMetrics>> {
        info!("Loading UE metrics from: {:?}", file_path);
        
        // Check cache first
        let cache_key = file_path.to_string_lossy().to_string();
        if let Some(cached) = self.cache.get(&cache_key) {
            if !cached.is_expired() {
                debug!("Cache hit for file: {:?}", file_path);
                return Ok(cached.ue_metrics.clone());
            }
        }
        
        // Load from file
        let df = LazyFrame::scan_parquet(file_path, ScanArgsParquet::default())?
            .collect()?;
        
        let mut metrics = Vec::new();
        
        // Convert DataFrame to UEMetrics
        let rows = df.height();
        for i in 0..rows {
            let row = df.get_row(i)?;
            
            let ue_metrics = UEMetrics {
                ue_id: self.extract_string_value(&row, "ue_id")?,
                pcell_throughput_mbps: self.extract_f32_value(&row, "pcell_throughput_mbps")?,
                buffer_status_report_bytes: self.extract_i64_value(&row, "buffer_status_report_bytes")?,
                pcell_cqi: self.extract_f32_value(&row, "pcell_cqi")?,
                pcell_rsrp: self.extract_f32_value(&row, "pcell_rsrp")?,
                pcell_sinr: self.extract_f32_value(&row, "pcell_sinr")?,
                active_bearers: self.extract_i32_value(&row, "active_bearers")?,
                data_rate_req_mbps: self.extract_f32_value(&row, "data_rate_req_mbps")?,
                timestamp_utc: self.extract_datetime_value(&row, "timestamp_utc")?,
            };
            
            metrics.push(ue_metrics);
        }
        
        // Cache the results
        self.cache_dataset(cache_key, CachedDataset::new(metrics.clone()));
        
        info!("Loaded {} UE metrics records", metrics.len());
        Ok(metrics)
    }
    
    /// Save UE metrics to Parquet file
    pub fn save_ue_metrics(&self, metrics: &[UEMetrics], file_path: &Path) -> Result<()> {
        info!("Saving {} UE metrics to: {:?}", metrics.len(), file_path);
        
        // Create parent directory
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Convert to Polars DataFrame
        let df = self.metrics_to_dataframe(metrics)?;
        
        // Write to Parquet
        let mut file = File::create(file_path)?;
        ParquetWriter::new(&mut file)
            .finish(&mut df.clone())?;
        
        info!("Successfully saved UE metrics to: {:?}", file_path);
        Ok(())
    }
    
    /// Load training examples from Parquet file
    pub fn load_training_examples(&mut self, file_path: &Path) -> Result<Vec<TrainingExample>> {
        info!("Loading training examples from: {:?}", file_path);
        
        let df = LazyFrame::scan_parquet(file_path, ScanArgsParquet::default())?
            .collect()?;
        
        let mut examples = Vec::new();
        let rows = df.height();
        
        for i in 0..rows {
            let row = df.get_row(i)?;
            
            // Extract input metrics
            let input_metrics = UEMetrics {
                ue_id: self.extract_string_value(&row, "ue_id")?,
                pcell_throughput_mbps: self.extract_f32_value(&row, "input_pcell_throughput_mbps")?,
                buffer_status_report_bytes: self.extract_i64_value(&row, "input_buffer_status_report_bytes")?,
                pcell_cqi: self.extract_f32_value(&row, "input_pcell_cqi")?,
                pcell_rsrp: self.extract_f32_value(&row, "input_pcell_rsrp")?,
                pcell_sinr: self.extract_f32_value(&row, "input_pcell_sinr")?,
                active_bearers: self.extract_i32_value(&row, "input_active_bearers")?,
                data_rate_req_mbps: self.extract_f32_value(&row, "input_data_rate_req_mbps")?,
                timestamp_utc: self.extract_datetime_value(&row, "input_timestamp_utc")?,
            };
            
            // For simplicity, we'll not load historical sequence from this format
            // In a real implementation, you might use a nested structure or separate files
            let historical_sequence = Vec::new();
            
            let example = TrainingExample {
                input_metrics,
                historical_sequence,
                actual_scell_needed: self.extract_bool_value(&row, "actual_scell_needed")?,
                actual_throughput_demand: self.extract_f32_value(&row, "actual_throughput_demand")?,
            };
            
            examples.push(example);
        }
        
        info!("Loaded {} training examples", examples.len());
        Ok(examples)
    }
    
    /// Save training examples to Parquet file
    pub fn save_training_examples(&self, examples: &[TrainingExample], file_path: &Path) -> Result<()> {
        info!("Saving {} training examples to: {:?}", examples.len(), file_path);
        
        // Create parent directory
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Convert to Polars DataFrame
        let df = self.training_examples_to_dataframe(examples)?;
        
        // Write to Parquet
        let mut file = File::create(file_path)?;
        ParquetWriter::new(&mut file)
            .finish(&mut df.clone())?;
        
        info!("Successfully saved training examples to: {:?}", file_path);
        Ok(())
    }
    
    /// Generate synthetic training data for testing
    pub fn generate_synthetic_training_data(&self, num_samples: usize) -> Vec<TrainingExample> {
        info!("Generating {} synthetic training examples", num_samples);
        
        let mut examples = Vec::new();
        let mut rng = fastrand::Rng::new();
        
        for i in 0..num_samples {
            let ue_id = format!("synthetic_ue_{:06}", i);
            
            // Generate realistic UE metrics
            let base_throughput = rng.f32() * 200.0; // 0-200 Mbps
            let buffer_size = rng.i64() % 1000000; // 0-1MB
            let cqi = rng.f32() * 15.0; // 0-15
            let rsrp = -120.0 + rng.f32() * 50.0; // -120 to -70 dBm
            let sinr = -10.0 + rng.f32() * 30.0; // -10 to 20 dB
            let active_bearers = 1 + (rng.u32() % 4) as i32; // 1-4 bearers
            let data_rate_req = base_throughput * (0.8 + rng.f32() * 0.4); // 80-120% of base
            
            let input_metrics = UEMetrics {
                ue_id: ue_id.clone(),
                pcell_throughput_mbps: base_throughput,
                buffer_status_report_bytes: buffer_size,
                pcell_cqi: cqi,
                pcell_rsrp: rsrp,
                pcell_sinr: sinr,
                active_bearers,
                data_rate_req_mbps: data_rate_req,
                timestamp_utc: Utc::now() - chrono::Duration::seconds(rng.i64() % 3600),
            };
            
            // Generate historical sequence
            let sequence_length = 5 + (rng.usize() % 10); // 5-15 historical points
            let mut historical_sequence = Vec::new();
            
            for j in 0..sequence_length {
                let historical_metrics = UEMetrics {
                    ue_id: ue_id.clone(),
                    pcell_throughput_mbps: base_throughput * (0.7 + rng.f32() * 0.6),
                    buffer_status_report_bytes: (buffer_size as f64 * (0.5 + rng.f64() * 1.0)) as i64,
                    pcell_cqi: (cqi * (0.8 + rng.f32() * 0.4)).max(0.0).min(15.0),
                    pcell_rsrp: rsrp + (rng.f32() - 0.5) * 10.0,
                    pcell_sinr: sinr + (rng.f32() - 0.5) * 5.0,
                    active_bearers,
                    data_rate_req_mbps: data_rate_req * (0.6 + rng.f32() * 0.8),
                    timestamp_utc: input_metrics.timestamp_utc - chrono::Duration::seconds((sequence_length - j) as i64 * 10),
                };
                historical_sequence.push(historical_metrics);
            }
            
            // Determine if SCell is needed based on heuristics
            let high_throughput = base_throughput > 80.0;
            let high_demand = data_rate_req > 100.0;
            let good_signal = cqi > 8.0 && sinr > 5.0;
            let large_buffer = buffer_size > 100000;
            
            let score = (high_throughput as u32) + (high_demand as u32) + 
                       (good_signal as u32) + (large_buffer as u32);
            
            let actual_scell_needed = score >= 2;
            let actual_throughput_demand = if actual_scell_needed {
                data_rate_req * (1.2 + rng.f32() * 0.5) // 120-170% of requested
            } else {
                data_rate_req * (0.7 + rng.f32() * 0.4) // 70-110% of requested
            };
            
            let example = TrainingExample {
                input_metrics,
                historical_sequence,
                actual_scell_needed,
                actual_throughput_demand,
            };
            
            examples.push(example);
        }
        
        info!("Generated {} synthetic training examples", examples.len());
        examples
    }
    
    /// Preprocess UE metrics for model training
    pub fn preprocess_metrics(&self, metrics: &[UEMetrics]) -> Result<Vec<UEMetrics>> {
        info!("Preprocessing {} UE metrics", metrics.len());
        
        let mut processed = Vec::new();
        
        for metric in metrics {
            let mut processed_metric = metric.clone();
            
            // Clip values to reasonable ranges
            processed_metric.pcell_throughput_mbps = processed_metric.pcell_throughput_mbps.max(0.0).min(1000.0);
            processed_metric.pcell_cqi = processed_metric.pcell_cqi.max(0.0).min(15.0);
            processed_metric.pcell_rsrp = processed_metric.pcell_rsrp.max(-140.0).min(-40.0);
            processed_metric.pcell_sinr = processed_metric.pcell_sinr.max(-20.0).min(40.0);
            processed_metric.active_bearers = processed_metric.active_bearers.max(0).min(10);
            processed_metric.data_rate_req_mbps = processed_metric.data_rate_req_mbps.max(0.0).min(2000.0);
            
            // Filter out invalid records
            if processed_metric.pcell_throughput_mbps >= 0.0 && 
               processed_metric.pcell_cqi >= 0.0 &&
               processed_metric.buffer_status_report_bytes >= 0 {
                processed.push(processed_metric);
            }
        }
        
        info!("Preprocessed {} valid metrics from {} input metrics", processed.len(), metrics.len());
        Ok(processed)
    }
    
    /// Get data statistics
    pub fn get_data_statistics(&self, metrics: &[UEMetrics]) -> DataStatistics {
        if metrics.is_empty() {
            return DataStatistics::default();
        }
        
        let throughputs: Vec<f32> = metrics.iter().map(|m| m.pcell_throughput_mbps).collect();
        let cqis: Vec<f32> = metrics.iter().map(|m| m.pcell_cqi).collect();
        let rsrps: Vec<f32> = metrics.iter().map(|m| m.pcell_rsrp).collect();
        let sinrs: Vec<f32> = metrics.iter().map(|m| m.pcell_sinr).collect();
        
        DataStatistics {
            total_records: metrics.len(),
            unique_ues: metrics.iter().map(|m| &m.ue_id).collect::<std::collections::HashSet<_>>().len(),
            throughput_stats: self.calculate_stats(&throughputs),
            cqi_stats: self.calculate_stats(&cqis),
            rsrp_stats: self.calculate_stats(&rsrps),
            sinr_stats: self.calculate_stats(&sinrs),
            time_range: self.calculate_time_range(metrics),
        }
    }
    
    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        info!("Data cache cleared");
    }
    
    // Private helper methods
    
    fn cache_dataset(&mut self, key: String, dataset: CachedDataset) {
        if self.cache.len() >= self.max_cache_size {
            // Remove oldest entry (simple LRU-like behavior)
            if let Some(oldest_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest_key);
            }
        }
        self.cache.insert(key, dataset);
    }
    
    fn metrics_to_dataframe(&self, metrics: &[UEMetrics]) -> Result<DataFrame> {
        let ue_ids: Vec<String> = metrics.iter().map(|m| m.ue_id.clone()).collect();
        let throughputs: Vec<f32> = metrics.iter().map(|m| m.pcell_throughput_mbps).collect();
        let buffer_sizes: Vec<i64> = metrics.iter().map(|m| m.buffer_status_report_bytes).collect();
        let cqis: Vec<f32> = metrics.iter().map(|m| m.pcell_cqi).collect();
        let rsrps: Vec<f32> = metrics.iter().map(|m| m.pcell_rsrp).collect();
        let sinrs: Vec<f32> = metrics.iter().map(|m| m.pcell_sinr).collect();
        let bearers: Vec<i32> = metrics.iter().map(|m| m.active_bearers).collect();
        let data_rates: Vec<f32> = metrics.iter().map(|m| m.data_rate_req_mbps).collect();
        let timestamps: Vec<i64> = metrics.iter().map(|m| m.timestamp_utc.timestamp()).collect();
        
        let df = df! [
            "ue_id" => ue_ids,
            "pcell_throughput_mbps" => throughputs,
            "buffer_status_report_bytes" => buffer_sizes,
            "pcell_cqi" => cqis,
            "pcell_rsrp" => rsrps,
            "pcell_sinr" => sinrs,
            "active_bearers" => bearers,
            "data_rate_req_mbps" => data_rates,
            "timestamp_utc" => timestamps,
        ]?;
        
        Ok(df)
    }
    
    fn training_examples_to_dataframe(&self, examples: &[TrainingExample]) -> Result<DataFrame> {
        let ue_ids: Vec<String> = examples.iter().map(|e| e.input_metrics.ue_id.clone()).collect();
        let input_throughputs: Vec<f32> = examples.iter().map(|e| e.input_metrics.pcell_throughput_mbps).collect();
        let input_buffers: Vec<i64> = examples.iter().map(|e| e.input_metrics.buffer_status_report_bytes).collect();
        let input_cqis: Vec<f32> = examples.iter().map(|e| e.input_metrics.pcell_cqi).collect();
        let input_rsrps: Vec<f32> = examples.iter().map(|e| e.input_metrics.pcell_rsrp).collect();
        let input_sinrs: Vec<f32> = examples.iter().map(|e| e.input_metrics.pcell_sinr).collect();
        let input_bearers: Vec<i32> = examples.iter().map(|e| e.input_metrics.active_bearers).collect();
        let input_data_rates: Vec<f32> = examples.iter().map(|e| e.input_metrics.data_rate_req_mbps).collect();
        let input_timestamps: Vec<i64> = examples.iter().map(|e| e.input_metrics.timestamp_utc.timestamp()).collect();
        let actual_scell_needed: Vec<bool> = examples.iter().map(|e| e.actual_scell_needed).collect();
        let actual_throughput_demand: Vec<f32> = examples.iter().map(|e| e.actual_throughput_demand).collect();
        
        let df = df! [
            "ue_id" => ue_ids,
            "input_pcell_throughput_mbps" => input_throughputs,
            "input_buffer_status_report_bytes" => input_buffers,
            "input_pcell_cqi" => input_cqis,
            "input_pcell_rsrp" => input_rsrps,
            "input_pcell_sinr" => input_sinrs,
            "input_active_bearers" => input_bearers,
            "input_data_rate_req_mbps" => input_data_rates,
            "input_timestamp_utc" => input_timestamps,
            "actual_scell_needed" => actual_scell_needed,
            "actual_throughput_demand" => actual_throughput_demand,
        ]?;
        
        Ok(df)
    }
    
    fn extract_string_value(&self, row: &Row, column: &str) -> Result<String> {
        row.0.iter()
            .find(|(name, _)| name == column)
            .and_then(|(_, value)| value.to_string().ok())
            .ok_or_else(|| anyhow!("Column not found or invalid type: {}", column))
    }
    
    fn extract_f32_value(&self, row: &Row, column: &str) -> Result<f32> {
        row.0.iter()
            .find(|(name, _)| name == column)
            .and_then(|(_, value)| {
                match value {
                    AnyValue::Float32(v) => Some(*v),
                    AnyValue::Float64(v) => Some(*v as f32),
                    AnyValue::Int32(v) => Some(*v as f32),
                    AnyValue::Int64(v) => Some(*v as f32),
                    _ => None,
                }
            })
            .ok_or_else(|| anyhow!("Column not found or invalid type: {}", column))
    }
    
    fn extract_i32_value(&self, row: &Row, column: &str) -> Result<i32> {
        row.0.iter()
            .find(|(name, _)| name == column)
            .and_then(|(_, value)| {
                match value {
                    AnyValue::Int32(v) => Some(*v),
                    AnyValue::Int64(v) => Some(*v as i32),
                    _ => None,
                }
            })
            .ok_or_else(|| anyhow!("Column not found or invalid type: {}", column))
    }
    
    fn extract_i64_value(&self, row: &Row, column: &str) -> Result<i64> {
        row.0.iter()
            .find(|(name, _)| name == column)
            .and_then(|(_, value)| {
                match value {
                    AnyValue::Int64(v) => Some(*v),
                    AnyValue::Int32(v) => Some(*v as i64),
                    _ => None,
                }
            })
            .ok_or_else(|| anyhow!("Column not found or invalid type: {}", column))
    }
    
    fn extract_bool_value(&self, row: &Row, column: &str) -> Result<bool> {
        row.0.iter()
            .find(|(name, _)| name == column)
            .and_then(|(_, value)| {
                match value {
                    AnyValue::Boolean(v) => Some(*v),
                    _ => None,
                }
            })
            .ok_or_else(|| anyhow!("Column not found or invalid type: {}", column))
    }
    
    fn extract_datetime_value(&self, row: &Row, column: &str) -> Result<DateTime<Utc>> {
        row.0.iter()
            .find(|(name, _)| name == column)
            .and_then(|(_, value)| {
                match value {
                    AnyValue::Int64(timestamp) => {
                        DateTime::from_timestamp(*timestamp, 0)
                    }
                    _ => None,
                }
            })
            .ok_or_else(|| anyhow!("Column not found or invalid type: {}", column))
    }
    
    fn calculate_stats(&self, values: &[f32]) -> StatsSummary {
        if values.is_empty() {
            return StatsSummary::default();
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        
        StatsSummary {
            mean,
            std,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            p25: sorted[sorted.len() / 4],
            p50: sorted[sorted.len() / 2],
            p75: sorted[3 * sorted.len() / 4],
        }
    }
    
    fn calculate_time_range(&self, metrics: &[UEMetrics]) -> (DateTime<Utc>, DateTime<Utc>) {
        let timestamps: Vec<DateTime<Utc>> = metrics.iter().map(|m| m.timestamp_utc).collect();
        let min_time = timestamps.iter().min().copied().unwrap_or_else(Utc::now);
        let max_time = timestamps.iter().max().copied().unwrap_or_else(Utc::now);
        (min_time, max_time)
    }
}

/// Cached dataset with expiration
#[derive(Debug, Clone)]
struct CachedDataset {
    ue_metrics: Vec<UEMetrics>,
    cached_at: DateTime<Utc>,
    ttl: chrono::Duration,
}

impl CachedDataset {
    fn new(ue_metrics: Vec<UEMetrics>) -> Self {
        Self {
            ue_metrics,
            cached_at: Utc::now(),
            ttl: chrono::Duration::minutes(30), // 30 minute TTL
        }
    }
    
    fn is_expired(&self) -> bool {
        Utc::now() > self.cached_at + self.ttl
    }
}

/// Data statistics summary
#[derive(Debug, Clone, Default)]
pub struct DataStatistics {
    pub total_records: usize,
    pub unique_ues: usize,
    pub throughput_stats: StatsSummary,
    pub cqi_stats: StatsSummary,
    pub rsrp_stats: StatsSummary,
    pub sinr_stats: StatsSummary,
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
}

/// Statistical summary for a metric
#[derive(Debug, Clone, Default)]
pub struct StatsSummary {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_data_pipeline_creation() {
        let temp_dir = tempdir().unwrap();
        let pipeline = DataPipeline::new(temp_dir.path().to_path_buf(), 100);
        assert_eq!(pipeline.cache.len(), 0);
    }
    
    #[test]
    fn test_synthetic_data_generation() {
        let temp_dir = tempdir().unwrap();
        let pipeline = DataPipeline::new(temp_dir.path().to_path_buf(), 100);
        
        let examples = pipeline.generate_synthetic_training_data(10);
        assert_eq!(examples.len(), 10);
        
        for example in &examples {
            assert!(!example.input_metrics.ue_id.is_empty());
            assert!(example.input_metrics.pcell_throughput_mbps >= 0.0);
            assert!(example.input_metrics.pcell_cqi >= 0.0 && example.input_metrics.pcell_cqi <= 15.0);
            assert!(!example.historical_sequence.is_empty());
        }
    }
    
    #[test]
    fn test_data_preprocessing() {
        let temp_dir = tempdir().unwrap();
        let pipeline = DataPipeline::new(temp_dir.path().to_path_buf(), 100);
        
        let mut metrics = vec![
            UEMetrics::new("ue1".to_string()),
            UEMetrics::new("ue2".to_string()),
        ];
        
        // Add some invalid data
        metrics[0].pcell_throughput_mbps = -10.0; // Invalid
        metrics[1].pcell_cqi = 20.0; // Will be clipped
        
        let processed = pipeline.preprocess_metrics(&metrics).unwrap();
        
        // Should filter out invalid record
        assert_eq!(processed.len(), 1);
        assert_eq!(processed[0].pcell_cqi, 15.0); // Clipped to max
    }
    
    #[test]
    fn test_data_statistics() {
        let temp_dir = tempdir().unwrap();
        let pipeline = DataPipeline::new(temp_dir.path().to_path_buf(), 100);
        
        let mut metrics = vec![
            UEMetrics::new("ue1".to_string()),
            UEMetrics::new("ue2".to_string()),
            UEMetrics::new("ue1".to_string()), // Duplicate UE
        ];
        
        metrics[0].pcell_throughput_mbps = 50.0;
        metrics[1].pcell_throughput_mbps = 100.0;
        metrics[2].pcell_throughput_mbps = 75.0;
        
        let stats = pipeline.get_data_statistics(&metrics);
        
        assert_eq!(stats.total_records, 3);
        assert_eq!(stats.unique_ues, 2); // ue1 and ue2
        assert_eq!(stats.throughput_stats.mean, 75.0);
        assert_eq!(stats.throughput_stats.min, 50.0);
        assert_eq!(stats.throughput_stats.max, 100.0);
    }
}