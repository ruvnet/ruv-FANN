//! Monitoring and metrics collection for the ingestion service

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::config::IngestionConfig;
use crate::error::ErrorStats;

/// Ingestion monitoring and metrics collection
pub struct IngestionMonitor {
    config: IngestionConfig,
    metrics: Arc<RwLock<IngestionMetrics>>,
    job_metrics: Arc<RwLock<HashMap<String, JobMetrics>>>,
    system_metrics: Arc<RwLock<SystemMetrics>>,
    start_time: Instant,
}

impl IngestionMonitor {
    pub fn new(config: IngestionConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(IngestionMetrics::new())),
            job_metrics: Arc::new(RwLock::new(HashMap::new())),
            system_metrics: Arc::new(RwLock::new(SystemMetrics::new())),
            start_time: Instant::now(),
        }
    }
    
    /// Start the monitoring background task
    pub async fn start_monitoring(&self) {
        let metrics = Arc::clone(&self.metrics);
        let system_metrics = Arc::clone(&self.system_metrics);
        let interval_seconds = self.config.metrics_interval_seconds;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_seconds));
            
            loop {
                interval.tick().await;
                
                // Update system metrics
                let mut sys_metrics = system_metrics.write().await;
                sys_metrics.update().await;
                
                // Log current metrics
                let current_metrics = metrics.read().await;
                debug!("Current metrics: {:?}", *current_metrics);
            }
        });
    }
    
    /// Record file processing start
    pub async fn record_file_start(&self, job_id: &str, file_path: &str, file_size: u64) {
        let mut job_metrics = self.job_metrics.write().await;
        let job_metric = job_metrics.entry(job_id.to_string()).or_insert_with(JobMetrics::new);
        
        job_metric.files_in_progress += 1;
        job_metric.current_file = Some(file_path.to_string());
        job_metric.total_input_size += file_size;
        job_metric.last_activity = SystemTime::now();
        
        let mut metrics = self.metrics.write().await;
        metrics.active_files += 1;
        metrics.total_input_size += file_size;
    }
    
    /// Record file processing completion
    pub async fn record_file_completion(
        &self,
        job_id: &str,
        file_path: &str,
        rows_processed: u64,
        rows_failed: u64,
        processing_time: Duration,
        output_size: u64,
    ) {
        let mut job_metrics = self.job_metrics.write().await;
        if let Some(job_metric) = job_metrics.get_mut(job_id) {
            job_metric.files_processed += 1;
            job_metric.files_in_progress = job_metric.files_in_progress.saturating_sub(1);
            job_metric.rows_processed += rows_processed;
            job_metric.rows_failed += rows_failed;
            job_metric.total_processing_time += processing_time;
            job_metric.total_output_size += output_size;
            job_metric.current_file = None;
            job_metric.last_activity = SystemTime::now();
            
            // Update file processing times
            job_metric.file_processing_times.push(processing_time);
            if job_metric.file_processing_times.len() > 100 {
                job_metric.file_processing_times.remove(0);
            }
        }
        
        let mut metrics = self.metrics.write().await;
        metrics.active_files = metrics.active_files.saturating_sub(1);
        metrics.files_processed += 1;
        metrics.rows_processed += rows_processed;
        metrics.rows_failed += rows_failed;
        metrics.total_output_size += output_size;
        metrics.total_processing_time += processing_time;
        
        // Update processing times for throughput calculation
        metrics.processing_times.push(processing_time);
        if metrics.processing_times.len() > 1000 {
            metrics.processing_times.remove(0);
        }
        
        info!(
            "File completed: {} ({} rows processed, {} rows failed) in {:?}",
            file_path, rows_processed, rows_failed, processing_time
        );
    }
    
    /// Record file processing failure
    pub async fn record_file_failure(&self, job_id: &str, file_path: &str, error: &str) {
        let mut job_metrics = self.job_metrics.write().await;
        if let Some(job_metric) = job_metrics.get_mut(job_id) {
            job_metric.files_failed += 1;
            job_metric.files_in_progress = job_metric.files_in_progress.saturating_sub(1);
            job_metric.current_file = None;
            job_metric.last_activity = SystemTime::now();
            job_metric.errors.push(error.to_string());
            
            // Keep only last 50 errors
            if job_metric.errors.len() > 50 {
                job_metric.errors.remove(0);
            }
        }
        
        let mut metrics = self.metrics.write().await;
        metrics.active_files = metrics.active_files.saturating_sub(1);
        metrics.files_failed += 1;
        
        warn!("File processing failed: {} - {}", file_path, error);
    }
    
    /// Record job completion
    pub async fn record_job_completion(&self, job_id: &str, success: bool) {
        let mut job_metrics = self.job_metrics.write().await;
        if let Some(job_metric) = job_metrics.get_mut(job_id) {
            job_metric.completed = true;
            job_metric.success = success;
            job_metric.completion_time = Some(SystemTime::now());
            job_metric.last_activity = SystemTime::now();
        }
        
        let mut metrics = self.metrics.write().await;
        if success {
            metrics.jobs_completed += 1;
        } else {
            metrics.jobs_failed += 1;
        }
        
        info!("Job {} completed with success: {}", job_id, success);
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> IngestionMetrics {
        let metrics = self.metrics.read().await;
        let mut result = metrics.clone();
        
        // Calculate derived metrics
        result.uptime_seconds = self.start_time.elapsed().as_secs();
        result.calculate_derived_metrics();
        
        result
    }
    
    /// Get job-specific metrics
    pub async fn get_job_metrics(&self, job_id: &str) -> Option<JobMetrics> {
        let job_metrics = self.job_metrics.read().await;
        job_metrics.get(job_id).cloned()
    }
    
    /// Get all job metrics
    pub async fn get_all_job_metrics(&self) -> HashMap<String, JobMetrics> {
        self.job_metrics.read().await.clone()
    }
    
    /// Get system metrics
    pub async fn get_system_metrics(&self) -> SystemMetrics {
        self.system_metrics.read().await.clone()
    }
    
    /// Clean up old job metrics
    pub async fn cleanup_old_jobs(&self, retention_hours: u64) {
        let mut job_metrics = self.job_metrics.write().await;
        let cutoff_time = SystemTime::now() - Duration::from_secs(retention_hours * 3600);
        
        let mut to_remove = Vec::new();
        for (job_id, metrics) in job_metrics.iter() {
            if metrics.completed && metrics.last_activity < cutoff_time {
                to_remove.push(job_id.clone());
            }
        }
        
        for job_id in to_remove {
            job_metrics.remove(&job_id);
            debug!("Cleaned up old job metrics: {}", job_id);
        }
    }
    
    /// Get performance summary
    pub async fn get_performance_summary(&self) -> PerformanceSummary {
        let metrics = self.get_metrics().await;
        let system_metrics = self.get_system_metrics().await;
        
        PerformanceSummary {
            throughput_mb_per_second: metrics.throughput_mb_per_second,
            rows_per_second: metrics.rows_per_second,
            average_file_processing_time_ms: metrics.average_file_processing_time_ms,
            error_rate: metrics.error_rate,
            compression_ratio: metrics.compression_ratio,
            cpu_usage_percent: system_metrics.cpu_usage_percent,
            memory_usage_mb: system_metrics.memory_usage_mb,
            active_files: metrics.active_files,
            queue_depth: metrics.active_files,
        }
    }
    
    /// Export metrics to JSON
    pub async fn export_metrics(&self) -> serde_json::Value {
        let metrics = self.get_metrics().await;
        let job_metrics = self.get_all_job_metrics().await;
        let system_metrics = self.get_system_metrics().await;
        
        serde_json::json!({
            "timestamp": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "global_metrics": metrics,
            "job_metrics": job_metrics,
            "system_metrics": system_metrics
        })
    }
}

/// Global ingestion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionMetrics {
    // Counters
    pub files_processed: u64,
    pub files_failed: u64,
    pub active_files: u64,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub rows_processed: u64,
    pub rows_failed: u64,
    
    // Sizes
    pub total_input_size: u64,
    pub total_output_size: u64,
    
    // Timing
    pub total_processing_time: Duration,
    pub uptime_seconds: u64,
    
    // Derived metrics (calculated)
    pub throughput_mb_per_second: f64,
    pub rows_per_second: f64,
    pub average_file_processing_time_ms: f64,
    pub error_rate: f64,
    pub compression_ratio: f64,
    
    // Internal data for calculations
    #[serde(skip)]
    pub processing_times: Vec<Duration>,
}

impl IngestionMetrics {
    pub fn new() -> Self {
        Self {
            files_processed: 0,
            files_failed: 0,
            active_files: 0,
            jobs_completed: 0,
            jobs_failed: 0,
            rows_processed: 0,
            rows_failed: 0,
            total_input_size: 0,
            total_output_size: 0,
            total_processing_time: Duration::from_secs(0),
            uptime_seconds: 0,
            throughput_mb_per_second: 0.0,
            rows_per_second: 0.0,
            average_file_processing_time_ms: 0.0,
            error_rate: 0.0,
            compression_ratio: 0.0,
            processing_times: Vec::new(),
        }
    }
    
    pub fn calculate_derived_metrics(&mut self) {
        // Calculate throughput
        if self.total_processing_time.as_secs_f64() > 0.0 {
            self.throughput_mb_per_second = (self.total_input_size as f64 / 1024.0 / 1024.0) 
                / self.total_processing_time.as_secs_f64();
            self.rows_per_second = self.rows_processed as f64 
                / self.total_processing_time.as_secs_f64();
        }
        
        // Calculate average processing time
        if !self.processing_times.is_empty() {
            let sum_ms: f64 = self.processing_times.iter()
                .map(|d| d.as_millis() as f64)
                .sum();
            self.average_file_processing_time_ms = sum_ms / self.processing_times.len() as f64;
        }
        
        // Calculate error rate
        let total_rows = self.rows_processed + self.rows_failed;
        if total_rows > 0 {
            self.error_rate = self.rows_failed as f64 / total_rows as f64;
        }
        
        // Calculate compression ratio
        if self.total_input_size > 0 {
            self.compression_ratio = self.total_output_size as f64 / self.total_input_size as f64;
        }
    }
}

/// Job-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetrics {
    pub files_processed: u64,
    pub files_failed: u64,
    pub files_in_progress: u64,
    pub rows_processed: u64,
    pub rows_failed: u64,
    pub total_input_size: u64,
    pub total_output_size: u64,
    pub total_processing_time: Duration,
    pub current_file: Option<String>,
    pub completed: bool,
    pub success: bool,
    pub start_time: SystemTime,
    pub completion_time: Option<SystemTime>,
    pub last_activity: SystemTime,
    pub errors: Vec<String>,
    
    #[serde(skip)]
    pub file_processing_times: Vec<Duration>,
}

impl JobMetrics {
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            files_processed: 0,
            files_failed: 0,
            files_in_progress: 0,
            rows_processed: 0,
            rows_failed: 0,
            total_input_size: 0,
            total_output_size: 0,
            total_processing_time: Duration::from_secs(0),
            current_file: None,
            completed: false,
            success: false,
            start_time: now,
            completion_time: None,
            last_activity: now,
            errors: Vec::new(),
            file_processing_times: Vec::new(),
        }
    }
    
    pub fn progress_percentage(&self) -> f64 {
        let total_files = self.files_processed + self.files_failed + self.files_in_progress;
        if total_files == 0 {
            0.0
        } else {
            (self.files_processed as f64 / total_files as f64) * 100.0
        }
    }
    
    pub fn error_rate(&self) -> f64 {
        let total_rows = self.rows_processed + self.rows_failed;
        if total_rows == 0 {
            0.0
        } else {
            self.rows_failed as f64 / total_rows as f64
        }
    }
    
    pub fn average_file_processing_time(&self) -> Duration {
        if self.file_processing_times.is_empty() {
            Duration::from_secs(0)
        } else {
            let total_ms: u64 = self.file_processing_times.iter()
                .map(|d| d.as_millis() as u64)
                .sum();
            Duration::from_millis(total_ms / self.file_processing_times.len() as u64)
        }
    }
    
    pub fn estimated_completion_time(&self) -> Option<SystemTime> {
        if self.completed || self.files_in_progress == 0 {
            return None;
        }
        
        let avg_time = self.average_file_processing_time();
        if avg_time.as_secs() == 0 {
            return None;
        }
        
        let estimated_remaining = avg_time * self.files_in_progress as u32;
        Some(SystemTime::now() + estimated_remaining)
    }
}

/// System resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub disk_usage_mb: f64,
    pub network_bytes_per_second: f64,
    pub open_file_descriptors: u64,
    pub thread_count: u64,
    pub last_updated: SystemTime,
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            disk_usage_mb: 0.0,
            network_bytes_per_second: 0.0,
            open_file_descriptors: 0,
            thread_count: 0,
            last_updated: SystemTime::now(),
        }
    }
    
    pub async fn update(&mut self) {
        // Simple system metrics collection
        // In a real implementation, you would use system monitoring crates
        self.last_updated = SystemTime::now();
        
        // Placeholder values - in real implementation use sysinfo or similar
        self.cpu_usage_percent = self.get_cpu_usage();
        self.memory_usage_mb = self.get_memory_usage();
        self.disk_usage_mb = self.get_disk_usage();
        self.open_file_descriptors = self.get_open_file_descriptors();
        self.thread_count = self.get_thread_count();
    }
    
    fn get_cpu_usage(&self) -> f64 {
        // Placeholder - use sysinfo crate for real implementation
        0.0
    }
    
    fn get_memory_usage(&self) -> f64 {
        // Placeholder - use sysinfo crate for real implementation
        0.0
    }
    
    fn get_disk_usage(&self) -> f64 {
        // Placeholder - use sysinfo crate for real implementation
        0.0
    }
    
    fn get_open_file_descriptors(&self) -> u64 {
        // Placeholder - use system calls for real implementation
        0
    }
    
    fn get_thread_count(&self) -> u64 {
        // Placeholder - use system calls for real implementation
        0
    }
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub throughput_mb_per_second: f64,
    pub rows_per_second: f64,
    pub average_file_processing_time_ms: f64,
    pub error_rate: f64,
    pub compression_ratio: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub active_files: u64,
    pub queue_depth: u64,
}

impl PerformanceSummary {
    pub fn is_healthy(&self) -> bool {
        self.error_rate < 0.01 && // Less than 1% error rate
        self.cpu_usage_percent < 80.0 && // Less than 80% CPU
        self.memory_usage_mb < 8192.0 && // Less than 8GB memory
        self.queue_depth < 100 // Less than 100 files in queue
    }
    
    pub fn performance_grade(&self) -> char {
        let score = self.calculate_performance_score();
        match score {
            90..=100 => 'A',
            80..=89 => 'B', 
            70..=79 => 'C',
            60..=69 => 'D',
            _ => 'F',
        }
    }
    
    fn calculate_performance_score(&self) -> u8 {
        let mut score = 100;
        
        // Deduct points for high error rate
        if self.error_rate > 0.01 {
            score -= 30;
        }
        
        // Deduct points for high CPU usage
        if self.cpu_usage_percent > 80.0 {
            score -= 20;
        }
        
        // Deduct points for low throughput
        if self.throughput_mb_per_second < 10.0 {
            score -= 20;
        }
        
        // Deduct points for high queue depth
        if self.queue_depth > 50 {
            score -= 15;
        }
        
        score.max(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ingestion_monitor_creation() {
        let config = IngestionConfig::default();
        let monitor = IngestionMonitor::new(config);
        
        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.files_processed, 0);
        assert_eq!(metrics.files_failed, 0);
    }
    
    #[tokio::test]
    async fn test_file_processing_metrics() {
        let config = IngestionConfig::default();
        let monitor = IngestionMonitor::new(config);
        
        let job_id = "test_job";
        let file_path = "test.csv";
        
        monitor.record_file_start(job_id, file_path, 1000).await;
        monitor.record_file_completion(
            job_id,
            file_path,
            100,
            5,
            Duration::from_millis(500),
            800,
        ).await;
        
        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.files_processed, 1);
        assert_eq!(metrics.rows_processed, 100);
        assert_eq!(metrics.rows_failed, 5);
        
        let job_metrics = monitor.get_job_metrics(job_id).await.unwrap();
        assert_eq!(job_metrics.files_processed, 1);
        assert_eq!(job_metrics.error_rate(), 0.047619047619047616); // 5/105
    }
    
    #[test]
    fn test_performance_summary_grading() {
        let summary = PerformanceSummary {
            throughput_mb_per_second: 50.0,
            rows_per_second: 10000.0,
            average_file_processing_time_ms: 1000.0,
            error_rate: 0.005,
            compression_ratio: 0.3,
            cpu_usage_percent: 50.0,
            memory_usage_mb: 2048.0,
            active_files: 5,
            queue_depth: 10,
        };
        
        assert!(summary.is_healthy());
        assert_eq!(summary.performance_grade(), 'A');
    }
    
    #[test]
    fn test_job_metrics_progress() {
        let mut job_metrics = JobMetrics::new();
        job_metrics.files_processed = 7;
        job_metrics.files_failed = 2;
        job_metrics.files_in_progress = 1;
        
        assert_eq!(job_metrics.progress_percentage(), 70.0);
    }
}