use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, instrument};

/// Statistics collector for feature engineering operations
#[derive(Debug)]
pub struct StatsCollector {
    start_time: Instant,
    total_rows_processed: AtomicU64,
    total_processing_time_ms: AtomicU64,
    total_operations: AtomicU64,
    feature_stats: Arc<RwLock<HashMap<String, FeatureStats>>>,
    error_counts: Arc<RwLock<HashMap<String, u64>>>,
    memory_usage_history: Arc<RwLock<Vec<MemoryUsageSample>>>,
}

impl StatsCollector {
    /// Create a new statistics collector
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_rows_processed: AtomicU64::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            total_operations: AtomicU64::new(0),
            feature_stats: Arc::new(RwLock::new(HashMap::new())),
            error_counts: Arc::new(RwLock::new(HashMap::new())),
            memory_usage_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record processing time
    #[instrument(skip(self))]
    pub async fn record_processing_time(&self, duration: Duration) {
        let millis = duration.as_millis() as u64;
        self.total_processing_time_ms.fetch_add(millis, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        
        debug!("Recorded processing time: {}ms", millis);
    }

    /// Record rows processed
    #[instrument(skip(self))]
    pub async fn record_rows_processed(&self, rows: usize) {
        self.total_rows_processed.fetch_add(rows as u64, Ordering::Relaxed);
        debug!("Recorded {} rows processed", rows);
    }

    /// Record feature statistics
    #[instrument(skip(self, stats))]
    pub async fn record_feature_stats(&self, feature_name: String, stats: FeatureStats) {
        let mut feature_stats = self.feature_stats.write().await;
        feature_stats.insert(feature_name.clone(), stats);
        debug!("Recorded feature stats for: {}", feature_name);
    }

    /// Record error
    #[instrument(skip(self))]
    pub async fn record_error(&self, error_type: String) {
        let mut error_counts = self.error_counts.write().await;
        *error_counts.entry(error_type.clone()).or_insert(0) += 1;
        debug!("Recorded error: {}", error_type);
    }

    /// Record memory usage
    #[instrument(skip(self))]
    pub async fn record_memory_usage(&self, memory_mb: u64) {
        let mut memory_history = self.memory_usage_history.write().await;
        memory_history.push(MemoryUsageSample {
            timestamp: chrono::Utc::now(),
            memory_mb,
        });
        
        // Keep only last 1000 samples
        if memory_history.len() > 1000 {
            memory_history.drain(0..memory_history.len() - 1000);
        }
        
        debug!("Recorded memory usage: {}MB", memory_mb);
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> ProcessingStatistics {
        let total_operations = self.total_operations.load(Ordering::Relaxed);
        let total_processing_time = self.total_processing_time_ms.load(Ordering::Relaxed);
        let average_processing_time = if total_operations > 0 {
            total_processing_time as f64 / total_operations as f64
        } else {
            0.0
        };

        let feature_stats = self.feature_stats.read().await.clone();
        let error_counts = self.error_counts.read().await.clone();
        let memory_history = self.memory_usage_history.read().await.clone();
        
        let current_memory_mb = memory_history.last().map(|m| m.memory_mb).unwrap_or(0);
        let peak_memory_mb = memory_history.iter().map(|m| m.memory_mb).max().unwrap_or(0);

        ProcessingStatistics {
            uptime_seconds: self.start_time.elapsed().as_secs(),
            total_rows_processed: self.total_rows_processed.load(Ordering::Relaxed),
            total_processing_time_ms: total_processing_time,
            total_operations,
            average_processing_time_ms: average_processing_time,
            current_memory_mb,
            peak_memory_mb,
            feature_stats,
            error_counts,
            throughput_rows_per_second: self.calculate_throughput().await,
            memory_usage_history,
        }
    }

    /// Reset all statistics
    pub async fn reset_statistics(&self) {
        self.total_rows_processed.store(0, Ordering::Relaxed);
        self.total_processing_time_ms.store(0, Ordering::Relaxed);
        self.total_operations.store(0, Ordering::Relaxed);
        
        let mut feature_stats = self.feature_stats.write().await;
        feature_stats.clear();
        
        let mut error_counts = self.error_counts.write().await;
        error_counts.clear();
        
        let mut memory_history = self.memory_usage_history.write().await;
        memory_history.clear();
        
        debug!("Reset all statistics");
    }

    /// Calculate current throughput
    async fn calculate_throughput(&self) -> f64 {
        let uptime_seconds = self.start_time.elapsed().as_secs();
        let total_rows = self.total_rows_processed.load(Ordering::Relaxed);
        
        if uptime_seconds > 0 {
            total_rows as f64 / uptime_seconds as f64
        } else {
            0.0
        }
    }

    /// Get feature statistics for a specific feature
    pub async fn get_feature_stats(&self, feature_name: &str) -> Option<FeatureStats> {
        let feature_stats = self.feature_stats.read().await;
        feature_stats.get(feature_name).cloned()
    }

    /// Get error counts
    pub async fn get_error_counts(&self) -> HashMap<String, u64> {
        self.error_counts.read().await.clone()
    }

    /// Get memory usage history
    pub async fn get_memory_usage_history(&self) -> Vec<MemoryUsageSample> {
        self.memory_usage_history.read().await.clone()
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Overall processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    pub uptime_seconds: u64,
    pub total_rows_processed: u64,
    pub total_processing_time_ms: u64,
    pub total_operations: u64,
    pub average_processing_time_ms: f64,
    pub current_memory_mb: u64,
    pub peak_memory_mb: u64,
    pub feature_stats: HashMap<String, FeatureStats>,
    pub error_counts: HashMap<String, u64>,
    pub throughput_rows_per_second: f64,
    pub memory_usage_history: Vec<MemoryUsageSample>,
}

/// Statistics for a specific feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_name: String,
    pub generation_count: u64,
    pub total_processing_time_ms: u64,
    pub average_processing_time_ms: f64,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub mean_value: Option<f64>,
    pub std_dev: Option<f64>,
    pub null_count: u64,
    pub total_count: u64,
    pub percentiles: Vec<f64>, // [p25, p50, p75, p90, p95, p99]
    pub last_generated: chrono::DateTime<chrono::Utc>,
}

impl FeatureStats {
    /// Create new feature statistics
    pub fn new(feature_name: String) -> Self {
        Self {
            feature_name,
            generation_count: 0,
            total_processing_time_ms: 0,
            average_processing_time_ms: 0.0,
            min_value: None,
            max_value: None,
            mean_value: None,
            std_dev: None,
            null_count: 0,
            total_count: 0,
            percentiles: Vec::new(),
            last_generated: chrono::Utc::now(),
        }
    }

    /// Update statistics with new data
    pub fn update(&mut self, values: &[f64], processing_time_ms: u64) {
        self.generation_count += 1;
        self.total_processing_time_ms += processing_time_ms;
        self.average_processing_time_ms = 
            self.total_processing_time_ms as f64 / self.generation_count as f64;
        self.last_generated = chrono::Utc::now();
        
        if !values.is_empty() {
            let mut sorted_values = values.to_vec();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // Filter out NaN values
            let valid_values: Vec<f64> = sorted_values.into_iter()
                .filter(|x| !x.is_nan())
                .collect();
            
            self.total_count += values.len() as u64;
            self.null_count += (values.len() - valid_values.len()) as u64;
            
            if !valid_values.is_empty() {
                self.min_value = Some(valid_values[0]);
                self.max_value = Some(valid_values[valid_values.len() - 1]);
                
                // Calculate mean
                let sum: f64 = valid_values.iter().sum();
                self.mean_value = Some(sum / valid_values.len() as f64);
                
                // Calculate standard deviation
                if let Some(mean) = self.mean_value {
                    let variance = valid_values.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / valid_values.len() as f64;
                    self.std_dev = Some(variance.sqrt());
                }
                
                // Calculate percentiles
                self.percentiles = self.calculate_percentiles(&valid_values);
            }
        }
    }

    /// Calculate percentiles
    fn calculate_percentiles(&self, sorted_values: &[f64]) -> Vec<f64> {
        let percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
        let mut results = Vec::new();
        
        for &p in &percentiles {
            let index = ((sorted_values.len() - 1) as f64 * p) as usize;
            results.push(sorted_values[index]);
        }
        
        results
    }
}

/// Memory usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageSample {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memory_mb: u64,
}

/// Performance metrics calculator
#[derive(Debug)]
pub struct PerformanceMetrics {
    stats_collector: Arc<StatsCollector>,
}

impl PerformanceMetrics {
    /// Create new performance metrics calculator
    pub fn new(stats_collector: Arc<StatsCollector>) -> Self {
        Self { stats_collector }
    }

    /// Calculate efficiency metrics
    pub async fn calculate_efficiency_metrics(&self) -> EfficiencyMetrics {
        let stats = self.stats_collector.get_statistics().await;
        
        let rows_per_ms = if stats.total_processing_time_ms > 0 {
            stats.total_rows_processed as f64 / stats.total_processing_time_ms as f64
        } else {
            0.0
        };
        
        let operations_per_second = if stats.uptime_seconds > 0 {
            stats.total_operations as f64 / stats.uptime_seconds as f64
        } else {
            0.0
        };
        
        let memory_efficiency = if stats.peak_memory_mb > 0 {
            stats.total_rows_processed as f64 / stats.peak_memory_mb as f64
        } else {
            0.0
        };
        
        let error_rate = if stats.total_operations > 0 {
            let total_errors: u64 = stats.error_counts.values().sum();
            total_errors as f64 / stats.total_operations as f64
        } else {
            0.0
        };

        EfficiencyMetrics {
            rows_per_millisecond: rows_per_ms,
            operations_per_second,
            memory_efficiency_rows_per_mb: memory_efficiency,
            error_rate,
            cpu_efficiency: self.calculate_cpu_efficiency(&stats).await,
            feature_generation_rate: self.calculate_feature_generation_rate(&stats).await,
        }
    }

    /// Calculate CPU efficiency
    async fn calculate_cpu_efficiency(&self, stats: &ProcessingStatistics) -> f64 {
        // Simple CPU efficiency metric based on processing time vs wall time
        let wall_time_ms = stats.uptime_seconds * 1000;
        if wall_time_ms > 0 {
            stats.total_processing_time_ms as f64 / wall_time_ms as f64
        } else {
            0.0
        }
    }

    /// Calculate feature generation rate
    async fn calculate_feature_generation_rate(&self, stats: &ProcessingStatistics) -> f64 {
        let total_features: u64 = stats.feature_stats.values()
            .map(|fs| fs.generation_count)
            .sum();
        
        if stats.uptime_seconds > 0 {
            total_features as f64 / stats.uptime_seconds as f64
        } else {
            0.0
        }
    }
}

/// Efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub rows_per_millisecond: f64,
    pub operations_per_second: f64,
    pub memory_efficiency_rows_per_mb: f64,
    pub error_rate: f64,
    pub cpu_efficiency: f64,
    pub feature_generation_rate: f64,
}

/// Real-time monitoring
#[derive(Debug)]
pub struct RealTimeMonitor {
    stats_collector: Arc<StatsCollector>,
    alert_thresholds: AlertThresholds,
}

impl RealTimeMonitor {
    /// Create new real-time monitor
    pub fn new(stats_collector: Arc<StatsCollector>, alert_thresholds: AlertThresholds) -> Self {
        Self {
            stats_collector,
            alert_thresholds,
        }
    }

    /// Check for alerts
    pub async fn check_alerts(&self) -> Vec<Alert> {
        let stats = self.stats_collector.get_statistics().await;
        let mut alerts = Vec::new();

        // Memory alert
        if stats.current_memory_mb > self.alert_thresholds.max_memory_mb {
            alerts.push(Alert {
                alert_type: AlertType::HighMemoryUsage,
                message: format!(
                    "Memory usage {}MB exceeds threshold {}MB",
                    stats.current_memory_mb,
                    self.alert_thresholds.max_memory_mb
                ),
                severity: AlertSeverity::Warning,
                timestamp: chrono::Utc::now(),
            });
        }

        // Error rate alert
        let total_errors: u64 = stats.error_counts.values().sum();
        let error_rate = if stats.total_operations > 0 {
            total_errors as f64 / stats.total_operations as f64
        } else {
            0.0
        };

        if error_rate > self.alert_thresholds.max_error_rate {
            alerts.push(Alert {
                alert_type: AlertType::HighErrorRate,
                message: format!(
                    "Error rate {:.2}% exceeds threshold {:.2}%",
                    error_rate * 100.0,
                    self.alert_thresholds.max_error_rate * 100.0
                ),
                severity: AlertSeverity::Critical,
                timestamp: chrono::Utc::now(),
            });
        }

        // Processing time alert
        if stats.average_processing_time_ms > self.alert_thresholds.max_processing_time_ms {
            alerts.push(Alert {
                alert_type: AlertType::SlowProcessing,
                message: format!(
                    "Average processing time {:.2}ms exceeds threshold {}ms",
                    stats.average_processing_time_ms,
                    self.alert_thresholds.max_processing_time_ms
                ),
                severity: AlertSeverity::Warning,
                timestamp: chrono::Utc::now(),
            });
        }

        alerts
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub max_memory_mb: u64,
    pub max_error_rate: f64,
    pub max_processing_time_ms: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048,
            max_error_rate: 0.05, // 5%
            max_processing_time_ms: 1000.0,
        }
    }
}

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighMemoryUsage,
    HighErrorRate,
    SlowProcessing,
    ServiceDown,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_stats_collector() {
        let collector = StatsCollector::new();
        
        // Record some data
        collector.record_processing_time(Duration::from_millis(100)).await;
        collector.record_rows_processed(1000).await;
        collector.record_memory_usage(512).await;
        
        let stats = collector.get_statistics().await;
        
        assert_eq!(stats.total_rows_processed, 1000);
        assert_eq!(stats.total_processing_time_ms, 100);
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.current_memory_mb, 512);
        assert_eq!(stats.peak_memory_mb, 512);
    }

    #[tokio::test]
    async fn test_feature_stats() {
        let mut stats = FeatureStats::new("test_feature".to_string());
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        stats.update(&values, 50);
        
        assert_eq!(stats.generation_count, 1);
        assert_eq!(stats.total_processing_time_ms, 50);
        assert_eq!(stats.average_processing_time_ms, 50.0);
        assert_eq!(stats.min_value, Some(1.0));
        assert_eq!(stats.max_value, Some(5.0));
        assert_eq!(stats.mean_value, Some(3.0));
        assert_eq!(stats.total_count, 5);
        assert_eq!(stats.null_count, 0);
    }

    #[tokio::test]
    async fn test_real_time_monitor() {
        let collector = Arc::new(StatsCollector::new());
        let thresholds = AlertThresholds {
            max_memory_mb: 100,
            max_error_rate: 0.1,
            max_processing_time_ms: 50.0,
        };
        
        let monitor = RealTimeMonitor::new(collector.clone(), thresholds);
        
        // Trigger memory alert
        collector.record_memory_usage(200).await;
        
        let alerts = monitor.check_alerts().await;
        assert_eq!(alerts.len(), 1);
        assert!(matches!(alerts[0].alert_type, AlertType::HighMemoryUsage));
    }
}