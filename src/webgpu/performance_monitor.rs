//! WebGPU Performance Monitoring System
//! Real-time performance tracking and optimization for neural network operations

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use crate::webgpu::shaders::webgpu_shaders::ShaderType;

#[cfg(not(feature = "gpu"))]
use crate::webgpu::pipeline_cache::ShaderType;

use crate::webgpu::kernel_optimizer::OptimizationMetrics;

/// Performance measurement for a single GPU operation
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Shader type used
    pub shader_type: ShaderType,

    /// Data size processed
    pub data_size: usize,

    /// Actual execution time
    pub execution_time: Duration,

    /// Memory bandwidth achieved (GB/s)
    pub memory_bandwidth_gbps: f32,

    /// Compute throughput achieved (GFLOPS)
    pub compute_throughput_gflops: f32,

    /// GPU occupancy (0.0 to 1.0)
    pub gpu_occupancy: f32,

    /// Memory efficiency (0.0 to 1.0)
    pub memory_efficiency: f32,

    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Performance statistics aggregated over multiple runs
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Number of measurements
    pub sample_count: usize,

    /// Average execution time
    pub avg_execution_time: Duration,

    /// Minimum execution time (best case)
    pub min_execution_time: Duration,

    /// Maximum execution time (worst case)
    pub max_execution_time: Duration,

    /// Standard deviation of execution times
    pub execution_time_stddev: Duration,

    /// Average memory bandwidth
    pub avg_memory_bandwidth_gbps: f32,

    /// Average compute throughput
    pub avg_compute_throughput_gflops: f32,

    /// Average GPU occupancy
    pub avg_gpu_occupancy: f32,

    /// Performance trend (improving, stable, degrading)
    pub trend: PerformanceTrend,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Insufficient, // Not enough data
}

/// Real-time performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Raw performance measurements
    measurements: HashMap<(ShaderType, usize), Vec<PerformanceMeasurement>>,

    /// Aggregated statistics
    stats_cache: HashMap<(ShaderType, usize), PerformanceStats>,

    /// Maximum number of measurements to keep per operation
    max_measurements_per_operation: usize,

    /// Minimum samples required for reliable statistics
    min_samples_for_stats: usize,

    /// Performance alert thresholds
    alert_thresholds: AlertThresholds,

    /// Real-time metrics
    real_time_metrics: RealTimeMetrics,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum acceptable execution time degradation (ratio)
    max_execution_time_degradation: f32,

    /// Minimum acceptable memory bandwidth utilization
    min_memory_bandwidth_utilization: f32,

    /// Minimum acceptable GPU occupancy
    min_gpu_occupancy: f32,

    /// Maximum acceptable execution time variance
    max_execution_time_variance: f32,
}

#[derive(Debug, Default)]
pub struct RealTimeMetrics {
    /// Total operations executed
    pub total_operations: u64,

    /// Total GPU time used
    pub total_gpu_time: Duration,

    /// Current operations per second
    pub current_ops_per_second: f32,

    /// Average memory bandwidth over last 100 operations
    pub rolling_avg_memory_bandwidth: f32,

    /// Average compute throughput over last 100 operations
    pub rolling_avg_compute_throughput: f32,

    /// GPU utilization percentage
    pub gpu_utilization_percent: f32,
}

#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub shader_type: ShaderType,
    pub data_size: usize,
    pub message: String,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    PerformanceDegradation,
    LowMemoryBandwidth,
    LowGpuOccupancy,
    HighExecutionVariance,
    UnexpectedSlowdown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl PerformanceMonitor {
    /// Create a new performance monitor with default settings
    pub fn new() -> Self {
        Self {
            measurements: HashMap::new(),
            stats_cache: HashMap::new(),
            max_measurements_per_operation: 100,
            min_samples_for_stats: 5,
            alert_thresholds: AlertThresholds::default(),
            real_time_metrics: RealTimeMetrics::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        max_measurements: usize,
        min_samples: usize,
        thresholds: AlertThresholds,
    ) -> Self {
        Self {
            measurements: HashMap::new(),
            stats_cache: HashMap::new(),
            max_measurements_per_operation: max_measurements,
            min_samples_for_stats: min_samples,
            alert_thresholds: thresholds,
            real_time_metrics: RealTimeMetrics::default(),
        }
    }

    /// Record a performance measurement
    pub fn record_measurement(
        &mut self,
        measurement: PerformanceMeasurement,
    ) -> Vec<PerformanceAlert> {
        let key = (measurement.shader_type.clone(), measurement.data_size);
        let mut alerts = Vec::new();

        // Add measurement to history
        let measurements = self.measurements.entry(key.clone()).or_default();
        measurements.push(measurement.clone());

        // Limit history size
        if measurements.len() > self.max_measurements_per_operation {
            measurements.remove(0);
        }

        // Invalidate stats cache for this operation
        self.stats_cache.remove(&key);

        // Check for performance alerts
        let measurements_len = measurements.len();
        if measurements_len >= self.min_samples_for_stats {
            alerts.extend(self.check_performance_alerts(&key, &measurement));
        }

        // Update real-time metrics
        self.update_real_time_metrics(&measurement);

        alerts
    }

    /// Get performance statistics for an operation
    pub fn get_stats(
        &mut self,
        shader_type: &ShaderType,
        data_size: usize,
    ) -> Option<PerformanceStats> {
        let key = (shader_type.clone(), data_size);

        // Check cache first
        if let Some(cached_stats) = self.stats_cache.get(&key) {
            return Some(cached_stats.clone());
        }

        // Calculate stats if we have enough measurements
        if let Some(measurements) = self.measurements.get(&key) {
            if measurements.len() >= self.min_samples_for_stats {
                let stats = self.calculate_stats(measurements);
                self.stats_cache.insert(key, stats.clone());
                return Some(stats);
            }
        }

        None
    }

    /// Get real-time performance metrics
    pub fn get_real_time_metrics(&self) -> &RealTimeMetrics {
        &self.real_time_metrics
    }

    /// Get performance prediction for an operation
    pub fn predict_performance(
        &self,
        shader_type: &ShaderType,
        data_size: usize,
    ) -> Option<OptimizationMetrics> {
        let key = (shader_type.clone(), data_size);

        if let Some(measurements) = self.measurements.get(&key) {
            if !measurements.is_empty() {
                let latest = &measurements[measurements.len() - 1];

                return Some(OptimizationMetrics {
                    memory_utilization: latest.memory_efficiency,
                    compute_utilization: latest.gpu_occupancy,
                    occupancy: latest.gpu_occupancy,
                    memory_efficiency: latest.memory_efficiency,
                    estimated_execution_time_us: latest.execution_time.as_micros() as f32,
                });
            }
        }

        None
    }

    /// Get performance trend for an operation
    pub fn get_performance_trend(
        &self,
        shader_type: &ShaderType,
        data_size: usize,
    ) -> PerformanceTrend {
        let key = (shader_type.clone(), data_size);

        if let Some(measurements) = self.measurements.get(&key) {
            if measurements.len() < 3 {
                return PerformanceTrend::Insufficient;
            }

            // Compare recent performance to older performance
            let recent_count = measurements.len().min(5);
            let older_count = measurements.len().min(10) - recent_count;

            if older_count == 0 {
                return PerformanceTrend::Insufficient;
            }

            let recent_avg = measurements
                .iter()
                .rev()
                .take(recent_count)
                .map(|m| m.execution_time.as_nanos() as f64)
                .sum::<f64>()
                / recent_count as f64;

            let older_avg = measurements
                .iter()
                .rev()
                .skip(recent_count)
                .take(older_count)
                .map(|m| m.execution_time.as_nanos() as f64)
                .sum::<f64>()
                / older_count as f64;

            let improvement_ratio = older_avg / recent_avg;

            if improvement_ratio > 1.05 {
                PerformanceTrend::Improving
            } else if improvement_ratio < 0.95 {
                PerformanceTrend::Degrading
            } else {
                PerformanceTrend::Stable
            }
        } else {
            PerformanceTrend::Insufficient
        }
    }

    /// Clear all performance data
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.stats_cache.clear();
        self.real_time_metrics = RealTimeMetrics::default();
    }

    /// Export performance data for analysis
    pub fn export_data(&self) -> HashMap<(ShaderType, usize), Vec<PerformanceMeasurement>> {
        self.measurements.clone()
    }

    // Private helper methods

    fn update_real_time_metrics(&mut self, measurement: &PerformanceMeasurement) {
        self.real_time_metrics.total_operations += 1;
        self.real_time_metrics.total_gpu_time += measurement.execution_time;

        // Update rolling averages (simplified)
        let alpha = 0.1; // Smoothing factor
        self.real_time_metrics.rolling_avg_memory_bandwidth = alpha
            * measurement.memory_bandwidth_gbps
            + (1.0 - alpha) * self.real_time_metrics.rolling_avg_memory_bandwidth;

        self.real_time_metrics.rolling_avg_compute_throughput = alpha
            * measurement.compute_throughput_gflops
            + (1.0 - alpha) * self.real_time_metrics.rolling_avg_compute_throughput;

        // Calculate operations per second (over last second)
        let total_seconds = self.real_time_metrics.total_gpu_time.as_secs_f32();
        if total_seconds > 0.0 {
            self.real_time_metrics.current_ops_per_second =
                self.real_time_metrics.total_operations as f32 / total_seconds;
        }

        // GPU utilization approximation
        self.real_time_metrics.gpu_utilization_percent =
            (measurement.gpu_occupancy * 100.0).min(100.0);
    }

    fn calculate_stats(&self, measurements: &[PerformanceMeasurement]) -> PerformanceStats {
        let count = measurements.len();

        // Calculate execution time statistics
        let execution_times: Vec<u64> = measurements
            .iter()
            .map(|m| m.execution_time.as_nanos() as u64)
            .collect();

        let avg_execution_nanos = execution_times.iter().sum::<u64>() / count as u64;
        let min_execution_nanos = *execution_times.iter().min().unwrap();
        let max_execution_nanos = *execution_times.iter().max().unwrap();

        // Calculate standard deviation
        let variance = execution_times
            .iter()
            .map(|&time| {
                let diff = time as f64 - avg_execution_nanos as f64;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let stddev_nanos = variance.sqrt() as u64;

        // Calculate averages for other metrics
        let avg_memory_bandwidth = measurements
            .iter()
            .map(|m| m.memory_bandwidth_gbps)
            .sum::<f32>()
            / count as f32;

        let avg_compute_throughput = measurements
            .iter()
            .map(|m| m.compute_throughput_gflops)
            .sum::<f32>()
            / count as f32;

        let avg_gpu_occupancy =
            measurements.iter().map(|m| m.gpu_occupancy).sum::<f32>() / count as f32;

        PerformanceStats {
            sample_count: count,
            avg_execution_time: Duration::from_nanos(avg_execution_nanos),
            min_execution_time: Duration::from_nanos(min_execution_nanos),
            max_execution_time: Duration::from_nanos(max_execution_nanos),
            execution_time_stddev: Duration::from_nanos(stddev_nanos),
            avg_memory_bandwidth_gbps: avg_memory_bandwidth,
            avg_compute_throughput_gflops: avg_compute_throughput,
            avg_gpu_occupancy,
            trend: PerformanceTrend::Insufficient, // Will be calculated separately
        }
    }

    fn check_performance_alerts(
        &self,
        key: &(ShaderType, usize),
        measurement: &PerformanceMeasurement,
    ) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();

        if let Some(measurements) = self.measurements.get(key) {
            if measurements.len() >= 2 {
                let previous = &measurements[measurements.len() - 2];

                // Check for execution time degradation
                let time_ratio = measurement.execution_time.as_nanos() as f32
                    / previous.execution_time.as_nanos() as f32;
                if time_ratio > self.alert_thresholds.max_execution_time_degradation {
                    alerts.push(PerformanceAlert {
                        alert_type: AlertType::PerformanceDegradation,
                        severity: if time_ratio > 2.0 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        shader_type: measurement.shader_type.clone(),
                        data_size: measurement.data_size,
                        message: format!(
                            "Execution time increased by {:.1}%",
                            (time_ratio - 1.0) * 100.0
                        ),
                        timestamp: measurement.timestamp,
                    });
                }

                // Check memory bandwidth
                if measurement.memory_bandwidth_gbps
                    < self.alert_thresholds.min_memory_bandwidth_utilization
                {
                    alerts.push(PerformanceAlert {
                        alert_type: AlertType::LowMemoryBandwidth,
                        severity: AlertSeverity::Info,
                        shader_type: measurement.shader_type.clone(),
                        data_size: measurement.data_size,
                        message: format!(
                            "Low memory bandwidth: {:.1} GB/s",
                            measurement.memory_bandwidth_gbps
                        ),
                        timestamp: measurement.timestamp,
                    });
                }

                // Check GPU occupancy
                if measurement.gpu_occupancy < self.alert_thresholds.min_gpu_occupancy {
                    alerts.push(PerformanceAlert {
                        alert_type: AlertType::LowGpuOccupancy,
                        severity: AlertSeverity::Warning,
                        shader_type: measurement.shader_type.clone(),
                        data_size: measurement.data_size,
                        message: format!(
                            "Low GPU occupancy: {:.1}%",
                            measurement.gpu_occupancy * 100.0
                        ),
                        timestamp: measurement.timestamp,
                    });
                }
            }
        }

        alerts
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_execution_time_degradation: 1.2,     // 20% degradation
            min_memory_bandwidth_utilization: 100.0, // 100 GB/s
            min_gpu_occupancy: 0.5,                  // 50%
            max_execution_time_variance: 0.3,        // 30%
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert_eq!(monitor.max_measurements_per_operation, 100);
        assert_eq!(monitor.min_samples_for_stats, 5);
    }

    #[test]
    fn test_measurement_recording() {
        let mut monitor = PerformanceMonitor::new();
        let measurement = PerformanceMeasurement {
            shader_type: ShaderType::MatrixVectorMultiply,
            data_size: 1024,
            execution_time: Duration::from_millis(5),
            memory_bandwidth_gbps: 500.0,
            compute_throughput_gflops: 1000.0,
            gpu_occupancy: 0.8,
            memory_efficiency: 0.9,
            timestamp: Instant::now(),
        };

        let alerts = monitor.record_measurement(measurement);
        assert!(alerts.is_empty()); // No alerts on first measurement

        assert_eq!(monitor.real_time_metrics.total_operations, 1);
    }

    #[test]
    fn test_performance_stats_calculation() {
        let mut monitor = PerformanceMonitor::new();

        // Record multiple measurements
        for i in 0..10 {
            let measurement = PerformanceMeasurement {
                shader_type: ShaderType::ActivationReLU,
                data_size: 512,
                execution_time: Duration::from_millis(i + 1),
                memory_bandwidth_gbps: 400.0 + i as f32 * 10.0,
                compute_throughput_gflops: 800.0,
                gpu_occupancy: 0.7,
                memory_efficiency: 0.8,
                timestamp: Instant::now(),
            };
            monitor.record_measurement(measurement);
        }

        let stats = monitor.get_stats(&ShaderType::ActivationReLU, 512);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.sample_count, 10);
        assert!(stats.avg_execution_time.as_millis() > 0);
    }
}
