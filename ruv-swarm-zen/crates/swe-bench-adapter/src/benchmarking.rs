//! Performance benchmarking API for SWE-Bench evaluation

use crate::{ExecutionResult, SWEBenchInstance};
use anyhow::Result;
use metrics::{counter, gauge, histogram};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Performance benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    metrics_registry: Arc<RwLock<MetricsRegistry>>,
}

use std::sync::Arc;

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: crate::BenchmarkConfig) -> Self {
        Self {
            config: BenchmarkConfig::from(config),
            metrics_registry: Arc::new(RwLock::new(MetricsRegistry::new())),
        }
    }

    /// Run performance benchmarks on execution results
    pub async fn run_benchmark(
        &self,
        instance: &SWEBenchInstance,
        execution: &ExecutionResult,
    ) -> Result<BenchmarkMetrics> {
        info!("Running benchmarks for instance: {}", instance.instance_id);

        let mut metrics = BenchmarkMetrics {
            instance_id: instance.instance_id.clone(),
            execution_time: execution.duration,
            iterations: self.config.iterations,
            measurements: HashMap::new(),
            memory_usage: None,
            profile_data: None,
        };

        // Warm-up phase
        if self.config.warm_up > 0 {
            debug!("Running {} warm-up iterations", self.config.warm_up);
            for _ in 0..self.config.warm_up {
                self.simulate_execution(execution).await?;
            }
        }

        // Benchmark iterations
        let mut timings = Vec::new();
        for i in 0..self.config.iterations {
            debug!(
                "Running benchmark iteration {}/{}",
                i + 1,
                self.config.iterations
            );
            let start = Instant::now();

            self.simulate_execution(execution).await?;

            let duration = start.elapsed();
            timings.push(duration);

            // Record metrics
            histogram!("swe_bench_execution_time", duration.as_secs_f64());
            counter!("swe_bench_iterations", 1);
        }

        // Calculate statistics
        let stats = self.calculate_statistics(&timings);
        metrics
            .measurements
            .insert("execution_time".to_string(), stats);

        // Memory benchmarking
        if self.config.measure_memory {
            let memory_stats = self.measure_memory_usage(execution).await?;
            metrics.memory_usage = Some(memory_stats);
        }

        // Profiling
        if self.config.profile_enabled {
            let profile = self.generate_profile(execution).await?;
            metrics.profile_data = Some(profile);
        }

        // Update registry
        {
            let mut registry = self.metrics_registry.write().await;
            registry.record_benchmark(&metrics);
        }

        info!("Benchmark completed for instance: {}", instance.instance_id);
        Ok(metrics)
    }

    /// Run comparative benchmarks between multiple executions
    pub async fn run_comparison(
        &self,
        baseline: &ExecutionResult,
        candidate: &ExecutionResult,
        instance: &SWEBenchInstance,
    ) -> Result<ComparisonReport> {
        info!(
            "Running comparative benchmark for instance: {}",
            instance.instance_id
        );

        let baseline_metrics = self.run_benchmark(instance, baseline).await?;
        let candidate_metrics = self.run_benchmark(instance, candidate).await?;

        let speedup = baseline_metrics.execution_time.as_secs_f64()
            / candidate_metrics.execution_time.as_secs_f64();

        let memory_comparison = if let (Some(base_mem), Some(cand_mem)) = (
            &baseline_metrics.memory_usage,
            &candidate_metrics.memory_usage,
        ) {
            Some(MemoryComparison {
                baseline_peak: base_mem.peak_usage,
                candidate_peak: cand_mem.peak_usage,
                improvement: (base_mem.peak_usage as f64 - cand_mem.peak_usage as f64)
                    / base_mem.peak_usage as f64,
            })
        } else {
            None
        };

        Ok(ComparisonReport {
            instance_id: instance.instance_id.clone(),
            baseline_metrics,
            candidate_metrics,
            speedup,
            memory_comparison,
            recommendation: self.generate_recommendation(speedup),
        })
    }

    /// Simulate execution for benchmarking
    async fn simulate_execution(&self, _execution: &ExecutionResult) -> Result<()> {
        // Simulate the execution workload
        // In real implementation, this would replay the actual execution
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    /// Calculate statistics from timing measurements
    fn calculate_statistics(&self, timings: &[Duration]) -> MeasurementStats {
        let total: Duration = timings.iter().sum();
        let mean = total / timings.len() as u32;

        let variance = timings
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / timings.len() as f64;

        let std_dev = variance.sqrt();

        let mut sorted = timings.to_vec();
        sorted.sort();

        let median = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
        let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

        MeasurementStats {
            mean,
            median,
            std_dev: Duration::from_secs_f64(std_dev),
            min: *sorted.first().unwrap(),
            max: *sorted.last().unwrap(),
            p95,
            p99,
        }
    }

    /// Measure memory usage during execution
    async fn measure_memory_usage(&self, _execution: &ExecutionResult) -> Result<MemoryStats> {
        // In a real implementation, this would use system calls or profiling tools
        // For now, return mock data
        Ok(MemoryStats {
            peak_usage: 1024 * 1024 * 100, // 100MB
            avg_usage: 1024 * 1024 * 80,   // 80MB
            allocations: 10000,
            deallocations: 9500,
            gc_cycles: 5,
        })
    }

    /// Generate execution profile
    async fn generate_profile(&self, _execution: &ExecutionResult) -> Result<ProfileData> {
        // In a real implementation, this would use profiling tools
        // For now, return mock profile data
        Ok(ProfileData {
            hot_functions: vec![
                HotFunction {
                    name: "parse_code".to_string(),
                    self_time: Duration::from_millis(100),
                    total_time: Duration::from_millis(150),
                    call_count: 1000,
                },
                HotFunction {
                    name: "apply_patch".to_string(),
                    self_time: Duration::from_millis(200),
                    total_time: Duration::from_millis(250),
                    call_count: 1,
                },
            ],
            call_graph: HashMap::new(),
        })
    }

    /// Generate recommendation based on speedup
    fn generate_recommendation(&self, speedup: f64) -> String {
        match speedup {
            s if s > 1.5 => {
                "Significant performance improvement. Candidate is recommended.".to_string()
            }
            s if s > 1.1 => {
                "Moderate performance improvement. Candidate shows promise.".to_string()
            }
            s if s > 0.9 => {
                "Comparable performance. Decision should be based on other factors.".to_string()
            }
            _ => "Performance regression detected. Baseline is recommended.".to_string(),
        }
    }

    /// Get aggregated benchmark statistics
    pub async fn get_statistics(&self) -> BenchmarkStatistics {
        let registry = self.metrics_registry.read().await;
        registry.get_statistics()
    }
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    iterations: usize,
    warm_up: usize,
    measure_memory: bool,
    profile_enabled: bool,
}

impl From<crate::BenchmarkConfig> for BenchmarkConfig {
    fn from(config: crate::BenchmarkConfig) -> Self {
        Self {
            iterations: config.iterations,
            warm_up: config.warm_up,
            measure_memory: config.measure_memory,
            profile_enabled: config.profile_enabled,
        }
    }
}

/// Benchmark metrics for a single run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub instance_id: String,
    pub execution_time: Duration,
    pub iterations: usize,
    pub measurements: HashMap<String, MeasurementStats>,
    pub memory_usage: Option<MemoryStats>,
    pub profile_data: Option<ProfileData>,
}

/// Statistics for a measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementStats {
    pub mean: Duration,
    pub median: Duration,
    pub std_dev: Duration,
    pub min: Duration,
    pub max: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub peak_usage: usize,
    pub avg_usage: usize,
    pub allocations: usize,
    pub deallocations: usize,
    pub gc_cycles: usize,
}

/// Profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileData {
    pub hot_functions: Vec<HotFunction>,
    pub call_graph: HashMap<String, Vec<String>>,
}

/// Hot function in profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotFunction {
    pub name: String,
    pub self_time: Duration,
    pub total_time: Duration,
    pub call_count: usize,
}

/// Comparison report between two executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub instance_id: String,
    pub baseline_metrics: BenchmarkMetrics,
    pub candidate_metrics: BenchmarkMetrics,
    pub speedup: f64,
    pub memory_comparison: Option<MemoryComparison>,
    pub recommendation: String,
}

/// Memory comparison data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryComparison {
    pub baseline_peak: usize,
    pub candidate_peak: usize,
    pub improvement: f64,
}

/// Performance report aggregating multiple benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub total_benchmarks: usize,
    pub avg_execution_time: Duration,
    pub fastest_instance: String,
    pub slowest_instance: String,
    pub by_difficulty: HashMap<String, DifficultyPerformance>,
}

/// Performance metrics by difficulty level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyPerformance {
    pub avg_time: Duration,
    pub success_rate: f64,
    pub instances: usize,
}

/// Metrics registry for tracking benchmarks
struct MetricsRegistry {
    benchmarks: Vec<BenchmarkMetrics>,
}

impl MetricsRegistry {
    fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    fn record_benchmark(&mut self, metrics: &BenchmarkMetrics) {
        self.benchmarks.push(metrics.clone());

        // Emit Prometheus metrics
        gauge!("swe_bench_total_benchmarks", self.benchmarks.len() as f64);
    }

    fn get_statistics(&self) -> BenchmarkStatistics {
        let total = self.benchmarks.len();
        let avg_time = if total > 0 {
            let sum: Duration = self.benchmarks.iter().map(|b| b.execution_time).sum();
            sum / total as u32
        } else {
            Duration::default()
        };

        BenchmarkStatistics {
            total_benchmarks: total,
            average_execution_time: avg_time,
            total_iterations: self.benchmarks.iter().map(|b| b.iterations).sum(),
        }
    }
}

/// Aggregate benchmark statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    pub total_benchmarks: usize,
    pub average_execution_time: Duration,
    pub total_iterations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_calculation() {
        let config = crate::BenchmarkConfig::default();
        let runner = BenchmarkRunner::new(config);

        let timings = vec![
            Duration::from_millis(100),
            Duration::from_millis(110),
            Duration::from_millis(120),
            Duration::from_millis(130),
            Duration::from_millis(140),
        ];

        let stats = runner.calculate_statistics(&timings);
        assert_eq!(stats.mean, Duration::from_millis(120));
        assert_eq!(stats.median, Duration::from_millis(120));
        assert_eq!(stats.min, Duration::from_millis(100));
        assert_eq!(stats.max, Duration::from_millis(140));
    }

    #[test]
    fn test_recommendation_generation() {
        let config = crate::BenchmarkConfig::default();
        let runner = BenchmarkRunner::new(config);

        assert!(runner.generate_recommendation(2.0).contains("Significant"));
        assert!(runner.generate_recommendation(1.2).contains("Moderate"));
        assert!(runner.generate_recommendation(1.0).contains("Comparable"));
        assert!(runner.generate_recommendation(0.5).contains("regression"));
    }
}
