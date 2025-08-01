//! RUV-SWARM Benchmarking Framework
//!
//! Comprehensive benchmarking system for evaluating Claude Code CLI performance
//! with ML-optimized swarm intelligence against baseline implementations.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::timeout;
use tracing::{error, info, warn};
use uuid::Uuid;

pub mod claude_executor;
pub mod comparator;
pub mod metrics;
pub mod realtime;
pub mod storage;
pub mod stream_parser;
pub mod swe_bench_evaluator;

use comparator::PerformanceComparator;
use metrics::{MetricsCollector, PerformanceMetrics};
use realtime::RealTimeMonitor;
use storage::BenchmarkStorage;
use stream_parser::{ClaudeStreamEvent, StreamMetricsCollector};

/// Main benchmarking framework that orchestrates all components
pub struct BenchmarkingFramework {
    /// Executes Claude Code CLI commands
    pub executor: BenchmarkExecutor,
    /// Collects performance metrics during execution
    pub collector: MetricsCollector,
    /// SQLite storage for benchmark data
    pub storage: BenchmarkStorage,
    /// Compares baseline vs ML-optimized performance
    pub comparator: PerformanceComparator,
    /// Real-time monitoring system
    pub monitor: Option<RealTimeMonitor>,
}

impl BenchmarkingFramework {
    /// Create a new benchmarking framework
    pub async fn new(config: BenchmarkConfig) -> Result<Self> {
        let storage = BenchmarkStorage::new(&config.database_path).await?;

        let monitor = if config.enable_real_time_monitoring {
            Some(RealTimeMonitor::new(config.monitor_port).await?)
        } else {
            None
        };

        Ok(Self {
            executor: BenchmarkExecutor::new(config.clone()),
            collector: MetricsCollector::new(),
            storage,
            comparator: PerformanceComparator::new(),
            monitor,
        })
    }

    /// Execute a complete benchmark suite
    pub async fn run_benchmark_suite(
        &mut self,
        scenarios: Vec<BenchmarkScenario>,
    ) -> Result<BenchmarkReport> {
        info!(
            "Starting benchmark suite with {} scenarios",
            scenarios.len()
        );

        let mut results = Vec::new();

        for scenario in scenarios {
            info!("Executing scenario: {}", scenario.name);

            // Run baseline
            let baseline_result = self
                .run_scenario(&scenario, ExecutionMode::Baseline)
                .await?;

            // Run ML-optimized
            let ml_result = self
                .run_scenario(&scenario, ExecutionMode::MLOptimized)
                .await?;

            // Compare results
            let comparison = self.comparator.compare(&baseline_result, &ml_result)?;

            results.push(BenchmarkResult {
                scenario: scenario.clone(),
                baseline: baseline_result,
                ml_optimized: ml_result,
                comparison,
            });
        }

        let summary = self.generate_summary(&results);

        Ok(BenchmarkReport {
            timestamp: Utc::now(),
            results,
            summary,
        })
    }

    /// Run a single scenario
    async fn run_scenario(
        &mut self,
        scenario: &BenchmarkScenario,
        mode: ExecutionMode,
    ) -> Result<ScenarioResult> {
        let run_id = Uuid::new_v4().to_string();

        // Initialize benchmark run in storage
        self.storage
            .create_benchmark_run(
                &run_id,
                &scenario.instance_id,
                &scenario.repository,
                &scenario.issue_description,
                &scenario.difficulty.to_string(),
                &mode.to_string(),
                &scenario.claude_command,
            )
            .await?;

        // Start monitoring if enabled
        if let Some(monitor) = &self.monitor {
            monitor.start_monitoring(&run_id).await?;
        }

        // Execute the scenario
        let start_time = Instant::now();
        let execution_result = self.executor.execute(scenario, mode.clone(), &run_id).await;
        let duration = start_time.elapsed();

        match execution_result {
            Ok(metrics) => {
                // Store metrics
                self.storage.store_metrics(&run_id, &metrics).await?;

                // Update run status
                self.storage.update_run_status(&run_id, "completed").await?;

                Ok(ScenarioResult {
                    run_id,
                    scenario_name: scenario.name.clone(),
                    mode,
                    duration,
                    metrics,
                    status: ExecutionStatus::Completed,
                    error: None,
                })
            }
            Err(e) => {
                error!("Scenario execution failed: {}", e);

                // Update run status
                self.storage.update_run_status(&run_id, "failed").await?;

                Ok(ScenarioResult {
                    run_id,
                    scenario_name: scenario.name.clone(),
                    mode,
                    duration,
                    metrics: PerformanceMetrics::default(),
                    status: ExecutionStatus::Failed,
                    error: Some(e.to_string()),
                })
            }
        }
    }

    /// Generate summary report
    fn generate_summary(&self, results: &[BenchmarkResult]) -> BenchmarkSummary {
        let total_scenarios = results.len();
        let successful_scenarios = results
            .iter()
            .filter(|r| {
                r.baseline.status == ExecutionStatus::Completed
                    && r.ml_optimized.status == ExecutionStatus::Completed
            })
            .count();

        let avg_improvement = if successful_scenarios > 0 {
            results
                .iter()
                .filter_map(|r| r.comparison.overall_improvement)
                .sum::<f64>()
                / successful_scenarios as f64
        } else {
            0.0
        };

        BenchmarkSummary {
            total_scenarios,
            successful_scenarios,
            failed_scenarios: total_scenarios - successful_scenarios,
            average_improvement: avg_improvement,
            key_findings: self.extract_key_findings(results),
        }
    }

    fn extract_key_findings(&self, results: &[BenchmarkResult]) -> Vec<String> {
        let mut findings = Vec::new();

        // Calculate average metrics
        let avg_speed_improvement = results
            .iter()
            .filter_map(|r| r.comparison.speed_improvement)
            .collect::<Vec<_>>();

        if !avg_speed_improvement.is_empty() {
            let avg =
                avg_speed_improvement.iter().sum::<f64>() / avg_speed_improvement.len() as f64;
            findings.push(format!("Average speed improvement: {:.1}%", avg * 100.0));
        }

        findings
    }
}

/// Configuration for the benchmarking framework
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Path to SQLite database
    pub database_path: PathBuf,
    /// Enable real-time monitoring
    pub enable_real_time_monitoring: bool,
    /// Port for monitoring server
    pub monitor_port: u16,
    /// Claude CLI executable path
    pub claude_executable: PathBuf,
    /// Timeout for command execution
    pub execution_timeout: Duration,
    /// Number of trials per scenario
    pub trial_count: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("benchmarks.db"),
            enable_real_time_monitoring: true,
            monitor_port: 8080,
            claude_executable: PathBuf::from("claude"),
            execution_timeout: Duration::from_secs(1800), // 30 minutes
            trial_count: 3,
        }
    }
}

/// Benchmark scenario definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScenario {
    pub name: String,
    pub instance_id: String,
    pub repository: String,
    pub issue_description: String,
    pub difficulty: Difficulty,
    pub claude_command: String,
    pub expected_files_modified: Vec<String>,
    pub validation_tests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

impl ToString for Difficulty {
    fn to_string(&self) -> String {
        match self {
            Difficulty::Easy => "easy".to_string(),
            Difficulty::Medium => "medium".to_string(),
            Difficulty::Hard => "hard".to_string(),
        }
    }
}

/// Execution mode for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionMode {
    Baseline,
    MLOptimized,
}

impl ToString for ExecutionMode {
    fn to_string(&self) -> String {
        match self {
            ExecutionMode::Baseline => "baseline".to_string(),
            ExecutionMode::MLOptimized => "ml_optimized".to_string(),
        }
    }
}

/// Result of a single scenario execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub run_id: String,
    pub scenario_name: String,
    pub mode: ExecutionMode,
    pub duration: Duration,
    pub metrics: PerformanceMetrics,
    pub status: ExecutionStatus,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Timeout,
}

/// Result of comparing baseline vs ML-optimized
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub scenario: BenchmarkScenario,
    pub baseline: ScenarioResult,
    pub ml_optimized: ScenarioResult,
    pub comparison: ComparisonResult,
}

/// Comparison results between baseline and ML-optimized
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub overall_improvement: Option<f64>,
    pub speed_improvement: Option<f64>,
    pub quality_improvement: Option<f64>,
    pub resource_efficiency: Option<f64>,
    pub statistical_significance: StatisticalSignificance,
}

#[derive(Debug, Clone)]
pub struct StatisticalSignificance {
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub effect_size: f64,
}

/// Complete benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub timestamp: DateTime<Utc>,
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub total_scenarios: usize,
    pub successful_scenarios: usize,
    pub failed_scenarios: usize,
    pub average_improvement: f64,
    pub key_findings: Vec<String>,
}

/// Benchmark executor that runs Claude Code CLI commands
pub struct BenchmarkExecutor {
    config: BenchmarkConfig,
    stream_collector: Arc<Mutex<StreamMetricsCollector>>,
}

impl BenchmarkExecutor {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            stream_collector: Arc::new(Mutex::new(StreamMetricsCollector::new())),
        }
    }

    /// Execute a benchmark scenario
    pub async fn execute(
        &self,
        scenario: &BenchmarkScenario,
        mode: ExecutionMode,
        run_id: &str,
    ) -> Result<PerformanceMetrics> {
        info!("Executing scenario {} in {:?} mode", scenario.name, mode);

        let command = self.build_command(scenario, mode)?;
        let mut child = self.spawn_command(&command)?;

        // Set up stream processing
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture stdout"))?;

        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        // Process output stream
        let collector = self.stream_collector.clone();
        let metrics_task = tokio::spawn(async move {
            let mut collector = collector.lock().await;
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(event) = serde_json::from_str::<ClaudeStreamEvent>(&line) {
                    collector.process_event(event).await;
                }
            }
        });

        // Wait for completion with timeout
        let result = timeout(self.config.execution_timeout, child.wait()).await;

        match result {
            Ok(Ok(status)) => {
                if !status.success() {
                    warn!("Command exited with non-zero status: {:?}", status);
                }
            }
            Ok(Err(e)) => {
                error!("Command execution error: {}", e);
                return Err(anyhow!("Command execution failed: {}", e));
            }
            Err(_) => {
                error!("Command execution timeout");
                // Kill the process
                let _ = child.kill().await;
                return Err(anyhow!("Command execution timeout"));
            }
        }

        // Wait for metrics collection to complete
        let _ = metrics_task.await;

        // Extract metrics
        let collector = self.stream_collector.lock().await;
        Ok(collector.finalize())
    }

    /// Build the command to execute
    fn build_command(
        &self,
        scenario: &BenchmarkScenario,
        mode: ExecutionMode,
    ) -> Result<Vec<String>> {
        let mut command = vec![self.config.claude_executable.to_string_lossy().to_string()];

        match mode {
            ExecutionMode::Baseline => {
                command.push(format!(
                    "solve SWE-bench instance {} without ML optimization",
                    scenario.instance_id
                ));
            }
            ExecutionMode::MLOptimized => {
                command.push(format!(
                    "solve SWE-bench instance {} using ML-optimized swarm coordination",
                    scenario.instance_id
                ));
            }
        }

        // Add common flags
        command.extend(vec![
            "-p".to_string(),
            "--dangerously-skip-permissions".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string(),
        ]);

        Ok(command)
    }

    /// Spawn the command process
    fn spawn_command(&self, args: &[String]) -> Result<Child> {
        if args.is_empty() {
            return Err(anyhow!("No command provided"));
        }

        let mut cmd = Command::new(&args[0]);
        for arg in &args[1..] {
            cmd.arg(arg);
        }

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn command: {}", e))
    }
}

// Module exports
pub use claude_executor::ExecutionResult;
pub use comparator::{ComparisonFramework, StatisticalAnalyzer};
pub use metrics::{
    CodeQualityScore, ErrorRecovery, MetricType, ResourceUsage, SwarmCoordinationMetrics,
    ThinkingSequence, ToolInvocation,
};
pub use realtime::{MonitoringServer, RealTimeEvent};
pub use storage::{
    BenchmarkRun, CodeQualityMetric, ErrorRecoveryEvent, ResourceUsageRecord, StreamEvent,
    ThinkingSequenceRecord, ToolInvocationRecord,
};
pub use stream_parser::{
    ErrorRecoveryTracker, EventProcessor, StreamJSONParser, ThinkingPatternAnalyzer,
    ToolUsageProcessor,
};
pub use swe_bench_evaluator::{
    ComprehensiveEvaluationReport, ModelConfig, ModelEvaluationReport, ModelType,
    SWEBenchEvaluator, SWEBenchEvaluatorConfig, SWEBenchInstance, SWEBenchInstances, SolveStatus,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_framework_creation() {
        let temp_dir = tempdir().unwrap();
        let config = BenchmarkConfig {
            database_path: temp_dir.path().join("test.db"),
            enable_real_time_monitoring: false,
            ..Default::default()
        };

        let framework = BenchmarkingFramework::new(config).await;
        assert!(framework.is_ok());
    }

    #[tokio::test]
    async fn test_scenario_execution() {
        let scenario = BenchmarkScenario {
            name: "test_scenario".to_string(),
            instance_id: "test_001".to_string(),
            repository: "test/repo".to_string(),
            issue_description: "Test issue".to_string(),
            difficulty: Difficulty::Easy,
            claude_command: "echo 'test'".to_string(),
            expected_files_modified: vec![],
            validation_tests: vec![],
        };

        // Test scenario creation
        assert_eq!(scenario.difficulty.to_string(), "easy");
    }
}
