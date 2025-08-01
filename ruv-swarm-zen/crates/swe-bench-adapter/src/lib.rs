//! SWE-Bench adapter for ruv-swarm orchestration system
//!
//! This crate provides integration with SWE-Bench, a benchmark for evaluating
//! software engineering AI systems on real-world GitHub issues.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use ruv_swarm_core::{
    agent::Agent as AgentTrait,
    task::{Task, TaskId, TaskPayload, TaskPriority},
};
use ruv_swarm_persistence::SqliteStorage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{error, info};

pub mod benchmarking;
pub mod evaluation;
pub mod loader;
pub mod prompts;
pub mod stream_parser;

pub use benchmarking::{BenchmarkMetrics, BenchmarkRunner, PerformanceReport};
pub use evaluation::{EvaluationResult, PatchEvaluator};
pub use loader::{DifficultyLevel, InstanceLoader, SWEBenchInstance};
pub use prompts::{ClaudePromptGenerator, PromptTemplate};
pub use stream_parser::{MetricsCollector, StreamParser};

/// Error types for SWE-Bench adapter operations
#[derive(Error, Debug)]
pub enum SWEBenchError {
    #[error("Failed to load instance: {0}")]
    LoadError(String),

    #[error("Failed to generate prompt: {0}")]
    PromptError(String),

    #[error("Failed to evaluate patch: {0}")]
    EvaluationError(String),

    #[error("Failed to parse stream: {0}")]
    ParseError(String),

    #[error("Benchmark error: {0}")]
    BenchmarkError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Git error: {0}")]
    GitError(#[from] git2::Error),
}

/// Main SWE-Bench adapter for orchestrating evaluation tasks
pub struct SWEBenchAdapter {
    loader: Arc<RwLock<InstanceLoader>>,
    prompt_generator: Arc<ClaudePromptGenerator>,
    evaluator: Arc<PatchEvaluator>,
    benchmark_runner: Arc<BenchmarkRunner>,
    metrics_collector: Arc<RwLock<MetricsCollector>>,
    memory_store: Arc<SqliteStorage>,
}

impl SWEBenchAdapter {
    /// Create a new SWE-Bench adapter instance
    pub async fn new(config: SWEBenchConfig) -> Result<Self> {
        let loader = Arc::new(RwLock::new(InstanceLoader::new(&config.instances_path)?));
        let prompt_generator = Arc::new(ClaudePromptGenerator::new(config.prompt_config));
        let evaluator = Arc::new(PatchEvaluator::new(config.eval_config));
        let benchmark_runner = Arc::new(BenchmarkRunner::new(config.benchmark_config));
        let metrics_collector = Arc::new(RwLock::new(MetricsCollector::new()));
        let memory_store =
            Arc::new(SqliteStorage::new(&config.memory_path.to_string_lossy()).await?);

        Ok(Self {
            loader,
            prompt_generator,
            evaluator,
            benchmark_runner,
            metrics_collector,
            memory_store,
        })
    }

    /// Run evaluation on a specific SWE-Bench instance
    pub async fn evaluate_instance<A>(
        &self,
        instance_id: &str,
        _agent: &A,
    ) -> Result<EvaluationReport>
    where
        A: AgentTrait,
    {
        info!("Starting evaluation for instance: {}", instance_id);
        let start_time = Instant::now();

        // Load the instance
        let instance = self
            .loader
            .write()
            .await
            .load_instance(instance_id)
            .await
            .context("Failed to load instance")?;

        // Generate prompt for Claude Code CLI
        let prompt = self
            .prompt_generator
            .generate_prompt(&instance)
            .context("Failed to generate prompt")?;

        // Create evaluation task
        let _task = Task {
            id: TaskId::new(format!("swe-bench-{}", instance_id)),
            task_type: "swe-bench-evaluation".to_string(),
            priority: match instance.difficulty.priority() {
                1 => TaskPriority::Low,
                2 => TaskPriority::Normal,
                3 => TaskPriority::High,
                _ => TaskPriority::Critical,
            },
            payload: TaskPayload::Json(
                serde_json::json!({
                    "prompt": prompt.content.clone(),
                    "instance_id": instance_id,
                    "difficulty": instance.difficulty.to_string(),
                    "repo": instance.repo.clone(),
                })
                .to_string(),
            ),
            required_capabilities: vec!["code_generation".to_string()],
            timeout_ms: Some(300000),
            retry_count: 0,
            max_retries: 3,
        };

        // Execute task through agent (mock for now)
        let execution_result = ExecutionResult {
            output: "Mock execution output".to_string(),
            patch: "diff --git a/fix.py b/fix.py\n+fix".to_string(),
            exit_code: 0,
            duration: Duration::from_secs(10),
        };

        // Parse output stream and collect metrics
        let stream_metrics = {
            let mut collector = self.metrics_collector.write().await;
            collector.parse_stream(&execution_result.output)?
        };

        // Evaluate the generated patch
        let patch_result = self
            .evaluator
            .evaluate_patch(&instance, &execution_result.patch)
            .await?;

        // Run performance benchmarks
        let benchmark_metrics = self
            .benchmark_runner
            .run_benchmark(&instance, &execution_result)
            .await?;

        // Store results in memory
        let report = EvaluationReport {
            instance_id: instance_id.to_string(),
            difficulty: instance.difficulty,
            prompt_tokens: prompt.token_count,
            execution_time: start_time.elapsed(),
            patch_result,
            stream_metrics,
            benchmark_metrics,
            timestamp: Utc::now(),
        };

        // Store results in persistence layer
        // Note: SqliteStorage doesn't have a simple store method,
        // would need to implement a proper storage pattern

        info!(
            "Completed evaluation for instance: {} in {:?}",
            instance_id, report.execution_time
        );

        Ok(report)
    }

    /// Run batch evaluation on multiple instances
    pub async fn evaluate_batch<A>(
        &self,
        instance_ids: Vec<String>,
        agents: Vec<A>,
        parallel: bool,
    ) -> Result<BatchEvaluationReport>
    where
        A: AgentTrait + Clone + 'static,
    {
        info!(
            "Starting batch evaluation for {} instances",
            instance_ids.len()
        );
        let start_time = Instant::now();

        let reports = if parallel {
            self.evaluate_parallel(instance_ids, agents).await?
        } else {
            self.evaluate_sequential(instance_ids, agents).await?
        };

        let total_time = start_time.elapsed();
        let success_rate =
            reports.iter().filter(|r| r.patch_result.passed).count() as f64 / reports.len() as f64;

        let batch_report = BatchEvaluationReport {
            total_instances: reports.len(),
            successful: reports.iter().filter(|r| r.patch_result.passed).count(),
            failed: reports.iter().filter(|r| !r.patch_result.passed).count(),
            success_rate,
            total_time,
            reports,
            timestamp: Utc::now(),
        };

        // Store batch report
        // Store batch report in persistence layer
        // Note: SqliteStorage doesn't have a simple store method

        Ok(batch_report)
    }

    /// Evaluate instances in parallel
    async fn evaluate_parallel<A>(
        &self,
        instance_ids: Vec<String>,
        agents: Vec<A>,
    ) -> Result<Vec<EvaluationReport>>
    where
        A: AgentTrait + Clone + 'static,
    {
        use futures::stream::{self, StreamExt};

        let num_agents = agents.len();
        let agent_pool = Arc::new(RwLock::new(agents));

        let results = stream::iter(instance_ids)
            .map(|instance_id| {
                let adapter = self.clone();
                let agent_pool = agent_pool.clone();

                async move {
                    // Get an available agent
                    let agent = {
                        let mut pool = agent_pool.write().await;
                        pool.pop()
                            .ok_or_else(|| anyhow::anyhow!("No available agents"))?
                    };

                    let result = adapter.evaluate_instance(&instance_id, &agent).await;

                    // Return agent to pool
                    {
                        let mut pool = agent_pool.write().await;
                        pool.push(agent);
                    }

                    result
                }
            })
            .buffer_unordered(num_agents)
            .collect::<Vec<_>>()
            .await;

        results.into_iter().collect()
    }

    /// Evaluate instances sequentially
    async fn evaluate_sequential<A>(
        &self,
        instance_ids: Vec<String>,
        mut agents: Vec<A>,
    ) -> Result<Vec<EvaluationReport>>
    where
        A: AgentTrait,
    {
        let mut reports = Vec::new();
        let agent = agents
            .pop()
            .ok_or_else(|| anyhow::anyhow!("No agents provided"))?;

        for instance_id in instance_ids {
            let report = self.evaluate_instance(&instance_id, &agent).await?;
            reports.push(report);
        }

        Ok(reports)
    }

    /// Get evaluation statistics
    pub async fn get_statistics(&self) -> Result<EvaluationStatistics> {
        // Query all reports from storage
        let all_reports = vec![];

        let total_evaluations = all_reports.len();
        let successful = all_reports
            .iter()
            .filter_map(|v: &serde_json::Value| {
                serde_json::from_value::<EvaluationReport>(v.clone()).ok()
            })
            .filter(|r| r.patch_result.passed)
            .count();

        let by_difficulty = self.calculate_difficulty_stats(&all_reports)?;
        let avg_execution_time = self.calculate_avg_execution_time(&all_reports)?;

        Ok(EvaluationStatistics {
            total_evaluations,
            successful,
            failed: total_evaluations - successful,
            success_rate: successful as f64 / total_evaluations as f64,
            by_difficulty,
            avg_execution_time,
            timestamp: Utc::now(),
        })
    }

    fn calculate_difficulty_stats(
        &self,
        reports: &[serde_json::Value],
    ) -> Result<HashMap<DifficultyLevel, DifficultyStats>> {
        let mut stats = HashMap::new();

        for report_value in reports {
            if let Ok(report) = serde_json::from_value::<EvaluationReport>(report_value.clone()) {
                let entry = stats.entry(report.difficulty).or_insert(DifficultyStats {
                    total: 0,
                    successful: 0,
                    avg_time: Duration::default(),
                });

                entry.total += 1;
                if report.patch_result.passed {
                    entry.successful += 1;
                }
            }
        }

        Ok(stats)
    }

    fn calculate_avg_execution_time(&self, reports: &[serde_json::Value]) -> Result<Duration> {
        let times: Vec<Duration> = reports
            .iter()
            .filter_map(|v: &serde_json::Value| {
                serde_json::from_value::<EvaluationReport>(v.clone()).ok()
            })
            .map(|r| r.execution_time)
            .collect();

        if times.is_empty() {
            return Ok(Duration::default());
        }

        let total: Duration = times.iter().sum();
        Ok(total / times.len() as u32)
    }
}

impl Clone for SWEBenchAdapter {
    fn clone(&self) -> Self {
        Self {
            loader: Arc::clone(&self.loader),
            prompt_generator: Arc::clone(&self.prompt_generator),
            evaluator: Arc::clone(&self.evaluator),
            benchmark_runner: Arc::clone(&self.benchmark_runner),
            metrics_collector: Arc::clone(&self.metrics_collector),
            memory_store: Arc::clone(&self.memory_store),
        }
    }
}

/// Configuration for SWE-Bench adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SWEBenchConfig {
    pub instances_path: PathBuf,
    pub memory_path: PathBuf,
    pub prompt_config: PromptConfig,
    pub eval_config: EvalConfig,
    pub benchmark_config: BenchmarkConfig,
}

impl Default for SWEBenchConfig {
    fn default() -> Self {
        Self {
            instances_path: PathBuf::from("./swe-bench-instances"),
            memory_path: PathBuf::from("./swe-bench-memory"),
            prompt_config: PromptConfig::default(),
            eval_config: EvalConfig::default(),
            benchmark_config: BenchmarkConfig::default(),
        }
    }
}

/// Configuration for prompt generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    pub max_tokens: usize,
    pub include_test_hints: bool,
    pub include_context_files: bool,
    pub template_style: PromptStyle,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4000,
            include_test_hints: true,
            include_context_files: true,
            template_style: PromptStyle::ClaudeCode,
        }
    }
}

/// Prompt generation styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptStyle {
    ClaudeCode,
    Minimal,
    Detailed,
    Custom(String),
}

/// Configuration for patch evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub timeout: Duration,
    pub test_command: String,
    pub sandbox_enabled: bool,
    pub max_retries: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            test_command: "pytest".to_string(),
            sandbox_enabled: true,
            max_retries: 3,
        }
    }
}

/// Configuration for performance benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub iterations: usize,
    pub warm_up: usize,
    pub measure_memory: bool,
    pub profile_enabled: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            warm_up: 3,
            measure_memory: true,
            profile_enabled: false,
        }
    }
}

/// Evaluation report for a single instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub instance_id: String,
    pub difficulty: DifficultyLevel,
    pub prompt_tokens: usize,
    pub execution_time: Duration,
    pub patch_result: EvaluationResult,
    pub stream_metrics: StreamMetrics,
    pub benchmark_metrics: BenchmarkMetrics,
    pub timestamp: DateTime<Utc>,
}

/// Batch evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEvaluationReport {
    pub total_instances: usize,
    pub successful: usize,
    pub failed: usize,
    pub success_rate: f64,
    pub total_time: Duration,
    pub reports: Vec<EvaluationReport>,
    pub timestamp: DateTime<Utc>,
}

/// Stream metrics collected during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    pub total_tokens: usize,
    pub tool_calls: usize,
    pub file_operations: FileOperationStats,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// File operation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileOperationStats {
    pub reads: usize,
    pub writes: usize,
    pub creates: usize,
    pub deletes: usize,
}

/// Evaluation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationStatistics {
    pub total_evaluations: usize,
    pub successful: usize,
    pub failed: usize,
    pub success_rate: f64,
    pub by_difficulty: HashMap<DifficultyLevel, DifficultyStats>,
    pub avg_execution_time: Duration,
    pub timestamp: DateTime<Utc>,
}

/// Statistics by difficulty level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyStats {
    pub total: usize,
    pub successful: usize,
    pub avg_time: Duration,
}

/// Agent execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub output: String,
    pub patch: String,
    pub exit_code: i32,
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() {
        let config = SWEBenchConfig::default();
        let adapter = SWEBenchAdapter::new(config).await;
        assert!(adapter.is_ok());
    }

    #[tokio::test]
    async fn test_prompt_config() {
        let config = PromptConfig::default();
        assert_eq!(config.max_tokens, 4000);
        assert!(config.include_test_hints);
        assert!(config.include_context_files);
    }

    #[tokio::test]
    async fn test_eval_config() {
        let config = EvalConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.test_command, "pytest");
        assert!(config.sandbox_enabled);
        assert_eq!(config.max_retries, 3);
    }
}
