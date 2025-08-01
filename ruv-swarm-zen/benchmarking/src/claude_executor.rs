//! Claude command executor module

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::BufReader;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{error, info};

use crate::metrics::PerformanceMetrics;
use crate::storage::BenchmarkStorage;
use crate::stream_parser::StreamJSONParser;
use crate::{BenchmarkScenario, ExecutionMode};

/// Claude command executor
pub struct ClaudeCommandExecutor {
    claude_path: PathBuf,
    timeout_duration: Duration,
    working_directory: Option<PathBuf>,
}

impl ClaudeCommandExecutor {
    /// Create a new Claude command executor
    pub fn new(claude_path: PathBuf) -> Self {
        Self {
            claude_path,
            timeout_duration: Duration::from_secs(1800), // 30 minutes default
            working_directory: None,
        }
    }

    /// Set execution timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_duration = timeout;
        self
    }

    /// Set working directory
    pub fn with_working_directory(mut self, dir: PathBuf) -> Self {
        self.working_directory = Some(dir);
        self
    }

    /// Execute a Claude command for benchmarking
    pub async fn execute_benchmark(
        &self,
        scenario: &BenchmarkScenario,
        mode: ExecutionMode,
        run_id: &str,
        storage: Option<&BenchmarkStorage>,
    ) -> Result<ExecutionResult> {
        info!(
            "Executing Claude command for scenario: {} ({})",
            scenario.name,
            mode.to_string()
        );

        let command_args = self.build_command_args(scenario, &mode)?;
        let mut child = self.spawn_command(&command_args)?;

        // Capture stdout for stream processing
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture stdout"))?;

        // Create stream parser
        let parser = StreamJSONParser::new();
        let reader = BufReader::new(stdout);
        let run_id_owned = run_id.to_string();

        // Process stream in background
        let parse_handle =
            tokio::spawn(async move { parser.process_stream(reader, &run_id_owned).await });

        // Wait for command completion with timeout
        let exit_status = match timeout(self.timeout_duration, child.wait()).await {
            Ok(Ok(status)) => {
                info!("Claude command completed with status: {:?}", status);
                status
            }
            Ok(Err(e)) => {
                error!("Claude command failed: {}", e);
                return Err(anyhow!("Command execution failed: {}", e));
            }
            Err(_) => {
                error!("Claude command timed out after {:?}", self.timeout_duration);
                // Kill the process
                let _ = child.kill().await;
                return Err(anyhow!("Command execution timeout"));
            }
        };

        // Get metrics from parser
        let metrics = parse_handle.await??;

        Ok(ExecutionResult {
            run_id: run_id.to_string(),
            exit_code: exit_status.code().unwrap_or(-1),
            success: exit_status.success(),
            metrics,
            output_size: 0, // Would need to track this during streaming
        })
    }

    /// Execute a specific SWE-Bench instance
    pub async fn execute_swe_bench(
        &self,
        instance_id: &str,
        mode: ExecutionMode,
        run_id: &str,
    ) -> Result<SWEBenchResult> {
        let command = match mode {
            ExecutionMode::Baseline => {
                format!(
                    "solve SWE-bench instance {} without ML optimization",
                    instance_id
                )
            }
            ExecutionMode::MLOptimized => {
                format!(
                    "solve SWE-bench instance {} using ML-optimized swarm coordination",
                    instance_id
                )
            }
        };

        let args = [self.claude_path.to_string_lossy().to_string(),
            command,
            "-p".to_string(),
            "--dangerously-skip-permissions".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string()];

        let mut child = Command::new(&args[0])
            .args(&args[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture stdout"))?;

        // Process stream
        let parser = StreamJSONParser::new();
        let reader = BufReader::new(stdout);
        let run_id_owned = run_id.to_string();

        let parse_handle =
            tokio::spawn(async move { parser.process_stream(reader, &run_id_owned).await });

        // Wait for completion
        let exit_status = timeout(self.timeout_duration, child.wait()).await??;
        let metrics = parse_handle.await??;

        // Extract SWE-Bench specific results
        let swe_result = self.extract_swe_bench_results(&metrics, instance_id);

        Ok(swe_result)
    }

    /// Build command arguments based on scenario and mode
    fn build_command_args(
        &self,
        scenario: &BenchmarkScenario,
        mode: &ExecutionMode,
    ) -> Result<Vec<String>> {
        let mut args = vec![self.claude_path.to_string_lossy().to_string()];

        // Add the main command based on mode
        match mode {
            ExecutionMode::Baseline => {
                args.push(scenario.claude_command.clone());
            }
            ExecutionMode::MLOptimized => {
                // Modify command for ML-optimized execution
                args.push(format!(
                    "{} with ML-optimized swarm coordination",
                    scenario.claude_command
                ));
            }
        }

        // Add standard flags
        args.extend(vec![
            "-p".to_string(),
            "--dangerously-skip-permissions".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string(),
        ]);

        Ok(args)
    }

    /// Spawn the command process
    fn spawn_command(&self, args: &[String]) -> Result<tokio::process::Child> {
        if args.is_empty() {
            return Err(anyhow!("No command arguments provided"));
        }

        let mut cmd = Command::new(&args[0]);

        for arg in &args[1..] {
            cmd.arg(arg);
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        if let Some(dir) = &self.working_directory {
            cmd.current_dir(dir);
        }

        cmd.spawn()
            .map_err(|e| anyhow!("Failed to spawn command: {}", e))
    }

    /// Extract SWE-Bench specific results from metrics
    fn extract_swe_bench_results(
        &self,
        metrics: &PerformanceMetrics,
        instance_id: &str,
    ) -> SWEBenchResult {
        SWEBenchResult {
            instance_id: instance_id.to_string(),
            tests_passed: 0, // Would need to parse from output
            tests_failed: 0,
            patch_applied: true,
            files_modified: Vec::new(),
            patch_size_bytes: 0,
            execution_time: metrics.task_completion_time,
            tool_invocations: metrics.tool_invocations.len(),
            thinking_sequences: metrics.thinking_sequences.len(),
            error_recoveries: metrics.error_recoveries.len(),
        }
    }
}

/// Result of executing a Claude command
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub run_id: String,
    pub exit_code: i32,
    pub success: bool,
    pub metrics: PerformanceMetrics,
    pub output_size: usize,
}

/// SWE-Bench specific execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SWEBenchResult {
    pub instance_id: String,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub patch_applied: bool,
    pub files_modified: Vec<String>,
    pub patch_size_bytes: usize,
    pub execution_time: Duration,
    pub tool_invocations: usize,
    pub thinking_sequences: usize,
    pub error_recoveries: usize,
}

/// Batch executor for running multiple benchmarks
pub struct BatchExecutor {
    executor: ClaudeCommandExecutor,
    parallel_jobs: usize,
}

impl BatchExecutor {
    pub fn new(claude_path: PathBuf) -> Self {
        Self {
            executor: ClaudeCommandExecutor::new(claude_path),
            parallel_jobs: 4,
        }
    }

    /// Set number of parallel jobs
    pub fn with_parallel_jobs(mut self, jobs: usize) -> Self {
        self.parallel_jobs = jobs;
        self
    }

    /// Execute multiple scenarios in parallel
    pub async fn execute_batch(
        &self,
        scenarios: Vec<BenchmarkScenario>,
        mode: ExecutionMode,
    ) -> Result<Vec<ExecutionResult>> {
        use futures::stream::{self, StreamExt};

        let results = stream::iter(scenarios)
            .map(|scenario| {
                let executor = self.executor.clone();
                let mode = mode.clone();
                let run_id = format!(
                    "{}-{}-{}",
                    scenario.instance_id,
                    mode.to_string(),
                    uuid::Uuid::new_v4()
                );

                async move {
                    executor
                        .execute_benchmark(&scenario, mode, &run_id, None)
                        .await
                }
            })
            .buffer_unordered(self.parallel_jobs)
            .collect::<Vec<_>>()
            .await;

        // Collect successful results and log errors
        let mut successful_results = Vec::new();
        for result in results {
            match result {
                Ok(res) => successful_results.push(res),
                Err(e) => error!("Batch execution error: {}", e),
            }
        }

        Ok(successful_results)
    }
}

// Make ClaudeCommandExecutor cloneable for parallel execution
impl Clone for ClaudeCommandExecutor {
    fn clone(&self) -> Self {
        Self {
            claude_path: self.claude_path.clone(),
            timeout_duration: self.timeout_duration,
            working_directory: self.working_directory.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_executor_creation() {
        let executor = ClaudeCommandExecutor::new(PathBuf::from("claude"));
        assert_eq!(executor.timeout_duration, Duration::from_secs(1800));
    }

    #[test]
    fn test_command_args_building() {
        let executor = ClaudeCommandExecutor::new(PathBuf::from("claude"));

        let scenario = BenchmarkScenario {
            name: "test".to_string(),
            instance_id: "test-001".to_string(),
            repository: "test/repo".to_string(),
            issue_description: "Test issue".to_string(),
            difficulty: crate::Difficulty::Easy,
            claude_command: "solve test problem".to_string(),
            expected_files_modified: vec![],
            validation_tests: vec![],
        };

        let args = executor
            .build_command_args(&scenario, &ExecutionMode::Baseline)
            .unwrap();

        assert_eq!(args[0], "claude");
        assert_eq!(args[1], "solve test problem");
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"--output-format".to_string()));
        assert!(args.contains(&"stream-json".to_string()));
    }

    #[test]
    fn test_batch_executor() {
        let batch = BatchExecutor::new(PathBuf::from("claude")).with_parallel_jobs(8);

        assert_eq!(batch.parallel_jobs, 8);
    }
}
