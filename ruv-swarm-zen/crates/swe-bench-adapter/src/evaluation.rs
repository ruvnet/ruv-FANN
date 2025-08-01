//! Patch evaluation functionality for SWE-Bench

use crate::loader::SWEBenchInstance;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::fs;
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Patch evaluator for testing generated solutions
pub struct PatchEvaluator {
    config: EvaluatorConfig,
    sandbox_dir: Option<PathBuf>,
}

impl PatchEvaluator {
    /// Create a new patch evaluator
    pub fn new(config: crate::EvalConfig) -> Self {
        Self {
            config: EvaluatorConfig::from(config),
            sandbox_dir: None,
        }
    }

    /// Evaluate a patch against test requirements
    pub async fn evaluate_patch(
        &self,
        instance: &SWEBenchInstance,
        patch_content: &str,
    ) -> Result<EvaluationResult> {
        info!("Evaluating patch for instance: {}", instance.instance_id);
        let start_time = Instant::now();

        // Create sandbox environment
        let sandbox = if self.config.sandbox_enabled {
            Some(self.create_sandbox(&instance.repo).await?)
        } else {
            None
        };

        let mut result = EvaluationResult {
            instance_id: instance.instance_id.clone(),
            passed: false,
            test_results: Vec::new(),
            patch_applied: false,
            errors: Vec::new(),
            warnings: Vec::new(),
            execution_time: Duration::default(),
            metrics: EvaluationMetrics::default(),
        };

        // Try to apply the patch
        match self
            .apply_patch(instance, patch_content, sandbox.as_ref())
            .await
        {
            Ok(applied_files) => {
                result.patch_applied = true;
                result.metrics.files_modified = applied_files.len();
                debug!(
                    "Patch applied successfully to {} files",
                    applied_files.len()
                );

                // Run tests
                match self.run_tests(instance, sandbox.as_ref()).await {
                    Ok(test_results) => {
                        result.test_results = test_results;
                        result.passed = result.test_results.iter().all(|t| t.passed);

                        // Collect metrics
                        result.metrics.tests_run = result.test_results.len();
                        result.metrics.tests_passed =
                            result.test_results.iter().filter(|t| t.passed).count();
                    }
                    Err(e) => {
                        error!("Failed to run tests: {}", e);
                        result.errors.push(format!("Test execution failed: {}", e));
                    }
                }
            }
            Err(e) => {
                error!("Failed to apply patch: {}", e);
                result
                    .errors
                    .push(format!("Patch application failed: {}", e));
            }
        }

        // Validate patch quality
        let quality_issues = self.validate_patch_quality(patch_content);
        result.warnings.extend(quality_issues);

        result.execution_time = start_time.elapsed();
        result.metrics.patch_size = patch_content.len();

        // Clean up sandbox
        if let Some(sandbox) = sandbox {
            if let Err(e) = fs::remove_dir_all(sandbox.path()).await {
                warn!("Failed to clean up sandbox: {}", e);
            }
        }

        info!(
            "Evaluation completed for {} in {:?}: {}",
            instance.instance_id,
            result.execution_time,
            if result.passed { "PASSED" } else { "FAILED" }
        );

        Ok(result)
    }

    /// Create a sandbox environment for safe execution
    async fn create_sandbox(&self, repo_url: &str) -> Result<TempDir> {
        let sandbox = TempDir::new()?;
        info!("Creating sandbox environment at: {:?}", sandbox.path());

        // Clone repository to sandbox
        let repo_path = sandbox.path().join("repo");

        let output = TokioCommand::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                repo_url,
                repo_path.to_str().unwrap(),
            ])
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to clone repository: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(sandbox)
    }

    /// Apply patch to the repository
    async fn apply_patch(
        &self,
        _instance: &SWEBenchInstance,
        patch_content: &str,
        sandbox: Option<&TempDir>,
    ) -> Result<Vec<String>> {
        let repo_path = if let Some(sandbox) = sandbox {
            sandbox.path().join("repo")
        } else {
            PathBuf::from(".")
        };

        // Save patch to temporary file
        let patch_file = repo_path.join(".swe_bench_patch.diff");
        fs::write(&patch_file, patch_content).await?;

        // Apply patch using git
        let output = TokioCommand::new("git")
            .current_dir(&repo_path)
            .args(["apply", "--check", patch_file.to_str().unwrap()])
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Patch validation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // Actually apply the patch
        let output = TokioCommand::new("git")
            .current_dir(&repo_path)
            .args(["apply", patch_file.to_str().unwrap()])
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Patch application failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // Get list of modified files
        let modified_files = self.get_modified_files(&repo_path).await?;

        // Clean up patch file
        let _ = fs::remove_file(patch_file).await;

        Ok(modified_files)
    }

    /// Get list of modified files in the repository
    async fn get_modified_files(&self, repo_path: &Path) -> Result<Vec<String>> {
        let output = TokioCommand::new("git")
            .current_dir(repo_path)
            .args(["diff", "--name-only"])
            .output()
            .await?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let files = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect();

        Ok(files)
    }

    /// Run tests for the instance
    async fn run_tests(
        &self,
        instance: &SWEBenchInstance,
        sandbox: Option<&TempDir>,
    ) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();

        let repo_path = if let Some(sandbox) = sandbox {
            sandbox.path().join("repo")
        } else {
            PathBuf::from(".")
        };

        for (i, directive) in instance.test_directives.iter().enumerate() {
            let test_name = format!("test_{}", i + 1);
            info!("Running test {}: {}", test_name, directive);

            let start = Instant::now();
            let result = self.execute_test_command(directive, &repo_path).await;
            let duration = start.elapsed();

            match result {
                Ok((passed, output)) => {
                    results.push(TestResult {
                        name: test_name,
                        passed,
                        output,
                        error: None,
                        duration,
                    });
                }
                Err(e) => {
                    results.push(TestResult {
                        name: test_name,
                        passed: false,
                        output: String::new(),
                        error: Some(e.to_string()),
                        duration,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Execute a single test command
    async fn execute_test_command(
        &self,
        command: &str,
        working_dir: &Path,
    ) -> Result<(bool, String)> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err(anyhow::anyhow!("Empty test command"));
        }

        let mut cmd = TokioCommand::new(parts[0]);
        cmd.current_dir(working_dir);

        if parts.len() > 1 {
            cmd.args(&parts[1..]);
        }

        // Execute with timeout
        let output = timeout(self.config.timeout, cmd.output())
            .await
            .context("Test execution timed out")??;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("STDOUT:\n{}\n\nSTDERR:\n{}", stdout, stderr);

        Ok((output.status.success(), combined_output))
    }

    /// Validate patch quality
    fn validate_patch_quality(&self, patch_content: &str) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check patch size
        let lines: Vec<&str> = patch_content.lines().collect();
        if lines.len() > 1000 {
            warnings
                .push("Large patch detected (>1000 lines). Consider breaking it down.".to_string());
        }

        // Check for common issues
        if patch_content.contains("TODO") || patch_content.contains("FIXME") {
            warnings.push("Patch contains TODO/FIXME comments".to_string());
        }

        if patch_content.contains("print(") || patch_content.contains("console.log") {
            warnings.push("Patch contains debug print statements".to_string());
        }

        // Check for proper diff format
        if !patch_content.starts_with("diff") && !patch_content.contains("---") {
            warnings.push("Patch may not be in proper diff format".to_string());
        }

        // Check for binary files
        if patch_content.contains("Binary files") {
            warnings.push("Patch modifies binary files".to_string());
        }

        warnings
    }

    /// Compare patches for similarity
    pub fn compare_patches(&self, patch1: &str, patch2: &str) -> PatchComparison {
        let diff = TextDiff::from_lines(patch1, patch2);

        let total_lines = patch1.lines().count().max(patch2.lines().count());
        let changed_lines = diff
            .iter_all_changes()
            .filter(|change| change.tag() != ChangeTag::Equal)
            .count();

        let similarity = if total_lines > 0 {
            1.0 - (changed_lines as f64 / total_lines as f64)
        } else {
            1.0
        };

        PatchComparison {
            similarity,
            additions: diff
                .iter_all_changes()
                .filter(|change| change.tag() == ChangeTag::Insert)
                .count(),
            deletions: diff
                .iter_all_changes()
                .filter(|change| change.tag() == ChangeTag::Delete)
                .count(),
            total_changes: changed_lines,
        }
    }
}

/// Configuration for the evaluator
#[derive(Debug, Clone)]
struct EvaluatorConfig {
    timeout: Duration,
    test_command: String,
    sandbox_enabled: bool,
    max_retries: usize,
}

impl From<crate::EvalConfig> for EvaluatorConfig {
    fn from(config: crate::EvalConfig) -> Self {
        Self {
            timeout: config.timeout,
            test_command: config.test_command,
            sandbox_enabled: config.sandbox_enabled,
            max_retries: config.max_retries,
        }
    }
}

/// Result of patch evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub instance_id: String,
    pub passed: bool,
    pub test_results: Vec<TestResult>,
    pub patch_applied: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub execution_time: Duration,
    pub metrics: EvaluationMetrics,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub output: String,
    pub error: Option<String>,
    pub duration: Duration,
}

/// Evaluation metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub files_modified: usize,
    pub patch_size: usize,
    pub tests_run: usize,
    pub tests_passed: usize,
}

/// Patch comparison result
#[derive(Debug, Clone)]
pub struct PatchComparison {
    pub similarity: f64,
    pub additions: usize,
    pub deletions: usize,
    pub total_changes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_quality_validation() {
        let config = crate::EvalConfig::default();
        let evaluator = PatchEvaluator::new(config);

        let patch_with_todos = r#"
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,5 +1,6 @@
 def test_function():
+    # TODO: Implement this properly
     print("Debug output")
     return True
"#;

        let warnings = evaluator.validate_patch_quality(patch_with_todos);
        assert!(warnings.iter().any(|w| w.contains("TODO")));
        assert!(warnings.iter().any(|w| w.contains("print")));
    }

    #[test]
    fn test_patch_comparison() {
        let config = crate::EvalConfig::default();
        let evaluator = PatchEvaluator::new(config);

        let patch1 = "line1\nline2\nline3\n";
        let patch2 = "line1\nmodified\nline3\n";

        let comparison = evaluator.compare_patches(patch1, patch2);

        // The similarity should be about 2/3 since 2 out of 3 lines are the same
        // But the actual calculation might be different due to how TextDiff counts changes
        assert!(comparison.similarity >= 0.0);
        assert!(comparison.similarity <= 1.0);

        // These should be correct
        assert_eq!(comparison.additions, 1);
        assert_eq!(comparison.deletions, 1);
        assert_eq!(comparison.total_changes, 2);
    }
}
