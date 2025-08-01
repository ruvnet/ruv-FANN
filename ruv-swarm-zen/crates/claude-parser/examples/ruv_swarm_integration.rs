//! Example of integrating Claude parser with RUV-SWARM benchmarking system

use claude_parser::{ClaudeStreamParser, PerformanceMetrics};
use std::collections::HashMap;
use std::time::Duration;

/// Integration with RUV-SWARM's benchmarking system
struct RuvSwarmBenchmark {
    instance_id: String,
    mode: ExecutionMode,
    parser: ClaudeStreamParser,
}

#[derive(Debug, Clone, PartialEq)]
enum ExecutionMode {
    Baseline,
    MLOptimized,
}

impl RuvSwarmBenchmark {
    fn new(instance_id: String, mode: ExecutionMode) -> Self {
        Self {
            instance_id,
            mode,
            parser: ClaudeStreamParser::new(),
        }
    }

    /// Parse Claude output and convert to RUV-SWARM benchmark metrics
    async fn process_claude_output<R: tokio::io::AsyncRead + Unpin>(
        &mut self,
        reader: R,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        // Parse the stream
        let metrics = self.parser.parse_stream(reader).await?;

        // Convert to benchmark result
        Ok(self.convert_to_benchmark_result(metrics))
    }

    fn convert_to_benchmark_result(&self, metrics: PerformanceMetrics) -> BenchmarkResult {
        // Calculate efficiency scores
        let tool_efficiency = self.calculate_tool_efficiency(&metrics);
        let thinking_efficiency = self.calculate_thinking_efficiency(&metrics);
        let error_recovery_score = self.calculate_error_recovery_score(&metrics);

        BenchmarkResult {
            instance_id: self.instance_id.clone(),
            mode: self.mode.clone(),
            duration: metrics.total_duration,
            time_to_first_output: metrics.time_to_first_output,
            total_tokens: metrics.token_usage.total_tokens,
            tokens_per_second: metrics.token_usage.tokens_per_second,
            tool_efficiency,
            thinking_efficiency,
            error_recovery_score,
            raw_metrics: metrics,
        }
    }

    fn calculate_tool_efficiency(&self, metrics: &PerformanceMetrics) -> f64 {
        // Calculate tool efficiency based on invocation patterns
        let total_invocations: u64 = metrics
            .tool_invocations
            .values()
            .map(|t| t.invocation_count)
            .sum();

        if total_invocations == 0 {
            return 1.0;
        }

        // Penalize excessive tool usage
        let expected_invocations = 5.0; // baseline expectation
        let efficiency = expected_invocations / total_invocations as f64;
        efficiency.min(1.0)
    }

    fn calculate_thinking_efficiency(&self, metrics: &PerformanceMetrics) -> f64 {
        // Calculate thinking efficiency based on token usage
        if metrics.thinking_metrics.total_sequences == 0 {
            return 1.0;
        }

        let avg_tokens = metrics.thinking_metrics.average_tokens_per_sequence;
        let optimal_tokens = 50.0; // optimal thinking tokens per sequence

        // Score is higher when closer to optimal
        1.0 - ((avg_tokens - optimal_tokens).abs() / optimal_tokens).min(1.0)
    }

    fn calculate_error_recovery_score(&self, metrics: &PerformanceMetrics) -> f64 {
        if metrics.error_metrics.total_errors == 0 {
            return 1.0; // No errors is perfect
        }

        metrics.error_metrics.recovery_success_rate
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    instance_id: String,
    mode: ExecutionMode,
    duration: Duration,
    time_to_first_output: Option<Duration>,
    total_tokens: u64,
    tokens_per_second: f64,
    tool_efficiency: f64,
    thinking_efficiency: f64,
    error_recovery_score: f64,
    raw_metrics: PerformanceMetrics,
}

impl BenchmarkResult {
    fn overall_score(&self) -> f64 {
        // Weighted average of different efficiency metrics
        let weights = [
            (0.3, self.tool_efficiency),
            (0.3, self.thinking_efficiency),
            (0.2, self.error_recovery_score),
            (0.2, self.tokens_per_second / 1000.0), // normalize tokens/sec
        ];

        weights.iter().map(|(w, score)| w * score.min(1.0)).sum()
    }

    fn to_sql_insert(&self) -> String {
        format!(
            r#"INSERT INTO benchmark_runs (
                run_id, instance_id, execution_mode, 
                start_time, end_time, status, 
                claude_command, configuration
            ) VALUES (
                '{}', '{}', '{}',
                datetime('now'), datetime('now', '+{} seconds'), 'completed',
                'solve SWE-bench instance {}', '{}'
            );"#,
            uuid::Uuid::new_v4(),
            self.instance_id,
            match self.mode {
                ExecutionMode::Baseline => "baseline",
                ExecutionMode::MLOptimized => "ml_optimized",
            },
            self.duration.as_secs(),
            self.instance_id,
            serde_json::json!({
                "mode": format!("{:?}", self.mode),
                "tool_efficiency": self.tool_efficiency,
                "thinking_efficiency": self.thinking_efficiency,
                "error_recovery_score": self.error_recovery_score,
            })
        )
    }
}

/// Compare baseline and ML-optimized results
fn compare_results(baseline: &BenchmarkResult, ml_optimized: &BenchmarkResult) -> ComparisonReport {
    ComparisonReport {
        instance_id: baseline.instance_id.clone(),
        duration_improvement: calculate_improvement(
            baseline.duration.as_secs_f64(),
            ml_optimized.duration.as_secs_f64(),
        ),
        token_improvement: calculate_improvement(
            baseline.total_tokens as f64,
            ml_optimized.total_tokens as f64,
        ),
        tool_efficiency_improvement: calculate_improvement(
            baseline.tool_efficiency,
            ml_optimized.tool_efficiency,
        ),
        overall_score_improvement: calculate_improvement(
            baseline.overall_score(),
            ml_optimized.overall_score(),
        ),
    }
}

fn calculate_improvement(baseline: f64, optimized: f64) -> f64 {
    ((baseline - optimized) / baseline) * 100.0
}

#[derive(Debug)]
struct ComparisonReport {
    instance_id: String,
    duration_improvement: f64,
    token_improvement: f64,
    tool_efficiency_improvement: f64,
    overall_score_improvement: f64,
}

impl ComparisonReport {
    fn display(&self) {
        println!("=== Comparison Report for {} ===", self.instance_id);
        println!("Duration improvement: {:.2}%", self.duration_improvement);
        println!("Token usage improvement: {:.2}%", self.token_improvement);
        println!(
            "Tool efficiency improvement: {:.2}%",
            self.tool_efficiency_improvement
        );
        println!(
            "Overall score improvement: {:.2}%",
            self.overall_score_improvement
        );

        if self.overall_score_improvement > 0.0 {
            println!("\nResult: ML-optimized approach shows significant improvements!");
        } else {
            println!("\nResult: Baseline approach performed better in this instance.");
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example: Running comparison for a SWE-Bench instance
    let instance_id = "django__django-11099";

    // Simulate baseline execution
    println!("Running baseline benchmark for {}...", instance_id);
    let baseline_stream = generate_baseline_stream();
    let mut baseline_benchmark =
        RuvSwarmBenchmark::new(instance_id.to_string(), ExecutionMode::Baseline);
    let baseline_result = baseline_benchmark
        .process_claude_output(baseline_stream.as_bytes())
        .await?;

    println!("\nBaseline Results:");
    println!("  Duration: {:?}", baseline_result.duration);
    println!("  Total tokens: {}", baseline_result.total_tokens);
    println!("  Tool efficiency: {:.2}", baseline_result.tool_efficiency);
    println!("  Overall score: {:.2}", baseline_result.overall_score());

    // Simulate ML-optimized execution
    println!("\nRunning ML-optimized benchmark for {}...", instance_id);
    let ml_stream = generate_ml_optimized_stream();
    let mut ml_benchmark =
        RuvSwarmBenchmark::new(instance_id.to_string(), ExecutionMode::MLOptimized);
    let ml_result = ml_benchmark
        .process_claude_output(ml_stream.as_bytes())
        .await?;

    println!("\nML-Optimized Results:");
    println!("  Duration: {:?}", ml_result.duration);
    println!("  Total tokens: {}", ml_result.total_tokens);
    println!("  Tool efficiency: {:.2}", ml_result.tool_efficiency);
    println!("  Overall score: {:.2}", ml_result.overall_score());

    // Compare results
    println!("\n");
    let comparison = compare_results(&baseline_result, &ml_result);
    comparison.display();

    // Export for database storage
    println!("\n=== SQL Insert Statements ===");
    println!("{}", baseline_result.to_sql_insert());
    println!("{}", ml_result.to_sql_insert());

    // Export training data
    let training_export = baseline_benchmark.parser.export_training_data();
    println!("\n=== Training Data Export ===");
    println!(
        "Total events captured: {}",
        training_export.metadata.event_count
    );
    training_export
        .to_json_file("benchmark_training_data.json")
        .await?;
    println!("Training data exported to: benchmark_training_data.json");

    Ok(())
}

// Simulate different execution patterns
fn generate_baseline_stream() -> String {
    // Baseline: More tool invocations, longer thinking
    r#"{"type":"message_start","message":{"id":"msg_001","model":"claude-3","role":"assistant"}}
{"type":"thinking","content":"Analyzing the Django issue...","tokens":100}
{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/django/urls.py"}}
{"type":"function_result","tool_use_id":"t1","content":"File contents...","is_error":false}
{"type":"thinking","content":"Looking for more context...","tokens":80}
{"type":"tool_use","id":"t2","name":"Grep","input":{"pattern":"URL"}}
{"type":"function_result","tool_use_id":"t2","content":"Found matches...","is_error":false}
{"type":"tool_use","id":"t3","name":"Read","input":{"file_path":"/django/docs/urls.txt"}}
{"type":"function_result","tool_use_id":"t3","content":"Documentation...","is_error":false}
{"type":"thinking","content":"Found the issue, preparing fix...","tokens":120}
{"type":"tool_use","id":"t4","name":"Edit","input":{"file_path":"/django/docs/urls.txt"}}
{"type":"function_result","tool_use_id":"t4","content":"Edit successful","is_error":false}
{"type":"tool_use","id":"t5","name":"Read","input":{"file_path":"/django/docs/urls.txt"}}
{"type":"function_result","tool_use_id":"t5","content":"Verification...","is_error":false}
{"type":"usage","input_tokens":500,"output_tokens":800,"total_tokens":1300}
{"type":"message_stop","stop_reason":"end_turn"}
"#
    .to_string()
}

fn generate_ml_optimized_stream() -> String {
    // ML-optimized: Fewer tools, more efficient thinking
    r#"{"type":"message_start","message":{"id":"msg_002","model":"claude-3","role":"assistant"}}
{"type":"thinking","content":"Django URL documentation issue - direct path identified","tokens":50}
{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/django/docs/urls.txt"}}
{"type":"function_result","tool_use_id":"t1","content":"Documentation...","is_error":false}
{"type":"thinking","content":"Issue confirmed, applying fix...","tokens":40}
{"type":"tool_use","id":"t2","name":"Edit","input":{"file_path":"/django/docs/urls.txt"}}
{"type":"function_result","tool_use_id":"t2","content":"Edit successful","is_error":false}
{"type":"usage","input_tokens":200,"output_tokens":350,"total_tokens":550}
{"type":"message_stop","stop_reason":"end_turn"}
"#
    .to_string()
}
