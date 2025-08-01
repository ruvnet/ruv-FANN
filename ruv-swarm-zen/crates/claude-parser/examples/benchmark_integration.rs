//! Example of integrating Claude parser with benchmarking system

use claude_parser::{ClaudeStreamParser, PerformanceMetrics};
use std::process::Stdio;
use tokio::io::BufReader;
use tokio::process::Command;

/// Validates that an instance ID contains only safe characters
fn validate_instance_id(instance_id: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Only allow alphanumeric characters, hyphens, underscores, and double underscores
    if !instance_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Err("Invalid instance ID: only alphanumeric characters, hyphens, and underscores are allowed".into());
    }
    
    // Validate length to prevent buffer overflow attacks
    if instance_id.len() > 256 {
        return Err("Invalid instance ID: maximum length is 256 characters".into());
    }
    
    // Ensure it doesn't start with special characters that could be interpreted as flags
    if instance_id.starts_with('-') {
        return Err("Invalid instance ID: cannot start with hyphen".into());
    }
    
    Ok(instance_id.to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example: Running Claude on a SWE-Bench instance
    let instance_id = "django__django-11099";
    
    // Validate instance ID to prevent command injection
    let safe_instance_id = validate_instance_id(instance_id)?;

    println!("Executing Claude on SWE-Bench instance: {}", safe_instance_id);

    // Launch Claude process with stream-json output
    // SECURITY: Using separate arguments prevents command injection (CVE-2024-24576)
    let mut child = Command::new("claude")
        .arg("solve")
        .arg("SWE-bench")
        .arg("instance")
        .arg(&safe_instance_id)
        .arg("-p")
        // NOTE: --dangerously-skip-permissions removed for security
        // Users must explicitly add if needed
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    // Get stdout for parsing
    let stdout = child.stdout.take().expect("Failed to get stdout");
    let reader = BufReader::new(stdout);

    // Create parser and process stream
    let mut parser = ClaudeStreamParser::new();

    println!("Parsing Claude output stream...\n");
    let start = std::time::Instant::now();

    // Parse the stream
    let metrics = parser.parse_stream(reader).await?;

    // Wait for process to complete
    let exit_status = child.wait().await?;
    println!("Claude process exited with status: {}", exit_status);

    // Display comprehensive metrics
    display_benchmark_results(&metrics, instance_id, start.elapsed());

    // Export for further analysis
    export_benchmark_data(&parser, instance_id).await?;

    // Compare with baseline (if available)
    if let Ok(baseline) = load_baseline_metrics(instance_id).await {
        compare_performance(&baseline, &metrics);
    }

    Ok(())
}

fn display_benchmark_results(
    metrics: &PerformanceMetrics,
    instance_id: &str,
    wall_time: std::time::Duration,
) {
    println!("=== Benchmark Results for {} ===", instance_id);
    println!("Wall clock time: {:?}", wall_time);
    println!("Stream duration: {:?}", metrics.total_duration);

    // Performance summary
    println!("\n--- Performance Summary ---");
    println!("Time to first output: {:?}", metrics.time_to_first_output);
    println!("Total tokens: {}", metrics.token_usage.total_tokens);
    println!(
        "Tokens/second: {:.2}",
        metrics.token_usage.tokens_per_second
    );

    // Tool efficiency
    println!("\n--- Tool Efficiency ---");
    let total_tool_calls: u64 = metrics
        .tool_invocations
        .values()
        .map(|t| t.invocation_count)
        .sum();
    println!("Total tool invocations: {}", total_tool_calls);

    for (tool, tool_metrics) in &metrics.tool_invocations {
        let percentage = (tool_metrics.invocation_count as f64 / total_tool_calls as f64) * 100.0;
        println!(
            "  {}: {} calls ({:.1}%)",
            tool, tool_metrics.invocation_count, percentage
        );
    }

    // Thinking analysis
    println!("\n--- Thinking Analysis ---");
    println!(
        "Thinking sequences: {}",
        metrics.thinking_metrics.total_sequences
    );
    println!(
        "Total thinking tokens: {}",
        metrics.thinking_metrics.total_tokens
    );
    if metrics.thinking_metrics.total_sequences > 0 {
        let avg_duration = metrics.thinking_metrics.total_duration.as_millis() as f64
            / metrics.thinking_metrics.total_sequences as f64;
        println!("Average thinking duration: {:.2}ms", avg_duration);
    }

    // Error handling
    println!("\n--- Error Handling ---");
    if metrics.error_metrics.total_errors > 0 {
        println!("Total errors: {}", metrics.error_metrics.total_errors);
        println!(
            "Recovery rate: {:.2}%",
            metrics.error_metrics.recovery_success_rate * 100.0
        );
        for (error_type, count) in &metrics.error_metrics.error_types {
            println!("  {}: {}", error_type, count);
        }
    } else {
        println!("No errors encountered");
    }
}

async fn export_benchmark_data(
    parser: &ClaudeStreamParser,
    instance_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let export = parser.export_training_data();

    // Create benchmark-specific filename
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("benchmark_{}_{}.json", instance_id, timestamp);

    export.to_json_file(&filename).await?;
    println!("\n--- Export Complete ---");
    println!("Benchmark data exported to: {}", filename);
    println!("Total events captured: {}", export.metadata.event_count);

    Ok(())
}

async fn load_baseline_metrics(
    _instance_id: &str,
) -> Result<PerformanceMetrics, Box<dyn std::error::Error>> {
    // In a real implementation, this would load from a database or file
    // For demo purposes, we'll return an error to skip comparison
    Err("No baseline metrics available".into())
}

fn compare_performance(baseline: &PerformanceMetrics, current: &PerformanceMetrics) {
    println!("\n=== Performance Comparison ===");

    // Duration comparison
    let duration_improvement = (baseline.total_duration.as_millis() as f64
        - current.total_duration.as_millis() as f64)
        / baseline.total_duration.as_millis() as f64
        * 100.0;

    println!(
        "Duration: {:?} -> {:?} ({:+.1}%)",
        baseline.total_duration, current.total_duration, duration_improvement
    );

    // Token efficiency
    let token_improvement = (baseline.token_usage.total_tokens as f64
        - current.token_usage.total_tokens as f64)
        / baseline.token_usage.total_tokens as f64
        * 100.0;

    println!(
        "Total tokens: {} -> {} ({:+.1}%)",
        baseline.token_usage.total_tokens, current.token_usage.total_tokens, token_improvement
    );

    // Tool usage efficiency
    let baseline_tool_calls: u64 = baseline
        .tool_invocations
        .values()
        .map(|t| t.invocation_count)
        .sum();
    let current_tool_calls: u64 = current
        .tool_invocations
        .values()
        .map(|t| t.invocation_count)
        .sum();

    let tool_improvement = (baseline_tool_calls as f64 - current_tool_calls as f64)
        / baseline_tool_calls as f64
        * 100.0;

    println!(
        "Tool invocations: {} -> {} ({:+.1}%)",
        baseline_tool_calls, current_tool_calls, tool_improvement
    );

    // Summary
    println!("\n--- Summary ---");
    if duration_improvement > 0.0 && token_improvement > 0.0 {
        println!("Performance improved across all metrics!");
    } else if duration_improvement > 0.0 || token_improvement > 0.0 {
        println!("Mixed performance results - some improvements, some regressions");
    } else {
        println!("Performance regression detected - investigation needed");
    }
}
