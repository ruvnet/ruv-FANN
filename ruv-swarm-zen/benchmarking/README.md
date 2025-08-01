# RUV-SWARM Benchmarking Framework

A comprehensive benchmarking framework for evaluating Claude Code CLI performance with ML-optimized swarm intelligence against baseline implementations.

## Features

### Core Components

1. **BenchmarkExecutor**: Runs Claude Code CLI commands with stream-json output parsing
2. **MetricsCollector**: Gathers detailed performance metrics including:
   - Task completion time
   - Tool invocation patterns
   - Thinking sequences and token counts
   - Error recovery tracking
   - Resource usage (CPU, memory, network)
   - Code quality metrics

3. **SQLite Storage**: Persistent storage with comprehensive schema for:
   - Benchmark runs and configurations
   - Stream events from Claude
   - Tool invocations and thinking sequences
   - Performance metrics and comparisons
   - SWE-Bench specific results

4. **Comparison Framework**: Statistical analysis for before/after comparison:
   - Speed improvements
   - Quality improvements
   - Resource efficiency
   - Statistical significance testing (p-values, confidence intervals)
   - Cohen's d effect size calculation

5. **Real-time Monitoring**: Web-based dashboard for live monitoring:
   - WebSocket-based real-time updates
   - Performance charts and trends
   - Event stream visualization
   - Active run management

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruv-swarm-benchmarking = { path = "../benchmarking" }
```

## Usage

### Basic Example

```rust
use ruv_swarm_benchmarking::{
    BenchmarkingFramework, BenchmarkConfig, BenchmarkScenario, Difficulty
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Configure benchmarking
    let config = BenchmarkConfig {
        database_path: PathBuf::from("benchmarks.db"),
        enable_real_time_monitoring: true,
        monitor_port: 8080,
        claude_executable: PathBuf::from("claude"),
        ..Default::default()
    };
    
    // Create framework
    let mut framework = BenchmarkingFramework::new(config).await?;
    
    // Define scenarios
    let scenarios = vec![
        BenchmarkScenario {
            name: "test_scenario".to_string(),
            instance_id: "test-001".to_string(),
            repository: "test/repo".to_string(),
            issue_description: "Test issue".to_string(),
            difficulty: Difficulty::Easy,
            claude_command: "solve test problem".to_string(),
            expected_files_modified: vec![],
            validation_tests: vec![],
        },
    ];
    
    // Run benchmarks
    let report = framework.run_benchmark_suite(scenarios).await?;
    
    // Access results
    println!("Average improvement: {:.1}%", 
        report.summary.average_improvement * 100.0);
    
    Ok(())
}
```

### SWE-Bench Integration

```rust
use ruv_swarm_benchmarking::claude_executor::ClaudeCommandExecutor;

let executor = ClaudeCommandExecutor::new(PathBuf::from("claude"));

// Execute SWE-Bench instance
let result = executor.execute_swe_bench(
    "django__django-11099",
    ExecutionMode::MLOptimized,
    "run-001"
).await?;

println!("Tests passed: {}", result.tests_passed);
println!("Tool invocations: {}", result.tool_invocations);
```

### Real-time Monitoring

Start the monitoring server:

```rust
let monitor = RealTimeMonitor::new(8080).await?;
tokio::spawn(async move {
    monitor.start_server().await
});
```

Then access the dashboard at `http://localhost:8080`

### Stream Event Processing

```rust
use ruv_swarm_benchmarking::stream_parser::{
    StreamJSONParser, ClaudeStreamEvent
};

let parser = StreamJSONParser::new();

// Process Claude's stream-json output
let metrics = parser.process_stream(reader, "run-001").await?;

println!("Total tool invocations: {}", metrics.tool_invocations.len());
println!("Thinking sequences: {}", metrics.thinking_sequences.len());
```

## Database Schema

The framework uses SQLite with the following main tables:

- `benchmark_runs`: Core run information
- `stream_events`: All Claude stream events
- `tool_invocations`: Tool usage details
- `thinking_sequences`: Thinking patterns
- `performance_metrics`: Numeric metrics
- `comparison_results`: Before/after comparisons
- `swe_bench_results`: SWE-Bench specific data

## Command Line Usage

### Running benchmarks with Claude CLI

```bash
# Baseline execution
claude "solve SWE-bench instance django__django-11099" \
  -p --dangerously-skip-permissions \
  --output-format stream-json \
  --verbose | ./benchmark-collector

# ML-optimized execution
./claude-flow swarm "solve SWE-bench instance django__django-11099" \
  --strategy development \
  --mode ml-optimized \
  --monitor
```

### Batch execution

```bash
# Run multiple scenarios
./benchmark-runner --scenarios scenarios.json \
  --parallel 4 \
  --output results/
```

## Metrics Collected

### Performance Metrics
- Task completion time
- Time to first output
- Agent coordination overhead
- ML inference time

### Code Quality Metrics
- Overall quality score
- Readability
- Modularity
- Security score
- Test coverage

### Resource Usage
- CPU usage (average, peak, p95, p99)
- Memory usage
- Network bandwidth
- Disk I/O

### Swarm Coordination
- Agent utilization
- Communication efficiency
- Task distribution balance
- Conflict resolution time

## Statistical Analysis

The framework provides comprehensive statistical analysis:

- **Mean and median improvements**
- **Standard deviation**
- **Cohen's d effect size**
- **Student's t-test for significance**
- **95% confidence intervals**

## License

MIT OR Apache-2.0