# SWE-Bench Adapter

A high-performance adapter for integrating SWE-Bench evaluation with the ruv-swarm orchestration system and Claude Code CLI.

## Features

- **Instance Loading**: Load and categorize SWE-Bench instances with automatic difficulty assessment
- **Prompt Generation**: Generate optimized prompts for Claude Code CLI based on instance complexity
- **Patch Evaluation**: Evaluate generated patches against test requirements in sandboxed environments
- **Performance Benchmarking**: Comprehensive performance metrics and comparative analysis
- **Stream Parsing**: Real-time metrics collection from Claude Code CLI output streams

## Architecture

The adapter consists of five main components:

### 1. Instance Loader (`loader.rs`)
- Loads SWE-Bench instances from local storage or remote repositories
- Automatically categorizes instances by difficulty (Easy, Medium, Hard, Expert)
- Supports filtering and batch loading
- Caches instances for improved performance

### 2. Prompt Generator (`prompts.rs`)
- Generates context-aware prompts for Claude Code CLI
- Adapts prompt complexity based on instance difficulty
- Supports multiple prompt templates (Simple, Standard, Detailed, Expert)
- Manages token limits and prompt truncation

### 3. Patch Evaluator (`evaluation.rs`)
- Applies patches in sandboxed environments
- Runs test suites with configurable timeouts
- Validates patch quality and identifies common issues
- Compares patches for similarity analysis

### 4. Performance Benchmarking (`benchmarking.rs`)
- Measures execution time, memory usage, and resource consumption
- Supports comparative benchmarking between solutions
- Generates detailed performance reports
- Integrates with Prometheus metrics

### 5. Stream Parser (`stream_parser.rs`)
- Parses Claude Code CLI output in real-time
- Extracts metrics: token usage, tool calls, file operations
- Identifies errors and warnings
- Supports concurrent stream analysis

## Usage

### Basic Evaluation

```rust
use swe_bench_adapter::{SWEBenchAdapter, SWEBenchConfig};
use ruv_swarm_agents::Agent;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize adapter
    let config = SWEBenchConfig::default();
    let adapter = SWEBenchAdapter::new(config).await?;
    
    // Create an agent
    let agent = Agent::new(
        AgentId::new(),
        "claude-coder",
        vec![AgentCapability::CodeGeneration],
    );
    
    // Evaluate a single instance
    let report = adapter.evaluate_instance("django-12345", &agent).await?;
    
    println!("Evaluation completed: {}", report.passed);
    println!("Execution time: {:?}", report.execution_time);
    
    Ok(())
}
```

### Batch Evaluation

```rust
// Evaluate multiple instances in parallel
let instance_ids = vec![
    "django-12345".to_string(),
    "flask-67890".to_string(),
    "numpy-11111".to_string(),
];

let agents = vec![agent1, agent2, agent3];

let batch_report = adapter.evaluate_batch(
    instance_ids,
    agents,
    true, // parallel execution
).await?;

println!("Success rate: {:.2}%", batch_report.success_rate * 100.0);
```

### Custom Prompt Generation

```rust
use swe_bench_adapter::{ClaudePromptGenerator, PromptConfig};

let config = PromptConfig {
    max_tokens: 6000,
    include_test_hints: true,
    include_context_files: true,
    template_style: PromptStyle::Expert,
};

let generator = ClaudePromptGenerator::new(config);
let prompt = generator.generate_prompt(&instance)?;

println!("Generated prompt ({} tokens):\n{}", prompt.token_count, prompt.content);
```

### Stream Metrics Collection

```rust
use swe_bench_adapter::stream_parser::{MetricsCollector, StreamAnalyzer};

// Single stream collection
let mut collector = MetricsCollector::new();
let metrics = collector.parse_stream(&claude_output)?;

println!("Tool calls: {}", metrics.tool_calls);
println!("File operations: {:?}", metrics.file_operations);

// Multi-stream analysis
let mut analyzer = StreamAnalyzer::new();
let mut rx = analyzer.add_stream("claude-1".to_string());

// Process output as it arrives
analyzer.process("claude-1", &output_chunk)?;

// Get global summary
let summary = analyzer.get_global_summary();
println!("Active streams: {}", summary.active_streams);
```

## Configuration

### SWEBenchConfig

```rust
let config = SWEBenchConfig {
    instances_path: PathBuf::from("./swe-bench-instances"),
    memory_path: PathBuf::from("./swe-bench-memory"),
    prompt_config: PromptConfig {
        max_tokens: 4000,
        include_test_hints: true,
        include_context_files: true,
        template_style: PromptStyle::ClaudeCode,
    },
    eval_config: EvalConfig {
        timeout: Duration::from_secs(300),
        test_command: "pytest".to_string(),
        sandbox_enabled: true,
        max_retries: 3,
    },
    benchmark_config: BenchmarkConfig {
        iterations: 10,
        warm_up: 3,
        measure_memory: true,
        profile_enabled: false,
    },
};
```

## Performance

The adapter is optimized for high-throughput evaluation:

- **Parallel Processing**: Evaluate multiple instances concurrently
- **Caching**: Instance caching reduces redundant I/O operations
- **Stream Processing**: Efficient parsing of large output streams
- **Memory Management**: Configurable memory limits and garbage collection

Benchmark results on standard hardware:
- Instance loading: ~5ms per instance
- Prompt generation: ~10ms for expert prompts
- Stream parsing: ~1ms per KB of output
- Full evaluation: ~30s per instance (including test execution)

## Integration with Claude Code CLI

The adapter generates prompts specifically optimized for Claude Code CLI:

1. **Context-Aware**: Includes repository information, issue details, and test requirements
2. **Difficulty-Adapted**: Adjusts prompt complexity based on instance difficulty
3. **Token-Optimized**: Manages token limits while preserving essential information
4. **Tool-Friendly**: Structures prompts to encourage effective tool usage

## Error Handling

The adapter provides comprehensive error handling:

- **Patch Application Failures**: Detailed error messages with git output
- **Test Execution Timeouts**: Configurable timeouts with graceful termination
- **Stream Parsing Errors**: Resilient parsing with error recovery
- **Sandbox Failures**: Fallback to non-sandboxed execution when necessary

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test`
2. Benchmarks show no regression: `cargo bench`
3. Code follows Rust formatting: `cargo fmt`
4. No clippy warnings: `cargo clippy`

## License

This project is licensed under the MIT OR Apache-2.0 license.