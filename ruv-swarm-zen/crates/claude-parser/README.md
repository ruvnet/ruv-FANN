# Claude Parser

A high-performance stream-JSON parser for Claude Code CLI output, designed for benchmarking and training data collection.

## Features

- **Stream Event Parsing**: Supports all Claude event types:
  - `message_start`, `message_stop`
  - `content_block_start`, `content_block_delta`, `content_block_stop`
  - `tool_use`, `function_result`
  - `thinking`
  - `error`
  - `usage`

- **Performance Metrics Extraction**:
  - Timing measurements (total duration, time to first output)
  - Token usage tracking (input, output, total)
  - Tool invocation statistics
  - Thinking sequence analysis
  - Error and recovery patterns

- **Error Handling**: Robust error recovery with detailed tracking of error types and recovery strategies

- **Export Functionality**: Export parsed data for training in JSON or JSONL format

## Usage

```rust
use claude_parser::ClaudeStreamParser;
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create parser
    let mut parser = ClaudeStreamParser::new();
    
    // Parse stream from file
    let file = File::open("claude_output.jsonl").await?;
    let metrics = parser.parse_stream(file).await?;
    
    // Access performance metrics
    println!("Total duration: {:?}", metrics.total_duration);
    println!("Tool invocations: {}", metrics.tool_invocations.len());
    println!("Total tokens: {}", metrics.token_usage.total_tokens);
    
    // Export training data
    let export = parser.export_training_data();
    export.to_json_file("training_data.json").await?;
    
    Ok(())
}
```

## Metrics Collected

### Tool Metrics
- Invocation count per tool
- Average duration
- Success rate
- Parameter sizes

### Thinking Metrics
- Total sequences
- Token count per sequence
- Duration analysis
- Pattern recognition

### Error Metrics
- Error types and frequencies
- Recovery success rates
- Recovery strategies used

### Timeline
- Chronological event sequence
- Relative timing information
- Event correlations

## Integration with Benchmarking

This parser is designed to work with the RUV-SWARM benchmarking system for:
- SWE-Bench instance evaluation
- Real-time performance monitoring
- Comparative analysis between baseline and ML-optimized approaches
- Training data collection for model improvement

## License

Licensed under MIT OR Apache-2.0