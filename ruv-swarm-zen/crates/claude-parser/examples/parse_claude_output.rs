//! Example of parsing Claude Code CLI output stream

use claude_parser::{ClaudeStreamEvent, ClaudeStreamParser};
use std::io::Cursor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example Claude stream output (typically from `claude ... --output-format stream-json`)
    let sample_stream = r#"{"type":"message_start","message":{"id":"msg_01234","model":"claude-3-opus-20240229","role":"assistant"}}
{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
{"type":"thinking","content":"Let me analyze the user's request...","tokens":15}
{"type":"tool_use","id":"tool_001","name":"Read","input":{"file_path":"/src/main.rs"},"timestamp":"2024-01-15T10:00:00Z"}
{"type":"function_result","tool_use_id":"tool_001","content":"File contents...","is_error":false}
{"type":"thinking","content":"Now I need to make the requested changes...","tokens":25}
{"type":"tool_use","id":"tool_002","name":"Edit","input":{"file_path":"/src/main.rs","old_string":"fn main()","new_string":"async fn main()"},"timestamp":"2024-01-15T10:00:05Z"}
{"type":"function_result","tool_use_id":"tool_002","content":"Edit successful","is_error":false}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I've updated the main function to be async as requested."}}
{"type":"error","error_type":"NetworkError","message":"Temporary connection issue","recoverable":true,"timestamp":"2024-01-15T10:00:10Z"}
{"type":"thinking","content":"Retrying after network error...","tokens":10}
{"type":"tool_use","id":"tool_003","name":"Write","input":{"file_path":"/src/lib.rs","content":"// New file content"},"timestamp":"2024-01-15T10:00:15Z"}
{"type":"function_result","tool_use_id":"tool_003","content":"File written","is_error":false}
{"type":"usage","input_tokens":150,"output_tokens":250,"total_tokens":400}
{"type":"message_stop","stop_reason":"end_turn"}
"#;

    // Create parser
    let mut parser = ClaudeStreamParser::new();

    // Parse the stream
    println!("Parsing Claude stream output...\n");
    let cursor = Cursor::new(sample_stream.as_bytes());
    let metrics = parser.parse_stream(cursor).await?;

    // Display performance metrics
    println!("=== Performance Metrics ===");
    println!("Total duration: {:?}", metrics.total_duration);
    println!("Time to first output: {:?}", metrics.time_to_first_output);

    // Tool usage analysis
    println!("\n=== Tool Usage ===");
    for (tool_name, tool_metrics) in &metrics.tool_invocations {
        println!("Tool: {}", tool_name);
        println!("  - Invocations: {}", tool_metrics.invocation_count);
        println!(
            "  - Success rate: {:.2}%",
            tool_metrics.success_rate * 100.0
        );
    }

    // Thinking pattern analysis
    println!("\n=== Thinking Patterns ===");
    println!(
        "Total sequences: {}",
        metrics.thinking_metrics.total_sequences
    );
    println!("Total tokens: {}", metrics.thinking_metrics.total_tokens);
    println!(
        "Average tokens per sequence: {:.1}",
        metrics.thinking_metrics.average_tokens_per_sequence
    );

    for (i, pattern) in metrics
        .thinking_metrics
        .thinking_patterns
        .iter()
        .enumerate()
    {
        println!("\nSequence {}: {} tokens", i + 1, pattern.token_count);
        println!("  Preview: {}", pattern.content_preview);
    }

    // Error analysis
    println!("\n=== Error Metrics ===");
    println!("Total errors: {}", metrics.error_metrics.total_errors);
    println!(
        "Recoverable errors: {}",
        metrics.error_metrics.recoverable_errors
    );
    println!(
        "Recovery success rate: {:.2}%",
        metrics.error_metrics.recovery_success_rate * 100.0
    );

    for (error_type, count) in &metrics.error_metrics.error_types {
        println!("  - {}: {} occurrences", error_type, count);
    }

    // Token usage
    println!("\n=== Token Usage ===");
    println!("Input tokens: {}", metrics.token_usage.input_tokens);
    println!("Output tokens: {}", metrics.token_usage.output_tokens);
    println!("Total tokens: {}", metrics.token_usage.total_tokens);
    println!(
        "Tokens per second: {:.2}",
        metrics.token_usage.tokens_per_second
    );

    // Event timeline
    println!("\n=== Event Timeline ===");
    for event in metrics.event_timeline.iter().take(5) {
        println!("{:>6}ms: {}", event.relative_time_ms, event.event_type);
    }
    if metrics.event_timeline.len() > 5 {
        println!("... and {} more events", metrics.event_timeline.len() - 5);
    }

    // Export training data
    println!("\n=== Exporting Training Data ===");
    let export = parser.export_training_data();

    // Create temp directory for output
    let temp_dir = std::env::temp_dir();
    let json_path = temp_dir.join("claude_training_data.json");
    let jsonl_path = temp_dir.join("claude_training_data.jsonl");

    export.to_json_file(json_path.to_str().unwrap()).await?;
    export.to_jsonl_file(jsonl_path.to_str().unwrap()).await?;

    println!("Exported training data:");
    println!("  - JSON: {}", json_path.display());
    println!("  - JSONL: {}", jsonl_path.display());
    println!("  - Total events: {}", export.metadata.event_count);

    Ok(())
}

// Helper function to demonstrate custom event processing
fn process_event_custom(event: &ClaudeStreamEvent) {
    match event {
        ClaudeStreamEvent::ToolUse { name, .. } => {
            println!("Tool invoked: {}", name);
        }
        ClaudeStreamEvent::Thinking { tokens, .. } => {
            println!("Thinking: {} tokens", tokens);
        }
        ClaudeStreamEvent::Error {
            error_type,
            recoverable,
            ..
        } => {
            println!("Error: {} (recoverable: {})", error_type, recoverable);
        }
        _ => {}
    }
}
