//! Example of real-time monitoring of Claude stream events

use claude_parser::ClaudeStreamEvent;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
struct RealTimeStats {
    tool_calls: Arc<DashMap<String, u64>>,
    thinking_tokens: Arc<DashMap<String, u64>>,
    errors: Arc<DashMap<String, u64>>,
    start_time: Instant,
}

impl RealTimeStats {
    fn new() -> Self {
        Self {
            tool_calls: Arc::new(DashMap::new()),
            thinking_tokens: Arc::new(DashMap::new()),
            errors: Arc::new(DashMap::new()),
            start_time: Instant::now(),
        }
    }

    fn update_tool_call(&self, tool_name: String) {
        *self.tool_calls.entry(tool_name).or_insert(0) += 1;
    }

    fn update_thinking(&self, tokens: usize) {
        *self.thinking_tokens.entry("total".to_string()).or_insert(0) += tokens as u64;
        *self
            .thinking_tokens
            .entry("sequences".to_string())
            .or_insert(0) += 1;
    }

    fn update_error(&self, error_type: String) {
        *self.errors.entry(error_type).or_insert(0) += 1;
    }

    fn display(&self) {
        let elapsed = self.start_time.elapsed();

        println!("\n=== Real-Time Stats ({}s) ===", elapsed.as_secs());

        // Tool usage
        println!("\nTool Usage:");
        for entry in self.tool_calls.iter() {
            println!("  {}: {} calls", entry.key(), entry.value());
        }

        // Thinking stats
        if let Some(total) = self.thinking_tokens.get("total") {
            if let Some(sequences) = self.thinking_tokens.get("sequences") {
                let avg = *total.value() as f64 / *sequences.value() as f64;
                println!("\nThinking:");
                println!("  Total tokens: {}", total.value());
                println!("  Sequences: {}", sequences.value());
                println!("  Average tokens/sequence: {:.1}", avg);
            }
        }

        // Error stats
        if !self.errors.is_empty() {
            println!("\nErrors:");
            for entry in self.errors.iter() {
                println!("  {}: {} occurrences", entry.key(), entry.value());
            }
        }

        println!("\n{}", "-".repeat(40));
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulated Claude stream for demonstration
    let sample_stream = generate_sample_stream();

    // Create channels for real-time processing
    let (event_tx, mut event_rx) = mpsc::channel::<ClaudeStreamEvent>(100);

    // Create real-time stats tracker
    let stats = RealTimeStats::new();
    let stats_clone = stats.clone();

    // Spawn stats display task
    let display_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));
        loop {
            interval.tick().await;
            stats_clone.display();
        }
    });

    // Spawn event processor
    let stats_for_processor = stats.clone();
    let processor_handle = tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            process_event_realtime(&event, &stats_for_processor);
        }
    });

    // Parse stream with real-time event emission
    println!("Starting real-time monitoring of Claude stream...\n");

    let reader = BufReader::new(sample_stream.as_bytes());
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        // Parse event
        if let Ok(event) = serde_json::from_str::<ClaudeStreamEvent>(&line) {
            // Display immediate feedback for certain events
            match &event {
                ClaudeStreamEvent::ToolUse { name, .. } => {
                    println!("[TOOL] {} invoked", name);
                }
                ClaudeStreamEvent::Thinking { tokens, .. } => {
                    println!("[THINK] Processing {} tokens...", tokens);
                }
                ClaudeStreamEvent::Error { error_type, .. } => {
                    println!("[ERROR] {} occurred", error_type);
                }
                _ => {}
            }

            // Send to processor
            event_tx.send(event).await?;
        }

        // Simulate real-time delay
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Cleanup
    drop(event_tx);
    let _ = processor_handle.await;
    display_handle.abort();

    println!("\n=== Final Statistics ===");
    stats.display();

    Ok(())
}

fn process_event_realtime(event: &ClaudeStreamEvent, stats: &RealTimeStats) {
    match event {
        ClaudeStreamEvent::ToolUse { name, .. } => {
            stats.update_tool_call(name.clone());
        }
        ClaudeStreamEvent::Thinking { tokens, .. } => {
            stats.update_thinking(*tokens);
        }
        ClaudeStreamEvent::Error { error_type, .. } => {
            stats.update_error(error_type.clone());
        }
        _ => {}
    }
}

fn generate_sample_stream() -> String {
    // Simulate a longer stream for real-time monitoring
    r#"{"type":"message_start","message":{"id":"msg_001","model":"claude-3","role":"assistant"}}
{"type":"thinking","content":"Analyzing the request...","tokens":20}
{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/app/src/main.rs"}}
{"type":"function_result","tool_use_id":"t1","content":"File contents...","is_error":false}
{"type":"thinking","content":"Found the issue, preparing fix...","tokens":35}
{"type":"tool_use","id":"t2","name":"Edit","input":{"file_path":"/app/src/main.rs","old_string":"old","new_string":"new"}}
{"type":"function_result","tool_use_id":"t2","content":"Edit successful","is_error":false}
{"type":"thinking","content":"Checking for related files...","tokens":25}
{"type":"tool_use","id":"t3","name":"Grep","input":{"pattern":"import.*main","include":"*.rs"}}
{"type":"function_result","tool_use_id":"t3","content":"Found 3 matches","is_error":false}
{"type":"error","error_type":"NetworkTimeout","message":"Request timeout","recoverable":true}
{"type":"thinking","content":"Retrying after network error...","tokens":15}
{"type":"tool_use","id":"t4","name":"Read","input":{"file_path":"/app/src/lib.rs"}}
{"type":"function_result","tool_use_id":"t4","content":"File contents...","is_error":false}
{"type":"tool_use","id":"t5","name":"Edit","input":{"file_path":"/app/src/lib.rs","old_string":"use","new_string":"use crate"}}
{"type":"function_result","tool_use_id":"t5","content":"Edit successful","is_error":false}
{"type":"thinking","content":"Running tests to verify changes...","tokens":30}
{"type":"tool_use","id":"t6","name":"Bash","input":{"command":"cargo test"}}
{"type":"function_result","tool_use_id":"t6","content":"All tests passed","is_error":false}
{"type":"thinking","content":"Preparing final response...","tokens":20}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Successfully updated the code."}}
{"type":"usage","input_tokens":250,"output_tokens":350,"total_tokens":600}
{"type":"message_stop","stop_reason":"end_turn"}
"#.to_string()
}
