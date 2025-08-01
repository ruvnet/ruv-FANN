//! Stream JSON parser for Claude Code CLI output

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::Mutex;
use tracing::warn;

use crate::metrics::{ErrorRecovery, PerformanceMetrics, ThinkingSequence, ToolInvocation};
use crate::storage::BenchmarkStorage;

/// Claude stream event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageInfo },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: ContentDelta },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        timestamp: Option<String>,
    },
    #[serde(rename = "thinking")]
    Thinking { content: String, tokens: usize },
    #[serde(rename = "error")]
    Error {
        error_type: String,
        message: String,
        recoverable: bool,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaContent },
    #[serde(rename = "message_stop")]
    MessageStop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageInfo {
    pub id: String,
    pub model: String,
    pub role: String,
    pub content: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaContent {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

/// Stream metrics collector
pub struct StreamMetricsCollector {
    start_time: Instant,
    events: Vec<(Instant, ClaudeStreamEvent)>,
    tool_usage: HashMap<String, ToolUsageStats>,
    thinking_stats: ThinkingStats,
    error_recovery: Vec<ErrorRecoveryEvent>,
    first_output_time: Option<Instant>,
}

impl Default for StreamMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamMetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            events: Vec::new(),
            tool_usage: HashMap::new(),
            thinking_stats: ThinkingStats::default(),
            error_recovery: Vec::new(),
            first_output_time: None,
        }
    }

    /// Process a stream event
    pub async fn process_event(&mut self, event: ClaudeStreamEvent) {
        let timestamp = Instant::now();

        // Record first output
        if self.first_output_time.is_none() {
            match &event {
                ClaudeStreamEvent::ContentBlockStart { .. } | ClaudeStreamEvent::ToolUse { .. } => {
                    self.first_output_time = Some(timestamp);
                }
                _ => {}
            }
        }

        match &event {
            ClaudeStreamEvent::ToolUse { name, .. } => {
                self.tool_usage
                    .entry(name.clone())
                    .or_default()
                    .record_invocation(timestamp);
            }
            ClaudeStreamEvent::Thinking { tokens, .. } => {
                self.thinking_stats.record_thinking(*tokens, timestamp);
            }
            ClaudeStreamEvent::Error { recoverable, .. } => {
                if *recoverable {
                    self.error_recovery.push(ErrorRecoveryEvent {
                        timestamp,
                        recovery_time: None,
                    });
                }
            }
            _ => {}
        }

        self.events.push((timestamp, event));
    }

    /// Finalize and return metrics
    pub fn finalize(&self) -> PerformanceMetrics {
        let total_duration = self.start_time.elapsed();

        let mut metrics = PerformanceMetrics::default();
        metrics.task_completion_time = total_duration;
        metrics.time_to_first_output = self
            .first_output_time
            .map(|t| t.duration_since(self.start_time))
            .unwrap_or_default();

        // Extract tool invocations
        metrics.tool_invocations = self.analyze_tool_patterns();

        // Extract thinking sequences
        metrics.thinking_sequences = self.analyze_thinking_patterns();

        // Extract error recoveries
        metrics.error_recoveries = self.analyze_error_recovery();

        // Calculate swarm metrics
        metrics.swarm_metrics.messages_passed = self.events.len() as u32;

        metrics
    }

    fn analyze_tool_patterns(&self) -> Vec<ToolInvocation> {
        let mut invocations = Vec::new();
        let mut current_tool: Option<(String, Instant)> = None;

        for (timestamp, event) in &self.events {
            if let ClaudeStreamEvent::ToolUse { name, .. } = event {
                // End previous tool if any
                if let Some((prev_name, prev_start)) = current_tool.take() {
                    invocations.push(ToolInvocation {
                        tool_name: prev_name,
                        timestamp: Utc::now()
                            - chrono::Duration::from_std(
                                self.start_time.elapsed() - prev_start.elapsed(),
                            )
                            .unwrap_or_default(),
                        duration: timestamp.duration_since(prev_start),
                        parameters: serde_json::Value::Null,
                        result_size: 0,
                        success: true,
                        error_message: None,
                    });
                }

                current_tool = Some((name.clone(), *timestamp));
            }
        }

        // Handle last tool
        if let Some((name, start)) = current_tool {
            invocations.push(ToolInvocation {
                tool_name: name,
                timestamp: Utc::now()
                    - chrono::Duration::from_std(self.start_time.elapsed() - start.elapsed())
                        .unwrap_or_default(),
                duration: self
                    .start_time
                    .elapsed()
                    .saturating_sub(start.duration_since(self.start_time)),
                parameters: serde_json::Value::Null,
                result_size: 0,
                success: true,
                error_message: None,
            });
        }

        invocations
    }

    fn analyze_thinking_patterns(&self) -> Vec<ThinkingSequence> {
        let mut sequences = Vec::new();
        let mut current_thinking: Option<(Instant, usize)> = None;

        for (timestamp, event) in &self.events {
            match event {
                ClaudeStreamEvent::Thinking { tokens, .. } => {
                    if let Some((_, token_count)) = &mut current_thinking {
                        *token_count += tokens;
                    } else {
                        current_thinking = Some((*timestamp, *tokens));
                    }
                }
                _ => {
                    // End thinking sequence
                    if let Some((start, tokens)) = current_thinking.take() {
                        sequences.push(ThinkingSequence {
                            start_time: Utc::now()
                                - chrono::Duration::from_std(
                                    self.start_time.elapsed() - start.elapsed(),
                                )
                                .unwrap_or_default(),
                            duration: timestamp.duration_since(start),
                            token_count: tokens,
                            decision_points: Vec::new(),
                        });
                    }
                }
            }
        }

        // Handle last thinking sequence
        if let Some((start, tokens)) = current_thinking {
            sequences.push(ThinkingSequence {
                start_time: Utc::now()
                    - chrono::Duration::from_std(self.start_time.elapsed() - start.elapsed())
                        .unwrap_or_default(),
                duration: self
                    .start_time
                    .elapsed()
                    .saturating_sub(start.duration_since(self.start_time)),
                token_count: tokens,
                decision_points: Vec::new(),
            });
        }

        sequences
    }

    fn analyze_error_recovery(&self) -> Vec<ErrorRecovery> {
        let mut recoveries = Vec::new();

        for (i, (timestamp, event)) in self.events.iter().enumerate() {
            if let ClaudeStreamEvent::Error {
                error_type,
                message,
                recoverable,
            } = event
            {
                if *recoverable {
                    // Look for recovery completion
                    let recovery_completed = self.events[i + 1..]
                        .iter()
                        .find(|(_, e)| !matches!(e, ClaudeStreamEvent::Error { .. }))
                        .map(|(t, _)| *t);

                    recoveries.push(ErrorRecovery {
                        error_time: Utc::now()
                            - chrono::Duration::from_std(
                                self.start_time.elapsed() - timestamp.elapsed(),
                            )
                            .unwrap_or_default(),
                        error_type: error_type.clone(),
                        error_message: message.clone(),
                        recovery_started: Some(
                            Utc::now()
                                - chrono::Duration::from_std(
                                    self.start_time.elapsed() - timestamp.elapsed(),
                                )
                                .unwrap_or_default(),
                        ),
                        recovery_completed: recovery_completed.map(|t| {
                            Utc::now()
                                - chrono::Duration::from_std(
                                    self.start_time.elapsed() - t.elapsed(),
                                )
                                .unwrap_or_default()
                        }),
                        recovery_strategy: Some("automatic".to_string()),
                        recovery_success: recovery_completed.is_some(),
                    });
                }
            }
        }

        recoveries
    }
}

#[derive(Debug, Default)]
struct ToolUsageStats {
    invocation_count: u32,
    first_invocation: Option<Instant>,
    last_invocation: Option<Instant>,
}

impl ToolUsageStats {
    fn new() -> Self {
        Self::default()
    }

    fn record_invocation(&mut self, timestamp: Instant) {
        self.invocation_count += 1;
        if self.first_invocation.is_none() {
            self.first_invocation = Some(timestamp);
        }
        self.last_invocation = Some(timestamp);
    }
}

#[derive(Debug, Default)]
struct ThinkingStats {
    total_sequences: u32,
    total_tokens: usize,
    total_duration: Duration,
}

impl ThinkingStats {
    fn record_thinking(&mut self, tokens: usize, _timestamp: Instant) {
        self.total_sequences += 1;
        self.total_tokens += tokens;
        // Estimate duration based on tokens (50ms per token as approximation)
        self.total_duration += Duration::from_millis(tokens as u64 * 50);
    }
}

#[derive(Debug)]
struct ErrorRecoveryEvent {
    timestamp: Instant,
    recovery_time: Option<Duration>,
}

/// Stream JSON parser
pub struct StreamJSONParser {
    collector: Arc<Mutex<StreamMetricsCollector>>,
    storage: Option<Arc<Mutex<BenchmarkStorage>>>,
}

impl Default for StreamJSONParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamJSONParser {
    pub fn new() -> Self {
        Self {
            collector: Arc::new(Mutex::new(StreamMetricsCollector::new())),
            storage: None,
        }
    }

    pub fn with_storage(mut self, storage: Arc<Mutex<BenchmarkStorage>>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Process stream from reader
    pub async fn process_stream<R: AsyncRead + Unpin>(
        &self,
        reader: R,
        run_id: &str,
    ) -> Result<PerformanceMetrics> {
        let mut lines = BufReader::new(reader).lines();
        let mut sequence_number = 0;
        let start_time = Instant::now();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<ClaudeStreamEvent>(&line) {
                Ok(event) => {
                    // Store in database if storage is available
                    if let Some(storage) = &self.storage {
                        let relative_time_ms = start_time.elapsed().as_millis() as i64;
                        let event_type = match &event {
                            ClaudeStreamEvent::MessageStart { .. } => "message_start",
                            ClaudeStreamEvent::ContentBlockStart { .. } => "content_block_start",
                            ClaudeStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
                            ClaudeStreamEvent::ToolUse { .. } => "tool_use",
                            ClaudeStreamEvent::Thinking { .. } => "thinking",
                            ClaudeStreamEvent::Error { .. } => "error",
                            _ => "other",
                        };

                        let _ = storage
                            .lock()
                            .await
                            .store_stream_event(
                                run_id,
                                event_type,
                                &serde_json::to_value(&event)?,
                                sequence_number,
                                relative_time_ms,
                            )
                            .await;
                    }

                    // Process event for metrics
                    self.collector.lock().await.process_event(event).await;
                    sequence_number += 1;
                }
                Err(e) => {
                    warn!("Failed to parse stream event: {} - Line: {}", e, line);
                }
            }
        }

        Ok(self.collector.lock().await.finalize())
    }
}

/// Event processor trait for extensibility
#[async_trait]
pub trait EventProcessor: Send + Sync {
    async fn process(&self, event: &ClaudeStreamEvent) -> Result<()>;
}

/// Tool usage processor
pub struct ToolUsageProcessor {
    metrics: Arc<Mutex<HashMap<String, u32>>>,
}

impl Default for ToolUsageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolUsageProcessor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl EventProcessor for ToolUsageProcessor {
    async fn process(&self, event: &ClaudeStreamEvent) -> Result<()> {
        if let ClaudeStreamEvent::ToolUse { name, .. } = event {
            let mut metrics = self.metrics.lock().await;
            *metrics.entry(name.clone()).or_insert(0) += 1;
        }
        Ok(())
    }
}

/// Thinking pattern analyzer
pub struct ThinkingPatternAnalyzer {
    patterns: Arc<Mutex<Vec<ThinkingPattern>>>,
}

#[derive(Debug, Clone)]
pub struct ThinkingPattern {
    pub tokens: usize,
    pub timestamp: Instant,
    pub context: String,
}

impl Default for ThinkingPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkingPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl EventProcessor for ThinkingPatternAnalyzer {
    async fn process(&self, event: &ClaudeStreamEvent) -> Result<()> {
        if let ClaudeStreamEvent::Thinking { content, tokens } = event {
            let pattern = ThinkingPattern {
                tokens: *tokens,
                timestamp: Instant::now(),
                context: content.clone(),
            };
            self.patterns.lock().await.push(pattern);
        }
        Ok(())
    }
}

/// Error recovery tracker
pub struct ErrorRecoveryTracker {
    events: Arc<Mutex<Vec<ErrorRecoveryTracking>>>,
}

#[derive(Debug, Clone)]
struct ErrorRecoveryTracking {
    error_type: String,
    timestamp: Instant,
    recovered: bool,
}

impl Default for ErrorRecoveryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorRecoveryTracker {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl EventProcessor for ErrorRecoveryTracker {
    async fn process(&self, event: &ClaudeStreamEvent) -> Result<()> {
        if let ClaudeStreamEvent::Error {
            error_type,
            recoverable,
            ..
        } = event
        {
            let tracking = ErrorRecoveryTracking {
                error_type: error_type.clone(),
                timestamp: Instant::now(),
                recovered: *recoverable,
            };
            self.events.lock().await.push(tracking);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[tokio::test]
    async fn test_stream_event_parsing() {
        let json = r#"{"type":"tool_use","id":"123","name":"Read","input":{"file":"test.rs"}}"#;
        let event: ClaudeStreamEvent = serde_json::from_str(json).unwrap();

        match event {
            ClaudeStreamEvent::ToolUse { name, .. } => {
                assert_eq!(name, "Read");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_stream_processing() {
        let stream_data = r#"{"type":"message_start","message":{"id":"msg1","model":"claude","role":"assistant","content":[]}}
{"type":"thinking","content":"Analyzing the request","tokens":5}
{"type":"tool_use","id":"1","name":"Read","input":{"file":"test.rs"}}
{"type":"message_stop"}"#;

        let cursor = Cursor::new(stream_data.as_bytes());
        let parser = StreamJSONParser::new();

        let metrics = parser.process_stream(cursor, "test-run").await.unwrap();

        assert_eq!(metrics.tool_invocations.len(), 1);
        assert_eq!(metrics.thinking_sequences.len(), 1);
    }
}
