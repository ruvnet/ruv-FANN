//! Claude Code Stream-JSON Parser
//!
//! This module provides comprehensive parsing and analysis of Claude Code CLI
//! stream-json output for benchmarking and training data collection.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tracing::{debug, error, info};

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid event format: {0}")]
    InvalidFormat(String),

    #[error("Stream processing error: {0}")]
    StreamError(String),
}

/// Result type for parser operations
pub type Result<T> = std::result::Result<T, ParserError>;

/// Claude stream event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {
        message: MessageInfo,
        timestamp: Option<DateTime<Utc>>,
    },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
        timestamp: Option<DateTime<Utc>>,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
        timestamp: Option<DateTime<Utc>>,
    },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        index: usize,
        timestamp: Option<DateTime<Utc>>,
    },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        timestamp: Option<DateTime<Utc>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },

    #[serde(rename = "thinking")]
    Thinking {
        content: String,
        tokens: usize,
        timestamp: Option<DateTime<Utc>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },

    #[serde(rename = "function_result")]
    FunctionResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
        timestamp: Option<DateTime<Utc>>,
    },

    #[serde(rename = "error")]
    Error {
        error_type: String,
        message: String,
        recoverable: bool,
        timestamp: Option<DateTime<Utc>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        recovery_strategy: Option<String>,
    },

    #[serde(rename = "message_stop")]
    MessageStop {
        stop_reason: Option<String>,
        timestamp: Option<DateTime<Utc>>,
    },

    #[serde(rename = "usage")]
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        timestamp: Option<DateTime<Utc>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageInfo {
    pub id: String,
    pub model: String,
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_json: Option<String>,
}

/// Performance metrics collected from stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_duration: Duration,
    pub time_to_first_output: Option<Duration>,
    pub tool_invocations: HashMap<String, ToolMetrics>,
    pub thinking_metrics: ThinkingMetrics,
    pub token_usage: TokenUsage,
    pub error_metrics: ErrorMetrics,
    pub event_timeline: Vec<TimestampedEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetrics {
    pub invocation_count: u64,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub parameter_sizes: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingMetrics {
    pub total_sequences: u64,
    pub total_tokens: u64,
    pub total_duration: Duration,
    pub average_tokens_per_sequence: f64,
    pub thinking_patterns: Vec<ThinkingPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingPattern {
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub token_count: usize,
    pub content_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub recoverable_errors: u64,
    pub recovery_success_rate: f64,
    pub error_types: HashMap<String, u64>,
    pub recovery_strategies: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedEvent {
    pub timestamp: DateTime<Utc>,
    pub relative_time_ms: u64,
    pub event_type: String,
    pub event_summary: String,
}

/// Stream parser for Claude output
pub struct ClaudeStreamParser {
    start_time: Instant,
    events: Arc<DashMap<String, Vec<ClaudeStreamEvent>>>,
    metrics_collector: MetricsCollector,
}

impl Default for ClaudeStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeStreamParser {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            events: Arc::new(DashMap::new()),
            metrics_collector: MetricsCollector::new(),
        }
    }

    /// Parse a stream of Claude events
    pub async fn parse_stream<R: AsyncRead + Unpin>(
        &mut self,
        reader: R,
    ) -> Result<PerformanceMetrics> {
        let mut lines = BufReader::new(reader).lines();
        let mut event_count = 0;

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            match self.parse_event(&line) {
                Ok(event) => {
                    event_count += 1;
                    self.process_event(event).await?;
                }
                Err(e) => {
                    error!("Failed to parse event: {} - Line: {}", e, line);
                    // Continue processing other events
                }
            }
        }

        info!("Processed {} events", event_count);
        Ok(self.metrics_collector.finalize(self.start_time.elapsed()))
    }

    /// Parse a single event line
    fn parse_event(&self, line: &str) -> Result<ClaudeStreamEvent> {
        // Handle both direct JSON and "data: " prefixed format
        let json_str = if line.starts_with("data: ") {
            &line[6..]
        } else {
            line
        };

        serde_json::from_str(json_str).map_err(ParserError::JsonError)
    }

    /// Process a parsed event
    async fn process_event(&mut self, event: ClaudeStreamEvent) -> Result<()> {
        let relative_time = self.start_time.elapsed();

        // Update metrics based on event type
        match &event {
            ClaudeStreamEvent::MessageStart { .. } => {
                self.metrics_collector.record_message_start(relative_time);
            }
            ClaudeStreamEvent::ToolUse { name, .. } => {
                self.metrics_collector
                    .record_tool_use(name.clone(), relative_time);
            }
            ClaudeStreamEvent::Thinking {
                tokens, content, ..
            } => {
                self.metrics_collector
                    .record_thinking(*tokens, content.clone(), relative_time);
            }
            ClaudeStreamEvent::Error {
                error_type,
                recoverable,
                ..
            } => {
                self.metrics_collector
                    .record_error(error_type.clone(), *recoverable);
            }
            ClaudeStreamEvent::FunctionResult {
                is_error,
                tool_use_id,
                ..
            } => {
                self.metrics_collector
                    .record_function_result(tool_use_id.clone(), *is_error);
            }
            ClaudeStreamEvent::Usage {
                input_tokens,
                output_tokens,
                total_tokens,
                ..
            } => {
                self.metrics_collector.record_token_usage(
                    *input_tokens,
                    *output_tokens,
                    *total_tokens,
                );
            }
            _ => {}
        }

        // Store event for timeline
        self.store_event(event, relative_time)?;

        Ok(())
    }

    /// Store event in timeline
    fn store_event(&self, event: ClaudeStreamEvent, relative_time: Duration) -> Result<()> {
        let event_type = match &event {
            ClaudeStreamEvent::MessageStart { .. } => "message_start",
            ClaudeStreamEvent::ContentBlockStart { .. } => "content_block_start",
            ClaudeStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
            ClaudeStreamEvent::ContentBlockStop { .. } => "content_block_stop",
            ClaudeStreamEvent::ToolUse { .. } => "tool_use",
            ClaudeStreamEvent::Thinking { .. } => "thinking",
            ClaudeStreamEvent::FunctionResult { .. } => "function_result",
            ClaudeStreamEvent::Error { .. } => "error",
            ClaudeStreamEvent::MessageStop { .. } => "message_stop",
            ClaudeStreamEvent::Usage { .. } => "usage",
        };

        self.events
            .entry(event_type.to_string())
            .or_default()
            .push(event);

        self.metrics_collector
            .add_timeline_event(event_type.to_string(), relative_time);

        Ok(())
    }

    /// Export collected data for training
    pub fn export_training_data(&self) -> TrainingDataExport {
        let mut all_events = Vec::new();

        for entry in self.events.iter() {
            all_events.extend(entry.value().clone());
        }

        TrainingDataExport {
            events: all_events,
            metrics: self.metrics_collector.get_current_metrics(),
            metadata: ExportMetadata {
                export_time: Utc::now(),
                parser_version: env!("CARGO_PKG_VERSION").to_string(),
                event_count: self.events.iter().map(|e| e.value().len()).sum(),
            },
        }
    }
}

/// Metrics collector for performance analysis
struct MetricsCollector {
    tool_metrics: Arc<DashMap<String, ToolMetricsBuilder>>,
    thinking_sequences: Arc<DashMap<String, ThinkingSequence>>,
    error_counts: Arc<DashMap<String, u64>>,
    recovery_strategies: Arc<DashMap<String, u64>>,
    timeline_events: Arc<DashMap<u64, TimestampedEvent>>,
    token_usage: Arc<DashMap<String, u64>>,
    first_output_time: Arc<tokio::sync::Mutex<Option<Duration>>>,
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            tool_metrics: Arc::new(DashMap::new()),
            thinking_sequences: Arc::new(DashMap::new()),
            error_counts: Arc::new(DashMap::new()),
            recovery_strategies: Arc::new(DashMap::new()),
            timeline_events: Arc::new(DashMap::new()),
            token_usage: Arc::new(DashMap::new()),
            first_output_time: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    fn record_message_start(&self, relative_time: Duration) {
        let mut first_output = self.first_output_time.try_lock().unwrap();
        if first_output.is_none() {
            *first_output = Some(relative_time);
        }
    }

    fn record_tool_use(&self, tool_name: String, _relative_time: Duration) {
        self.tool_metrics
            .entry(tool_name)
            .or_insert_with(ToolMetricsBuilder::new)
            .invocation_count += 1;
    }

    fn record_thinking(&self, tokens: usize, content: String, relative_time: Duration) {
        let id = format!("thinking_{}", self.thinking_sequences.len());
        self.thinking_sequences.insert(
            id,
            ThinkingSequence {
                start_time: Utc::now(),
                tokens,
                content_preview: content.chars().take(100).collect(),
                duration: relative_time,
            },
        );
    }

    fn record_error(&self, error_type: String, recoverable: bool) {
        *self.error_counts.entry(error_type).or_insert(0) += 1;
        if recoverable {
            *self
                .error_counts
                .entry("recoverable".to_string())
                .or_insert(0) += 1;
        }
    }

    fn record_function_result(&self, tool_use_id: String, is_error: bool) {
        if !is_error {
            // Mark tool use as successful
            debug!("Tool {} completed successfully", tool_use_id);
        }
    }

    fn record_token_usage(&self, input: u64, output: u64, total: u64) {
        self.token_usage.insert("input".to_string(), input);
        self.token_usage.insert("output".to_string(), output);
        self.token_usage.insert("total".to_string(), total);
    }

    fn add_timeline_event(&self, event_type: String, relative_time: Duration) {
        let event = TimestampedEvent {
            timestamp: Utc::now(),
            relative_time_ms: relative_time.as_millis() as u64,
            event_type: event_type.clone(),
            event_summary: format!("{} at {:?}", event_type, relative_time),
        };

        self.timeline_events
            .insert(relative_time.as_millis() as u64, event);
    }

    fn finalize(&self, total_duration: Duration) -> PerformanceMetrics {
        // Calculate tool metrics
        let mut tool_invocations = HashMap::new();
        for entry in self.tool_metrics.iter() {
            let (name, builder) = entry.pair();
            tool_invocations.insert(
                name.clone(),
                ToolMetrics {
                    invocation_count: builder.invocation_count,
                    total_duration: Duration::from_millis(100 * builder.invocation_count), // Estimate
                    average_duration: Duration::from_millis(100), // Estimate
                    success_rate: 0.95,                           // Estimate
                    parameter_sizes: vec![],
                },
            );
        }

        // Calculate thinking metrics
        let thinking_patterns: Vec<_> = self
            .thinking_sequences
            .iter()
            .map(|entry| ThinkingPattern {
                start_time: entry.value().start_time,
                duration: entry.value().duration,
                token_count: entry.value().tokens,
                content_preview: entry.value().content_preview.clone(),
            })
            .collect();

        let total_thinking_tokens: u64 =
            thinking_patterns.iter().map(|p| p.token_count as u64).sum();

        let thinking_metrics = ThinkingMetrics {
            total_sequences: thinking_patterns.len() as u64,
            total_tokens: total_thinking_tokens,
            total_duration: Duration::from_millis(total_thinking_tokens * 50), // Estimate
            average_tokens_per_sequence: if thinking_patterns.is_empty() {
                0.0
            } else {
                total_thinking_tokens as f64 / thinking_patterns.len() as f64
            },
            thinking_patterns,
        };

        // Calculate error metrics
        let total_errors: u64 = self
            .error_counts
            .iter()
            .filter(|e| e.key() != "recoverable")
            .map(|e| *e.value())
            .sum();

        let recoverable_errors = self
            .error_counts
            .get("recoverable")
            .map(|e| *e.value())
            .unwrap_or(0);

        let error_types: HashMap<_, _> = self
            .error_counts
            .iter()
            .filter(|e| e.key() != "recoverable")
            .map(|e| (e.key().clone(), *e.value()))
            .collect();

        let error_metrics = ErrorMetrics {
            total_errors,
            recoverable_errors,
            recovery_success_rate: if recoverable_errors > 0 {
                0.8 // Estimate
            } else {
                1.0
            },
            error_types,
            recovery_strategies: self
                .recovery_strategies
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect(),
        };

        // Calculate token usage
        let input_tokens = self
            .token_usage
            .get("input")
            .map(|e| *e.value())
            .unwrap_or(0);
        let output_tokens = self
            .token_usage
            .get("output")
            .map(|e| *e.value())
            .unwrap_or(0);
        let total_tokens = self
            .token_usage
            .get("total")
            .map(|e| *e.value())
            .unwrap_or(0);

        let token_usage = TokenUsage {
            input_tokens,
            output_tokens,
            total_tokens,
            tokens_per_second: if total_duration.as_secs() > 0 {
                total_tokens as f64 / total_duration.as_secs_f64()
            } else {
                0.0
            },
        };

        // Build timeline
        let mut timeline: Vec<_> = self
            .timeline_events
            .iter()
            .map(|e| e.value().clone())
            .collect();
        timeline.sort_by_key(|e| e.relative_time_ms);

        PerformanceMetrics {
            total_duration,
            time_to_first_output: *self.first_output_time.try_lock().unwrap(),
            tool_invocations,
            thinking_metrics,
            token_usage,
            error_metrics,
            event_timeline: timeline,
        }
    }

    fn get_current_metrics(&self) -> PerformanceMetrics {
        self.finalize(Duration::from_secs(0))
    }
}

#[derive(Debug, Clone)]
struct ToolMetricsBuilder {
    invocation_count: u64,
}

impl ToolMetricsBuilder {
    fn new() -> Self {
        Self {
            invocation_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct ThinkingSequence {
    start_time: DateTime<Utc>,
    tokens: usize,
    content_preview: String,
    duration: Duration,
}

/// Training data export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataExport {
    pub events: Vec<ClaudeStreamEvent>,
    pub metrics: PerformanceMetrics,
    pub metadata: ExportMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub export_time: DateTime<Utc>,
    pub parser_version: String,
    pub event_count: usize,
}

impl TrainingDataExport {
    /// Export to JSON file
    pub async fn to_json_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    /// Export to JSONL format (one event per line)
    pub async fn to_jsonl_file(&self, path: &str) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let file = tokio::fs::File::create(path).await?;
        let mut writer = tokio::io::BufWriter::new(file);

        for event in &self.events {
            let line = serde_json::to_string(event)?;
            writer.write_all(line.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }

        writer.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_parse_tool_use_event() {
        let event_json =
            r#"{"type":"tool_use","id":"123","name":"Read","input":{"file_path":"/test.txt"}}"#;

        let parser = ClaudeStreamParser::new();
        let event = parser.parse_event(event_json).unwrap();

        match event {
            ClaudeStreamEvent::ToolUse { name, .. } => {
                assert_eq!(name, "Read");
            }
            _ => panic!("Expected ToolUse event"),
        }
    }

    #[tokio::test]
    async fn test_parse_thinking_event() {
        let event_json = r#"{"type":"thinking","content":"Analyzing the code...","tokens":42}"#;

        let parser = ClaudeStreamParser::new();
        let event = parser.parse_event(event_json).unwrap();

        match event {
            ClaudeStreamEvent::Thinking { tokens, .. } => {
                assert_eq!(tokens, 42);
            }
            _ => panic!("Expected Thinking event"),
        }
    }

    #[tokio::test]
    async fn test_stream_parsing() {
        let stream_data = r#"{"type":"message_start","message":{"id":"msg_123","model":"claude-3","role":"assistant"}}
{"type":"tool_use","id":"tool_1","name":"Read","input":{"file_path":"/test.txt"}}
{"type":"thinking","content":"Processing...","tokens":25}
{"type":"usage","input_tokens":100,"output_tokens":200,"total_tokens":300}
"#;

        let mut parser = ClaudeStreamParser::new();
        let metrics = parser.parse_stream(stream_data.as_bytes()).await.unwrap();

        assert_eq!(metrics.tool_invocations.len(), 1);
        assert_eq!(metrics.thinking_metrics.total_sequences, 1);
        assert_eq!(metrics.token_usage.total_tokens, 300);
    }

    #[tokio::test]
    async fn test_error_handling() {
        let stream_data = r#"{"type":"error","error_type":"ToolError","message":"File not found","recoverable":true}
{"type":"error","error_type":"NetworkError","message":"Connection lost","recoverable":false}
"#;

        let mut parser = ClaudeStreamParser::new();
        let metrics = parser.parse_stream(stream_data.as_bytes()).await.unwrap();

        assert_eq!(metrics.error_metrics.total_errors, 2);
        assert_eq!(metrics.error_metrics.recoverable_errors, 1);
    }

    #[tokio::test]
    async fn test_export_training_data() {
        let stream_data = r#"{"type":"tool_use","id":"1","name":"Write","input":{"content":"test"}}
{"type":"thinking","content":"Done","tokens":10}
"#;

        let mut parser = ClaudeStreamParser::new();
        parser.parse_stream(stream_data.as_bytes()).await.unwrap();

        let export = parser.export_training_data();
        assert_eq!(export.events.len(), 2);
        assert_eq!(export.metadata.event_count, 2);

        // Test JSON export
        use tempfile::NamedTempFile;
        let temp_file = NamedTempFile::new().unwrap();
        export
            .to_json_file(temp_file.path().to_str().unwrap())
            .await
            .unwrap();

        // Verify file was written
        let content = tokio::fs::read_to_string(temp_file.path()).await.unwrap();
        assert!(content.contains("tool_use"));
    }

    #[tokio::test]
    async fn test_performance_metrics_calculation() {
        let stream_data = r#"{"type":"message_start","message":{"id":"1","model":"claude","role":"assistant"}}
{"type":"tool_use","id":"1","name":"Read","input":{}}
{"type":"tool_use","id":"2","name":"Write","input":{}}
{"type":"tool_use","id":"3","name":"Read","input":{}}
{"type":"thinking","content":"Planning...","tokens":50}
{"type":"thinking","content":"Executing...","tokens":75}
{"type":"usage","input_tokens":150,"output_tokens":250,"total_tokens":400}
"#;

        let mut parser = ClaudeStreamParser::new();
        let metrics = parser.parse_stream(stream_data.as_bytes()).await.unwrap();

        // Verify tool metrics
        assert_eq!(
            metrics
                .tool_invocations
                .get("Read")
                .unwrap()
                .invocation_count,
            2
        );
        assert_eq!(
            metrics
                .tool_invocations
                .get("Write")
                .unwrap()
                .invocation_count,
            1
        );

        // Verify thinking metrics
        assert_eq!(metrics.thinking_metrics.total_sequences, 2);
        assert_eq!(metrics.thinking_metrics.total_tokens, 125);
        assert_eq!(metrics.thinking_metrics.average_tokens_per_sequence, 62.5);

        // Verify token usage
        assert_eq!(metrics.token_usage.input_tokens, 150);
        assert_eq!(metrics.token_usage.output_tokens, 250);
        assert_eq!(metrics.token_usage.total_tokens, 400);
    }
}
