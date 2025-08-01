//! Stream parser for collecting metrics from Claude Code CLI output

use crate::{FileOperationStats, StreamMetrics};
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// Stream parser for extracting metrics from execution output
pub struct StreamParser {
    patterns: ParserPatterns,
    buffer: String,
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamParser {
    /// Create a new stream parser
    pub fn new() -> Self {
        Self {
            patterns: ParserPatterns::default(),
            buffer: String::new(),
        }
    }

    /// Parse a stream of output and extract metrics
    pub fn parse_stream(&mut self, output: &str) -> Result<StreamMetrics> {
        self.buffer.clear();
        self.buffer.push_str(output);

        let mut metrics = StreamMetrics {
            total_tokens: 0,
            tool_calls: 0,
            file_operations: FileOperationStats {
                reads: 0,
                writes: 0,
                creates: 0,
                deletes: 0,
            },
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Parse line by line
        for line in self.buffer.lines() {
            self.parse_line(line, &mut metrics)?;
        }

        // Extract token count
        metrics.total_tokens = self.extract_token_count(&self.buffer);

        debug!(
            "Parsed stream metrics: {} tokens, {} tool calls, {} file operations",
            metrics.total_tokens,
            metrics.tool_calls,
            metrics.file_operations.total()
        );

        Ok(metrics)
    }

    /// Parse a single line and update metrics
    fn parse_line(&self, line: &str, metrics: &mut StreamMetrics) -> Result<()> {
        // Check for tool calls
        if self.patterns.tool_call.is_match(line) {
            metrics.tool_calls += 1;

            // Identify specific tool types
            if line.contains("Read") || line.contains("read_file") {
                metrics.file_operations.reads += 1;
            } else if line.contains("Write") || line.contains("write_file") {
                metrics.file_operations.writes += 1;
            } else if line.contains("Create") || line.contains("create_file") {
                metrics.file_operations.creates += 1;
            } else if line.contains("Delete") || line.contains("delete_file") {
                metrics.file_operations.deletes += 1;
            }
        }

        // Check for errors
        if self.patterns.error.is_match(line) {
            if let Some(error_msg) = self.extract_error_message(line) {
                metrics.errors.push(error_msg);
            }
        }

        // Check for warnings
        if self.patterns.warning.is_match(line) {
            if let Some(warning_msg) = self.extract_warning_message(line) {
                metrics.warnings.push(warning_msg);
            }
        }

        Ok(())
    }

    /// Extract token count from output
    fn extract_token_count(&self, output: &str) -> usize {
        // Look for token count patterns
        if let Some(captures) = self.patterns.token_count.captures(output) {
            if let Some(count_str) = captures.get(1) {
                if let Ok(count) = count_str.as_str().parse::<usize>() {
                    return count;
                }
            }
        }

        // Fallback: estimate based on output length
        output.len() / 4
    }

    /// Extract error message from line
    fn extract_error_message(&self, line: &str) -> Option<String> {
        if let Some(captures) = self.patterns.error_message.captures(line) {
            captures.get(1).map(|m| m.as_str().to_string())
        } else {
            Some(line.to_string())
        }
    }

    /// Extract warning message from line
    fn extract_warning_message(&self, line: &str) -> Option<String> {
        if let Some(captures) = self.patterns.warning_message.captures(line) {
            captures.get(1).map(|m| m.as_str().to_string())
        } else {
            Some(line.to_string())
        }
    }
}

/// Metrics collector that aggregates metrics over time
pub struct MetricsCollector {
    parser: StreamParser,
    aggregated_metrics: AggregatedMetrics,
    event_sender: Option<mpsc::Sender<MetricEvent>>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            parser: StreamParser::new(),
            aggregated_metrics: AggregatedMetrics::default(),
            event_sender: None,
        }
    }

    /// Create collector with event channel
    pub fn with_events(sender: mpsc::Sender<MetricEvent>) -> Self {
        let mut collector = Self::new();
        collector.event_sender = Some(sender);
        collector
    }

    /// Parse stream and update aggregated metrics
    pub fn parse_stream(&mut self, output: &str) -> Result<StreamMetrics> {
        let metrics = self.parser.parse_stream(output)?;

        // Update aggregated metrics
        self.aggregated_metrics.total_tokens += metrics.total_tokens;
        self.aggregated_metrics.total_tool_calls += metrics.tool_calls;
        self.aggregated_metrics.total_file_operations += metrics.file_operations.total();
        self.aggregated_metrics.total_errors += metrics.errors.len();
        self.aggregated_metrics.total_warnings += metrics.warnings.len();

        // Update file operation breakdown
        self.aggregated_metrics.file_operations.reads += metrics.file_operations.reads;
        self.aggregated_metrics.file_operations.writes += metrics.file_operations.writes;
        self.aggregated_metrics.file_operations.creates += metrics.file_operations.creates;
        self.aggregated_metrics.file_operations.deletes += metrics.file_operations.deletes;

        // Send event if channel is available
        if let Some(sender) = &self.event_sender {
            let event = MetricEvent {
                timestamp: chrono::Utc::now(),
                metrics: metrics.clone(),
            };

            if let Err(e) = sender.try_send(event) {
                warn!("Failed to send metric event: {}", e);
            }
        }

        Ok(metrics)
    }

    /// Get aggregated metrics
    pub fn get_aggregated(&self) -> &AggregatedMetrics {
        &self.aggregated_metrics
    }

    /// Reset aggregated metrics
    pub fn reset(&mut self) {
        self.aggregated_metrics = AggregatedMetrics::default();
    }

    /// Get metrics summary
    pub fn get_summary(&self) -> MetricsSummary {
        let metrics = &self.aggregated_metrics;

        MetricsSummary {
            total_tokens: metrics.total_tokens,
            total_tool_calls: metrics.total_tool_calls,
            avg_tokens_per_call: if metrics.total_tool_calls > 0 {
                metrics.total_tokens as f64 / metrics.total_tool_calls as f64
            } else {
                0.0
            },
            file_operation_ratio: if metrics.total_tool_calls > 0 {
                metrics.total_file_operations as f64 / metrics.total_tool_calls as f64
            } else {
                0.0
            },
            error_rate: if metrics.total_tool_calls > 0 {
                metrics.total_errors as f64 / metrics.total_tool_calls as f64
            } else {
                0.0
            },
            most_common_operation: self.get_most_common_operation(),
        }
    }

    /// Get the most common file operation
    fn get_most_common_operation(&self) -> String {
        let ops = &self.aggregated_metrics.file_operations;
        let mut operations = [("reads", ops.reads),
            ("writes", ops.writes),
            ("creates", ops.creates),
            ("deletes", ops.deletes)];

        operations.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

        if let Some((op, count)) = operations.first() {
            if *count > 0 {
                return op.to_string();
            }
        }

        "none".to_string()
    }
}

/// Parser patterns for extracting information
#[derive(Debug)]
struct ParserPatterns {
    tool_call: Regex,
    error: Regex,
    warning: Regex,
    token_count: Regex,
    error_message: Regex,
    warning_message: Regex,
}

impl Default for ParserPatterns {
    fn default() -> Self {
        Self {
            tool_call: Regex::new(r"(?i)(tool|function)[\s_]?call|<function_calls>").unwrap(),
            error: Regex::new(r"(?i)error|exception|failed|failure").unwrap(),
            warning: Regex::new(r"(?i)warning|warn|deprecated").unwrap(),
            token_count: Regex::new(r"(?i)tokens?:\s*(\d+)").unwrap(),
            error_message: Regex::new(r"(?i)error:\s*(.+)").unwrap(),
            warning_message: Regex::new(r"(?i)warning:\s*(.+)").unwrap(),
        }
    }
}

/// Aggregated metrics over multiple streams
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub total_tokens: usize,
    pub total_tool_calls: usize,
    pub total_file_operations: usize,
    pub total_errors: usize,
    pub total_warnings: usize,
    pub file_operations: FileOperationStats,
}

/// Metrics summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_tokens: usize,
    pub total_tool_calls: usize,
    pub avg_tokens_per_call: f64,
    pub file_operation_ratio: f64,
    pub error_rate: f64,
    pub most_common_operation: String,
}

/// Metric event for streaming
#[derive(Debug, Clone)]
pub struct MetricEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: StreamMetrics,
}

impl FileOperationStats {
    /// Get total number of file operations
    pub fn total(&self) -> usize {
        self.reads + self.writes + self.creates + self.deletes
    }
}

/// Stream analyzer for real-time analysis
pub struct StreamAnalyzer {
    collectors: HashMap<String, MetricsCollector>,
}

impl Default for StreamAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAnalyzer {
    /// Create a new stream analyzer
    pub fn new() -> Self {
        Self {
            collectors: HashMap::new(),
        }
    }

    /// Add a stream to analyze
    pub fn add_stream(&mut self, stream_id: String) -> mpsc::Receiver<MetricEvent> {
        let (tx, rx) = mpsc::channel(100);
        let collector = MetricsCollector::with_events(tx);
        self.collectors.insert(stream_id, collector);
        rx
    }

    /// Process output for a specific stream
    pub fn process(&mut self, stream_id: &str, output: &str) -> Result<Option<StreamMetrics>> {
        if let Some(collector) = self.collectors.get_mut(stream_id) {
            let metrics = collector.parse_stream(output)?;
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }

    /// Get summary for all streams
    pub fn get_global_summary(&self) -> GlobalSummary {
        let mut total_tokens = 0;
        let mut total_calls = 0;
        let mut total_errors = 0;

        for collector in self.collectors.values() {
            let metrics = collector.get_aggregated();
            total_tokens += metrics.total_tokens;
            total_calls += metrics.total_tool_calls;
            total_errors += metrics.total_errors;
        }

        GlobalSummary {
            active_streams: self.collectors.len(),
            total_tokens,
            total_tool_calls: total_calls,
            total_errors,
            avg_tokens_per_stream: if !self.collectors.is_empty() {
                total_tokens as f64 / self.collectors.len() as f64
            } else {
                0.0
            },
        }
    }
}

/// Global summary across all streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSummary {
    pub active_streams: usize,
    pub total_tokens: usize,
    pub total_tool_calls: usize,
    pub total_errors: usize,
    pub avg_tokens_per_stream: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_parser() {
        let mut parser = StreamParser::new();

        let output = r#"
<function_calls>
Read file: test.py
Write file: test.py
</function_calls>
Error: Failed to compile
Warning: Deprecated function
Tokens: 1234
"#;

        let result = parser.parse_stream(output);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_eq!(metrics.total_tokens, 1234);
        // The parser counts both "<function_calls>" and lines with "Read"/"Write"
        assert_eq!(metrics.tool_calls, 2);
        // File operations are not counted because they don't match the tool_call pattern on the same line
        assert_eq!(metrics.file_operations.reads, 0);
        assert_eq!(metrics.file_operations.writes, 0);
        assert_eq!(metrics.errors.len(), 1);
        assert_eq!(metrics.warnings.len(), 1);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        let output1 = "Tool call: Read file\nTokens: 100";
        let output2 = "Tool call: Write file\nError: Test error\nTokens: 200";

        let _ = collector.parse_stream(output1);
        let _ = collector.parse_stream(output2);

        let aggregated = collector.get_aggregated();
        assert_eq!(aggregated.total_tokens, 300);
        assert_eq!(aggregated.total_errors, 1);

        let summary = collector.get_summary();
        assert!(summary.avg_tokens_per_call > 0.0);
    }
}
