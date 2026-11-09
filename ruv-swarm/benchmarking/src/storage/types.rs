use crate::metrics::{ErrorRecovery, ThinkingSequence, ToolInvocation};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub run_id: String,
    pub instance_id: String,
    pub repository: String,
    pub issue_description: Option<String>,
    pub difficulty: Option<String>,
    pub execution_mode: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status: String,
    pub claude_command: String,
    pub configuration: serde_json::Value,
    pub environment: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub run_id: String,
    pub metric_type: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub unit: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

pub type StreamEvent = serde_json::Value;
pub type ToolInvocationRecord = ToolInvocation;
pub type ThinkingSequenceRecord = ThinkingSequence;
pub type ErrorRecoveryEvent = ErrorRecovery;
pub type CodeQualityMetric = serde_json::Value;
pub type ResourceUsageRecord = serde_json::Value;
