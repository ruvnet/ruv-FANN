//! Metrics collection and analysis module

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Core performance metrics collected during benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    // Timing Metrics
    pub task_completion_time: Duration,
    pub time_to_first_output: Duration,
    pub agent_coordination_overhead: Duration,
    pub ml_inference_time: Duration,

    // Code Quality Metrics
    pub code_quality_score: CodeQualityScore,
    pub test_coverage: f64,
    pub cyclomatic_complexity: u32,
    pub maintainability_index: f64,
    pub documentation_completeness: f64,

    // Resource Usage Metrics
    pub cpu_usage: ResourceUsage,
    pub memory_usage: ResourceUsage,
    pub network_bandwidth: ResourceUsage,
    pub disk_io: ResourceUsage,

    // Accuracy/Correctness Metrics
    pub functional_correctness: f64,
    pub test_pass_rate: f64,
    pub edge_case_handling: f64,
    pub error_handling_quality: f64,

    // Swarm Coordination Metrics
    pub swarm_metrics: SwarmCoordinationMetrics,

    // Tool usage and thinking patterns
    pub tool_invocations: Vec<ToolInvocation>,
    pub thinking_sequences: Vec<ThinkingSequence>,
    pub error_recoveries: Vec<ErrorRecovery>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeQualityScore {
    pub overall: f64,
    pub readability: f64,
    pub modularity: f64,
    pub best_practices_adherence: f64,
    pub security_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    pub average: f64,
    pub peak: f64,
    pub p95: f64,
    pub p99: f64,
    pub timeline: Vec<(DateTime<Utc>, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SwarmCoordinationMetrics {
    pub agent_utilization: f64,
    pub communication_efficiency: f64,
    pub task_distribution_balance: f64,
    pub conflict_resolution_time: Duration,
    pub consensus_achievement_rate: f64,
    pub active_agents: u32,
    pub messages_passed: u32,
    pub conflicts_resolved: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub tool_name: String,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub parameters: serde_json::Value,
    pub result_size: usize,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingSequence {
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub token_count: usize,
    pub decision_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecovery {
    pub error_time: DateTime<Utc>,
    pub error_type: String,
    pub error_message: String,
    pub recovery_started: Option<DateTime<Utc>>,
    pub recovery_completed: Option<DateTime<Utc>>,
    pub recovery_strategy: Option<String>,
    pub recovery_success: bool,
}

/// Metrics collector that aggregates performance data
pub struct MetricsCollector {
    start_time: Instant,
    first_output_time: Option<Instant>,
    tool_metrics: HashMap<String, ToolUsageStats>,
    resource_samples: Vec<ResourceSample>,
    thinking_stats: ThinkingStats,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            first_output_time: None,
            tool_metrics: HashMap::new(),
            resource_samples: Vec::new(),
            thinking_stats: ThinkingStats::default(),
        }
    }

    /// Record first output time
    pub fn record_first_output(&mut self) {
        if self.first_output_time.is_none() {
            self.first_output_time = Some(Instant::now());
        }
    }

    /// Record a tool invocation
    pub fn record_tool_invocation(&mut self, invocation: &ToolInvocation) {
        let stats = self
            .tool_metrics
            .entry(invocation.tool_name.clone())
            .or_default();

        stats.invocation_count += 1;
        stats.total_duration += invocation.duration;
        if invocation.success {
            stats.success_count += 1;
        }
    }

    /// Record resource usage sample
    pub fn record_resource_usage(&mut self, sample: ResourceSample) {
        self.resource_samples.push(sample);
    }

    /// Record thinking sequence
    pub fn record_thinking(&mut self, sequence: &ThinkingSequence) {
        self.thinking_stats.total_sequences += 1;
        self.thinking_stats.total_tokens += sequence.token_count;
        self.thinking_stats.total_duration += sequence.duration;
    }

    /// Calculate final metrics
    pub fn finalize(self) -> PerformanceMetrics {
        let elapsed = self.start_time.elapsed();

        PerformanceMetrics {
            task_completion_time: elapsed,
            time_to_first_output: self
                .first_output_time
                .map(|t| t.duration_since(self.start_time))
                .unwrap_or_default(),
            agent_coordination_overhead: self.calculate_coordination_overhead(),
            ml_inference_time: self.calculate_ml_inference_time(),
            code_quality_score: self.calculate_code_quality(),
            test_coverage: 0.0, // Would be filled by test results
            cyclomatic_complexity: 0,
            maintainability_index: 0.0,
            documentation_completeness: 0.0,
            cpu_usage: self.calculate_resource_usage(&self.resource_samples, |s| s.cpu_percent),
            memory_usage: self.calculate_resource_usage(&self.resource_samples, |s| s.memory_mb),
            network_bandwidth: self
                .calculate_resource_usage(&self.resource_samples, |s| s.network_kb_per_sec),
            disk_io: self
                .calculate_resource_usage(&self.resource_samples, |s| s.disk_io_kb_per_sec),
            functional_correctness: 0.0, // Would be filled by test results
            test_pass_rate: 0.0,
            edge_case_handling: 0.0,
            error_handling_quality: 0.0,
            swarm_metrics: SwarmCoordinationMetrics::default(),
            tool_invocations: Vec::new(),
            thinking_sequences: Vec::new(),
            error_recoveries: Vec::new(),
        }
    }

    fn calculate_coordination_overhead(&self) -> Duration {
        // Calculate based on message passing and synchronization
        Duration::from_millis(0) // Placeholder
    }

    fn calculate_ml_inference_time(&self) -> Duration {
        // Calculate ML model inference time
        Duration::from_millis(0) // Placeholder
    }

    fn calculate_code_quality(&self) -> CodeQualityScore {
        CodeQualityScore {
            overall: 0.8,
            readability: 0.85,
            modularity: 0.75,
            best_practices_adherence: 0.82,
            security_score: 0.9,
        }
    }

    fn calculate_resource_usage<F>(&self, samples: &[ResourceSample], extractor: F) -> ResourceUsage
    where
        F: Fn(&ResourceSample) -> f64,
    {
        if samples.is_empty() {
            return ResourceUsage::default();
        }

        let mut values: Vec<f64> = samples.iter().map(&extractor).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sum: f64 = values.iter().sum();
        let average = sum / values.len() as f64;
        let peak = values.last().copied().unwrap_or(0.0);
        let p95_idx = (values.len() as f64 * 0.95) as usize;
        let p99_idx = (values.len() as f64 * 0.99) as usize;

        ResourceUsage {
            average,
            peak,
            p95: values.get(p95_idx).copied().unwrap_or(peak),
            p99: values.get(p99_idx).copied().unwrap_or(peak),
            timeline: samples
                .iter()
                .map(|s| (s.timestamp, extractor(s)))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceSample {
    pub timestamp: DateTime<Utc>,
    pub cpu_percent: f64,
    pub memory_mb: f64,
    pub network_kb_per_sec: f64,
    pub disk_io_kb_per_sec: f64,
}

#[derive(Debug, Default)]
struct ToolUsageStats {
    invocation_count: u32,
    success_count: u32,
    total_duration: Duration,
}

#[derive(Debug, Default)]
struct ThinkingStats {
    total_sequences: u32,
    total_tokens: usize,
    total_duration: Duration,
}

/// Metric type enumeration for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    TaskCompletionTime,
    TimeToFirstOutput,
    AgentCoordinationOverhead,
    MLInferenceTime,
    CodeQualityOverall,
    TestCoverage,
    CyclomaticComplexity,
    MaintainabilityIndex,
    CpuUsageAverage,
    CpuUsagePeak,
    MemoryUsageAverage,
    MemoryUsagePeak,
    NetworkBandwidth,
    DiskIO,
    FunctionalCorrectness,
    TestPassRate,
    AgentUtilization,
    CommunicationEfficiency,
    TaskDistributionBalance,
}

impl MetricType {
    pub fn to_string(&self) -> String {
        format!("{:?}", self)
    }

    pub fn unit(&self) -> &'static str {
        match self {
            MetricType::TaskCompletionTime
            | MetricType::TimeToFirstOutput
            | MetricType::AgentCoordinationOverhead
            | MetricType::MLInferenceTime => "ms",

            MetricType::CodeQualityOverall
            | MetricType::TestCoverage
            | MetricType::FunctionalCorrectness
            | MetricType::TestPassRate
            | MetricType::AgentUtilization
            | MetricType::CommunicationEfficiency
            | MetricType::TaskDistributionBalance => "percentage",

            MetricType::CyclomaticComplexity => "count",
            MetricType::MaintainabilityIndex => "index",

            MetricType::CpuUsageAverage | MetricType::CpuUsagePeak => "percent",

            MetricType::MemoryUsageAverage | MetricType::MemoryUsagePeak => "MB",

            MetricType::NetworkBandwidth | MetricType::DiskIO => "KB/s",
        }
    }
}

/// Advanced metrics for ML-specific analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMetrics {
    // ML-Specific Metrics
    pub model_accuracy: f64,
    pub prediction_confidence: f64,
    pub feature_importance: HashMap<String, f64>,
    pub optimization_convergence_rate: f64,

    // Swarm Intelligence Metrics
    pub collective_problem_solving_efficiency: f64,
    pub emergent_behavior_quality: f64,
    pub adaptation_rate: f64,
    pub knowledge_sharing_effectiveness: f64,

    // Developer Experience Metrics
    pub api_usability_score: f64,
    pub debugging_ease: f64,
    pub integration_complexity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        // Record some metrics
        collector.record_first_output();

        let tool_invocation = ToolInvocation {
            tool_name: "Read".to_string(),
            timestamp: Utc::now(),
            duration: Duration::from_millis(100),
            parameters: serde_json::json!({"file": "test.rs"}),
            result_size: 1024,
            success: true,
            error_message: None,
        };

        collector.record_tool_invocation(&tool_invocation);

        // Finalize and check
        let metrics = collector.finalize();
        assert!(metrics.time_to_first_output > Duration::ZERO);
    }

    #[test]
    fn test_metric_type_units() {
        assert_eq!(MetricType::TaskCompletionTime.unit(), "ms");
        assert_eq!(MetricType::CpuUsageAverage.unit(), "percent");
        assert_eq!(MetricType::MemoryUsagePeak.unit(), "MB");
    }
}
