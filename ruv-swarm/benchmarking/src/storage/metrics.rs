use super::{encode_time, BenchmarkStorage};
use crate::metrics::{
    ErrorRecovery, MetricType, PerformanceMetrics, SwarmCoordinationMetrics, ThinkingSequence,
    ToolInvocation,
};
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Transaction};
use std::time::Duration;

impl BenchmarkStorage {
    /// Persist a complete metrics batch atomically, including its detailed
    /// events. Any constraint, serialization or conversion error rolls it back.
    pub async fn store_metrics(&self, run_id: &str, metrics: &PerformanceMetrics) -> Result<()> {
        let (run_id, metrics) = (run_id.to_owned(), StoredMetrics::from(metrics));
        self.with_connection(move |connection| {
            let transaction = connection.transaction()?;
            write_metrics(&transaction, &run_id, &metrics)?;
            transaction.commit()?;
            Ok(())
        })
        .await
    }
}

// Own only the records this API persists. Copying the full public metrics value
// would also clone potentially large, unused CPU/memory/network/I/O timelines.
struct StoredMetrics {
    values: [(MetricType, f64); 5],
    tool_invocations: Vec<ToolInvocation>,
    thinking_sequences: Vec<ThinkingSequence>,
    error_recoveries: Vec<ErrorRecovery>,
    swarm_metrics: SwarmCoordinationMetrics,
}

impl From<&PerformanceMetrics> for StoredMetrics {
    fn from(metrics: &PerformanceMetrics) -> Self {
        Self {
            values: [
                (
                    MetricType::TaskCompletionTime,
                    metrics.task_completion_time.as_millis() as f64,
                ),
                (
                    MetricType::TimeToFirstOutput,
                    metrics.time_to_first_output.as_millis() as f64,
                ),
                (
                    MetricType::CodeQualityOverall,
                    metrics.code_quality_score.overall,
                ),
                (MetricType::CpuUsageAverage, metrics.cpu_usage.average),
                (MetricType::MemoryUsageAverage, metrics.memory_usage.average),
            ],
            tool_invocations: metrics.tool_invocations.clone(),
            thinking_sequences: metrics.thinking_sequences.clone(),
            error_recoveries: metrics.error_recoveries.clone(),
            swarm_metrics: metrics.swarm_metrics.clone(),
        }
    }
}

fn write_metrics(
    transaction: &Transaction<'_>,
    run_id: &str,
    metrics: &StoredMetrics,
) -> Result<()> {
    let mut insert_metric = transaction.prepare_cached(
        "INSERT INTO performance_metrics (
        run_id, metric_type, metric_name, metric_value, unit, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
    )?;
    for (kind, value) in &metrics.values {
        let name = kind.to_string();
        insert_metric.execute(params![
            run_id,
            name,
            name,
            value,
            kind.unit(),
            encode_time(Utc::now())
        ])?;
    }
    let mut insert_tool = transaction.prepare_cached(
        "INSERT INTO tool_invocations (
        run_id, tool_name, invocation_time, duration_ms, parameters, result_size,
        success, error_message, sequence_in_run) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )?;
    for (index, tool) in metrics.tool_invocations.iter().enumerate() {
        insert_tool.execute(params![
            run_id,
            tool.tool_name,
            encode_time(tool.timestamp),
            milliseconds(tool.duration)?,
            tool.parameters.to_string(),
            i64::try_from(tool.result_size).context("tool result size exceeds SQLite integer")?,
            tool.success,
            tool.error_message,
            i32::try_from(index).context("tool sequence exceeds SQLite sequence range")?
        ])?;
    }
    let mut insert_thinking = transaction.prepare_cached(
        "INSERT INTO thinking_sequences (
        run_id, start_time, duration_ms, token_count, decision_points) VALUES (?, ?, ?, ?, ?)",
    )?;
    for sequence in &metrics.thinking_sequences {
        insert_thinking.execute(params![
            run_id,
            encode_time(sequence.start_time),
            milliseconds(sequence.duration)?,
            i64::try_from(sequence.token_count)
                .context("thinking token count exceeds SQLite integer")?,
            serde_json::to_string(&sequence.decision_points)?
        ])?;
    }
    let mut insert_recovery = transaction.prepare_cached(
        "INSERT INTO error_recovery_events (
        run_id, error_time, error_type, error_message, recovery_started, recovery_completed,
        recovery_strategy, recovery_success) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )?;
    for recovery in &metrics.error_recoveries {
        insert_recovery.execute(params![
            run_id,
            encode_time(recovery.error_time),
            recovery.error_type,
            recovery.error_message,
            recovery.recovery_started.map(encode_time),
            recovery.recovery_completed.map(encode_time),
            recovery.recovery_strategy,
            recovery.recovery_success
        ])?;
    }
    let swarm = &metrics.swarm_metrics;
    transaction.execute(
        "INSERT INTO swarm_coordination_metrics (
        run_id, timestamp, active_agents, messages_passed, conflicts_resolved,
        consensus_rounds, coordination_efficiency) VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            run_id,
            encode_time(Utc::now()),
            i64::from(swarm.active_agents),
            i64::from(swarm.messages_passed),
            i64::from(swarm.conflicts_resolved),
            0i64,
            swarm.communication_efficiency
        ],
    )?;
    Ok(())
}

fn milliseconds(value: Duration) -> Result<i64> {
    i64::try_from(value.as_millis()).context("duration exceeds SQLite integer milliseconds")
}
