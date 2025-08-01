//! SQLite storage module for benchmark data

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
use std::path::Path;
use tracing::info;

use crate::metrics::{
    ErrorRecovery, MetricType, PerformanceMetrics, ThinkingSequence, ToolInvocation,
};

/// Benchmark storage using SQLite
pub struct BenchmarkStorage {
    pool: Pool<Sqlite>,
}

impl BenchmarkStorage {
    /// Create a new storage instance
    pub async fn new(db_path: &Path) -> Result<Self> {
        // Create database directory if it doesn't exist
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let db_url = format!("sqlite:{}", db_path.display());

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await?;

        let storage = Self { pool };
        storage.initialize_schema().await?;

        Ok(storage)
    }

    /// Initialize database schema
    async fn initialize_schema(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                instance_id TEXT NOT NULL,
                repository TEXT NOT NULL,
                issue_description TEXT,
                difficulty TEXT CHECK(difficulty IN ('easy', 'medium', 'hard')),
                execution_mode TEXT NOT NULL CHECK(execution_mode IN ('baseline', 'ml_optimized')),
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'timeout')),
                claude_command TEXT NOT NULL,
                configuration TEXT NOT NULL,
                environment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS stream_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_timestamp TIMESTAMP NOT NULL,
                relative_time_ms INTEGER NOT NULL,
                event_data TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS tool_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                invocation_time TIMESTAMP NOT NULL,
                duration_ms INTEGER,
                parameters TEXT,
                result_size INTEGER,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                sequence_in_run INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS thinking_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                duration_ms INTEGER NOT NULL,
                token_count INTEGER,
                decision_points TEXT,
                context_before TEXT,
                context_after TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS error_recovery_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                error_time TIMESTAMP NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT,
                recovery_started TIMESTAMP,
                recovery_completed TIMESTAMP,
                recovery_strategy TEXT,
                recovery_success BOOLEAN,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                unit TEXT,
                timestamp TIMESTAMP NOT NULL,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS code_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                quality_score REAL NOT NULL,
                complexity_score INTEGER NOT NULL,
                test_coverage REAL,
                documentation_score REAL,
                security_score REAL,
                issues TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                cpu_usage REAL NOT NULL,
                memory_usage REAL NOT NULL,
                disk_io_read REAL,
                disk_io_write REAL,
                network_in REAL,
                network_out REAL,
                agent_count INTEGER,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS swarm_coordination_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                active_agents INTEGER NOT NULL,
                messages_passed INTEGER NOT NULL,
                conflicts_resolved INTEGER,
                consensus_rounds INTEGER,
                coordination_efficiency REAL,
                task_distribution TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS comparison_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_id TEXT UNIQUE NOT NULL,
                instance_id TEXT NOT NULL,
                baseline_run_id TEXT NOT NULL,
                ml_run_id TEXT NOT NULL,
                metric_improvements TEXT NOT NULL,
                statistical_analysis TEXT NOT NULL,
                patch_diff_analysis TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (baseline_run_id) REFERENCES benchmark_runs(run_id),
                FOREIGN KEY (ml_run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS swe_bench_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                tests_passed INTEGER NOT NULL,
                tests_failed INTEGER NOT NULL,
                patch_applied BOOLEAN NOT NULL,
                files_modified TEXT NOT NULL,
                patch_size_bytes INTEGER,
                validation_output TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create indexes for performance
        self.create_indexes().await?;

        info!("Database schema initialized successfully");
        Ok(())
    }

    /// Create indexes for better query performance
    async fn create_indexes(&self) -> Result<()> {
        let indexes = vec![
            "CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON performance_metrics(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_type ON performance_metrics(metric_type)",
            "CREATE INDEX IF NOT EXISTS idx_resource_run_timestamp ON resource_usage(run_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_comparison_instance ON comparison_results(instance_id)",
            "CREATE INDEX IF NOT EXISTS idx_stream_events_run ON stream_events(run_id, sequence_number)",
            "CREATE INDEX IF NOT EXISTS idx_tool_invocations_run ON tool_invocations(run_id, invocation_time)",
            "CREATE INDEX IF NOT EXISTS idx_thinking_run ON thinking_sequences(run_id, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_error_recovery_run ON error_recovery_events(run_id, error_time)",
            "CREATE INDEX IF NOT EXISTS idx_benchmark_instance ON benchmark_runs(instance_id, execution_mode)",
        ];

        for index_sql in indexes {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }

        Ok(())
    }

    /// Create a new benchmark run
    pub async fn create_benchmark_run(
        &self,
        run_id: &str,
        instance_id: &str,
        repository: &str,
        issue_description: &str,
        difficulty: &str,
        execution_mode: &str,
        claude_command: &str,
    ) -> Result<()> {
        let config = serde_json::json!({
            "execution_mode": execution_mode,
            "timeout": 1800,
            "max_retries": 3,
        });

        let environment = serde_json::json!({
            "rust_version": "1.75.0",
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        });

        sqlx::query(
            r#"
            INSERT INTO benchmark_runs (
                run_id, instance_id, repository, issue_description,
                difficulty, execution_mode, start_time, status,
                claude_command, configuration, environment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
            "#,
        )
        .bind(run_id)
        .bind(instance_id)
        .bind(repository)
        .bind(issue_description)
        .bind(difficulty)
        .bind(execution_mode)
        .bind(Utc::now())
        .bind(claude_command)
        .bind(config.to_string())
        .bind(environment.to_string())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update run status
    pub async fn update_run_status(&self, run_id: &str, status: &str) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE benchmark_runs 
            SET status = ?, end_time = ?
            WHERE run_id = ?
            "#,
        )
        .bind(status)
        .bind(Utc::now())
        .bind(run_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store performance metrics
    pub async fn store_metrics(&self, run_id: &str, metrics: &PerformanceMetrics) -> Result<()> {
        // Store individual metrics
        self.store_metric(
            run_id,
            MetricType::TaskCompletionTime,
            metrics.task_completion_time.as_millis() as f64,
        )
        .await?;
        self.store_metric(
            run_id,
            MetricType::TimeToFirstOutput,
            metrics.time_to_first_output.as_millis() as f64,
        )
        .await?;
        self.store_metric(
            run_id,
            MetricType::CodeQualityOverall,
            metrics.code_quality_score.overall,
        )
        .await?;
        self.store_metric(
            run_id,
            MetricType::CpuUsageAverage,
            metrics.cpu_usage.average,
        )
        .await?;
        self.store_metric(
            run_id,
            MetricType::MemoryUsageAverage,
            metrics.memory_usage.average,
        )
        .await?;

        // Store tool invocations
        for (i, tool) in metrics.tool_invocations.iter().enumerate() {
            self.store_tool_invocation(run_id, tool, i as i32).await?;
        }

        // Store thinking sequences
        for sequence in &metrics.thinking_sequences {
            self.store_thinking_sequence(run_id, sequence).await?;
        }

        // Store error recoveries
        for recovery in &metrics.error_recoveries {
            self.store_error_recovery(run_id, recovery).await?;
        }

        // Store swarm metrics
        self.store_swarm_metrics(run_id, &metrics.swarm_metrics)
            .await?;

        Ok(())
    }

    /// Store a single metric
    async fn store_metric(&self, run_id: &str, metric_type: MetricType, value: f64) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO performance_metrics (
                run_id, metric_type, metric_name, metric_value, unit, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(run_id)
        .bind(metric_type.to_string())
        .bind(metric_type.to_string())
        .bind(value)
        .bind(metric_type.unit())
        .bind(Utc::now())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store tool invocation
    async fn store_tool_invocation(
        &self,
        run_id: &str,
        tool: &ToolInvocation,
        sequence: i32,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO tool_invocations (
                run_id, tool_name, invocation_time, duration_ms,
                parameters, result_size, success, error_message, sequence_in_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(run_id)
        .bind(&tool.tool_name)
        .bind(tool.timestamp)
        .bind(tool.duration.as_millis() as i64)
        .bind(tool.parameters.to_string())
        .bind(tool.result_size as i64)
        .bind(tool.success)
        .bind(&tool.error_message)
        .bind(sequence)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store thinking sequence
    async fn store_thinking_sequence(
        &self,
        run_id: &str,
        sequence: &ThinkingSequence,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO thinking_sequences (
                run_id, start_time, duration_ms, token_count, decision_points
            ) VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(run_id)
        .bind(sequence.start_time)
        .bind(sequence.duration.as_millis() as i64)
        .bind(sequence.token_count as i64)
        .bind(serde_json::to_string(&sequence.decision_points)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store error recovery event
    async fn store_error_recovery(&self, run_id: &str, recovery: &ErrorRecovery) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO error_recovery_events (
                run_id, error_time, error_type, error_message,
                recovery_started, recovery_completed, recovery_strategy, recovery_success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(run_id)
        .bind(recovery.error_time)
        .bind(&recovery.error_type)
        .bind(&recovery.error_message)
        .bind(recovery.recovery_started)
        .bind(recovery.recovery_completed)
        .bind(&recovery.recovery_strategy)
        .bind(recovery.recovery_success)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store swarm coordination metrics
    async fn store_swarm_metrics(
        &self,
        run_id: &str,
        metrics: &crate::metrics::SwarmCoordinationMetrics,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO swarm_coordination_metrics (
                run_id, timestamp, active_agents, messages_passed,
                conflicts_resolved, consensus_rounds, coordination_efficiency
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(run_id)
        .bind(Utc::now())
        .bind(metrics.active_agents as i64)
        .bind(metrics.messages_passed as i64)
        .bind(metrics.conflicts_resolved as i64)
        .bind(0i64) // consensus_rounds placeholder
        .bind(metrics.communication_efficiency)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store stream event
    pub async fn store_stream_event(
        &self,
        run_id: &str,
        event_type: &str,
        event_data: &serde_json::Value,
        sequence: i32,
        relative_time_ms: i64,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO stream_events (
                run_id, event_type, event_timestamp, relative_time_ms,
                event_data, sequence_number
            ) VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(run_id)
        .bind(event_type)
        .bind(Utc::now())
        .bind(relative_time_ms)
        .bind(event_data.to_string())
        .bind(sequence)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store comparison result
    pub async fn store_comparison(
        &self,
        comparison_id: &str,
        instance_id: &str,
        baseline_run_id: &str,
        ml_run_id: &str,
        improvements: &serde_json::Value,
        analysis: &serde_json::Value,
        summary: &str,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO comparison_results (
                comparison_id, instance_id, baseline_run_id, ml_run_id,
                metric_improvements, statistical_analysis, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(comparison_id)
        .bind(instance_id)
        .bind(baseline_run_id)
        .bind(ml_run_id)
        .bind(improvements.to_string())
        .bind(analysis.to_string())
        .bind(summary)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get benchmark run by ID
    pub async fn get_run(&self, run_id: &str) -> Result<Option<BenchmarkRun>> {
        let row = sqlx::query_as::<_, BenchmarkRunRow>(
            r#"
            SELECT * FROM benchmark_runs WHERE run_id = ?
            "#,
        )
        .bind(run_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| r.into()))
    }

    /// Get runs by instance ID
    pub async fn get_runs_by_instance(&self, instance_id: &str) -> Result<Vec<BenchmarkRun>> {
        let rows = sqlx::query_as::<_, BenchmarkRunRow>(
            r#"
            SELECT * FROM benchmark_runs 
            WHERE instance_id = ?
            ORDER BY start_time DESC
            "#,
        )
        .bind(instance_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| r.into()).collect())
    }

    /// Get metrics for a run
    pub async fn get_run_metrics(&self, run_id: &str) -> Result<Vec<MetricRecord>> {
        let rows = sqlx::query_as::<_, MetricRow>(
            r#"
            SELECT * FROM performance_metrics
            WHERE run_id = ?
            ORDER BY timestamp
            "#,
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| r.into()).collect())
    }
}

// Database row types
#[derive(sqlx::FromRow)]
struct BenchmarkRunRow {
    id: i64,
    run_id: String,
    instance_id: String,
    repository: String,
    issue_description: Option<String>,
    difficulty: Option<String>,
    execution_mode: String,
    start_time: DateTime<Utc>,
    end_time: Option<DateTime<Utc>>,
    status: String,
    claude_command: String,
    configuration: String,
    environment: String,
    created_at: DateTime<Utc>,
}

#[derive(sqlx::FromRow)]
struct MetricRow {
    id: i64,
    run_id: String,
    metric_type: String,
    metric_name: String,
    metric_value: f64,
    unit: Option<String>,
    timestamp: DateTime<Utc>,
    metadata: Option<String>,
}

// Public types
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

impl From<BenchmarkRunRow> for BenchmarkRun {
    fn from(row: BenchmarkRunRow) -> Self {
        Self {
            run_id: row.run_id,
            instance_id: row.instance_id,
            repository: row.repository,
            issue_description: row.issue_description,
            difficulty: row.difficulty,
            execution_mode: row.execution_mode,
            start_time: row.start_time,
            end_time: row.end_time,
            status: row.status,
            claude_command: row.claude_command,
            configuration: serde_json::from_str(&row.configuration).unwrap_or_default(),
            environment: serde_json::from_str(&row.environment).unwrap_or_default(),
        }
    }
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

impl From<MetricRow> for MetricRecord {
    fn from(row: MetricRow) -> Self {
        Self {
            run_id: row.run_id,
            metric_type: row.metric_type,
            metric_name: row.metric_name,
            metric_value: row.metric_value,
            unit: row.unit,
            timestamp: row.timestamp,
            metadata: row.metadata.and_then(|s| serde_json::from_str(&s).ok()),
        }
    }
}

// Additional public types for completeness
pub type StreamEvent = serde_json::Value;
pub type ToolInvocationRecord = ToolInvocation;
pub type ThinkingSequenceRecord = ThinkingSequence;
pub type ErrorRecoveryEvent = ErrorRecovery;
pub type CodeQualityMetric = serde_json::Value;
pub type ResourceUsageRecord = serde_json::Value;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_storage_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let storage = BenchmarkStorage::new(&db_path).await;
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_benchmark_run_crud() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let storage = BenchmarkStorage::new(&db_path).await.unwrap();

        // Create a run
        let run_id = "test-run-001";
        storage
            .create_benchmark_run(
                run_id,
                "test-instance",
                "test/repo",
                "Test issue",
                "easy",
                "baseline",
                "test command",
            )
            .await
            .unwrap();

        // Get the run
        let run = storage.get_run(run_id).await.unwrap();
        assert!(run.is_some());

        let run = run.unwrap();
        assert_eq!(run.run_id, run_id);
        assert_eq!(run.status, "running");

        // Update status
        storage
            .update_run_status(run_id, "completed")
            .await
            .unwrap();

        // Verify update
        let updated_run = storage.get_run(run_id).await.unwrap().unwrap();
        assert_eq!(updated_run.status, "completed");
        assert!(updated_run.end_time.is_some());
    }
}
