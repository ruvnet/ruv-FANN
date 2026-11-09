//! Async benchmark storage with a synchronized, owned SQLite connection.
//!
//! Filesystem and database operations run on Tokio's blocking pool. An async
//! semaphore admits one operation before spawning, so contended callers wait
//! without occupying blocking threads or blocking an async runtime worker.
use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection, OpenFlags};
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::Semaphore;

mod metrics;
mod query;
mod timestamp;
mod types;
pub use types::*;
#[cfg(test)]
mod tests;

/// SQLite storage preserving the existing benchmark tables and public API.
/// Clones share one connection, including a single `:memory:` database.
#[derive(Clone)]
pub struct BenchmarkStorage {
    connection: Arc<Mutex<Connection>>,
    admission: Arc<Semaphore>,
}

impl BenchmarkStorage {
    /// Open or create a database and initialize its schema atomically.
    pub async fn new(db_path: &Path) -> Result<Self> {
        let db_path = db_path.to_owned();
        let connection = tokio::task::spawn_blocking(move || -> Result<Connection> {
            if let Some(parent) = db_path.parent().filter(|path| !path.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).context("create benchmark database directory")?;
            }
            // Paths remain filenames, rather than being interpreted as URI SQL
            // options. The external mutex serializes access to this connection.
            let mut connection = Connection::open_with_flags(
                &db_path,
                OpenFlags::SQLITE_OPEN_READ_WRITE
                    | OpenFlags::SQLITE_OPEN_CREATE
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
            .context("open benchmark database")?;
            connection.busy_timeout(Duration::from_secs(5))?;
            connection.pragma_update(None, "foreign_keys", "ON")?;
            let transaction = connection.transaction()?;
            transaction.execute_batch(include_str!("storage/schema.sql"))?;
            transaction.commit()?;
            Ok(connection)
        })
        .await
        .context("benchmark database initialization task failed")??;
        tracing::info!("Database schema initialized successfully");
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            admission: Arc::new(Semaphore::new(1)),
        })
    }

    async fn with_connection<R, F>(&self, operation: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<R> + Send + 'static,
    {
        let permit = Arc::clone(&self.admission)
            .acquire_owned()
            .await
            .context("benchmark database admission closed")?;
        let connection = Arc::clone(&self.connection);
        tokio::task::spawn_blocking(move || {
            // A cancelled caller cannot release admission while its SQL is
            // still running: the blocking operation owns both permit and Arc.
            let _permit = permit;
            let mut connection = connection
                .lock()
                .map_err(|_| anyhow::anyhow!("benchmark database connection mutex poisoned"))?;
            operation(&mut connection)
        })
        .await
        .context("benchmark database operation task failed")?
    }

    /// Create a new benchmark run. All user values use bound SQL parameters.
    #[allow(clippy::too_many_arguments)]
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
        let (
            run_id,
            instance_id,
            repository,
            issue_description,
            difficulty,
            execution_mode,
            claude_command,
        ) = (
            run_id.to_owned(),
            instance_id.to_owned(),
            repository.to_owned(),
            issue_description.to_owned(),
            difficulty.to_owned(),
            execution_mode.to_owned(),
            claude_command.to_owned(),
        );
        self.with_connection(move |connection| {
            let configuration = serde_json::json!({
                "execution_mode": execution_mode, "timeout": 1800, "max_retries": 3,
            });
            let environment = serde_json::json!({
                "rust_version": "1.75.0", "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
            });
            connection.prepare_cached("INSERT INTO benchmark_runs (
                run_id, instance_id, repository, issue_description, difficulty,
                execution_mode, start_time, status, claude_command, configuration, environment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)")?.execute(
                params![run_id, instance_id, repository, issue_description, difficulty, execution_mode,
                    encode_time(Utc::now()), claude_command, configuration.to_string(), environment.to_string()])?;
            Ok(())
        }).await
    }

    pub async fn update_run_status(&self, run_id: &str, status: &str) -> Result<()> {
        let (run_id, status) = (run_id.to_owned(), status.to_owned());
        self.with_connection(move |connection| {
            connection
                .prepare_cached(
                    "UPDATE benchmark_runs SET status = ?, end_time = ? WHERE run_id = ?",
                )?
                .execute(params![status, encode_time(Utc::now()), run_id])?;
            Ok(())
        })
        .await
    }

    pub async fn store_stream_event(
        &self,
        run_id: &str,
        event_type: &str,
        event_data: &serde_json::Value,
        sequence: i32,
        relative_time_ms: i64,
    ) -> Result<()> {
        let (run_id, event_type, event_data) =
            (run_id.to_owned(), event_type.to_owned(), event_data.clone());
        self.with_connection(move |connection| {
            connection
                .prepare_cached(
                    "INSERT INTO stream_events (
                run_id, event_type, event_timestamp, relative_time_ms, event_data, sequence_number
                ) VALUES (?, ?, ?, ?, ?, ?)",
                )?
                .execute(params![
                    run_id,
                    event_type,
                    encode_time(Utc::now()),
                    relative_time_ms,
                    event_data.to_string(),
                    sequence
                ])?;
            Ok(())
        })
        .await
    }

    #[allow(clippy::too_many_arguments)]
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
        let (comparison_id, instance_id, baseline_run_id, ml_run_id, summary) = (
            comparison_id.to_owned(),
            instance_id.to_owned(),
            baseline_run_id.to_owned(),
            ml_run_id.to_owned(),
            summary.to_owned(),
        );
        let (improvements, analysis) = (improvements.clone(), analysis.clone());
        self.with_connection(move |connection| {
            connection
                .prepare_cached(
                    "INSERT INTO comparison_results (
                comparison_id, instance_id, baseline_run_id, ml_run_id,
                metric_improvements, statistical_analysis, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?)",
                )?
                .execute(params![
                    comparison_id,
                    instance_id,
                    baseline_run_id,
                    ml_run_id,
                    improvements.to_string(),
                    analysis.to_string(),
                    summary
                ])?;
            Ok(())
        })
        .await
    }
}

// SQLx's former DateTime<Utc> representation: RFC3339, fractional seconds as
// needed, and an explicit +00:00. Keep byte ordering and on-disk data compatible.
fn encode_time(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::AutoSi, false)
}
