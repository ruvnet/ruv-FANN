use super::{timestamp::SqlTimestamp, BenchmarkRun, BenchmarkStorage, MetricRecord};
use anyhow::Result;
use rusqlite::{OptionalExtension, Row};

impl BenchmarkStorage {
    pub async fn get_run(&self, run_id: &str) -> Result<Option<BenchmarkRun>> {
        let run_id = run_id.to_owned();
        self.with_connection(move |connection| {
            Ok(connection
                .prepare_cached("SELECT * FROM benchmark_runs WHERE run_id = ?")?
                .query_row([run_id], decode_run)
                .optional()?)
        })
        .await
    }

    pub async fn get_runs_by_instance(&self, instance_id: &str) -> Result<Vec<BenchmarkRun>> {
        let instance_id = instance_id.to_owned();
        self.with_connection(move |connection| {
            let mut statement = connection.prepare_cached(
                "SELECT * FROM benchmark_runs WHERE instance_id = ? ORDER BY start_time DESC",
            )?;
            let rows = statement.query_map([instance_id], decode_run)?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
    }

    pub async fn get_run_metrics(&self, run_id: &str) -> Result<Vec<MetricRecord>> {
        let run_id = run_id.to_owned();
        self.with_connection(move |connection| {
            let mut statement = connection.prepare_cached(
                "SELECT * FROM performance_metrics WHERE run_id = ? ORDER BY timestamp",
            )?;
            let rows = statement.query_map([run_id], decode_metric)?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
    }
}

fn decode_run(row: &Row<'_>) -> rusqlite::Result<BenchmarkRun> {
    let configuration: String = row.get("configuration")?;
    let environment: String = row.get("environment")?;
    Ok(BenchmarkRun {
        run_id: row.get("run_id")?,
        instance_id: row.get("instance_id")?,
        repository: row.get("repository")?,
        issue_description: row.get("issue_description")?,
        difficulty: row.get("difficulty")?,
        execution_mode: row.get("execution_mode")?,
        start_time: row.get::<_, SqlTimestamp>("start_time")?.0,
        end_time: row
            .get::<_, Option<SqlTimestamp>>("end_time")?
            .map(|value| value.0),
        status: row.get("status")?,
        claude_command: row.get("claude_command")?,
        configuration: serde_json::from_str(&configuration).unwrap_or_default(),
        environment: serde_json::from_str(&environment).unwrap_or_default(),
    })
}

fn decode_metric(row: &Row<'_>) -> rusqlite::Result<MetricRecord> {
    let metadata: Option<String> = row.get("metadata")?;
    Ok(MetricRecord {
        run_id: row.get("run_id")?,
        metric_type: row.get("metric_type")?,
        metric_name: row.get("metric_name")?,
        metric_value: row.get("metric_value")?,
        unit: row.get("unit")?,
        timestamp: row.get::<_, SqlTimestamp>("timestamp")?.0,
        metadata: metadata.and_then(|value| serde_json::from_str(&value).ok()),
    })
}
