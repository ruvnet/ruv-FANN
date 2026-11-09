use super::*;
use crate::metrics::{ErrorRecovery, PerformanceMetrics, ThinkingSequence, ToolInvocation};
use chrono::TimeZone;
use rusqlite::types::Value;
use serde_json::json;
use tempfile::tempdir;

async fn create(storage: &BenchmarkStorage, id: &str, instance: &str) {
    storage
        .create_benchmark_run(
            id,
            instance,
            "test/repo",
            "an issue",
            "easy",
            "baseline",
            "test command",
        )
        .await
        .unwrap();
}

fn metrics() -> PerformanceMetrics {
    let now = Utc.with_ymd_and_hms(2025, 3, 1, 12, 0, 0).unwrap();
    PerformanceMetrics {
        task_completion_time: Duration::from_millis(1234),
        time_to_first_output: Duration::from_millis(42),
        tool_invocations: vec![ToolInvocation {
            tool_name: "Read 'quoted'".into(),
            timestamp: now,
            duration: Duration::from_millis(7),
            parameters: json!({"file":"test's.rs"}),
            result_size: 17,
            success: false,
            error_message: Some("expected error".into()),
        }],
        thinking_sequences: vec![ThinkingSequence {
            start_time: now,
            duration: Duration::from_millis(9),
            token_count: 33,
            decision_points: vec!["inspect".into(), "retry".into()],
        }],
        error_recoveries: vec![ErrorRecovery {
            error_time: now,
            error_type: "tool".into(),
            error_message: "expected error".into(),
            recovery_started: Some(now),
            recovery_completed: None,
            recovery_strategy: Some("retry".into()),
            recovery_success: true,
        }],
        ..PerformanceMetrics::default()
    }
}

#[tokio::test]
async fn database_reopen_crud_query_order_and_parameter_binding() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("nested/benchmark.db");
    let storage = BenchmarkStorage::new(&path).await.unwrap();
    let malicious_id = "run'; DROP TABLE benchmark_runs;--";
    create(&storage, malicious_id, "same-instance").await;
    create(&storage, "newer", "same-instance").await;
    assert!(storage.get_run("missing").await.unwrap().is_none());
    assert!(storage
        .get_runs_by_instance("x' OR 1=1 --")
        .await
        .unwrap()
        .is_empty());
    assert!(storage
        .create_benchmark_run(malicious_id, "x", "r", "i", "easy", "baseline", "c")
        .await
        .is_err());
    storage
        .update_run_status(malicious_id, "completed")
        .await
        .unwrap();
    assert!(storage
        .update_run_status(malicious_id, "not-a-status")
        .await
        .is_err());
    storage
        .with_connection(move |connection| {
            connection.execute(
                "UPDATE benchmark_runs SET start_time = ? WHERE run_id = ?",
                params!["2024-01-01T00:00:00+00:00", malicious_id],
            )?;
            connection.execute(
                "UPDATE benchmark_runs SET start_time = ? WHERE run_id = ?",
                params!["2025-01-01T00:00:00+00:00", "newer"],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let runs = storage.get_runs_by_instance("same-instance").await.unwrap();
    assert_eq!(
        runs.iter()
            .map(|run| run.run_id.as_str())
            .collect::<Vec<_>>(),
        ["newer", malicious_id]
    );
    assert_eq!(runs[1].status, "completed");
    assert!(runs[1].end_time.is_some());
    assert_eq!(runs[1].configuration["execution_mode"], "baseline");
    drop(storage);
    let reopened = BenchmarkStorage::new(&path).await.unwrap();
    assert_eq!(
        reopened
            .get_run(malicious_id)
            .await
            .unwrap()
            .unwrap()
            .repository,
        "test/repo"
    );
    // Delete is intentionally a direct schema regression, not a new public API.
    reopened
        .with_connection(|connection| {
            connection.execute("DELETE FROM benchmark_runs WHERE run_id = ?", ["newer"])?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(reopened.get_run("newer").await.unwrap().is_none());
}

#[tokio::test]
async fn complete_metrics_events_comparisons_and_foreign_keys() {
    let directory = tempdir().unwrap();
    let storage = BenchmarkStorage::new(&directory.path().join("test.db"))
        .await
        .unwrap();
    create(&storage, "baseline", "instance").await;
    create(&storage, "candidate", "instance").await;
    let metrics = metrics();
    storage.store_metrics("baseline", &metrics).await.unwrap();
    let records = storage.get_run_metrics("baseline").await.unwrap();
    assert_eq!(records.len(), 5);
    assert_eq!(
        records
            .iter()
            .find(|metric| metric.metric_type == "TaskCompletionTime")
            .unwrap()
            .metric_value,
        1234.0
    );
    assert_eq!(
        records
            .iter()
            .find(|metric| metric.metric_type == "TimeToFirstOutput")
            .unwrap()
            .unit
            .as_deref(),
        Some("ms")
    );
    assert!(records
        .windows(2)
        .all(|pair| pair[0].timestamp <= pair[1].timestamp));
    storage
        .store_stream_event("baseline", "tool 'event'", &json!({"input":"'; --"}), 2, 17)
        .await
        .unwrap();
    storage
        .store_comparison(
            "compare",
            "instance",
            "baseline",
            "candidate",
            &json!({"time":0.5}),
            &json!({"p_value":1.0}),
            "summary 'quoted'",
        )
        .await
        .unwrap();
    storage.with_connection(move |connection| {
        let tool: (String, i64, String, i64, bool, Option<String>, i32) = connection.query_row(
            "SELECT tool_name,duration_ms,parameters,result_size,success,error_message,sequence_in_run FROM tool_invocations", [],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?)))?;
        assert_eq!(tool, ("Read 'quoted'".into(),7,metrics.tool_invocations[0].parameters.to_string(),17,false,Some("expected error".into()),0));
        let thinking: (i64, i64, String) = connection.query_row(
            "SELECT duration_ms,token_count,decision_points FROM thinking_sequences", [], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
        assert_eq!(thinking, (9,33,"[\"inspect\",\"retry\"]".into()));
        let recovery: (Option<String>, Option<String>, bool) = connection.query_row(
            "SELECT recovery_started,recovery_completed,recovery_success FROM error_recovery_events", [],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
        assert_eq!(recovery, (Some(encode_time(metrics.error_recoveries[0].error_time)),None,true));
        let stream: (String, i64, String, i32) = connection.query_row(
            "SELECT event_type,relative_time_ms,event_data,sequence_number FROM stream_events", [],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?)))?;
        assert_eq!(stream, ("tool 'event'".into(),17,"{\"input\":\"'; --\"}".into(),2));
        let comparison: (String, String, String) = connection.query_row(
            "SELECT metric_improvements,statistical_analysis,summary FROM comparison_results", [],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
        assert_eq!(comparison, ("{\"time\":0.5}".into(),"{\"p_value\":1.0}".into(),"summary 'quoted'".into()));
        let foreign_keys: bool = connection.pragma_query_value(None, "foreign_keys", |row| row.get(0))?;
        assert!(foreign_keys);
        assert!(connection.execute("DELETE FROM benchmark_runs WHERE run_id = ?", ["baseline"]).is_err());
        Ok(())
    }).await.unwrap();
    assert!(storage
        .store_stream_event("missing", "event", &json!({}), 0, 0)
        .await
        .is_err());
}

#[tokio::test]
async fn metrics_batch_rolls_back_on_mid_batch_failure_then_recovers() {
    let storage = BenchmarkStorage::new(Path::new(":memory:")).await.unwrap();
    create(&storage, "run", "instance").await;
    storage
        .with_connection(|connection| {
            connection.execute_batch(
                "CREATE TRIGGER reject_tool BEFORE INSERT ON tool_invocations
            BEGIN SELECT RAISE(ABORT, 'forced test failure'); END;",
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(storage.store_metrics("run", &metrics()).await.is_err());
    assert!(storage.get_run_metrics("run").await.unwrap().is_empty());
    storage
        .with_connection(|connection| {
            for table in [
                "tool_invocations",
                "thinking_sequences",
                "error_recovery_events",
                "swarm_coordination_metrics",
            ] {
                let count: i64 =
                    connection.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                        row.get(0)
                    })?;
                assert_eq!(count, 0);
            }
            connection.execute_batch("DROP TRIGGER reject_tool")?;
            Ok(())
        })
        .await
        .unwrap();
    storage.store_metrics("run", &metrics()).await.unwrap();
    assert_eq!(storage.get_run_metrics("run").await.unwrap().len(), 5);
    assert!(storage.store_metrics("unknown", &metrics()).await.is_err());
    assert!(storage.get_run_metrics("unknown").await.unwrap().is_empty());
}

#[tokio::test]
async fn legacy_schema_and_timestamp_representations_remain_readable() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("legacy.db");
    // A database created before the new backend: exact previous table schema,
    // RFC3339 run values, SQLite defaults, and SQLx's other accepted date forms.
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(include_str!("schema.sql"))
        .unwrap();
    connection
        .execute(
            "INSERT INTO benchmark_runs (run_id,instance_id,repository,difficulty,execution_mode,
        start_time,status,claude_command,configuration,environment) VALUES (?,?,?,?,?,?,?,?,?,?)",
            params![
                "legacy",
                "instance",
                "repo",
                "easy",
                "baseline",
                "2025-03-01T12:00:00.125+00:00",
                "running",
                "cmd",
                "bad-json",
                "{}"
            ],
        )
        .unwrap();
    let timestamps = [
        Value::Text("2025-03-01T12:00:00.125+00:00".into()),
        Value::Text("2025-03-01 12:00:00".into()),
        Value::Integer(0),
        // TIMESTAMP has NUMERIC affinity: use a fractional Julian day so SQLite
        // retains REAL storage instead of converting an integral value to INTEGER.
        Value::Real(2_440_588.25),
        Value::Text("2025-03-01T12:00Z".into()),
        Value::Integer(2_440_588),
    ];
    for (index, value) in timestamps.into_iter().enumerate() {
        connection.execute("INSERT INTO performance_metrics (run_id,metric_type,metric_name,metric_value,timestamp,metadata)
            VALUES ('legacy','test',?,1,?,?)", params![format!("metric-{index}"),value,if index == 0 {"{\"source\":\"legacy\"}"} else {"bad-json"}]).unwrap();
    }
    let stored_types: (String, String) = connection.query_row(
        "SELECT (SELECT typeof(timestamp) FROM performance_metrics WHERE metric_name = 'metric-3'),
                (SELECT typeof(timestamp) FROM performance_metrics WHERE metric_name = 'metric-5')",
        [], |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(stored_types, ("real".into(), "integer".into()));
    drop(connection);
    let storage = BenchmarkStorage::new(&path).await.unwrap();
    let run = storage.get_run("legacy").await.unwrap().unwrap();
    assert_eq!(run.start_time.timestamp_subsec_millis(), 125);
    assert!(run.configuration.is_null()); // Preserve the existing JSON fallback.
    let records = storage.get_run_metrics("legacy").await.unwrap();
    assert_eq!(records.len(), 6);
    assert_eq!(
        records
            .iter()
            .find(|row| row.metric_name == "metric-2")
            .unwrap()
            .timestamp
            .timestamp(),
        0
    );
    assert_eq!(
        records
            .iter()
            .find(|row| row.metric_name == "metric-3")
            .unwrap()
            .timestamp
            .timestamp(),
        64_800
    );
    assert_eq!(
        records
            .iter()
            .find(|row| row.metric_name == "metric-5")
            .unwrap()
            .timestamp
            .timestamp(),
        2_440_588
    );
    assert_eq!(
        records
            .iter()
            .find(|row| row.metric_name == "metric-0")
            .unwrap()
            .metadata,
        Some(json!({"source":"legacy"}))
    );
    assert!(records
        .iter()
        .filter(|row| row.metric_name != "metric-0")
        .all(|row| row.metadata.is_none()));
}

#[tokio::test]
async fn concurrent_callers_share_memory_database_without_losing_writes() {
    let storage = BenchmarkStorage::new(Path::new(":memory:")).await.unwrap();
    create(&storage, "run", "instance").await;
    let mut writes = Vec::new();
    for sequence in 0..32 {
        let storage = storage.clone();
        writes.push(tokio::spawn(async move {
            storage
                .store_stream_event(
                    "run",
                    "event",
                    &json!({"sequence":sequence}),
                    sequence,
                    i64::from(sequence),
                )
                .await
        }));
    }
    for write in writes {
        write.await.unwrap().unwrap();
    }
    storage
        .with_connection(|connection| {
            let (count, unique): (i64, i64) = connection.query_row(
                "SELECT count(*),count(DISTINCT sequence_number) FROM stream_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            assert_eq!((count, unique), (32, 32));
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn sqlite_lock_wait_does_not_block_the_async_runtime() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("locked.db");
    let storage = BenchmarkStorage::new(&path).await.unwrap();
    let (locked_sender, locked_receiver) = tokio::sync::oneshot::channel();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let lock = tokio::task::spawn_blocking(move || {
        let connection = Connection::open(path).unwrap();
        connection.execute_batch("BEGIN EXCLUSIVE").unwrap();
        locked_sender.send(()).unwrap();
        release_receiver.recv().unwrap();
        connection.execute_batch("ROLLBACK").unwrap();
    });
    locked_receiver.await.unwrap();
    let start = std::time::Instant::now();
    let write = tokio::spawn(async move {
        storage
            .create_benchmark_run(
                "run", "instance", "repo", "issue", "easy", "baseline", "cmd",
            )
            .await
    });
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(5)).await;
    let remained_responsive = start.elapsed() < Duration::from_secs(2);
    release_sender.send(()).unwrap();
    lock.await.unwrap();
    write.await.unwrap().unwrap();
    assert!(
        remained_responsive,
        "SQLite busy wait blocked the current-thread runtime"
    );
}

#[tokio::test]
async fn cancelled_waiter_keeps_connection_owned_until_its_transaction_finishes() {
    let storage = BenchmarkStorage::new(Path::new(":memory:")).await.unwrap();
    create(&storage, "run", "instance").await;
    let (started_sender, started_receiver) = tokio::sync::oneshot::channel();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let writer_storage = storage.clone();
    let writer = tokio::spawn(async move {
        writer_storage
            .with_connection(move |connection| {
                let transaction = connection.transaction()?;
                transaction.execute(
                    "UPDATE benchmark_runs SET status = 'completed' WHERE run_id = ?",
                    ["run"],
                )?;
                started_sender.send(()).unwrap();
                release_receiver.recv().unwrap();
                transaction.commit()?;
                Ok(())
            })
            .await
    });
    started_receiver.await.unwrap();
    writer.abort();
    assert!(writer.await.unwrap_err().is_cancelled());
    // Cancellation drops the awaiting future, not the running SQL operation.
    assert_eq!(storage.admission.available_permits(), 0);
    let reader_storage = storage.clone();
    let reader = tokio::spawn(async move { reader_storage.get_run("run").await });
    tokio::task::yield_now().await;
    assert!(!reader.is_finished());
    release_sender.send(()).unwrap();
    assert_eq!(reader.await.unwrap().unwrap().unwrap().status, "completed");
}
