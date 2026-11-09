//! Native supplier-version and durable-storage checks against the actual C library.
#![cfg(not(target_arch = "wasm32"))]

use ruv_swarm_persistence::{models::TaskPriority, AgentModel, SqliteStorage, Storage, TaskModel};

#[test]
fn bundled_sqlite_is_patched_for_mismatched_wal_page_sizes() {
    let connection = rusqlite::Connection::open_in_memory().unwrap();
    let (version, source_id): (String, String) = connection
        .query_row("SELECT sqlite_version(), sqlite_source_id()", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    let number = rusqlite::version_number();
    eprintln!("SQLite runtime: version={version}, version_number={number}, source_id={source_id}");
    // RUSTSEC-2026-0459 / CVE-2026-119375 is fixed in SQLite 3.54.0.
    assert!(
        number >= 3_054_000,
        "linked vulnerable SQLite {version} ({number})"
    );
    assert_eq!(version, rusqlite::version());
}

#[tokio::test]
async fn migrated_wal_database_roundtrips_bound_json_after_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("persistent.sqlite");
    let storage = SqliteStorage::new(path.to_str().unwrap()).await.unwrap();
    let mut agent = AgentModel::new(
        "quote '; DROP TABLE agents; -- 日本語".into(),
        "worker".into(),
        vec!["persist".into()],
    );
    agent.metadata.insert(
        "nested".into(),
        serde_json::json!({"text":"quote ' and \u{0000} byte", "values":[1,2,3]}),
    );
    storage.store_agent(&agent).await.unwrap();
    let task = TaskModel::new(
        "persist".into(),
        serde_json::json!({"agent":agent.id,"unicode":"日本語"}),
        TaskPriority::High,
    );
    storage.store_task(&task).await.unwrap();
    assert_eq!(
        storage.get_agent(&agent.id).await.unwrap(),
        Some(agent.clone())
    );
    assert_eq!(
        storage.get_task(&task.id).await.unwrap(),
        Some(task.clone())
    );
    drop(storage);

    let connection = rusqlite::Connection::open(&path).unwrap();
    let mode: String = connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();
    let migration: u32 = connection
        .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(mode.to_lowercase(), "wal");
    assert_eq!(migration, 4);
    drop(connection);

    let reopened = SqliteStorage::new(path.to_str().unwrap()).await.unwrap();
    assert_eq!(reopened.get_agent(&agent.id).await.unwrap(), Some(agent));
    assert_eq!(reopened.get_task(&task.id).await.unwrap(), Some(task));
    assert!(reopened.get_storage_size().await.unwrap() > 0);
}
