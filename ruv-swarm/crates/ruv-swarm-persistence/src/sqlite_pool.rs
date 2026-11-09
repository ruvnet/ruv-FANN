//! File-backed pool adapter kept with the persistence backend so its SQLite
//! wrapper and bundled C library can be upgraded together.
use rusqlite::Connection;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct SqliteConnectionManager {
    path: PathBuf,
}

impl SqliteConnectionManager {
    pub(crate) fn file(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_owned(),
        }
    }
}

impl r2d2::ManageConnection for SqliteConnectionManager {
    type Connection = Connection;
    type Error = rusqlite::Error;

    fn connect(&self) -> Result<Connection, Self::Error> {
        // Connection::open preserves rusqlite's normal file/URI open flags.
        Connection::open(&self.path)
    }

    fn is_valid(&self, connection: &mut Connection) -> Result<(), Self::Error> {
        connection
            .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
            .map(|_| ())
    }

    fn has_broken(&self, connection: &mut Connection) -> bool {
        // Never lend an unfinished transaction to an unrelated pool borrower.
        // Closing the discarded connection also rolls that transaction back.
        !connection.is_autocommit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2d2::ManageConnection;

    #[test]
    fn pooled_file_connections_share_committed_data() {
        let directory = tempfile::tempdir().unwrap();
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .min_idle(Some(2))
            .build(SqliteConnectionManager::file(
                directory.path().join("pool.sqlite"),
            ))
            .unwrap();
        let first = pool.get().unwrap();
        let second = pool.get().unwrap();
        first
            .execute_batch("CREATE TABLE entries(value TEXT NOT NULL)")
            .unwrap();
        let value = "quote ' ; DROP TABLE entries; -- 日本語";
        first
            .execute("INSERT INTO entries(value) VALUES (?1)", [value])
            .unwrap();
        let actual: String = second
            .query_row("SELECT value FROM entries", [], |row| row.get(0))
            .unwrap();
        assert_eq!(actual, value);
    }

    #[test]
    fn unfinished_transaction_is_discarded_and_rolled_back() {
        let directory = tempfile::tempdir().unwrap();
        let pool = r2d2::Pool::builder()
            .max_size(1)
            .min_idle(Some(1))
            .build(SqliteConnectionManager::file(
                directory.path().join("rollback.sqlite"),
            ))
            .unwrap();
        {
            let connection = pool.get().unwrap();
            connection.execute_batch("CREATE TABLE entries(value INTEGER NOT NULL); BEGIN; INSERT INTO entries VALUES (7);").unwrap();
        }
        let connection = pool.get().unwrap();
        assert!(connection.is_autocommit());
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM entries", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn invalid_database_path_returns_an_error() {
        let directory = tempfile::tempdir().unwrap();
        let manager =
            SqliteConnectionManager::file(directory.path().join("absent/database.sqlite"));
        assert!(manager.connect().is_err());
    }

    #[tokio::test]
    async fn sqlite_query_limits_are_checked_without_truncation() {
        use crate::{AgentModel, EventModel, SqliteStorage, Storage, StorageError};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("query-limits.sqlite");
        let storage = SqliteStorage::new(path.to_str().unwrap()).await.unwrap();
        let agent = AgentModel::new("limits".into(), "worker".into(), vec![]);
        storage.store_agent(&agent).await.unwrap();
        let event = EventModel::new("limits".into(), serde_json::json!({"value":7}))
            .with_agent(agent.id.clone());
        storage.store_event(&event).await.unwrap();
        for limit in [0, 1] {
            assert_eq!(
                storage
                    .get_events_by_agent(&agent.id, limit)
                    .await
                    .unwrap()
                    .len(),
                limit
            );
            assert_eq!(
                storage
                    .get_events_by_type("limits", limit)
                    .await
                    .unwrap()
                    .len(),
                limit
            );
            assert!(storage
                .get_messages_between(&agent.id, "other", limit)
                .await
                .unwrap()
                .is_empty());
        }
        if let Err(expected) = i64::try_from(usize::MAX) {
            let errors = [
                storage
                    .get_events_by_agent(&agent.id, usize::MAX)
                    .await
                    .unwrap_err(),
                storage
                    .get_events_by_type("limits", usize::MAX)
                    .await
                    .unwrap_err(),
                storage
                    .get_messages_between(&agent.id, "other", usize::MAX)
                    .await
                    .unwrap_err(),
            ];
            for error in errors {
                match error {
                    StorageError::Database(message) => assert_eq!(message, expected.to_string()),
                    other => panic!("expected checked SQLite binding failure, got {other:?}"),
                }
            }
        }
        let recovered = storage.get_events_by_type("limits", 1).await.unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].id, event.id);
        assert_eq!(recovered[0].payload, event.payload);
    }
}
