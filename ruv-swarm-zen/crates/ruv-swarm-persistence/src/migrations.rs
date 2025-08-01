//! Database migration management

use crate::StorageError;
use rusqlite::Connection;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Migration definition
pub struct Migration {
    pub version: u32,
    pub name: &'static str,
    pub up: &'static str,
    pub down: Option<&'static str>,
}

/// Migration manager
pub struct MigrationManager {
    migrations: Vec<Migration>,
}

impl MigrationManager {
    /// Create new migration manager
    pub fn new() -> Self {
        Self {
            migrations: Self::load_migrations(),
        }
    }

    /// Load all migrations
    fn load_migrations() -> Vec<Migration> {
        vec![
            Migration {
                version: 1,
                name: "initial_schema",
                up: include_str!("../sql/schema.sql"),
                down: None,
            },
            Migration {
                version: 2,
                name: "add_agent_capabilities_index",
                up: r#"
                    CREATE INDEX IF NOT EXISTS idx_agents_capabilities 
                    ON agents(json_extract(capabilities, '$'));
                "#,
                down: Some("DROP INDEX IF EXISTS idx_agents_capabilities;"),
            },
            Migration {
                version: 3,
                name: "add_task_dependencies_tracking",
                up: r#"
                    CREATE TABLE IF NOT EXISTS task_dependencies (
                        task_id TEXT NOT NULL,
                        depends_on TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        PRIMARY KEY (task_id, depends_on),
                        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                        FOREIGN KEY (depends_on) REFERENCES tasks(id) ON DELETE CASCADE
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_task_deps_task ON task_dependencies(task_id);
                    CREATE INDEX IF NOT EXISTS idx_task_deps_depends ON task_dependencies(depends_on);
                "#,
                down: Some("DROP TABLE IF EXISTS task_dependencies;"),
            },
            Migration {
                version: 4,
                name: "add_agent_groups",
                up: r#"
                    CREATE TABLE IF NOT EXISTS agent_groups (
                        id TEXT PRIMARY KEY NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        metadata TEXT NOT NULL, -- JSON
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL
                    );
                    
                    CREATE TABLE IF NOT EXISTS agent_group_members (
                        group_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        role TEXT,
                        joined_at INTEGER NOT NULL,
                        PRIMARY KEY (group_id, agent_id),
                        FOREIGN KEY (group_id) REFERENCES agent_groups(id) ON DELETE CASCADE,
                        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_group_members_group ON agent_group_members(group_id);
                    CREATE INDEX IF NOT EXISTS idx_group_members_agent ON agent_group_members(agent_id);
                "#,
                down: Some(
                    r#"
                    DROP TABLE IF EXISTS agent_group_members;
                    DROP TABLE IF EXISTS agent_groups;
                "#,
                ),
            },
        ]
    }

    /// Run pending migrations
    pub fn migrate(&self, conn: &Connection) -> Result<(), StorageError> {
        // Ensure migrations table exists
        self.ensure_migrations_table(conn)?;

        // Get current version
        let current_version = self.get_current_version(conn)?;
        info!("Current schema version: {}", current_version);

        // Run pending migrations
        let pending_migrations: Vec<_> = self
            .migrations
            .iter()
            .filter(|m| m.version > current_version)
            .collect();

        if pending_migrations.is_empty() {
            info!("No pending migrations");
            return Ok(());
        }

        info!("Found {} pending migrations", pending_migrations.len());

        for migration in pending_migrations {
            self.run_migration(conn, migration)?;
        }

        Ok(())
    }

    /// Rollback to specific version
    pub fn rollback_to(&self, conn: &Connection, target_version: u32) -> Result<(), StorageError> {
        let current_version = self.get_current_version(conn)?;

        if target_version >= current_version {
            return Err(StorageError::Migration(format!(
                "Cannot rollback to version {} from current version {}",
                target_version, current_version
            )));
        }

        // Get migrations to rollback in reverse order
        let rollback_migrations: Vec<_> = self
            .migrations
            .iter()
            .filter(|m| m.version > target_version && m.version <= current_version)
            .rev()
            .collect();

        for migration in rollback_migrations {
            if let Some(down_sql) = migration.down {
                info!(
                    "Rolling back migration {}: {}",
                    migration.version, migration.name
                );

                conn.execute_batch(down_sql).map_err(|e| {
                    StorageError::Migration(format!(
                        "Failed to rollback migration {}: {}",
                        migration.version, e
                    ))
                })?;

                // Remove migration record
                conn.execute(
                    "DELETE FROM schema_migrations WHERE version = ?1",
                    rusqlite::params![migration.version],
                )
                .map_err(|e| StorageError::Database(e.to_string()))?;

                debug!("Rolled back migration {}", migration.version);
            } else {
                warn!("Migration {} has no rollback script", migration.version);
            }
        }

        Ok(())
    }

    /// Ensure migrations table exists
    fn ensure_migrations_table(&self, conn: &Connection) -> Result<(), StorageError> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                applied_at INTEGER NOT NULL
            );
        "#,
        )
        .map_err(|e| {
            StorageError::Migration(format!("Failed to create migrations table: {}", e))
        })?;

        Ok(())
    }

    /// Get current schema version
    fn get_current_version(&self, conn: &Connection) -> Result<u32, StorageError> {
        let version: Option<u32> = conn
            .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(version.unwrap_or(0))
    }

    /// Run a single migration
    fn run_migration(&self, conn: &Connection, migration: &Migration) -> Result<(), StorageError> {
        info!(
            "Running migration {}: {}",
            migration.version, migration.name
        );

        // Start transaction
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| StorageError::Transaction(e.to_string()))?;

        // Run migration
        tx.execute_batch(migration.up).map_err(|e| {
            StorageError::Migration(format!(
                "Failed to run migration {}: {}",
                migration.version, e
            ))
        })?;

        // Record migration
        tx.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![
                migration.version,
                migration.name,
                chrono::Utc::now().timestamp()
            ],
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        // Commit transaction
        tx.commit()
            .map_err(|e| StorageError::Transaction(e.to_string()))?;

        debug!("Completed migration {}", migration.version);
        Ok(())
    }

    /// Get migration status
    pub fn get_status(&self, conn: &Connection) -> Result<MigrationStatus, StorageError> {
        let current_version = self.get_current_version(conn)?;

        // Get applied migrations
        let mut stmt = conn
            .prepare("SELECT version, name, applied_at FROM schema_migrations ORDER BY version")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let applied_migrations: HashMap<u32, AppliedMigration> = stmt
            .query_map([], |row| {
                Ok(AppliedMigration {
                    version: row.get(0)?,
                    name: row.get(1)?,
                    applied_at: row.get(2)?,
                })
            })
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .map(|m| (m.version, m))
            .collect();

        // Get pending migrations
        let pending: Vec<_> = self
            .migrations
            .iter()
            .filter(|m| !applied_migrations.contains_key(&m.version))
            .map(|m| PendingMigration {
                version: m.version,
                name: m.name.to_string(),
            })
            .collect();

        Ok(MigrationStatus {
            current_version,
            applied: applied_migrations.into_values().collect(),
            pending,
        })
    }
}

impl Default for MigrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration status information
#[derive(Debug)]
pub struct MigrationStatus {
    pub current_version: u32,
    pub applied: Vec<AppliedMigration>,
    pub pending: Vec<PendingMigration>,
}

/// Applied migration information
#[derive(Debug)]
pub struct AppliedMigration {
    pub version: u32,
    pub name: String,
    pub applied_at: i64,
}

/// Pending migration information
#[derive(Debug)]
pub struct PendingMigration {
    pub version: u32,
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_migrations() {
        let temp_file = NamedTempFile::new().unwrap();
        let conn = Connection::open(temp_file.path()).unwrap();

        let manager = MigrationManager::new();

        // Run migrations
        manager.migrate(&conn).unwrap();

        // Check version
        let version = manager.get_current_version(&conn).unwrap();
        assert!(version > 0);

        // Get status
        let status = manager.get_status(&conn).unwrap();
        assert_eq!(status.current_version, version);
        assert!(status.pending.is_empty());
    }
}
