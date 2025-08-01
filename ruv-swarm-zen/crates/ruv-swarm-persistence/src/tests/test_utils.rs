//! Test utilities for database isolation and fixture management

use crate::{Storage, StorageError};
#[cfg(not(target_arch = "wasm32"))]
use crate::sqlite::SqliteStorage;
use crate::memory::MemoryStorage;
use tempfile::NamedTempFile;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::debug;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Test fixture with automatic cleanup
pub struct TestFixture {
    storage: Box<dyn Storage<Error = StorageError>>,
    #[allow(dead_code)]
    _temp_file: Option<NamedTempFile>,
}

impl TestFixture {
    /// Create a new test fixture with isolated database
    pub async fn new() -> Result<Self, StorageError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let temp_file = NamedTempFile::new()
                .map_err(|e| StorageError::Other(format!("Failed to create temp file: {}", e)))?;
            let path = temp_file.path().to_str()
                .ok_or_else(|| StorageError::Other("Invalid temp file path".to_string()))?;
            
            let storage = SqliteStorage::new(path).await?;
            
            Ok(Self {
                storage: Box::new(storage),
                _temp_file: Some(temp_file),
            })
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            Ok(Self {
                storage: Box::new(MemoryStorage::new()),
                _temp_file: None,
            })
        }
    }
    
    /// Create a test fixture with custom pool size
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn with_pool_size(pool_size: u32) -> Result<Self, StorageError> {
        let temp_file = NamedTempFile::new()
            .map_err(|e| StorageError::Other(format!("Failed to create temp file: {}", e)))?;
        let path = temp_file.path().to_str()
            .ok_or_else(|| StorageError::Other("Invalid temp file path".to_string()))?;
        
        // Create storage with custom pool size
        let manager = r2d2_sqlite::SqliteConnectionManager::file(path);
        let pool = r2d2::Pool::builder()
            .max_size(pool_size)
            .min_idle(Some(2))
            .connection_timeout(std::time::Duration::from_millis(100))
            .build(manager)
            .map_err(|e| StorageError::Pool(e.to_string()))?;
        
        // Initialize schema
        let conn = pool.get().map_err(|e| StorageError::Pool(e.to_string()))?;
        conn.execute_batch(include_str!("../../sql/schema.sql"))
            .map_err(|e| StorageError::Migration(format!("Schema initialization failed: {}", e)))?;
        
        // Configure for testing
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            PRAGMA busy_timeout = 10000;
            PRAGMA foreign_keys = ON;
            "#
        ).map_err(|e| StorageError::Database(format!("Failed to configure SQLite: {}", e)))?;
        
        drop(conn);
        
        // Create storage wrapper
        let storage = SqliteStorage::from_pool(pool).await?;
        
        Ok(Self {
            storage: Box::new(storage),
            _temp_file: Some(temp_file),
        })
    }
    
    /// Get the storage instance
    pub fn storage(&self) -> &dyn Storage<Error = StorageError> {
        &*self.storage
    }
    
    /// Get mutable storage instance
    pub fn storage_mut(&mut self) -> &mut dyn Storage<Error = StorageError> {
        &mut *self.storage
    }
}

// Automatic cleanup on drop
impl Drop for TestFixture {
    fn drop(&mut self) {
        // Database file automatically deleted when TempFile drops
    }
}

/// Create isolated SQLite storage for testing
#[cfg(not(target_arch = "wasm32"))]
pub async fn create_test_storage() -> Result<SqliteStorage, StorageError> {
    let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let temp_file = NamedTempFile::new()
        .map_err(|e| StorageError::Other(format!("Failed to create temp file: {}", e)))?;
    
    let path = temp_file.path().to_str()
        .ok_or_else(|| StorageError::Other("Invalid temp file path".to_string()))?
        .to_owned();
    
    // Keep file alive for test duration
    std::mem::forget(temp_file);
    
    let storage = SqliteStorage::new(&path).await?;
    
    // Configure for testing
    let conn = storage.get_conn_test()?;
    conn.execute_batch(
        "PRAGMA journal_mode = MEMORY;
         PRAGMA synchronous = OFF;
         PRAGMA temp_store = MEMORY;"
    ).map_err(|e| StorageError::Database(format!("Failed to configure SQLite: {}", e)))?;
    
    debug!("Created test storage {}", test_id);
    Ok(storage)
}

/// Create in-memory storage for unit tests
pub fn create_memory_storage() -> MemoryStorage {
    MemoryStorage::new()
}

/// Test configuration options
#[derive(Default)]
pub struct TestConfig {
    pub pool_size: Option<u32>,
    pub timeout_ms: Option<u64>,
    pub enable_wal: bool,
}

impl TestConfig {
    /// Configuration for high concurrency tests
    pub fn high_concurrency() -> Self {
        Self {
            pool_size: Some(32),
            timeout_ms: Some(100),
            enable_wal: true,
        }
    }
    
    /// Configuration for low latency tests
    pub fn low_latency() -> Self {
        Self {
            pool_size: Some(4),
            timeout_ms: Some(10),
            enable_wal: false,
        }
    }
}

/// Assertion helpers for tests
pub mod assertions {
    use crate::{Storage, StorageError};
    use std::collections::HashSet;
    
    /// Verify referential integrity across storage
    pub async fn assert_referential_integrity(
        storage: &dyn Storage<Error = StorageError>
    ) -> Result<(), String> {
        let agents = storage.list_agents().await
            .map_err(|e| format!("Failed to list agents: {}", e))?;
        let tasks = storage.get_pending_tasks().await
            .map_err(|e| format!("Failed to get tasks: {}", e))?;
        
        let agent_ids: HashSet<_> = agents.iter()
            .map(|a| &a.id)
            .collect();
        
        for task in tasks {
            if let Some(agent_id) = &task.assigned_to {
                if !agent_ids.contains(agent_id) {
                    return Err(format!(
                        "Task {} assigned to non-existent agent {}",
                        task.id, agent_id
                    ));
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fixture_isolation() {
        let fixture1 = TestFixture::new().await.unwrap();
        let fixture2 = TestFixture::new().await.unwrap();
        
        // Each fixture should have its own isolated database
        let agent = crate::models::AgentModel::new(
            "test-agent".to_string(),
            "worker".to_string(),
            vec!["compute".to_string()],
        );
        
        // Store in first fixture
        fixture1.storage().store_agent(&agent).await.unwrap();
        
        // Should not exist in second fixture
        let result = fixture2.storage().get_agent(&agent.id).await.unwrap();
        assert!(result.is_none());
        
        // Should exist in first fixture
        let result = fixture1.storage().get_agent(&agent.id).await.unwrap();
        assert!(result.is_some());
    }
}