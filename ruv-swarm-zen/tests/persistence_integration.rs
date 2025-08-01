//! Integration tests for swarm state persistence and recovery
//! 
//! These tests validate state management, checkpointing, recovery,
//! and distributed state synchronization across the swarm.

use ruv_swarm_core::{
    agent::{Agent, AgentId, AgentType, AgentState},
    swarm::{Swarm, SwarmConfig, SwarmState, Topology},
    task::{Task, TaskState, TaskResult},
    checkpoint::{Checkpoint, CheckpointManager},
};
use ruv_swarm_persistence::{
    PersistenceBackend, SqlitePersistence, RedisPersistence,
    MemoryPersistence, S3Persistence, PersistenceError,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tempfile::TempDir;
use std::path::PathBuf;

/// Helper to create a test swarm with specific persistence backend
async fn create_swarm_with_persistence(
    backend: Box<dyn PersistenceBackend>,
) -> Result<Swarm, Box<dyn std::error::Error>> {
    let config = SwarmConfig {
        topology: Topology::Star,
        max_agents: 10,
        heartbeat_interval: std::time::Duration::from_secs(1),
        task_timeout: std::time::Duration::from_secs(30),
        persistence: backend,
        checkpoint_interval: Some(std::time::Duration::from_secs(5)),
    };
    
    Swarm::new(config).await.map_err(Into::into)
}

/// Helper to generate test task data
fn create_test_task() -> Task {
    Task::DataProcessing {
        input_data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        operations: vec!["normalize", "transform", "aggregate"],
        output_format: "json",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_persistence_basic() {
        let persistence = Box::new(MemoryPersistence::new());
        let mut swarm = create_swarm_with_persistence(persistence).await.unwrap();
        
        // Create and save agent state
        let agent_id = swarm.spawn(AgentType::DataProcessor).await.unwrap();
        
        // Save swarm state
        let checkpoint = swarm.create_checkpoint().await.unwrap();
        swarm.save_checkpoint(&checkpoint).await.unwrap();
        
        // Retrieve and verify
        let loaded_checkpoint = swarm.load_checkpoint(checkpoint.id()).await.unwrap();
        assert_eq!(checkpoint.id(), loaded_checkpoint.id());
        assert_eq!(checkpoint.agent_count(), 1);
        assert!(checkpoint.has_agent(&agent_id));
    }

    #[tokio::test]
    async fn test_sqlite_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("swarm_state.db");
        
        let persistence = Box::new(
            SqlitePersistence::new(db_path.to_str().unwrap()).await.unwrap()
        );
        
        let mut swarm = create_swarm_with_persistence(persistence).await.unwrap();
        
        // Create complex state
        let agents = vec![
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::DataAnalyzer).await.unwrap(),
            swarm.spawn(AgentType::Coordinator).await.unwrap(),
        ];
        
        // Start a task
        let task = create_test_task();
        let task_id = swarm.submit_task(task).await.unwrap();
        
        // Create checkpoint
        let checkpoint = swarm.create_checkpoint().await.unwrap();
        swarm.save_checkpoint(&checkpoint).await.unwrap();
        
        // Simulate restart - create new swarm with same persistence
        drop(swarm);
        
        let persistence2 = Box::new(
            SqlitePersistence::new(db_path.to_str().unwrap()).await.unwrap()
        );
        let mut swarm2 = create_swarm_with_persistence(persistence2).await.unwrap();
        
        // Restore from checkpoint
        let checkpoints = swarm2.list_checkpoints().await.unwrap();
        assert_eq!(checkpoints.len(), 1);
        
        swarm2.restore_from_checkpoint(checkpoint.id()).await.unwrap();
        
        // Verify state was restored
        assert_eq!(swarm2.agent_count(), 3);
        for agent_id in &agents {
            assert!(swarm2.has_agent(agent_id));
        }
        
        // Verify task state
        let task_state = swarm2.get_task_state(&task_id).await.unwrap();
        assert!(matches!(task_state, TaskState::Pending | TaskState::Running));
    }

    #[tokio::test]
    async fn test_distributed_state_sync() {
        // Create multiple swarm instances with shared persistence
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("shared_state.db");
        
        let mut swarms = vec![];
        for i in 0..3 {
            let persistence = Box::new(
                SqlitePersistence::new(db_path.to_str().unwrap()).await.unwrap()
            );
            let swarm = create_swarm_with_persistence(persistence).await.unwrap();
            swarms.push(swarm);
        }
        
        // Make changes in first swarm
        let agent_id = swarms[0].spawn(AgentType::NeuralProcessor).await.unwrap();
        swarms[0].sync_state().await.unwrap();
        
        // Verify other swarms can see the change
        for swarm in &mut swarms[1..] {
            swarm.sync_state().await.unwrap();
            assert!(swarm.has_agent(&agent_id));
        }
    }

    #[tokio::test]
    async fn test_checkpoint_versioning() {
        let persistence = Box::new(MemoryPersistence::new());
        let mut swarm = create_swarm_with_persistence(persistence).await.unwrap();
        
        let mut checkpoints = vec![];
        
        // Create multiple checkpoints with different states
        for i in 0..5 {
            // Add an agent for each checkpoint
            swarm.spawn(AgentType::DataProcessor).await.unwrap();
            
            let checkpoint = swarm.create_checkpoint().await.unwrap();
            swarm.save_checkpoint(&checkpoint).await.unwrap();
            checkpoints.push(checkpoint);
            
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        
        // Verify we can restore to any checkpoint
        for (i, checkpoint) in checkpoints.iter().enumerate() {
            swarm.restore_from_checkpoint(checkpoint.id()).await.unwrap();
            assert_eq!(swarm.agent_count(), i + 1);
        }
        
        // Verify checkpoint ordering
        let listed = swarm.list_checkpoints().await.unwrap();
        assert_eq!(listed.len(), 5);
        
        // Checkpoints should be ordered by timestamp
        for i in 1..listed.len() {
            assert!(listed[i].timestamp() > listed[i-1].timestamp());
        }
    }

    #[tokio::test]
    async fn test_incremental_checkpoints() {
        let persistence = Box::new(MemoryPersistence::new());
        let mut swarm = create_swarm_with_persistence(persistence).await.unwrap();
        
        // Create base checkpoint
        let agent1 = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        let base_checkpoint = swarm.create_checkpoint().await.unwrap();
        swarm.save_checkpoint(&base_checkpoint).await.unwrap();
        
        // Make changes and create incremental checkpoint
        let agent2 = swarm.spawn(AgentType::DataAnalyzer).await.unwrap();
        let task = create_test_task();
        let task_id = swarm.submit_task(task).await.unwrap();
        
        let incremental = swarm.create_incremental_checkpoint(&base_checkpoint).await.unwrap();
        swarm.save_checkpoint(&incremental).await.unwrap();
        
        // Verify incremental checkpoint is smaller
        assert!(incremental.size() < base_checkpoint.size() * 0.5);
        
        // Restore and verify
        swarm.restore_from_checkpoint(incremental.id()).await.unwrap();
        assert_eq!(swarm.agent_count(), 2);
        assert!(swarm.has_agent(&agent1));
        assert!(swarm.has_agent(&agent2));
    }

    #[tokio::test]
    async fn test_persistence_failure_recovery() {
        // Create a flaky persistence backend that fails intermittently
        struct FlakyPersistence {
            inner: MemoryPersistence,
            fail_count: Arc<RwLock<usize>>,
            fail_threshold: usize,
        }
        
        #[async_trait::async_trait]
        impl PersistenceBackend for FlakyPersistence {
            async fn save(&self, key: &str, data: &[u8]) -> Result<(), PersistenceError> {
                let mut count = self.fail_count.write().await;
                *count += 1;
                
                if *count % self.fail_threshold == 0 {
                    Err(PersistenceError::WriteError("Simulated failure".into()))
                } else {
                    self.inner.save(key, data).await
                }
            }
            
            async fn load(&self, key: &str) -> Result<Vec<u8>, PersistenceError> {
                self.inner.load(key).await
            }
            
            async fn delete(&self, key: &str) -> Result<(), PersistenceError> {
                self.inner.delete(key).await
            }
            
            async fn list_keys(&self, prefix: &str) -> Result<Vec<String>, PersistenceError> {
                self.inner.list_keys(prefix).await
            }
        }
        
        let flaky = Box::new(FlakyPersistence {
            inner: MemoryPersistence::new(),
            fail_count: Arc::new(RwLock::new(0)),
            fail_threshold: 3,
        });
        
        let mut swarm = create_swarm_with_persistence(flaky).await.unwrap();
        
        // Enable retry mechanism
        swarm.set_persistence_retry_policy(3, std::time::Duration::from_millis(100)).await;
        
        // Attempt multiple saves - some will fail but should be retried
        for i in 0..10 {
            swarm.spawn(AgentType::DataProcessor).await.unwrap();
            
            // This should succeed despite intermittent failures
            let checkpoint = swarm.create_checkpoint().await.unwrap();
            let result = swarm.save_checkpoint(&checkpoint).await;
            assert!(result.is_ok(), "Save failed on iteration {}", i);
        }
    }

    #[tokio::test]
    async fn test_state_compaction() {
        let persistence = Box::new(MemoryPersistence::new());
        let mut swarm = create_swarm_with_persistence(persistence).await.unwrap();
        
        // Create many checkpoints
        for _ in 0..20 {
            swarm.spawn(AgentType::DataProcessor).await.unwrap();
            let checkpoint = swarm.create_checkpoint().await.unwrap();
            swarm.save_checkpoint(&checkpoint).await.unwrap();
        }
        
        // Check storage size before compaction
        let size_before = swarm.get_storage_size().await.unwrap();
        
        // Run compaction
        let compacted = swarm.compact_checkpoints(5).await.unwrap();
        
        // Verify compaction results
        assert!(compacted > 10, "Should have compacted at least 10 checkpoints");
        
        let size_after = swarm.get_storage_size().await.unwrap();
        assert!(size_after < size_before * 0.5, "Storage should be significantly reduced");
        
        // Verify we still have recent checkpoints
        let remaining = swarm.list_checkpoints().await.unwrap();
        assert!(remaining.len() >= 5);
    }

    #[tokio::test]
    async fn test_cross_version_compatibility() {
        let persistence = Box::new(MemoryPersistence::new());
        let mut swarm = create_swarm_with_persistence(persistence).await.unwrap();
        
        // Save state with version tag
        swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        let checkpoint = swarm.create_checkpoint().await.unwrap();
        swarm.save_checkpoint_with_version(&checkpoint, "1.0.0").await.unwrap();
        
        // Simulate version upgrade
        swarm.set_version("2.0.0").await;
        
        // Attempt to load old checkpoint
        let result = swarm.restore_from_checkpoint(checkpoint.id()).await;
        
        // Should handle version mismatch gracefully
        assert!(result.is_ok(), "Should handle version differences");
        
        // Verify migration was applied if needed
        let migrations = swarm.get_applied_migrations().await.unwrap();
        assert!(!migrations.is_empty(), "Migrations should be tracked");
    }

    #[tokio::test]
    async fn test_parallel_state_operations() {
        let persistence = Box::new(MemoryPersistence::new());
        let swarm = Arc::new(RwLock::new(
            create_swarm_with_persistence(persistence).await.unwrap()
        ));
        
        // Spawn multiple tasks performing state operations concurrently
        let mut handles = vec![];
        
        for i in 0..10 {
            let swarm_clone = Arc::clone(&swarm);
            let handle = tokio::spawn(async move {
                let mut swarm = swarm_clone.write().await;
                
                // Perform various state operations
                if i % 2 == 0 {
                    swarm.spawn(AgentType::DataProcessor).await.unwrap();
                } else {
                    let task = create_test_task();
                    swarm.submit_task(task).await.unwrap();
                }
                
                // Create checkpoint
                let checkpoint = swarm.create_checkpoint().await.unwrap();
                swarm.save_checkpoint(&checkpoint).await.unwrap();
            });
            
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify state consistency
        let swarm = swarm.read().await;
        let checkpoints = swarm.list_checkpoints().await.unwrap();
        assert_eq!(checkpoints.len(), 10);
        
        // Each checkpoint should be valid
        for checkpoint in checkpoints {
            let loaded = swarm.load_checkpoint(checkpoint.id()).await;
            assert!(loaded.is_ok());
        }
    }

    #[tokio::test]
    async fn test_state_migration() {
        // Start with SQLite
        let temp_dir = TempDir::new().unwrap();
        let sqlite_path = temp_dir.path().join("state.db");
        
        let sqlite_persistence = Box::new(
            SqlitePersistence::new(sqlite_path.to_str().unwrap()).await.unwrap()
        );
        
        let mut swarm = create_swarm_with_persistence(sqlite_persistence).await.unwrap();
        
        // Create state
        let agents = vec![
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::DataAnalyzer).await.unwrap(),
        ];
        
        let checkpoint = swarm.create_checkpoint().await.unwrap();
        swarm.save_checkpoint(&checkpoint).await.unwrap();
        
        // Export state
        let export_path = temp_dir.path().join("export.json");
        swarm.export_state(&export_path).await.unwrap();
        
        // Create new swarm with different persistence
        let memory_persistence = Box::new(MemoryPersistence::new());
        let mut new_swarm = create_swarm_with_persistence(memory_persistence).await.unwrap();
        
        // Import state
        new_swarm.import_state(&export_path).await.unwrap();
        
        // Verify state was migrated correctly
        assert_eq!(new_swarm.agent_count(), 2);
        for agent_id in &agents {
            assert!(new_swarm.has_agent(agent_id));
        }
    }

    #[tokio::test]
    async fn test_garbage_collection() {
        let persistence = Box::new(MemoryPersistence::new());
        let mut swarm = create_swarm_with_persistence(persistence).await.unwrap();
        
        // Configure GC policy
        swarm.set_gc_policy(
            std::time::Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            100, // max checkpoints
        ).await;
        
        // Create many checkpoints with different ages
        for i in 0..150 {
            let checkpoint = swarm.create_checkpoint().await.unwrap();
            
            // Simulate age by modifying timestamp
            if i < 50 {
                // Old checkpoints (should be GC'd)
                checkpoint.set_timestamp_for_test(
                    std::time::SystemTime::now() - std::time::Duration::from_secs(10 * 24 * 60 * 60)
                );
            }
            
            swarm.save_checkpoint(&checkpoint).await.unwrap();
        }
        
        // Run garbage collection
        let collected = swarm.run_garbage_collection().await.unwrap();
        
        // Verify old checkpoints were removed
        assert!(collected >= 50, "Should have collected old checkpoints");
        
        let remaining = swarm.list_checkpoints().await.unwrap();
        assert!(remaining.len() <= 100, "Should respect max checkpoint limit");
    }
}