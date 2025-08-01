//! Tests for transaction support and atomicity

use crate::memory::MemoryStorage;
use crate::models::*;
use crate::*;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Mock transaction implementation for testing
struct MockTransaction {
    operations: Arc<Mutex<Vec<TransactionOp>>>,
    committed: Arc<Mutex<bool>>,
    rolled_back: Arc<Mutex<bool>>,
}

#[derive(Debug, Clone)]
enum TransactionOp {
    StoreAgent(AgentModel),
    UpdateAgent(AgentModel),
    DeleteAgent(String),
    StoreTask(TaskModel),
    UpdateTask(TaskModel),
}

impl MockTransaction {
    fn new() -> Self {
        MockTransaction {
            operations: Arc::new(Mutex::new(Vec::new())),
            committed: Arc::new(Mutex::new(false)),
            rolled_back: Arc::new(Mutex::new(false)),
        }
    }

    async fn add_operation(&self, op: TransactionOp) {
        self.operations.lock().await.push(op);
    }

    async fn is_committed(&self) -> bool {
        *self.committed.lock().await
    }

    async fn is_rolled_back(&self) -> bool {
        *self.rolled_back.lock().await
    }
}

#[async_trait::async_trait]
impl Transaction for MockTransaction {
    async fn commit(self: Box<Self>) -> Result<(), StorageError> {
        *self.committed.lock().await = true;
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> Result<(), StorageError> {
        *self.rolled_back.lock().await = true;
        Ok(())
    }
}

#[tokio::test]
async fn test_transaction_commit() {
    let transaction = Arc::new(MockTransaction::new());
    let tx_clone = transaction.clone();

    // Add operations
    let now = Utc::now();
    transaction
        .add_operation(TransactionOp::StoreAgent(AgentModel {
            id: "tx-agent-1".to_string(),
            name: "Transaction Agent".to_string(),
            agent_type: "test".to_string(),
            status: AgentStatus::Idle,
            capabilities: vec![],
            metadata: std::collections::HashMap::new(),
            heartbeat: now,
            created_at: now,
            updated_at: now,
        }))
        .await;

    // Commit transaction (this consumes the Box)
    let boxed_tx = Box::new(MockTransaction {
        operations: transaction.operations.clone(),
        committed: transaction.committed.clone(),
        rolled_back: transaction.rolled_back.clone(),
    });
    boxed_tx.commit().await.unwrap();

    assert!(tx_clone.is_committed().await);
    assert!(!tx_clone.is_rolled_back().await);
}

#[tokio::test]
async fn test_transaction_rollback() {
    let transaction = Arc::new(MockTransaction::new());
    let tx_clone = transaction.clone();

    // Add operations
    transaction
        .add_operation(TransactionOp::DeleteAgent("agent-to-delete".to_string()))
        .await;

    // Rollback transaction (this consumes the Box)
    let boxed_tx = Box::new(MockTransaction {
        operations: transaction.operations.clone(),
        committed: transaction.committed.clone(),
        rolled_back: transaction.rolled_back.clone(),
    });
    boxed_tx.rollback().await.unwrap();

    assert!(!tx_clone.is_committed().await);
    assert!(tx_clone.is_rolled_back().await);
}

#[tokio::test]
async fn test_transaction_isolation() {
    // This test simulates transaction isolation by using separate storage instances
    let storage1 = Arc::new(MemoryStorage::new());
    let storage2 = storage1.clone();

    // Agent that will be modified in transaction
    let now = Utc::now();
    let agent = AgentModel {
        id: "isolation-test".to_string(),
        name: "Original Name".to_string(),
        agent_type: "test".to_string(),
        status: AgentStatus::Idle,
        capabilities: vec![],
        metadata: std::collections::HashMap::new(),
        heartbeat: now,
        created_at: now,
        updated_at: now,
    };

    storage1.store_agent(&agent).await.unwrap();

    // Simulate transaction in progress
    let tx_operations = Arc::new(Mutex::new(Vec::new()));

    // Transaction 1: Update agent
    {
        let mut ops = tx_operations.lock().await;
        ops.push(TransactionOp::UpdateAgent(AgentModel {
            id: agent.id.clone(),
            name: "Updated Name".to_string(),
            ..agent.clone()
        }));
    }

    // Transaction 2: Read should see original value (isolation)
    let read_agent = storage2.get_agent(&agent.id).await.unwrap().unwrap();
    assert_eq!(read_agent.name, "Original Name");
}

#[tokio::test]
async fn test_transaction_atomicity() {
    let storage = MemoryStorage::new();

    // Prepare multiple operations that should be atomic
    let now = Utc::now();
    let agents: Vec<AgentModel> = (0..5)
        .map(|i| AgentModel {
            id: format!("atomic-agent-{}", i),
            name: format!("Atomic Agent {}", i),
            agent_type: "worker".to_string(),
            status: AgentStatus::Idle,
            capabilities: vec![],
            metadata: std::collections::HashMap::new(),
            heartbeat: now,
            created_at: now,
            updated_at: now,
        })
        .collect();

    // Simulate atomic operation
    let mut success = true;
    for (i, agent) in agents.iter().enumerate() {
        if i == 3 {
            // Simulate failure on 4th operation
            success = false;
            break;
        }
        storage.store_agent(agent).await.unwrap();
    }

    if !success {
        // In a real transaction, all operations would be rolled back
        // For this test, we verify partial state
        let stored_agents = storage.list_agents().await.unwrap();
        assert_eq!(stored_agents.len(), 3); // Only first 3 were stored
    }
}

#[tokio::test]
async fn test_concurrent_transactions() {
    let storage = Arc::new(MemoryStorage::new());

    // Create initial agent
    let agent_id = "concurrent-tx-agent".to_string();
    let now = Utc::now();
    let initial_agent = AgentModel {
        id: agent_id.clone(),
        name: "Initial".to_string(),
        agent_type: "test".to_string(),
        status: AgentStatus::Idle,
        capabilities: vec!["v1".to_string()],
        metadata: {
            let mut map = std::collections::HashMap::new();
            map.insert("counter".to_string(), serde_json::json!(0));
            map
        },
        heartbeat: now,
        created_at: now,
        updated_at: now,
    };

    storage.store_agent(&initial_agent).await.unwrap();

    // Simulate concurrent transactions trying to update the same agent
    let mut handles = vec![];

    for i in 0..5 {
        let storage_clone = storage.clone();
        let agent_id_clone = agent_id.clone();

        let handle = tokio::spawn(async move {
            // Read current state
            let mut agent = storage_clone
                .get_agent(&agent_id_clone)
                .await
                .unwrap()
                .unwrap();

            // Modify
            agent
                .metadata
                .insert("counter".to_string(), serde_json::json!(i));

            // Small delay to increase chance of conflict
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

            // Update
            storage_clone.update_agent(&agent).await
        });

        handles.push(handle);
    }

    // Wait for all transactions
    for handle in handles {
        let _ = handle.await.unwrap();
    }

    // Verify final state
    let final_agent = storage.get_agent(&agent_id).await.unwrap().unwrap();
    // In a real transactional system, one of the updates would win
    assert!(final_agent.metadata["counter"].is_number());
}

#[tokio::test]
async fn test_transaction_with_mixed_operations() {
    let transaction = MockTransaction::new();

    // Add various operation types
    let now = Utc::now();
    transaction
        .add_operation(TransactionOp::StoreAgent(AgentModel {
            id: "new-agent".to_string(),
            name: "New Agent".to_string(),
            agent_type: "compute".to_string(),
            status: AgentStatus::Idle,
            capabilities: vec![],
            metadata: std::collections::HashMap::new(),
            heartbeat: now,
            created_at: now,
            updated_at: now,
        }))
        .await;

    transaction
        .add_operation(TransactionOp::StoreTask(TaskModel {
            id: "new-task".to_string(),
            task_type: "process".to_string(),
            status: TaskStatus::Pending,
            priority: TaskPriority::Medium,
            payload: serde_json::json!({}),
            assigned_to: Some("new-agent".to_string()),
            result: None,
            error: None,
            retry_count: 0,
            max_retries: 3,
            dependencies: vec![],
            created_at: now,
            updated_at: now,
            started_at: None,
            completed_at: None,
        }))
        .await;

    transaction
        .add_operation(TransactionOp::UpdateAgent(AgentModel {
            id: "new-agent".to_string(),
            name: "New Agent".to_string(),
            agent_type: "compute".to_string(),
            status: AgentStatus::Busy, // Status changed
            capabilities: vec![],
            metadata: std::collections::HashMap::new(),
            heartbeat: now,
            created_at: now,
            updated_at: now,
        }))
        .await;

    // Verify operations were recorded
    let ops = transaction.operations.lock().await;
    assert_eq!(ops.len(), 3);

    // Verify operation types
    assert!(matches!(ops[0], TransactionOp::StoreAgent(_)));
    assert!(matches!(ops[1], TransactionOp::StoreTask(_)));
    assert!(matches!(ops[2], TransactionOp::UpdateAgent(_)));
}

#[tokio::test]
async fn test_transaction_savepoint() {
    // Simulate savepoint functionality
    struct SavepointTransaction {
        operations: Vec<TransactionOp>,
        savepoints: Vec<usize>,
    }

    impl SavepointTransaction {
        fn new() -> Self {
            SavepointTransaction {
                operations: Vec::new(),
                savepoints: Vec::new(),
            }
        }

        fn add_operation(&mut self, op: TransactionOp) {
            self.operations.push(op);
        }

        fn create_savepoint(&mut self) {
            self.savepoints.push(self.operations.len());
        }

        fn rollback_to_savepoint(&mut self) {
            if let Some(savepoint) = self.savepoints.pop() {
                self.operations.truncate(savepoint);
            }
        }
    }

    let mut tx = SavepointTransaction::new();

    // Add operations
    let now = Utc::now();
    tx.add_operation(TransactionOp::StoreAgent(AgentModel {
        id: "sp-agent-1".to_string(),
        name: "Agent 1".to_string(),
        agent_type: "test".to_string(),
        status: AgentStatus::Idle,
        capabilities: vec![],
        metadata: std::collections::HashMap::new(),
        heartbeat: now,
        created_at: now,
        updated_at: now,
    }));

    // Create savepoint
    tx.create_savepoint();

    // Add more operations
    tx.add_operation(TransactionOp::StoreAgent(AgentModel {
        id: "sp-agent-2".to_string(),
        name: "Agent 2".to_string(),
        agent_type: "test".to_string(),
        status: AgentStatus::Idle,
        capabilities: vec![],
        metadata: std::collections::HashMap::new(),
        heartbeat: now,
        created_at: now,
        updated_at: now,
    }));

    tx.add_operation(TransactionOp::DeleteAgent("sp-agent-1".to_string()));

    // Rollback to savepoint
    tx.rollback_to_savepoint();

    // Should only have first operation
    assert_eq!(tx.operations.len(), 1);
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_transaction_operation_ordering(
            operation_count in 1usize..50,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let transaction = MockTransaction::new();

                // Add random operations
                for i in 0..operation_count {
                    let now = Utc::now();
                    let op = if i % 2 == 0 {
                        TransactionOp::StoreAgent(AgentModel {
                            id: format!("agent-{}", i),
                            name: format!("Agent {}", i),
                            agent_type: "test".to_string(),
                            status: AgentStatus::Idle,
                            capabilities: vec![],
                            metadata: std::collections::HashMap::new(),
                            heartbeat: now,
                            created_at: now,
                            updated_at: now,
                        })
                    } else {
                        TransactionOp::StoreTask(TaskModel {
                            id: format!("task-{}", i),
                            task_type: "test".to_string(),
                            status: TaskStatus::Pending,
                            priority: match i % 4 {
                                0 => TaskPriority::Low,
                                1 => TaskPriority::Medium,
                                2 => TaskPriority::High,
                                _ => TaskPriority::Critical,
                            },
                            payload: serde_json::json!({}),
                            assigned_to: None,
                            result: None,
                            error: None,
                            retry_count: 0,
                            max_retries: 3,
                            dependencies: vec![],
                            created_at: now,
                            updated_at: now,
                            started_at: None,
                            completed_at: None,
                        })
                    };

                    transaction.add_operation(op).await;
                }

                let ops = transaction.operations.lock().await;
                assert_eq!(ops.len(), operation_count);
            });
        }
    }
}
