//! Property-based tests for persistence layer
//! 
//! These tests use proptest to verify invariants and edge cases
//! that are critical for production reliability.

use proptest::prelude::*;
use proptest::test_runner::TestRunner;
use proptest::strategy::{ValueTree, Just};
use tokio::runtime::Runtime;
use chrono::{DateTime, Utc, Duration};
use std::collections::HashSet;
use uuid::Uuid;

use crate::{Storage, StorageError, AgentModel, TaskModel, EventModel, MessageModel, MetricModel};
use crate::models::{AgentStatus, TaskStatus, TaskPriority};
use crate::memory::MemoryStorage;
#[cfg(not(target_arch = "wasm32"))]
use crate::sqlite::SqliteStorage;

// ===== Arbitrary Instance Generators =====

prop_compose! {
    /// Generate valid agent IDs
    fn arb_agent_id()(id in "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}") -> String {
        id
    }
}

prop_compose! {
    /// Generate valid agent models with constraints
    fn arb_agent_model()(
        id in arb_agent_id(),
        name in "[a-zA-Z0-9-_]{3,50}",
        agent_type in prop::sample::select(vec!["compute", "analysis", "coordination", "storage"]),
        status in prop::sample::select(vec![AgentStatus::Idle, AgentStatus::Busy, AgentStatus::Error, AgentStatus::Active]),
        capabilities in prop::collection::vec("[a-z]+", 0..10),
        metadata_keys in prop::collection::vec("[a-z]+", 0..5),
        heartbeat_offset in 0i64..3600,
        created_offset in 3600i64..7200,
    ) -> AgentModel {
        let now = Utc::now();
        let created_at = now - Duration::seconds(created_offset);
        let updated_at = now - Duration::seconds(heartbeat_offset / 2);
        let heartbeat = now - Duration::seconds(heartbeat_offset);
        
        let metadata = metadata_keys.into_iter()
            .map(|k| (k.clone(), serde_json::Value::String(k)))
            .collect();
        
        AgentModel {
            id,
            name,
            agent_type: agent_type.to_string(),
            status,
            capabilities,
            metadata,
            heartbeat,
            created_at,
            updated_at,
        }
    }
}

/// Generate valid task models
fn arb_task_model(existing_agents: Vec<String>) -> impl Strategy<Value = TaskModel> {
    let agent_count = existing_agents.len();
    let agents = existing_agents.clone();
    
    (
        arb_agent_id(),
        "[a-zA-Z0-9 ]{10,100}",
        prop::sample::select(vec![TaskPriority::Low, TaskPriority::Medium, TaskPriority::High, TaskPriority::Critical]),
        prop::sample::select(vec![TaskStatus::Pending, TaskStatus::Assigned, TaskStatus::Running, TaskStatus::Completed, TaskStatus::Failed]),
        if agent_count == 0 { 
            Just(None).boxed()
        } else { 
            (0..agent_count).prop_map(move |idx| Some(agents[idx].clone())).boxed()
        },
        0..1000usize,
        0i64..3600,
    ).prop_map(|(id, description, priority, status, assigned_to, data_size, created_offset)| {
        let now = Utc::now();
        let created_at = now - Duration::seconds(created_offset);
        let updated_at = created_at + Duration::seconds(10);
        
        // Generate some payload data
        let payload = (0..data_size)
            .map(|i| (format!("key_{}", i), serde_json::Value::Number(i.into())))
            .collect();
        
        TaskModel {
            id,
            task_type: description,
            priority,
            status,
            assigned_to,
            payload,
            result: None,
            error: None,
            retry_count: 0,
            max_retries: 3,
            dependencies: vec![],
            created_at,
            updated_at,
            started_at: None,
            completed_at: None,
        }
    })
}

// ===== Property Tests =====

proptest! {
    #[test]
    fn test_agent_model_invariants(agent in arb_agent_model()) {
        // Property: Timestamps must be ordered correctly
        prop_assert!(agent.created_at <= agent.updated_at);
        prop_assert!(agent.heartbeat <= Utc::now());
        
        // Property: Required fields must not be empty
        prop_assert!(!agent.id.is_empty());
        prop_assert!(!agent.name.is_empty());
        prop_assert!(!agent.agent_type.is_empty());
    }
    
    #[test]
    fn test_agent_storage_consistency(agents in prop::collection::vec(arb_agent_model(), 1..20)) {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let storage = MemoryStorage::new();
            
            // Store all agents
            for agent in &agents {
                storage.store_agent(agent).await.unwrap();
            }
            
            // Property: All stored agents must be retrievable
            let stored_agents = storage.list_agents().await.unwrap();
            prop_assert_eq!(stored_agents.len(), agents.len());
            
            // Property: Each agent must be retrievable by ID
            for agent in &agents {
                let retrieved = storage.get_agent(&agent.id).await.unwrap();
                prop_assert!(retrieved.is_some());
                prop_assert_eq!(retrieved.unwrap().id, agent.id.clone());
            }
            
            // Property: No phantom agents should exist
            let phantom_id = Uuid::new_v4().to_string();
            let phantom = storage.get_agent(&phantom_id).await.unwrap();
            prop_assert!(phantom.is_none());
            
            Ok(())
        })?;
    }
    
    #[test]
    fn test_agent_status_filtering(
        agents in prop::collection::vec(arb_agent_model(), 10..50),
        query_status in prop::sample::select(vec!["idle", "busy", "error", "active", "paused"])
    ) {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let storage = MemoryStorage::new();
            
            // Store all agents
            for agent in &agents {
                storage.store_agent(agent).await.unwrap();
            }
            
            // Query by status
            let filtered = storage.list_agents_by_status(query_status).await.unwrap();
            
            // Property: All returned agents must have the queried status
            for agent in &filtered {
                prop_assert_eq!(agent.status.to_string(), query_status);
            }
            
            // Property: Count must match manual filter
            let expected_count = agents.iter().filter(|a| a.status.to_string() == query_status).count();
            prop_assert_eq!(filtered.len(), expected_count);
            
            Ok(())
        })?;
    }
    
    #[test]
    fn test_task_referential_integrity(
        agents in prop::collection::vec(arb_agent_model(), 1..10),
        task_count in 10usize..50,
    ) {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let storage = MemoryStorage::new();
            
            // Store agents first
            let agent_ids: Vec<String> = agents.iter().map(|a| a.id.clone()).collect();
            for agent in &agents {
                storage.store_agent(agent).await.unwrap();
            }
            
            // Generate tasks that may reference agents
            let mut tasks = vec![];
            for i in 0..task_count {
                let task = arb_task_model(agent_ids.clone())
                    .new_tree(&mut TestRunner::default())
                    .unwrap()
                    .current();
                tasks.push(task);
            }
            
            // Store tasks
            for task in &tasks {
                storage.store_task(task).await.unwrap();
            }
            
            // Property: All assigned tasks must reference existing agents
            let stored_tasks = storage.get_pending_tasks().await.unwrap();
            let stored_agent_ids: HashSet<_> = storage.list_agents().await.unwrap()
                .into_iter()
                .map(|a| a.id)
                .collect();
            
            for task in stored_tasks {
                if let Some(agent_id) = task.assigned_to {
                    prop_assert!(
                        stored_agent_ids.contains(&agent_id),
                        "Task {} assigned to non-existent agent {}",
                        task.id,
                        agent_id
                    );
                }
            }
            
            Ok(())
        })?;
    }
    
    #[test]
    fn test_concurrent_agent_updates(
        initial_agent in arb_agent_model(),
        update_count in 10usize..50,
        statuses in prop::collection::vec(
            prop::sample::select(vec![AgentStatus::Idle, AgentStatus::Busy, AgentStatus::Error, AgentStatus::Active]),
            10..50
        )
    ) {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let storage = MemoryStorage::new();
            
            // Store initial agent
            storage.store_agent(&initial_agent).await.unwrap();
            
            // Property: Last update should win in sequential updates
            let mut current_agent = initial_agent.clone();
            for (i, status) in statuses.iter().enumerate().take(update_count) {
                current_agent.status = status.clone();
                current_agent.updated_at = Utc::now() + Duration::seconds(i as i64);
                storage.update_agent(&current_agent).await.unwrap();
            }
            
            let final_agent = storage.get_agent(&initial_agent.id).await.unwrap().unwrap();
            prop_assert_eq!(final_agent.status, current_agent.status);
            
            Ok(())
        })?;
    }
    
    #[test]
    fn test_storage_idempotency(agent in arb_agent_model()) {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let storage = MemoryStorage::new();
            
            // Property: Storing the same agent multiple times should be idempotent
            storage.store_agent(&agent).await.unwrap();
            storage.store_agent(&agent).await.unwrap();
            storage.store_agent(&agent).await.unwrap();
            
            let agents = storage.list_agents().await.unwrap();
            prop_assert_eq!(agents.len(), 1);
            prop_assert_eq!(agents[0].id.clone(), agent.id);
            
            Ok(())
        })?;
    }
}

// ===== Cross-Storage Consistency Tests =====

#[cfg(not(target_arch = "wasm32"))]
proptest! {
    #[test]
    fn test_cross_storage_consistency(
        agents in prop::collection::vec(arb_agent_model(), 1..20)
    ) {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let memory_storage = MemoryStorage::new();
            let fixture = crate::tests::test_utils::TestFixture::new().await.unwrap();
            let sqlite_storage = fixture.storage();
            
            // Store same data in both storages
            for agent in &agents {
                memory_storage.store_agent(agent).await.unwrap();
                sqlite_storage.store_agent(agent).await.unwrap();
            }
            
            // Property: Both storages should return identical results
            let memory_agents = memory_storage.list_agents().await.unwrap();
            let sqlite_agents = sqlite_storage.list_agents().await.unwrap();
            
            // Sort by ID for comparison
            let mut memory_ids: Vec<_> = memory_agents.iter().map(|a| &a.id).collect();
            let mut sqlite_ids: Vec<_> = sqlite_agents.iter().map(|a| &a.id).collect();
            memory_ids.sort();
            sqlite_ids.sort();
            
            prop_assert_eq!(memory_ids, sqlite_ids);
            
            // Property: Status filtering should work identically
            for status in ["idle", "busy", "offline", "error"] {
                let memory_filtered = memory_storage.list_agents_by_status(status).await.unwrap();
                let sqlite_filtered = sqlite_storage.list_agents_by_status(status).await.unwrap();
                
                prop_assert_eq!(
                    memory_filtered.len(),
                    sqlite_filtered.len(),
                    "Status '{}' filtering differs between storages",
                    status
                );
            }
            
            Ok(())
        })?;
    }
}

// ===== Stateful Property Tests =====

/// Represents operations that can be performed on storage
#[derive(Debug, Clone)]
enum StorageOp {
    StoreAgent(AgentModel),
    UpdateAgent { id: String, new_status: String },
    DeleteAgent(String),
    StoreTask(TaskModel),
    ClaimTask { task_id: String, agent_id: String },
}

fn arb_storage_op(existing_agents: Vec<String>) -> impl Strategy<Value = StorageOp> {
    let agent_count = existing_agents.len();
    let agents = existing_agents.clone();
    let agents2 = existing_agents.clone();
    
    if agent_count == 0 {
        // If no agents exist, only allow StoreAgent or StoreTask operations
        prop_oneof![
            arb_agent_model().prop_map(StorageOp::StoreAgent),
            arb_task_model(vec![]).prop_map(StorageOp::StoreTask),
        ].boxed()
    } else {
        prop_oneof![
            // StoreAgent - 20% chance
            arb_agent_model().prop_map(StorageOp::StoreAgent),
            
            // UpdateAgent - 20% chance
            (0..agent_count, prop::sample::select(vec![AgentStatus::Idle, AgentStatus::Busy, AgentStatus::Error, AgentStatus::Active]))
                .prop_map(move |(idx, status)| StorageOp::UpdateAgent {
                    id: agents[idx].clone(),
                    new_status: status.to_string(),
                }),
            
            // DeleteAgent - 20% chance
            (0..agent_count).prop_map(move |idx| StorageOp::DeleteAgent(agents2[idx].clone())),
            
            // StoreTask - 20% chance
            arb_task_model(existing_agents.clone()).prop_map(StorageOp::StoreTask),
            
            // ClaimTask - 20% chance
            (arb_task_model(existing_agents.clone()), 0..agent_count)
                .prop_map(move |(task, idx)| StorageOp::ClaimTask {
                    task_id: task.id,
                    agent_id: existing_agents[idx].clone(),
                }),
        ].boxed()
    }
}

proptest! {
    #[test]
    fn test_storage_invariants_under_operations(
        initial_agents in prop::collection::vec(arb_agent_model(), 1..5),
        op_count in 10usize..50,
    ) {
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let storage = MemoryStorage::new();
            let mut agent_ids = vec![];
            
            // Store initial agents
            for agent in initial_agents {
                storage.store_agent(&agent).await.unwrap();
                agent_ids.push(agent.id);
            }
            
            // Generate and execute operations
            for _ in 0..op_count {
                let strategy = arb_storage_op(agent_ids.clone());
                let op = strategy
                    .new_tree(&mut TestRunner::default())
                    .unwrap()
                    .current();
                
                match op {
                    StorageOp::StoreAgent(agent) => {
                        agent_ids.push(agent.id.clone());
                        storage.store_agent(&agent).await.unwrap();
                    }
                    StorageOp::UpdateAgent { id, new_status } => {
                        if let Some(mut agent) = storage.get_agent(&id).await.unwrap() {
                            // Convert string back to AgentStatus
                            agent.status = match new_status.as_str() {
                                "idle" => AgentStatus::Idle,
                                "busy" => AgentStatus::Busy,
                                "error" => AgentStatus::Error,
                                "active" => AgentStatus::Active,
                                _ => AgentStatus::Idle, // Default fallback
                            };
                            agent.updated_at = Utc::now();
                            storage.update_agent(&agent).await.unwrap();
                        }
                    }
                    StorageOp::DeleteAgent(id) => {
                        // Check if agent has assigned tasks before deleting
                        let tasks = storage.get_tasks_by_agent(&id).await.unwrap();
                        if tasks.is_empty() {
                            storage.delete_agent(&id).await.unwrap();
                            agent_ids.retain(|aid| aid != &id);
                        }
                        // If agent has tasks, skip deletion to maintain referential integrity
                    }
                    StorageOp::StoreTask(task) => {
                        storage.store_task(&task).await.unwrap();
                    }
                    StorageOp::ClaimTask { task_id, agent_id } => {
                        let _ = storage.claim_task(&task_id, &agent_id).await;
                    }
                }
                
                // Check invariants after each operation
                check_storage_invariants(&storage).await?;
            }
            
            Ok(()) as Result<(), TestCaseError>
        })?;
    }
}

async fn check_storage_invariants(storage: &dyn Storage<Error = StorageError>) -> Result<(), TestCaseError> {
    // Invariant 1: All agents have valid timestamps
    let agents = storage.list_agents().await.unwrap();
    for agent in agents {
        prop_assert!(agent.created_at <= agent.updated_at);
        prop_assert!(agent.heartbeat <= Utc::now() + Duration::seconds(60)); // Allow small clock drift
    }
    
    // Invariant 2: Task assignments reference existing agents
    let agent_ids: HashSet<_> = storage.list_agents().await.unwrap()
        .into_iter()
        .map(|a| a.id)
        .collect();
    
    let tasks = storage.get_pending_tasks().await.unwrap();
    for task in tasks {
        if let Some(agent_id) = &task.assigned_to {
            prop_assert!(
                agent_ids.contains(agent_id),
                "Task {} assigned to non-existent agent {}",
                task.id,
                agent_id
            );
        }
    }
    
    Ok(())
}