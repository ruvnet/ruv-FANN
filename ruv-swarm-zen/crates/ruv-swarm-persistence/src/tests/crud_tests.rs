//! CRUD operation tests for persistence layer

use crate::memory::MemoryStorage;
use crate::models::*;
use crate::*;
use chrono::Utc;
use uuid::Uuid;

#[tokio::test]
async fn test_agent_create_read() {
    let storage = MemoryStorage::new();

    let now = Utc::now();
    let agent = AgentModel {
        id: "test-agent-1".to_string(),
        name: "Test Agent".to_string(),
        agent_type: "neural".to_string(),
        status: AgentStatus::Idle,
        capabilities: vec!["compute".to_string(), "analyze".to_string()],
        metadata: {
            let mut map = std::collections::HashMap::new();
            map.insert("threads".to_string(), serde_json::json!(4));
            map
        },
        heartbeat: now,
        created_at: now,
        updated_at: now,
    };

    // Create
    storage.store_agent(&agent).await.unwrap();

    // Read
    let retrieved = storage.get_agent(&agent.id).await.unwrap();
    assert!(retrieved.is_some());

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, agent.id);
    assert_eq!(retrieved.name, agent.name);
    assert_eq!(retrieved.agent_type, agent.agent_type);
    assert_eq!(retrieved.status, agent.status);
}

#[tokio::test]
async fn test_agent_update() {
    let storage = MemoryStorage::new();

    let now = Utc::now();
    let mut agent = AgentModel {
        id: "test-agent-2".to_string(),
        name: "Update Test Agent".to_string(),
        agent_type: "compute".to_string(),
        status: AgentStatus::Idle,
        capabilities: vec![],
        metadata: std::collections::HashMap::new(),
        heartbeat: now,
        created_at: now,
        updated_at: now,
    };

    storage.store_agent(&agent).await.unwrap();

    // Update fields
    agent.status = AgentStatus::Busy;
    agent.capabilities = vec!["neural".to_string(), "quantum".to_string()];
    agent.updated_at = Utc::now();

    storage.update_agent(&agent).await.unwrap();

    // Verify update
    let retrieved = storage.get_agent(&agent.id).await.unwrap().unwrap();
    assert_eq!(retrieved.status, AgentStatus::Busy);
    assert_eq!(
        retrieved.capabilities,
        vec!["neural".to_string(), "quantum".to_string()]
    );
}

#[tokio::test]
async fn test_agent_delete() {
    let storage = MemoryStorage::new();

    let now = Utc::now();
    let agent = AgentModel {
        id: "test-agent-3".to_string(),
        name: "Delete Test Agent".to_string(),
        agent_type: "temporary".to_string(),
        status: AgentStatus::Idle,
        capabilities: vec![],
        metadata: std::collections::HashMap::new(),
        heartbeat: now,
        created_at: now,
        updated_at: now,
    };

    storage.store_agent(&agent).await.unwrap();

    // Verify exists
    assert!(storage.get_agent(&agent.id).await.unwrap().is_some());

    // Delete
    storage.delete_agent(&agent.id).await.unwrap();

    // Verify deleted
    assert!(storage.get_agent(&agent.id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_task_crud_operations() {
    let storage = MemoryStorage::new();

    let now = Utc::now();
    let task = TaskModel {
        id: Uuid::new_v4().to_string(),
        task_type: "analysis".to_string(),
        priority: TaskPriority::High,
        status: TaskStatus::Pending,
        assigned_to: None,
        payload: serde_json::json!({"data": "test"}),
        result: None,
        error: None,
        retry_count: 0,
        max_retries: 3,
        dependencies: vec![],
        created_at: now,
        updated_at: now,
        started_at: None,
        completed_at: None,
    };

    // Create
    storage.store_task(&task).await.unwrap();

    // Read
    let retrieved = storage.get_task(&task.id).await.unwrap();
    assert!(retrieved.is_some());

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, task.id);
    assert_eq!(retrieved.task_type, task.task_type);
    assert_eq!(retrieved.priority, task.priority);
    assert_eq!(retrieved.status, task.status);

    // Update
    let mut updated_task = retrieved;
    updated_task.status = TaskStatus::Running;
    updated_task.assigned_to = Some("agent-1".to_string());
    updated_task.started_at = Some(Utc::now());
    updated_task.updated_at = Utc::now();

    storage.update_task(&updated_task).await.unwrap();

    // Verify update
    let retrieved = storage.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(retrieved.status, TaskStatus::Running);
    assert_eq!(retrieved.assigned_to, Some("agent-1".to_string()));
    assert!(retrieved.started_at.is_some());

    // Note: delete_task is not implemented in the Storage trait
    // Tasks are typically marked as completed/failed rather than deleted
}

#[tokio::test]
async fn test_event_crud() {
    let storage = MemoryStorage::new();

    let event = EventModel {
        id: Uuid::new_v4().to_string(),
        event_type: "task_completed".to_string(),
        agent_id: Some("agent-1".to_string()),
        task_id: Some("task-1".to_string()),
        payload: serde_json::json!({"duration": 100}),
        metadata: std::collections::HashMap::new(),
        timestamp: Utc::now(),
        sequence: 1,
    };

    // Create
    storage.store_event(&event).await.unwrap();

    // Read by agent
    let events = storage.get_events_by_agent("agent-1", 10).await.unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].id, event.id);

    // Read by type
    let events = storage
        .get_events_by_type("task_completed", 10)
        .await
        .unwrap();
    assert_eq!(events.len(), 1);
}

#[tokio::test]
async fn test_message_crud() {
    let storage = MemoryStorage::new();

    let message = MessageModel {
        id: Uuid::new_v4().to_string(),
        from_agent: "agent-1".to_string(),
        to_agent: "agent-2".to_string(),
        message_type: "coordination".to_string(),
        content: serde_json::json!({"action": "sync"}),
        priority: MessagePriority::High,
        read: false,
        created_at: Utc::now(),
        read_at: None,
    };

    // Create
    storage.store_message(&message).await.unwrap();

    // Read unread
    let unread = storage.get_unread_messages("agent-2").await.unwrap();
    assert_eq!(unread.len(), 1);
    assert_eq!(unread[0].id, message.id);

    // Mark as read
    storage.mark_message_read(&message.id).await.unwrap();

    // Verify marked as read
    let unread = storage.get_unread_messages("agent-2").await.unwrap();
    assert_eq!(unread.len(), 0);
}

#[tokio::test]
async fn test_metric_crud() {
    let storage = MemoryStorage::new();

    let metric = MetricModel {
        id: Uuid::new_v4().to_string(),
        metric_type: "performance".to_string(),
        agent_id: Some("agent-1".to_string()),
        value: 95.5,
        unit: "percent".to_string(),
        tags: {
            let mut tags = std::collections::HashMap::new();
            tags.insert("component".to_string(), "neural".to_string());
            tags
        },
        timestamp: Utc::now(),
    };

    // Create
    storage.store_metric(&metric).await.unwrap();

    // Read by agent
    let metrics = storage
        .get_metrics_by_agent("agent-1", "performance")
        .await
        .unwrap();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].value, 95.5);
}

#[tokio::test]
async fn test_batch_operations() {
    let storage = MemoryStorage::new();

    // Create multiple agents
    for i in 0..5 {
        let now = Utc::now();
        let agent = AgentModel {
            id: format!("batch-agent-{}", i),
            name: format!("Batch Agent {}", i),
            agent_type: "worker".to_string(),
            status: if i % 2 == 0 {
                AgentStatus::Idle
            } else {
                AgentStatus::Busy
            },
            capabilities: vec!["compute".to_string()],
            metadata: std::collections::HashMap::new(),
            heartbeat: now,
            created_at: now,
            updated_at: now,
        };
        storage.store_agent(&agent).await.unwrap();
    }

    // List all agents
    let agents = storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), 5);

    // List by status
    let idle_agents = storage
        .list_agents_by_status(&AgentStatus::Idle.to_string())
        .await
        .unwrap();
    assert_eq!(idle_agents.len(), 3);

    let busy_agents = storage
        .list_agents_by_status(&AgentStatus::Busy.to_string())
        .await
        .unwrap();
    assert_eq!(busy_agents.len(), 2);
}

#[tokio::test]
async fn test_time_based_queries() {
    let storage = MemoryStorage::new();

    let start_time = Utc::now();

    // Create events at different times
    for i in 0..5 {
        let event = EventModel {
            id: Uuid::new_v4().to_string(),
            event_type: "test_event".to_string(),
            agent_id: Some("agent-1".to_string()),
            task_id: None,
            payload: serde_json::json!({"index": i}),
            metadata: std::collections::HashMap::new(),
            timestamp: start_time + chrono::Duration::seconds(i * 10),
            sequence: i as u64,
        };
        storage.store_event(&event).await.unwrap();
    }

    // Query events since a specific time
    let mid_time = start_time + chrono::Duration::seconds(25);
    let recent_events = storage
        .get_events_since(mid_time.timestamp())
        .await
        .unwrap();
    assert_eq!(recent_events.len(), 2);
}

#[tokio::test]
async fn test_error_scenarios() {
    let storage = MemoryStorage::new();

    // Non-existent reads should return None
    assert!(storage.get_agent("non-existent").await.unwrap().is_none());
    assert!(storage.get_task("non-existent").await.unwrap().is_none());

    // Delete non-existent should not error
    storage.delete_agent("non-existent").await.unwrap();
    // Note: delete_task is not implemented in the Storage trait
}

#[tokio::test]
async fn test_task_priority_ordering() {
    let storage = MemoryStorage::new();

    // Create tasks with different priorities
    let priorities = vec![
        TaskPriority::Low,
        TaskPriority::Critical,
        TaskPriority::Medium,
        TaskPriority::High,
    ];

    for (i, priority) in priorities.into_iter().enumerate() {
        let now = Utc::now();
        let task = TaskModel {
            id: format!("priority-task-{}", i),
            task_type: "test".to_string(),
            priority,
            status: TaskStatus::Pending,
            assigned_to: None,
            payload: serde_json::json!({}),
            result: None,
            error: None,
            retry_count: 0,
            max_retries: 3,
            dependencies: vec![],
            created_at: now + chrono::Duration::seconds(i as i64),
            updated_at: now,
            started_at: None,
            completed_at: None,
        };
        storage.store_task(&task).await.unwrap();
    }

    // Get pending tasks - should be ordered by priority
    let pending = storage.get_pending_tasks().await.unwrap();
    assert_eq!(pending.len(), 4);

    // Verify highest priority task is first
    assert_eq!(pending[0].priority, TaskPriority::Critical);
}
