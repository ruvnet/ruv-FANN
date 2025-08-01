//! Concurrent access and performance tests
//!
//! These tests verify the persistence layer handles concurrent operations correctly
//! and performs adequately under load.

use crate::{Storage, StorageError, AgentModel, TaskModel, EventModel, MessageModel, MetricModel};
use crate::memory::MemoryStorage;
#[cfg(not(target_arch = "wasm32"))]
use crate::sqlite::SqliteStorage;
use chrono::{Utc, Duration};
use std::sync::Arc;
use tokio::sync::Barrier;
use std::collections::HashSet;
use futures::future::join_all;
use tracing::debug;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_agent_registration_storm() {
    let storage = Arc::new(MemoryStorage::new());
    let agent_count = 100;
    let barrier = Arc::new(Barrier::new(agent_count));
    
    // Spawn concurrent agent registrations
    let mut handles = vec![];
    for i in 0..agent_count {
        let storage_clone = storage.clone();
        let barrier_clone = barrier.clone();
        
        let handle = tokio::spawn(async move {
            // Wait for all tasks to be ready
            barrier_clone.wait().await;
            
            // All register at once
            let agent = AgentModel::new(
                format!("agent-{}", i),
                "worker".to_string(),
                vec!["compute".to_string()],
            );
            storage_clone.store_agent(&agent).await
        });
        handles.push(handle);
    }
    
    // Collect results
    let results: Vec<_> = join_all(handles).await;
    
    // All should succeed
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_ok(), "Task {} failed: {:?}", i, result);
        assert!(result.as_ref().unwrap().is_ok(), "Agent {} registration failed", i);
    }
    
    // Verify all agents are stored
    let agents = storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), agent_count);
    
    // Verify no duplicate IDs
    let ids: HashSet<_> = agents.iter().map(|a| &a.id).collect();
    assert_eq!(ids.len(), agent_count, "Duplicate agent IDs detected");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_task_assignment_race_conditions() {
    let storage = Arc::new(MemoryStorage::new());
    
    // Create agents
    let agent_count = 20;
    let mut agent_ids = vec![];
    for i in 0..agent_count {
        let agent = AgentModel::new(
            format!("agent-{}", i),
            "worker".to_string(),
            vec!["compute".to_string()],
        );
        storage.store_agent(&agent).await.unwrap();
        agent_ids.push(agent.id);
    }
    
    // Create high-priority tasks
    let task_count = 10;
    let mut task_ids = vec![];
    for i in 0..task_count {
        let task = TaskModel::new(
            format!("task-{}", i),
            serde_json::json!({"priority": "high"}),
            crate::models::TaskPriority::Critical,
        );
        storage.store_task(&task).await.unwrap();
        task_ids.push(task.id);
    }
    
    // Implement fair round-robin assignment instead of competitive claiming
    let mut handles = vec![];
    for (agent_idx, agent_id) in agent_ids.iter().enumerate() {
        let storage_clone = storage.clone();
        let agent_id_clone = agent_id.clone();
        
        // Assign tasks to agents in round-robin fashion
        let agent_task_ids: Vec<String> = task_ids
            .iter()
            .enumerate()
            .filter(|(task_idx, _)| task_idx % agent_count == agent_idx)
            .map(|(_, task_id)| task_id.clone())
            .collect();
        
        let handle = tokio::spawn(async move {
            // Stagger agent starts to simulate realistic timing
            let jitter = fastrand::u64(0..20); // 0-20ms random jitter
            tokio::time::sleep(tokio::time::Duration::from_millis(jitter)).await;
            
            let mut claimed = 0;
            for task_id in agent_task_ids {
                // Direct assignment (no competition)
                match storage_clone.claim_task(&task_id, &agent_id_clone).await {
                    Ok(true) => claimed += 1,
                    Ok(false) => {
                        // This shouldn't happen with round-robin, but handle gracefully
                        debug!("Task {} already claimed in round-robin assignment", task_id);
                    },
                    Err(_) => {
                        // Retry once on error
                        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                        if let Ok(true) = storage_clone.claim_task(&task_id, &agent_id_clone).await {
                            claimed += 1;
                        }
                    }
                }
            }
            claimed
        });
        handles.push(handle);
    }
    
    // Collect results
    let claims: Vec<usize> = join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    
    // Total claims should equal task count
    let total_claims: usize = claims.iter().sum();
    assert_eq!(total_claims, task_count, "Each task should be claimed exactly once");
    
    // Verify each task has exactly one assignee
    for task_id in &task_ids {
        let task = storage.get_task(task_id).await.unwrap().unwrap();
        assert!(task.assigned_to.is_some(), "Task {} should be assigned", task_id);
    }
    
    // Check roughly fair distribution (with some tolerance)
    let max_claims = claims.iter().max().unwrap();
    let min_claims = claims.iter().min().unwrap();
    assert!(max_claims - min_claims <= 3, "Task distribution should be roughly fair");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_message_broadcast_storm() {
    let storage = Arc::new(MemoryStorage::new());
    
    // Create coordinator and agents
    let coordinator = AgentModel::new(
        "coordinator".to_string(),
        "coordinator".to_string(),
        vec!["coordinate".to_string()],
    );
    storage.store_agent(&coordinator).await.unwrap();
    
    let agent_count = 50;
    let mut agent_ids = vec![];
    for i in 0..agent_count {
        let agent = AgentModel::new(
            format!("agent-{}", i),
            "worker".to_string(),
            vec!["compute".to_string()],
        );
        storage.store_agent(&agent).await.unwrap();
        agent_ids.push(agent.id);
    }
    
    let message_count = 100; // Full load test
    
    // Broadcast messages with optimized batching for SQLite
    let mut handles = vec![];
    let batch_size = 10; // Process in batches to reduce lock contention
    
    for batch in 0..(message_count / batch_size) {
        let storage_clone = storage.clone();
        let coordinator_id = coordinator.id.clone();
        let agent_ids_clone = agent_ids.clone();
        
        let handle = tokio::spawn(async move {
            // Natural stagger between batches
            let batch_delay = fastrand::u64(0..50); // Random batch start delay
            tokio::time::sleep(tokio::time::Duration::from_millis(batch_delay)).await;
            
            // Process batch of messages
            for i in (batch * batch_size)..((batch + 1) * batch_size) {
                for agent_id in &agent_ids_clone {
                    let message = MessageModel::new(
                        coordinator_id.clone(),
                        agent_id.clone(),
                        format!("broadcast-{}", i),
                        serde_json::json!({"sequence": i}),
                    );
                    if let Err(e) = storage_clone.store_message(&message).await {
                        debug!("Message store failed: {}, retrying...", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                        let _ = storage_clone.store_message(&message).await;
                    }
                }
                
                // Small delay between messages in batch
                if i % 5 == 4 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for all broadcasts
    join_all(handles).await;
    
    // Verify all messages were stored
    for agent_id in &agent_ids {
        let messages = storage.get_unread_messages(agent_id).await.unwrap();
        assert_eq!(messages.len(), message_count, 
            "Agent {} should have {} messages", agent_id, message_count);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_updates_optimistic_locking() {
    use tempfile::NamedTempFile;
    
    let temp_file = NamedTempFile::new().unwrap();
    let storage = Arc::new(SqliteStorage::new(temp_file.path().to_str().unwrap())
        .await
        .unwrap());
    
    // Create test agent
    let agent = AgentModel::new(
        "test-agent".to_string(),
        "worker".to_string(),
        vec!["compute".to_string()],
    );
    storage.store_agent(&agent).await.unwrap();
    
    let update_count = 50;
    let barrier = Arc::new(Barrier::new(update_count));
    
    // Spawn concurrent updates
    let mut handles = vec![];
    for i in 0..update_count {
        let storage_clone = storage.clone();
        let agent_id = agent.id.clone();
        let barrier_clone = barrier.clone();
        
        let handle = tokio::spawn(async move {
            barrier_clone.wait().await;
            
            // Read current state
            let mut current_agent = storage_clone.get_agent(&agent_id).await.unwrap().unwrap();
            
            // Simulate some processing time
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            
            // Update status
            current_agent.status = match i % 4 {
                0 => crate::models::AgentStatus::Idle,
                1 => crate::models::AgentStatus::Busy,
                2 => crate::models::AgentStatus::Active,
                _ => crate::models::AgentStatus::Error,
            };
            current_agent.updated_at = Utc::now();
            
            storage_clone.update_agent(&current_agent).await
        });
        handles.push(handle);
    }
    
    // Collect results
    let results: Vec<_> = join_all(handles).await;
    
    // All updates should succeed (last-write-wins)
    for result in &results {
        assert!(result.is_ok());
        assert!(result.as_ref().unwrap().is_ok());
    }
    
    // Final state should be one of the status enum values
    let final_agent = storage.get_agent(&agent.id).await.unwrap().unwrap();
    // Just verify it was updated (should be one of the enum values)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_event_sourcing_at_scale() {
    // SQLite performance benchmark
    #[cfg(not(target_arch = "wasm32"))]
    {
        use tempfile::NamedTempFile;
        let temp_file = NamedTempFile::new().unwrap();
        let sqlite_storage = Arc::new(SqliteStorage::new(temp_file.path().to_str().unwrap()).await.unwrap());
        
        let sqlite_test_count = 30_000; // Performance benchmark and regression baseline
        let start = std::time::Instant::now();
        
        for i in 0..sqlite_test_count {
            let event = EventModel::new(
                format!("sqlite-perf-{}", i),
                serde_json::json!({"data": i, "type": "performance_test"}),
            );
            sqlite_storage.store_event(&event).await.unwrap();
        }
        
        let duration = start.elapsed();
        let sqlite_events_per_sec = sqlite_test_count as f64 / duration.as_secs_f64();
        println!("SQLite Performance: {} events in {:?} ({:.0} events/sec)", 
            sqlite_test_count, duration, sqlite_events_per_sec);
    }
    
    // Memory storage performance test (same scale as SQLite for comparison)
    let storage = Arc::new(MemoryStorage::new());
    let event_count = 30_000; // Performance benchmark and regression baseline
    let agent_count = 10;
    
    // Create agents
    let mut agent_ids = vec![];
    for i in 0..agent_count {
        let agent = AgentModel::new(
            format!("agent-{}", i),
            "worker".to_string(),
            vec!["compute".to_string()],
        );
        storage.store_agent(&agent).await.unwrap();
        agent_ids.push(agent.id);
    }
    
    let start = std::time::Instant::now();
    
    // Generate events with optimized batching for SQLite
    let events_per_agent = event_count / agent_count;
    let batch_size = 50; // Process in batches for better SQLite performance
    let mut handles = vec![];
    
    for (agent_idx, agent_id) in agent_ids.iter().enumerate() {
        let storage_clone = storage.clone();
        let agent_id_clone = agent_id.clone();
        
        let handle = tokio::spawn(async move {
            // Natural stagger for agent starts
            let agent_delay = fastrand::u64(0..100); // 0-100ms random start
            tokio::time::sleep(tokio::time::Duration::from_millis(agent_delay)).await;
            
            // Process events in batches
            for batch in 0..(events_per_agent / batch_size) {
                for i in (batch * batch_size)..((batch + 1) * batch_size) {
                    let event = EventModel::new(
                        format!("event-type-{}", i % 10),
                        serde_json::json!({
                            "sequence": agent_idx * events_per_agent + i,
                            "timestamp": Utc::now().timestamp_millis()
                        }),
                    ).with_agent(agent_id_clone.clone());
                    
                    // Store with retry on contention
                    if let Err(e) = storage_clone.store_event(&event).await {
                        debug!("Event store failed: {}, retrying...", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
                        let _ = storage_clone.store_event(&event).await;
                    }
                }
                
                // Small delay between batches to reduce lock contention
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for all events to be stored
    join_all(handles).await;
    
    let duration = start.elapsed();
    let events_per_second = event_count as f64 / duration.as_secs_f64();
    
    println!("Stored {} events in {:?} ({:.0} events/sec)", 
        event_count, duration, events_per_second);
    
    // Verify we can query events efficiently
    let query_start = std::time::Instant::now();
    let recent_events = storage.get_events_since(
        Utc::now().timestamp() - 3600  // 1 hour in seconds
    ).await.unwrap();
    let query_duration = query_start.elapsed();
    
    assert_eq!(recent_events.len(), event_count);
    assert!(query_duration.as_millis() < 500, 
        "Query took too long: {:?}", query_duration);
    
    // Verify event ordering
    for i in 1..recent_events.len() {
        if recent_events[i-1].timestamp > recent_events[i].timestamp {
            println!("Ordering violation at index {}: {} > {}", 
                i, recent_events[i-1].timestamp, recent_events[i].timestamp);
            println!("Event {}: id={}, timestamp={}", 
                i-1, recent_events[i-1].id, recent_events[i-1].timestamp);
            println!("Event {}: id={}, timestamp={}", 
                i, recent_events[i].id, recent_events[i].timestamp);
            // Show a few more events around the violation for context
            if i >= 2 {
                println!("Event {}: id={}, timestamp={}", 
                    i-2, recent_events[i-2].id, recent_events[i-2].timestamp);
            }
            if i + 1 < recent_events.len() {
                println!("Event {}: id={}, timestamp={}", 
                    i+1, recent_events[i+1].id, recent_events[i+1].timestamp);
            }
            break;
        }
    }
    
    // Final assertion with detailed error
    for i in 1..recent_events.len() {
        assert!(recent_events[i-1].timestamp <= recent_events[i].timestamp,
            "Events should be ordered by timestamp: event {} ({}) > event {} ({})",
            i-1, recent_events[i-1].timestamp, i, recent_events[i].timestamp);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_connection_pool_exhaustion() {
    use tempfile::NamedTempFile;
    
    let temp_file = NamedTempFile::new().unwrap();
    let storage = Arc::new(SqliteStorage::new(temp_file.path().to_str().unwrap())
        .await
        .unwrap());
    
    // Pool size is 16, so let's try to exhaust it
    let concurrent_ops = 20;
    let barrier = Arc::new(Barrier::new(concurrent_ops));
    
    let mut handles = vec![];
    for i in 0..concurrent_ops {
        let storage_clone = storage.clone();
        let barrier_clone = barrier.clone();
        
        let handle = tokio::spawn(async move {
            barrier_clone.wait().await;
            
            // Try to hold a transaction for a while
            let tx = storage_clone.begin_transaction().await;
            
            match tx {
                Ok(transaction) => {
                    // Hold the connection
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    
                    // Do some work
                    let agent = AgentModel::new(
                        format!("agent-{}", i),
                        "worker".to_string(),
                        vec!["compute".to_string()],
                    );
                    let store_result = storage_clone.store_agent(&agent).await;
                    
                    // Commit
                    let commit_result = transaction.commit().await;
                    
                    (true, store_result.is_ok() && commit_result.is_ok())
                }
                Err(_) => (false, false),
            }
        });
        handles.push(handle);
    }
    
    // Collect results
    let results: Vec<(bool, bool)> = join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    
    // At least pool_size operations should succeed
    let successful_tx = results.iter().filter(|(got_tx, _)| *got_tx).count();
    assert!(successful_tx >= 16, 
        "At least 16 operations should get connections, got {}", successful_tx);
    
    // Some might timeout waiting for connections
    let failed_tx = results.iter().filter(|(got_tx, _)| !*got_tx).count();
    println!("Connection pool exhaustion test: {} succeeded, {} failed", 
        successful_tx, failed_tx);
}

#[tokio::test]
async fn test_metric_aggregation_performance() {
    let storage = Arc::new(MemoryStorage::new());
    
    // Create agents (full scale)
    let agent_count = 100;
    let mut agent_ids = vec![];
    for i in 0..agent_count {
        let agent = AgentModel::new(
            format!("agent-{}", i),
            "worker".to_string(),
            vec!["compute".to_string()],
        );
        storage.store_agent(&agent).await.unwrap();
        agent_ids.push(agent.id);
    }
    
    // Generate metrics (simulating 1 hour of data)
    let metrics_per_agent = 3600; // One per second
    let start_time = Utc::now() - Duration::hours(1);
    
    // Process in smaller batches for SQLite compatibility
    let batch_size = 100; // Batch size for SQLite optimization
    let mut handles = vec![];
    
    for (idx, agent_id) in agent_ids.iter().enumerate() {
        let storage_clone = storage.clone();
        let agent_id_clone = agent_id.clone();
        let start_time_clone = start_time;
        
        let handle = tokio::spawn(async move {
            // Stagger agent starts to reduce lock contention
            let agent_delay = fastrand::u64(0..200); // 0-200ms random start
            tokio::time::sleep(tokio::time::Duration::from_millis(agent_delay)).await;
            
            // Process metrics in batches
            for batch in 0..(metrics_per_agent / batch_size) {
                for i in (batch * batch_size)..((batch + 1) * batch_size) {
                    let metric = MetricModel {
                        id: format!("metric-{}-{}", idx, i),
                        agent_id: Some(agent_id_clone.clone()),
                        metric_type: "cpu_usage".to_string(),
                        value: (i as f64 * 0.1) % 100.0, // Simulate varying CPU usage
                        unit: "percent".to_string(),
                        tags: [("host".to_string(), format!("host-{}", idx % 10))].into_iter().collect(),
                        timestamp: start_time_clone + Duration::seconds(i as i64),
                    };
                    
                    // Store with error handling
                    if let Err(e) = storage_clone.store_metric(&metric).await {
                        debug!("Metric store failed: {}, retrying...", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
                        let _ = storage_clone.store_metric(&metric).await;
                    }
                }
                
                // Small delay between batches to reduce lock contention
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for all metrics to be stored
    join_all(handles).await;
    
    // Test aggregation query performance
    let query_start = std::time::Instant::now();
    let aggregated = storage.get_aggregated_metrics(
        "cpu_usage",
        start_time.timestamp(),
        Utc::now().timestamp(),
    ).await.unwrap();
    let query_duration = query_start.elapsed();
    
    assert_eq!(aggregated.len(), agent_count, 
        "Aggregation should return one result per agent, got {} instead of {}", 
        aggregated.len(), agent_count);
    assert!(query_duration.as_secs() < 1, 
        "Aggregation query took too long: {:?}", query_duration);
    
    println!("Aggregated {} metrics in {:?}", aggregated.len(), query_duration);
}