//! Simple task tests that match the actual API

use crate::task::*;
use crate::agent::Capability;

#[test]
fn test_task_creation() {
    let task = Task::new("task-1", "compute");
    
    assert_eq!(task.id.0, "task-1");
    assert_eq!(task.task_type, "compute");
    assert_eq!(task.priority, TaskPriority::Normal);
    assert_eq!(task.retry_count, 0);
    assert_eq!(task.max_retries, 3);
}

#[test]
fn test_task_builder_pattern() {
    let task = Task::new("task-1", "analysis")
        .with_priority(TaskPriority::High)
        .with_timeout(5000)
        .with_payload(TaskPayload::Text("Analyze this data".into()))
        .require_capability("analyze");
    
    assert_eq!(task.priority, TaskPriority::High);
    assert_eq!(task.timeout_ms, Some(5000));
    assert!(matches!(task.payload, TaskPayload::Text(_)));
    assert_eq!(task.required_capabilities.len(), 1);
    assert_eq!(task.required_capabilities[0], "analyze");
}

#[test]
fn test_task_priority_ordering() {
    assert!(TaskPriority::Critical > TaskPriority::High);
    assert!(TaskPriority::High > TaskPriority::Normal);
    assert!(TaskPriority::Normal > TaskPriority::Low);
}

#[test]
fn test_task_retry_logic() {
    let mut task = Task::new("retry-task", "compute");
    
    assert!(task.can_retry());
    assert_eq!(task.retry_count, 0);
    
    task.increment_retry();
    assert_eq!(task.retry_count, 1);
    assert!(task.can_retry());
    
    // Max out retries
    task.increment_retry();
    task.increment_retry();
    assert_eq!(task.retry_count, 3);
    assert!(!task.can_retry());
}

#[test]
fn test_task_result_success() {
    let result = TaskResult::success("Operation completed")
        .with_task_id(TaskId::new("task-1"))
        .with_execution_time(150);
    
    assert_eq!(result.status, TaskStatus::Completed);
    assert!(result.output.is_some());
    assert!(result.error.is_none());
    assert_eq!(result.execution_time_ms, 150);
}

#[test]
fn test_task_result_failure() {
    let result = TaskResult::failure("Operation failed")
        .with_task_id(TaskId::new("task-2"))
        .with_execution_time(75);
    
    assert_eq!(result.status, TaskStatus::Failed);
    assert!(result.output.is_none());
    assert!(result.error.is_some());
    assert_eq!(result.execution_time_ms, 75);
}

#[test]
fn test_task_payload_variants() {
    let empty_payload = TaskPayload::Empty;
    let text_payload = TaskPayload::Text("Hello".into());
    let binary_payload = TaskPayload::Binary(vec![1, 2, 3, 4]);
    let json_payload = TaskPayload::Json(r#"{"key": "value"}"#.into());
    
    match empty_payload {
        TaskPayload::Empty => {},
        _ => panic!("Expected empty payload"),
    }
    
    match text_payload {
        TaskPayload::Text(s) => assert_eq!(s, "Hello"),
        _ => panic!("Expected text payload"),
    }
    
    match binary_payload {
        TaskPayload::Binary(bytes) => assert_eq!(bytes, vec![1, 2, 3, 4]),
        _ => panic!("Expected binary payload"),
    }
    
    match json_payload {
        TaskPayload::Json(s) => assert!(s.contains("key")),
        _ => panic!("Expected JSON payload"),
    }
}

#[test]
fn test_task_status_transitions() {
    let statuses = [TaskStatus::Pending,
        TaskStatus::Assigned,
        TaskStatus::Running,
        TaskStatus::Completed,
        TaskStatus::Failed,
        TaskStatus::Cancelled,
        TaskStatus::TimedOut];
    
    // Verify all statuses are distinct
    for (i, status1) in statuses.iter().enumerate() {
        for (j, status2) in statuses.iter().enumerate() {
            if i == j {
                assert_eq!(status1, status2);
            } else {
                assert_ne!(status1, status2);
            }
        }
    }
}

#[test]
fn test_task_id_display() {
    let task_id = TaskId::new("my-task-123");
    assert_eq!(format!("{task_id}"), "my-task-123");
}

#[test]
fn test_distribution_strategy_default() {
    let strategy = DistributionStrategy::default();
    assert_eq!(strategy, DistributionStrategy::LeastLoaded);
}

#[test]
fn test_task_with_multiple_capabilities() {
    let task = Task::new("multi-cap-task", "complex")
        .require_capability("compute")
        .require_capability("storage")
        .require_capability("network");
    
    assert_eq!(task.required_capabilities.len(), 3);
    assert!(task.required_capabilities.contains(&"compute".to_string()));
    assert!(task.required_capabilities.contains(&"storage".to_string()));
    assert!(task.required_capabilities.contains(&"network".to_string()));
}

#[test]
fn test_task_custom_retry_limit() {
    let mut task = Task::new("custom-retry", "compute");
    task.max_retries = 5; // Set directly since no builder method exists
    
    assert_eq!(task.max_retries, 5);
    
    // Test custom retry limit
    for _ in 0..5 {
        assert!(task.can_retry());
        task.increment_retry();
    }
    
    assert!(!task.can_retry());
    assert_eq!(task.retry_count, 5);
}

#[test]
fn test_task_result_builder_chain() {
    let result = TaskResult::success("Success")
        .with_task_id(TaskId::new("chain-task"))
        .with_execution_time(200);
    
    assert_eq!(result.status, TaskStatus::Completed);
    assert_eq!(result.task_id.to_string(), "chain-task");
    assert_eq!(result.execution_time_ms, 200);
    assert!(result.output.is_some());
}

#[test]
fn test_task_timeout_behavior() {
    let task_no_timeout = Task::new("no-timeout", "compute");
    assert_eq!(task_no_timeout.timeout_ms, None);
    
    let task_with_timeout = Task::new("with-timeout", "compute")
        .with_timeout(1000);
    assert_eq!(task_with_timeout.timeout_ms, Some(1000));
}


#[test]
fn test_distribution_strategy_all_variants() {
    let strategies = [DistributionStrategy::RoundRobin,
        DistributionStrategy::LeastLoaded,
        DistributionStrategy::Random,
        DistributionStrategy::Priority,
        DistributionStrategy::CapabilityBased];
    
    // Verify all strategies are distinct
    for (i, strategy1) in strategies.iter().enumerate() {
        for (j, strategy2) in strategies.iter().enumerate() {
            if i == j {
                assert_eq!(strategy1, strategy2);
            } else {
                assert_ne!(strategy1, strategy2);
            }
        }
    }
}

#[test]
fn test_task_payload_size() {
    // Test large payloads
    let large_text = "x".repeat(10_000);
    let large_task = Task::new("large-task", "process")
        .with_payload(TaskPayload::Text(large_text.clone()));
    
    match large_task.payload {
        TaskPayload::Text(s) => assert_eq!(s.len(), 10_000),
        _ => panic!("Expected text payload"),
    }
    
    // Test binary payload
    let large_binary = vec![0u8; 5_000];
    let binary_task = Task::new("binary-task", "process")
        .with_payload(TaskPayload::Binary(large_binary.clone()));
    
    match binary_task.payload {
        TaskPayload::Binary(b) => assert_eq!(b.len(), 5_000),
        _ => panic!("Expected binary payload"),
    }
}

#[test]
fn test_task_equality() {
    let task1 = Task::new("task-eq", "compute");
    let task2 = Task::new("task-eq", "compute");
    let task3 = Task::new("task-different", "compute");
    
    // Tasks with same ID should be equal
    assert_eq!(task1.id, task2.id);
    // Tasks with different IDs should not be equal
    assert_ne!(task1.id, task3.id);
}

#[test]
fn test_task_result_error_details() {
    let detailed_error = "Connection timeout after 5 attempts";
    let result = TaskResult::failure(detailed_error)
        .with_task_id(TaskId::new("error-task"));
    
    assert_eq!(result.status, TaskStatus::Failed);
    assert_eq!(result.error, Some(detailed_error.to_string()));
}
