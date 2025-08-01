//! Tests for agent messaging and communication types

use crate::agent::{AgentMessage, MessageType};

#[test]
fn test_agent_message_creation() {
    let msg = AgentMessage {
        from: "agent-1".to_string(),
        to: "agent-2".to_string(),
        payload: "test data".to_string(),
        msg_type: MessageType::TaskAssignment,
        correlation_id: Some("corr-123".to_string()),
    };
    
    assert_eq!(msg.from, "agent-1");
    assert_eq!(msg.to, "agent-2");
    assert_eq!(msg.payload, "test data");
    assert_eq!(msg.msg_type, MessageType::TaskAssignment);
    assert_eq!(msg.correlation_id, Some("corr-123".to_string()));
}

#[derive(Debug, Clone, PartialEq)]
struct TaskInfo {
    task_id: String,
    priority: u8,
}

#[test]
fn test_agent_message_with_different_payloads() {
    // Test with string payload
    let string_msg = AgentMessage {
        from: "sender".to_string(),
        to: "receiver".to_string(),
        payload: "Hello, agent!".to_string(),
        msg_type: MessageType::Coordination,
        correlation_id: None,
    };
    assert_eq!(string_msg.payload, "Hello, agent!");
    
    // Test with numeric payload
    let numeric_msg = AgentMessage {
        from: "agent-a".to_string(),
        to: "agent-b".to_string(),
        payload: 42u32,
        msg_type: MessageType::StatusUpdate,
        correlation_id: Some("status-42".to_string()),
    };
    assert_eq!(numeric_msg.payload, 42);
    
    // Test with struct payload
    let task_msg = AgentMessage {
        from: "coordinator".to_string(),
        to: "worker".to_string(),
        payload: TaskInfo {
            task_id: "task-123".to_string(),
            priority: 5,
        },
        msg_type: MessageType::TaskAssignment,
        correlation_id: Some("assign-123".to_string()),
    };
    
    assert_eq!(task_msg.payload.task_id, "task-123");
    assert_eq!(task_msg.payload.priority, 5);
}

#[test]
fn test_message_type_variants() {
    let types = [
        MessageType::TaskAssignment,
        MessageType::TaskResult,
        MessageType::StatusUpdate,
        MessageType::Coordination,
        MessageType::Error,
    ];
    
    // Verify all types are distinct
    for (i, t1) in types.iter().enumerate() {
        for (j, t2) in types.iter().enumerate() {
            if i == j {
                assert_eq!(t1, t2);
            } else {
                assert_ne!(t1, t2);
            }
        }
    }
}

#[test]
fn test_message_type_serialization() {
    let types = [
        MessageType::TaskAssignment,
        MessageType::TaskResult,
        MessageType::StatusUpdate,
        MessageType::Coordination,
        MessageType::Error,
    ];
    
    for msg_type in types {
        // Test serialization
        let json = serde_json::to_string(&msg_type).unwrap();
        
        // Test deserialization
        let deserialized: MessageType = serde_json::from_str(&json).unwrap();
        assert_eq!(msg_type, deserialized);
    }
}

#[test]
fn test_agent_message_serialization() {
    let msg = AgentMessage {
        from: "agent-1".to_string(),
        to: "agent-2".to_string(),
        payload: "test payload".to_string(),
        msg_type: MessageType::Coordination,
        correlation_id: Some("corr-456".to_string()),
    };
    
    // Serialize to JSON
    let json = serde_json::to_string(&msg).unwrap();
    
    // Deserialize back
    let deserialized: AgentMessage<String> = serde_json::from_str(&json).unwrap();
    
    assert_eq!(msg.from, deserialized.from);
    assert_eq!(msg.to, deserialized.to);
    assert_eq!(msg.payload, deserialized.payload);
    assert_eq!(msg.msg_type, deserialized.msg_type);
    assert_eq!(msg.correlation_id, deserialized.correlation_id);
}

#[test]
fn test_message_patterns() {
    // Test request-response pattern
    let request = AgentMessage {
        from: "client".to_string(),
        to: "server".to_string(),
        payload: "get_status".to_string(),
        msg_type: MessageType::Coordination,
        correlation_id: Some("req-001".to_string()),
    };
    
    let response = AgentMessage {
        from: "server".to_string(),
        to: "client".to_string(),
        payload: "status: ok".to_string(),
        msg_type: MessageType::StatusUpdate,
        correlation_id: request.correlation_id.clone(),
    };
    
    // Verify correlation
    assert_eq!(request.correlation_id, response.correlation_id);
    assert_eq!(response.from, request.to);
    assert_eq!(response.to, request.from);
    
    // Test error response
    let error_response = AgentMessage {
        from: "server".to_string(),
        to: "client".to_string(),
        payload: "Task processing failed".to_string(),
        msg_type: MessageType::Error,
        correlation_id: Some("req-001".to_string()),
    };
    
    assert_eq!(error_response.msg_type, MessageType::Error);
    assert_eq!(error_response.correlation_id, request.correlation_id);
}

#[test]
fn test_broadcast_pattern() {
    // Simulate broadcast message (no specific target)
    let broadcast = AgentMessage {
        from: "coordinator".to_string(),
        to: "*".to_string(), // Convention for broadcast
        payload: "system_shutdown".to_string(),
        msg_type: MessageType::Coordination,
        correlation_id: None, // Broadcasts typically don't need correlation
    };
    
    assert_eq!(broadcast.to, "*");
    assert!(broadcast.correlation_id.is_none());
}