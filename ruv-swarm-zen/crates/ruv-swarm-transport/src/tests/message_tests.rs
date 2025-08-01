//! Tests for message serialization and protocol

use crate::protocol::{Message, MessageType, ProtocolVersion};
use serde_json;
use uuid::Uuid;

#[test]
fn test_message_creation() {
    let msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Task,
        sender: "agent-1".to_string(),
        recipient: "agent-2".to_string(),
        payload: b"Hello, Agent!".to_vec(),
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    assert_eq!(msg.sender, "agent-1");
    assert_eq!(msg.recipient, "agent-2");
    assert_eq!(msg.payload, b"Hello, Agent!");
}

#[test]
fn test_message_serialization() {
    let msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Data,
        sender: "test-sender".to_string(),
        recipient: "test-recipient".to_string(),
        payload: vec![1, 2, 3, 4, 5],
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    // Serialize to JSON
    let serialized = serde_json::to_string(&msg).unwrap();
    assert!(serialized.contains("test-sender"));
    assert!(serialized.contains("test-recipient"));
    
    // Deserialize back
    let deserialized: Message = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.id, msg.id);
    assert_eq!(deserialized.sender, msg.sender);
    assert_eq!(deserialized.recipient, msg.recipient);
    assert_eq!(deserialized.payload, msg.payload);
}

#[test]
fn test_message_types() {
    let types = vec![
        MessageType::Task,
        MessageType::TaskResult,
        MessageType::Control,
        MessageType::Data,
        MessageType::Heartbeat,
        MessageType::Broadcast,
        MessageType::Error,
    ];
    
    for msg_type in types {
        let msg = Message {
            id: Uuid::new_v4(),
            msg_type: msg_type.clone(),
            sender: "test".to_string(),
            recipient: "test".to_string(),
            payload: vec![],
            timestamp: chrono::Utc::now(),
            version: ProtocolVersion::V1,
        };
        
        // Verify serialization works for all message types
        let serialized = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.msg_type, msg_type);
    }
}

#[test]
fn test_protocol_version() {
    let v1_msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Data,
        sender: "v1-sender".to_string(),
        recipient: "v1-recipient".to_string(),
        payload: b"V1 data".to_vec(),
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    assert!(matches!(v1_msg.version, ProtocolVersion::V1));
}

#[test]
fn test_message_with_binary_payload() {
    // Test with various binary patterns
    let test_payloads = vec![
        vec![0x00, 0xFF, 0xAA, 0x55], // Mixed bytes
        vec![0; 1024],                 // All zeros
        (0..=255).collect::<Vec<u8>>(), // All byte values
        vec![],                        // Empty payload
    ];
    
    for payload in test_payloads {
        let msg = Message {
            id: Uuid::new_v4(),
            msg_type: MessageType::Data,
            sender: "binary-test".to_string(),
            recipient: "binary-test".to_string(),
            payload: payload.clone(),
            timestamp: chrono::Utc::now(),
            version: ProtocolVersion::V1,
        };
        
        let serialized = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.payload, payload);
    }
}

#[test]
fn test_message_timestamp_ordering() {
    use std::thread;
    use std::time::Duration;
    
    let msg1 = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Task,
        sender: "test".to_string(),
        recipient: "test".to_string(),
        payload: vec![],
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    thread::sleep(Duration::from_millis(10));
    
    let msg2 = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Task,
        sender: "test".to_string(),
        recipient: "test".to_string(),
        payload: vec![],
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    assert!(msg2.timestamp > msg1.timestamp);
}

#[test]
fn test_message_id_uniqueness() {
    let ids: Vec<Uuid> = (0..1000)
        .map(|_| Uuid::new_v4())
        .collect();
    
    // Check all IDs are unique
    let unique_count = ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, 1000);
}

#[test]
fn test_message_size_calculation() {
    let small_msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Data,
        sender: "a".to_string(),
        recipient: "b".to_string(),
        payload: vec![1, 2, 3],
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    let large_msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Data,
        sender: "agent-with-long-name".to_string(),
        recipient: "another-agent-with-long-name".to_string(),
        payload: vec![0; 10000],
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    let small_size = serde_json::to_vec(&small_msg).unwrap().len();
    let large_size = serde_json::to_vec(&large_msg).unwrap().len();
    
    assert!(large_size > small_size);
    assert!(large_size > 10000); // At least payload size
}

#[test]
fn test_broadcast_message() {
    let broadcast_msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Broadcast,
        sender: "coordinator".to_string(),
        recipient: "*".to_string(), // Broadcast indicator
        payload: b"System announcement".to_vec(),
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    assert_eq!(broadcast_msg.recipient, "*");
    assert!(matches!(broadcast_msg.msg_type, MessageType::Broadcast));
}

#[test]
fn test_error_message() {
    let error_payload = serde_json::json!({
        "error_code": "AGENT_TIMEOUT",
        "message": "Agent failed to respond within timeout",
        "details": {
            "agent_id": "agent-123",
            "timeout_ms": 5000
        }
    });
    
    let error_msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Error,
        sender: "orchestrator".to_string(),
        recipient: "monitor".to_string(),
        payload: serde_json::to_vec(&error_payload).unwrap(),
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    assert!(matches!(error_msg.msg_type, MessageType::Error));
    
    // Verify we can parse the error payload
    let parsed: serde_json::Value = serde_json::from_slice(&error_msg.payload).unwrap();
    assert_eq!(parsed["error_code"], "AGENT_TIMEOUT");
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_message_serialization_roundtrip(
            sender in "[a-zA-Z0-9-]{1,50}",
            recipient in "[a-zA-Z0-9-]{1,50}",
            payload_size in 0usize..10000,
        ) {
            let payload: Vec<u8> = (0..payload_size).map(|i| (i % 256) as u8).collect();
            
            let msg = Message {
                id: Uuid::new_v4(),
                msg_type: MessageType::Data,
                sender: sender.clone(),
                recipient: recipient.clone(),
                payload: payload.clone(),
                timestamp: chrono::Utc::now(),
                version: ProtocolVersion::V1,
            };
            
            let serialized = serde_json::to_string(&msg).unwrap();
            let deserialized: Message = serde_json::from_str(&serialized).unwrap();
            
            assert_eq!(deserialized.sender, sender);
            assert_eq!(deserialized.recipient, recipient);
            assert_eq!(deserialized.payload, payload);
            assert_eq!(deserialized.msg_type, MessageType::Data);
        }
        
        #[test]
        fn test_message_id_generation_unique(count in 1usize..100) {
            let ids: Vec<Uuid> = (0..count).map(|_| Uuid::new_v4()).collect();
            let unique_ids: std::collections::HashSet<_> = ids.iter().collect();
            assert_eq!(ids.len(), unique_ids.len());
        }
    }
}