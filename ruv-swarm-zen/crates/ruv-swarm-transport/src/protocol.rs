//! Message protocol definitions for swarm communication

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Protocol version for compatibility checking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProtocolVersion {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
}

impl ProtocolVersion {
    pub const CURRENT: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };

    /// Check if this version is compatible with another
    pub fn is_compatible(&self, other: &Self) -> bool {
        // Major version must match, minor/patch can differ
        self.major == other.major
    }
}

impl std::fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Message types in the swarm protocol
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageType {
    /// Request message expecting a response
    Request {
        /// Correlation ID for matching responses
        correlation_id: Uuid,
        /// Request method/operation
        method: String,
        /// Request parameters
        params: serde_json::Value,
    },

    /// Response to a request
    Response {
        /// Correlation ID matching the request
        correlation_id: Uuid,
        /// Response result (Ok) or error (Err)
        result: Result<serde_json::Value, ErrorResponse>,
    },

    /// Event notification (no response expected)
    Event {
        /// Event name
        name: String,
        /// Event data
        data: serde_json::Value,
    },

    /// Broadcast message to all peers
    Broadcast {
        /// Topic for filtering
        topic: String,
        /// Broadcast data
        data: serde_json::Value,
    },

    /// Heartbeat for connection monitoring
    Heartbeat {
        /// Sequence number
        seq: u64,
    },

    /// Control message for protocol negotiation
    Control {
        /// Control operation
        operation: ControlOperation,
    },
}

/// Error response structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorResponse {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional error data
    pub data: Option<serde_json::Value>,
}

/// Control operations for protocol management
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControlOperation {
    /// Handshake initiation
    Hello {
        /// Protocol version
        version: ProtocolVersion,
        /// Client capabilities
        capabilities: Vec<String>,
        /// Client metadata
        metadata: HashMap<String, String>,
    },

    /// Handshake acknowledgment
    HelloAck {
        /// Negotiated protocol version
        version: ProtocolVersion,
        /// Server capabilities
        capabilities: Vec<String>,
        /// Server metadata
        metadata: HashMap<String, String>,
    },

    /// Graceful disconnect
    Goodbye {
        /// Reason for disconnect
        reason: String,
    },

    /// Request compression mode change
    SetCompression {
        /// Enable/disable compression
        enabled: bool,
        /// Compression algorithm (e.g., "gzip", "zstd")
        algorithm: Option<String>,
    },

    /// Flow control
    FlowControl {
        /// Pause/resume message flow
        pause: bool,
        /// Buffer pressure (0.0 - 1.0)
        pressure: f32,
    },
}

/// Complete message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: Uuid,
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
    /// Source agent/node ID
    pub source: String,
    /// Optional destination (None for broadcasts)
    pub destination: Option<String>,
    /// Protocol version
    pub version: ProtocolVersion,
    /// Message payload
    pub payload: MessageType,
    /// Optional message headers
    pub headers: HashMap<String, String>,
    /// Message priority (0-255, higher = more important)
    pub priority: u8,
    /// Time-to-live (hop count)
    pub ttl: Option<u8>,
}

impl Message {
    /// Create a new message with default values
    pub fn new(source: String, payload: MessageType) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            source,
            destination: None,
            version: ProtocolVersion::CURRENT,
            payload,
            headers: HashMap::new(),
            priority: 128, // Medium priority
            ttl: None,
        }
    }

    /// Create a request message
    pub fn request(source: String, method: String, params: serde_json::Value) -> Self {
        let correlation_id = Uuid::new_v4();
        Self::new(
            source,
            MessageType::Request {
                correlation_id,
                method,
                params,
            },
        )
    }

    /// Create a response message
    pub fn response(
        source: String,
        correlation_id: Uuid,
        result: Result<serde_json::Value, ErrorResponse>,
    ) -> Self {
        Self::new(
            source,
            MessageType::Response {
                correlation_id,
                result,
            },
        )
    }

    /// Create an event message
    pub fn event(source: String, name: String, data: serde_json::Value) -> Self {
        Self::new(source, MessageType::Event { name, data })
    }

    /// Create a broadcast message
    pub fn broadcast(source: String, topic: String, data: serde_json::Value) -> Self {
        Self::new(source, MessageType::Broadcast { topic, data })
    }

    /// Set destination
    pub fn to(mut self, destination: String) -> Self {
        self.destination = Some(destination);
        self
    }

    /// Add header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set TTL
    pub fn with_ttl(mut self, ttl: u8) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Check if message has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            ttl == 0
        } else {
            false
        }
    }

    /// Decrement TTL and return if message should be dropped
    pub fn decrement_ttl(&mut self) -> bool {
        if let Some(ttl) = &mut self.ttl {
            if *ttl > 0 {
                *ttl -= 1;
                false
            } else {
                true // Drop message
            }
        } else {
            false
        }
    }
}

/// Message codec trait for serialization
pub trait MessageCodec: Send + Sync {
    /// Encode a message to bytes
    fn encode(&self, msg: &Message) -> Result<Vec<u8>, crate::TransportError>;

    /// Decode a message from bytes
    fn decode(&self, data: &[u8]) -> Result<Message, crate::TransportError>;
}

/// Binary codec using MessagePack
pub struct BinaryCodec;

impl MessageCodec for BinaryCodec {
    fn encode(&self, msg: &Message) -> Result<Vec<u8>, crate::TransportError> {
        rmp_serde::to_vec(msg).map_err(|e| crate::TransportError::SerializationError(e.to_string()))
    }

    fn decode(&self, data: &[u8]) -> Result<Message, crate::TransportError> {
        rmp_serde::from_slice(data)
            .map_err(|e| crate::TransportError::SerializationError(e.to_string()))
    }
}

/// JSON codec for human-readable format
pub struct JsonCodec;

impl MessageCodec for JsonCodec {
    fn encode(&self, msg: &Message) -> Result<Vec<u8>, crate::TransportError> {
        serde_json::to_vec(msg)
            .map_err(|e| crate::TransportError::SerializationError(e.to_string()))
    }

    fn decode(&self, data: &[u8]) -> Result<Message, crate::TransportError> {
        serde_json::from_slice(data)
            .map_err(|e| crate::TransportError::SerializationError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version_compatibility() {
        let v1 = ProtocolVersion {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let v2 = ProtocolVersion {
            major: 1,
            minor: 1,
            patch: 0,
        };
        let v3 = ProtocolVersion {
            major: 2,
            minor: 0,
            patch: 0,
        };

        assert!(v1.is_compatible(&v2));
        assert!(!v1.is_compatible(&v3));
    }

    #[test]
    fn test_message_creation() {
        let msg = Message::request(
            "agent1".to_string(),
            "compute".to_string(),
            serde_json::json!({"input": 42}),
        );

        assert_eq!(msg.source, "agent1");
        assert!(matches!(msg.payload, MessageType::Request { .. }));
    }

    #[test]
    fn test_message_ttl() {
        let mut msg = Message::event(
            "agent1".to_string(),
            "test".to_string(),
            serde_json::Value::Null,
        )
        .with_ttl(2);

        assert!(!msg.is_expired());
        assert!(!msg.decrement_ttl());
        assert!(!msg.is_expired());
        assert!(!msg.decrement_ttl());
        assert!(msg.is_expired());
        assert!(msg.decrement_ttl());
    }

    #[test]
    fn test_binary_codec() {
        let codec = BinaryCodec;
        let msg = Message::event(
            "test".to_string(),
            "event".to_string(),
            serde_json::json!({"data": "test"}),
        );

        let encoded = codec.encode(&msg).unwrap();
        let decoded = codec.decode(&encoded).unwrap();

        assert_eq!(msg.id, decoded.id);
        assert_eq!(msg.source, decoded.source);
    }
}
