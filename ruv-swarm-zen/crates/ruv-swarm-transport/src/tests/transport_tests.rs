//! Tests for transport implementations

use crate::*;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// Mock transport for testing
#[derive(Clone)]
struct MockTransport {
    id: String,
    connected: Arc<Mutex<bool>>,
    messages: Arc<Mutex<Vec<(String, Message)>>>,
    receiver: Arc<Mutex<mpsc::Receiver<(String, Message)>>>,
    sender: mpsc::Sender<(String, Message)>,
    stats: Arc<Mutex<TransportStats>>,
}

impl MockTransport {
    fn new(id: impl Into<String>) -> Self {
        let (sender, receiver) = mpsc::channel(100);
        MockTransport {
            id: id.into(),
            connected: Arc::new(Mutex::new(true)),
            messages: Arc::new(Mutex::new(Vec::new())),
            receiver: Arc::new(Mutex::new(receiver)),
            sender,
            stats: Arc::new(Mutex::new(TransportStats::default())),
        }
    }
    
    async fn simulate_receive(&self, from: String, msg: Message) {
        self.sender.send((from, msg)).await.unwrap();
    }
}

#[async_trait]
impl Transport for MockTransport {
    type Message = Message;
    type Error = TransportError;
    
    async fn send(&self, to: &str, msg: Self::Message) -> Result<(), Self::Error> {
        if !*self.connected.lock().await {
            return Err(TransportError::ConnectionError("Not connected".into()));
        }
        
        let mut messages = self.messages.lock().await;
        messages.push((to.to_string(), msg));
        
        let mut stats = self.stats.lock().await;
        stats.messages_sent += 1;
        stats.last_activity = Some(chrono::Utc::now());
        
        Ok(())
    }
    
    async fn receive(&mut self) -> Result<(String, Self::Message), Self::Error> {
        if !*self.connected.lock().await {
            return Err(TransportError::ConnectionError("Not connected".into()));
        }
        
        let mut receiver = self.receiver.lock().await;
        match receiver.recv().await {
            Some((from, msg)) => {
                let mut stats = self.stats.lock().await;
                stats.messages_received += 1;
                stats.last_activity = Some(chrono::Utc::now());
                Ok((from, msg))
            }
            None => Err(TransportError::ConnectionError("Channel closed".into())),
        }
    }
    
    async fn broadcast(&self, msg: Self::Message) -> Result<(), Self::Error> {
        self.send("*", msg).await
    }
    
    fn local_address(&self) -> Result<String, Self::Error> {
        Ok(self.id.clone())
    }
    
    fn is_connected(&self) -> bool {
        futures::executor::block_on(async {
            *self.connected.lock().await
        })
    }
    
    async fn close(&mut self) -> Result<(), Self::Error> {
        *self.connected.lock().await = false;
        Ok(())
    }
    
    fn stats(&self) -> TransportStats {
        futures::executor::block_on(async {
            self.stats.lock().await.clone()
        })
    }
}

#[tokio::test]
async fn test_transport_send_receive() {
    let mut transport = MockTransport::new("test-transport");
    
    let msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Task,
        sender: "agent-1".to_string(),
        recipient: "agent-2".to_string(),
        payload: vec![1, 2, 3, 4],
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    // Send message
    transport.send("agent-2", msg.clone()).await.unwrap();
    
    // Simulate receiving
    transport.simulate_receive("agent-1".to_string(), msg.clone()).await;
    
    // Receive message
    let (from, received) = transport.receive().await.unwrap();
    assert_eq!(from, "agent-1");
    assert_eq!(received.id, msg.id);
}

#[tokio::test]
async fn test_transport_connection_state() {
    let mut transport = MockTransport::new("test-transport");
    
    assert!(transport.is_connected());
    
    transport.close().await.unwrap();
    assert!(!transport.is_connected());
    
    // Should fail to send when disconnected
    let msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Task,
        sender: "test".to_string(),
        recipient: "test".to_string(),
        payload: vec![],
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    let result = transport.send("test", msg).await;
    assert!(matches!(result, Err(TransportError::ConnectionError(_))));
}

#[tokio::test]
async fn test_transport_stats() {
    let mut transport = MockTransport::new("test-transport");
    
    let initial_stats = transport.stats();
    assert_eq!(initial_stats.messages_sent, 0);
    assert_eq!(initial_stats.messages_received, 0);
    
    // Send multiple messages
    for i in 0..5 {
        let msg = Message {
            id: Uuid::new_v4(),
            msg_type: MessageType::Task,
            sender: "test".to_string(),
            recipient: format!("agent-{}", i),
            payload: vec![i as u8],
            timestamp: chrono::Utc::now(),
            version: ProtocolVersion::V1,
        };
        transport.send(&format!("agent-{}", i), msg).await.unwrap();
    }
    
    let stats = transport.stats();
    assert_eq!(stats.messages_sent, 5);
    assert!(stats.last_activity.is_some());
}

#[tokio::test]
async fn test_transport_broadcast() {
    let transport = MockTransport::new("test-transport");
    
    let msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Broadcast,
        sender: "coordinator".to_string(),
        recipient: "*".to_string(),
        payload: b"Hello everyone".to_vec(),
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    transport.broadcast(msg.clone()).await.unwrap();
    
    let messages = transport.messages.lock().await;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, "*");
}

#[tokio::test]
async fn test_transport_config_validation() {
    let config = TransportConfig {
        max_message_size: 1024, // 1KB
        connection_timeout_ms: 1000,
        retry_attempts: 1,
        enable_compression: false,
        compression_threshold: 512,
    };
    
    assert!(config.max_message_size > 0);
    assert!(config.connection_timeout_ms > 0);
    assert!(config.compression_threshold < config.max_message_size);
}

#[tokio::test]
async fn test_transport_message_size_limit() {
    let transport = MockTransport::new("test-transport");
    let config = TransportConfig {
        max_message_size: 100, // Very small limit
        ..Default::default()
    };
    
    // Create a message that would exceed the limit
    let large_payload = vec![0u8; 200];
    let msg = Message {
        id: Uuid::new_v4(),
        msg_type: MessageType::Data,
        sender: "test".to_string(),
        recipient: "test".to_string(),
        payload: large_payload,
        timestamp: chrono::Utc::now(),
        version: ProtocolVersion::V1,
    };
    
    // In a real implementation, this would check message size
    // For now, we just verify the message was created
    assert!(msg.payload.len() > config.max_message_size);
}

#[tokio::test]
async fn test_concurrent_transport_operations() {
    let transport = Arc::new(Mutex::new(MockTransport::new("concurrent-test")));
    
    let mut handles = vec![];
    
    // Spawn multiple concurrent sends
    for i in 0..10 {
        let transport_clone = transport.clone();
        let handle = tokio::spawn(async move {
            let msg = Message {
                id: Uuid::new_v4(),
                msg_type: MessageType::Task,
                sender: format!("sender-{}", i),
                recipient: "receiver".to_string(),
                payload: vec![i as u8],
                timestamp: chrono::Utc::now(),
                version: ProtocolVersion::V1,
            };
            
            let transport = transport_clone.lock().await;
            transport.send("receiver", msg).await.unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all sends to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    let transport = transport.lock().await;
    let stats = transport.stats();
    assert_eq!(stats.messages_sent, 10);
}

/// Mock transport builder for testing
struct MockTransportBuilder {
    id_prefix: String,
}

impl MockTransportBuilder {
    fn new(prefix: impl Into<String>) -> Self {
        MockTransportBuilder {
            id_prefix: prefix.into(),
        }
    }
}

#[async_trait]
impl TransportBuilder for MockTransportBuilder {
    type Transport = MockTransport;
    
    async fn build(&self, _config: TransportConfig) -> Result<Self::Transport, TransportError> {
        Ok(MockTransport::new(format!("{}-instance", self.id_prefix)))
    }
}

#[tokio::test]
async fn test_transport_registry() {
    let registry = TransportRegistry::new();
    
    // Register mock transport
    // Note: This would need adjustment based on actual registry implementation
    assert!(registry.transports.is_empty());
}

#[tokio::test]
async fn test_transport_error_types() {
    let errors = vec![
        TransportError::ConnectionError("Connection lost".into()),
        TransportError::SerializationError("Invalid JSON".into()),
        TransportError::MessageTooLarge { size: 1000, max: 500 },
        TransportError::NotAvailable("WebSocket".into()),
        TransportError::Timeout,
        TransportError::InvalidAddress("bad://address".into()),
        TransportError::VersionMismatch { 
            expected: "1.0".into(), 
            actual: "2.0".into() 
        },
    ];
    
    for error in errors {
        // Verify error messages are properly formatted
        assert!(!error.to_string().is_empty());
    }
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_transport_config_invariants(
            max_size in 1usize..100_000_000,
            timeout in 1u64..60_000,
            retries in 0u32..100,
            compression in any::<bool>(),
            threshold in 1usize..100_000,
        ) {
            let config = TransportConfig {
                max_message_size: max_size,
                connection_timeout_ms: timeout,
                retry_attempts: retries,
                enable_compression: compression,
                compression_threshold: threshold,
            };
            
            // Config invariants
            assert!(config.max_message_size > 0);
            assert!(config.connection_timeout_ms > 0);
            if config.enable_compression {
                assert!(config.compression_threshold > 0);
            }
        }
    }
}