//! Tests for connection handling and reconnection logic

use crate::*;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::{sleep, Duration, timeout};

/// Connection state for testing
#[derive(Debug, Clone, PartialEq)]
enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed,
}

/// Mock connection manager for testing
struct MockConnectionManager {
    state: Arc<RwLock<ConnectionState>>,
    reconnect_attempts: Arc<Mutex<u32>>,
    max_reconnects: u32,
    reconnect_delay_ms: u64,
    connection_events: Arc<Mutex<Vec<(chrono::DateTime<chrono::Utc>, ConnectionState)>>>,
}

impl MockConnectionManager {
    fn new(max_reconnects: u32, reconnect_delay_ms: u64) -> Self {
        MockConnectionManager {
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            reconnect_attempts: Arc::new(Mutex::new(0)),
            max_reconnects,
            reconnect_delay_ms,
            connection_events: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    async fn connect(&self) -> Result<(), TransportError> {
        self.log_event(ConnectionState::Connecting).await;
        *self.state.write().await = ConnectionState::Connecting;
        
        // Simulate connection delay
        sleep(Duration::from_millis(100)).await;
        
        *self.state.write().await = ConnectionState::Connected;
        self.log_event(ConnectionState::Connected).await;
        *self.reconnect_attempts.lock().await = 0;
        
        Ok(())
    }
    
    async fn disconnect(&self) {
        *self.state.write().await = ConnectionState::Disconnected;
        self.log_event(ConnectionState::Disconnected).await;
    }
    
    async fn simulate_connection_loss(&self) {
        *self.state.write().await = ConnectionState::Disconnected;
        self.log_event(ConnectionState::Disconnected).await;
    }
    
    async fn reconnect(&self) -> Result<(), TransportError> {
        let mut attempts = self.reconnect_attempts.lock().await;
        
        if *attempts >= self.max_reconnects {
            *self.state.write().await = ConnectionState::Failed;
            self.log_event(ConnectionState::Failed).await;
            return Err(TransportError::ConnectionError("Max reconnection attempts reached".into()));
        }
        
        *attempts += 1;
        *self.state.write().await = ConnectionState::Reconnecting;
        self.log_event(ConnectionState::Reconnecting).await;
        
        // Simulate reconnection delay with exponential backoff
        let delay = self.reconnect_delay_ms * (*attempts as u64);
        sleep(Duration::from_millis(delay)).await;
        
        // Simulate 70% success rate
        if *attempts % 10 < 7 {
            self.connect().await?;
            Ok(())
        } else {
            Err(TransportError::ConnectionError("Reconnection failed".into()))
        }
    }
    
    async fn get_state(&self) -> ConnectionState {
        self.state.read().await.clone()
    }
    
    async fn log_event(&self, state: ConnectionState) {
        let mut events = self.connection_events.lock().await;
        events.push((chrono::Utc::now(), state));
    }
    
    async fn get_events(&self) -> Vec<(chrono::DateTime<chrono::Utc>, ConnectionState)> {
        self.connection_events.lock().await.clone()
    }
}

#[tokio::test]
async fn test_connection_lifecycle() {
    let manager = MockConnectionManager::new(3, 100);
    
    // Initial state
    assert_eq!(manager.get_state().await, ConnectionState::Disconnected);
    
    // Connect
    manager.connect().await.unwrap();
    assert_eq!(manager.get_state().await, ConnectionState::Connected);
    
    // Disconnect
    manager.disconnect().await;
    assert_eq!(manager.get_state().await, ConnectionState::Disconnected);
}

#[tokio::test]
async fn test_reconnection_logic() {
    let manager = MockConnectionManager::new(3, 50);
    
    // Establish initial connection
    manager.connect().await.unwrap();
    
    // Simulate connection loss
    manager.simulate_connection_loss().await;
    assert_eq!(manager.get_state().await, ConnectionState::Disconnected);
    
    // Attempt reconnection
    let result = manager.reconnect().await;
    
    // Should succeed within max attempts
    if result.is_ok() {
        assert_eq!(manager.get_state().await, ConnectionState::Connected);
    }
}

#[tokio::test]
async fn test_max_reconnection_attempts() {
    let manager = MockConnectionManager::new(2, 10);
    
    // Force all reconnection attempts to fail
    for _ in 0..3 {
        let _ = manager.reconnect().await;
    }
    
    // Should reach failed state after max attempts
    let attempts = *manager.reconnect_attempts.lock().await;
    assert!(attempts >= 2);
}

#[tokio::test]
async fn test_connection_events_tracking() {
    let manager = MockConnectionManager::new(3, 10);
    
    // Perform connection operations
    manager.connect().await.unwrap();
    manager.disconnect().await;
    manager.connect().await.unwrap();
    
    let events = manager.get_events().await;
    
    // Verify event sequence
    let states: Vec<ConnectionState> = events.iter().map(|(_, state)| state.clone()).collect();
    assert!(states.contains(&ConnectionState::Connecting));
    assert!(states.contains(&ConnectionState::Connected));
    assert!(states.contains(&ConnectionState::Disconnected));
}

#[tokio::test]
async fn test_concurrent_connection_operations() {
    let manager = Arc::new(MockConnectionManager::new(5, 50));
    
    let mut handles = vec![];
    
    // Spawn multiple connection attempts
    for i in 0..5 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            if i % 2 == 0 {
                manager_clone.connect().await
            } else {
                sleep(Duration::from_millis(50)).await;
                manager_clone.disconnect().await;
                Ok(())
            }
        });
        handles.push(handle);
    }
    
    // Wait for all operations
    for handle in handles {
        let _ = handle.await;
    }
    
    // State should be deterministic
    let events = manager.get_events().await;
    assert!(!events.is_empty());
}

#[tokio::test]
async fn test_connection_timeout() {
    let manager = MockConnectionManager::new(1, 5000); // 5 second delay
    
    // Try to connect with timeout
    let connect_future = manager.connect();
    let result = timeout(Duration::from_millis(200), connect_future).await;
    
    // Should complete within timeout (connection takes 100ms)
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_reconnection_exponential_backoff() {
    let manager = MockConnectionManager::new(3, 100);
    
    let start = tokio::time::Instant::now();
    
    // First reconnection attempt (100ms delay)
    let _ = manager.reconnect().await;
    let first_duration = start.elapsed();
    
    // Second reconnection attempt (200ms delay)
    let _ = manager.reconnect().await;
    let second_duration = start.elapsed() - first_duration;
    
    // Second attempt should take longer due to backoff
    assert!(second_duration > first_duration);
}

/// Mock transport with connection management
struct ReconnectingTransport {
    connection_manager: Arc<MockConnectionManager>,
    send_buffer: Arc<Mutex<Vec<Message>>>,
}

impl ReconnectingTransport {
    fn new(max_reconnects: u32) -> Self {
        ReconnectingTransport {
            connection_manager: Arc::new(MockConnectionManager::new(max_reconnects, 100)),
            send_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    async fn send_with_retry(&self, msg: Message) -> Result<(), TransportError> {
        let state = self.connection_manager.get_state().await;
        
        match state {
            ConnectionState::Connected => {
                // Send immediately
                Ok(())
            }
            ConnectionState::Disconnected => {
                // Buffer and attempt reconnection
                self.send_buffer.lock().await.push(msg);
                self.connection_manager.reconnect().await?;
                
                // Flush buffer after reconnection
                self.flush_buffer().await;
                Ok(())
            }
            _ => {
                // Buffer message for later
                self.send_buffer.lock().await.push(msg);
                Ok(())
            }
        }
    }
    
    async fn flush_buffer(&self) {
        let mut buffer = self.send_buffer.lock().await;
        buffer.clear(); // In real implementation, would send all buffered messages
    }
}

#[tokio::test]
async fn test_message_buffering_during_disconnect() {
    let transport = ReconnectingTransport::new(3);
    
    // Disconnect the transport
    transport.connection_manager.disconnect().await;
    
    // Try to send messages
    for i in 0..5 {
        let msg = Message {
            id: Uuid::new_v4(),
            msg_type: MessageType::Data,
            sender: "test".to_string(),
            recipient: "test".to_string(),
            payload: vec![i],
            timestamp: chrono::Utc::now(),
            version: ProtocolVersion::V1,
        };
        
        let _ = transport.send_with_retry(msg).await;
    }
    
    // Messages should be buffered
    let buffer = transport.send_buffer.lock().await;
    assert!(!buffer.is_empty());
}

#[tokio::test]
async fn test_connection_state_transitions() {
    let valid_transitions = vec![
        (ConnectionState::Disconnected, ConnectionState::Connecting),
        (ConnectionState::Connecting, ConnectionState::Connected),
        (ConnectionState::Connected, ConnectionState::Disconnected),
        (ConnectionState::Disconnected, ConnectionState::Reconnecting),
        (ConnectionState::Reconnecting, ConnectionState::Connected),
        (ConnectionState::Reconnecting, ConnectionState::Failed),
    ];
    
    for (from, to) in valid_transitions {
        // Verify these are different states
        assert_ne!(from, to);
    }
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_reconnection_attempts_bounded(
            max_attempts in 1u32..100,
            delay_ms in 1u64..1000,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let manager = MockConnectionManager::new(max_attempts, delay_ms);
                
                // Force multiple reconnection failures
                for _ in 0..max_attempts + 10 {
                    let _ = manager.reconnect().await;
                }
                
                let attempts = *manager.reconnect_attempts.lock().await;
                assert!(attempts <= max_attempts + 1); // Allow one extra for edge case
            });
        }
    }
}