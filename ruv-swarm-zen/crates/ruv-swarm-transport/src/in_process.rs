//! In-process transport for local communication within the same process

use crate::{protocol::Message, Transport, TransportConfig, TransportError, TransportStats};
use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, info};

/// In-process transport using channels for zero-cost local communication
pub struct InProcessTransport {
    id: String,
    config: TransportConfig,
    registry: Arc<InProcessRegistry>,
    incoming_rx: mpsc::Receiver<(String, Message)>,
    incoming_tx: mpsc::Sender<(String, Message)>,
    broadcast_tx: broadcast::Sender<Message>,
    is_running: Arc<AtomicBool>,
    stats: Arc<RwLock<TransportStats>>,
}

/// Registry for in-process transports
pub struct InProcessRegistry {
    transports: DashMap<String, InProcessEndpoint>,
}

/// Endpoint for a registered transport
struct InProcessEndpoint {
    tx: mpsc::Sender<(String, Message)>,
    broadcast_tx: broadcast::Sender<Message>,
}

impl InProcessRegistry {
    /// Create a new registry
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            transports: DashMap::new(),
        })
    }

    /// Register a transport
    fn register(&self, id: String, endpoint: InProcessEndpoint) {
        self.transports.insert(id.clone(), endpoint);
        info!("Registered in-process transport: {}", id);
    }

    /// Unregister a transport
    fn unregister(&self, id: &str) {
        self.transports.remove(id);
        info!("Unregistered in-process transport: {}", id);
    }

    /// Send a message to a specific transport
    async fn send(&self, from: &str, to: &str, msg: Message) -> Result<(), TransportError> {
        if let Some(endpoint) = self.transports.get(to) {
            endpoint
                .tx
                .send((from.to_string(), msg))
                .await
                .map_err(|_| {
                    TransportError::ConnectionError(format!("Transport {} is not receiving", to))
                })?;
            Ok(())
        } else {
            Err(TransportError::ConnectionError(format!(
                "Transport {} not found",
                to
            )))
        }
    }

    /// Broadcast a message to all transports
    async fn broadcast(&self, from: &str, msg: Message) -> Result<(), TransportError> {
        let mut errors = Vec::new();

        for entry in self.transports.iter() {
            let (id, endpoint) = entry.pair();

            // Don't send to self
            if id == from {
                continue;
            }

            // Clone message for each recipient
            let msg_clone = msg.clone();
            if endpoint.broadcast_tx.send(msg_clone).is_err() {
                errors.push(id.clone());
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(TransportError::Other(anyhow::anyhow!(
                "Failed to broadcast to: {}",
                errors.join(", ")
            )))
        }
    }

    /// Get list of registered transport IDs
    pub fn list_transports(&self) -> Vec<String> {
        self.transports
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

impl Default for InProcessRegistry {
    fn default() -> Self {
        Self {
            transports: DashMap::new(),
        }
    }
}

impl InProcessTransport {
    /// Create a new in-process transport
    pub async fn new(
        id: String,
        config: TransportConfig,
        registry: Arc<InProcessRegistry>,
    ) -> Result<Self, TransportError> {
        let (incoming_tx, incoming_rx) = mpsc::channel(config.max_message_size / 1024);
        let (broadcast_tx, _) = broadcast::channel(1024);

        // Register with the registry
        let endpoint = InProcessEndpoint {
            tx: incoming_tx.clone(),
            broadcast_tx: broadcast_tx.clone(),
        };
        registry.register(id.clone(), endpoint);

        Ok(Self {
            id,
            config,
            registry,
            incoming_rx,
            incoming_tx,
            broadcast_tx,
            is_running: Arc::new(AtomicBool::new(true)),
            stats: Arc::new(RwLock::new(TransportStats::default())),
        })
    }

    /// Create a pair of connected transports
    pub async fn create_pair(
        id1: String,
        id2: String,
        config: TransportConfig,
    ) -> Result<(Self, Self), TransportError> {
        let registry = InProcessRegistry::new();

        let transport1 = Self::new(id1, config.clone(), registry.clone()).await?;
        let transport2 = Self::new(id2, config, registry).await?;

        Ok((transport1, transport2))
    }

    /// Get the shared registry
    pub fn registry(&self) -> Arc<InProcessRegistry> {
        Arc::clone(&self.registry)
    }
}

#[async_trait]
impl Transport for InProcessTransport {
    type Message = Message;
    type Error = TransportError;

    async fn send(&self, to: &str, msg: Self::Message) -> Result<(), Self::Error> {
        // Check message size
        let size_estimate = bincode::serialized_size(&msg)
            .map_err(|e| TransportError::SerializationError(e.to_string()))?
            as usize;

        if size_estimate > self.config.max_message_size {
            return Err(TransportError::MessageTooLarge {
                size: size_estimate,
                max: self.config.max_message_size,
            });
        }

        // Send through registry
        self.registry.send(&self.id, to, msg).await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.messages_sent += 1;
        stats.bytes_sent += size_estimate as u64;
        stats.last_activity = Some(chrono::Utc::now());

        Ok(())
    }

    async fn receive(&mut self) -> Result<(String, Self::Message), Self::Error> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Err(TransportError::ConnectionError(
                "Transport is closed".to_string(),
            ));
        }

        let result = self
            .incoming_rx
            .recv()
            .await
            .ok_or_else(|| TransportError::ConnectionError("Channel closed".to_string()))?;

        // Update stats
        let msg_size = bincode::serialized_size(&result.1).unwrap_or(0) as u64;

        let mut stats = self.stats.write().await;
        stats.messages_received += 1;
        stats.bytes_received += msg_size;
        stats.last_activity = Some(chrono::Utc::now());

        Ok(result)
    }

    async fn broadcast(&self, msg: Self::Message) -> Result<(), Self::Error> {
        // Check message size
        let size_estimate = bincode::serialized_size(&msg)
            .map_err(|e| TransportError::SerializationError(e.to_string()))?
            as usize;

        if size_estimate > self.config.max_message_size {
            return Err(TransportError::MessageTooLarge {
                size: size_estimate,
                max: self.config.max_message_size,
            });
        }

        // Broadcast through registry
        self.registry.broadcast(&self.id, msg).await?;

        // Update stats
        let peer_count = self.registry.transports.len() - 1; // Exclude self
        let mut stats = self.stats.write().await;
        stats.messages_sent += peer_count as u64;
        stats.bytes_sent += size_estimate as u64 * peer_count as u64;
        stats.last_activity = Some(chrono::Utc::now());

        Ok(())
    }

    fn local_address(&self) -> Result<String, Self::Error> {
        Ok(format!("inproc://{}", self.id))
    }

    fn is_connected(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.is_running.store(false, Ordering::SeqCst);
        self.registry.unregister(&self.id);
        Ok(())
    }

    fn stats(&self) -> TransportStats {
        futures::executor::block_on(async { self.stats.read().await.clone() })
    }
}

/// Builder for in-process transports
pub struct InProcessTransportBuilder {
    registry: Arc<InProcessRegistry>,
}

impl InProcessTransportBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            registry: InProcessRegistry::new(),
        }
    }

    /// Use an existing registry
    pub fn with_registry(registry: Arc<InProcessRegistry>) -> Self {
        Self { registry }
    }

    /// Build a transport
    pub async fn build(
        &self,
        id: String,
        config: TransportConfig,
    ) -> Result<InProcessTransport, TransportError> {
        InProcessTransport::new(id, config, Arc::clone(&self.registry)).await
    }

    /// Get the registry
    pub fn registry(&self) -> Arc<InProcessRegistry> {
        Arc::clone(&self.registry)
    }
}

impl Default for InProcessTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Message, MessageType};

    #[tokio::test]
    async fn test_in_process_transport_pair() {
        let config = TransportConfig::default();
        let (transport1, mut transport2) =
            InProcessTransport::create_pair("agent1".to_string(), "agent2".to_string(), config)
                .await
                .unwrap();

        // Test send from transport1 to transport2
        let msg = Message::event(
            "agent1".to_string(),
            "test".to_string(),
            serde_json::json!({"data": "test"}),
        );
        transport1.send("agent2", msg.clone()).await.unwrap();

        // Receive on transport2
        let (from, received) = transport2.receive().await.unwrap();
        assert_eq!(from, "agent1");
        assert_eq!(received.id, msg.id);
    }

    #[tokio::test]
    async fn test_in_process_broadcast() {
        let builder = InProcessTransportBuilder::new();
        let config = TransportConfig::default();

        // Create multiple transports
        let transport1 = builder
            .build("agent1".to_string(), config.clone())
            .await
            .unwrap();
        let transport2 = builder
            .build("agent2".to_string(), config.clone())
            .await
            .unwrap();
        let transport3 = builder
            .build("agent3".to_string(), config.clone())
            .await
            .unwrap();

        // Subscribe to broadcasts
        let mut broadcast_rx2 = transport2.broadcast_tx.subscribe();
        let mut broadcast_rx3 = transport3.broadcast_tx.subscribe();

        // Broadcast from transport1
        let msg = Message::broadcast(
            "agent1".to_string(),
            "topic".to_string(),
            serde_json::json!({"broadcast": "data"}),
        );
        transport1.broadcast(msg.clone()).await.unwrap();

        // Check that both transport2 and transport3 received the broadcast
        let received2 = broadcast_rx2.recv().await.unwrap();
        let received3 = broadcast_rx3.recv().await.unwrap();

        assert_eq!(received2.id, msg.id);
        assert_eq!(received3.id, msg.id);
    }

    #[tokio::test]
    async fn test_message_size_limit() {
        let mut config = TransportConfig::default();
        config.max_message_size = 100; // Very small limit

        let (transport1, _) =
            InProcessTransport::create_pair("agent1".to_string(), "agent2".to_string(), config)
                .await
                .unwrap();

        // Create a large message
        let large_data = vec![0u8; 1000];
        let msg = Message::event(
            "agent1".to_string(),
            "large".to_string(),
            serde_json::to_value(large_data).unwrap(),
        );

        // Should fail due to size limit
        let result = transport1.send("agent2", msg).await;
        assert!(matches!(
            result,
            Err(TransportError::MessageTooLarge { .. })
        ));
    }

    #[tokio::test]
    async fn test_registry_list() {
        let registry = InProcessRegistry::new();
        let config = TransportConfig::default();

        // Create multiple transports
        let _t1 = InProcessTransport::new("agent1".to_string(), config.clone(), registry.clone())
            .await
            .unwrap();
        let _t2 = InProcessTransport::new("agent2".to_string(), config.clone(), registry.clone())
            .await
            .unwrap();
        let _t3 = InProcessTransport::new("agent3".to_string(), config, registry.clone())
            .await
            .unwrap();

        // List should contain all three
        let list = registry.list_transports();
        assert_eq!(list.len(), 3);
        assert!(list.contains(&"agent1".to_string()));
        assert!(list.contains(&"agent2".to_string()));
        assert!(list.contains(&"agent3".to_string()));
    }
}
