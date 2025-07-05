//! Intelligent swarm communication infrastructure
//!
//! This module provides the core communication layer for intelligent swarm agents,
//! enabling context-aware messaging, shared knowledge bases, and sophisticated
//! coordination patterns.

use crate::agent::{AgentId, AgentMessage, MessageType, MessageUrgency};
use crate::error::{Result, SwarmError};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::sync::Arc;

#[cfg(feature = "std")]
use dashmap::DashMap;

#[cfg(feature = "std")]
use tokio::sync::mpsc;

/// Trait for communication channels between agents
#[async_trait]
pub trait CommunicationChannel: Send + Sync {
    /// Send a message through the communication channel
    async fn send_message<T: Serialize + Send + Sync + 'static>(
        &self,
        message: AgentMessage<T>,
    ) -> Result<()>;
}

/// In-memory communication bus for agents within the same swarm instance
#[cfg(feature = "std")]
pub struct InProcessCommunicationBus {
    /// Agent-specific mailboxes for message delivery
    mailboxes: DashMap<AgentId, mpsc::Sender<AgentMessage<Value>>>,
}

#[cfg(feature = "std")]
impl InProcessCommunicationBus {
    /// Create a new in-process communication bus
    pub fn new() -> Self {
        InProcessCommunicationBus {
            mailboxes: DashMap::new(),
        }
    }

    /// Register an agent's mailbox for message delivery
    pub fn register_agent_mailbox(&self, agent_id: AgentId, sender: mpsc::Sender<AgentMessage<Value>>) {
        self.mailboxes.insert(agent_id, sender);
    }

    /// Unregister an agent's mailbox
    pub fn unregister_agent_mailbox(&self, agent_id: &AgentId) {
        self.mailboxes.remove(agent_id);
    }

    /// Get the number of registered mailboxes
    pub fn mailbox_count(&self) -> usize {
        self.mailboxes.len()
    }

    /// Check if an agent has a registered mailbox
    pub fn has_mailbox(&self, agent_id: &AgentId) -> bool {
        self.mailboxes.contains_key(agent_id)
    }
}

#[cfg(feature = "std")]
#[async_trait]
impl CommunicationChannel for InProcessCommunicationBus {
    async fn send_message<T: Serialize + Send + Sync + 'static>(
        &self,
        message: AgentMessage<T>,
    ) -> Result<()> {
        let target_id = message.to.clone();
        
        // Convert the payload to a JSON value for type erasure
        let payload_value = serde_json::to_value(message.payload)
            .map_err(|e| SwarmError::Custom(format!("Failed to serialize message payload: {}", e)))?;
        
        // Create the message with JSON payload
        let msg_for_send = AgentMessage {
            from: message.from,
            to: message.to,
            payload: payload_value,
            msg_type: message.msg_type,
            correlation_id: message.correlation_id,
            info_type: message.info_type,
            context: message.context,
            urgency: message.urgency,
        };

        // Send the message to the target agent's mailbox
        if let Some(sender) = self.mailboxes.get(&target_id) {
            sender.send(msg_for_send).await
                .map_err(|_| SwarmError::Custom(format!("Failed to send message to agent {}", target_id)))?;
            Ok(())
        } else {
            Err(SwarmError::AgentNotFound { id: target_id })
        }
    }
}

/// Knowledge base entry for shared information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    /// The knowledge value
    pub value: Value,
    /// Timestamp when the knowledge was created/updated
    pub timestamp: u64,
    /// Source agent that contributed this knowledge
    pub source: Option<AgentId>,
    /// Tags for categorization and search
    pub tags: Vec<String>,
    /// Expiration time (optional)
    pub expires_at: Option<u64>,
}

impl KnowledgeEntry {
    /// Create a new knowledge entry
    pub fn new(value: Value) -> Self {
        KnowledgeEntry {
            value,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            source: None,
            tags: Vec::new(),
            expires_at: None,
        }
    }

    /// Create a new knowledge entry with source
    pub fn with_source(value: Value, source: AgentId) -> Self {
        let mut entry = Self::new(value);
        entry.source = Some(source);
        entry
    }

    /// Add tags to the knowledge entry
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set expiration time
    pub fn with_expiration(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if the knowledge entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now > expires_at
        } else {
            false
        }
    }
}

/// Manages inter-agent communication and shared knowledge within the swarm
#[cfg(feature = "std")]
pub struct SwarmCommunicationManager {
    /// Message bus for inter-agent communication
    message_bus: InProcessCommunicationBus,
    /// Shared knowledge base for stigmergic communication
    shared_knowledge_base: DashMap<String, KnowledgeEntry>,
    /// Statistics for communication monitoring
    message_stats: DashMap<MessageType, u64>,
}

#[cfg(feature = "std")]
impl SwarmCommunicationManager {
    /// Create a new swarm communication manager
    pub fn new() -> Self {
        SwarmCommunicationManager {
            message_bus: InProcessCommunicationBus::new(),
            shared_knowledge_base: DashMap::new(),
            message_stats: DashMap::new(),
        }
    }

    /// Send a message through the communication bus
    pub async fn send_message<T: Serialize + Send + Sync + 'static>(
        &self,
        message: AgentMessage<T>,
    ) -> Result<()> {
        // Update message statistics
        let msg_type = message.msg_type;
        self.message_stats.entry(msg_type)
            .and_modify(|count| *count += 1)
            .or_insert(1);

        // Send the message
        self.message_bus.send_message(message).await
    }

    /// Update the shared knowledge base with new information
    pub fn update_knowledge(&self, key: String, value: Value) {
        let entry = KnowledgeEntry::new(value);
        self.shared_knowledge_base.insert(key, entry);
    }

    /// Update the shared knowledge base with detailed entry
    pub fn update_knowledge_entry(&self, key: String, entry: KnowledgeEntry) {
        self.shared_knowledge_base.insert(key, entry);
    }

    /// Get a specific piece of knowledge by key
    pub fn get_knowledge(&self, key: &str) -> Option<KnowledgeEntry> {
        self.shared_knowledge_base.get(key).map(|entry| {
            let entry = entry.value().clone();
            if entry.is_expired() {
                // Remove expired entries
                self.shared_knowledge_base.remove(key);
                None
            } else {
                Some(entry)
            }
        }).flatten()
    }

    /// Query the knowledge base with a search string
    pub fn query_knowledge(&self, query: &str) -> Vec<(String, KnowledgeEntry)> {
        let query_lower = query.to_lowercase();
        self.shared_knowledge_base
            .iter()
            .filter_map(|entry| {
                let key = entry.key();
                let knowledge_entry = entry.value();
                
                // Skip expired entries
                if knowledge_entry.is_expired() {
                    return None;
                }

                // Search in key, value string representation, and tags
                let matches = key.to_lowercase().contains(&query_lower)
                    || knowledge_entry.value.to_string().to_lowercase().contains(&query_lower)
                    || knowledge_entry.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower));

                if matches {
                    Some((key.clone(), knowledge_entry.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Query knowledge by tags
    pub fn query_knowledge_by_tags(&self, tags: &[String]) -> Vec<(String, KnowledgeEntry)> {
        self.shared_knowledge_base
            .iter()
            .filter_map(|entry| {
                let key = entry.key();
                let knowledge_entry = entry.value();
                
                // Skip expired entries
                if knowledge_entry.is_expired() {
                    return None;
                }

                // Check if any of the requested tags match
                let has_matching_tag = tags.iter().any(|tag| {
                    knowledge_entry.tags.iter().any(|entry_tag| {
                        entry_tag.to_lowercase() == tag.to_lowercase()
                    })
                });

                if has_matching_tag {
                    Some((key.clone(), knowledge_entry.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clean up expired knowledge entries
    pub fn cleanup_expired_knowledge(&self) {
        let expired_keys: Vec<String> = self.shared_knowledge_base
            .iter()
            .filter_map(|entry| {
                if entry.value().is_expired() {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();

        for key in expired_keys {
            self.shared_knowledge_base.remove(&key);
        }
    }

    /// Get the number of knowledge entries
    pub fn knowledge_count(&self) -> usize {
        self.shared_knowledge_base.len()
    }

    /// Register an agent's mailbox with the message bus
    pub fn register_agent_mailbox(&self, agent_id: AgentId, sender: mpsc::Sender<AgentMessage<Value>>) {
        self.message_bus.register_agent_mailbox(agent_id, sender);
    }

    /// Unregister an agent's mailbox from the message bus
    pub fn unregister_agent_mailbox(&self, agent_id: &AgentId) {
        self.message_bus.unregister_agent_mailbox(agent_id);
    }

    /// Get message statistics
    pub fn get_message_stats(&self) -> Vec<(MessageType, u64)> {
        self.message_stats
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }

    /// Get the number of registered agent mailboxes
    pub fn mailbox_count(&self) -> usize {
        self.message_bus.mailbox_count()
    }

    /// Check if an agent has a registered mailbox
    pub fn has_agent_mailbox(&self, agent_id: &AgentId) -> bool {
        self.message_bus.has_mailbox(agent_id)
    }
}

#[cfg(feature = "std")]
impl Default for SwarmCommunicationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Communication statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStats {
    /// Total messages sent
    pub total_messages: u64,
    /// Messages by type
    pub messages_by_type: Vec<(MessageType, u64)>,
    /// Number of active agent mailboxes
    pub active_mailboxes: usize,
    /// Number of knowledge entries
    pub knowledge_entries: usize,
}

#[cfg(feature = "std")]
impl SwarmCommunicationManager {
    /// Get comprehensive communication statistics
    pub fn get_stats(&self) -> CommunicationStats {
        let messages_by_type = self.get_message_stats();
        let total_messages = messages_by_type.iter().map(|(_, count)| count).sum();

        CommunicationStats {
            total_messages,
            messages_by_type,
            active_mailboxes: self.mailbox_count(),
            knowledge_entries: self.knowledge_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::MessageType;
    use tokio::sync::mpsc;

    #[test]
    fn test_knowledge_entry_creation() {
        let value = serde_json::json!({"test": "data"});
        let entry = KnowledgeEntry::new(value.clone());
        
        assert_eq!(entry.value, value);
        assert!(entry.source.is_none());
        assert!(entry.tags.is_empty());
        assert!(entry.expires_at.is_none());
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_knowledge_entry_with_source() {
        let value = serde_json::json!({"test": "data"});
        let agent_id = "test_agent".to_string();
        let entry = KnowledgeEntry::with_source(value.clone(), agent_id.clone());
        
        assert_eq!(entry.value, value);
        assert_eq!(entry.source, Some(agent_id));
    }

    #[test]
    fn test_knowledge_entry_expiration() {
        let value = serde_json::json!({"test": "data"});
        let past_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() - 1000; // 1000 seconds ago
        
        let entry = KnowledgeEntry::new(value).with_expiration(past_time);
        assert!(entry.is_expired());
    }

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn test_communication_manager_creation() {
        let manager = SwarmCommunicationManager::new();
        assert_eq!(manager.knowledge_count(), 0);
        assert_eq!(manager.mailbox_count(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_knowledge_base_operations() {
        let manager = SwarmCommunicationManager::new();
        let key = "test_key".to_string();
        let value = serde_json::json!({"data": "test_value"});
        
        // Test knowledge update
        manager.update_knowledge(key.clone(), value.clone());
        assert_eq!(manager.knowledge_count(), 1);
        
        // Test knowledge retrieval
        let retrieved = manager.get_knowledge(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, value);
        
        // Test knowledge query
        let results = manager.query_knowledge("test_value");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, key);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_knowledge_query_by_tags() {
        let manager = SwarmCommunicationManager::new();
        
        let entry1 = KnowledgeEntry::new(serde_json::json!({"data": "value1"}))
            .with_tags(vec!["tag1".to_string(), "common".to_string()]);
        let entry2 = KnowledgeEntry::new(serde_json::json!({"data": "value2"}))
            .with_tags(vec!["tag2".to_string(), "common".to_string()]);
        
        manager.update_knowledge_entry("key1".to_string(), entry1);
        manager.update_knowledge_entry("key2".to_string(), entry2);
        
        // Query by specific tag
        let results = manager.query_knowledge_by_tags(&["tag1".to_string()]);
        assert_eq!(results.len(), 1);
        
        // Query by common tag
        let results = manager.query_knowledge_by_tags(&["common".to_string()]);
        assert_eq!(results.len(), 2);
    }

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn test_message_routing() {
        let manager = SwarmCommunicationManager::new();
        let (tx, mut rx) = mpsc::channel(10);
        
        // Register agent mailbox
        let agent_id = "test_agent".to_string();
        manager.register_agent_mailbox(agent_id.clone(), tx);
        
        // Test message sending
        let message = AgentMessage {
            from: "sender".to_string(),
            to: agent_id.clone(),
            payload: serde_json::json!({"test": "message"}),
            msg_type: MessageType::InformationShare,
            correlation_id: Some("test_corr_id".to_string()),
            info_type: Some("test_info".to_string()),
            context: Some(serde_json::json!({"context": "test"})),
            urgency: Some(crate::agent::MessageUrgency::Medium),
        };
        
        // Send message
        let result = manager.send_message(message.clone()).await;
        assert!(result.is_ok());
        
        // Receive message
        let received = rx.recv().await;
        assert!(received.is_some());
        let received_msg = received.unwrap();
        assert_eq!(received_msg.from, message.from);
        assert_eq!(received_msg.to, message.to);
        assert_eq!(received_msg.msg_type, message.msg_type);
    }

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn test_message_routing_to_nonexistent_agent() {
        let manager = SwarmCommunicationManager::new();
        
        let message = AgentMessage {
            from: "sender".to_string(),
            to: "nonexistent_agent".to_string(),
            payload: serde_json::json!({"test": "message"}),
            msg_type: MessageType::InformationShare,
            correlation_id: None,
            info_type: None,
            context: None,
            urgency: None,
        };
        
        // Should fail to send to nonexistent agent
        let result = manager.send_message(message).await;
        assert!(result.is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_knowledge_cleanup() {
        let manager = SwarmCommunicationManager::new();
        
        // Add expired knowledge
        let expired_entry = KnowledgeEntry::new(serde_json::json!({"data": "expired"}))
            .with_expiration(0); // Already expired
        manager.update_knowledge_entry("expired_key".to_string(), expired_entry);
        
        // Add valid knowledge
        let valid_entry = KnowledgeEntry::new(serde_json::json!({"data": "valid"}));
        manager.update_knowledge_entry("valid_key".to_string(), valid_entry);
        
        assert_eq!(manager.knowledge_count(), 2);
        
        // Cleanup expired entries
        manager.cleanup_expired_knowledge();
        
        // Should only have valid entry left
        assert_eq!(manager.knowledge_count(), 1);
        assert!(manager.get_knowledge("expired_key").is_none());
        assert!(manager.get_knowledge("valid_key").is_some());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_communication_stats() {
        let manager = SwarmCommunicationManager::new();
        let (tx1, _rx1) = mpsc::channel(10);
        let (tx2, _rx2) = mpsc::channel(10);
        
        // Register agents
        manager.register_agent_mailbox("agent1".to_string(), tx1);
        manager.register_agent_mailbox("agent2".to_string(), tx2);
        
        // Add knowledge entries
        manager.update_knowledge("key1".to_string(), serde_json::json!({"data": "value1"}));
        manager.update_knowledge("key2".to_string(), serde_json::json!({"data": "value2"}));
        
        let stats = manager.get_stats();
        assert_eq!(stats.active_mailboxes, 2);
        assert_eq!(stats.knowledge_entries, 2);
        assert_eq!(stats.total_messages, 0); // No messages sent yet
    }

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn test_complex_knowledge_operations() {
        let manager = SwarmCommunicationManager::new();
        
        // Create knowledge with different tags
        let research_entry = KnowledgeEntry::with_source(
            serde_json::json!({"topic": "neural_networks", "confidence": 0.9}),
            "researcher_agent".to_string()
        ).with_tags(vec!["research".to_string(), "ai".to_string(), "neural".to_string()]);
        
        let implementation_entry = KnowledgeEntry::with_source(
            serde_json::json!({"code": "class NeuralNet:", "language": "python"}),
            "coder_agent".to_string()
        ).with_tags(vec!["implementation".to_string(), "code".to_string(), "neural".to_string()]);
        
        manager.update_knowledge_entry("research:neural_networks".to_string(), research_entry);
        manager.update_knowledge_entry("code:neural_net_impl".to_string(), implementation_entry);
        
        // Test various query patterns
        let neural_results = manager.query_knowledge("neural");
        assert_eq!(neural_results.len(), 2);
        
        let research_results = manager.query_knowledge_by_tags(&["research".to_string()]);
        assert_eq!(research_results.len(), 1);
        
        let code_results = manager.query_knowledge_by_tags(&["code".to_string()]);
        assert_eq!(code_results.len(), 1);
        
        let ai_results = manager.query_knowledge_by_tags(&["ai".to_string()]);
        assert_eq!(ai_results.len(), 1);
        
        // Test complex queries
        let all_neural = manager.query_knowledge_by_tags(&["neural".to_string()]);
        assert_eq!(all_neural.len(), 2);
    }

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn test_enhanced_message_features() {
        use crate::agent::{MessageUrgency, MessageType};
        
        let manager = SwarmCommunicationManager::new();
        let (tx, mut rx) = mpsc::channel(10);
        
        manager.register_agent_mailbox("agent1".to_string(), tx);
        
        // Test message with all enhanced features
        let enhanced_message = AgentMessage {
            from: "smart_agent".to_string(),
            to: "agent1".to_string(),
            payload: serde_json::json!({
                "research_topic": "swarm_intelligence",
                "findings": ["emergent_behavior", "collective_intelligence"],
                "confidence": 0.85
            }),
            msg_type: MessageType::InformationShare,
            correlation_id: Some("research_session_123".to_string()),
            info_type: Some("research_findings".to_string()),
            context: Some(serde_json::json!({
                "project": "intelligent_swarms",
                "phase": "discovery",
                "deadline": "2024-02-01",
                "collaboration_needed": true
            })),
            urgency: Some(MessageUrgency::High),
        };
        
        // Send enhanced message
        let result = manager.send_message(enhanced_message).await;
        assert!(result.is_ok());
        
        // Receive and verify enhanced message
        let received = rx.recv().await;
        assert!(received.is_some());
        let received_msg = received.unwrap();
        
        assert_eq!(received_msg.msg_type, MessageType::InformationShare);
        assert_eq!(received_msg.correlation_id, Some("research_session_123".to_string()));
        assert_eq!(received_msg.info_type, Some("research_findings".to_string()));
        assert!(received_msg.context.is_some());
        assert_eq!(received_msg.urgency, Some(MessageUrgency::High));
        
        // Verify payload content
        assert!(received_msg.payload["research_topic"].as_str().unwrap() == "swarm_intelligence");
        assert!(received_msg.payload["confidence"].as_f64().unwrap() == 0.85);
    }
}