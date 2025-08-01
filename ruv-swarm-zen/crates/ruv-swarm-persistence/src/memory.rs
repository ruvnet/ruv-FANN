//! In-memory storage implementation for testing and development

use crate::{models::*, Storage, StorageError, Transaction as TransactionTrait};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// In-memory storage implementation
pub struct MemoryStorage {
    agents: Arc<RwLock<HashMap<String, AgentModel>>>,
    tasks: Arc<RwLock<HashMap<String, TaskModel>>>,
    events: Arc<RwLock<Vec<EventModel>>>,
    messages: Arc<RwLock<Vec<MessageModel>>>,
    metrics: Arc<RwLock<Vec<MetricModel>>>,
    next_sequence: Arc<RwLock<u64>>,
}

impl MemoryStorage {
    /// Create new in-memory storage instance
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            events: Arc::new(RwLock::new(Vec::new())),
            messages: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(Vec::new())),
            next_sequence: Arc::new(RwLock::new(1)),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Storage for MemoryStorage {
    type Error = StorageError;

    // Agent operations
    async fn store_agent(&self, agent: &AgentModel) -> Result<(), Self::Error> {
        let mut agents = self.agents.write();
        agents.insert(agent.id.clone(), agent.clone());
        debug!("Stored agent in memory: {}", agent.id);
        Ok(())
    }

    async fn get_agent(&self, id: &str) -> Result<Option<AgentModel>, Self::Error> {
        let agents = self.agents.read();
        Ok(agents.get(id).cloned())
    }

    async fn update_agent(&self, agent: &AgentModel) -> Result<(), Self::Error> {
        let mut agents = self.agents.write();
        if !agents.contains_key(&agent.id) {
            return Err(StorageError::NotFound(format!(
                "Agent {} not found",
                agent.id
            )));
        }
        agents.insert(agent.id.clone(), agent.clone());
        debug!("Updated agent in memory: {}", agent.id);
        Ok(())
    }

    async fn delete_agent(&self, id: &str) -> Result<(), Self::Error> {
        let mut agents = self.agents.write();
        if agents.remove(id).is_some() {
            debug!("Deleted agent from memory: {}", id);
        } else {
            debug!("Agent {} not found, delete is idempotent", id);
        }
        Ok(())
    }

    async fn list_agents(&self) -> Result<Vec<AgentModel>, Self::Error> {
        let agents = self.agents.read();
        let mut list: Vec<_> = agents.values().cloned().collect();
        list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(list)
    }

    async fn list_agents_by_status(&self, status: &str) -> Result<Vec<AgentModel>, Self::Error> {
        let agents = self.agents.read();
        let mut list: Vec<_> = agents
            .values()
            .filter(|a| a.status.to_string() == status)
            .cloned()
            .collect();
        list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(list)
    }

    // Task operations
    async fn store_task(&self, task: &TaskModel) -> Result<(), Self::Error> {
        let mut tasks = self.tasks.write();
        tasks.insert(task.id.clone(), task.clone());
        debug!("Stored task in memory: {}", task.id);
        Ok(())
    }

    async fn get_task(&self, id: &str) -> Result<Option<TaskModel>, Self::Error> {
        let tasks = self.tasks.read();
        Ok(tasks.get(id).cloned())
    }

    async fn update_task(&self, task: &TaskModel) -> Result<(), Self::Error> {
        let mut tasks = self.tasks.write();
        if !tasks.contains_key(&task.id) {
            return Err(StorageError::NotFound(format!(
                "Task {} not found",
                task.id
            )));
        }
        tasks.insert(task.id.clone(), task.clone());
        debug!("Updated task in memory: {}", task.id);
        Ok(())
    }

    async fn get_pending_tasks(&self) -> Result<Vec<TaskModel>, Self::Error> {
        let tasks = self.tasks.read();
        let mut pending: Vec<_> = tasks
            .values()
            .filter(|t| t.status == TaskStatus::Pending)
            .cloned()
            .collect();
        pending.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });
        Ok(pending)
    }

    async fn get_tasks_by_agent(&self, agent_id: &str) -> Result<Vec<TaskModel>, Self::Error> {
        let tasks = self.tasks.read();
        let mut agent_tasks: Vec<_> = tasks
            .values()
            .filter(|t| t.assigned_to.as_deref() == Some(agent_id))
            .cloned()
            .collect();
        agent_tasks.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });
        Ok(agent_tasks)
    }

    async fn claim_task(&self, task_id: &str, agent_id: &str) -> Result<bool, Self::Error> {
        let mut tasks = self.tasks.write();
        if let Some(task) = tasks.get_mut(task_id) {
            if task.status == TaskStatus::Pending {
                task.assign_to(agent_id);
                return Ok(true);
            }
        }
        Ok(false)
    }

    // Event operations
    async fn store_event(&self, event: &EventModel) -> Result<(), Self::Error> {
        let mut events = self.events.write();
        let mut sequence = self.next_sequence.write();

        let mut event_with_seq = event.clone();
        event_with_seq.sequence = *sequence;
        *sequence += 1;

        events.push(event_with_seq);
        debug!("Stored event in memory: {}", event.id);
        Ok(())
    }

    async fn get_events_by_agent(
        &self,
        agent_id: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error> {
        let events = self.events.read();
        let mut agent_events: Vec<_> = events
            .iter()
            .filter(|e| e.agent_id.as_deref() == Some(agent_id))
            .cloned()
            .collect();
        agent_events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        agent_events.truncate(limit);
        Ok(agent_events)
    }

    async fn get_events_by_type(
        &self,
        event_type: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error> {
        let events = self.events.read();
        let mut type_events: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect();
        type_events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        type_events.truncate(limit);
        Ok(type_events)
    }

    async fn get_events_since(&self, timestamp: i64) -> Result<Vec<EventModel>, Self::Error> {
        let events = self.events.read();
        let mut since_events: Vec<_> = events
            .iter()
            .filter(|e| e.timestamp.timestamp() >= timestamp)
            .cloned()
            .collect();
        
        // Sort by timestamp (with full precision) then by id for deterministic ordering
        since_events.sort_by(|a, b| {
            a.timestamp.cmp(&b.timestamp)
                .then_with(|| a.id.cmp(&b.id))
        });
        
        Ok(since_events)
    }

    // Message operations
    async fn store_message(&self, message: &MessageModel) -> Result<(), Self::Error> {
        let mut messages = self.messages.write();
        messages.push(message.clone());
        debug!("Stored message in memory: {}", message.id);
        Ok(())
    }

    async fn get_messages_between(
        &self,
        agent1: &str,
        agent2: &str,
        limit: usize,
    ) -> Result<Vec<MessageModel>, Self::Error> {
        let messages = self.messages.read();
        let mut between: Vec<_> = messages
            .iter()
            .filter(|m| {
                (m.from_agent == agent1 && m.to_agent == agent2)
                    || (m.from_agent == agent2 && m.to_agent == agent1)
            })
            .cloned()
            .collect();
        between.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        between.truncate(limit);
        Ok(between)
    }

    async fn get_unread_messages(&self, agent_id: &str) -> Result<Vec<MessageModel>, Self::Error> {
        let messages = self.messages.read();
        let unread: Vec<_> = messages
            .iter()
            .filter(|m| m.to_agent == agent_id && !m.read)
            .cloned()
            .collect();
        Ok(unread)
    }

    async fn mark_message_read(&self, message_id: &str) -> Result<(), Self::Error> {
        let mut messages = self.messages.write();
        if let Some(msg) = messages.iter_mut().find(|m| m.id == message_id) {
            msg.mark_read();
            Ok(())
        } else {
            Err(StorageError::NotFound(format!(
                "Message {} not found",
                message_id
            )))
        }
    }

    // Metric operations
    async fn store_metric(&self, metric: &MetricModel) -> Result<(), Self::Error> {
        let mut metrics = self.metrics.write();
        metrics.push(metric.clone());
        debug!("Stored metric in memory: {}", metric.id);
        Ok(())
    }

    async fn get_metrics_by_agent(
        &self,
        agent_id: &str,
        metric_type: &str,
    ) -> Result<Vec<MetricModel>, Self::Error> {
        let metrics = self.metrics.read();
        let agent_metrics: Vec<_> = metrics
            .iter()
            .filter(|m| m.agent_id.as_deref() == Some(agent_id) && m.metric_type == metric_type)
            .cloned()
            .collect();
        Ok(agent_metrics)
    }

    async fn get_aggregated_metrics(
        &self,
        metric_type: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<MetricModel>, Self::Error> {
        let metrics = self.metrics.read();

        // Group metrics by agent_id and unit
        let mut grouped: HashMap<(Option<String>, String), Vec<f64>> = HashMap::new();

        for metric in metrics.iter() {
            let timestamp = metric.timestamp.timestamp();
            if metric.metric_type == metric_type && timestamp >= start_time && timestamp <= end_time
            {
                let key = (metric.agent_id.clone(), metric.unit.clone());
                grouped
                    .entry(key)
                    .or_default()
                    .push(metric.value);
            }
        }

        // Create aggregated metrics
        let mut results = Vec::new();
        for ((agent_id, unit), values) in grouped {
            if !values.is_empty() {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                let mut metric = MetricModel::new(metric_type.to_string(), avg, unit);
                metric.agent_id = agent_id;
                metric
                    .tags
                    .insert("count".to_string(), values.len().to_string());
                results.push(metric);
            }
        }

        Ok(results)
    }

    // Transaction support
    async fn begin_transaction(&self) -> Result<Box<dyn TransactionTrait>, Self::Error> {
        Ok(Box::new(MemoryTransaction::new()))
    }

    // Maintenance operations
    async fn vacuum(&self) -> Result<(), Self::Error> {
        // No-op for in-memory storage
        debug!("Vacuum called on memory storage (no-op)");
        Ok(())
    }

    async fn checkpoint(&self) -> Result<(), Self::Error> {
        // No-op for in-memory storage
        debug!("Checkpoint called on memory storage (no-op)");
        Ok(())
    }

    async fn get_storage_size(&self) -> Result<u64, Self::Error> {
        // Estimate size based on number of items
        let agents_count = self.agents.read().len();
        let tasks_count = self.tasks.read().len();
        let events_count = self.events.read().len();
        let messages_count = self.messages.read().len();
        let metrics_count = self.metrics.read().len();

        // Rough estimate: 1KB per item
        let total_items =
            agents_count + tasks_count + events_count + messages_count + metrics_count;
        Ok((total_items * 1024) as u64)
    }
}

/// Memory transaction (no-op implementation)
struct MemoryTransaction;

impl MemoryTransaction {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TransactionTrait for MemoryTransaction {
    async fn commit(self: Box<Self>) -> Result<(), StorageError> {
        // No-op for in-memory storage
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> Result<(), StorageError> {
        // No-op for in-memory storage
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = MemoryStorage::new();

        // Test agent operations
        let agent = AgentModel::new(
            "test-agent".to_string(),
            "worker".to_string(),
            vec!["compute".to_string()],
        );

        storage.store_agent(&agent).await.unwrap();
        let retrieved = storage.get_agent(&agent.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test-agent");

        // Test task operations
        let task = TaskModel::new(
            "process".to_string(),
            serde_json::json!({"data": "test"}),
            TaskPriority::High,
        );

        storage.store_task(&task).await.unwrap();
        let pending = storage.get_pending_tasks().await.unwrap();
        assert_eq!(pending.len(), 1);

        // Test claim task
        let claimed = storage.claim_task(&task.id, &agent.id).await.unwrap();
        assert!(claimed);

        let updated_task = storage.get_task(&task.id).await.unwrap().unwrap();
        assert_eq!(updated_task.assigned_to, Some(agent.id.clone()));
    }
}
