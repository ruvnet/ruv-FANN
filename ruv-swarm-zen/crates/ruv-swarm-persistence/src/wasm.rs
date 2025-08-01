//! WASM storage implementation using IndexedDB

use crate::{models::*, Storage, StorageError, Transaction as TransactionTrait};
use async_trait::async_trait;
use rexie::{Database, ObjectStore, TransactionMode};
use serde_json;
use std::sync::Arc;
use wasm_bindgen::JsValue;
use web_sys::console;

/// WASM storage implementation using IndexedDB
pub struct WasmStorage {
    db: Arc<Database>,
}

impl WasmStorage {
    /// Create new WASM storage instance
    pub async fn new() -> Result<Self, StorageError> {
        console::log_1(&"Initializing WASM storage with IndexedDB".into());

        // Create database
        let db = Database::create("ruv_swarm", 1, |builder| {
            // Create object stores
            builder.create_object_store("agents", |store| {
                store.key_path("id");
                store.add_index("status", "status", false);
                store.add_index("agent_type", "agent_type", false);
            });

            builder.create_object_store("tasks", |store| {
                store.key_path("id");
                store.add_index("status", "status", false);
                store.add_index("priority", "priority", false);
                store.add_index("assigned_to", "assigned_to", false);
            });

            builder.create_object_store("events", |store| {
                store.key_path("id");
                store.add_index("event_type", "event_type", false);
                store.add_index("agent_id", "agent_id", false);
                store.add_index("timestamp", "timestamp", false);
            });

            builder.create_object_store("messages", |store| {
                store.key_path("id");
                store.add_index("from_agent", "from_agent", false);
                store.add_index("to_agent", "to_agent", false);
                store.add_index("read", "read", false);
            });

            builder.create_object_store("metrics", |store| {
                store.key_path("id");
                store.add_index("metric_type", "metric_type", false);
                store.add_index("agent_id", "agent_id", false);
                store.add_index("timestamp", "timestamp", false);
            });
        })
        .await
        .map_err(|e| StorageError::Database(format!("Failed to create database: {:?}", e)))?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Convert model to JsValue
    fn to_js_value<T: serde::Serialize>(value: &T) -> Result<JsValue, StorageError> {
        let json = serde_json::to_string(value)?;
        Ok(JsValue::from_str(&json))
    }

    /// Convert JsValue to model
    fn from_js_value<T: serde::de::DeserializeOwned>(value: JsValue) -> Result<T, StorageError> {
        let json = value.as_string().ok_or_else(|| {
            StorageError::Serialization(serde_json::Error::custom("Invalid JS value"))
        })?;
        Ok(serde_json::from_str(&json)?)
    }
}

#[async_trait]
impl Storage for WasmStorage {
    type Error = StorageError;

    // Agent operations
    async fn store_agent(&self, agent: &AgentModel) -> Result<(), Self::Error> {
        let tx = self
            .db
            .transaction(&["agents"], TransactionMode::ReadWrite)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("agents")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let value = Self::to_js_value(agent)?;
        store
            .put(&value)
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        tx.commit()
            .await
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        Ok(())
    }

    async fn get_agent(&self, id: &str) -> Result<Option<AgentModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["agents"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("agents")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        match store.get(&JsValue::from_str(id)).await {
            Ok(value) => Ok(Some(Self::from_js_value(value)?)),
            Err(_) => Ok(None),
        }
    }

    async fn update_agent(&self, agent: &AgentModel) -> Result<(), Self::Error> {
        // In IndexedDB, put operation updates if key exists
        self.store_agent(agent).await
    }

    async fn delete_agent(&self, id: &str) -> Result<(), Self::Error> {
        let tx = self
            .db
            .transaction(&["agents"], TransactionMode::ReadWrite)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("agents")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        store
            .delete(&JsValue::from_str(id))
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        tx.commit()
            .await
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        Ok(())
    }

    async fn list_agents(&self) -> Result<Vec<AgentModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["agents"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("agents")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = store
            .get_all()
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        values.into_iter().map(Self::from_js_value).collect()
    }

    async fn list_agents_by_status(&self, status: &str) -> Result<Vec<AgentModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["agents"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("agents")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let index = store
            .index("status")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = index
            .get_all(Some(&JsValue::from_str(status)))
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        values.into_iter().map(Self::from_js_value).collect()
    }

    // Task operations
    async fn store_task(&self, task: &TaskModel) -> Result<(), Self::Error> {
        let tx = self
            .db
            .transaction(&["tasks"], TransactionMode::ReadWrite)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("tasks")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let value = Self::to_js_value(task)?;
        store
            .put(&value)
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        tx.commit()
            .await
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        Ok(())
    }

    async fn get_task(&self, id: &str) -> Result<Option<TaskModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["tasks"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("tasks")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        match store.get(&JsValue::from_str(id)).await {
            Ok(value) => Ok(Some(Self::from_js_value(value)?)),
            Err(_) => Ok(None),
        }
    }

    async fn update_task(&self, task: &TaskModel) -> Result<(), Self::Error> {
        self.store_task(task).await
    }

    async fn get_pending_tasks(&self) -> Result<Vec<TaskModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["tasks"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("tasks")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let index = store
            .index("status")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = index
            .get_all(Some(&JsValue::from_str("pending")))
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let mut tasks: Vec<TaskModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .collect();

        // Sort by priority and creation time
        tasks.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });

        Ok(tasks)
    }

    async fn get_tasks_by_agent(&self, agent_id: &str) -> Result<Vec<TaskModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["tasks"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("tasks")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let index = store
            .index("assigned_to")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = index
            .get_all(Some(&JsValue::from_str(agent_id)))
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        values.into_iter().map(Self::from_js_value).collect()
    }

    async fn claim_task(&self, task_id: &str, agent_id: &str) -> Result<bool, Self::Error> {
        let mut task = match self.get_task(task_id).await? {
            Some(t) => t,
            None => return Ok(false),
        };

        if task.status == TaskStatus::Pending {
            task.assign_to(agent_id);
            self.update_task(&task).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // Event operations
    async fn store_event(&self, event: &EventModel) -> Result<(), Self::Error> {
        let tx = self
            .db
            .transaction(&["events"], TransactionMode::ReadWrite)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("events")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let value = Self::to_js_value(event)?;
        store
            .put(&value)
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        tx.commit()
            .await
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        Ok(())
    }

    async fn get_events_by_agent(
        &self,
        agent_id: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["events"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("events")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let index = store
            .index("agent_id")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = index
            .get_all(Some(&JsValue::from_str(agent_id)))
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let mut events: Vec<EventModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .collect();

        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);

        Ok(events)
    }

    async fn get_events_by_type(
        &self,
        event_type: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["events"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("events")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let index = store
            .index("event_type")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = index
            .get_all(Some(&JsValue::from_str(event_type)))
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let mut events: Vec<EventModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .collect();

        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);

        Ok(events)
    }

    async fn get_events_since(&self, timestamp: i64) -> Result<Vec<EventModel>, Self::Error> {
        // IndexedDB doesn't support range queries easily, so we get all and filter
        let tx = self
            .db
            .transaction(&["events"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("events")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = store
            .get_all()
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let events: Vec<EventModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .filter(|e| e.timestamp.timestamp() > timestamp)
            .collect();

        Ok(events)
    }

    // Message operations
    async fn store_message(&self, message: &MessageModel) -> Result<(), Self::Error> {
        let tx = self
            .db
            .transaction(&["messages"], TransactionMode::ReadWrite)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("messages")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let value = Self::to_js_value(message)?;
        store
            .put(&value)
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        tx.commit()
            .await
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        Ok(())
    }

    async fn get_messages_between(
        &self,
        agent1: &str,
        agent2: &str,
        limit: usize,
    ) -> Result<Vec<MessageModel>, Self::Error> {
        // We need to get all messages and filter manually
        let tx = self
            .db
            .transaction(&["messages"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("messages")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = store
            .get_all()
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let mut messages: Vec<MessageModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .filter(|m| {
                (m.from_agent == agent1 && m.to_agent == agent2)
                    || (m.from_agent == agent2 && m.to_agent == agent1)
            })
            .collect();

        messages.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        messages.truncate(limit);

        Ok(messages)
    }

    async fn get_unread_messages(&self, agent_id: &str) -> Result<Vec<MessageModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["messages"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("messages")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let index = store
            .index("to_agent")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = index
            .get_all(Some(&JsValue::from_str(agent_id)))
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let messages: Vec<MessageModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .filter(|m| !m.read)
            .collect();

        Ok(messages)
    }

    async fn mark_message_read(&self, message_id: &str) -> Result<(), Self::Error> {
        let mut message = match self.get_message(message_id).await? {
            Some(m) => m,
            None => {
                return Err(StorageError::NotFound(format!(
                    "Message {} not found",
                    message_id
                )))
            }
        };

        message.mark_read();
        self.store_message(&message).await
    }

    // Metric operations
    async fn store_metric(&self, metric: &MetricModel) -> Result<(), Self::Error> {
        let tx = self
            .db
            .transaction(&["metrics"], TransactionMode::ReadWrite)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("metrics")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let value = Self::to_js_value(metric)?;
        store
            .put(&value)
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        tx.commit()
            .await
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        Ok(())
    }

    async fn get_metrics_by_agent(
        &self,
        agent_id: &str,
        metric_type: &str,
    ) -> Result<Vec<MetricModel>, Self::Error> {
        let tx = self
            .db
            .transaction(&["metrics"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("metrics")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = store
            .get_all()
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let metrics: Vec<MetricModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .filter(|m| m.agent_id.as_deref() == Some(agent_id) && m.metric_type == metric_type)
            .collect();

        Ok(metrics)
    }

    async fn get_aggregated_metrics(
        &self,
        metric_type: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<MetricModel>, Self::Error> {
        // For WASM, we'll do simple client-side aggregation
        let tx = self
            .db
            .transaction(&["metrics"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("metrics")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let values = store
            .get_all()
            .await
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        let filtered_metrics: Vec<MetricModel> = values
            .into_iter()
            .filter_map(|v| Self::from_js_value(v).ok())
            .filter(|m| {
                let ts = m.timestamp.timestamp();
                m.metric_type == metric_type && ts >= start_time && ts <= end_time
            })
            .collect();

        // Simple aggregation by agent_id
        use std::collections::HashMap;
        let mut grouped: HashMap<(Option<String>, String), Vec<f64>> = HashMap::new();

        for metric in filtered_metrics {
            let key = (metric.agent_id.clone(), metric.unit.clone());
            grouped
                .entry(key)
                .or_insert_with(Vec::new)
                .push(metric.value);
        }

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
        Ok(Box::new(WasmTransaction))
    }

    // Maintenance operations
    async fn vacuum(&self) -> Result<(), Self::Error> {
        // No-op for IndexedDB
        console::log_1(&"Vacuum called on WASM storage (no-op)".into());
        Ok(())
    }

    async fn checkpoint(&self) -> Result<(), Self::Error> {
        // No-op for IndexedDB
        console::log_1(&"Checkpoint called on WASM storage (no-op)".into());
        Ok(())
    }

    async fn get_storage_size(&self) -> Result<u64, Self::Error> {
        // Browser storage estimation API could be used here
        // For now, return a placeholder
        Ok(0)
    }
}

impl WasmStorage {
    /// Helper to get a message by ID
    async fn get_message(&self, id: &str) -> Result<Option<MessageModel>, StorageError> {
        let tx = self
            .db
            .transaction(&["messages"], TransactionMode::ReadOnly)
            .map_err(|e| StorageError::Transaction(format!("{:?}", e)))?;

        let store = tx
            .object_store("messages")
            .map_err(|e| StorageError::Database(format!("{:?}", e)))?;

        match store.get(&JsValue::from_str(id)).await {
            Ok(value) => Ok(Some(Self::from_js_value(value)?)),
            Err(_) => Ok(None),
        }
    }
}

/// WASM transaction (no-op implementation)
struct WasmTransaction;

#[async_trait]
impl TransactionTrait for WasmTransaction {
    async fn commit(self: Box<Self>) -> Result<(), StorageError> {
        // IndexedDB transactions auto-commit
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> Result<(), StorageError> {
        // IndexedDB transactions auto-rollback on error
        Ok(())
    }
}
