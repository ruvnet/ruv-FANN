//! Browser-local IndexedDB persistence with awaited commits and atomic writes.
use crate::{models::*, Storage, StorageError, Transaction as TransactionTrait};
use async_trait::async_trait;
use rexie::{Index, KeyRange, ObjectStore, Rexie, Transaction, TransactionMode};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use wasm_bindgen::JsValue;

const STORES: [&str; 6] = [
    "agents", "tasks", "events", "messages", "metrics", "metadata",
];
fn database_error(error: impl std::fmt::Debug) -> StorageError {
    StorageError::Database(format!("{error:?}"))
}

#[path = "wasm_batch.rs"]
mod batch;
pub use batch::WasmMutation;

/// IndexedDB handles and futures stay on their JavaScript thread.
pub struct WasmStorage {
    db: Option<Rexie>,
}
impl Drop for WasmStorage {
    fn drop(&mut self) {
        if let Some(db) = self.db.take() {
            db.close();
        }
    }
}
impl WasmStorage {
    pub async fn new() -> Result<Self, StorageError> {
        Self::open("ruv_swarm").await
    }
    /// Open a named database; version 2 adds the transactional event counter.
    pub async fn open(name: &str) -> Result<Self, StorageError> {
        if name.is_empty() {
            return Err(StorageError::Database(
                "Database name must not be empty".into(),
            ));
        }
        let schema = [
            ("agents", &["status", "agent_type"][..]),
            ("tasks", &["status", "assigned_to"][..]),
            ("events", &["event_type", "agent_id"][..]),
            ("messages", &["from_agent", "to_agent"][..]),
            ("metrics", &["metric_type", "agent_id"][..]),
            ("metadata", &[][..]),
        ];
        let mut builder = Rexie::builder(name).version(2);
        for (name, indexes) in schema {
            let mut store = ObjectStore::new(name).key_path("id");
            for index in indexes {
                store = store.add_index(Index::new(index, index));
            }
            builder = builder.add_object_store(store);
        }
        Ok(Self {
            db: Some(builder.build().await.map_err(database_error)?),
        })
    }
    fn database(&self) -> &Rexie {
        self.db
            .as_ref()
            .expect("open database remains owned until drop")
    }
    fn transaction(
        &self,
        stores: &[&str],
        mode: TransactionMode,
    ) -> Result<Transaction, StorageError> {
        self.database()
            .transaction(stores, mode)
            .map_err(database_error)
    }
    // Keep exact JSON in a string envelope: JS numbers would round large u64
    // fields. IndexedDB still receives an object with a real inline id and indexes.
    fn encode<T: Serialize>(model: &T) -> Result<JsValue, StorageError> {
        let value = serde_json::to_value(model)?;
        let id = value
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| StorageError::Database("Record requires a string id".into()))?;
        let object = js_sys::Object::new();
        js_sys::Reflect::set(&object, &"id".into(), &id.into()).map_err(database_error)?;
        js_sys::Reflect::set(&object, &"record".into(), &value.to_string().into())
            .map_err(database_error)?;
        for field in [
            "status",
            "agent_type",
            "assigned_to",
            "agent_id",
            "event_type",
            "from_agent",
            "to_agent",
            "metric_type",
        ] {
            if let Some(text) = value.get(field).and_then(|v| v.as_str()) {
                js_sys::Reflect::set(&object, &field.into(), &text.into())
                    .map_err(database_error)?;
            }
        }
        Ok(object.into())
    }
    fn decode<T: DeserializeOwned>(value: JsValue) -> Result<T, StorageError> {
        let json = js_sys::Reflect::get(&value, &"record".into())
            .map_err(database_error)?
            .as_string()
            .ok_or_else(|| {
                StorageError::Database("Stored record has an invalid envelope".into())
            })?;
        Ok(serde_json::from_str(&json)?)
    }
    async fn finish<T>(
        tx: Transaction,
        result: Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        match result {
            Err(error) => {
                let _ = tx.abort().await;
                Err(error)
            }
            Ok(value) => {
                let completion = tx.done().await.map_err(database_error)?;
                if !completion.is_committed() {
                    return Err(StorageError::Transaction(
                        "IndexedDB transaction aborted".into(),
                    ));
                }
                Ok(value)
            }
        }
    }
    async fn get<T: DeserializeOwned>(
        &self,
        name: &str,
        id: &str,
    ) -> Result<Option<T>, StorageError> {
        let tx = self.transaction(&[name], TransactionMode::ReadOnly)?;
        let result = async {
            tx.store(name)
                .map_err(database_error)?
                .get(id.into())
                .await
                .map_err(database_error)?
                .map(Self::decode)
                .transpose()
        }
        .await;
        Self::finish(tx, result).await
    }
    async fn records<T: DeserializeOwned>(
        &self,
        name: &str,
        filter: Option<(&str, &str)>,
    ) -> Result<Vec<T>, StorageError> {
        let tx = self.transaction(&[name], TransactionMode::ReadOnly)?;
        let result = async {
            let store = tx.store(name).map_err(database_error)?;
            let values = if let Some((field, value)) = filter {
                let range = KeyRange::only(&JsValue::from_str(value)).map_err(database_error)?;
                store
                    .index(field)
                    .map_err(database_error)?
                    .get_all(Some(range), None)
                    .await
                    .map_err(database_error)?
            } else {
                store.get_all(None, None).await.map_err(database_error)?
            };
            values.into_iter().map(Self::decode).collect()
        }
        .await;
        Self::finish(tx, result).await
    }
    async fn next_sequence(tx: &Transaction) -> Result<u64, StorageError> {
        let store = tx.store("metadata").map_err(database_error)?;
        let previous = store
            .get("event_sequence".into())
            .await
            .map_err(database_error)?;
        let current = if let Some(value) = previous {
            js_sys::Reflect::get(&value, &"value".into())
                .map_err(database_error)?
                .as_string()
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| StorageError::Database("Invalid event sequence counter".into()))?
        } else {
            0
        };
        let next = current
            .checked_add(1)
            .ok_or_else(|| StorageError::Database("Event sequence exhausted".into()))?;
        let value = js_sys::Object::new();
        js_sys::Reflect::set(&value, &"id".into(), &"event_sequence".into())
            .map_err(database_error)?;
        js_sys::Reflect::set(&value, &"value".into(), &next.to_string().into())
            .map_err(database_error)?;
        store.put(&value, None).await.map_err(database_error)?;
        Ok(next)
    }
}

#[async_trait(?Send)]
impl Storage for WasmStorage {
    type Error = StorageError;
    async fn store_agent(&self, value: &AgentModel) -> Result<(), StorageError> {
        self.write_batch(&[WasmMutation::Agent(value.clone())])
            .await
    }
    async fn get_agent(&self, id: &str) -> Result<Option<AgentModel>, StorageError> {
        self.get("agents", id).await
    }
    async fn update_agent(&self, value: &AgentModel) -> Result<(), StorageError> {
        self.store_agent(value).await
    }
    async fn delete_agent(&self, id: &str) -> Result<(), StorageError> {
        self.write_batch(&[WasmMutation::DeleteAgent(id.into())])
            .await
    }
    async fn list_agents(&self) -> Result<Vec<AgentModel>, StorageError> {
        self.records("agents", None).await
    }
    async fn list_agents_by_status(&self, status: &str) -> Result<Vec<AgentModel>, StorageError> {
        self.records("agents", Some(("status", status))).await
    }
    async fn store_task(&self, value: &TaskModel) -> Result<(), StorageError> {
        self.write_batch(&[WasmMutation::Task(value.clone())]).await
    }
    async fn get_task(&self, id: &str) -> Result<Option<TaskModel>, StorageError> {
        self.get("tasks", id).await
    }
    async fn update_task(&self, value: &TaskModel) -> Result<(), StorageError> {
        self.store_task(value).await
    }
    async fn get_pending_tasks(&self) -> Result<Vec<TaskModel>, StorageError> {
        let mut tasks: Vec<TaskModel> = self.records("tasks", Some(("status", "pending"))).await?;
        tasks.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(tasks)
    }
    async fn get_tasks_by_agent(&self, agent: &str) -> Result<Vec<TaskModel>, StorageError> {
        self.records("tasks", Some(("assigned_to", agent))).await
    }
    async fn claim_task(&self, id: &str, agent: &str) -> Result<bool, StorageError> {
        let tx = self.transaction(&["tasks"], TransactionMode::ReadWrite)?;
        let result = async {
            let store = tx.store("tasks").map_err(database_error)?;
            let Some(value) = store.get(id.into()).await.map_err(database_error)? else {
                return Ok(false);
            };
            let mut task: TaskModel = Self::decode(value)?;
            if task.status != TaskStatus::Pending {
                return Ok(false);
            }
            task.assign_to(agent);
            store
                .put(&Self::encode(&task)?, None)
                .await
                .map_err(database_error)?;
            Ok(true)
        }
        .await;
        Self::finish(tx, result).await
    }
    async fn store_event(&self, value: &EventModel) -> Result<(), StorageError> {
        self.write_batch(&[WasmMutation::Event(value.clone())])
            .await
    }
    async fn get_events_by_agent(
        &self,
        agent: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, StorageError> {
        let mut events: Vec<EventModel> = self.records("events", Some(("agent_id", agent))).await?;
        events.sort_by(|a, b| {
            b.timestamp
                .cmp(&a.timestamp)
                .then_with(|| b.sequence.cmp(&a.sequence))
        });
        events.truncate(limit);
        Ok(events)
    }
    async fn get_events_by_type(
        &self,
        kind: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, StorageError> {
        let mut events: Vec<EventModel> =
            self.records("events", Some(("event_type", kind))).await?;
        events.sort_by(|a, b| {
            b.timestamp
                .cmp(&a.timestamp)
                .then_with(|| b.sequence.cmp(&a.sequence))
        });
        events.truncate(limit);
        Ok(events)
    }
    async fn get_events_since(&self, timestamp: i64) -> Result<Vec<EventModel>, StorageError> {
        let mut events: Vec<EventModel> = self.records("events", None).await?;
        events.retain(|e| e.timestamp.timestamp() >= timestamp);
        events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp).then_with(|| a.id.cmp(&b.id)));
        Ok(events)
    }
    async fn store_message(&self, value: &MessageModel) -> Result<(), StorageError> {
        self.write_batch(&[WasmMutation::Message(value.clone())])
            .await
    }
    async fn get_messages_between(
        &self,
        first: &str,
        second: &str,
        limit: usize,
    ) -> Result<Vec<MessageModel>, StorageError> {
        let mut messages: Vec<MessageModel> = self.records("messages", None).await?;
        messages.retain(|m| {
            (m.from_agent == first && m.to_agent == second)
                || (m.from_agent == second && m.to_agent == first)
        });
        messages.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        messages.truncate(limit);
        Ok(messages)
    }
    async fn get_unread_messages(&self, agent: &str) -> Result<Vec<MessageModel>, StorageError> {
        let mut messages: Vec<MessageModel> =
            self.records("messages", Some(("to_agent", agent))).await?;
        messages.retain(|m| !m.read);
        Ok(messages)
    }
    async fn mark_message_read(&self, id: &str) -> Result<(), StorageError> {
        let tx = self.transaction(&["messages"], TransactionMode::ReadWrite)?;
        let result = async {
            let store = tx.store("messages").map_err(database_error)?;
            let value = store
                .get(id.into())
                .await
                .map_err(database_error)?
                .ok_or_else(|| StorageError::NotFound(format!("Message {id} not found")))?;
            let mut message: MessageModel = Self::decode(value)?;
            message.mark_read();
            store
                .put(&Self::encode(&message)?, None)
                .await
                .map_err(database_error)?;
            Ok(())
        }
        .await;
        Self::finish(tx, result).await
    }
    async fn store_metric(&self, value: &MetricModel) -> Result<(), StorageError> {
        self.write_batch(&[WasmMutation::Metric(value.clone())])
            .await
    }
    async fn get_metrics_by_agent(
        &self,
        agent: &str,
        kind: &str,
    ) -> Result<Vec<MetricModel>, StorageError> {
        let mut metrics: Vec<MetricModel> =
            self.records("metrics", Some(("agent_id", agent))).await?;
        metrics.retain(|m| m.metric_type == kind);
        Ok(metrics)
    }
    async fn get_aggregated_metrics(
        &self,
        kind: &str,
        start: i64,
        end: i64,
    ) -> Result<Vec<MetricModel>, StorageError> {
        let metrics: Vec<MetricModel> =
            self.records("metrics", Some(("metric_type", kind))).await?;
        let mut groups: HashMap<(Option<String>, String), (f64, usize)> = HashMap::new();
        for metric in metrics {
            let time = metric.timestamp.timestamp();
            if time >= start && time <= end {
                let entry = groups.entry((metric.agent_id, metric.unit)).or_default();
                entry.0 += metric.value;
                entry.1 += 1;
            }
        }
        Ok(groups
            .into_iter()
            .map(|((agent, unit), (sum, count))| {
                let mut value = MetricModel::new(kind.into(), sum / count as f64, unit);
                value.agent_id = agent;
                value.tags.insert("count".into(), count.to_string());
                value
            })
            .collect())
    }
    async fn begin_transaction(&self) -> Result<Box<dyn TransactionTrait>, StorageError> {
        Err(StorageError::Transaction("IndexedDB transactions are scoped to write_batch; use WasmStorage::write_batch for atomic multi-record updates".into()))
    }
    async fn vacuum(&self) -> Result<(), StorageError> {
        Err(StorageError::Other(
            "IndexedDB compaction is managed by the browser".into(),
        ))
    }
    /// Commit barrier for previously queued writes; IndexedDB has no exposed WAL.
    async fn checkpoint(&self) -> Result<(), StorageError> {
        let tx = self.transaction(&STORES, TransactionMode::ReadOnly)?;
        Self::finish(tx, Ok(())).await
    }
    /// Serialized payload bytes, not physical origin-wide disk usage.
    async fn get_storage_size(&self) -> Result<u64, StorageError> {
        let tx = self.transaction(&STORES, TransactionMode::ReadOnly)?;
        let result = async {
            let mut size = 0u64;
            for name in STORES {
                for value in tx
                    .store(name)
                    .map_err(database_error)?
                    .get_all(None, None)
                    .await
                    .map_err(database_error)?
                {
                    let json = js_sys::JSON::stringify(&value)
                        .map_err(database_error)?
                        .as_string()
                        .ok_or_else(|| StorageError::Database("Invalid stored value".into()))?;
                    size = size
                        .checked_add(json.len() as u64)
                        .ok_or_else(|| StorageError::Database("Storage size overflow".into()))?;
                }
            }
            Ok(size)
        }
        .await;
        Self::finish(tx, result).await
    }
}
