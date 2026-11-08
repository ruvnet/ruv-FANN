//! Atomic multi-record browser writes.
use super::{database_error, WasmStorage};
use crate::{models::*, StorageError};
use rexie::TransactionMode;

/// A mutation in one atomic IndexedDB transaction. Inputs are copied.
#[derive(Debug, Clone)]
pub enum WasmMutation {
    Agent(AgentModel),
    Task(TaskModel),
    Event(EventModel),
    Message(MessageModel),
    Metric(MetricModel),
    DeleteAgent(String),
}

impl WasmStorage {
    /// Commit all mutations or abort all of them, including event sequence changes.
    pub async fn write_batch(&self, mutations: &[WasmMutation]) -> Result<(), StorageError> {
        let mut stores = Vec::new();
        for mutation in mutations {
            let name = match mutation {
                WasmMutation::Agent(_) | WasmMutation::DeleteAgent(_) => "agents",
                WasmMutation::Task(_) => "tasks",
                WasmMutation::Event(_) => {
                    stores.push("metadata");
                    "events"
                }
                WasmMutation::Message(_) => "messages",
                WasmMutation::Metric(_) => "metrics",
            };
            stores.push(name);
        }
        stores.sort_unstable();
        stores.dedup();
        if stores.is_empty() {
            return Ok(());
        }
        let tx = self.transaction(&stores, TransactionMode::ReadWrite)?;
        let result = async {
            for mutation in mutations {
                let (name, value) = match mutation {
                    WasmMutation::Agent(model) => ("agents", Self::encode(model)?),
                    WasmMutation::Task(model) => ("tasks", Self::encode(model)?),
                    WasmMutation::Event(model) => {
                        let mut model = model.clone();
                        model.sequence = Self::next_sequence(&tx).await?;
                        ("events", Self::encode(&model)?)
                    }
                    WasmMutation::Message(model) => ("messages", Self::encode(model)?),
                    WasmMutation::Metric(model) => {
                        if !model.value.is_finite() {
                            return Err(StorageError::Database(
                                "Metric values must be finite".into(),
                            ));
                        }
                        ("metrics", Self::encode(model)?)
                    }
                    WasmMutation::DeleteAgent(id) => {
                        tx.store("agents")
                            .map_err(database_error)?
                            .delete(id.into())
                            .await
                            .map_err(database_error)?;
                        continue;
                    }
                };
                let store = tx.store(name).map_err(database_error)?;
                if name == "events" {
                    store.add(&value, None).await.map_err(database_error)?;
                } else {
                    store.put(&value, None).await.map_err(database_error)?;
                }
            }
            Ok(())
        }
        .await;
        Self::finish(tx, result).await
    }
}
