//! Persistence layer for RUV-Swarm with SQLite and ORM support
//!
//! This crate provides a flexible persistence layer with support for:
//! - SQLite backend for native platforms
//! - IndexedDB backend for WASM targets
//! - In-memory storage for testing
//! - Repository pattern with type-safe queries
//! - Transaction support and connection pooling

pub mod memory;
pub mod migrations;
pub mod models;
#[cfg(not(target_arch = "wasm32"))]
pub mod sqlite;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use async_trait::async_trait;
use std::error::Error as StdError;
use thiserror::Error;

pub use models::{AgentModel, EventModel, MessageModel, MetricModel, TaskModel};

pub use memory::MemoryStorage;
#[cfg(not(target_arch = "wasm32"))]
pub use sqlite::SqliteStorage;
#[cfg(target_arch = "wasm32")]
pub use wasm::WasmStorage;

/// Storage error types
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Transaction error: {0}")]
    Transaction(String),

    #[error("Migration error: {0}")]
    Migration(String),

    #[error("Connection pool error: {0}")]
    Pool(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<Box<dyn StdError + Send + Sync>> for StorageError {
    fn from(err: Box<dyn StdError + Send + Sync>) -> Self {
        StorageError::Other(err.to_string())
    }
}

/// Storage trait for persistence operations
#[async_trait]
pub trait Storage: Send + Sync {
    type Error: StdError + Send + Sync + 'static;

    // Agent operations
    async fn store_agent(&self, agent: &AgentModel) -> Result<(), Self::Error>;
    async fn get_agent(&self, id: &str) -> Result<Option<AgentModel>, Self::Error>;
    async fn update_agent(&self, agent: &AgentModel) -> Result<(), Self::Error>;
    async fn delete_agent(&self, id: &str) -> Result<(), Self::Error>;
    async fn list_agents(&self) -> Result<Vec<AgentModel>, Self::Error>;
    async fn list_agents_by_status(&self, status: &str) -> Result<Vec<AgentModel>, Self::Error>;

    // Task operations
    async fn store_task(&self, task: &TaskModel) -> Result<(), Self::Error>;
    async fn get_task(&self, id: &str) -> Result<Option<TaskModel>, Self::Error>;
    async fn update_task(&self, task: &TaskModel) -> Result<(), Self::Error>;
    async fn get_pending_tasks(&self) -> Result<Vec<TaskModel>, Self::Error>;
    async fn get_tasks_by_agent(&self, agent_id: &str) -> Result<Vec<TaskModel>, Self::Error>;
    async fn claim_task(&self, task_id: &str, agent_id: &str) -> Result<bool, Self::Error>;

    // Event operations
    async fn store_event(&self, event: &EventModel) -> Result<(), Self::Error>;
    async fn get_events_by_agent(
        &self,
        agent_id: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error>;
    async fn get_events_by_type(
        &self,
        event_type: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error>;
    async fn get_events_since(&self, timestamp: i64) -> Result<Vec<EventModel>, Self::Error>;

    // Message operations
    async fn store_message(&self, message: &MessageModel) -> Result<(), Self::Error>;
    async fn get_messages_between(
        &self,
        agent1: &str,
        agent2: &str,
        limit: usize,
    ) -> Result<Vec<MessageModel>, Self::Error>;
    async fn get_unread_messages(&self, agent_id: &str) -> Result<Vec<MessageModel>, Self::Error>;
    async fn mark_message_read(&self, message_id: &str) -> Result<(), Self::Error>;

    // Metric operations
    async fn store_metric(&self, metric: &MetricModel) -> Result<(), Self::Error>;
    async fn get_metrics_by_agent(
        &self,
        agent_id: &str,
        metric_type: &str,
    ) -> Result<Vec<MetricModel>, Self::Error>;
    async fn get_aggregated_metrics(
        &self,
        metric_type: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<MetricModel>, Self::Error>;

    // Transaction support
    async fn begin_transaction(&self) -> Result<Box<dyn Transaction>, Self::Error>;

    // Maintenance operations
    async fn vacuum(&self) -> Result<(), Self::Error>;
    async fn checkpoint(&self) -> Result<(), Self::Error>;
    async fn get_storage_size(&self) -> Result<u64, Self::Error>;
}

/// Transaction trait for atomic operations
#[async_trait]
pub trait Transaction: Send + Sync {
    async fn commit(self: Box<Self>) -> Result<(), StorageError>;
    async fn rollback(self: Box<Self>) -> Result<(), StorageError>;
}

/// Query builder for type-safe queries with parameterized values
pub struct QueryBuilder<T> {
    table: String,
    conditions: Vec<String>,
    parameters: Vec<String>,
    order_by: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> QueryBuilder<T> {
    pub fn new(table: &str) -> Self {
        Self {
            table: table.to_string(),
            conditions: Vec::new(),
            parameters: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn where_eq(mut self, field: &str, value: &str) -> Self {
        // Use parameterized placeholders to prevent SQL injection
        self.conditions.push(format!("{} = ?", field));
        self.parameters.push(value.to_string());
        self
    }

    pub fn where_like(mut self, field: &str, pattern: &str) -> Self {
        // Use parameterized placeholders to prevent SQL injection
        self.conditions.push(format!("{} LIKE ?", field));
        self.parameters.push(pattern.to_string());
        self
    }

    pub fn where_gt(mut self, field: &str, value: i64) -> Self {
        self.conditions.push(format!("{} > {}", field, value));
        self
    }

    pub fn order_by(mut self, field: &str, desc: bool) -> Self {
        self.order_by = Some(format!("{} {}", field, if desc { "DESC" } else { "ASC" }));
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn build(&self) -> (String, Vec<String>) {
        let mut query = format!("SELECT * FROM {}", self.table);

        if !self.conditions.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&self.conditions.join(" AND "));
        }

        if let Some(ref order) = self.order_by {
            query.push_str(" ORDER BY ");
            query.push_str(order);
        }

        if let Some(limit) = self.limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        if let Some(offset) = self.offset {
            query.push_str(&format!(" OFFSET {}", offset));
        }

        (query, self.parameters.clone())
    }
}

/// Repository pattern implementation
pub trait Repository<T> {
    type Error: StdError + Send + Sync + 'static;

    fn find_by_id(&self, id: &str) -> Result<Option<T>, Self::Error>;
    fn find_all(&self) -> Result<Vec<T>, Self::Error>;
    fn save(&self, entity: &T) -> Result<(), Self::Error>;
    fn update(&self, entity: &T) -> Result<(), Self::Error>;
    fn delete(&self, id: &str) -> Result<(), Self::Error>;
    fn query(&self, builder: QueryBuilder<T>) -> Result<Vec<T>, Self::Error>;
}

/// Initialize storage based on target platform
pub async fn init_storage(
    path: Option<&str>,
) -> Result<Box<dyn Storage<Error = StorageError>>, StorageError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = path.unwrap_or("swarm.db");
        Ok(Box::new(SqliteStorage::new(path).await?))
    }

    #[cfg(target_arch = "wasm32")]
    {
        Ok(Box::new(WasmStorage::new().await?))
    }
}

#[cfg(test)]
mod tests;
