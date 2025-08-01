# ruv-FANN Swarm Persistence Architecture

## Executive Summary

This document outlines the design of a lightweight, WASM-compatible ORM and SQLite persistence layer for the ruv-FANN swarm state management system. The architecture prioritizes performance, type safety, and cross-platform compatibility while maintaining minimal overhead for swarm coordination.

## ORM Options Analysis

### 1. Diesel
**Type**: Full-featured ORM with compile-time SQL verification

**Pros:**
- Compile-time SQL type checking
- Powerful migration system
- Mature and well-tested
- Strong typing with automatic code generation

**Cons:**
- Heavy dependency footprint
- Not WASM-compatible
- Requires build.rs and compile-time code generation
- Overkill for lightweight swarm state

**Verdict**: Not suitable due to WASM incompatibility and heavyweight nature

### 2. SeaORM
**Type**: Async-first ORM with dynamic query building

**Pros:**
- Async/await native support
- Dynamic query builder
- Good abstraction layer
- Active development

**Cons:**
- Requires async runtime (tokio/async-std)
- Not fully WASM-compatible
- Larger dependency tree
- More complex than needed for swarm state

**Verdict**: Good for server-side, but async overhead and WASM issues make it unsuitable

### 3. sqlx
**Type**: Async SQL toolkit with compile-time checked queries

**Pros:**
- Compile-time SQL verification
- Lightweight compared to full ORMs
- Good performance
- Direct SQL with type safety

**Cons:**
- Async-only
- Limited WASM support
- Requires macros for compile-time checking
- Network-oriented (not ideal for embedded SQLite)

**Verdict**: Good middle ground but async requirement limits WASM usage

### 4. rusqlite
**Type**: Direct SQLite bindings for Rust

**Pros:**
- Minimal overhead
- Direct SQLite access
- Synchronous API
- Can be made WASM-compatible with sql.js
- Lightweight dependency

**Cons:**
- No ORM features
- Manual SQL writing
- No automatic migrations

**Verdict**: **RECOMMENDED** - Best choice for lightweight, WASM-compatible persistence

## Recommended Architecture

### Core Stack
- **rusqlite**: Direct SQLite access for native platforms
- **sql.js**: SQLite compiled to WASM for browser environments
- **Custom lightweight ORM**: Minimal abstraction layer for type safety
- **IndexedDB**: Fallback storage for browser persistence

## Database Schema Design

### Core Tables

```sql
-- Agent registry and metadata
CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL CHECK (status IN ('idle', 'busy', 'failed', 'terminated')),
    capabilities JSON NOT NULL DEFAULT '[]',
    configuration JSON NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    last_heartbeat INTEGER NOT NULL,
    terminated_at INTEGER,
    UNIQUE(name)
);

-- Task queue and history
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    parent_task_id TEXT,
    type TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5,
    payload JSON NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'assigned', 'running', 'completed', 'failed', 'cancelled')),
    result JSON,
    error_message TEXT,
    created_at INTEGER NOT NULL,
    assigned_at INTEGER,
    started_at INTEGER,
    completed_at INTEGER,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE SET NULL,
    FOREIGN KEY (parent_task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

-- Inter-agent communication logs
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    from_agent TEXT NOT NULL,
    to_agent TEXT,
    topic TEXT NOT NULL,
    content JSON NOT NULL,
    correlation_id TEXT,
    timestamp INTEGER NOT NULL,
    delivered BOOLEAN DEFAULT FALSE,
    acknowledged BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (from_agent) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (to_agent) REFERENCES agents(id) ON DELETE CASCADE
);

-- Performance metrics and telemetry
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    tags JSON,
    timestamp INTEGER NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- Swarm configuration and topology
CREATE TABLE swarm_config (
    key TEXT PRIMARY KEY,
    value JSON NOT NULL,
    updated_at INTEGER NOT NULL,
    updated_by TEXT
);

-- Agent relationships and coordination groups
CREATE TABLE agent_groups (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    configuration JSON NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE TABLE agent_group_members (
    group_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    role TEXT,
    joined_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, agent_id),
    FOREIGN KEY (group_id) REFERENCES agent_groups(id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- Event sourcing for state reconstruction
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aggregate_id TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSON NOT NULL,
    metadata JSON,
    timestamp INTEGER NOT NULL,
    sequence_number INTEGER NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_heartbeat ON agents(last_heartbeat);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_agent ON tasks(agent_id);
CREATE INDEX idx_tasks_priority ON tasks(priority DESC, created_at ASC);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
CREATE INDEX idx_messages_topic ON messages(topic);
CREATE INDEX idx_messages_correlation ON messages(correlation_id);
CREATE INDEX idx_metrics_agent_time ON metrics(agent_id, timestamp);
CREATE INDEX idx_metrics_type ON metrics(metric_type, metric_name);
CREATE INDEX idx_events_aggregate ON events(aggregate_id, sequence_number);
```

## Lightweight ORM Design

### Core Traits

```rust
use std::error::Error;
use rusqlite::{Connection, Transaction};

/// Base trait for all persistent entities
pub trait Entity: Sized {
    /// Primary key type
    type Id: ToString + FromStr;
    
    /// Table name
    const TABLE_NAME: &'static str;
    
    /// Create table SQL
    const CREATE_TABLE: &'static str;
    
    /// Convert to SQL parameters
    fn to_params(&self) -> Vec<(&str, &dyn ToSql)>;
    
    /// Create from row
    fn from_row(row: &Row) -> Result<Self, Box<dyn Error>>;
    
    /// Get primary key
    fn id(&self) -> &Self::Id;
}

/// Repository pattern for data access
pub trait Repository<T: Entity> {
    fn insert(&self, entity: &T) -> Result<(), Box<dyn Error>>;
    fn update(&self, entity: &T) -> Result<(), Box<dyn Error>>;
    fn delete(&self, id: &T::Id) -> Result<(), Box<dyn Error>>;
    fn find_by_id(&self, id: &T::Id) -> Result<Option<T>, Box<dyn Error>>;
    fn find_all(&self) -> Result<Vec<T>, Box<dyn Error>>;
}

/// Unit of Work pattern for transactions
pub trait UnitOfWork {
    fn begin(&mut self) -> Result<Transaction, Box<dyn Error>>;
    fn commit(self) -> Result<(), Box<dyn Error>>;
    fn rollback(self) -> Result<(), Box<dyn Error>>;
}

/// Query builder for type-safe queries
pub struct QueryBuilder<T: Entity> {
    table: &'static str,
    wheres: Vec<String>,
    order_by: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    _phantom: PhantomData<T>,
}
```

### Connection Management

```rust
/// Connection pool for native environments
pub struct ConnectionPool {
    path: PathBuf,
    connections: Arc<Mutex<Vec<Connection>>>,
    max_connections: usize,
}

/// Abstract connection interface for cross-platform support
pub trait DbConnection {
    fn execute(&self, sql: &str, params: &[&dyn ToSql]) -> Result<usize, Box<dyn Error>>;
    fn query<T, F>(&self, sql: &str, params: &[&dyn ToSql], f: F) -> Result<Vec<T>, Box<dyn Error>>
    where
        F: Fn(&Row) -> Result<T, Box<dyn Error>>;
    fn transaction<F, R>(&mut self, f: F) -> Result<R, Box<dyn Error>>
    where
        F: FnOnce(&Transaction) -> Result<R, Box<dyn Error>>;
}
```

## WASM-Compatible Persistence Layer

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│                   (Swarm Coordination)                   │
├─────────────────────────────────────────────────────────┤
│                  Persistence Abstraction                 │
│              (Platform-agnostic interface)               │
├─────────────────────────────────────────────────────────┤
│   Native Backend    │       WASM Backend                │
│   (rusqlite)        │    (sql.js + IndexedDB)          │
├─────────────────────────────────────────────────────────┤
│      SQLite         │    In-Memory SQLite + IndexedDB   │
└─────────────────────────────────────────────────────────┘
```

### Platform Detection and Backend Selection

```rust
#[cfg(target_arch = "wasm32")]
pub use wasm_backend::*;

#[cfg(not(target_arch = "wasm32"))]
pub use native_backend::*;

/// Cross-platform persistence factory
pub struct PersistenceFactory;

impl PersistenceFactory {
    pub fn create_connection(config: &PersistenceConfig) -> Result<Box<dyn DbConnection>, Box<dyn Error>> {
        #[cfg(target_arch = "wasm32")]
        {
            WasmConnection::new(config)
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            NativeConnection::new(config)
        }
    }
}
```

### WASM Backend Implementation

```rust
/// WASM-specific connection using sql.js and IndexedDB
pub struct WasmConnection {
    sql_js: SqlJs,
    indexed_db: IndexedDbHandle,
    sync_interval: Duration,
}

impl WasmConnection {
    /// Initialize sql.js with periodic sync to IndexedDB
    pub async fn new(config: &PersistenceConfig) -> Result<Self, Box<dyn Error>> {
        // Load sql.js WASM module
        let sql_js = SqlJs::load().await?;
        
        // Open IndexedDB for persistent storage
        let indexed_db = IndexedDbHandle::open("ruv_swarm", 1).await?;
        
        // Load existing data from IndexedDB if available
        if let Some(data) = indexed_db.get("swarm_state").await? {
            sql_js.load_from_binary(&data)?;
        }
        
        Ok(Self {
            sql_js,
            indexed_db,
            sync_interval: config.sync_interval,
        })
    }
    
    /// Sync in-memory SQLite to IndexedDB
    async fn sync_to_indexed_db(&self) -> Result<(), Box<dyn Error>> {
        let binary = self.sql_js.export_to_binary()?;
        self.indexed_db.put("swarm_state", &binary).await?;
        Ok(())
    }
}
```

## Migration Strategy

### Schema Versioning

```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at INTEGER NOT NULL,
    checksum TEXT NOT NULL
);
```

### Migration System

```rust
pub struct MigrationManager {
    migrations: Vec<Migration>,
}

pub struct Migration {
    version: u32,
    name: String,
    up: String,
    down: String,
    checksum: String,
}

impl MigrationManager {
    pub fn migrate(&self, conn: &dyn DbConnection) -> Result<(), Box<dyn Error>> {
        let current_version = self.get_current_version(conn)?;
        
        for migration in &self.migrations {
            if migration.version > current_version {
                conn.transaction(|tx| {
                    tx.execute(&migration.up, &[])?;
                    self.record_migration(tx, migration)?;
                    Ok(())
                })?;
            }
        }
        
        Ok(())
    }
}
```

## Backup and Recovery

### Backup Strategy

1. **Continuous Backup**: Event sourcing for state reconstruction
2. **Periodic Snapshots**: Full database exports at intervals
3. **Incremental Backups**: Change tracking for efficient updates

### Recovery Procedures

```rust
pub struct BackupManager {
    storage: Box<dyn BackupStorage>,
}

pub trait BackupStorage {
    fn store(&self, backup: &Backup) -> Result<(), Box<dyn Error>>;
    fn restore(&self, id: &str) -> Result<Backup, Box<dyn Error>>;
    fn list(&self) -> Result<Vec<BackupMetadata>, Box<dyn Error>>;
}

impl BackupManager {
    /// Create full backup
    pub fn backup_full(&self, conn: &dyn DbConnection) -> Result<String, Box<dyn Error>> {
        let backup = Backup {
            id: generate_id(),
            timestamp: SystemTime::now(),
            type_: BackupType::Full,
            data: conn.export()?,
        };
        
        self.storage.store(&backup)?;
        Ok(backup.id)
    }
    
    /// Restore from backup
    pub fn restore(&self, id: &str, conn: &mut dyn DbConnection) -> Result<(), Box<dyn Error>> {
        let backup = self.storage.restore(id)?;
        conn.import(&backup.data)?;
        Ok(())
    }
}
```

## Performance Optimization

### Query Optimization

1. **Prepared Statements**: Cache and reuse for frequent queries
2. **Batch Operations**: Group inserts/updates for efficiency
3. **Index Strategy**: Carefully designed indexes for common queries
4. **Connection Pooling**: Reuse connections in native environments

### Caching Layer

```rust
pub struct CachedRepository<T: Entity> {
    inner: Box<dyn Repository<T>>,
    cache: Arc<RwLock<HashMap<T::Id, Arc<T>>>>,
    ttl: Duration,
}

impl<T: Entity> Repository<T> for CachedRepository<T> {
    fn find_by_id(&self, id: &T::Id) -> Result<Option<T>, Box<dyn Error>> {
        // Check cache first
        if let Some(cached) = self.cache.read().unwrap().get(id) {
            return Ok(Some((**cached).clone()));
        }
        
        // Fetch from database
        let result = self.inner.find_by_id(id)?;
        
        // Update cache
        if let Some(ref entity) = result {
            self.cache.write().unwrap().insert(
                id.clone(),
                Arc::new(entity.clone())
            );
        }
        
        Ok(result)
    }
}
```

### Performance Monitoring

```rust
pub struct PerformanceMonitor {
    metrics: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl PerformanceMonitor {
    pub fn track<F, R>(&self, operation: &str, f: F) -> Result<R, Box<dyn Error>>
    where
        F: FnOnce() -> Result<R, Box<dyn Error>>
    {
        let start = Instant::now();
        let result = f()?;
        let duration = start.elapsed();
        
        self.metrics.lock().unwrap()
            .entry(operation.to_string())
            .or_default()
            .push(duration);
            
        Ok(result)
    }
}
```

## Event Sourcing Implementation

### Event Store

```rust
pub struct EventStore {
    conn: Box<dyn DbConnection>,
}

impl EventStore {
    pub fn append(&self, event: &Event) -> Result<(), Box<dyn Error>> {
        let sql = "INSERT INTO events (aggregate_id, aggregate_type, event_type, event_data, metadata, timestamp, sequence_number) 
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)";
        
        self.conn.execute(sql, &[
            &event.aggregate_id,
            &event.aggregate_type,
            &event.event_type,
            &event.event_data,
            &event.metadata,
            &event.timestamp,
            &event.sequence_number,
        ])?;
        
        Ok(())
    }
    
    pub fn get_events(&self, aggregate_id: &str, after_sequence: Option<u64>) -> Result<Vec<Event>, Box<dyn Error>> {
        let sql = "SELECT * FROM events WHERE aggregate_id = ?1 AND sequence_number > ?2 ORDER BY sequence_number";
        
        self.conn.query(sql, &[
            &aggregate_id,
            &after_sequence.unwrap_or(0),
        ], Event::from_row)
    }
}
```

## Implementation Checklist

- [ ] Create ruv-swarm crate with persistence module
- [ ] Implement rusqlite backend for native platforms
- [ ] Implement sql.js + IndexedDB backend for WASM
- [ ] Create lightweight ORM abstractions
- [ ] Implement migration system
- [ ] Add connection pooling for native
- [ ] Create caching layer
- [ ] Implement event sourcing
- [ ] Add performance monitoring
- [ ] Write comprehensive tests
- [ ] Create benchmarks
- [ ] Document API usage

## Conclusion

This persistence architecture provides a lightweight, type-safe, and cross-platform solution for ruv-FANN swarm state management. By using rusqlite with custom ORM abstractions and supporting both native and WASM environments, the system maintains high performance while enabling deployment flexibility. The event sourcing capability ensures reliable state reconstruction and audit trails for distributed swarm coordination.