# Storage Backends

This guide helps you choose the right storage backend for your use case and understand the trade-offs between different implementations.

## Overview

ruv-swarm-persistence supports three storage backends:

| Backend | Use Case | Persistence | Performance | Platform |
|---------|----------|-------------|-------------|----------|
| **SQLite** | Production applications | Persistent | High | Native only |
| **Memory** | Testing & development | In-memory | Very high | All platforms |
| **WASM** | Browser applications | Persistent* | Medium | WASM only |

*WASM storage uses IndexedDB which persists data in the browser

## SQLite Storage

### When to Use SQLite

✅ **Choose SQLite when:**
- Building production applications
- Need persistent data across restarts
- Require ACID transactions
- Working with large datasets
- Need SQL query capabilities
- Building server applications

❌ **Avoid SQLite when:**
- Running in WASM/browser environments
- Need the absolute fastest performance
- Building temporary/testing applications
- Memory constraints are critical

### SQLite Features

#### ACID Compliance
```rust
// Full ACID transaction support
let mut tx = storage.begin_transaction().await?;

// Multiple operations in single transaction
storage.store_agent(&agent).await?;
storage.store_task(&task).await?;
storage.store_event(&event).await?;

// Commit all changes atomically
tx.commit().await?;
```

#### Connection Pooling
```rust
use std::time::Duration;

let storage = SqliteStorage::builder()
    .max_connections(32)                    // Max concurrent connections
    .min_idle_connections(4)                // Keep connections warm
    .connection_timeout(Duration::from_secs(30))  // Connection timeout
    .idle_timeout(Duration::from_secs(600))       // Idle connection timeout
    .build("production.db")
    .await?;
```

#### Optimized Configuration

The SQLite backend uses production-optimized settings:

```sql
-- Optimized for concurrent access
PRAGMA foreign_keys = ON;           -- Enforce referential integrity
PRAGMA journal_mode = WAL;          -- Write-Ahead Logging for concurrency
PRAGMA synchronous = NORMAL;        -- Balance between safety and performance
PRAGMA busy_timeout = 30000;        -- 30 second timeout for busy database
```

#### Performance Indexing

Automatic indexes for common query patterns:

```sql
-- Performance-optimized indexes
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_tasks_priority ON tasks(priority DESC);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_messages_unread ON messages(to_agent, read);
CREATE INDEX idx_metrics_type_time ON metrics(metric_type, timestamp);
```

### SQLite Performance

**Benchmarks on MacBook Pro M2 Max, 32GB RAM:**

| Operation | Throughput | Latency (avg) | Hardware Context |
|-----------|------------|---------------|------------------|
| Agent Insert | 15,000 ops/sec | 0.067ms | M2 Max |
| Task Query | 25,000 ops/sec | 0.04ms | M2 Max |
| Event Batch | 12,166 events/sec | 0.082ms | M2 Max |
| Transaction | 5,000 ops/sec | 0.2ms | M2 Max |

### SQLite Best Practices

#### Connection Management
```rust
// Use connection pooling in production
let storage = SqliteStorage::builder()
    .max_connections(std::cmp::min(32, num_cpus::get() * 4))
    .build("app.db")
    .await?;

// Monitor pool health
let pool_status = storage.pool_status().await?;
if pool_status.active > pool_status.max * 0.8 {
    eprintln!("Warning: High connection pool utilization");
}
```

#### Transaction Scoping
```rust
// Keep transactions short and focused
async fn process_batch(storage: &SqliteStorage, items: Vec<Item>) -> Result<(), StorageError> {
    let mut tx = storage.begin_transaction().await?;
    
    // Process all items in single transaction
    for item in items {
        storage.store_item(&item).await?;
    }
    
    tx.commit().await?;
    Ok(())
}
```

## Memory Storage

### When to Use Memory

✅ **Choose Memory when:**
- Writing tests and benchmarks
- Building development prototypes
- Need maximum performance
- Working with temporary data
- Running in constrained environments

❌ **Avoid Memory when:**
- Need data persistence across restarts
- Working with large datasets
- Building production applications
- Need ACID guarantees

### Memory Features

#### Ultra-High Performance
```rust
// Memory storage provides the fastest possible operations
let storage = MemoryStorage::new();

// No I/O overhead - all operations are in-memory
let agent = AgentModel::new("worker".to_string(), "type".to_string(), vec![]);
storage.store_agent(&agent).await?;  // < 1μs typical
```

#### Thread-Safe Concurrent Access
```rust
use std::sync::Arc;

let storage = Arc::new(MemoryStorage::new());

// Safe to share across multiple tasks
let storage_clone = storage.clone();
tokio::spawn(async move {
    storage_clone.store_agent(&agent).await.unwrap();
});
```

#### Zero Configuration
```rust
// No setup required - ready to use immediately
let storage = MemoryStorage::new();
// Start using immediately
```

### Memory Performance

**Benchmarks on MacBook Pro M2 Max, 32GB RAM:**

| Operation | Throughput | Latency (avg) | Hardware Context |
|-----------|------------|---------------|------------------|
| Agent Insert | 100,000+ ops/sec | 0.01ms | M2 Max |
| Task Query | 200,000+ ops/sec | 0.005ms | M2 Max |
| Event Batch | 48,706 events/sec | 0.02ms | M2 Max |
| Transaction | 50,000+ ops/sec | 0.02ms | M2 Max |

### Memory Best Practices

#### Test Data Management
```rust
// Create isolated storage for each test
#[tokio::test]
async fn test_agent_operations() {
    let storage = MemoryStorage::new();
    
    // Test operations
    let agent = AgentModel::new("test".to_string(), "type".to_string(), vec![]);
    storage.store_agent(&agent).await.unwrap();
    
    // Storage is automatically cleaned up when dropped
}
```

#### Development Prototyping
```rust
// Use memory storage for rapid prototyping
async fn prototype_feature() -> Result<(), StorageError> {
    let storage = MemoryStorage::new();
    
    // Rapid iteration without I/O overhead
    for i in 0..10000 {
        let agent = AgentModel::new(format!("agent-{}", i), "worker".to_string(), vec![]);
        storage.store_agent(&agent).await?;
    }
    
    Ok(())
}
```

## WASM Storage

### When to Use WASM

✅ **Choose WASM when:**
- Building browser applications
- Need persistent data in web apps
- Working with WebAssembly
- Building cross-platform web tools

❌ **Avoid WASM when:**
- Building native applications
- Need maximum performance
- Working with large datasets
- Building server applications

### WASM Features

#### IndexedDB Integration
```rust
// Automatically uses IndexedDB in browser environments
#[cfg(target_arch = "wasm32")]
let storage = WasmStorage::new().await?;

// Persistent across browser sessions
storage.store_agent(&agent).await?;
```

#### Automatic Platform Detection
```rust
// Platform-aware initialization
let storage = init_storage(None).await?;
// Uses WasmStorage on WASM, SqliteStorage on native
```

### WASM Performance

**Typical browser performance (varies by browser):**

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Agent Insert | 1,000-5,000 ops/sec | IndexedDB overhead |
| Task Query | 2,000-10,000 ops/sec | Depends on browser |
| Event Batch | 500-2,000 events/sec | Async I/O limitations |

### WASM Best Practices

#### Browser Compatibility
```rust
// Handle browser-specific limitations
#[cfg(target_arch = "wasm32")]
async fn web_app_storage() -> Result<(), StorageError> {
    let storage = WasmStorage::new().await?;
    
    // Use smaller batch sizes for web
    let batch_size = 100;
    
    for batch in data.chunks(batch_size) {
        for item in batch {
            storage.store_item(item).await?;
        }
        
        // Yield control to browser
        wasm_bindgen_futures::JsFuture::from(
            js_sys::Promise::resolve(&wasm_bindgen::JsValue::NULL)
        ).await.unwrap();
    }
    
    Ok(())
}
```

## Choosing the Right Backend

### Decision Matrix

| Requirement | SQLite | Memory | WASM |
|-------------|--------|--------|------|
| **Persistence** | ✅ Full | ❌ None | ✅ Browser |
| **Performance** | ⚡ High | ⚡⚡ Ultra | ⚡ Medium |
| **ACID Transactions** | ✅ Full | ⚡ Fast | ⚡ Limited |
| **Production Ready** | ✅ Yes | ❌ No | ⚡ Browser only |
| **Testing** | ⚡ Good | ✅ Excellent | ❌ Limited |
| **Large Datasets** | ✅ Yes | ⚡ Limited | ❌ No |
| **Cross-Platform** | ✅ Native | ✅ All | ✅ WASM only |

### Use Case Recommendations

#### Production Web Service
```rust
// Use SQLite for persistent, high-performance backend
let storage = SqliteStorage::builder()
    .max_connections(32)
    .build("production.db")
    .await?;
```

#### Test Suite
```rust
// Use Memory for fast, isolated tests
#[tokio::test]
async fn test_feature() {
    let storage = MemoryStorage::new();
    // Test logic
}
```

#### Browser Application
```rust
// Use WASM for browser-based persistence
#[cfg(target_arch = "wasm32")]
let storage = WasmStorage::new().await?;
```

#### Development/Prototyping
```rust
// Use Memory for rapid iteration
let storage = MemoryStorage::new();
// Fast development cycle
```

## Migration Between Backends

### Development to Production

During development, you might start with Memory storage and migrate to SQLite:

```rust
// Development configuration
#[cfg(debug_assertions)]
let storage = MemoryStorage::new();

// Production configuration
#[cfg(not(debug_assertions))]
let storage = SqliteStorage::new("production.db").await?;
```

### Platform-Aware Code

Write backend-agnostic code using the Storage trait:

```rust
async fn business_logic<S: Storage>(storage: &S) -> Result<(), S::Error> {
    // This works with any storage backend
    let agent = AgentModel::new("worker".to_string(), "type".to_string(), vec![]);
    storage.store_agent(&agent).await?;
    
    let retrieved = storage.get_agent(&agent.id).await?;
    assert!(retrieved.is_some());
    
    Ok(())
}
```

## Security Considerations

### SQLite Security

#### Connection Security
```rust
// SQLite provides secure local storage
let storage = SqliteStorage::new("secure.db").await?;

// File permissions should be set appropriately
std::fs::set_permissions("secure.db", std::fs::Permissions::from_mode(0o600))?;
```

#### Query Security
```rust
// Always use parameterized queries
let (query, params) = QueryBuilder::<AgentModel>::new("agents")
    .where_eq("status", user_input)  // Safe - uses parameters
    .build();
```

### Memory Security

#### Data Lifetime
```rust
// Memory storage is automatically cleaned up
let storage = MemoryStorage::new();
// Data exists only while storage is in scope
// No persistence means no data leakage
```

### WASM Security

#### Browser Sandbox
```rust
// WASM storage is sandboxed by browser
let storage = WasmStorage::new().await?;
// Data is isolated per origin
// Subject to browser security policies
```

## Performance Tuning

### SQLite Optimization

#### Connection Pool Tuning
```rust
let storage = SqliteStorage::builder()
    .max_connections(std::cmp::min(32, num_cpus::get() * 4))
    .min_idle_connections(4)
    .connection_timeout(Duration::from_secs(30))
    .idle_timeout(Duration::from_secs(600))
    .build("optimized.db")
    .await?;
```

#### Batch Operations
```rust
// Use transactions for batch inserts
let mut tx = storage.begin_transaction().await?;

for item in large_batch {
    storage.store_item(&item).await?;
}

tx.commit().await?;
```

### Memory Optimization

#### Data Structure Efficiency
```rust
// Memory storage uses efficient data structures
let storage = MemoryStorage::new();

// Uses HashMap for O(1) lookups
// Uses Vec for ordered collections
// Optimized for in-memory operations
```

## Troubleshooting

### SQLite Issues

#### Database Locks
```rust
// If experiencing lock contention
let storage = SqliteStorage::builder()
    .max_connections(16)  // Reduce connection pool size
    .build("db.sqlite")
    .await?;

// Keep transactions short
let mut tx = storage.begin_transaction().await?;
// Minimal work here
tx.commit().await?;
```

#### Performance Issues
```rust
// Check connection pool utilization
let pool_status = storage.pool_status().await?;
println!("Pool utilization: {}/{}", pool_status.active, pool_status.max);

// Monitor query performance
let start = std::time::Instant::now();
let result = storage.get_agent("agent-123").await?;
let duration = start.elapsed();
println!("Query took: {:?}", duration);
```

### Memory Issues

#### Memory Usage
```rust
// Monitor memory usage in long-running processes
let storage = MemoryStorage::new();

// Consider periodically clearing old data
storage.vacuum().await?;  // Removes deleted items
```

### WASM Issues

#### Browser Limitations
```rust
// Handle quota exceeded errors
#[cfg(target_arch = "wasm32")]
match storage.store_large_data(&data).await {
    Err(StorageError::Other(msg)) if msg.contains("quota") => {
        // Handle quota exceeded
        eprintln!("Storage quota exceeded");
    },
    result => result?,
}
```

## Next Steps

- [Getting Started](./getting-started.md) - Learn how to use each backend
- [API Reference](./api-reference.md) - Detailed API documentation
- [Security Guide](./security-guide.md) - Security best practices
- [Testing Guide](./testing-guide.md) - Testing with different backends

---

*Choose the right storage backend for your specific needs and requirements!*