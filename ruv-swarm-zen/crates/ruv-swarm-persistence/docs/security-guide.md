# Security Guide

This guide covers security best practices for ruv-swarm-persistence, including SQL injection prevention, ACID transactions, and secure deployment patterns.

## Table of Contents

- [SQL Injection Prevention](#sql-injection-prevention)
- [ACID Transactions](#acid-transactions)
- [Thread Safety](#thread-safety)
- [Input Validation](#input-validation)
- [Connection Security](#connection-security)
- [Deployment Security](#deployment-security)
- [Security Testing](#security-testing)

## SQL Injection Prevention

### Overview

ruv-swarm-persistence provides built-in SQL injection protection through parameterized queries and safe query building patterns.

### ✅ Safe Query Building

Always use the `QueryBuilder` with parameterized queries:

```rust
use ruv_swarm_persistence::QueryBuilder;

// ✅ SECURE: Uses parameterized queries
let (query, params) = QueryBuilder::<AgentModel>::new("agents")
    .where_eq("status", user_input)        // Safe - uses ? placeholder
    .where_like("name", search_pattern)    // Safe - uses ? placeholder
    .build();

// Generated SQL: "SELECT * FROM agents WHERE status = ? AND name LIKE ?"
// Parameters: [user_input, search_pattern]
```

### ❌ Unsafe Patterns to Avoid

Never use string concatenation or formatting for SQL queries:

```rust
// ❌ DANGEROUS: Never do this
let query = format!("SELECT * FROM agents WHERE status = '{}'", user_input);

// ❌ DANGEROUS: Vulnerable to injection
let query = "SELECT * FROM agents WHERE status = '".to_string() + user_input + "'";
```

### Security Features

#### Automatic Parameterization

The QueryBuilder automatically uses parameterized queries:

```rust
// All these methods use safe parameterization
let builder = QueryBuilder::<AgentModel>::new("agents")
    .where_eq("field", "value")     // Uses ? placeholder
    .where_like("field", "pattern") // Uses ? placeholder
    .where_gt("field", 123);        // Direct value (safe for numbers)
```

#### Injection Attack Prevention

The system prevents common injection attacks:

```rust
// Malicious input is safely handled
let malicious_input = "'; DROP TABLE agents; --";

let (query, params) = QueryBuilder::<AgentModel>::new("agents")
    .where_eq("name", malicious_input)
    .build();

// Safe result:
// SQL: "SELECT * FROM agents WHERE name = ?"
// Params: ["'; DROP TABLE agents; --"]
// The malicious input is treated as a literal string value
```

### Testing SQL Injection Protection

Comprehensive tests verify injection protection:

```rust
#[tokio::test]
async fn test_sql_injection_prevention() {
    let storage = SqliteStorage::new("test.db").await.unwrap();
    
    // Test various injection attempts
    let injection_attempts = vec![
        "'; DROP TABLE agents; --",
        "' OR '1'='1",
        "'; DELETE FROM agents; --",
        "' UNION SELECT * FROM sqlite_master; --",
    ];
    
    for attempt in injection_attempts {
        let (query, params) = QueryBuilder::<AgentModel>::new("agents")
            .where_eq("name", attempt)
            .build();
        
        // Query should use parameterized form
        assert!(query.contains("?"));
        assert!(!query.contains(attempt));
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], attempt);
    }
}
```

## ACID Transactions

### Overview

ruv-swarm-persistence provides full ACID (Atomicity, Consistency, Isolation, Durability) transaction support for data integrity.

### Transaction Basics

#### Atomic Operations

```rust
use ruv_swarm_persistence::{SqliteStorage, Transaction};

async fn atomic_operation(storage: &SqliteStorage) -> Result<(), StorageError> {
    // Begin transaction
    let mut tx = storage.begin_transaction().await?;
    
    // Multiple operations that must succeed together
    storage.store_agent(&agent1).await?;
    storage.store_agent(&agent2).await?;
    storage.store_task(&task).await?;
    
    // All operations committed together
    tx.commit().await?;
    
    Ok(())
}
```

#### Automatic Rollback on Error

```rust
async fn safe_batch_operation(storage: &SqliteStorage) -> Result<(), StorageError> {
    let mut tx = storage.begin_transaction().await?;
    
    // Process multiple items
    for item in items {
        storage.store_item(&item).await?;
        
        // If validation fails, transaction automatically rolls back
        if !validate_item(&item) {
            return Err(StorageError::Other("Invalid item".to_string()));
        }
    }
    
    // Only commit if all operations succeed
    tx.commit().await?;
    Ok(())
}
```

### Transaction Isolation

#### Preventing Race Conditions

```rust
// Transaction isolation prevents race conditions
async fn atomic_task_claim(storage: &SqliteStorage, task_id: &str, agent_id: &str) -> Result<bool, StorageError> {
    let mut tx = storage.begin_transaction().await?;
    
    // Check if task is available (within transaction)
    if let Some(task) = storage.get_task(task_id).await? {
        if task.assigned_to.is_none() {
            // Claim task atomically
            let mut claimed_task = task;
            claimed_task.assigned_to = Some(agent_id.to_string());
            storage.update_task(&claimed_task).await?;
            
            tx.commit().await?;
            return Ok(true);
        }
    }
    
    tx.rollback().await?;
    Ok(false)
}
```

### Transaction Best Practices

#### Keep Transactions Short

```rust
// ✅ GOOD: Short, focused transaction
async fn update_agent_status(storage: &SqliteStorage, agent_id: &str, status: AgentStatus) -> Result<(), StorageError> {
    let mut tx = storage.begin_transaction().await?;
    
    let mut agent = storage.get_agent(agent_id).await?
        .ok_or_else(|| StorageError::NotFound(agent_id.to_string()))?;
    
    agent.status = status;
    agent.updated_at = Utc::now();
    
    storage.update_agent(&agent).await?;
    tx.commit().await?;
    
    Ok(())
}
```

```rust
// ❌ BAD: Long-running transaction
async fn bad_transaction(storage: &SqliteStorage) -> Result<(), StorageError> {
    let mut tx = storage.begin_transaction().await?;
    
    // Don't do expensive work in transactions
    let data = fetch_data_from_external_api().await?;  // Slow!
    let processed = complex_processing(data).await?;   // Slow!
    
    storage.store_data(&processed).await?;
    tx.commit().await?;
    
    Ok(())
}
```

#### Handle Transaction Errors

```rust
async fn robust_transaction(storage: &SqliteStorage) -> Result<(), StorageError> {
    let mut tx = storage.begin_transaction().await?;
    
    match perform_operations(&storage).await {
        Ok(_) => {
            tx.commit().await?;
            Ok(())
        },
        Err(e) => {
            tx.rollback().await?;
            Err(e)
        }
    }
}
```

### Consistency Guarantees

#### Foreign Key Constraints

```sql
-- Foreign key constraints ensure referential integrity
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    assigned_to TEXT,
    FOREIGN KEY (assigned_to) REFERENCES agents(id)
);
```

#### Data Validation

```rust
impl TaskModel {
    fn validate(&self) -> Result<(), ValidationError> {
        // Validate task data before storage
        if self.task_type.is_empty() {
            return Err(ValidationError::EmptyTaskType);
        }
        
        if self.priority == TaskPriority::Critical && self.assigned_to.is_none() {
            return Err(ValidationError::CriticalTaskNotAssigned);
        }
        
        Ok(())
    }
}

// Use validation before storage
async fn store_validated_task(storage: &SqliteStorage, task: &TaskModel) -> Result<(), StorageError> {
    task.validate().map_err(|e| StorageError::Other(e.to_string()))?;
    
    let mut tx = storage.begin_transaction().await?;
    storage.store_task(task).await?;
    tx.commit().await?;
    
    Ok(())
}
```

## Thread Safety

### Overview

All storage implementations are thread-safe and can be safely shared across async tasks.

### Concurrent Access

#### Safe Sharing

```rust
use std::sync::Arc;

// Storage can be safely shared across tasks
let storage = Arc::new(SqliteStorage::new("app.db").await?);

// Clone and use in multiple tasks
let storage_clone = storage.clone();
tokio::spawn(async move {
    storage_clone.store_agent(&agent).await.unwrap();
});
```

#### Connection Pool Safety

```rust
// Connection pooling handles concurrent access
let storage = SqliteStorage::builder()
    .max_connections(32)  // Pool handles concurrent requests
    .build("concurrent.db")
    .await?;

// Multiple tasks can safely use the same storage
let mut handles = vec![];
for i in 0..100 {
    let storage_clone = storage.clone();
    let handle = tokio::spawn(async move {
        let agent = AgentModel::new(format!("agent-{}", i), "worker".to_string(), vec![]);
        storage_clone.store_agent(&agent).await.unwrap();
    });
    handles.push(handle);
}

// All operations complete safely
for handle in handles {
    handle.await.unwrap();
}
```

### Preventing Race Conditions

#### Atomic Operations

```rust
// Use atomic operations for race-free updates
async fn atomic_counter_increment(storage: &SqliteStorage, counter_id: &str) -> Result<i64, StorageError> {
    let mut tx = storage.begin_transaction().await?;
    
    let mut counter = storage.get_counter(counter_id).await?
        .ok_or_else(|| StorageError::NotFound(counter_id.to_string()))?;
    
    counter.value += 1;
    storage.update_counter(&counter).await?;
    
    tx.commit().await?;
    
    Ok(counter.value)
}
```

#### Task Claiming

```rust
// Atomic task claiming prevents double-assignment
async fn claim_task_safely(storage: &SqliteStorage, task_id: &str, agent_id: &str) -> Result<bool, StorageError> {
    // This operation is atomic at the database level
    storage.claim_task(task_id, agent_id).await
}
```

### Performance Considerations

#### Connection Pool Tuning

```rust
// Configure connection pool for your concurrency needs
let storage = SqliteStorage::builder()
    .max_connections(std::cmp::min(32, num_cpus::get() * 4))  // Scale with CPU cores
    .min_idle_connections(4)                                  // Keep connections warm
    .connection_timeout(Duration::from_secs(30))              // Reasonable timeout
    .build("production.db")
    .await?;
```

## Input Validation

### Data Sanitization

#### Validate Input Data

```rust
fn validate_agent_name(name: &str) -> Result<(), ValidationError> {
    if name.is_empty() {
        return Err(ValidationError::EmptyName);
    }
    
    if name.len() > 100 {
        return Err(ValidationError::NameTooLong);
    }
    
    // Prevent control characters
    if name.chars().any(|c| c.is_control()) {
        return Err(ValidationError::InvalidCharacters);
    }
    
    Ok(())
}

// Use validation before storage
async fn store_agent_safely(storage: &SqliteStorage, name: &str) -> Result<(), StorageError> {
    validate_agent_name(name).map_err(|e| StorageError::Other(e.to_string()))?;
    
    let agent = AgentModel::new(name.to_string(), "worker".to_string(), vec![]);
    storage.store_agent(&agent).await?;
    
    Ok(())
}
```

#### JSON Payload Validation

```rust
use serde_json::Value;

fn validate_task_payload(payload: &Value) -> Result<(), ValidationError> {
    // Ensure payload is an object
    if !payload.is_object() {
        return Err(ValidationError::InvalidPayloadType);
    }
    
    // Check payload size
    let serialized = serde_json::to_string(payload)
        .map_err(|_| ValidationError::PayloadSerializationError)?;
    
    if serialized.len() > 1024 * 1024 {  // 1MB limit
        return Err(ValidationError::PayloadTooLarge);
    }
    
    // Validate required fields
    if payload.get("task_type").is_none() {
        return Err(ValidationError::MissingTaskType);
    }
    
    Ok(())
}
```

### Capability Validation

```rust
fn validate_capabilities(capabilities: &[String]) -> Result<(), ValidationError> {
    // Check capability count
    if capabilities.len() > 50 {
        return Err(ValidationError::TooManyCapabilities);
    }
    
    // Validate each capability
    for capability in capabilities {
        if capability.is_empty() {
            return Err(ValidationError::EmptyCapability);
        }
        
        if capability.len() > 100 {
            return Err(ValidationError::CapabilityTooLong);
        }
        
        // Only allow alphanumeric and common separators
        if !capability.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return Err(ValidationError::InvalidCapabilityFormat);
        }
    }
    
    Ok(())
}
```

## Connection Security

### File Permissions

#### SQLite Database Security

```rust
use std::fs;
use std::os::unix::fs::PermissionsExt;

async fn create_secure_database(db_path: &str) -> Result<SqliteStorage, StorageError> {
    // Create database
    let storage = SqliteStorage::new(db_path).await?;
    
    // Set restrictive permissions (owner read/write only)
    fs::set_permissions(db_path, fs::Permissions::from_mode(0o600))
        .map_err(|e| StorageError::Other(format!("Failed to set permissions: {}", e)))?;
    
    Ok(storage)
}
```

### Connection Limits

#### Prevent Resource Exhaustion

```rust
// Configure connection limits to prevent DoS
let storage = SqliteStorage::builder()
    .max_connections(32)                          // Limit concurrent connections
    .connection_timeout(Duration::from_secs(30))  // Timeout slow connections
    .idle_timeout(Duration::from_secs(600))       // Clean up idle connections
    .build("secure.db")
    .await?;
```

### Memory Protection

#### Sensitive Data Handling

```rust
// Use secure memory handling for sensitive data
use zeroize::Zeroize;

struct SecureTaskData {
    secret_key: String,
    payload: serde_json::Value,
}

impl Drop for SecureTaskData {
    fn drop(&mut self) {
        self.secret_key.zeroize();
    }
}

// Clear sensitive data after use
async fn process_secure_task(storage: &SqliteStorage, mut task_data: SecureTaskData) -> Result<(), StorageError> {
    // Process task
    let task = TaskModel::new("secure_task".to_string(), task_data.payload.clone(), TaskPriority::High);
    storage.store_task(&task).await?;
    
    // Sensitive data is automatically cleared when task_data is dropped
    Ok(())
}
```

## Deployment Security

### Environment Configuration

#### Secure Configuration

```rust
use std::env;

fn get_secure_db_path() -> Result<String, SecurityError> {
    // Use environment variable for database path
    env::var("SWARM_DB_PATH")
        .map_err(|_| SecurityError::MissingConfiguration("SWARM_DB_PATH".to_string()))
}

fn get_connection_limits() -> (u32, u32) {
    let max_connections = env::var("SWARM_MAX_CONNECTIONS")
        .unwrap_or_else(|_| "32".to_string())
        .parse()
        .unwrap_or(32);
    
    let min_connections = env::var("SWARM_MIN_CONNECTIONS")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .unwrap_or(4);
    
    (max_connections, min_connections)
}
```

### Logging Security

#### Secure Logging

```rust
use tracing::{info, warn, error};

// Log security events without exposing sensitive data
async fn secure_logging_example(storage: &SqliteStorage, user_id: &str) -> Result<(), StorageError> {
    info!("User {} attempting to access agents", user_id);
    
    match storage.list_agents().await {
        Ok(agents) => {
            info!("Successfully retrieved {} agents for user {}", agents.len(), user_id);
        },
        Err(e) => {
            error!("Failed to retrieve agents for user {}: {}", user_id, e);
            // Don't log sensitive error details
        }
    }
    
    Ok(())
}
```

### Monitoring

#### Security Monitoring

```rust
use std::time::Instant;

// Monitor for suspicious activity
async fn monitor_query_performance(storage: &SqliteStorage, query: &str) -> Result<(), StorageError> {
    let start = Instant::now();
    
    // Execute query
    let result = storage.execute_query(query).await;
    
    let duration = start.elapsed();
    
    // Log slow queries (potential DoS)
    if duration > Duration::from_secs(1) {
        warn!("Slow query detected: {} (took {:?})", query, duration);
    }
    
    result
}
```

## Security Testing

### Injection Testing

#### Comprehensive Injection Tests

```rust
#[tokio::test]
async fn test_comprehensive_injection_protection() {
    let storage = SqliteStorage::new("test.db").await.unwrap();
    
    let injection_payloads = vec![
        // SQL injection attempts
        "'; DROP TABLE agents; --",
        "' OR '1'='1",
        "'; DELETE FROM agents; --",
        "' UNION SELECT * FROM sqlite_master; --",
        "'; INSERT INTO agents VALUES ('evil', 'hacker'); --",
        
        // XSS attempts (should be safely stored)
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        
        // Path traversal attempts
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        
        // NoSQL injection attempts
        "{ \"$ne\": null }",
        "{ \"$where\": \"this.name == 'admin'\" }",
    ];
    
    for payload in injection_payloads {
        // Test query builder safety
        let (query, params) = QueryBuilder::<AgentModel>::new("agents")
            .where_eq("name", payload)
            .build();
        
        // Verify parameterized query
        assert!(query.contains("?"));
        assert!(!query.contains(payload));
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], payload);
        
        // Test storage safety
        let agent = AgentModel::new(payload.to_string(), "worker".to_string(), vec![]);
        assert!(storage.store_agent(&agent).await.is_ok());
        
        // Verify data is stored safely
        let retrieved = storage.get_agent(&agent.id).await.unwrap().unwrap();
        assert_eq!(retrieved.name, payload);
    }
}
```

### Transaction Testing

#### ACID Property Testing

```rust
#[tokio::test]
async fn test_transaction_isolation() {
    let storage = Arc::new(SqliteStorage::new("test.db").await.unwrap());
    
    // Test concurrent transactions
    let mut handles = vec![];
    
    for i in 0..10 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            let mut tx = storage_clone.begin_transaction().await.unwrap();
            
            // Simulate work
            tokio::time::sleep(Duration::from_millis(10)).await;
            
            let agent = AgentModel::new(format!("agent-{}", i), "worker".to_string(), vec![]);
            storage_clone.store_agent(&agent).await.unwrap();
            
            tx.commit().await.unwrap();
        });
        handles.push(handle);
    }
    
    // All transactions should complete successfully
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all agents were stored
    let agents = storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), 10);
}
```

### Concurrency Testing

#### Thread Safety Testing

```rust
#[tokio::test]
async fn test_concurrent_access_safety() {
    let storage = Arc::new(SqliteStorage::new("test.db").await.unwrap());
    
    // Test high concurrency
    let mut handles = vec![];
    
    for i in 0..1000 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            let agent = AgentModel::new(format!("agent-{}", i), "worker".to_string(), vec![]);
            storage_clone.store_agent(&agent).await.unwrap();
        });
        handles.push(handle);
    }
    
    // All operations should complete without errors
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify data integrity
    let agents = storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), 1000);
    
    // Verify no duplicate IDs
    let mut ids = HashSet::new();
    for agent in agents {
        assert!(ids.insert(agent.id), "Duplicate agent ID found");
    }
}
```

## Security Checklist

### ✅ Implementation Checklist

- [ ] Use parameterized queries for all SQL operations
- [ ] Validate all input data before storage
- [ ] Use transactions for multi-step operations
- [ ] Configure appropriate connection limits
- [ ] Set restrictive file permissions on database files
- [ ] Implement proper error handling without information leakage
- [ ] Use secure logging practices
- [ ] Test for SQL injection vulnerabilities
- [ ] Test transaction isolation and rollback
- [ ] Verify thread safety under high concurrency

### ✅ Deployment Checklist

- [ ] Database files have restricted permissions (600 or 640)
- [ ] Connection limits are configured appropriately
- [ ] Environment variables are used for sensitive configuration
- [ ] Logging doesn't expose sensitive data
- [ ] Monitoring is in place for suspicious activity
- [ ] Regular security testing is performed
- [ ] Dependencies are kept up to date
- [ ] Backup procedures are secure

## Next Steps

- [Getting Started](./getting-started.md) - Learn secure usage patterns
- [API Reference](./api-reference.md) - Detailed security features
- [Testing Guide](./testing-guide.md) - Security testing practices
- [Storage Backends](./storage-backends.md) - Backend-specific security

---

*Security is a shared responsibility. Always validate inputs, use transactions appropriately, and test thoroughly!*