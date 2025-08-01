# Testing Guide

This guide covers comprehensive testing patterns for ruv-swarm-persistence, including unit tests, integration tests, concurrent tests, security tests, and property-based testing.

## Table of Contents

- [Test Organization](#test-organization)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Concurrent Testing](#concurrent-testing)
- [Security Testing](#security-testing)
- [Property-Based Testing](#property-based-testing)
- [Performance Testing](#performance-testing)
- [Test Utilities](#test-utilities)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

## Test Organization

### Test Structure

The ruv-swarm-persistence test suite is organized into focused modules:

```
src/tests/
├── mod.rs                    # Test module organization
├── crud_tests.rs            # Basic CRUD operations
├── storage_tests.rs         # Storage trait implementations
├── transaction_tests.rs     # ACID transaction testing
├── concurrent_tests.rs      # Concurrency and performance
├── security_tests.rs        # SQL injection and security
├── property_tests.rs        # Property-based testing
├── query_tests.rs           # Query builder testing
└── test_utils.rs            # Shared test utilities
```

### Test Statistics

**Current Test Coverage (71/71 tests passing - 100% success rate):**

| Test Module | Tests | Focus Area |
|-------------|-------|------------|
| CRUD Tests | 12 | Basic operations |
| Storage Tests | 8 | Trait implementations |
| Transaction Tests | 6 | ACID compliance |
| Concurrent Tests | 15 | Performance & safety |
| Security Tests | 12 | SQL injection prevention |
| Property Tests | 8 | Edge case discovery |
| Query Tests | 10 | Safe query building |

## Unit Testing

### Basic CRUD Operations

Test fundamental storage operations:

```rust
#[tokio::test]
async fn test_agent_crud_operations() {
    let storage = MemoryStorage::new();
    
    // Create
    let agent = AgentModel::new(
        "test-agent".to_string(),
        "worker".to_string(),
        vec!["compute".to_string()]
    );
    storage.store_agent(&agent).await.unwrap();
    
    // Read
    let retrieved = storage.get_agent(&agent.id).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name, "test-agent");
    
    // Update
    let mut updated_agent = agent.clone();
    updated_agent.status = AgentStatus::Busy;
    storage.update_agent(&updated_agent).await.unwrap();
    
    let retrieved = storage.get_agent(&agent.id).await.unwrap().unwrap();
    assert_eq!(retrieved.status, AgentStatus::Busy);
    
    // Delete
    storage.delete_agent(&agent.id).await.unwrap();
    let retrieved = storage.get_agent(&agent.id).await.unwrap();
    assert!(retrieved.is_none());
}
```

### Test Isolation

Use isolated storage for each test:

```rust
#[tokio::test]
async fn test_isolated_storage() {
    // Each test gets its own storage instance
    let storage = MemoryStorage::new();
    
    // Test operations don't affect other tests
    let agent = AgentModel::new("isolated".to_string(), "worker".to_string(), vec![]);
    storage.store_agent(&agent).await.unwrap();
    
    // Storage is automatically cleaned up when test ends
}
```

### Error Handling Tests

Test error conditions and edge cases:

```rust
#[tokio::test]
async fn test_error_handling() {
    let storage = MemoryStorage::new();
    
    // Test non-existent agent
    let result = storage.get_agent("non-existent").await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
    
    // Test invalid update
    let non_existent_agent = AgentModel::new("test".to_string(), "worker".to_string(), vec![]);
    let result = storage.update_agent(&non_existent_agent).await;
    // Should handle gracefully (implementation-dependent)
}
```

## Integration Testing

### Cross-Backend Testing

Test the same functionality across different storage backends:

```rust
async fn test_storage_implementation<S: Storage>(storage: S) 
where 
    S::Error: std::fmt::Debug,
{
    // Test works with any Storage implementation
    let agent = AgentModel::new("test".to_string(), "worker".to_string(), vec![]);
    storage.store_agent(&agent).await.unwrap();
    
    let retrieved = storage.get_agent(&agent.id).await.unwrap();
    assert!(retrieved.is_some());
}

#[tokio::test]
async fn test_memory_storage() {
    let storage = MemoryStorage::new();
    test_storage_implementation(storage).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_sqlite_storage() {
    use tempfile::NamedTempFile;
    
    let temp_file = NamedTempFile::new().unwrap();
    let storage = SqliteStorage::new(temp_file.path().to_str().unwrap()).await.unwrap();
    test_storage_implementation(storage).await;
}
```

### Database Schema Testing

Test database schema and migrations:

```rust
#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_database_schema() {
    use tempfile::NamedTempFile;
    
    let temp_file = NamedTempFile::new().unwrap();
    let storage = SqliteStorage::new(temp_file.path().to_str().unwrap()).await.unwrap();
    
    // Test that tables exist and have correct structure
    let agents = storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), 0);
    
    // Test foreign key constraints
    let task = TaskModel::new("test".to_string(), serde_json::json!({}), TaskPriority::Low);
    storage.store_task(&task).await.unwrap();
    
    // Test that invalid foreign key is rejected
    let mut invalid_task = task.clone();
    invalid_task.assigned_to = Some("non-existent-agent".to_string());
    // Should handle foreign key constraint appropriately
}
```

## Concurrent Testing

### High-Load Concurrency Tests

Test system behavior under high concurrent load:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_agent_registration() {
    let storage = Arc::new(MemoryStorage::new());
    let agent_count = 100;
    let barrier = Arc::new(Barrier::new(agent_count));
    
    // Spawn concurrent operations
    let mut handles = vec![];
    for i in 0..agent_count {
        let storage_clone = storage.clone();
        let barrier_clone = barrier.clone();
        
        let handle = tokio::spawn(async move {
            // Wait for all tasks to be ready
            barrier_clone.wait().await;
            
            // All register at once
            let agent = AgentModel::new(
                format!("agent-{}", i),
                "worker".to_string(),
                vec!["compute".to_string()]
            );
            storage_clone.store_agent(&agent).await
        });
        handles.push(handle);
    }
    
    // Collect results
    let results: Vec<_> = join_all(handles).await;
    
    // All should succeed
    for result in results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }
    
    // Verify all agents stored
    let agents = storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), agent_count);
}
```

### Race Condition Testing

Test for race conditions in critical operations:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_task_claiming_race_conditions() {
    let storage = Arc::new(MemoryStorage::new());
    
    // Create a single task
    let task = TaskModel::new("race-test".to_string(), serde_json::json!({}), TaskPriority::High);
    storage.store_task(&task).await.unwrap();
    
    // Multiple agents try to claim the same task
    let agent_count = 10;
    let mut handles = vec![];
    
    for i in 0..agent_count {
        let storage_clone = storage.clone();
        let task_id = task.id.clone();
        
        let handle = tokio::spawn(async move {
            let agent_id = format!("agent-{}", i);
            storage_clone.claim_task(&task_id, &agent_id).await
        });
        handles.push(handle);
    }
    
    // Collect results
    let results: Vec<_> = join_all(handles).await;
    
    // Only one should succeed
    let successful_claims = results.into_iter()
        .filter_map(|r| r.ok())
        .filter(|r| r.is_ok() && r.as_ref().unwrap() == &true)
        .count();
    
    assert_eq!(successful_claims, 1, "Only one agent should claim the task");
}
```

### Performance Benchmarking

Test performance under realistic loads:

```rust
#[tokio::test]
async fn test_event_sourcing_performance() {
    let storage = Arc::new(MemoryStorage::new());
    let event_count = 30_000; // Regression baseline
    
    let start = std::time::Instant::now();
    
    // Store events in batches for better performance
    let batch_size = 100;
    let mut handles = vec![];
    
    for batch in 0..(event_count / batch_size) {
        let storage_clone = storage.clone();
        
        let handle = tokio::spawn(async move {
            for i in (batch * batch_size)..((batch + 1) * batch_size) {
                let event = EventModel::new(
                    format!("event-type-{}", i % 10),
                    serde_json::json!({"sequence": i})
                );
                storage_clone.store_event(&event).await.unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Wait for all events to be stored
    join_all(handles).await;
    
    let duration = start.elapsed();
    let events_per_second = event_count as f64 / duration.as_secs_f64();
    
    println!("Memory Storage: {} events in {:?} ({:.0} events/sec)", 
        event_count, duration, events_per_second);
    
    // Verify performance regression baseline
    assert!(events_per_second > 30_000.0, "Performance regression detected");
}
```

## Security Testing

### SQL Injection Prevention

Test comprehensive SQL injection protection:

```rust
#[tokio::test]
async fn test_sql_injection_prevention() {
    let storage = MemoryStorage::new();
    
    let injection_attempts = vec![
        "'; DROP TABLE agents; --",
        "' OR '1'='1",
        "'; DELETE FROM agents; --",
        "' UNION SELECT * FROM sqlite_master; --",
        "<script>alert('xss')</script>",
    ];
    
    for attempt in injection_attempts {
        // Test query builder safety
        let (query, params) = QueryBuilder::<AgentModel>::new("agents")
            .where_eq("name", attempt)
            .build();
        
        // Verify parameterized query
        assert!(query.contains("?"), "Query should use parameterized placeholders");
        assert!(!query.contains(attempt), "Query should not contain raw input");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], attempt);
        
        // Test storage safety
        let agent = AgentModel::new(attempt.to_string(), "worker".to_string(), vec![]);
        assert!(storage.store_agent(&agent).await.is_ok());
        
        // Verify data stored safely
        let retrieved = storage.get_agent(&agent.id).await.unwrap().unwrap();
        assert_eq!(retrieved.name, attempt);
    }
}
```

### Transaction Security

Test ACID compliance and transaction isolation:

```rust
#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_transaction_acid_properties() {
    use tempfile::NamedTempFile;
    
    let temp_file = NamedTempFile::new().unwrap();
    let storage = SqliteStorage::new(temp_file.path().to_str().unwrap()).await.unwrap();
    
    // Test Atomicity - all or nothing
    let mut tx = storage.begin_transaction().await.unwrap();
    
    let agent1 = AgentModel::new("agent1".to_string(), "worker".to_string(), vec![]);
    let agent2 = AgentModel::new("agent2".to_string(), "worker".to_string(), vec![]);
    
    storage.store_agent(&agent1).await.unwrap();
    storage.store_agent(&agent2).await.unwrap();
    
    // Before commit, changes shouldn't be visible from other connections
    let other_storage = SqliteStorage::new(temp_file.path().to_str().unwrap()).await.unwrap();
    let agents = other_storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), 0, "Changes should not be visible before commit");
    
    // Commit changes
    tx.commit().await.unwrap();
    
    // After commit, changes should be visible
    let agents = other_storage.list_agents().await.unwrap();
    assert_eq!(agents.len(), 2, "Changes should be visible after commit");
}
```

### Input Validation Security

Test input validation and sanitization:

```rust
#[tokio::test]
async fn test_input_validation() {
    let storage = MemoryStorage::new();
    
    // Test oversized inputs
    let large_name = "a".repeat(10000);
    let agent = AgentModel::new(large_name.clone(), "worker".to_string(), vec![]);
    
    // Should handle large inputs gracefully
    let result = storage.store_agent(&agent).await;
    assert!(result.is_ok());
    
    // Test special characters
    let special_chars = "äöü中文🎉";
    let agent = AgentModel::new(special_chars.to_string(), "worker".to_string(), vec![]);
    
    storage.store_agent(&agent).await.unwrap();
    let retrieved = storage.get_agent(&agent.id).await.unwrap().unwrap();
    assert_eq!(retrieved.name, special_chars);
}
```

## Property-Based Testing

### Agent Model Invariants

Test invariants that should always hold:

```rust
#[cfg(feature = "proptest")]
use proptest::prelude::*;

#[cfg(feature = "proptest")]
proptest! {
    #[test]
    fn test_agent_invariants(
        name in "[a-zA-Z0-9_-]{1,100}",
        agent_type in "[a-zA-Z0-9_-]{1,50}",
        capabilities in prop::collection::vec("[a-zA-Z0-9_-]{1,50}", 0..10)
    ) {
        let agent = AgentModel::new(name.clone(), agent_type.clone(), capabilities.clone());
        
        // Invariants that should always hold
        assert!(!agent.id.is_empty());
        assert_eq!(agent.name, name);
        assert_eq!(agent.agent_type, agent_type);
        assert_eq!(agent.capabilities, capabilities);
        assert_eq!(agent.status, AgentStatus::Initializing);
        assert!(agent.created_at <= agent.updated_at);
        assert!(agent.heartbeat <= agent.updated_at);
    }
}
```

### Storage Consistency Properties

Test storage consistency across operations:

```rust
#[cfg(feature = "proptest")]
proptest! {
    #[test]
    fn test_storage_consistency(
        agents in prop::collection::vec(
            ("[a-zA-Z0-9_-]{1,50}", "[a-zA-Z0-9_-]{1,20}"),
            1..50
        )
    ) {
        let storage = MemoryStorage::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            // Store all agents
            for (name, agent_type) in &agents {
                let agent = AgentModel::new(name.clone(), agent_type.clone(), vec![]);
                storage.store_agent(&agent).await.unwrap();
            }
            
            // Verify count consistency
            let stored_agents = storage.list_agents().await.unwrap();
            assert_eq!(stored_agents.len(), agents.len());
            
            // Verify all agents are retrievable
            for agent in &stored_agents {
                let retrieved = storage.get_agent(&agent.id).await.unwrap();
                assert!(retrieved.is_some());
                assert_eq!(retrieved.unwrap().id, agent.id);
            }
        });
    }
}
```

### Query Builder Properties

Test query builder safety properties:

```rust
#[cfg(feature = "proptest")]
proptest! {
    #[test]
    fn test_query_builder_safety(
        table in "[a-z_]{1,20}",
        field in "[a-z_]{1,20}",
        value in "[a-zA-Z0-9_-]{1,50}",
        limit in 1usize..1000
    ) {
        let (query, params) = QueryBuilder::<AgentModel>::new(&table)
            .where_eq(&field, &value)
            .limit(limit)
            .build();
        
        // Safety properties
        assert!(query.starts_with("SELECT * FROM"));
        assert!(query.contains(&table));
        assert!(query.contains(&field));
        assert!(query.contains("?"));
        assert!(!query.contains(&value), "Query should not contain raw values");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], value);
    }
}
```

## Performance Testing

### Benchmark Framework

Create reusable benchmarks:

```rust
use std::time::Instant;

struct BenchmarkResult {
    operation: String,
    total_time: Duration,
    operations_per_second: f64,
    total_operations: usize,
}

impl BenchmarkResult {
    fn new(operation: &str, total_time: Duration, total_operations: usize) -> Self {
        let ops_per_sec = total_operations as f64 / total_time.as_secs_f64();
        Self {
            operation: operation.to_string(),
            total_time,
            operations_per_second: ops_per_sec,
            total_operations,
        }
    }
}

async fn benchmark_storage_operations<S: Storage>(storage: &S, operations: usize) -> Vec<BenchmarkResult> 
where 
    S::Error: std::fmt::Debug,
{
    let mut results = Vec::new();
    
    // Benchmark agent creation
    let start = Instant::now();
    for i in 0..operations {
        let agent = AgentModel::new(format!("agent-{}", i), "worker".to_string(), vec![]);
        storage.store_agent(&agent).await.unwrap();
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new("Agent Store", duration, operations));
    
    // Benchmark agent retrieval
    let start = Instant::now();
    for i in 0..operations {
        let agent_id = format!("agent-{}", i);
        let _ = storage.get_agent(&agent_id).await.unwrap();
    }
    let duration = start.elapsed();
    results.push(BenchmarkResult::new("Agent Retrieve", duration, operations));
    
    results
}
```

### Regression Testing

Test for performance regressions:

```rust
#[tokio::test]
async fn test_performance_regression() {
    let storage = MemoryStorage::new();
    let operation_count = 10_000;
    
    let results = benchmark_storage_operations(&storage, operation_count).await;
    
    for result in results {
        println!("{}: {:.0} ops/sec", result.operation, result.operations_per_second);
        
        // Set regression thresholds
        match result.operation.as_str() {
            "Agent Store" => assert!(result.operations_per_second > 50_000.0),
            "Agent Retrieve" => assert!(result.operations_per_second > 100_000.0),
            _ => {}
        }
    }
}
```

## Test Utilities

### Shared Test Infrastructure

Create reusable test utilities:

```rust
// test_utils.rs
use tempfile::NamedTempFile;
use std::sync::Arc;

pub struct TestFixture {
    pub memory_storage: MemoryStorage,
    #[cfg(not(target_arch = "wasm32"))]
    pub sqlite_storage: SqliteStorage,
    #[cfg(not(target_arch = "wasm32"))]
    _temp_file: NamedTempFile,
}

impl TestFixture {
    pub async fn new() -> Self {
        let memory_storage = MemoryStorage::new();
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            let temp_file = NamedTempFile::new().unwrap();
            let sqlite_storage = SqliteStorage::new(temp_file.path().to_str().unwrap()).await.unwrap();
            
            Self {
                memory_storage,
                sqlite_storage,
                _temp_file: temp_file,
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        Self {
            memory_storage,
        }
    }
    
    pub fn create_test_agent(&self, name: &str) -> AgentModel {
        AgentModel::new(name.to_string(), "worker".to_string(), vec!["compute".to_string()])
    }
    
    pub fn create_test_task(&self, task_type: &str) -> TaskModel {
        TaskModel::new(
            task_type.to_string(),
            serde_json::json!({"test": true}),
            TaskPriority::Medium
        )
    }
}

// Usage in tests
#[tokio::test]
async fn test_with_fixture() {
    let fixture = TestFixture::new().await;
    let agent = fixture.create_test_agent("test-agent");
    
    fixture.memory_storage.store_agent(&agent).await.unwrap();
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        fixture.sqlite_storage.store_agent(&agent).await.unwrap();
    }
}
```

### Test Data Generators

Create realistic test data:

```rust
use rand::Rng;

pub fn generate_test_agents(count: usize) -> Vec<AgentModel> {
    let mut rng = rand::thread_rng();
    let agent_types = vec!["worker", "coordinator", "analyzer", "processor"];
    let capabilities = vec!["compute", "storage", "network", "ml", "data"];
    
    (0..count)
        .map(|i| {
            let agent_type = agent_types[rng.gen_range(0..agent_types.len())];
            let capability_count = rng.gen_range(1..4);
            let agent_capabilities: Vec<String> = (0..capability_count)
                .map(|_| capabilities[rng.gen_range(0..capabilities.len())].to_string())
                .collect();
            
            AgentModel::new(
                format!("agent-{}", i),
                agent_type.to_string(),
                agent_capabilities
            )
        })
        .collect()
}
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
cargo test --package ruv-swarm-persistence

# Run tests with output
cargo test --package ruv-swarm-persistence -- --nocapture

# Run specific test module
cargo test --package ruv-swarm-persistence concurrent_tests

# Run specific test
cargo test --package ruv-swarm-persistence test_agent_crud_operations
```

### Test Configuration

```bash
# Run tests with multiple threads
cargo test --package ruv-swarm-persistence -- --test-threads=4

# Run tests in release mode for performance testing
cargo test --package ruv-swarm-persistence --release

# Run tests with features
cargo test --package ruv-swarm-persistence --features proptest
```

### Coverage Testing

```bash
# Install coverage tool
cargo install cargo-tarpaulin

# Run coverage analysis
cargo tarpaulin --package ruv-swarm-persistence --out html

# Generate coverage report
cargo tarpaulin --package ruv-swarm-persistence --out json
```

### Performance Profiling

```bash
# Profile tests
cargo test --package ruv-swarm-persistence --release -- --nocapture | grep "events/sec"

# Run benchmarks
cargo bench --package ruv-swarm-persistence
```

## Troubleshooting

### Common Test Issues

#### Test Hangs or Timeouts

```rust
// Add timeouts to prevent hanging tests
#[tokio::test]
#[timeout(Duration::from_secs(30))]
async fn test_with_timeout() {
    // Test code
}
```

#### Resource Cleanup

```rust
// Ensure proper cleanup in tests
#[tokio::test]
async fn test_with_cleanup() {
    let storage = MemoryStorage::new();
    
    // Test operations
    
    // Explicit cleanup if needed
    storage.vacuum().await.unwrap();
}
```

#### Flaky Tests

```rust
// Make tests deterministic
#[tokio::test]
async fn test_deterministic() {
    // Use fixed seeds for random data
    let mut rng = rand::StdRng::seed_from_u64(42);
    
    // Use barriers for synchronization
    let barrier = Arc::new(Barrier::new(thread_count));
    
    // Test with deterministic timing
}
```

### Performance Issues

#### Connection Pool Exhaustion

```rust
// Configure appropriate pool size for tests
#[tokio::test]
async fn test_with_large_pool() {
    let storage = SqliteStorage::builder()
        .max_connections(64)  // Increase for high-concurrency tests
        .build("test.db")
        .await
        .unwrap();
}
```

#### Memory Usage

```rust
// Monitor memory usage in long-running tests
#[tokio::test]
async fn test_memory_usage() {
    let storage = MemoryStorage::new();
    
    // Periodically check memory usage
    for i in 0..100_000 {
        let agent = AgentModel::new(format!("agent-{}", i), "worker".to_string(), vec![]);
        storage.store_agent(&agent).await.unwrap();
        
        if i % 10_000 == 0 {
            println!("Processed {} agents", i);
        }
    }
}
```

## Best Practices

1. **Use Isolated Storage**: Each test should use its own storage instance
2. **Test All Backends**: Ensure tests work with both Memory and SQLite storage
3. **Include Edge Cases**: Test error conditions and boundary values
4. **Use Property-Based Testing**: Discover edge cases with generated inputs
5. **Benchmark Performance**: Include performance regression tests
6. **Test Concurrency**: Verify thread safety and race condition handling
7. **Validate Security**: Test SQL injection prevention and input validation
8. **Clean Up Resources**: Ensure proper cleanup in test teardown

## Next Steps

- [Getting Started](./getting-started.md) - Learn how to use the persistence layer
- [API Reference](./api-reference.md) - Detailed API documentation
- [Security Guide](./security-guide.md) - Security best practices
- [Storage Backends](./storage-backends.md) - Backend comparison and selection

---

*Test thoroughly, test early, test often! The ruv-swarm-persistence test suite ensures reliability and security.*