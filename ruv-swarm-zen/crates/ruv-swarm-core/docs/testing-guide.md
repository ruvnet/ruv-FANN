# Testing Guide - ruv-swarm-core

## Quick Reference

```bash
# Run all tests
cargo test --package ruv-swarm-core

# Run tests with output
cargo test --package ruv-swarm-core -- --nocapture

# Run specific test module
cargo test --package ruv-swarm-core agent_tests

# Check test coverage
cargo tarpaulin --package ruv-swarm-core

# Run clippy linting
cargo clippy --package ruv-swarm-core --all-targets

# Format code
cargo fmt --package ruv-swarm-core
```

## Test Structure Overview

ruv-swarm-core has **169 tests** across **12 modules** with **87.6% coverage**:

```
src/tests/
├── mod.rs                     # Test module coordination
├── agent_tests.rs            # 30 tests - Agent functionality
├── swarm_tests.rs            # 21 tests - Basic swarm operations
├── topology_tests.rs         # 20 tests - Network topologies
├── task_tests.rs             # 18 tests - Task management
├── async_swarm_tests.rs      # 18 tests - AsyncSwarm functionality
├── swarm_trait_tests.rs      # 16 tests - Trait abstractions
├── error_handling_tests.rs   # 15 tests - Error scenarios
├── swarm_integration_tests.rs # 8 tests - End-to-end workflows
├── agent_message_tests.rs    # 7 tests - Message passing
├── custom_payload_tests.rs   # 6 tests - Custom payloads
└── agent_trait_tests.rs      # 6 tests - Agent trait implementation
```

## Running Tests

### Basic Test Execution

```bash
# All tests (recommended)
cargo test --package ruv-swarm-core

# With verbose output
cargo test --package ruv-swarm-core -- --nocapture --test-threads=1

# Only library tests (no examples)
cargo test --package ruv-swarm-core --lib

# Include examples and benchmarks
cargo test --package ruv-swarm-core --all-targets
```

### Specific Test Categories

```bash
# Core functionality tests
cargo test --package ruv-swarm-core agent_tests
cargo test --package ruv-swarm-core swarm_tests
cargo test --package ruv-swarm-core task_tests

# Advanced features
cargo test --package ruv-swarm-core async_swarm_tests
cargo test --package ruv-swarm-core topology_tests

# Edge cases and error handling
cargo test --package ruv-swarm-core error_handling_tests
cargo test --package ruv-swarm-core swarm_integration_tests
```

### Individual Tests

```bash
# Run a specific test
cargo test --package ruv-swarm-core test_swarm_creation

# Run tests matching a pattern
cargo test --package ruv-swarm-core lifecycle

# Run ignored tests
cargo test --package ruv-swarm-core -- --ignored
```

## Test Coverage Analysis

### Current Coverage: 87.6%

| Module | Coverage | Lines | Grade |
|--------|----------|-------|-------|
| error.rs | 100% | 7/7 | ✅ Perfect |
| lib.rs | 100% | 6/6 | ✅ Perfect |
| swarm_trait.rs | 100% | 3/3 | ✅ Perfect |
| topology.rs | 98.8% | 83/84 | 🟢 Excellent |
| swarm.rs | 92.2% | 83/90 | 🟢 Excellent |
| task.rs | 90.9% | 40/44 | 🟢 Very Good |
| async_swarm.rs | 81.9% | 131/160 | 🟡 Good |
| agent.rs | 77.8% | 63/81 | 🟡 Good |

### Generate Coverage Report

```bash
# Install tarpaulin if needed
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --package ruv-swarm-core --out Html

# Open coverage report
open tarpaulin-report.html
```

## Writing Tests

### Test Structure Template

```rust
// src/tests/your_module_tests.rs

use crate::{
    agent::DynamicAgent,
    swarm::{Swarm, SwarmConfig},
    task::Task,
    error::SwarmError,
};

#[test]
fn test_basic_functionality() {
    // Arrange
    let mut swarm = Swarm::new(SwarmConfig::default());
    let agent = DynamicAgent::new("test-agent", vec!["compute".to_string()]);
    
    // Act
    let result = swarm.register_agent(agent);
    
    // Assert
    assert!(result.is_ok());
    assert_eq!(swarm.metrics().total_agents, 1);
}

#[tokio::test]
async fn test_async_functionality() {
    // Arrange
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Act
    let result = swarm.start_all_agents().await;
    
    // Assert
    assert!(result.is_ok());
}

#[test]
fn test_error_conditions() {
    // Arrange
    let mut swarm = Swarm::new(SwarmConfig { max_agents: 0, ..Default::default() });
    let agent = DynamicAgent::new("agent", vec![]);
    
    // Act
    let result = swarm.register_agent(agent);
    
    // Assert
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), SwarmError::ResourceExhausted { .. }));
}
```

### Testing Patterns

#### 1. **Agent Testing**
```rust
#[tokio::test]
async fn test_agent_lifecycle() {
    let mut agent = DynamicAgent::new("test", vec!["compute".to_string()]);
    
    // Test initial state
    assert_eq!(agent.status(), AgentStatus::Idle);
    
    // Test start
    agent.start().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Running);
    
    // Test shutdown
    agent.shutdown().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Offline);
}
```

#### 2. **Task Testing**
```rust
#[test]
fn test_task_builder() {
    let task = Task::new("test-task", "computation")
        .with_priority(TaskPriority::High)
        .require_capability("compute")
        .with_timeout(5000);
    
    assert_eq!(task.priority, TaskPriority::High);
    assert!(task.required_capabilities.contains(&"compute".to_string()));
    assert_eq!(task.timeout_ms, Some(5000));
}
```

#### 3. **Error Testing**
```rust
#[test]
fn test_error_handling() {
    let error = SwarmError::AgentNotFound { id: "missing".to_string() };
    
    // Test error properties
    assert!(!error.is_retriable());
    assert!(format!("{}", error).contains("missing"));
    
    // Test error creation
    let custom_error = SwarmError::custom("Test message");
    assert!(matches!(custom_error, SwarmError::Custom(_)));
}
```

#### 4. **Integration Testing**
```rust
#[tokio::test]
async fn test_end_to_end_workflow() {
    // Setup
    let mut swarm = Swarm::new(SwarmConfig::default());
    let agent = DynamicAgent::new("worker", vec!["process".to_string()]);
    swarm.register_agent(agent).unwrap();
    
    // Create and submit task
    let task = Task::new("job-1", "process").require_capability("process");
    swarm.submit_task(task).unwrap();
    
    // Execute workflow
    swarm.start_all_agents().await.unwrap();
    let assignments = swarm.distribute_tasks().await.unwrap();
    
    // Verify results
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].1, "worker");
    assert_eq!(swarm.task_queue_size(), 0);
    assert_eq!(swarm.assigned_tasks_count(), 1);
}
```

### MockAgent for Testing

```rust
use crate::agent::MockAgent;

#[tokio::test]
async fn test_with_mock_agent() {
    let mut mock = MockAgent::new("mock")
        .with_capabilities(vec!["test".to_string()])
        .with_process_result(Ok(crate::task::TaskResult::success("Done")));
    
    // Test mock behavior
    assert!(mock.has_capability("test"));
    
    let result = mock.process(Task::new("test", "work")).await;
    assert!(result.is_ok());
}
```

## Test Data and Fixtures

### Common Test Data

```rust
// Helper functions for consistent test data
pub fn create_test_agent(id: &str) -> DynamicAgent {
    DynamicAgent::new(id, vec!["compute".to_string(), "analyze".to_string()])
}

pub fn create_test_task(id: &str) -> Task {
    Task::new(id, "test-task")
        .require_capability("compute")
        .with_priority(TaskPriority::Normal)
}

pub fn create_test_swarm() -> Swarm {
    Swarm::new(SwarmConfig {
        max_agents: 10,
        topology_type: TopologyType::Mesh,
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    })
}
```

### Async Test Helpers

```rust
// Helper for async swarm testing
pub async fn setup_async_swarm_with_agents(count: usize) -> AsyncSwarm {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    for i in 0..count {
        let agent = DynamicAgent::new(
            format!("agent-{}", i), 
            vec!["compute".to_string()]
        );
        swarm.register_agent(agent).await.unwrap();
    }
    
    swarm
}
```

## Performance Testing

### Benchmark Tests

```rust
#[ignore] // Run with --ignored flag
#[tokio::test]
async fn benchmark_task_distribution() {
    let swarm = setup_async_swarm_with_agents(100).await;
    
    // Submit many tasks
    for i in 0..1000 {
        let task = Task::new(format!("task-{}", i), "compute")
            .require_capability("compute");
        swarm.submit_task(task).await.unwrap();
    }
    
    // Measure distribution time
    let start = std::time::Instant::now();
    let assignments = swarm.distribute_tasks().await.unwrap();
    let duration = start.elapsed();
    
    println!("Distributed {} tasks in {:?}", assignments.len(), duration);
    assert!(duration.as_millis() < 1000); // Should be under 1 second
}
```

### Load Testing

```rust
#[ignore]
#[tokio::test]
async fn load_test_concurrent_operations() {
    let swarm = Arc::new(setup_async_swarm_with_agents(50).await);
    
    // Spawn multiple concurrent operations
    let mut handles = vec![];
    for i in 0..100 {
        let swarm_clone = swarm.clone();
        let handle = tokio::spawn(async move {
            let task = Task::new(format!("load-task-{}", i), "compute")
                .require_capability("compute");
            swarm_clone.submit_task(task).await
        });
        handles.push(handle);
    }
    
    // Wait for all operations
    for handle in handles {
        handle.await.unwrap().unwrap();
    }
    
    assert_eq!(swarm.task_queue_size().await, 100);
}
```

## Debugging Tests

### Debug Output

```rust
#[tokio::test]
async fn debug_test_with_output() {
    // Enable detailed logging for this test
    let _ = env_logger::try_init();
    
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Add debug prints
    println!("Created swarm with {} agents", swarm.metrics().total_agents);
    
    let agent = DynamicAgent::new("debug-agent", vec!["debug".to_string()]);
    swarm.register_agent(agent).unwrap();
    
    println!("Agent registered, metrics: {:?}", swarm.metrics());
    
    // Run with: cargo test debug_test_with_output -- --nocapture
}
```

### Test-only Code

```rust
#[cfg(test)]
impl Swarm {
    // Test-only methods
    pub fn test_get_internal_state(&self) -> &HashMap<AgentId, DynamicAgent> {
        &self.agents
    }
    
    pub fn test_set_agent_status(&mut self, agent_id: &str, status: AgentStatus) {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.set_status(status);
        }
    }
}
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Run tests
        run: cargo test --package ruv-swarm-core --all-targets
      
      - name: Check coverage
        run: |
          cargo install cargo-tarpaulin
          cargo tarpaulin --package ruv-swarm-core --out Xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Best Practices

### 1. **Test Organization**
- Group related tests in the same module
- Use descriptive test names that explain what is being tested
- Test one specific behavior per test function
- Use `#[ignore]` for expensive performance tests

### 2. **Test Data**
- Create helper functions for common test data
- Use meaningful test data that reflects real usage
- Avoid hardcoded values when possible

### 3. **Async Testing**
- Always use `#[tokio::test]` for async tests
- Test both success and failure paths
- Use `tokio::timeout` for tests that might hang

### 4. **Error Testing**
- Test all error conditions
- Verify error messages and types
- Test error recovery scenarios

### 5. **Performance**
- Mark performance tests with `#[ignore]`
- Set reasonable performance expectations
- Test with realistic data sizes

## Contributing Tests

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Ensure good coverage** (aim for >85%)
3. **Test edge cases** and error conditions
4. **Add integration tests** for new features
5. **Update this guide** if you add new testing patterns

## Troubleshooting

### Common Issues

#### Tests hanging
```bash
# Run with timeout and single thread
cargo test --package ruv-swarm-core -- --test-threads=1 --timeout=60
```

#### Flaky async tests
```rust
// Add timeout to prevent hanging
#[tokio::test]
async fn test_with_timeout() {
    tokio::time::timeout(
        Duration::from_secs(5),
        actual_test_logic()
    ).await.unwrap();
}
```

#### Memory leaks in tests
```rust
// Ensure proper cleanup
#[tokio::test]
async fn test_with_cleanup() {
    let swarm = AsyncSwarm::new(config);
    
    // Test logic here
    
    // Explicit cleanup
    swarm.shutdown_all_agents().await.unwrap();
}
```

Remember: **Good tests are documentation** - they show how the code should be used and what behavior is expected!