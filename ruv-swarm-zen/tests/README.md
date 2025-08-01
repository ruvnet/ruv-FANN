# ruv-swarm Integration Tests

This directory contains comprehensive integration tests for the ruv-swarm framework, including end-to-end tests for Claude Code integration, stream parsing, model training, and performance validation.

## Test Structure

### Main Integration Test Suite (`integration_test.rs`)

The main integration test file includes the following test scenarios:

1. **End-to-End Claude Code SWE-Bench Test** (`test_end_to_end_claude_code_swe_bench`)
   - Tests running Claude Code on SWE-Bench instances
   - Validates task orchestration and result collection
   - Verifies token usage and execution metrics

2. **Stream Parsing and Metrics Collection** (`test_stream_parsing_and_metrics_collection`)
   - Tests parsing of Claude Code stream-json output
   - Validates metrics collection (tokens, execution time, tool calls)
   - Ensures proper event handling and aggregation

3. **Model Training and Evaluation** (`test_model_training_and_evaluation`)
   - Tests distributed neural network training
   - Validates model accuracy and loss metrics
   - Tests model evaluation on test data

4. **Performance Improvement Validation** (`test_performance_improvement_validation`)
   - Compares baseline vs. optimized execution
   - Validates token reduction (target: >30%)
   - Validates execution time reduction (target: >40%)

5. **Fibonacci Example** (`test_fibonacci_example`)
   - Simple example demonstrating parallel computation
   - Tests code generation for Fibonacci function
   - Validates parallel speedup metrics

6. **Swarm Performance Scaling** (`test_swarm_performance_scaling`)
   - Tests performance with different swarm sizes (1, 2, 4, 8 agents)
   - Validates scaling efficiency
   - Measures parallel execution overhead

7. **Persistence and Recovery** (`test_persistence_and_recovery`)
   - Tests swarm state persistence to SQLite
   - Validates state recovery after restart
   - Ensures task history is preserved

8. **Error Handling and Retry** (`test_error_handling_and_retry`)
   - Tests retry logic for failing tasks
   - Validates retry count limits
   - Ensures proper error propagation

## Running the Tests

### Run All Integration Tests
```bash
cargo test --test integration_test
```

### Run Specific Test
```bash
cargo test --test integration_test test_fibonacci_example
```

### Run with Output
```bash
cargo test --test integration_test -- --nocapture
```

### Run Tests Sequentially (for debugging)
```bash
cargo test --test integration_test -- --test-threads=1
```

### Using the Test Script
```bash
./scripts/run-integration-tests.sh
```

## Test Data and Helpers

### Claude Stream Parser
The tests include a custom `ClaudeStreamParser` that processes stream-json events:
- Tool calls with timing information
- Function results with duration metrics
- Assistant messages with token counts
- Error handling and recovery

### Performance Metrics
Tests collect and validate the following metrics:
- Total tokens used
- Execution time (milliseconds)
- Tool call patterns and frequency
- Success rates
- Memory usage

### Test Helpers
- `create_test_swarm_with_config()`: Creates swarm with specific topology
- `generate_training_data()`: Generates synthetic data for ML tests
- `FibonacciTask`: Custom task payload for Fibonacci calculations

## Benchmarks

The integration tests include benchmarks for:
- Stream parsing performance
- Task distribution overhead

Run benchmarks with:
```bash
cargo bench --test integration_test
```

## Expected Results

### Performance Targets
- **Token Reduction**: >30% compared to baseline
- **Time Reduction**: >40% compared to baseline
- **Model Accuracy**: >95% on test data
- **Scaling Efficiency**: >70% for 2x agents, >50% for 4x agents

### Fibonacci Example
- Fibonacci(40) = 102,334,155
- Parallel speedup: >1.5x

## Troubleshooting

### Common Issues

1. **Test Timeout**: Increase timeout in `SwarmConfig` or test attributes
2. **SQLite Errors**: Ensure write permissions for `/tmp` directory
3. **Agent Spawn Failures**: Check max_agents limit in configuration

### Debug Mode
Enable debug logging:
```bash
RUST_LOG=debug cargo test --test integration_test
```

## Adding New Tests

To add new integration tests:

1. Add test function to `integration_test.rs`
2. Use existing helpers or create new ones
3. Follow naming convention: `test_<feature>_<scenario>`
4. Update this README with test description
5. Add to `run-integration-tests.sh` if needed

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:
```yaml
- name: Run Integration Tests
  run: cargo test --test integration_test --release
```

For faster CI runs, exclude long-running tests:
```bash
cargo test --test integration_test -- --skip scaling
```