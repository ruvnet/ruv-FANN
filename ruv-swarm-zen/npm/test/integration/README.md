# ruv-swarm Integration Test Suite

Comprehensive end-to-end integration tests for the ruv-swarm system, validating full system functionality, resilience, performance, and feature integration.

## 🏗️ Test Architecture

### Test Structure
```
test/integration/
├── scenarios/
│   ├── lifecycle/          # Full workflow tests
│   ├── resilience/         # Error recovery tests  
│   ├── performance/        # Load and stress tests
│   └── cross-feature/      # Feature integration tests
├── run-integration-tests.js # Test runner
├── integration-test-config.js # Configuration
└── README.md
```

### Test Categories

#### 1. Lifecycle Tests (`scenarios/lifecycle/`)
**Purpose**: Validate complete agent and swarm lifecycle from initialization to shutdown.

**Key Test Scenarios**:
- ✅ Full workflow from spawn to completion
- ✅ Agent communication throughout lifecycle  
- ✅ State persistence and restoration
- ✅ Neural integration lifecycle
- ✅ Memory management across lifecycle
- ✅ Training and learning cycles

**Critical Success Metrics**:
- All agents spawn successfully
- Tasks complete with >95% success rate
- State persists correctly across restarts
- Neural agents learn and improve
- Memory is managed without leaks

#### 2. Resilience Tests (`scenarios/resilience/`)
**Purpose**: Verify system recovery from various failure scenarios.

**Key Test Scenarios**:
- 🛡️ Component failure recovery
- 🌊 Cascading failure protection
- 🔄 Circuit breaker patterns
- 💾 Database connection failures
- 🌐 Network partition handling
- 📈 Resource leak prevention
- ⚡ Graceful degradation under load

**Critical Success Metrics**:
- <5 second recovery from failures
- No data loss during failures
- Circuit breakers activate properly
- Network partitions heal automatically
- Memory usage remains stable

#### 3. Performance Tests (`scenarios/performance/`)
**Purpose**: Ensure system performs efficiently under various load conditions.

**Key Test Scenarios**:
- 🚀 100+ agent stress testing
- ⏱️ Response time under load
- 📊 Dynamic scaling behavior
- 💾 Memory usage optimization
- 🔀 Mixed workload handling
- 🏭 Pipeline throughput optimization
- 🧹 Resource cleanup verification

**Critical Success Metrics**:
- Handle 100+ agents efficiently
- <300ms average response time
- <400MB peak memory usage
- >10 items/second throughput
- <10MB memory growth after cleanup

#### 4. Cross-Feature Integration (`scenarios/cross-feature/`)
**Purpose**: Validate integration between different system components.

**Key Test Scenarios**:
- 🧠 Neural + Swarm coordination
- 💭 Memory + Persistence integration
- 🔌 MCP + Agent lifecycle
- 🪝 Hooks + Event system
- 🌐 Full system integration

**Critical Success Metrics**:
- All features work together seamlessly
- Data flows correctly between components
- No conflicts between feature integrations
- Performance doesn't degrade with full integration

## 🚀 Running Tests

### Quick Start
```bash
# Run all integration tests
npm run test:integration

# Run with verbose output
VERBOSE=true npm run test:integration

# Run in parallel
PARALLEL_TESTS=true npm run test:integration

# Run with coverage
COVERAGE=true npm run test:integration
```

### Advanced Usage
```bash
# Run specific test suite
npx mocha test/integration/scenarios/lifecycle/full-workflow.test.js

# Run with custom timeout
npx mocha test/integration/scenarios/performance/load-testing.test.js --timeout 120000

# Run resilience tests only
npx mocha "test/integration/scenarios/resilience/*.test.js"

# Bail on first failure
BAIL_ON_FAILURE=true npm run test:integration
```

### Environment Variables
```bash
# Test Configuration
NODE_ENV=test                    # Test environment
PARALLEL_TESTS=true             # Run tests in parallel
VERBOSE=true                    # Detailed output
COVERAGE=true                   # Generate coverage report
BAIL_ON_FAILURE=true           # Stop on first failure

# System Configuration  
RUV_SWARM_TEST_MODE=true       # Enable test mode
RUV_SWARM_LOG_LEVEL=debug      # Logging level
RUV_SWARM_MAX_AGENTS=100       # Maximum agents for tests
RUV_SWARM_MEMORY_LIMIT=512MB   # Memory limit for tests
```

## 📊 Test Results and Reporting

### Result Categories
- ✅ **PASSED**: Test completed successfully
- ❌ **FAILED**: Test failed (check logs)  
- 💥 **ERROR**: Test encountered an error
- ⏭️ **SKIPPED**: Test was skipped

### Critical vs Non-Critical
- 🔴 **CRITICAL**: Must pass for production deployment
- 🟡 **NON-CRITICAL**: Performance optimizations, can be addressed later

### Sample Output
```
🧪 ruv-swarm Integration Test Suite
=====================================

Configuration:
  Environment: test
  Parallel: true
  Verbose: false
  Coverage: true

🔧 Setting up test environment...
✅ Environment ready

📋 Running Lifecycle Tests...
  PASSED in 15234ms

📋 Running Resilience Tests...
  PASSED in 23451ms

📋 Running Performance Tests...  
  PASSED in 45123ms

📋 Running Cross-Feature Integration...
  PASSED in 32156ms

📊 Integration Test Results
============================

Summary:
  Total Suites: 4
  Passed: 4
  Failed: 0
  Success Rate: 100.0%
  Duration: 116.96s

💡 Recommendations:
  🎉 All tests passed! System is ready for production.
```

## 🔧 Configuration

### Test Configuration (`integration-test-config.js`)
Centralized configuration for all test parameters:

```javascript
const config = {
  environment: {
    timeout: { suite: 60000, test: 30000 },
    retries: { flaky: 2, critical: 1 }
  },
  swarm: {
    maxAgents: { small: 10, large: 100, stress: 200 },
    performance: { targetResponseTime: 100 }
  },
  load: {
    profiles: { light: {...}, heavy: {...} }
  }
};
```

### Platform-Specific Limits
Tests automatically adjust based on platform:
- **Linux**: Up to 200 agents, 2GB memory
- **macOS**: Up to 100 agents, 1GB memory  
- **Windows**: Up to 50 agents, 512MB memory

### CI/CD Adjustments
In CI environments:
- Timeouts doubled for stability
- Agent limits reduced for resource constraints
- Less frequent sampling to reduce overhead

## 🐛 Debugging Failed Tests

### Common Failure Patterns

#### 1. Timeout Failures
```bash
# Increase timeout for specific test
npx mocha test.js --timeout 120000

# Check system resources
npm run test:system-check
```

#### 2. Memory Issues
```bash
# Run with memory profiling
node --inspect --max-old-space-size=2048 test/integration/run-integration-tests.js

# Check for memory leaks
npm run test:memory-profile
```

#### 3. Database Issues
```bash
# Clean test database
rm -f data/test-ruv-swarm.db

# Run with fresh database
npm run test:clean-db
```

#### 4. Network/Timing Issues
```bash
# Run with retries
RUV_SWARM_TEST_RETRIES=3 npm run test:integration

# Increase network timeouts
RUV_SWARM_NETWORK_TIMEOUT=10000 npm run test:integration
```

### Debugging Tools
```bash
# Verbose logging
DEBUG=ruv-swarm:* npm run test:integration

# Save test artifacts
SAVE_ARTIFACTS=true npm run test:integration

# Profile performance
PROFILE=true npm run test:integration
```

## 📈 Performance Benchmarks

### Expected Performance Metrics

#### Agent Management
- Agent spawn time: <50ms per agent
- 100 agents spawn time: <5 seconds
- Agent communication latency: <10ms
- Memory per agent: <5MB

#### Task Execution  
- Simple task completion: <100ms
- Complex task completion: <1000ms
- Task throughput: >10 tasks/second
- Parallel efficiency: >80%

#### System Resources
- Peak memory usage: <400MB (100 agents)
- CPU usage: <80% under load
- File handles: <1000 active
- Database connections: <10 concurrent

#### Network Performance
- Message latency: <5ms local
- Throughput: >1000 messages/second
- Network partition recovery: <3 seconds
- Consensus time: <100ms

## 🔄 Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Integration Tests
  run: |
    npm run test:integration:ci
  env:
    NODE_ENV: test
    CI: true
    PARALLEL_TESTS: true
    COVERAGE: true
```

### Test Result Artifacts
- JUnit XML reports
- Coverage reports (HTML/JSON)
- Performance metrics
- Failure screenshots
- System logs

### Quality Gates
- All critical tests must pass
- Coverage >80%
- Performance within benchmarks
- No memory leaks detected
- Error rate <1%

## 🚀 Best Practices

### Writing Integration Tests

#### 1. Test Structure
```javascript
describe('Feature Integration', () => {
  let swarm;
  
  beforeEach(async () => {
    swarm = new RuvSwarm();
    await swarm.init({ /* config */ });
  });
  
  afterEach(async () => {
    await swarm.shutdown();
  });
  
  it('should handle complete workflow', async () => {
    // Test implementation
  });
});
```

#### 2. Resource Management
- Always clean up resources in `afterEach`
- Use timeouts appropriately
- Monitor memory usage
- Handle async operations properly

#### 3. Error Handling
```javascript
it('should recover from failures', async () => {
  try {
    // Test that might fail
    await riskyOperation();
  } catch (error) {
    // Verify error handling
    expect(error.code).to.equal('EXPECTED_ERROR');
  }
});
```

#### 4. Performance Testing
```javascript
it('should meet performance requirements', async () => {
  const startTime = Date.now();
  await performOperation();
  const duration = Date.now() - startTime;
  
  expect(duration).to.be.lessThan(1000);
});
```

## 📚 Additional Resources

- [Unit Test Guide](../unit/README.md)
- [Performance Testing Guide](../performance/README.md)
- [MCP Integration Tests](../mcp-integration.test.js)
- [Neural Agent Tests](../neural-integration.test.js)
- [System Architecture](../../docs/architecture.md)

## 🤝 Contributing

When adding new integration tests:

1. Follow the existing test structure
2. Use appropriate timeouts and retries
3. Clean up resources properly
4. Add comprehensive assertions
5. Document test purpose and expectations
6. Update this README if needed

### Test Naming Convention
```
describe('Component Integration', () => {
  describe('Feature Scenario', () => {
    it('should behave correctly under conditions', () => {
      // Test implementation
    });
  });
});
```

### Performance Test Guidelines
- Set realistic performance expectations
- Test under various load conditions
- Monitor resource usage
- Include both positive and negative scenarios
- Verify cleanup after stress testing

---

For questions or issues with integration tests, please check the [troubleshooting guide](../../docs/troubleshooting.md) or open an issue.