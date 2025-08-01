# RUV-SWARM Test Suite Documentation

## Overview

This directory contains comprehensive test suites for the ruv-swarm MCP (Model Context Protocol) implementation, including integration tests, persistence layer tests, neural network integration tests, and example workflows.

## Test Structure

### 1. MCP Integration Tests (`mcp-integration.test.js`)
Comprehensive tests for all 16 MCP tools:
- **swarm_init** - Initialize swarm with topology
- **swarm_status** - Get swarm status
- **swarm_monitor** - Real-time monitoring
- **agent_spawn** - Create new agents
- **agent_list** - List active agents
- **agent_metrics** - Get performance metrics
- **task_orchestrate** - Orchestrate tasks across swarm
- **task_status** - Check task progress
- **task_results** - Retrieve task results
- **task_create** - Create individual tasks
- **benchmark_run** - Execute performance benchmarks
- **features_detect** - Detect runtime capabilities
- **memory_usage** - Get memory statistics
- **memory_store** - Store persistent data
- **memory_get** - Retrieve stored data
- **neural_status** - Neural network status
- **neural_train** - Train neural networks
- **neural_patterns** - Analyze cognitive patterns

### 2. Persistence Layer Tests (`persistence.test.js`)
Tests for SQLite database persistence:
- Agent CRUD operations
- Task management and status tracking
- Memory storage with TTL support
- Swarm state persistence
- Neural network weight storage
- Performance metrics recording
- Concurrent operations handling
- Database recovery and cleanup

### 3. Neural Network Integration Tests (`neural-integration.test.js`)
Tests for FANN neural network integration:
- Neural network creation and training
- Agent learning capabilities
- Decision making processes
- Swarm intelligence coordination
- Collective learning patterns
- Performance optimization
- Concurrent neural operations

### 4. Example Workflows (`../examples/mcp-workflows.js`)
Real-world usage examples:
- **Web Application Development** - Full development lifecycle
- **AI Research Swarm** - Distributed research coordination
- **CI/CD Pipeline** - Automated deployment workflows
- **Data Analysis Pipeline** - ML model training workflow
- **Swarm Coordination** - Multi-agent collaboration patterns

## Running Tests

### Quick Start
```bash
# Install dependencies
npm install

# Run all tests
npm run test:all

# Run specific test suites
npm run test:mcp          # MCP integration tests
npm run test:persistence  # Persistence layer tests
npm run test:neural       # Neural network tests

# Run with live reload
npm run test:watch
```

### MCP Server Setup
Some tests require the MCP server to be running:
```bash
# Start MCP server
npm run mcp:server

# Or with auto-reload for development
npm run mcp:server:dev
```

### Running Examples
```bash
# Run all workflow examples
npm run examples

# Run specific workflows
npm run examples:webapp    # Web app development workflow
npm run examples:research  # AI research workflow
npm run examples:cicd      # CI/CD pipeline workflow
npm run examples:data      # Data analysis workflow
npm run examples:swarm     # Swarm coordination workflow
```

## Test Configuration

### Environment Variables
- `NODE_ENV=test` - Set automatically during test runs
- `MCP_SERVER_URL` - Override default MCP server URL (default: ws://localhost:3000/mcp)
- `TEST_TIMEOUT` - Override test timeout (default: 60000ms)

### Test Database
- Location: `test/test-swarm.db` (created during tests)
- Automatically cleaned up after test completion
- Uses SQLite for cross-platform compatibility

## Test Features

### Comprehensive Coverage
- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end workflow testing
- **Performance Tests** - Benchmarking and optimization
- **Concurrency Tests** - Parallel operation handling
- **Error Handling** - Edge cases and failure scenarios

### Advanced Testing Capabilities
- **Persistence Verification** - Data integrity across sessions
- **Neural Learning Simulation** - Agent improvement over time
- **Swarm Coordination** - Multi-agent collaboration patterns
- **Real-time Monitoring** - Event streaming and notifications
- **Performance Metrics** - CPU, memory, and throughput tracking

## Test Report

After running `npm run test:all`, a detailed JSON report is generated:
- Location: `test/test-report-[timestamp].json`
- Contains: Test results, performance metrics, error details

## Writing New Tests

### Test Structure Template
```javascript
async function test(name, fn) {
    try {
        await fn();
        console.log(`✅ ${name}`);
        results.passed++;
    } catch (error) {
        console.error(`❌ ${name}`);
        console.error(`   ${error.message}`);
        results.failed++;
    }
}

// Example test
await test('Your Test Name', async () => {
    // Test implementation
    assert(condition, 'Error message');
});
```

### MCP Tool Testing Pattern
```javascript
const result = await client.sendRequest('tools/call', {
    name: 'ruv-swarm.tool_name',
    arguments: {
        param1: 'value1',
        param2: 'value2'
    }
});

assert(result.expected_field);
```

## Troubleshooting

### Common Issues

1. **MCP Server Connection Failed**
   - Ensure MCP server is running: `npm run mcp:server`
   - Check server logs for errors
   - Verify port 3000 is available

2. **SQLite Errors**
   - Install build tools: `npm install -g node-gyp`
   - On Windows: Install Visual Studio Build Tools
   - On macOS: Install Xcode Command Line Tools

3. **WebSocket Errors**
   - Check firewall settings
   - Ensure no proxy interference
   - Verify WebSocket support in environment

4. **Test Timeouts**
   - Increase timeout in test configuration
   - Check for infinite loops or deadlocks
   - Verify async operations complete

### Debug Mode
Enable debug logging:
```bash
DEBUG=ruv-swarm:* npm run test:all
```

## Continuous Integration

### GitHub Actions Configuration
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install
      - run: npm run build:all
      - run: npm run test:all
```

## Performance Benchmarks

Expected performance metrics:
- **Agent Spawn**: < 50ms per agent
- **Task Creation**: < 10ms per task
- **Memory Operations**: < 5ms per operation
- **Neural Forward Pass**: < 100ms for large networks
- **Concurrent Operations**: Linear scaling up to 100 agents

## Contributing

When adding new tests:
1. Follow existing test patterns
2. Include both success and failure cases
3. Test edge cases and error conditions
4. Add performance benchmarks for new features
5. Update this README with new test documentation

## License

Same as parent project: MIT OR Apache-2.0