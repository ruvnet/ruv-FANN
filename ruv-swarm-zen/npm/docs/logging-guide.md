# ruv-swarm Logging Guide

## Overview

The ruv-swarm logging system provides comprehensive diagnostics and debugging capabilities for MCP communication, swarm operations, and performance tracking.

## Features

### 1. Structured Logging
- **Hierarchical Logger**: Component-based logging with inheritance
- **Log Levels**: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- **Correlation IDs**: Track requests across the entire session
- **Metadata Support**: Attach contextual data to every log entry

### 2. Connection Lifecycle Tracking
- Connection establishment events
- Message exchange patterns
- Connection failures and retries
- Duration and performance metrics

### 3. Performance Monitoring
- Operation timing with start/end tracking
- Memory usage at operation boundaries
- Automatic threshold detection
- Aggregated performance metrics

### 4. Debug Levels for MCP
- **TRACE**: Raw message content
- **DEBUG**: Method calls and responses
- **INFO**: Connection events and tool execution
- **WARN**: Performance issues and retries
- **ERROR**: Failures and exceptions

### 5. Log Rotation
- Automatic file rotation at 10MB
- Keep last 5 log files
- Configurable directory and size limits

## Configuration

### Environment Variables

```bash
# Global log level
LOG_LEVEL=DEBUG

# Component-specific levels
MCP_LOG_LEVEL=TRACE      # MCP server logging
TOOLS_LOG_LEVEL=DEBUG     # Tool execution logging
SWARM_LOG_LEVEL=INFO      # Swarm operations
AGENT_LOG_LEVEL=DEBUG     # Agent activity
NEURAL_LOG_LEVEL=INFO     # Neural network operations
WASM_LOG_LEVEL=WARN       # WASM module loading
DB_LOG_LEVEL=INFO         # Database operations
HOOKS_LOG_LEVEL=DEBUG     # Hook execution
PERF_LOG_LEVEL=INFO       # Performance metrics
MEMORY_LOG_LEVEL=WARN     # Memory usage alerts

# Output configuration
LOG_TO_FILE=true          # Enable file logging
LOG_FORMAT=json           # Use JSON format (default: text)
LOG_DIR=./logs            # Log directory path
```

### Examples

```bash
# Debug MCP communication
MCP_LOG_LEVEL=TRACE npx ruv-swarm mcp start

# Enable file logging with JSON format
LOG_TO_FILE=true LOG_FORMAT=json npx ruv-swarm mcp start

# Debug specific components
TOOLS_LOG_LEVEL=DEBUG AGENT_LOG_LEVEL=TRACE npx ruv-swarm init

# Full debug mode
LOG_LEVEL=DEBUG LOG_TO_FILE=true npx ruv-swarm mcp start
```

## Diagnostic Commands

### Run Diagnostic Tests
```bash
npx ruv-swarm diagnose test
```
Tests:
- Memory allocation
- File system access
- WASM module availability

### Generate Diagnostic Report
```bash
# Display in console
npx ruv-swarm diagnose report

# Save to file
npx ruv-swarm diagnose report --output=report.json

# Markdown format
npx ruv-swarm diagnose report --output=report.md --format=markdown
```

### Monitor System
```bash
# Monitor for 60 seconds
npx ruv-swarm diagnose monitor

# Custom duration and interval
npx ruv-swarm diagnose monitor --duration=120 --interval=500
```

### Analyze Logs
```bash
# Search for errors
npx ruv-swarm diagnose logs

# Custom pattern search
npx ruv-swarm diagnose logs --pattern="connection.*failed"

# Different log directory
npx ruv-swarm diagnose logs --dir=/custom/logs --pattern="timeout"
```

### Show Configuration
```bash
npx ruv-swarm diagnose config
```

## Log Output Examples

### Text Format (Default)
```
[14:23:45.123] 🔌 INFO  [mcp-server] (sess-abc-0001) Connection established
   connection: {
     protocol: 'stdio',
     transport: 'stdin/stdout',
     timestamp: '2024-01-04T14:23:45.123Z'
   }
```

### JSON Format
```json
{
  "timestamp": "2024-01-04T14:23:45.123Z",
  "level": "INFO",
  "logger": "mcp-server",
  "message": "Connection established",
  "correlationId": "sess-abc-0001",
  "connection": {
    "protocol": "stdio",
    "transport": "stdin/stdout"
  }
}
```

## Debugging Connection Issues

### 1. Enable Trace Logging
```bash
MCP_LOG_LEVEL=TRACE LOG_TO_FILE=true npx ruv-swarm mcp start
```

### 2. Run Diagnostics
```bash
# Test system
npx ruv-swarm diagnose test

# Generate report after failure
npx ruv-swarm diagnose report --output=debug-report.json
```

### 3. Monitor in Real-time
```bash
# In one terminal
npx ruv-swarm diagnose monitor

# In another terminal, reproduce the issue
```

### 4. Analyze Patterns
```bash
# Look for connection failures
npx ruv-swarm diagnose logs --pattern="connection.*failed|error|timeout"
```

## Performance Tracking

### Automatic Metrics
- Tool execution duration
- Memory usage per operation
- Connection success/failure rates
- Resource usage at failure time

### Performance Thresholds
- `swarm_init`: 1000ms
- `agent_spawn`: 500ms
- `task_orchestrate`: 2000ms
- `neural_train`: 5000ms

Operations exceeding thresholds are logged as warnings.

## Best Practices

1. **Development**: Use `LOG_LEVEL=DEBUG` for detailed information
2. **Production**: Use `LOG_LEVEL=INFO` with `LOG_TO_FILE=true`
3. **Debugging**: Enable `MCP_LOG_LEVEL=TRACE` for protocol issues
4. **Performance**: Monitor with `PERF_LOG_LEVEL=DEBUG`
5. **Memory Issues**: Set `MEMORY_LOG_LEVEL=INFO` for tracking

## Troubleshooting

### High Memory Usage
```bash
MEMORY_LOG_LEVEL=INFO npx ruv-swarm diagnose monitor --duration=300
```

### Connection Failures
```bash
MCP_LOG_LEVEL=TRACE npx ruv-swarm mcp start 2> debug.log
npx ruv-swarm diagnose logs --pattern="failed|error" --dir=.
```

### Performance Issues
```bash
PERF_LOG_LEVEL=DEBUG LOG_TO_FILE=true npx ruv-swarm mcp start
npx ruv-swarm diagnose report --output=perf-report.json
```

## Integration with Claude Code

The logging system is designed to work seamlessly with Claude Code's MCP integration:

1. **Stderr Output**: In MCP stdio mode, all logs go to stderr
2. **Correlation IDs**: Track requests across Claude Code sessions
3. **Structured Data**: Easy to parse for automated analysis
4. **Performance Metrics**: Identify slow operations affecting responsiveness

## Advanced Usage

### Custom Logger Configuration
```javascript
import { Logger } from 'ruv-swarm/logger';

const logger = new Logger({
  name: 'my-component',
  level: 'DEBUG',
  enableFile: true,
  logDir: './my-logs',
  metadata: {
    service: 'custom-service',
    version: '1.0.0'
  }
});
```

### Child Loggers
```javascript
const childLogger = logger.child({
  module: 'sub-component',
  correlationId: 'request-123'
});
```

### Performance Tracking
```javascript
const opId = logger.startOperation('complex-task', { userId: 123 });
// ... perform operation ...
logger.endOperation(opId, true, { resultCount: 42 });
```

## Future Enhancements

- [ ] Log aggregation and search
- [ ] Real-time log streaming
- [ ] Alert thresholds and notifications
- [ ] Integration with external logging services
- [ ] Performance profiling visualization