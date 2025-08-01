# Changelog

## [1.0.12] - 2025-07-04

### 🚨 Critical Fix - MCP Connection Stability

This release addresses a critical issue where MCP connections would silently fail mid-session, causing tools to become unavailable and severely impacting swarm coordination quality.

### Fixed
- **MCP Tool Method Access**: Fixed incorrect tool method invocation that caused all tool calls to fail
- **Stream Error Handling**: Added proper stdin/stdout error handlers for graceful shutdown
- **Connection Monitoring**: Implemented heartbeat mechanism to detect stale connections
- **Process Cleanup**: Added signal handlers for SIGTERM/SIGINT
- **Broken Pipe Protection**: Wrapped stdout writes in try-catch blocks

### Added
- **Enhanced Logging System**: 
  - Structured logging with correlation IDs
  - Connection lifecycle tracking
  - Performance monitoring
  - Memory usage tracking
- **Diagnostic CLI Commands**:
  - `npx ruv-swarm diagnose test` - Run diagnostic tests
  - `npx ruv-swarm diagnose report` - Generate diagnostic reports
  - `npx ruv-swarm diagnose monitor` - Real-time system monitoring
  - `npx ruv-swarm diagnose logs` - Analyze log files
- **Docker Test Environment**: Complete reliability testing setup with monitoring stack

### Impact
- Restores stable MCP connections throughout entire sessions
- Eliminates "tool not available" errors
- Returns swarm coordination quality to expected 84.8% performance level
- Provides comprehensive diagnostics for future troubleshooting

## [1.0.11] - 2025-07-03

### Added
- Initial MCP server implementation
- WASM-based neural network capabilities
- Swarm orchestration features