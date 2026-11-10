# Changelog

## [2.0.1] - 2026-11-10

### Release pipeline

- Refresh the WASM build receipt source digest after the ruv-swarm Rust crates dropped OpenSSL (RUSTSEC-2026-0474/0476/0477, #205/#208). The WASM binaries and their hashes are unchanged; no runtime or dependency change.

## [2.0.0] - 2026-11-07

### Breaking corrections

- Keep the public API to supported local assignment, checked dense inference and statistical forecasts. Remove untyped prototype exports and unrestricted source subpaths.
- Replace simulated MCP training, invented task results and random telemetry with real SQLite records and generated WASM operations. Unsupported execution fails explicitly; 17 supported tools are discoverable.
- Local assignments report `assigned` or `unassigned`, preserve task descriptions and capabilities, and do not claim external model completion. The CLI uses the same records and real inference/forecast APIs.
- Keep Claude permission checks enabled by default. Explicit bypass remains available; `--` protects flag-shaped prompt text. Retire unsafe generated Windows batch prompt forwarding.
- Generate supported tool documentation and project MCP registration while preserving existing servers, permission rules and hooks; remove broad Git/shell permission grants and unsupported automation instructions.
- Require Node.js 22.13 or newer for the maintained toolchain. The optional Rust build tool wasm-pack is installed explicitly rather than downloaded by an npm postinstall hook.

### Security and release validation

- Use literal process arguments, bounded JSON-RPC frames/concurrency, allowlisted SQL updates, single-read WASM verification and connection-specific bounded statement caches.
- Correct production SQLite expiry comparisons and same-millisecond memory identifier collisions.
- Replace vulnerable development-only deepmerge with reviewed, pinned `@stackline/deepmerge` 1.1.2. Verify the actual poisoning payload, Jest configuration and Rollup resolution; all dependency audits pass.
- Build core and SIMD from reviewed locked Rust sources; validate generated bindings, types, licenses and source receipts before packaging. Keep unsupported legacy binaries out of the archive.
- Test actual Node MCP workflows, persistence across restarts, inference, forecasts and measured benchmarks. Isolate Rust integration tests by listener and database, preserving their failure exit codes.

The existing Rust crate versions remain independent of this npm major release.

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
- Historical performance claims had no reproducible benchmark evidence and are superseded by the supported 2.0 contract.
- Provides comprehensive diagnostics for future troubleshooting

## [1.0.11] - 2025-07-03

### Added
- Initial MCP server implementation
- WASM-based neural network capabilities
- Swarm orchestration features
