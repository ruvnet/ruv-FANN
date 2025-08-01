# Docker Test Suite for ruv-swarm v1.0.6

## Overview

This comprehensive Docker test suite validates ruv-swarm v1.0.6 across multiple Node.js versions, platforms, and scenarios to ensure full compatibility and performance before npm release.

## Test Categories

### 1. **Node.js Version Compatibility**
- Node.js 18 (LTS)
- Node.js 20 (LTS)
- Node.js 22 (Current)

### 2. **NPX Integration Tests**
All commands from issue #45:
```bash
npx ruv-swarm mcp start
npx ruv-swarm init [topology] [maxAgents]
npx ruv-swarm spawn <type> [name]
npx ruv-swarm orchestrate <task>
npx ruv-swarm status [--verbose]
npx ruv-swarm monitor [duration]
npx ruv-swarm neural <subcommand>
npx ruv-swarm benchmark <subcommand>
npx ruv-swarm performance <subcommand>
npx ruv-swarm hook <type> [options]
npx ruv-swarm claude-invoke <prompt>
```

### 3. **WASM Validation**
- Ensures WASM loads without fallback
- Validates 16MB memory allocation
- Tests all WASM exports
- Verifies no deprecation warnings

### 4. **Performance Benchmarks**
- Swarm creation (small/medium/large)
- Agent operations
- Neural network operations
- Memory operations
- Task orchestration
- WASM-specific benchmarks

### 5. **MCP Server Testing**
- WebSocket connectivity
- All MCP protocol methods
- Task orchestration
- Memory persistence
- Neural operations

### 6. **Cross-Platform Validation**
- Alpine Linux compatibility
- File system operations
- Process spawning
- Memory allocation
- Native module compatibility

## Running Tests

### Quick Start
```bash
# Run all Docker tests
npm run test:docker:all

# Run specific test
docker-compose -f docker-compose.test.yml run test-node20

# Build images only
npm run test:docker:build

# Clean up
npm run test:docker:clean
```

### Individual Test Commands
```bash
# Node.js compatibility
npm run test:docker:comprehensive

# NPX integration
npm run test:docker:npx

# WASM validation
npm run test:docker:wasm

# Performance benchmarks
npm run test:docker:performance

# MCP server
npm run test:docker:mcp

# Cross-platform
npm run test:docker:cross-platform
```

### Full Test Suite
```bash
# Run complete validation
./scripts/run-docker-tests.sh
```

## Test Results

Results are saved to `test-results/` directory:
- `docker-test-report.json` - Combined test results
- `docker-test-report.md` - Human-readable report
- Individual test outputs in subdirectories

## Expected Results for v1.0.6

### ✅ All tests should pass with:
- No WASM fallback warnings
- 16MB WASM memory allocation confirmed
- All NPX commands functional
- Performance grades A or B
- 100% MCP method compatibility
- Cross-platform compatibility score >90%

### 📊 Performance Targets:
- Swarm creation: <10ms mean
- Agent spawn: <5ms mean
- Neural forward pass: <1ms mean
- Memory operations: <0.5ms mean

## Troubleshooting

### Common Issues

1. **Docker not running**
   ```bash
   # Start Docker daemon
   sudo systemctl start docker
   ```

2. **Permission denied**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   ```

3. **Port conflicts**
   ```bash
   # Check port 3000 for MCP server
   lsof -i :3000
   ```

4. **Out of memory**
   - Increase Docker memory limit in settings
   - Recommended: 4GB minimum

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Docker Tests
  run: |
    npm run test:docker:build
    npm run test:docker:all
```

### Local Pre-release Validation
```bash
# Full validation before publishing
npm run test:docker:build && \
npm run test:docker:all && \
npm run deploy:check
```

## Updating Tests

When adding new features:
1. Update relevant test files in `test/`
2. Add new test cases to Docker suite
3. Update expected results in this README
4. Rebuild Docker images: `npm run test:docker:build`

## Contact

For issues or questions about the Docker test suite:
- GitHub Issues: https://github.com/ruvnet/ruv-FANN/issues
- Documentation: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm