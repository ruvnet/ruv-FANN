# ruv-swarm Documentation

## Overview
This directory contains all documentation for the ruv-swarm project, organized by category.

## Directory Structure

### 📊 `/reports/`
Performance reports, validation results, and training completion reports:
- `FINAL_PERFORMANCE_VALIDATION_REPORT.md` - Comprehensive performance validation
- `LSTM_TRAINING_COMPLETE.md` - LSTM model training results

### 🔧 `/implementation/`
Technical implementation details and architecture documentation:
- `MCP_IMPLEMENTATION.md` - Model Context Protocol implementation
- `DEPLOYMENT_RECOMMENDATIONS.md` - Production deployment guidelines
- `AGENT4_IMPLEMENTATION_REPORT.md` - Agent 4 implementation details
- `AGENT_TASK_BINDING_FIX_SUMMARY.md` - Task binding fixes
- `COMPREHENSIVE_SYSTEM_ANALYSIS_REPORT.md` - Full system analysis
- `FINAL_WASM_CONFIRMATION.md` - WASM implementation confirmation
- `MCP_TROUBLESHOOTING.md` - MCP troubleshooting guide
- `OPTIMIZATION_REPORT.md` - Optimization results
- `SIMD_IMPLEMENTATION_REPORT.md` - SIMD implementation details
- `WASM_IMPLEMENTATION_COMPLETE.md` - WASM completion report

### 📚 `/guides/`
User guides and how-to documentation:
- `CONTRIBUTING.md` - Contribution guidelines

### 📖 Core Documentation
- `API_REFERENCE.md` - Complete API reference
- `BENCHMARKS.md` - Benchmark results and methodology
- `FORECASTING_IMPLEMENTATION.md` - Forecasting feature implementation
- `MCP_USAGE.md` - MCP usage guide
- `NEURAL_INTEGRATION.md` - Neural network integration guide
- `AGENT2_COMPLETION_REPORT.md` - Agent 2 completion details

## Quick Links

### Getting Started
- [README](../README.md) - Main project documentation
- [Examples](../examples/README.md) - Code examples
- [NPM Documentation](../npm/README.md) - NPM package documentation

### API & Integration
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [MCP Usage Guide](MCP_USAGE.md) - Model Context Protocol usage
- [Neural Integration](NEURAL_INTEGRATION.md) - Neural network features

### Performance & Benchmarks
- [Benchmarks](BENCHMARKS.md) - Performance benchmarks
- [Performance Report](reports/FINAL_PERFORMANCE_VALIDATION_REPORT.md) - Validation results

### Development
- [Contributing Guide](guides/CONTRIBUTING.md) - How to contribute
- [Implementation Details](implementation/) - Technical implementation docs

## Claude Code Integration

For Claude Code integration with hooks:
```bash
npx ruv-swarm init --claude --force
```

This creates:
- `.claude/settings.json` with hook configurations
- `.claude/commands/` with documentation
- Automatic coordination and formatting hooks

See [Claude Hooks Documentation](../npm/docs/CLAUDE_HOOKS_IMPLEMENTATION_SUMMARY.md) for details.