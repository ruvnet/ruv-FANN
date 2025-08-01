# ruv-swarm NPM Package Documentation

## Overview
This directory contains documentation specific to the ruv-swarm NPM package.

## Quick Start

### Installation
```bash
npm install ruv-swarm
# or
npx ruv-swarm init --claude --force
```

### Basic Usage
```javascript
const { RuvSwarm } = require('ruv-swarm');

// Initialize swarm
const swarm = await RuvSwarm.initialize({
  loadingStrategy: 'progressive',
  enableNeuralNetworks: true,
  enableForecasting: true
});
```

## Directory Structure

### `/api/` 
API documentation and type definitions

### `/guides/`
- Setup guides
- Integration tutorials
- Best practices

### `/examples/`
- Code examples
- Sample workflows
- Integration patterns

## Key Features

### 🤖 Claude Code Integration
Automatic setup with hooks:
```bash
npx ruv-swarm init --claude --force
```

This creates:
- `.claude/settings.json` with pre-configured hooks
- `.claude/commands/` documentation
- Automatic coordination and formatting

### 🧠 Neural Networks
Built-in neural network capabilities with:
- 18 activation functions
- Configurable architectures
- Cognitive diversity patterns

### 📊 Performance
- 84.8% SWE-Bench solve rate
- 32.3% token reduction
- 2.8-4.4x speed improvement

## Package Structure

```
npm/
├── bin/              # CLI executables
├── src/              # Source code
│   ├── claude-integration/  # Claude Code integration
│   ├── hooks/              # Hook implementations
│   └── *.js/ts            # Core modules
├── wasm/             # WebAssembly modules
├── test/             # Test files
├── test-reports/     # Test results
├── config/           # Configuration files
├── examples/         # Example code
└── docs/            # Documentation
```

## Related Documentation
- [Main Documentation](../../docs/README.md)
- [API Reference](../../docs/API_REFERENCE.md)
- [Examples](../examples/)