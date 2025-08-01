# ruv-swarm Unified WASM Module

This crate provides a unified WebAssembly interface to the entire ruv-FANN ecosystem, including neural networks, forecasting models, and swarm orchestration capabilities.

## Architecture

The unified WASM module exposes:
- **Core**: Swarm orchestration, agent management, task distribution
- **Neural**: ruv-FANN neural networks with 18 activation functions and 5 training algorithms
- **Forecasting**: 27+ neuro-divergent forecasting models
- **Persistence**: SQLite-based storage with WASM optimizations

## Building

### Quick Build
```bash
# Build all modules with optimization
./scripts/build-wasm-unified.sh

# Build with specific features
./scripts/build-wasm-unified.sh release "simd,parallel,full"
```

### Advanced Build Pipeline
```bash
# Use the build orchestrator for fine-grained control
./scripts/build-orchestrator.sh

# Build specific module
./scripts/build-orchestrator.sh --module neural

# Build and run tests
./scripts/build-orchestrator.sh --test
```

## Features

- **SIMD Optimization**: 4x speedup for vector operations (when available)
- **Memory Management**: Smart pooling and allocation strategies
- **Progressive Loading**: Load only required modules
- **TypeScript Support**: Full type definitions generated

## Usage

### JavaScript/TypeScript
```javascript
import { RuvSwarmUnified } from '@ruv/swarm-wasm';

// Initialize with all modules
const swarm = new RuvSwarmUnified();
await swarm.init();

// Create agents
const agent = swarm.core.WasmAgent.new("researcher-1", "researcher");
agent.set_cognitive_pattern("divergent");

// Create swarm
const swarmInstance = swarm.core.WasmSwarm.new(100);
swarmInstance.set_topology("mesh");

// Use neural networks (if available)
if (swarm.neural) {
    const network = swarm.neural.NeuralNetwork.new();
    const netInfo = network.create_network([2, 4, 1]);
}
```

### Performance Optimization
```javascript
// Configure for neural network workloads
const config = swarm.core.WasmConfig.new();
config.optimize_for_neural_networks();

// Configure for swarm operations
config.optimize_for_swarm(50); // 50 agents

// Check system capabilities
const caps = swarm.core.get_system_capabilities();
console.log('SIMD available:', caps.simd);
console.log('Workers available:', caps.workers);
```

### Memory Management
```javascript
// Initialize memory manager
const memManager = swarm.core.WasmMemoryManager.new();
memManager.initialize_pools();

// Monitor memory usage
const stats = memManager.get_memory_stats();
console.log('Memory usage:', stats.current_usage_mb, 'MB');

// Set up memory pressure monitoring
const monitor = swarm.core.MemoryMonitor.new(50); // 50MB threshold
monitor.set_callback(() => {
    console.warn('Memory pressure detected!');
});
```

## Build Configuration

### Core Module (`wasm-config/core.toml`)
- Optimized for size (< 1MB)
- 4MB initial memory
- SIMD enabled

### Neural Module (`wasm-config/neural.toml`)
- Optimized for performance
- 16MB initial memory
- Full SIMD acceleration

### Forecasting Module (`wasm-config/forecasting.toml`)
- Balanced optimization
- 8MB initial memory
- Time series optimizations

## Performance Targets

- **Total Size**: < 5MB (all modules combined)
- **Load Time**: < 100ms
- **Memory Usage**: < 50MB for 10-agent swarm
- **SIMD Speedup**: 2-4x for supported operations

## Development

### Prerequisites
- Rust 1.70+
- wasm-pack 0.12+
- Node.js 16+ (for wasm-opt)

### Testing
```bash
# Run WASM tests
wasm-pack test --headless --chrome

# Run benchmarks
npm run benchmark:wasm
```

### CI/CD
GitHub Actions workflow automatically:
1. Builds WASM modules on push
2. Runs tests and benchmarks
3. Publishes to NPM on main branch

## Troubleshooting

### Build Issues
- Ensure wasm32-unknown-unknown target is installed: `rustup target add wasm32-unknown-unknown`
- Install wasm-pack: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`

### Size Optimization
- Use `wasm-opt -Oz` for maximum size reduction
- Enable LTO in Cargo.toml
- Use `wee_alloc` feature for smaller allocator

### Performance
- Check SIMD support: `ruvSwarm.core.get_features().simd`
- Monitor memory usage with MemoryMonitor
- Use batch operations for better performance