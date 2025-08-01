# RUV-Swarm Examples

This directory contains example applications demonstrating various features and use cases of the RUV-Swarm framework.

## Quick Start

To run any example:

```bash
# From the ruv-swarm directory
cargo run --example basic_swarm

# With logging enabled
RUST_LOG=info cargo run --example basic_swarm

# For release mode (optimized)
cargo run --release --example basic_swarm
```

## Examples Overview

### 1. Basic Swarm (`basic_swarm.rs`)

A simple introduction to swarm creation and task orchestration.

**Features demonstrated:**
- Creating a swarm with configuration
- Spawning worker agents
- Orchestrating a simple task
- Retrieving swarm statistics

**Use case:** Getting started with RUV-Swarm

### 2. Neural Training (`neural_training.rs`)

Distributed neural network training using swarm intelligence.

**Features demonstrated:**
- Cognitive diversity in agents
- Parallel training algorithms
- Data partitioning strategies
- Performance monitoring

**Use case:** Machine learning at scale

### 3. Cognitive Diversity (`cognitive_diversity.rs`)

Shows how different cognitive styles improve problem-solving.

**Features demonstrated:**
- Multiple cognitive styles (Analytical, Creative, Strategic, etc.)
- Task routing based on cognitive affinity
- Collaboration patterns analysis
- Diversity impact measurement

**Use case:** Complex problem solving requiring diverse approaches

### 4. Web Demo (`web_demo/`)

Interactive browser-based visualization of swarm behavior.

**Features:**
- Real-time swarm visualization
- Agent spawning interface
- Task creation and monitoring
- Performance metrics dashboard
- WASM integration

**To run:**
```bash
# Build WASM module first
wasm-pack build --target web --out-dir examples/web_demo

# Serve the demo (requires a web server)
python3 -m http.server --directory examples/web_demo 8000

# Open browser to http://localhost:8000
```

## Running Benchmarks

The `benches/` directory contains comprehensive performance benchmarks:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench agent_spawn

# Generate HTML report
cargo bench -- --save-baseline my_baseline

# Compare with baseline
cargo bench -- --baseline my_baseline
```

### Benchmark Categories

1. **Agent Spawning** (`agent_spawn_bench.rs`)
   - Single vs batch creation
   - Different agent types
   - Cognitive style impact
   - Topology effects

2. **Message Passing** (`message_passing_bench.rs`)
   - Unicast latency
   - Broadcast performance
   - Routing overhead
   - Concurrent messaging

3. **Task Orchestration** (`orchestration_bench.rs`)
   - Task distribution
   - Scheduling overhead
   - Load balancing strategies
   - Pipeline execution

4. **WASM Performance** (`wasm_bench.rs`)
   - Native vs WASM comparison
   - Serialization overhead
   - Memory allocation
   - Optimization impact

## Performance Results

### Agent Creation (Native)
- Single agent: ~50μs
- Batch (100 agents): ~4ms
- With cognitive style: +10% overhead

### Message Passing
- Unicast latency: ~20μs
- Broadcast (50 agents): ~1ms
- Gossip propagation: ~10ms for 100 nodes

### Task Orchestration
- Simple task (5 workers): ~2ms
- Parallel distribution (50 workers): ~15ms
- Hierarchical (3 levels): ~8ms

### WASM vs Native
- Agent spawn: 2.5x slower
- Task execution: 1.8x slower
- Message passing: 2.2x slower
- Acceptable for browser environments

## Best Practices

1. **Swarm Configuration**
   - Start with small agent counts and scale up
   - Choose topology based on communication patterns
   - Enable cognitive diversity for complex tasks

2. **Task Design**
   - Break large tasks into smaller subtasks
   - Use appropriate task types for routing
   - Set priorities for time-sensitive work

3. **Performance Optimization**
   - Use batch operations when possible
   - Monitor swarm statistics
   - Adjust heartbeat intervals based on needs

4. **Error Handling**
   - Always handle orchestration failures
   - Implement retry logic for critical tasks
   - Monitor agent health

## Advanced Usage

### Custom Topologies
```rust
let config = SwarmConfig {
    topology: Topology::SmallWorld,
    rewiring_probability: 0.1,
    ..Default::default()
};
```

### Cognitive Task Routing
```rust
// Task will be routed to agents with matching cognitive styles
let task = Task::new(
    "Creative problem",
    TaskType::Creative,
    subtasks
)?;

swarm.orchestrate_with_diversity(task).await?;
```

### Pipeline Processing
```rust
let pipeline = swarm.create_pipeline()
    .add_stage("Extract", extractors)
    .add_stage("Transform", transformers)
    .add_stage("Load", loaders)
    .build();

swarm.execute_pipeline(pipeline, data).await?;
```

## Troubleshooting

### Common Issues

1. **"Too many agents" error**
   - Increase `max_agents` in SwarmConfig
   - Check system resource limits

2. **Task timeout**
   - Increase `message_timeout` in config
   - Check agent responsiveness
   - Verify network connectivity

3. **WASM module not found**
   - Run `wasm-pack build` first
   - Check output directory path
   - Ensure correct import path in JS

## Contributing

To add a new example:
1. Create a new `.rs` file in `examples/`
2. Add entry to `Cargo.toml` `[[example]]` section
3. Document features demonstrated
4. Update this README

## License

See the main project LICENSE file.