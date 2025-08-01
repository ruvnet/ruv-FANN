# ruv-swarm-mcp

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-2.0-blue.svg)](https://modelcontextprotocol.io)
[![Claude Code](https://img.shields.io/badge/Claude_Code-Compatible-green.svg)](https://claude.ai/code)

**ruv-swarm-mcp** is a powerful Model Context Protocol (MCP) server implementation for the RUV-Swarm orchestration system. It provides Claude Code and other MCP-compatible clients with seamless access to advanced swarm intelligence capabilities through a standardized JSON-RPC interface.

## 🚀 Introduction

The ruv-swarm-mcp crate bridges the gap between Claude Code's AI capabilities and RUV-Swarm's distributed agent orchestration system. By implementing the Model Context Protocol specification, it enables Claude to directly control and coordinate intelligent agent swarms for complex task execution.

### Key Features

- **13+ Comprehensive MCP Tools** - Complete swarm orchestration capabilities
- **Claude Code Integration** - Seamless integration with Anthropic's Claude Code CLI
- **JSON-RPC 2.0 Protocol** - Standards-compliant MCP server implementation
- **WebSocket & Stdio Support** - Multiple communication protocols
- **Real-time Monitoring** - Live event streaming and performance metrics
- **Neural Agent Support** - Advanced AI agents with cognitive pattern recognition
- **WASM Integration** - High-performance WebAssembly modules
- **Persistent Memory** - Session-based and long-term data storage
- **Performance Optimization** - Built-in profiling and auto-optimization

## 📦 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp

# Build the MCP server
cargo build --release

# Install globally
cargo install --path .
```

### Using Cargo

```bash
cargo install ruv-swarm-mcp
```

## 🛠️ Usage

### Starting the MCP Server

```bash
# Basic startup
ruv-swarm-mcp

# With custom configuration
ruv-swarm-mcp --config mcp-config.json --port 3000

# Debug mode
RUST_LOG=debug ruv-swarm-mcp --debug
```

### Claude Code Integration

#### 1. Configure Claude Code MCP Server

Add to your Claude Code configuration:

```json
{
  "mcpServers": {
    "ruv-swarm": {
      "command": "ruv-swarm-mcp",
      "args": ["--stdio"],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

#### 2. Using ruv-swarm Tools in Claude Code

```bash
# Start Claude Code with ruv-swarm MCP tools
claude-code --mcp-server ruv-swarm

# The following tools will be available in Claude Code:
# - mcp__ruv-swarm__swarm_init
# - mcp__ruv-swarm__agent_spawn
# - mcp__ruv-swarm__task_orchestrate
# - mcp__ruv-swarm__swarm_monitor
# ... and 9 more tools
```

## 🔧 Complete MCP Tools Documentation

### 1. Swarm Initialization

#### `swarm_init`
Initialize a new swarm with specified topology and configuration.

**Parameters:**
- `topology` (required): `"mesh"` | `"hierarchical"` | `"ring"` | `"star"`
- `maxAgents` (optional): Maximum number of agents (default: 5)
- `strategy` (optional): `"balanced"` | `"specialized"` | `"adaptive"`

**Example:**
```json
{
  "name": "swarm_init",
  "arguments": {
    "topology": "mesh",
    "maxAgents": 10,
    "strategy": "balanced"
  }
}
```

### 2. Agent Management

#### `agent_spawn`
Create new agents with specific roles and capabilities.

**Parameters:**
- `type` (required): `"researcher"` | `"coder"` | `"analyst"` | `"optimizer"` | `"coordinator"`
- `name` (optional): Custom agent name
- `capabilities` (optional): Array of agent capabilities

**Example:**
```json
{
  "name": "agent_spawn",
  "arguments": {
    "type": "researcher",
    "name": "Research Agent Alpha",
    "capabilities": ["data_analysis", "literature_review", "report_generation"]
  }
}
```

#### `agent_list`
List all active agents in the swarm.

**Parameters:**
- `filter` (optional): `"all"` | `"active"` | `"idle"` | `"busy"`

#### `agent_metrics`
Get performance metrics for specific agents or all agents.

**Parameters:**
- `agentId` (optional): Specific agent ID
- `metric` (optional): `"all"` | `"cpu"` | `"memory"` | `"tasks"` | `"performance"`

### 3. Task Orchestration

#### `task_orchestrate`
Orchestrate complex tasks across the swarm using various strategies.

**Parameters:**
- `task` (required): Task description or objective
- `priority` (optional): `"low"` | `"medium"` | `"high"` | `"critical"`
- `strategy` (optional): `"parallel"` | `"sequential"` | `"adaptive"`
- `maxAgents` (optional): Maximum agents to use

**Example:**
```json
{
  "name": "task_orchestrate",
  "arguments": {
    "task": "Analyze market trends and generate investment recommendations",
    "priority": "high",
    "strategy": "adaptive",
    "maxAgents": 5
  }
}
```

#### `task_status`
Check the progress of running tasks.

**Parameters:**
- `taskId` (optional): Specific task ID
- `detailed` (optional): Include detailed progress information

#### `task_results`
Retrieve results from completed tasks.

**Parameters:**
- `taskId` (required): Task ID to retrieve results for
- `format` (optional): `"summary"` | `"detailed"` | `"raw"`

### 4. Monitoring & Analytics

#### `swarm_status`
Get comprehensive swarm status and health information.

**Parameters:**
- `verbose` (optional): Include detailed agent information

#### `swarm_monitor`
Monitor swarm activity in real-time.

**Parameters:**
- `duration` (optional): Monitoring duration in seconds (default: 10)
- `interval` (optional): Update interval in seconds (default: 1)

#### `memory_usage`
Get current memory usage statistics.

**Parameters:**
- `detail` (optional): `"summary"` | `"detailed"` | `"by-agent"`

### 5. Performance & Benchmarking

#### `benchmark_run`
Execute performance benchmarks.

**Parameters:**
- `type` (optional): `"all"` | `"wasm"` | `"swarm"` | `"agent"` | `"task"`
- `iterations` (optional): Number of iterations (default: 10)

#### `features_detect`
Detect runtime features and capabilities.

**Parameters:**
- `category` (optional): `"all"` | `"wasm"` | `"simd"` | `"memory"` | `"platform"`

### 6. Neural Agent Capabilities

#### `neural_status`
Get neural agent status and performance metrics.

**Parameters:**
- `agentId` (optional): Specific neural agent ID

#### `neural_train`
Train neural agents with sample tasks.

**Parameters:**
- `agentId` (optional): Specific agent ID to train
- `iterations` (optional): Number of training iterations (default: 10)

#### `neural_patterns`
Get cognitive pattern information for neural agents.

**Parameters:**
- `pattern` (optional): `"all"` | `"convergent"` | `"divergent"` | `"lateral"` | `"systems"` | `"critical"` | `"abstract"`

## 🔄 Claude Code Workflow Examples

### Research & Analysis Workflow

```python
# In Claude Code, use the MCP tools to orchestrate research
await mcp__ruv_swarm__swarm_init({
    "topology": "hierarchical",
    "maxAgents": 8,
    "strategy": "specialized"
})

# Spawn specialized research agents
await mcp__ruv_swarm__agent_spawn({
    "type": "researcher",
    "name": "Literature Researcher",
    "capabilities": ["academic_search", "citation_analysis"]
})

await mcp__ruv_swarm__agent_spawn({
    "type": "analyst", 
    "name": "Data Analyst",
    "capabilities": ["statistical_analysis", "visualization"]
})

# Orchestrate comprehensive research task
await mcp__ruv_swarm__task_orchestrate({
    "task": "Conduct comprehensive analysis of renewable energy trends",
    "priority": "high",
    "strategy": "parallel",
    "maxAgents": 4
})

# Monitor progress
await mcp__ruv_swarm__swarm_monitor({
    "duration": 30,
    "interval": 2
})
```

### Development Workflow

```python
# Initialize development-focused swarm
await mcp__ruv_swarm__swarm_init({
    "topology": "mesh",
    "maxAgents": 6,
    "strategy": "adaptive"
})

# Create coding agents
await mcp__ruv_swarm__agent_spawn({
    "type": "coder",
    "name": "Backend Developer",
    "capabilities": ["python", "rust", "api_development"]
})

await mcp__ruv_swarm__agent_spawn({
    "type": "coder",
    "name": "Frontend Developer", 
    "capabilities": ["javascript", "react", "ui_design"]
})

# Orchestrate development project
await mcp__ruv_swarm__task_orchestrate({
    "task": "Build a distributed task management system",
    "priority": "critical",
    "strategy": "sequential",
    "maxAgents": 5
})

# Get performance metrics
await mcp__ruv_swarm__agent_metrics({
    "metric": "all"
})
```

## ⚙️ Configuration

### Server Configuration (`mcp-config.json`)

```json
{
  "bind_addr": "127.0.0.1:3000",
  "max_connections": 100,
  "request_timeout_secs": 300,
  "enable_websocket": true,
  "enable_stdio": true,
  "log_level": "info",
  "features": {
    "neural_agents": true,
    "wasm_modules": true,
    "simd_support": true,
    "persistent_memory": true
  },
  "swarm_defaults": {
    "max_agents": 10,
    "strategy": "balanced",
    "topology": "mesh"
  }
}
```

### Environment Variables

```bash
# Server configuration
export RUV_SWARM_PORT=3000
export RUV_SWARM_HOST=127.0.0.1
export RUV_SWARM_MAX_CONNECTIONS=100

# Feature flags
export RUV_SWARM_ENABLE_NEURAL=true
export RUV_SWARM_ENABLE_WASM=true
export RUV_SWARM_ENABLE_SIMD=true

# Debugging
export RUST_LOG=debug
export RUV_SWARM_DEBUG=true
```

## 🧪 Development & Testing

### Running Tests

```bash
# Run all tests (24 unit tests + 4 doc tests)
cargo test

# Run specific test modules
cargo test tests::integration_tests
cargo test tests::security_tests

# Run with debug output
RUST_LOG=debug cargo test -- --nocapture

# Run individual test
cargo test test_mcp_server_creation
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp

# Install dependencies
cargo build

# Run in development mode
cargo run -- --debug --config dev-config.json

# Run tests with coverage
cargo tarpaulin --out Html
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 API Reference

### WebSocket Endpoints

- `ws://localhost:3000/mcp` - Main MCP WebSocket endpoint
- `ws://localhost:3000/events` - Real-time event streaming
- `ws://localhost:3000/metrics` - Performance metrics stream

### HTTP Endpoints

- `GET /` - Server information and health check
- `GET /tools` - List all available MCP tools
- `GET /status` - Current swarm status
- `GET /metrics` - Performance metrics
- `POST /execute` - Execute MCP tool directly

### JSON-RPC Methods

All MCP tools follow the JSON-RPC 2.0 specification:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": { ... }
  },
  "id": 1
}
```

## 🔗 Links

- **Main Repository**: [https://github.com/ruvnet/ruv-FANN](https://github.com/ruvnet/ruv-FANN)
- **Documentation**: [https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/docs](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/docs)
- **Model Context Protocol**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Claude Code**: [https://claude.ai/code](https://claude.ai/code)
- **Issues**: [https://github.com/ruvnet/ruv-FANN/issues](https://github.com/ruvnet/ruv-FANN/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## 🎯 Performance Benchmarks

| Tool | Average Latency | Throughput | Memory Usage |
|------|----------------|------------|--------------|
| swarm_init | 150ms | 100 ops/sec | 2.5MB |
| agent_spawn | 50ms | 500 ops/sec | 1.2MB |
| task_orchestrate | 200ms | 50 ops/sec | 5.1MB |
| swarm_monitor | 10ms | 1000 ops/sec | 0.8MB |

## 🤝 Acknowledgments

- **Anthropic** for Claude Code and MCP specification
- **WebAssembly Community** for WASM runtime capabilities
- **Rust Community** for excellent async/tokio ecosystem
- **Contributors** who have helped improve this project

---

**Created by [rUv](https://github.com/ruvnet)** - Pioneering the future of AI agent orchestration and swarm intelligence.

*ruv-swarm-mcp enables seamless integration between Claude Code and distributed AI agent systems, making complex multi-agent coordination accessible through standardized protocols.*