# ruv-swarm-core Overview

## What is ruv-swarm-core?

ruv-swarm-core is the foundational crate for orchestrating distributed AI agent swarms. It provides the core abstractions, types, and implementations needed to build scalable multi-agent systems.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Agent       │    │     Task        │    │   Topology      │
│   (Trait)       │    │  Management     │    │  Coordination   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              Swarm Orchestrator                 │
         │                                                 │
         │  ┌─────────────┐         ┌─────────────────┐    │
         │  │    Swarm    │         │   AsyncSwarm    │    │
         │  │ (Sync/ST)   │         │ (Async/MT)      │    │
         │  └─────────────┘         └─────────────────┘    │
         └─────────────────────────────────────────────────┘
```

## Core Components

### 1. **Agent Trait** (`src/agent.rs`)
- Defines the interface all agents must implement
- Async-first design with `async fn process()`
- Built-in lifecycle management (`start`, `shutdown`, `health_check`)
- Capability-based task matching

### 2. **Task System** (`src/task.rs`)
- Priority-based task scheduling
- Multiple payload types (Binary, Json, Custom)
- Builder pattern for task creation
- Retry and timeout mechanisms

### 3. **Topology Management** (`src/topology.rs`)
- Multiple topology types: Mesh, Star, Pipeline, Hierarchical
- Dynamic connection management
- Automatic coordinator election (Star topology)

### 4. **Swarm Orchestrators**
- **Swarm** (`src/swarm.rs`): Single-threaded, simpler API
- **AsyncSwarm** (`src/async_swarm.rs`): Multi-threaded, production-ready

### 5. **Error Handling** (`src/error.rs`)
- Comprehensive error types for all failure modes
- Retriable vs non-retriable error classification
- Rich error context and debugging information

## Key Features

- **🚀 Async-First**: Built for non-blocking operations
- **🔒 Thread-Safe**: AsyncSwarm supports concurrent access
- **📊 Observable**: Built-in metrics and health monitoring
- **🎯 Flexible**: Multiple distribution strategies and topologies
- **🛡️ Robust**: Comprehensive error handling and recovery
- **🧪 Well-Tested**: 87.6% test coverage with 169 tests

## Design Principles

1. **Performance**: Optimized for high-throughput scenarios
2. **Reliability**: Graceful failure handling and recovery
3. **Flexibility**: Pluggable strategies and topologies
4. **Observability**: Rich metrics and monitoring capabilities
5. **Developer Experience**: Clear APIs and comprehensive documentation

## When to Use

**Use ruv-swarm-core when you need:**
- Distributed AI agent coordination
- High-performance task processing
- Scalable multi-agent systems
- Robust error handling and monitoring
- Production-grade reliability

**Consider alternatives if:**
- You need simple single-agent processing
- Performance is not a concern
- You don't need distributed coordination

## Next Steps

- [Getting Started Guide](./getting-started.md)
- [API Reference](./api-reference.md)
- [Swarm vs AsyncSwarm](./swarm-vs-async-swarm.md)
- [Testing Guide](./testing-guide.md)