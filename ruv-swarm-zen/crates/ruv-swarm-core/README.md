# ruv-swarm-core 🧠🐝

[![Crates.io](https://img.shields.io/crates/v/ruv-swarm-core.svg)](https://crates.io/crates/ruv-swarm-core)
[![Documentation](https://docs.rs/ruv-swarm-core/badge.svg)](https://docs.rs/ruv-swarm-core)
[![License](https://img.shields.io/crates/l/ruv-swarm-core.svg)](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE)

**Core orchestration and agent traits for RUV Swarm - the foundational building blocks for creating distributed AI agent swarms with cognitive diversity patterns.**

## 🎯 What is ruv-swarm-core?

ruv-swarm-core is the foundational orchestration crate that powers the RUV Swarm ecosystem. It provides the core traits, abstractions, and coordination primitives needed to build distributed AI agent systems with cognitive diversity patterns and advanced swarm behaviors.

This crate serves as the bedrock for all swarm operations, defining how agents communicate, coordinate, and execute tasks across different topologies and distribution strategies.

## ✨ Key Features

### 🤖 **Agent Management**
- **Agent Trait**: Core abstraction for all swarm agents with async processing
- **Cognitive Patterns**: Support for diverse thinking patterns (convergent, divergent, lateral, etc.)
- **Health Monitoring**: Real-time agent status tracking and health checks
- **Resource Management**: Configurable resource limits and requirements
- **Capability Discovery**: Dynamic agent capability registration and matching

### 🌐 **Swarm Coordination**
- **Multiple Topologies**: Mesh, hierarchical, ring, and star network topologies
- **Distribution Strategies**: Balanced, specialized, and adaptive task distribution
- **Task Orchestration**: Priority-based task queue with sophisticated scheduling
- **Message Passing**: Efficient inter-agent communication primitives
- **Fault Tolerance**: Graceful degradation and error recovery mechanisms

### 🧠 **Cognitive Architecture**
- **Pattern Diversity**: 7 distinct cognitive patterns for varied problem-solving approaches
- **Adaptive Behavior**: Agents can switch cognitive patterns based on task requirements
- **Collective Intelligence**: Emergent behaviors from agent interactions
- **Learning Coordination**: Support for distributed learning and knowledge sharing

### 🔧 **Platform Support**
- **No-std Compatible**: Runs in embedded and resource-constrained environments
- **WASM Ready**: Full WebAssembly support for browser and edge deployment
- **Async/Await**: Modern Rust asynchronous programming throughout
- **Type Safety**: Comprehensive error handling with detailed error types

## 📦 Installation

Add ruv-swarm-core to your `Cargo.toml`:

```toml
[dependencies]
ruv-swarm-core = "0.1.0"
```

### Feature Flags

Enable optional features based on your deployment needs:

```toml
[dependencies]
ruv-swarm-core = { version = "0.1.0", features = ["std", "wasm"] }
```

Available features:
- `std` (default) - Standard library support with full functionality
- `no_std` - No standard library support for embedded environments
- `wasm` - WebAssembly support with JavaScript interop
- `minimal` - Minimal feature set for size optimization

## 🚀 Basic Usage Examples

### Simple Agent Implementation

```rust
use ruv_swarm_core::prelude::*;
use async_trait::async_trait;

// Define a compute agent
struct ComputeAgent {
    id: String,
    capabilities: Vec<String>,
}

#[async_trait]
impl Agent for ComputeAgent {
    type Input = f64;
    type Output = f64;
    type Error = std::io::Error;

    async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        // Simulate computational work
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        Ok(input * 2.0)
    }

    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn cognitive_pattern(&self) -> CognitivePattern {
        CognitivePattern::Convergent
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an agent
    let agent = ComputeAgent {
        id: "compute-001".to_string(),
        capabilities: vec!["mathematics".to_string(), "computation".to_string()],
    };

    // Process input
    let mut agent = agent;
    let result = agent.process(42.0).await?;
    println!("Agent processed 42.0 -> {}", result);

    Ok(())
}
```

### Swarm Creation and Task Distribution

```rust
use ruv_swarm_core::{Swarm, SwarmConfig, Task, Priority, TopologyType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure swarm with mesh topology
    let config = SwarmConfig {
        max_agents: 10,
        topology: TopologyType::Mesh,
        distribution_strategy: DistributionStrategy::Balanced,
        enable_monitoring: true,
        ..Default::default()
    };

    // Create swarm
    let mut swarm = Swarm::new(config).await?;

    // Add agents to swarm
    for i in 0..5 {
        let agent = ComputeAgent {
            id: format!("agent-{:03}", i),
            capabilities: vec!["computation".to_string()],
        };
        swarm.add_agent(Box::new(agent)).await?;
    }

    // Create and submit tasks
    for i in 0..20 {
        let task = Task::new(
            format!("task-{}", i),
            Priority::Medium,
            i as f64,
        );
        swarm.submit_task(task).await?;
    }

    // Process tasks
    swarm.start().await?;

    // Wait for completion
    while swarm.has_pending_tasks().await {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    println!("All tasks completed!");
    Ok(())
}
```

### Cognitive Pattern Switching

```rust
use ruv_swarm_core::{Agent, CognitivePattern};

struct AdaptiveAgent {
    id: String,
    current_pattern: CognitivePattern,
}

impl AdaptiveAgent {
    fn switch_pattern(&mut self, task_type: &str) {
        self.current_pattern = match task_type {
            "creative" => CognitivePattern::Divergent,
            "analytical" => CognitivePattern::Convergent,
            "innovative" => CognitivePattern::Lateral,
            "systematic" => CognitivePattern::Systems,
            _ => CognitivePattern::Critical,
        };
    }
}

#[async_trait]
impl Agent for AdaptiveAgent {
    type Input = (String, f64); // (task_type, data)
    type Output = f64;
    type Error = std::io::Error;

    async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        let (task_type, data) = input;
        
        // Switch cognitive pattern based on task
        self.switch_pattern(&task_type);
        
        // Process differently based on cognitive pattern
        let result = match self.current_pattern {
            CognitivePattern::Convergent => data * 1.1,
            CognitivePattern::Divergent => data * 1.5,
            CognitivePattern::Lateral => data.sqrt() * 2.0,
            CognitivePattern::Systems => data.ln() + 1.0,
            _ => data,
        };

        Ok(result)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn cognitive_pattern(&self) -> CognitivePattern {
        self.current_pattern
    }

    fn capabilities(&self) -> &[String] {
        static CAPS: &[String] = &[];
        CAPS
    }
}
```

### Multi-Topology Swarm

```rust
use ruv_swarm_core::{Topology, TopologyType};

async fn create_hierarchical_swarm() -> Result<(), Box<dyn std::error::Error>> {
    // Create hierarchical topology with coordinators and workers
    let topology = Topology::new(TopologyType::Hierarchical);
    
    let config = SwarmConfig {
        topology_type: TopologyType::Hierarchical,
        coordinator_count: 2,
        worker_count: 8,
        enable_fault_tolerance: true,
        ..Default::default()
    };

    let mut swarm = Swarm::with_topology(config, topology).await?;

    // Add coordinator agents
    for i in 0..2 {
        let coordinator = CoordinatorAgent::new(format!("coord-{}", i));
        swarm.add_coordinator(Box::new(coordinator)).await?;
    }

    // Add worker agents
    for i in 0..8 {
        let worker = WorkerAgent::new(format!("worker-{}", i));
        swarm.add_worker(Box::new(worker)).await?;
    }

    // Start coordinated processing
    swarm.start_coordinated().await?;

    Ok(())
}
```

## 🔗 Core API Documentation

### Agent Trait
The foundational trait that all swarm agents must implement:

```rust
#[async_trait]
pub trait Agent: Send + Sync {
    type Input: Send;
    type Output: Send;
    type Error: Send;

    async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    fn id(&self) -> &str;
    fn capabilities(&self) -> &[String];
    fn cognitive_pattern(&self) -> CognitivePattern;
    fn health_status(&self) -> HealthStatus;
}
```

### Cognitive Patterns
Seven distinct patterns for diverse problem-solving approaches:

- **Convergent**: Focused, analytical thinking
- **Divergent**: Creative, expansive exploration
- **Lateral**: Innovative, non-linear approaches
- **Systems**: Holistic, interconnected analysis
- **Critical**: Evaluative, skeptical assessment
- **Abstract**: High-level conceptual thinking
- **Concrete**: Practical, detail-oriented processing

### Task Management
Priority-based task orchestration with sophisticated scheduling:

```rust
pub struct Task<T> {
    pub id: TaskId,
    pub priority: Priority,
    pub data: T,
    pub requirements: Vec<String>,
    pub timeout: Option<Duration>,
}

pub enum Priority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}
```

## 🌐 Topology Support

### Mesh Topology
Full connectivity between all agents for maximum redundancy:
- **Advantages**: High fault tolerance, optimal load distribution
- **Use Cases**: Critical systems, real-time processing

### Hierarchical Topology
Coordinator-worker structure for organized task flow:
- **Advantages**: Clear command structure, efficient resource management
- **Use Cases**: Large-scale processing, enterprise applications

### Ring Topology
Agents connected in a circular pattern:
- **Advantages**: Predictable communication patterns, lower bandwidth
- **Use Cases**: Sequential processing, token-ring algorithms

### Star Topology
Central hub with spoke connections to all agents:
- **Advantages**: Simple coordination, centralized control
- **Use Cases**: Centralized processing, hub-and-spoke architectures

## 📚 API Documentation

Complete API documentation is available on [docs.rs](https://docs.rs/ruv-swarm-core):

- **[Agent Trait Documentation](https://docs.rs/ruv-swarm-core/latest/ruv_swarm_core/trait.Agent.html)**
- **[Swarm Management](https://docs.rs/ruv-swarm-core/latest/ruv_swarm_core/struct.Swarm.html)**  
- **[Task Coordination](https://docs.rs/ruv-swarm-core/latest/ruv_swarm_core/struct.Task.html)**
- **[Topology Types](https://docs.rs/ruv-swarm-core/latest/ruv_swarm_core/enum.TopologyType.html)**
- **[Cognitive Patterns](https://docs.rs/ruv-swarm-core/latest/ruv_swarm_core/enum.CognitivePattern.html)**

## 🔗 Links

- **[Main Repository](https://github.com/ruvnet/ruv-FANN)**: Complete RUV-FANN ecosystem
- **[ruv-swarm](../../../)**: Full swarm implementation using this core
- **[API Documentation](https://docs.rs/ruv-swarm-core)**: Complete API reference  
- **[Examples](../../examples/)**: Practical implementation examples
- **[Benchmarks](../../benchmarks/)**: Performance analysis and comparisons

## 🏗️ Architecture Integration

ruv-swarm-core integrates seamlessly with the broader RUV ecosystem:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ruv-FANN      │    │   ruv-swarm      │    │ neuro-divergent │
│ Neural Networks │◄──►│ Agent Swarms     │◄──►│   Forecasting   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              ▲
                              │
                    ┌─────────────────┐
                    │ ruv-swarm-core  │
                    │ Core Traits &   │
                    │ Orchestration   │
                    └─────────────────┘
```

## 🤝 Contributing

We welcome contributions to ruv-swarm-core! Please see the main repository's [Contributing Guide](https://github.com/ruvnet/ruv-FANN/blob/main/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the main repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm/crates/ruv-swarm-core

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench

# Check no-std compatibility
cargo check --no-default-features --features no_std

# Test WASM compatibility
cargo check --target wasm32-unknown-unknown --features wasm
```

## 📄 License

Licensed under either of:

- **Apache License, Version 2.0** ([LICENSE-APACHE](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- **MIT License** ([LICENSE-MIT](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

**Created by rUv** 

*Building the future of distributed AI agent orchestration - one cognitive pattern at a time.*

Part of the **[RUV-FANN](https://github.com/ruvnet/ruv-FANN)** ecosystem for neural networks, agent swarms, and AI forecasting.