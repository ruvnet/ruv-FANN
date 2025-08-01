# ruv-swarm-agents

Specialized AI agent implementations for the RUV Swarm neural orchestration system. This crate provides cognitive diversity through intelligent agent types that leverage different thinking patterns for optimal swarm performance.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-ready-green.svg)](https://webassembly.org/)

## 🧠 Introduction

The `ruv-swarm-agents` crate implements specialized AI agents with diverse cognitive patterns designed for high-performance swarm orchestration. Each agent type embodies different thinking approaches, enabling the swarm to tackle complex problems from multiple angles simultaneously.

Built on WebAssembly with SIMD optimization, these agents provide:

- **Cognitive Diversity**: Six distinct thinking patterns for comprehensive problem-solving
- **Neural Integration**: Built-in neural network support with 18+ activation functions  
- **Swarm Coordination**: Seamless inter-agent communication and task distribution
- **Performance Optimization**: WASM-powered execution with memory-efficient design

## ✨ Key Features

### 🎯 Cognitive Patterns

Six scientifically-backed cognitive patterns drive agent behavior:

- **Convergent Thinking**: Focused, analytical problem-solving
- **Divergent Thinking**: Creative, exploratory ideation
- **Lateral Thinking**: Unconventional, breakthrough approaches
- **Systems Thinking**: Holistic, interconnected analysis
- **Critical Thinking**: Evaluative, questioning methodology
- **Abstract Thinking**: Conceptual, theoretical reasoning

### 🤖 Specialized Agent Types

Five specialized agent implementations:

- **Researcher**: Data analysis, information gathering, pattern discovery
- **Coder**: Code generation, optimization, technical implementation
- **Analyst**: Performance evaluation, metrics analysis, insights generation
- **Optimizer**: Resource management, efficiency improvements, bottleneck resolution
- **Coordinator**: Task orchestration, agent synchronization, workflow management

### 🚀 Advanced Capabilities

- **Neural Network Integration**: Adaptive learning with 18+ activation functions
- **WebAssembly Performance**: SIMD-optimized execution for maximum throughput
- **Persistent Memory**: SQLite-backed agent memory for long-term learning
- **Real-time Monitoring**: Comprehensive metrics and health monitoring
- **Dynamic Scaling**: Automatic agent spawning based on workload demands

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruv-swarm-agents = "0.1.0"

# Required dependencies
ruv-swarm-core = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

For WebAssembly targets:

```toml
[dependencies]
ruv-swarm-agents = { version = "0.1.0", features = ["wasm"] }
```

## 🚀 Quick Start

### Basic Agent Creation

```rust
use ruv_swarm_agents::{Agent, CognitivePattern, ResearcherAgent};
use ruv_swarm_core::task::Task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a researcher agent with divergent thinking
    let mut agent = ResearcherAgent::new("researcher-1")
        .with_cognitive_pattern(CognitivePattern::Divergent)
        .with_capabilities(vec![
            "data-analysis".to_string(),
            "pattern-recognition".to_string(),
            "information-synthesis".to_string(),
        ]);

    // Start the agent
    agent.start().await?;

    // Create and process a task
    let task = Task::new("analyze-dataset", "research")
        .with_payload("Process customer behavior patterns");

    let result = agent.process(task).await?;
    println!("Analysis complete: {:?}", result);

    Ok(())
}
```

### Multi-Agent Swarm Setup

```rust
use ruv_swarm_agents::{
    Agent, CognitivePattern, AgentSwarm,
    ResearcherAgent, CoderAgent, AnalystAgent
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create agents with complementary cognitive patterns
    let researcher = ResearcherAgent::new("researcher-1")
        .with_cognitive_pattern(CognitivePattern::Divergent);
    
    let coder = CoderAgent::new("coder-1")
        .with_cognitive_pattern(CognitivePattern::Convergent);
        
    let analyst = AnalystAgent::new("analyst-1")
        .with_cognitive_pattern(CognitivePattern::Critical);

    // Form a coordinated swarm
    let mut swarm = AgentSwarm::new("problem-solving-swarm")
        .add_agent(researcher)
        .add_agent(coder)
        .add_agent(analyst)
        .with_topology(SwarmTopology::Mesh);

    // Initialize and orchestrate
    swarm.initialize().await?;
    
    let result = swarm.orchestrate_task(
        "Build an intelligent data processing pipeline"
    ).await?;

    println!("Swarm result: {:?}", result);
    Ok(())
}
```

### Neural Network Enhanced Agents

```rust
use ruv_swarm_agents::{Agent, NeuralEnhancedAgent, ActivationFunction};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create agent with neural network capabilities
    let mut agent = NeuralEnhancedAgent::new("neural-optimizer-1")
        .with_network_config(|config| {
            config
                .layers(&[784, 128, 64, 10])
                .activation(ActivationFunction::ReLU)
                .learning_rate(0.001)
                .enable_simd(true)
        });

    // Train the agent on task patterns
    agent.train_on_tasks(&training_tasks, 100).await?;
    
    // Use trained agent for optimized task processing
    let optimized_result = agent.process_with_learning(task).await?;
    
    Ok(())
}
```

## 📚 Agent Types Documentation

### 🔬 ResearcherAgent

Specialized for data analysis, information gathering, and pattern discovery.

**Cognitive Patterns**: Divergent, Systems, Abstract  
**Capabilities**: 
- `data-analysis` - Statistical analysis and pattern recognition
- `information-synthesis` - Combining insights from multiple sources  
- `hypothesis-generation` - Creating testable theories from observations
- `literature-review` - Comprehensive information gathering

**Example Use Cases**:
- Market research and trend analysis
- Scientific data exploration
- Competitive intelligence gathering
- User behavior pattern discovery

### 💻 CoderAgent

Optimized for code generation, technical implementation, and system development.

**Cognitive Patterns**: Convergent, Critical, Systems  
**Capabilities**:
- `code-generation` - Creating optimized code solutions
- `architecture-design` - System design and planning
- `debugging` - Error identification and resolution
- `optimization` - Performance improvement implementation

**Example Use Cases**:
- Automated code generation
- Legacy system modernization
- Performance optimization
- Technical debt resolution

### 📊 AnalystAgent

Focused on performance evaluation, metrics analysis, and insight generation.

**Cognitive Patterns**: Critical, Convergent, Abstract  
**Capabilities**:
- `metrics-analysis` - Statistical evaluation and reporting
- `performance-evaluation` - System performance assessment
- `trend-identification` - Pattern recognition in time series data
- `recommendation-generation` - Actionable insights from analysis

**Example Use Cases**:
- Business intelligence reporting
- System performance monitoring
- Financial analysis and forecasting
- Quality assurance evaluation

### ⚡ OptimizerAgent

Specialized in resource management, efficiency improvements, and bottleneck resolution.

**Cognitive Patterns**: Systems, Convergent, Critical  
**Capabilities**:
- `resource-optimization` - Memory and CPU efficiency improvements
- `workflow-streamlining` - Process optimization and automation
- `bottleneck-resolution` - Performance constraint identification
- `cost-reduction` - Efficiency-driven cost optimization

**Example Use Cases**:
- Infrastructure optimization
- Supply chain efficiency
- Database query optimization
- Algorithm performance tuning

### 🎯 CoordinatorAgent

Designed for task orchestration, agent synchronization, and workflow management.

**Cognitive Patterns**: Systems, Lateral, Abstract  
**Capabilities**:
- `task-orchestration` - Multi-agent task coordination
- `workflow-management` - Complex process coordination
- `resource-allocation` - Optimal agent task distribution
- `conflict-resolution` - Agent disagreement mediation

**Example Use Cases**:
- Project management automation
- Multi-agent system coordination
- Resource scheduling optimization
- Distributed computing orchestration

## 🧬 Cognitive Pattern Combinations

Agents can leverage multiple cognitive patterns simultaneously for enhanced problem-solving:

```rust
// Multi-pattern research agent
let versatile_researcher = ResearcherAgent::new("multi-pattern-researcher")
    .with_primary_pattern(CognitivePattern::Divergent)
    .with_secondary_patterns(vec![
        CognitivePattern::Systems,
        CognitivePattern::Abstract
    ])
    .enable_pattern_switching(true);

// Pattern switching based on task type
agent.configure_pattern_rules(|task| {
    match task.category {
        "creative" => CognitivePattern::Divergent,
        "analytical" => CognitivePattern::Convergent,
        "system-design" => CognitivePattern::Systems,
        _ => CognitivePattern::Critical
    }
});
```

## 🔗 Integration Examples

### MCP Server Integration

```rust
use ruv_swarm_agents::mcp::MCPAgentServer;

// Create MCP-compatible agent server
let mcp_server = MCPAgentServer::new()
    .register_agent_type::<ResearcherAgent>("researcher")
    .register_agent_type::<CoderAgent>("coder")
    .register_agent_type::<AnalystAgent>("analyst")
    .with_stdio_transport();

// Start MCP server for Claude Code integration
mcp_server.start().await?;
```

### Web Integration

```rust
use ruv_swarm_agents::web::WebAgentInterface;

// Web-based agent interface
let web_interface = WebAgentInterface::new()
    .bind("0.0.0.0:8080")
    .with_cors_enabled()
    .register_swarm(swarm);

web_interface.serve().await?;
```

## 🔧 Configuration

### Agent Configuration

```rust
use ruv_swarm_agents::config::AgentConfig;

let config = AgentConfig::new()
    .max_concurrent_tasks(10)
    .memory_limit_mb(512)
    .enable_neural_networks(true)
    .cognitive_flexibility(0.7)
    .learning_rate(0.001)
    .collaboration_threshold(0.8);

let agent = ResearcherAgent::with_config("researcher-1", config);
```

### Swarm Configuration

```rust
use ruv_swarm_agents::config::SwarmConfig;

let swarm_config = SwarmConfig::new()
    .topology(SwarmTopology::Hierarchical)
    .max_agents(50)
    .coordination_interval_ms(100)
    .enable_auto_scaling(true)
    .load_balancing_strategy(LoadBalancing::Cognitive);
```

## 📈 Performance Monitoring

### Real-time Metrics

```rust
// Monitor agent performance
let metrics = agent.get_metrics().await?;
println!("Tasks processed: {}", metrics.tasks_processed);
println!("Success rate: {:.2}%", metrics.success_rate * 100.0);
println!("Avg processing time: {}ms", metrics.avg_processing_time_ms);

// Cognitive pattern effectiveness
let pattern_metrics = agent.get_cognitive_metrics().await?;
for (pattern, effectiveness) in pattern_metrics {
    println!("{:?}: {:.2}% effective", pattern, effectiveness * 100.0);
}
```

### Health Monitoring

```rust
// Continuous health monitoring
tokio::spawn(async move {
    loop {
        let health = agent.health_check().await?;
        match health.status {
            HealthStatus::Healthy => info!("Agent {} healthy", agent.id()),
            HealthStatus::Degraded => warn!("Agent {} degraded: {}", agent.id(), health.message),
            HealthStatus::Unhealthy => error!("Agent {} unhealthy: {}", agent.id(), health.message),
        }
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
});
```

## 🧪 Testing

```bash
# Run agent tests
cargo test --package ruv-swarm-agents

# Run cognitive pattern tests
cargo test cognitive_patterns

# Run integration tests
cargo test --test integration

# Benchmark performance
cargo bench --package ruv-swarm-agents
```

## 🌐 Links

- **Main Repository**: [ruv-FANN](https://github.com/ruvnet/ruv-FANN)
- **Documentation**: [docs.rs/ruv-swarm-agents](https://docs.rs/ruv-swarm-agents)
- **Core Library**: [ruv-swarm-core](../ruv-swarm-core)
- **Examples**: [examples directory](../../examples)
- **Benchmarks**: [Performance Reports](../../benchmarks)

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

---

**Created by rUv** - *Advancing AI through cognitive diversity and neural orchestration*

For support and discussions, visit our [GitHub Discussions](https://github.com/ruvnet/ruv-FANN/discussions).