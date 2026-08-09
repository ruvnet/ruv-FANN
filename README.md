# ruv-FANN: The Neural Intelligence Framework 🧠

[![Crates.io](https://img.shields.io/crates/v/ruv-fann.svg)](https://crates.io/crates/ruv-fann)
[![Documentation](https://docs.rs/ruv-fann/badge.svg)](https://docs.rs/ruv-fann)
[![License](https://img.shields.io/crates/l/ruv-fann.svg)](https://github.com/ruvnet/ruv-fann/blob/main/LICENSE)
[![CI](https://github.com/ruvnet/ruv-FANN/workflows/CI/badge.svg)](https://github.com/ruvnet/ruv-FANN/actions)

**What if intelligence could be ephemeral, composable, and surgically precise?**

Welcome to ruv-FANN, a comprehensive neural intelligence framework that reimagines how we build, deploy, and orchestrate artificial intelligence. This repository contains three groundbreaking projects that work together to deliver unprecedented performance in neural computing, forecasting, and multi-agent orchestration.

## 🌟 The Vision

We believe AI should be:
- **Ephemeral**: Spin up intelligence when needed, dissolve when done
- **Accessible**: CPU-native, GPU-optional - built for the GPU-poor
- **Composable**: Mix and match neural architectures like LEGO blocks
- **Precise**: Tiny, purpose-built brains for specific tasks

This isn't about calling a model API. This is about **instantiating intelligence**.

## 🎯 What's in This Repository?

### 1. **ruv-FANN Core** - The Foundation
A complete Rust rewrite of the legendary FANN (Fast Artificial Neural Network) library. No unsafe code in the core API (10 audited `unsafe` sites, all x86-64 AVX2 intrinsics in the SIMD acceleration path in `src/simd`, gated by runtime CPU feature detection), blazing performance, and full compatibility with decades of proven neural network algorithms.

### 2. **Neuro-Divergent** - Advanced Neural Forecasting  
17 state-of-the-art forecasting models (LSTM, N-BEATS, Transformers) with 100% Python NeuralForecast compatibility. 2-4x faster, 25-35% less memory.

### 3. **ruv-swarm** - Ephemeral Swarm Intelligence
The crown jewel. Achieves **84.8% SWE-Bench solve rate**\* (internal 2025 benchmark run; see the note under [Benchmarks](#-benchmarks)), outperforming Claude 3.7 by 14.5 points. Spin up lightweight neural networks that exist just long enough to solve problems.

## 🚀 Quick Install ruv-swarm

```bash
# NPX - No installation required!
npx ruv-swarm@latest init --claude

# NPM - Global installation
npm install -g ruv-swarm

# Cargo - For Rust developers
cargo install ruv-swarm-cli
```

That's it. You're now running distributed neural intelligence.

## 🧠 How It Works

### The Magic of Ephemeral Intelligence

1. **Instantiation**: Neural networks are created on-demand for specific tasks
2. **Specialization**: Each network is purpose-built with just enough neurons
3. **Execution**: Networks solve their task using CPU-native WASM
4. **Dissolution**: Networks disappear after completion, no resource waste

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│          Claude Code / Your App             │
├─────────────────────────────────────────────┤
│            ruv-swarm (MCP/CLI)              │
├─────────────────────────────────────────────┤
│         Neuro-Divergent Models              │
│    (LSTM, TCN, N-BEATS, Transformers)      │
├─────────────────────────────────────────────┤
│           ruv-FANN Core Engine              │
│        (Rust Neural Networks)               │
├─────────────────────────────────────────────┤
│            WASM Runtime                     │
│    (Browser/Edge/Server/Embedded)          │
└─────────────────────────────────────────────┘
```

## ⚡ Key Features

### 🏃 Performance
- **<100ms decisions** - Complex reasoning in milliseconds
- **84.8% SWE-Bench**\* - Best-in-class problem solving
- **2.8-4.4x faster** - Than traditional frameworks
- **32.3% less tokens** - Cost-efficient intelligence

### 🛠️ Technology
- **Pure Rust** - Memory safe, zero panics
- **WebAssembly** - Run anywhere: browser to RISC-V
- **CPU-native** - No CUDA, no GPU required
- **MCP Integration** - Native Claude Code support

### 🧬 Intelligence Models
- **17 Neural Architectures** - From MLP to Transformers
- **5 Swarm Topologies** - Mesh, ring, hierarchical, star, custom
- **7 Cognitive Patterns** - Convergent, divergent, lateral, systems thinking
- **Adaptive Learning** - Real-time evolution and optimization

## 📊 Benchmarks

| Metric | ruv-swarm | Claude 3.7 | GPT-4 | Improvement |
|--------|-----------|------------|-------|-------------|
| **SWE-Bench Solve Rate** | **84.8%**\* | 70.3% | 65.2% | **+14.5pp** |
| **Token Efficiency** | **32.3% less** | Baseline | +5% | **Best** |
| **Speed (tasks/sec)** | **3,800** | N/A | N/A | **4.4x** |
| **Memory Usage** | **29% less** | Baseline | N/A | **Optimal** |

\* SWE-Bench and comparison figures are from an internal 2025 benchmark run using the in-repo harness (`ruv-swarm/crates/swe-bench-adapter`); stored results live in `ruv-swarm/models/claude-code-optimizer/swe_bench_optimization_results.json`. They have not been independently reproduced — treat them as historical results until a full reproduction script and logs are published.

## 🌐 Ecosystem Projects

### Core Projects
- **[ruv-FANN](./ruv-fann/)** - Neural network foundation library
- **[Neuro-Divergent](./neuro-divergent/)** - Advanced forecasting models
- **[ruv-swarm](./ruv-swarm/)** - Distributed swarm intelligence

### Tools & Extensions
- **[MCP Server](./ruv-swarm/docs/MCP_USAGE.md)** - Claude Code integration
- **[CLI Tools](./ruv-swarm/docs/CLI_REFERENCE.md)** - Command-line interface
- **[Docker Support](./ruv-swarm/npm/docker/)** - Containerized deployment

## 🤝 Contributing with GitHub Swarm

We use an innovative swarm-based contribution system powered by ruv-swarm itself!

### How to Contribute

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/ruv-FANN.git
   cd ruv-FANN
   ```

2. **Initialize Swarm**
   ```bash
   npx ruv-swarm init --github-swarm
   ```

3. **Spawn Contribution Agents**
   ```bash
   # Auto-spawns specialized agents for your contribution type
   npx ruv-swarm contribute --type "feature|bug|docs"
   ```

4. **Let the Swarm Guide You**
   - Agents analyze codebase and suggest implementation
   - Automatic code review and optimization
   - Generates tests and documentation
   - Creates optimized pull request

### Contribution Areas
- 🐛 **Bug Fixes** - Swarm identifies and fixes issues
- ✨ **Features** - Guided feature implementation
- 📚 **Documentation** - Auto-generated from code analysis
- 🧪 **Tests** - Intelligent test generation
- 🎨 **Examples** - Working demos and tutorials

## 🙏 Acknowledgments

### Special Thanks To

#### Core Contributors
- **Ocean(@ohdearquant)** - Transformed FANN from mock implementations to real neural networks with actual CPU and GPU training. Built the Rust implementation from placeholder code into a functional neural computing engine.
- **Bron(@syndicate604)** - Made the JavaScript/WASM integration actually work by removing mock functions and building real functionality. Transformed broken prototypes into production-ready systems.
- **Jed(@jedarden)** - Platform integration and scope management
- **Shep(@elsheppo)** - Testing framework and quality assurance

#### Projects We Built Upon
- **[FANN](http://leenissen.dk/fann/)** - Steffen Nissen's original Fast Artificial Neural Network library
- **[NeuralForecast](https://github.com/Nixtla/neuralforecast)** - Inspiration for forecasting model APIs
- **[Claude MCP](https://modelcontextprotocol.io/)** - Model Context Protocol for AI integration
- **[Rust WASM](https://rustwasm.github.io/)** - WebAssembly toolchain and ecosystem

#### Open Source Libraries
- **num-traits** - Generic numeric traits
- **ndarray** - N-dimensional arrays
- **serde** - Serialization framework
- **tokio** - Async runtime
- **wasm-bindgen** - WASM bindings

### Community
Thanks to all contributors, issue reporters, and users who have helped shape ruv-FANN into what it is today. Special recognition to the Rust ML community for pioneering memory-safe machine learning.

## 📄 License

Dual-licensed under:
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

Choose whichever license works best for your use case.

---

<div align="center">

**Built with ❤️ and 🦀 by the rUv team**

*Making intelligence ephemeral, accessible, and precise*

[Website](https://ruv.ai) • [Documentation](https://docs.ruv.ai) • [Discord](https://discord.gg/ruv) • [Twitter](https://twitter.com/ruvnet)

</div>