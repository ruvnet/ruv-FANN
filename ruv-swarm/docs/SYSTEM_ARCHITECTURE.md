# RUV Swarm System Architecture

## Overview
RUV Swarm is designed as a high-performance, modular, and distributed system for neural network orchestration, emphasizing ephemeral intelligence and efficient resource utilization. It leverages Rust for core logic and WebAssembly for cross-platform deployment, enabling powerful AI capabilities even on resource-constrained devices.

## Modular Crate System
The project is structured into several Rust crates, each responsible for a specific aspect of the swarm's functionality, promoting modularity, reusability, and clear separation of concerns.

- **`ruv-swarm-core`**: The foundational crate providing the core orchestration engine for managing the swarm.
- **`ruv-swarm-agents`**: Implements the various specialized agents (e.g., Researcher, Coder, Analyst) that form the cognitive units of the swarm.
- **`ruv-swarm-ml`**: Contains the machine learning and forecasting models, offering a diverse set of neural network architectures and training algorithms.
- **`ruv-swarm-wasm`**: Provides WebAssembly bindings for core functionalities, enabling high-speed execution in various environments (browser, edge, server).
- **`ruv-swarm-mcp`**: Integrates the Model Context Protocol (MCP), facilitating communication and coordination between agents and external systems. This can run as a standalone server.
- **`ruv-swarm-transport`**: Handles the communication protocols used within the swarm (e.g., WebSocket, SharedMemory).
- **`ruv-swarm-persistence`**: Manages state persistence, utilizing SQLite for ACID compliance.
- **`ruv-swarm-cli`**: Provides command-line tools for interacting with and managing the swarm.
- **`claude-parser`**: A specialized parser for Stream-JSON output, particularly for Claude Code integration.
- **`swe-bench-adapter`**: Facilitates direct integration with software engineering benchmarks like SWE-Bench.

## Technology Stack
- **Core Language**: Rust 1.75+ with asynchronous programming capabilities (tokio).
- **Machine Learning**: Custom-built neural networks and time series models.
- **WebAssembly**: `wasm-bindgen` for JavaScript interoperability, with SIMD (Single Instruction, Multiple Data) acceleration for performance.
- **Frontend (JavaScript SDK)**: TypeScript with WASM bindings for browser-based and Node.js environments.
- **Data Persistence**: SQLite for robust and reliable state management.
- **Communication Protocols**: WebSocket and SharedMemory for real-time, efficient inter-agent communication. MCP (JSON-RPC 2.0) for external integration.
- **Deployment**: Designed for flexible deployment across various environments, including Docker, Kubernetes, and edge computing devices.

## Component Interaction
- The `ruv-swarm` npm package acts as the primary JavaScript/TypeScript interface for users, loading WASM modules compiled from `ruv-swarm-wasm` to execute high-performance AI tasks.
- The `ruv-swarm-mcp` crate can be run as a separate server, providing Model Context Protocol functionalities that the npm package or other external systems can interact with over network protocols like WebSockets.
- Agents within the swarm communicate and coordinate using the defined transport and persistence layers, enabling collective learning and problem-solving.
- The CLI tools provide a direct interface for managing and monitoring the swarm, including deploying agents, orchestrating tasks, and running benchmarks.
