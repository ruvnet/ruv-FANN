# ruv-swarm-transport

[![Crates.io](https://img.shields.io/crates/v/ruv-swarm-transport.svg)](https://crates.io/crates/ruv-swarm-transport)
[![Documentation](https://docs.rs/ruv-swarm-transport/badge.svg)](https://docs.rs/ruv-swarm-transport)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/ruv-FANN#license)

**High-performance transport layer for distributed agent communication in the RUV-FANN swarm intelligence framework.**

The `ruv-swarm-transport` crate provides a unified, async-first networking abstraction designed specifically for intelligent agent swarms. It delivers multiple transport backends optimized for different deployment scenarios, from high-throughput in-process communication to secure networked connections and low-latency shared memory transfers.

## 🚀 Key Features

### **Multiple Transport Backends**
- **🌐 WebSocket Transport**: Full-duplex communication with TLS support, auto-reconnection, and connection pooling
- **🔗 Shared Memory Transport**: Zero-copy message passing with lock-free ring buffers for maximum throughput
- **⚡ In-Process Transport**: Ultra-low latency channels for co-located agents
- **🌍 WASM Support**: Browser and Web Workers compatibility with SharedArrayBuffer integration

### **Advanced Message Protocol**
- **Request/Response Pattern**: Correlated messaging with automatic timeout handling
- **Event Notifications**: Pub/sub messaging with topic filtering
- **Broadcast Messaging**: Efficient one-to-many communication
- **Heartbeat Monitoring**: Automatic connection health checking
- **Protocol Versioning**: Backward-compatible protocol negotiation

### **Performance Optimizations**
- **Zero-Copy Operations**: Direct memory access in shared memory transport
- **Message Compression**: Automatic gzip compression with configurable thresholds
- **Binary Serialization**: MessagePack for efficient wire format (10x faster than JSON)
- **Lock-Free Data Structures**: High-concurrency ring buffers and atomic operations
- **Connection Pooling**: Reusable WebSocket connections with backpressure handling

### **Enterprise-Grade Reliability**
- **Automatic Reconnection**: Exponential backoff with jitter for WebSocket connections
- **Flow Control**: Backpressure and congestion management
- **Message Ordering**: FIFO guarantees within channels
- **Error Recovery**: Graceful degradation and retry mechanisms
- **Comprehensive Monitoring**: Built-in statistics and health metrics

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruv-swarm-transport = "0.1.0"

# Optional: Enable specific backends
ruv-swarm-transport = { version = "0.1.0", features = ["websocket", "shared-memory", "wasm"] }
```

### Feature Flags

- `websocket` (default): WebSocket transport implementation
- `shared-memory` (default): Shared memory transport for IPC
- `wasm`: Web Assembly support for browser environments

## 🎯 Usage Examples

### In-Process Transport (Ultra-Low Latency)

Perfect for co-located agents requiring microsecond-level communication:

```rust
use ruv_swarm_transport::{Transport, TransportConfig, in_process::InProcessTransport, protocol::Message};

// Create connected transport pair
let config = TransportConfig::default();
let (mut agent1, mut agent2) = InProcessTransport::create_pair(
    "agent_1".to_string(),
    "agent_2".to_string(), 
    config,
).await?;

// Send structured message
let request = Message::request(
    "agent_1".to_string(),
    "compute_fibonacci".to_string(),
    serde_json::json!({"n": 42})
);

agent1.send("agent_2", request).await?;

// Receive and process
let (sender, message) = agent2.receive().await?;
println!("Received from {}: {:?}", sender, message);
```

### WebSocket Transport (Network Communication)

Ideal for distributed agents across network boundaries:

```rust
use ruv_swarm_transport::{websocket::{WebSocketTransport, WsMode}, TransportConfig};

// Server setup
let server = WebSocketTransport::new(
    WsMode::Server { 
        bind_addr: "0.0.0.0:8080".to_string() 
    },
    TransportConfig::default()
).await?;

// Client connection with TLS
let client = WebSocketTransport::new(
    WsMode::Client { 
        url: "wss://swarm-hub.example.com:8080".to_string() 
    },
    TransportConfig {
        connection_timeout_ms: 10000,
        retry_attempts: 5,
        enable_compression: true,
        ..Default::default()
    }
).await?;

// Broadcast to all connected agents
let announcement = Message::broadcast(
    "coordinator".to_string(),
    "task_assignment".to_string(),
    serde_json::json!({
        "task_id": "neural_training_batch_001",
        "model_params": {...},
        "deadline": "2024-01-15T10:00:00Z"
    })
);

client.broadcast(announcement).await?;
```

### Shared Memory Transport (High Throughput)

Optimized for high-volume message passing between processes:

```rust
use ruv_swarm_transport::{shared_memory::{SharedMemoryTransport, SharedMemoryInfo}};

let memory_config = SharedMemoryInfo {
    name: "neural_swarm_shared".to_string(),
    size: 100 * 1024 * 1024, // 100MB shared segment
    ring_buffer_size: 10 * 1024 * 1024, // 10MB ring buffer
};

let transport = SharedMemoryTransport::new(
    memory_config,
    TransportConfig {
        max_message_size: 50 * 1024 * 1024, // 50MB max message
        enable_compression: false, // Disable for raw throughput
        ..Default::default()
    }
).await?;

// High-frequency neural network weight updates
for epoch in 0..1000 {
    let weight_update = Message::event(
        "trainer_node".to_string(),
        "weights_updated".to_string(),
        serde_json::json!({
            "epoch": epoch,
            "weights": generate_weight_matrix(), // Large tensor data
            "loss": 0.001234
        })
    );
    
    transport.broadcast(weight_update).await?;
}
```

## 🔧 Advanced Configuration

### Custom Transport Configuration

```rust
let config = TransportConfig {
    max_message_size: 100 * 1024 * 1024,    // 100MB max
    connection_timeout_ms: 30000,           // 30 second timeout
    retry_attempts: 10,                     // Aggressive retry
    enable_compression: true,               // Enable compression
    compression_threshold: 4096,            // Compress >4KB messages
};
```

### Message Protocol Customization

```rust
use ruv_swarm_transport::protocol::{Message, MessageType, ProtocolVersion};

// Create versioned request with headers
let message = Message::request(
    "neural_coordinator".to_string(),
    "distribute_training".to_string(),
    serde_json::json!({
        "dataset": "imagenet_subset",
        "batch_size": 256,
        "learning_rate": 0.001
    })
)
.with_header("Authorization".to_string(), "Bearer <token>".to_string())
.with_header("Content-Encoding".to_string(), "gzip".to_string())
.with_priority(255) // Highest priority
.with_ttl(10);      // 10 hop limit
```

## 📊 Protocol Documentation

### Message Types

| Type | Purpose | Response Required | Use Case |
|------|---------|-------------------|----------|
| **Request** | Command execution | Yes | RPC calls, data queries |
| **Response** | Request acknowledgment | No | Results, acknowledgments |  
| **Event** | State notifications | No | Status updates, alerts |
| **Broadcast** | Topic-based messaging | No | Global announcements |
| **Heartbeat** | Connection monitoring | No | Health checking |
| **Control** | Protocol management | Varies | Handshakes, flow control |

### Protocol Handshake

```rust
// Client initiates handshake
let hello = Message::new("client_agent".to_string(), MessageType::Control {
    operation: ControlOperation::Hello {
        version: ProtocolVersion::CURRENT,
        capabilities: vec!["compression".to_string(), "binary_codec".to_string()],
        metadata: [("agent_type".to_string(), "neural_trainer".to_string())].into(),
    }
});

// Server responds with negotiated capabilities
let hello_ack = Message::new("server_hub".to_string(), MessageType::Control {
    operation: ControlOperation::HelloAck {
        version: ProtocolVersion::CURRENT,
        capabilities: vec!["compression".to_string()], // Intersection of capabilities
        metadata: [("hub_region".to_string(), "us-west-2".to_string())].into(),
    }
});
```

## ⚡ Performance Benchmarks

Performance characteristics on typical hardware (Intel i7-10700K, 32GB RAM):

| Transport | Latency (avg) | Throughput | Memory Usage | CPU Usage |
|-----------|---------------|------------|--------------|-----------|
| **In-Process** | ~500ns | 2M+ msgs/sec | Minimal | <1% |
| **Shared Memory** | ~2μs | 1M+ msgs/sec | ~100MB | 2-5% |
| **WebSocket (local)** | ~50μs | 100K msgs/sec | ~10MB | 5-10% |
| **WebSocket (network)** | ~10ms* | 10K msgs/sec | ~50MB | 10-15% |

*Network latency dependent

### Run Benchmarks

```bash
# Run all transport benchmarks
cargo bench

# Specific benchmark suites  
cargo bench --bench transport_benchmarks -- in_process
cargo bench --bench transport_benchmarks -- shared_memory
cargo bench --bench transport_benchmarks -- websocket
```

### Message Size Performance

```rust
// Benchmark different message sizes
for size in &[1_KB, 10_KB, 100_KB, 1_MB, 10_MB] {
    let data = vec![0u8; *size];
    let msg = Message::event("bench".to_string(), "test".to_string(), 
                           serde_json::to_value(&data)?);
    
    let start = Instant::now();
    transport.send("target", msg).await?;
    let duration = start.elapsed();
    
    println!("{}KB: {:?}", size / 1024, duration);
}
```

## 🔒 Security & Reliability

### TLS Configuration for WebSocket

```rust
let secure_client = WebSocketTransport::new(
    WsMode::Client { 
        url: "wss://secure-swarm.company.com:443".to_string() 
    },
    TransportConfig {
        connection_timeout_ms: 15000,
        retry_attempts: 3,
        // TLS handled automatically by tokio-tungstenite
        ..Default::default()
    }
).await?;
```

### Error Handling & Recovery

```rust
use ruv_swarm_transport::TransportError;

match transport.send("target_agent", message).await {
    Ok(()) => println!("Message sent successfully"),
    Err(TransportError::ConnectionError(e)) => {
        // Handle connection issues with retry
        eprintln!("Connection failed: {}", e);
        tokio::time::sleep(Duration::from_secs(1)).await;
        // Automatic reconnection will be attempted
    },
    Err(TransportError::MessageTooLarge { size, max }) => {
        eprintln!("Message {} bytes exceeds limit {}", size, max);
        // Implement message chunking or compression
    },
    Err(e) => eprintln!("Transport error: {}", e),
}
```

## 🌐 WASM Support

### Browser Integration

```rust
#[cfg(target_arch = "wasm32")]
use ruv_swarm_transport::wasm::WebWorkerTransport;

// Enable SharedArrayBuffer for high-performance communication
let transport = WebWorkerTransport::new(
    worker_handle,
    TransportConfig::default()
).await?;

// Works seamlessly with same Transport trait
transport.send("web_worker_agent", message).await?;
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Integration tests  
cargo test --test integration_test

# Test specific transport
cargo test websocket
cargo test shared_memory
cargo test in_process
```

## 📚 Documentation

- **[API Documentation](https://docs.rs/ruv-swarm-transport)**
- **[Examples](/examples)** - Complete usage examples
- **[Benchmarks](/benches)** - Performance testing suites
- **[Integration Tests](/tests)** - End-to-end validation

## 🔗 Links

- **Main Repository**: [github.com/ruvnet/ruv-FANN](https://github.com/ruvnet/ruv-FANN)
- **RUV-FANN Documentation**: [ruv-FANN Wiki](https://github.com/ruvnet/ruv-FANN/wiki)
- **Issue Tracker**: [GitHub Issues](https://github.com/ruvnet/ruv-FANN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/ruv-FANN/discussions)

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guide](https://github.com/ruvnet/ruv-FANN/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/ruvnet/ruv-FANN/blob/main/CODE_OF_CONDUCT.md).

### Development Setup

```bash
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm/crates/ruv-swarm-transport
cargo build
cargo test
```

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🎯 Roadmap

- [ ] **QUIC Transport**: UDP-based transport with multiplexing
- [ ] **gRPC Integration**: Native gRPC transport backend  
- [ ] **Redis Transport**: Pub/sub via Redis for scalable messaging
- [ ] **Kubernetes Integration**: Service mesh transport discovery
- [ ] **Metrics Export**: Prometheus/OpenTelemetry integration
- [ ] **Message Encryption**: End-to-end encryption support

---

**Created by rUv** - Building the future of distributed intelligence 🧠✨

*Part of the RUV-FANN (Robust Unified Virtual Functional Artificial Neural Network) ecosystem - enabling adaptive, self-organizing swarm intelligence systems.*