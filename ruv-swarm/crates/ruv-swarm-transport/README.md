# ruv-swarm-transport

Move serialized swarm messages through in-process channels or WebSockets. A shared-memory prototype also provides a synchronized ring buffer.

## Start with in-process communication

```rust
use ruv_swarm_transport::in_process::{InProcessRegistry, InProcessTransport};
use ruv_swarm_transport::{Transport, TransportConfig};

async fn endpoints() -> Result<(), Box<dyn std::error::Error>> {
    let registry = InProcessRegistry::new();
    let first = InProcessTransport::new("a".into(), TransportConfig::default(), registry.clone()).await?;
    let second = InProcessTransport::new("b".into(), TransportConfig::default(), registry).await?;
    assert_ne!(first.local_address()?, second.local_address()?);
    Ok(())
}
```

Import `ruv_swarm_transport::Transport` to call trait methods such as `send`, `receive`, `local_address` and `close`.

## Limits and behavior

- The default message budget is 10 MiB. Senders and receivers enforce their configured budgets; gzip decoding checks the expanded length before accepting a message.
- WebSocket handshakes, frames and messages use configured time/size limits. Gzip detection uses its header, so highly compressible payloads remain decodable.
- Duplicate in-process endpoint IDs are rejected. Broadcast reaches the ordinary `receive` API. Slow receivers apply channel backpressure.
- The ring buffer serializes complete read/write operations, including cursors, and copies wrapping slices in bulk. It is an in-process synchronized buffer. The current transport does not map the ring's data into the OS shared-memory segment, so it does not provide cross-process delivery.

Peer authentication, TLS deployment, reconnect policy and application-level delivery acknowledgments remain application responsibilities. Native all-target/all-feature acceptance passes on Rust 1.88. The `wasm` feature compiles and two tests execute in-process messages, broadcast, duplicate IDs and receiver limits as actual WASM under Node. Native WebSocket and shared-memory backends are not compiled for browsers. Serialization compatibility depends on the selected protocol version and codec. The registry drops its map guard before awaiting a transport builder; a regression checks concurrent replacement.

## Check it

```sh
cargo +1.88.0 test --locked -p ruv-swarm-transport --all-targets --all-features
cargo +1.88.0 check --locked -p ruv-swarm-transport --no-default-features \
  --features wasm --lib --target wasm32-unknown-unknown
```

Tests cover concurrent producers/consumers, full and wrapped rings, bounded gzip decoding, receiver limits, duplicate endpoints and broadcast delivery.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
