# ruv-swarm-core

Core types for registering agents, describing tasks and choosing which agent receives each task.

The scheduler supports capability matching, load-based assignment, persistent round robin and stable priority ordering. Duplicate agent IDs are rejected. A task assignment identifies a worker; your application is responsible for running that worker and recording the result.

## Try it

From `ruv-swarm/`:

```sh
cargo run --locked -p ruv-swarm-core --example basic_swarm
cargo test --locked -p ruv-swarm-core --lib
```

The example registers two workers and assigns three tasks in this order: `worker-a`, `worker-b`, `worker-a`. Its complete source is in `examples/basic_swarm.rs` and is included in the crate package.

```rust
use ruv_swarm_core::{DistributionStrategy, Swarm, SwarmConfig};

let swarm = Swarm::new(SwarmConfig {
    distribution_strategy: DistributionStrategy::RoundRobin,
    ..Default::default()
});
```

## Scope

Rust 1.88 native acceptance covers all targets and Cargo features, including 173 library tests and eight Criterion smoke cases. Minimal, `no_std` and standalone `--no-default-features --features wasm --target wasm32-unknown-unknown` configurations also compile. These compilation checks do not validate an embedded device or browser runtime. Topology describes connections in memory and does not create a network service. Automatic scaling, health monitoring and concurrent task processing include experimental behavior; they do not provide a complete distributed worker runtime.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
