# ruv-swarm-daa

Experimental autonomous-agent coordination types, learning state, feedback handling and resource telemetry. The crate provides a `StandardDAAAgent` builder and traits for applications that want to explore coordination policies.

## Create an agent

```rust
use ruv_swarm_daa::{CognitivePattern, StandardDAAAgent};

async fn create_agent() -> Result<(), Box<dyn std::error::Error>> {
    let _agent = StandardDAAAgent::builder()
        .with_learning_rate(0.001)
        .with_cognitive_pattern(CognitivePattern::Adaptive)
        .build().await?;
    Ok(())
}
```

Cognitive pattern names select local policy/configuration behavior. They do not provide a trained language model or establish autonomous coding ability. Learning and neural components include heuristic/prototype implementations and need application-specific evaluation.

Unused `tch` and Candle dependencies were removed: they were never connected to this crate's execution path and pulled vulnerable archive/compression dependencies into the lockfile. The `neural-networks` feature remains an empty compatibility marker. It does not enable an external tensor engine.

## Check it

```sh
cargo test --locked -p ruv-swarm-daa --lib
cargo check --locked -p ruv-swarm-daa --bins
```

Rust 1.88 native all-target/all-feature acceptance includes 16 library tests and two coordinator binary tests, including checked resource-accounting overflow. The standalone `--no-default-features --features wasm --lib --target wasm32-unknown-unknown` path compiles; that check does not establish browser execution or learning quality. This package is MIT-licensed.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
