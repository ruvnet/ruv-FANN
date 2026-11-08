# ruv-swarm-agents

Turn an application-owned async function into a typed swarm `Agent`. The handler performs the work, retains any state it owns, and returns its original error type.

## Use a function as an agent

Your application needs `ruv-swarm-agents`, `ruv-swarm-core` and Tokio.

```rust
use ruv_swarm_agents::FunctionAgent;
use ruv_swarm_core::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = FunctionAgent::new(
        "double", vec!["compute".into()],
        |number: i32| async move { Ok::<i32, std::io::Error>(number * 2) },
    )?;
    assert_eq!(agent.process(21).await?, 42);
    Ok(())
}
```

`FunctionAgent::new` validates identity and capability names. `with_metadata` adds descriptive metadata without replacing the ID. `process` takes mutable access to the handler, so a single agent does not invoke its handler concurrently.

The crate does not start a model-provider session, execute commands, or supply researcher/coder personas. Connect those services in your handler. Core scheduling and invoking a typed handler are separate application steps.

## Check it

```sh
cargo test --locked -p ruv-swarm-agents
```

Tests exercise stateful repeated calls, unchanged handler errors and identity validation.

The package requires Rust 1.88 and participates in the workspace all-target/all-feature acceptance gate.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
