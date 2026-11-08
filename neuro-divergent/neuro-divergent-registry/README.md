# neuro-divergent-registry

Describe, categorize, and look up forecasting model metadata. The crate includes discovery and factory interfaces, but generic model creation and native executable plugin loading are unfinished and return unsupported-operation errors. A catalog entry is metadata, not proof that its architecture is implemented.

![Model selection belongs inside the past-only training period](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/neuro-divergent/docs/assets/forecasting-flow.svg)

## Read the catalog

```rust
use neuro_divergent_registry::{initialize_registry, ModelFactory};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_registry()?;
    for model in ModelFactory::list_models() {
        println!("{}", model.name);
    }
    Ok(())
}
```

The registry supports categories, aliases, tags, and descriptive capabilities. Discovery and performance fields include scaffolding; do not use them as benchmark evidence. The registry's model trait is separate from the other forecasting crates' traits.

## Plugin descriptors

Plugin descriptor reads enforce canonical allowed and blocked paths, regular-file checks, and a 1 MiB limit. Blocked paths take precedence, including symlink resolutions. Native-loader entry points check both the binary and neighboring descriptor path. Compatibility uses semantic version ordering and rejects malformed version bounds. Checksum mode requires a SHA-256 checksum and verifies the referenced file using a bounded streaming buffer. Signature and sandbox modes fail closed because those protections have not been implemented.

A checksum supplied in a mutable descriptor does not authenticate its author. Use trusted descriptors and directories. Path checks do not protect against a hostile process concurrently replacing local files. Native plugin code is not executed by the current loader.

## Validate

From the repository root:

```sh
cargo test --manifest-path neuro-divergent/Cargo.toml -p neuro-divergent-registry --all-features --lib --locked
```

Tests cover catalog behavior, path restrictions, symlink escape, oversized descriptors, checksum requirements and tampering, and unsupported security modes. `plugin-system` enables loader dependencies, not a completed native plugin runtime. The crate is prepared as a 0.2.0 candidate; see the [family review](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-forecasting.md) for exact results and remaining work.
