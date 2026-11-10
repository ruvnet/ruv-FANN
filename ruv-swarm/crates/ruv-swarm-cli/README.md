# ruv-swarm-cli

A native command-line interface for configuring and exploring swarm operations. The executable is named `ruv-swarm`.

## Run from this checkout

```sh
cargo run --locked -p ruv-swarm-cli -- --help
cargo run --locked -p ruv-swarm-cli -- init --help
cargo test --locked -p ruv-swarm-cli
```

The help output is the source of truth for command names and flags. Configuration can be selected with `--config` or `RUV_SWARM_CONFIG`; profiles and output formatting are available through global options.

## Configuration handling

Loading validates nonzero limits and timeouts, minimum/maximum agent bounds and HTTP(S) API endpoints. API keys can be supplied at runtime or read from existing configuration, but saving configuration omits the key and debug output redacts it. This does not remove credentials that were already saved by older versions.

Several commands currently display local or simulated state. They do not deploy a distributed worker service, create a model-provider session or prove that a task was executed. Use the core and agent APIs when building an application worker loop. PostgreSQL/Redis command labels are not verified backend implementations.

The tests cover invalid configuration and omitted credentials. CLI help/build checks validate the binary surface; they do not validate a complete external service deployment.

The package requires Rust 1.88 and participates in the workspace all-target/all-feature acceptance gate.

Published on [crates.io](https://crates.io/crates/ruv-swarm-cli). Builds use rustls for TLS; no OpenSSL dependency. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
