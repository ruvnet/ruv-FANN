# ruv-swarm-benchmarking

Experimental tools for running configured Claude CLI scenarios, collecting event data, storing results in SQLite and comparing observations. This checkout has no registered crates.io release for this package.

## Work from the source checkout

```sh
cargo test --locked -p ruv-swarm-benchmarking --lib
cargo doc --locked -p ruv-swarm-benchmarking --no-deps
```

Start with `BenchmarkConfig`, `BenchmarkingFramework` and the `BenchmarkStorage` API. Choose an explicit executable, working directory and database path for any real execution. Process-launching paths run application commands; they are not a sandbox.

## Interpret results carefully

The event parser and SQLite storage have unit coverage, and statistical helpers can analyze repeated observations. A single baseline/candidate pair cannot establish significance: its comparison now uses `p_value = 1.0`, confidence interval `(0,0)` and effect size 0 as **not-calculated sentinels**. Empty trial sets return finite empty summaries instead of panicking. Use repeated real runs and inspect assumptions before reporting confidence intervals or effect sizes.

The comprehensive SWE evaluation entry point returns an explicit unsupported-integration error; it does not fabricate five trained model evaluations. Some executor modes, resource metrics, quality scores and dashboard paths remain prototypes or heuristics. A mode named `MLOptimized` is not proof that a trained model was used. Tests do not demonstrate a speedup, token saving, code quality gain or official SWE-bench solve rate.

The private storage backend uses exactly pinned `sqlx-core = 0.8.6` and `sqlx-sqlite = 0.8.6` with explicit row decoding. This avoids resolving unused MySQL/RSA drivers; keep these semver-exempt SQLx component versions aligned when updating them. Rust 1.88 all-target/all-feature acceptance includes 20 library tests and four integration tests. Storage regressions cover the component-dependency change.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
