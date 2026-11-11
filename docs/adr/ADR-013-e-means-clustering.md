# ADR-013: Shared e-means input clustering with RuVector

Status: Implemented source architecture; release requires publishing the shared dependencies first.
Date: 2026-11-11.
Reference: Bessa et al., [e-means: outperforming k-means (in one-tenth the epochs), arXiv:2611.09501v1](https://arxiv.org/abs/2611.09501v1).

## Context

ruv-FANN applications need to group training inputs and train representative codebooks. RuVector needs the same operation for vector quantization. Maintaining separate clustering algorithms would allow numerical behavior, validation and defaults to diverge between the two projects and their JavaScript interfaces.

The paper describes mini-batch centroid updates using exponentially weighted counts and sums, cosine scheduling, donor splitting to revive inactive clusters, and last-epoch mean finalization. Its batch-size and peak-step scaling laws provide defaults without application-specific tuning. The published performance measurements use different hardware and datasets from this implementation; they do not establish local CPU or WASM speedups.

## Decision

Use the independent `ruvector-e-means` Rust crate, version `0.1.0`, owned by the RuVector repository. Enable it through the optional `clustering` Cargo feature. Export its `fit`, `Config`, `Finalization`, `Model` and `Error` interfaces from `ruv_fann::clustering`.

Add `TrainingData<f32>::cluster_inputs(&Config)` as the bridge from nested training input rows to the shared flat row-major API. Validate that rows are nonempty and rectangular before flattening. Delegate configuration, finite-coordinate and clustering validation to the shared crate. Return a distinct adapter error for invalid input shapes. Outputs do not participate in this unsupervised operation; the method borrows data without changing inputs, outputs or network weights.

Add the explicit `ruv-fann/clustering` npm subpath, backed by an exact `@ruvector/e-means@0.1.0` dependency. Its JavaScript and TypeScript modules re-export the shared package directly, including asynchronous `init` and synchronous `fit`/`predict` after initialization. The npm package runs the same Rust algorithm compiled to WASM. The root neural inference and training entrypoint does not eagerly initialize clustering.

The initial scope includes exact squared-Euclidean assignment, seeded reproducibility, automatic scaling-law parameters, EMA updates, cosine annealing, dead-cluster revival and last-epoch mean finalization. The alternative `ema` finalization exposes the terminal EMA centroids. Slow-EMA finalization, approximate assignment and GPU kernels remain outside this first implementation.

## Alternatives

Embedding a second Rust implementation in ruv-FANN would duplicate maintenance and numerical testing. A JavaScript rewrite would also introduce a second algorithm and different random-number behavior. Both lose the shared implementation boundary.

Replacing backpropagation, RProp or AdamW with clustering would change the supervised learning contract. The new feature instead supports input analysis and codebook training through a separate API. Applications choose explicitly how to use the resulting assignments and centroids.

## Dependencies and release order

Cargo uses a version-only dependency and npm uses an exact registry version. There are no committed paths to a sibling checkout or unpublished Git revision. Publish `ruvector-e-means@0.1.0` and `@ruvector/e-means@0.1.0`, verify fresh registry installs, then release the consuming ruv-FANN artifacts. Until then, independent dependency resolution requires a local override; passing local checks does not prove registry availability.

For development with sibling checkouts:

```sh
cargo test --features clustering --test clustering_tests \
  --config 'patch.crates-io.ruvector-e-means.path="../ruvector/crates/ruvector-e-means"'
```

For npm validation, build and pack the shared package, install that exact archive in an isolated consumer, and import the packed ruv-FANN `clustering` subpath. An archive override is for qualification only; published dependency metadata retains the registry version.

Rollback removes the optional feature usage and npm subpath usage. Existing supervised inference and training callers have no API migration.

## Acceptance and evidence

| Requirement | Executable evidence |
| --- | --- |
| Use the shared Rust dependency | `tests/clustering_tests.rs` compares adapter and direct shared-crate centroids/assignments |
| Provide useful input clustering | Separated-cluster prediction and one-cluster mean numerical oracles |
| Preserve supervised inputs and targets | Adapter test compares borrowed data before and after clustering |
| Reject malformed boundaries | Empty/ragged rows and non-finite-coordinate regressions |
| Expose the same dependency to npm | `packages/core/test/clustering.test.mjs` imports `ruv-fann/clustering` and executes WASM training/prediction |
| Remain reproducible | Same seed/data/config yields identical centroids |

Focused validation must run against actual crate and WASM bytes. The shared crate owns algorithm-level tests and benchmarks; ruv-FANN owns the adapter and package-boundary checks. Broader Cargo and npm regressions remain required when preparing a release. No accuracy guarantee for arbitrary datasets or reproduction of the paper's GPU throughput is implied.
