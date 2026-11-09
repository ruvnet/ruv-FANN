# Changelog

Changes follow [Semantic Versioning](https://semver.org/). Versions in a checkout
are release candidates until publication is verified in the registry.

## [0.2.2] - Unreleased

### Performance

- `PreparedNetwork::run_into` reads a dense first layer's inputs in place
  instead of copying them into the workspace and reading them back, and
  activation dispatch is inlined. Single-row scoring of a `10 -> 1` sigmoid
  head went from 35.4 ns to 19.5 ns (x86-64 Windows, criterion,
  `benches/prepared_single_row.rs`); a `10 -> 4 -> 1` head is unchanged within
  noise. Results, error reporting and the output-commit rule are bit-identical
  (`tests/prepared_direct_first_layer.rs`). Sparse first layers keep the copy.

### Fixed

- `gpu_sweet_spot_benchmark` declares `required-features = ["gpu"]`, so a plain
  `cargo test` or `cargo check --examples` on default features compiles.

### Security

- Check SIMD slice lengths and dimension products in release builds before
  entering unsafe vector kernels. Dispatch AVX2/FMA only when supported.
- Bound model/training-data headers, text lines and decompressed output before
  allocating. Validate binary connection indices against the previous layer.
- Reject invalid public network topology at inference boundaries.
- Update the root lock to patched Crossbeam and Tokio versions; remove the stale
  Rand soundness-advisory exception. Audit every workspace lock independently.
- Restrict the core crate archive to library sources, examples, Rust tests,
  documentation assets and package metadata; exclude agent state and sibling
  projects.

### Performance and correctness

- Add `Network::run_into` for repeated inference into a caller-owned output
  slice. Regression tests prove zero allocations on this path and one final
  output allocation for `run`; no latency speedup is asserted.
- Borrow neighboring layer neurons during inference instead of allocating
  intermediate value buffers for each neuron/layer.
- Implement the scalar GELU, Swish and other exposed activation variants and
  correct SIMD gradient outer-product dimensions.

### Documentation and validation

- Replace unverified performance and compatibility promises in the root README
  with a working example, package map, feature limits and reproducible commands.
- Add accessible animated SVG diagrams with reduced-motion fallbacks and a
  primary-source research report.
- Add a verification driver for all five workspaces, binary/regression tests
  and dependency audits. CI now uses locked dependency resolution, tests pull
  requests, and reports failures without auto-fixing source or suppressing them.
- Replace the duplicate automatic npm publishing paths with a version-matched
  manual release workflow gated by repository and package validation.

This entry describes the core candidate and repository tooling. See
[the modernization report](docs/modernization.md) and family reports for changes,
behavior migrations and release blockers in the other packages. The previous
unreleased draft's speedup, memory and coverage numbers had no reproduced
comparison and are omitted.

## [0.2.1] - 2026-05-23

### Security
- Updated `bytes` transitive dependency to 1.11.1 (fixes RUSTSEC-2026-0007:
  integer overflow in `BytesMut::reserve`)
- Updated `slab` transitive dependency to 0.4.12 (fixes RUSTSEC-2025-0047:
  out-of-bounds access in `get_disjoint_mut`)
- Bumped minimum `tokio` to 1.44 to pull the patched `bytes` version
- Added pre-parse bounds check in FANN format parser: `layer_sizes` and
  `weights` sections are now limited to 10,000 and 100,000,000 entries
  respectively, preventing memory exhaustion from crafted files

### Fixed
- **Compile error with `--all-features`**: `shaders.rs` match on
  `ActivationFunction` was non-exhaustive — missing arms for `Gelu`, `Swish`,
  and `LeakyRelu` caused a build failure; mapped to existing GPU shader types
- **Adam L2 weight-decay bug**: the decay term previously applied a constant
  `learning_rate * lambda` offset to all weight updates instead of scaling by
  the actual weight value (`learning_rate * lambda * w`), violating L2
  regularisation semantics
- Replaced `lazy_static` macro with `std::sync::OnceLock` in
  `memory_manager.rs`; removed `lazy_static` dependency entirely

### Removed
- `lazy_static` dependency (replaced by `std::sync::OnceLock`, stable since
  Rust 1.70)

### Tests
- Added 9 new tests in `src/tests/network_tests.rs` that verify *computed
  output values* (not just structural properties): ReLU ordering, sigmoid/tanh
  range + non-NaN guarantees, output dimension correctness, input validation,
  deterministic forward pass, and Adam weight-decay activation

[0.2.2]: https://github.com/ruvnet/ruv-FANN/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/ruvnet/ruv-FANN/releases/tag/v0.2.1
