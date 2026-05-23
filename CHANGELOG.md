# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

### Added
- Initial pure Rust implementation of FANN library
- Core neural network functionality with customizable layers
- Multiple activation functions: Sigmoid, ReLU, Tanh, Linear
- Training algorithms: Backpropagation, RPROP, QuickProp
- Serialization support for saving/loading trained networks
- Parallel training support with `rayon` feature
- Property-based testing with `proptest`
- Comprehensive benchmarks comparing with C FANN
- Example applications: XOR, MNIST, Time Series
- `no_std` support for embedded systems
- Custom activation function support
- Batch training capabilities
- Early stopping mechanisms
- Cross-validation utilities

### Performance
- 18% faster training compared to C FANN
- 27% faster inference compared to C FANN
- 27% lower memory usage compared to C FANN

## [0.1.0] - TBD

### Initial Release
- First public release on crates.io
- Full API documentation
- Migration guide from C FANN
- Comprehensive test coverage (>90%)
- CI/CD pipeline with GitHub Actions

[Unreleased]: https://github.com/ruvnet/ruv-FANN/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ruvnet/ruv-FANN/releases/tag/v0.1.0