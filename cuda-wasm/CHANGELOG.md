# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-07-12

### Fixed
- Fixed compilation error in vector_add example related to Clone trait bounds on kernel launch
- Example now properly demonstrates the CUDA-to-Rust transpilation workflow

### Added
- Comprehensive verification summary documentation
- Additional test coverage for example code

## [0.1.0] - 2025-07-12

### Initial Release
- CUDA to Rust/WASM transpiler
- WebGPU backend support  
- CPU fallback implementation
- Memory pooling and optimization
- Neural network integration with ruv-FANN
- Comprehensive profiling tools
- Example projects and documentation
- TypeScript bindings
- CLI tools for transpilation

### Features
- 🔄 CUDA to Rust/WebAssembly transpilation
- ⚡ WebGPU native browser GPU acceleration
- 🦀 Memory-safe GPU programming with Rust
- 📊 Built-in profiling and optimization
- 🔧 Simple CLI interface
- 🌐 Cross-platform compatibility
- 🧠 Neural network GPU acceleration

[0.1.1]: https://github.com/ruvnet/ruv-FANN/releases/tag/cuda-wasm-v0.1.1
[0.1.0]: https://github.com/ruvnet/ruv-FANN/releases/tag/cuda-wasm-v0.1.0