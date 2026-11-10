# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - Unreleased

### Package metadata

- Set `homepage` to https://ruv.io. No code or dependency change.

## [2.0.0] - Unreleased

### Breaking changes

- Reimplement the npm package as a Node.js wrapper around a genuine external `cuda2rust` >=0.2.0 executable. The executable is an explicit prerequisite; installation downloads and builds nothing.
- Return generated Rust or WGSL source text. Remove browser/WASM exports, native FFI bindings, claimed kernel execution, synthetic benchmark comparisons and unsupported analysis APIs from the distribution.
- `transpileCuda` now accepts source text and returns a promise of `{ code, target, compilerVersion }`; use `transpileFile` for paths. Targets are `rust` and `wgsl`.
- Require Node.js >=22.12.0. Remove npm runtime/development dependencies from this package.

### Added

- Working CommonJS, ES module, TypeScript declaration and CLI entry points.
- Input/UTF-8, compiler-version, timeout and output bounds; fixed argument arrays without a shell; private temporary input cleanup; structured errors.
- CLI output-file protection, package artifact checks, executable-fixture tests and an explicit integration gate against the real Rust CLI.
- Keep legacy native/WASM source scripts in the repository but exclude them from npm packaging. See the README migration table.

## [1.0.0] - 2024-12-12

### Added
- Initial release of @cuda-wasm/core
- CUDA to WebAssembly transpilation
- WebGPU compute shader support
- Rust-based transpiler with memory safety
- Performance analysis and benchmarking tools
- Cross-platform support (Linux, macOS, Windows)
- Browser and Node.js compatibility
- TypeScript definitions
- Comprehensive test suite
- CLI tool for kernel transpilation
- Example projects and documentation

### Features
- 🔄 CUDA to WebAssembly transpilation
- ⚡ WebGPU native browser GPU acceleration
- 🦀 Memory-safe GPU programming with Rust
- 📊 Built-in profiling and optimization
- 🔧 Simple CLI interface
- 🌐 Cross-platform compatibility
- 📝 Full TypeScript support

### Supported Targets
- WebAssembly (WASM)
- WebGPU compute shaders
- Native GPU backends (CUDA, OpenCL, Vulkan)

### CLI Commands
- `cuda-wasm transpile` - Convert CUDA kernels to WASM/WebGPU
- `cuda-wasm analyze` - Analyze kernel performance characteristics
- `cuda-wasm benchmark` - Run performance benchmarks

### API Features
- `transpileCuda()` - Main transpilation function
- `analyzeKernel()` - Kernel analysis and optimization suggestions
- `benchmark()` - Performance benchmarking
- `createWebGPUKernel()` - WebGPU kernel creation and execution

### Installation Methods
- NPX: `npx @cuda-wasm/core`
- Global: `npm install -g @cuda-wasm/core`
- Project: `npm install @cuda-wasm/core`

[1.0.0]: https://github.com/ruvnet/ruv-FANN/releases/tag/cuda-wasm-v1.0.0