# Contributing to ruv-swarm

Thank you for considering contributing to ruv-swarm! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- **Rust 1.75+**: Install via [rustup](https://rustup.rs/)
- **Node.js 16+**: For JavaScript/NPM components
- **wasm-pack**: For WebAssembly builds
  ```bash
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
  ```

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ruvnet/ruv-FANN.git
   cd ruv-FANN/ruv-swarm
   ```

2. **Build the workspace**:
   ```bash
   cargo build --workspace
   ```

3. **Run tests**:
   ```bash
   cargo test --workspace
   ```

4. **Build WASM modules**:
   ```bash
   cd npm && npm run build:wasm
   ```

## Contributing Process

### 1. Issue First

- **Search existing issues** before creating a new one
- **Use issue templates** when available
- **Provide clear descriptions** with reproduction steps for bugs
- **Label appropriately** (bug, enhancement, documentation, etc.)

### 2. Fork and Branch

- **Fork the repository** to your GitHub account
- **Create a feature branch** from `main`:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- **Use descriptive branch names**: `feature/cognitive-patterns`, `fix/memory-leak`, `docs/user-guide`

### 3. Development Guidelines

#### Code Style

- **Follow Rust conventions**: Use `rustfmt` and `clippy`
- **TypeScript standards**: Use ESLint and Prettier for JavaScript/TypeScript
- **Documentation**: Add doc comments for public APIs
- **Error handling**: Use appropriate error types, avoid unwrap() in library code

#### Testing

- **Unit tests**: Test individual components thoroughly
- **Integration tests**: Test component interactions
- **Property-based tests**: Use `proptest` for complex scenarios
- **WASM tests**: Test WebAssembly builds with `wasm-pack test`

#### Performance

- **Benchmark changes**: Use criterion for performance-critical code
- **Profile memory usage**: Avoid unnecessary allocations
- **SIMD optimization**: Use vectorization where appropriate
- **Measure impact**: Document performance improvements/regressions

### 4. Commit Guidelines

#### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or fixing tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(cognitive): add lateral thinking pattern

Add new cognitive pattern for cross-domain problem solving.
Includes pattern implementation, tests, and documentation.

Closes #123
```

```
fix(transport): resolve WebSocket reconnection issue

Fix exponential backoff calculation that caused connection storms.
Add integration test to prevent regression.

Fixes #456
```

#### Commit Atomicity

- **One logical change per commit**
- **Self-contained commits** that build and test successfully
- **Meaningful commit messages** that explain the "why"

### 5. Pull Request Process

#### Before Submitting

- **Run all tests**: `cargo test --workspace && npm test`
- **Check formatting**: `cargo fmt --check && npm run lint`
- **Run clippy**: `cargo clippy --workspace -- -D warnings`
- **Build documentation**: `cargo doc --workspace --no-deps`
- **Update CHANGELOG**: Add entry for significant changes

#### PR Description

Use the PR template and include:

- **Clear title** summarizing the change
- **Problem description** and motivation
- **Solution approach** and alternatives considered
- **Testing performed** and results
- **Breaking changes** if any
- **Screenshots/demos** for UI changes

#### Review Process

- **Automated checks** must pass (CI/CD pipeline)
- **Code review** by maintainers
- **Address feedback** promptly and thoroughly
- **Squash commits** if requested before merge

## Architecture Guidelines

### Cognitive Patterns

When adding new cognitive patterns:

1. **Research basis**: Include scientific references
2. **Clear characteristics**: Define thinking style and use cases
3. **Implementation**: Add to `CognitivePattern` enum
4. **Testing**: Comprehensive tests for pattern behavior
5. **Documentation**: Update user guide with examples

### Transport Layer

For transport implementations:

1. **Async-first**: Use tokio for async operations
2. **Error handling**: Comprehensive error types
3. **Reconnection**: Implement robust reconnection logic
4. **Testing**: Test network failures and edge cases
5. **Performance**: Benchmark throughput and latency

### WASM Integration

For WebAssembly features:

1. **Size optimization**: Minimize bundle size
2. **Browser compatibility**: Test across browsers
3. **Memory management**: Avoid memory leaks
4. **Feature detection**: Graceful fallbacks
5. **TypeScript types**: Accurate type definitions

## Testing Standards

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interaction
├── benchmarks/     # Performance benchmarks
├── fixtures/       # Test data and utilities
└── wasm/          # WebAssembly-specific tests
```

### Test Requirements

- **Coverage**: Aim for >80% code coverage
- **Edge cases**: Test error conditions and limits
- **Concurrency**: Test concurrent operations
- **Property tests**: Use proptest for complex invariants
- **Performance**: Benchmark performance-critical paths

### Benchmark Guidelines

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_agent_spawning(c: &mut Criterion) {
    c.bench_function("spawn_agent", |b| {
        b.iter(|| {
            // Benchmark code here
        });
    });
}

criterion_group!(benches, benchmark_agent_spawning);
criterion_main!(benches);
```

## Documentation Standards

### Code Documentation

- **Public APIs**: Complete rustdoc documentation
- **Examples**: Include usage examples in doc comments
- **Safety**: Document unsafe code thoroughly
- **Panics**: Document when functions can panic

### User Documentation

- **Clear language**: Avoid jargon, explain concepts
- **Complete examples**: Working code snippets
- **Common patterns**: Document typical usage
- **Troubleshooting**: Address common issues

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

1. **Update version numbers** in all Cargo.toml files
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite** including benchmarks
4. **Build all targets** (native + WASM)
5. **Test NPM package** installation and usage
6. **Create release PR** with version bump
7. **Tag release** after merge
8. **Publish crates** to crates.io
9. **Publish NPM** package

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussion
- **Discord**: Real-time chat and support
- **Twitter**: Updates and announcements

### Getting Help

- **Documentation**: Check docs.rs and user guide first
- **Search issues**: Look for existing discussions
- **Ask questions**: Use GitHub Discussions for help
- **Join Discord**: Get real-time assistance

## Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: All contributors listed
- **Release notes**: Significant contributions highlighted
- **Git history**: All commits properly attributed

## License

By contributing to ruv-swarm, you agree that your contributions will be licensed under the same MIT OR Apache-2.0 license that covers the project.

---

Thank you for contributing to ruv-swarm! Your efforts help make distributed AI agent orchestration accessible to everyone.