# Release Process for ruv-swarm

This document outlines the complete release process for publishing both Rust crates and the NPM package.

## Pre-Release Checklist

### 1. Code Quality
- [ ] All tests pass: `cargo test --all` and `npm test`
- [ ] No clippy warnings: `cargo clippy --all -- -D warnings`
- [ ] No eslint errors: `npm run lint:check`
- [ ] Documentation is up-to-date
- [ ] Examples run without errors
- [ ] CHANGELOG.md is updated with all changes

### 2. Version Consistency
- [ ] All Rust crates have the same version in Cargo.toml (currently 0.2.0)
- [ ] NPM package.json version matches Rust crates (currently 0.2.0)
- [ ] Git tag is ready to be created: `v0.2.0`

### 3. Security & Performance
- [ ] Security audit passes: `cargo audit` and `npm audit`
- [ ] Performance benchmarks meet targets
- [ ] WASM size is optimized
- [ ] No unsafe code without proper documentation

## Publishing Process

### Phase 1: Final Validation

```bash
# Run comprehensive validation
cd /workspaces/ruv-FANN/ruv-swarm

# Rust validation
cargo test --all --release
cargo clippy --all -- -D warnings
cargo fmt --all -- --check
cargo audit

# NPM validation
cd npm
npm run test:comprehensive
npm run lint:check
npm audit
```

### Phase 2: Rust Crates Publishing

The Rust crates must be published in dependency order:

```bash
# Run the automated script
./scripts/publish-rust.sh
```

Or manually publish in this order:
1. `ruv-swarm-core` - Core traits and abstractions
2. `ruv-swarm-transport` - Network transport layer
3. `ruv-swarm-persistence` - Storage and state management
4. `ruv-swarm-agents` - Agent implementations
5. `ruv-swarm-ml` - Machine learning components
6. `claude-parser` - Claude format parser
7. `ruv-swarm-daa` - DAA integration
8. `ruv-swarm-mcp` - MCP server
9. `swe-bench-adapter` - SWE-Bench integration
10. `ruv-swarm-wasm` - WASM bindings
11. `ruv-swarm-cli` - Command-line interface

### Phase 3: NPM Package Publishing

After all Rust crates are published and indexed:

```bash
# Build and publish NPM package
cd npm
./scripts/publish-npm.sh
```

### Phase 4: Post-Release

1. **Create Git Tag**:
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0: MCP Integration & 100% Test Coverage"
   git push origin v0.2.0
   ```

2. **Create GitHub Release**:
   - Go to https://github.com/ruvnet/ruv-FANN/releases/new
   - Select the `v0.2.0` tag
   - Title: "ruv-swarm v0.2.0: MCP Integration & Performance Improvements"
   - Copy highlights from CHANGELOG.md
   - Attach any relevant binaries

3. **Update Documentation**:
   - Update README.md installation instructions
   - Update docs.rs documentation
   - Update NPM package documentation

4. **Announcements**:
   - Post release notes
   - Update project website
   - Notify users of breaking changes

## Rollback Process

If issues are discovered after publishing:

### For Rust Crates:
- You cannot unpublish versions from crates.io
- Publish a patch version (0.2.1) with fixes
- Yank the broken version: `cargo yank --vers 0.2.0`

### For NPM Package:
- Deprecate the broken version: `npm deprecate ruv-swarm@0.2.0 "Critical bug, use 0.2.1"`
- Publish a patch version immediately

## Version Numbering

We follow Semantic Versioning (SemVer):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

Current version: 0.2.0
Next patch: 0.2.1
Next minor: 0.3.0
Next major: 1.0.0

## Emergency Contacts

- Crates.io issues: https://github.com/rust-lang/crates.io/issues
- NPM support: https://www.npmjs.com/support
- Project maintainer: @ruvnet

## Automated CI/CD

Future releases will use GitHub Actions:
- Automated testing on PR
- Automated publishing on tag push
- Automated documentation updates

See `.github/workflows/release.yml` (to be implemented)