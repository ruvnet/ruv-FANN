# Release Checklist v0.2.0

## 🔍 Pre-Release Validation

### Rust Crates
- [ ] Run `cargo test --all --release` - All tests pass
- [ ] Run `cargo clippy --all -- -D warnings` - No warnings
- [ ] Run `cargo fmt --all -- --check` - Code is formatted
- [ ] Run `cargo audit` - No security vulnerabilities
- [ ] Run `cargo build --all --release` - Builds successfully
- [ ] Verify all crates are at version 0.2.0

### NPM Package
- [ ] Run `npm run test:comprehensive` - All tests pass
- [ ] Run `npm run lint:check` - No linting errors
- [ ] Run `npm audit` - No vulnerabilities
- [ ] Run `npm run build:all` - WASM builds successfully
- [ ] Verify package.json version is 0.2.0
- [ ] Verify publishConfig is set correctly

### Documentation
- [ ] CHANGELOG.md is up-to-date
- [ ] README.md installation instructions are correct
- [ ] All examples in examples/ directory work
- [ ] API documentation is complete

## 📦 Publishing Steps

### 1. Final Build Verification
```bash
# From workspace root
cd /workspaces/ruv-FANN/ruv-swarm

# Clean and rebuild everything
cargo clean
cargo build --all --release
cd npm && npm run build:all
```

### 2. Rust Crates Publishing
```bash
# Login to crates.io
cargo login

# Run the publishing script
./scripts/publish-rust.sh
```

### 3. Wait for Crates.io Indexing
- [ ] Wait ~5-10 minutes for all crates to be indexed
- [ ] Verify at https://crates.io/search?q=ruv-swarm

### 4. NPM Package Publishing
```bash
# Login to npm
npm login

# Run the publishing script
cd npm
../scripts/publish-npm.sh
```

### 5. Git Tag and Release
```bash
# Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0: MCP Integration & 100% Test Coverage"
git push origin v0.2.0
```

### 6. GitHub Release
- [ ] Go to https://github.com/ruvnet/ruv-FANN/releases/new
- [ ] Select tag: v0.2.0
- [ ] Release title: "ruv-swarm v0.2.0: MCP Integration & Performance Improvements"
- [ ] Copy release notes from CHANGELOG.md
- [ ] Publish release

## 🚀 Post-Release Verification

### Crates.io
- [ ] Verify all crates are live: https://crates.io/users/ruvnet
- [ ] Test installation: `cargo install ruv-swarm-cli`
- [ ] Verify CLI works: `ruv-swarm --version`

### NPM
- [ ] Verify package is live: https://www.npmjs.com/package/ruv-swarm
- [ ] Test installation: `npm install ruv-swarm`
- [ ] Test CLI: `npx ruv-swarm --version`

### Integration Testing
- [ ] Create a new project and test Rust integration
- [ ] Create a new Node.js project and test NPM integration
- [ ] Test MCP server: `npx ruv-swarm mcp start`

## 📊 Success Metrics

- All 11 Rust crates published successfully
- NPM package published with correct version
- No failed installations reported
- Documentation accessible and accurate
- Examples run without errors

## 🔄 Rollback Plan

If critical issues are found:

1. **Rust**: Yank affected versions
   ```bash
   cargo yank --vers 0.2.0 --crate ruv-swarm-core
   ```

2. **NPM**: Deprecate version
   ```bash
   npm deprecate ruv-swarm@0.2.0 "Critical issue, please use 0.2.1"
   ```

3. Fix issues and release 0.2.1 immediately

## 📝 Notes

- Current workspace: `/workspaces/ruv-FANN/ruv-swarm`
- All crates use workspace versioning
- NPM package includes WASM builds
- This is the first public release with MCP support

---

Release Prepared By: Release Preparation Agent
Date: 2025-07-02
Target Version: 0.2.0