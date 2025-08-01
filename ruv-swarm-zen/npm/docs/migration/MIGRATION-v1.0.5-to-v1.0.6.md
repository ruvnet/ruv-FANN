# Migration Guide: v1.0.5 to v1.0.6

## Overview

Version 1.0.6 is a **patch release** that fixes critical NPX/CLI execution issues without any breaking API changes. This guide helps you upgrade smoothly.

## 🚀 Quick Migration

```bash
# Update to v1.0.6
npm update ruv-swarm@latest

# Or for global installations
npm install -g ruv-swarm@latest
```

## 🔄 What Changed

### Fixed Issues

1. **NPX/CLI Execution** - The critical syntax error preventing CLI usage has been fixed
2. **WASM Deprecation Warnings** - Properly passing binary data to avoid console warnings
3. **Node.js v22 Compatibility** - Fixed private method syntax issues
4. **Build Process** - Resolved character encoding and escaping issues

### No Breaking Changes

- ✅ All APIs remain unchanged
- ✅ No configuration changes required
- ✅ Backward compatible with v1.0.5 code

## 📋 Migration Steps

### For NPX Users

If you were experiencing the "Invalid or unexpected token" error:

```bash
# Before (v1.0.5 - broken)
$ npx ruv-swarm --version
# SyntaxError: Invalid or unexpected token

# After (v1.0.6 - fixed)
$ npx ruv-swarm --version
# ruv-swarm v1.0.6
```

### For Library Users

No changes needed! Your existing code continues to work:

```javascript
// This code works in both v1.0.5 and v1.0.6
import { RuvSwarm } from 'ruv-swarm';

const swarm = await RuvSwarm.initialize({
  topology: 'mesh',
  maxAgents: 10
});
```

### For Global CLI Users

Update your global installation:

```bash
# Uninstall old version
npm uninstall -g ruv-swarm

# Install new version
npm install -g ruv-swarm@latest

# Verify it works
ruv-swarm --version
```

### For Docker Users

Update your Dockerfile:

```dockerfile
# Before
FROM node:18-alpine
RUN npm install -g ruv-swarm@1.0.5

# After
FROM node:18-alpine
RUN npm install -g ruv-swarm@1.0.6
```

## 🧪 Testing Your Migration

Run these commands to verify the upgrade:

```bash
# 1. Check version
npx ruv-swarm --version
# Should output: ruv-swarm v1.0.6

# 2. Test CLI functionality
npx ruv-swarm --help
# Should display help without errors

# 3. Test MCP server
npx ruv-swarm mcp start --port 3000
# Should start without syntax errors

# 4. Run a simple swarm
npx ruv-swarm init mesh 5
# Should initialize successfully
```

## ⚠️ Known Issues Fixed

### Issue #41: NPX Integration Failure

**Symptom**: 
```
SyntaxError: Invalid or unexpected token
    at file:///path/to/ruv-swarm/src/wasm-loader.js:255
```

**Status**: ✅ FIXED in v1.0.6

**Solution**: Update to v1.0.6

### WASM Deprecation Warning

**Symptom**:
```
Passing the module or path to 'init' is deprecated and will be removed...
```

**Status**: ✅ FIXED in v1.0.6

**Solution**: The warning no longer appears in v1.0.6

## 🔍 Verification Script

Create a `verify-upgrade.js` file:

```javascript
import { RuvSwarm } from 'ruv-swarm';
import { execSync } from 'child_process';

console.log('Verifying ruv-swarm v1.0.6 upgrade...\n');

// Test 1: Library import
try {
  const version = RuvSwarm.getVersion();
  console.log('✅ Library import working:', version);
} catch (error) {
  console.error('❌ Library import failed:', error.message);
}

// Test 2: CLI execution
try {
  const cliVersion = execSync('npx ruv-swarm --version', { encoding: 'utf8' });
  console.log('✅ CLI execution working:', cliVersion.trim());
} catch (error) {
  console.error('❌ CLI execution failed:', error.message);
}

// Test 3: WASM functionality
try {
  const swarm = await RuvSwarm.initialize({ topology: 'mesh' });
  console.log('✅ WASM initialization working');
  await swarm.destroy();
} catch (error) {
  console.error('❌ WASM initialization failed:', error.message);
}

console.log('\nVerification complete!');
```

Run with:
```bash
node verify-upgrade.js
```

## 📞 Support

If you encounter any issues during migration:

1. Check the [Troubleshooting Guide](README.md#-troubleshooting)
2. Review the [CHANGELOG](CHANGELOG.md)
3. Open an issue: https://github.com/ruvnet/ruv-FANN/issues

## 🎉 What's Next

With v1.0.6, you can now:

- ✅ Use `npx ruv-swarm` commands without errors
- ✅ Deploy to remote servers with confidence
- ✅ Integrate with Claude Code via MCP
- ✅ Enjoy improved error messages and debugging

Continue using ruv-swarm as before, but now with reliable CLI functionality!

---

*Migration guide for ruv-swarm v1.0.6 - Generated on 2025-07-03*