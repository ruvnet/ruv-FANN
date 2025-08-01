# WASM SIMD Build Guide

## 🚀 Overview

The `build-wasm-simd.sh` script creates production-ready WebAssembly modules with SIMD128 optimization for maximum performance in neural network operations and swarm coordination.

## 📋 Prerequisites

### Required:
- **Rust** (1.70+) with `wasm32-unknown-unknown` target
- **wasm-pack** (latest version recommended)
- **Node.js** (16+) for testing and validation

### Optional (for optimization):
- **binaryen** (`wasm-opt` tool) for size optimization
- **Modern browser** with SIMD128 support for testing

### Installation:
```bash
# Install Rust target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Install binaryen (optional)
# Ubuntu/Debian:
sudo apt install binaryen
# macOS:
brew install binaryen
# Windows:
# Download from https://github.com/WebAssembly/binaryen/releases
```

## 💻 Usage

### Basic Build:
```bash
cd ruv-swarm/scripts
./build-wasm-simd.sh
```

### Build Process:
1. **Prerequisite Check** - Validates required tools
2. **Clean Previous Builds** - Removes old artifacts
3. **SIMD Optimization** - Builds with SIMD128 flags
4. **Size Optimization** - Applies wasm-opt if available
5. **Advanced Loader** - Creates SIMD-aware JavaScript loader
6. **Performance Testing** - Generates test suite
7. **NPM Integration** - Copies files to npm package
8. **Validation** - Runs basic functionality tests

## 📊 Performance Targets

| Metric | Target | Typical Result |
|--------|--------|----------------|
| **Load Time** | <500ms | ~200ms |
| **Agent Spawn** | <100ms | ~35ms |
| **Memory (10 agents)** | <50MB | ~28MB |
| **SIMD Operations** | 4x baseline | 4-6x faster |

## 🔧 Advanced Features

### SIMD Detection:
The loader automatically detects SIMD128 support:
```javascript
import { initSIMD, getPerformanceMetrics } from './ruv_swarm_simd_loader.js';

await initSIMD();
const metrics = getPerformanceMetrics();
console.log('SIMD supported:', metrics.simdSupported);
```

### Memory Pooling:
Automatic agent pooling for performance:
```javascript
import { spawnAgentSIMD, releaseAgentSIMD } from './ruv_swarm_simd_loader.js';

// Spawn agent (reuses pooled instances)
const { agent } = await spawnAgentSIMD({ type: 'neural' });

// Release back to pool when done
releaseAgentSIMD(agent);
```

### Performance Monitoring:
Built-in metrics collection:
```javascript
const metrics = getPerformanceMetrics();
console.log({
    loadTime: metrics.loadTime,
    averageSpawnTime: metrics.averageSpawnTime,
    memoryUsage: metrics.memoryUsageMB,
    performanceGrade: metrics.performanceGrade()
});
```

## 🧪 Testing

### Browser Testing:
1. Run the build script
2. Open `pkg/simd_performance_test.html` in a modern browser
3. View real-time performance metrics and validation

### Node.js Testing:
```bash
cd ruv-swarm/npm
npm test
```

### Manual Validation:
```bash
# Check files were created
ls -la scripts/build-wasm-simd.sh
ls -la crates/ruv-swarm-wasm/pkg/

# Verify SIMD features
grep -r "simd" crates/ruv-swarm-wasm/pkg/
```

## 📁 Output Files

### Generated Files:
- `pkg/ruv_swarm_wasm.js` - Main WASM bindings
- `pkg/ruv_swarm_wasm_bg.wasm` - Optimized WASM module
- `pkg/ruv_swarm_simd_loader.js` - Advanced SIMD loader
- `pkg/simd_performance_test.html` - Performance test suite
- `pkg/package.json` - NPM package metadata

### NPM Integration:
Files are automatically copied to `npm/wasm/` for package distribution.

## 🔍 Troubleshooting

### Common Issues:

**1. "wasm-pack not found"**
```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
source ~/.cargo/env
```

**2. "wasm32-unknown-unknown target missing"**
```bash
rustup target add wasm32-unknown-unknown
```

**3. "Build failed"**
- Check Rust version: `rustc --version` (needs 1.70+)
- Verify in correct directory: should be in `crates/ruv-swarm-wasm`
- Check Cargo.toml has SIMD features

**4. "SIMD not detected in browser"**
- Use a modern browser (Chrome 91+, Firefox 89+, Safari 16.4+)
- Check browser compatibility at https://caniuse.com/wasm-simd

**5. "Performance targets not met"**
- Ensure `wasm-opt` is installed for size optimization
- Check browser supports SIMD128
- Verify hardware has sufficient performance

### Debug Mode:
```bash
# Enable verbose output
RUST_LOG=debug ./build-wasm-simd.sh

# Check WASM module info
wasm-objdump -h pkg/ruv_swarm_wasm_bg.wasm
```

## 🔧 Customization

### Build Configuration:
Edit `crates/ruv-swarm-wasm/Cargo.toml`:
```toml
[features]
default = ["simd", "simd128"]
performance = ["simd", "simd128", "full"]
minimal = []
```

### Performance Tuning:
Modify RUSTFLAGS in the script:
```bash
export RUSTFLAGS="-C target-feature=+simd128,+sign-ext,+mutable-globals -C opt-level=3"
```

### Memory Configuration:
Adjust loader settings:
```javascript
const memory = new WebAssembly.Memory({
    initial: 32,    // Increase for more agents
    maximum: 512,   // Set memory limit
    shared: false   // Enable for threading
});
```

## 📈 Integration Examples

### React Application:
```javascript
import { initSIMD, spawnAgentSIMD } from 'ruv-swarm/wasm/ruv_swarm_simd_loader.js';

function SwarmApp() {
    useEffect(() => {
        async function initSwarm() {
            await initSIMD();
            const agent = await spawnAgentSIMD({ type: 'researcher' });
            // Use agent...
        }
        initSwarm();
    }, []);
}
```

### Node.js Backend:
```javascript
// Use regular build for Node.js
const { RuvSwarm } = require('ruv-swarm');
const swarm = new RuvSwarm({ features: ['neural'] });
```

### Web Worker:
```javascript
// In worker thread
importScripts('./ruv_swarm_simd_loader.js');

self.onmessage = async function(e) {
    await initSIMD();
    const agent = await spawnAgentSIMD(e.data.config);
    // Process with agent...
};
```

## 🚀 Production Deployment

### CDN Usage:
```html
<script type="module">
    import { initSIMD } from 'https://cdn.jsdelivr.net/npm/ruv-swarm@latest/wasm/ruv_swarm_simd_loader.js';
    await initSIMD();
</script>
```

### Webpack Configuration:
```javascript
module.exports = {
    experiments: {
        asyncWebAssembly: true,
    },
    resolve: {
        fallback: {
            "crypto": require.resolve("crypto-browserify")
        }
    }
};
```

### Performance Monitoring:
```javascript
// Production monitoring
const metrics = getPerformanceMetrics();
if (metrics.performanceGrade() !== 'A') {
    console.warn('Performance below optimal:', metrics);
    // Send to monitoring service
}
```

## 📚 Related Documentation

- [WebAssembly SIMD](https://github.com/WebAssembly/simd)
- [wasm-pack Guide](https://rustwasm.github.io/wasm-pack/)
- [Rust and WebAssembly Book](https://rustwasm.github.io/docs/book/)
- [ruv-swarm API Documentation](../README.md)

---

**For support and issues**: https://github.com/ruvnet/ruv-FANN/issues