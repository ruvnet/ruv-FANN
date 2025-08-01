# WASM Compilation Strategy for ruv-FANN/neuro-divergent

## Overview

This document outlines the comprehensive strategy for compiling ruv-FANN and neuro-divergent libraries to WebAssembly (WASM) and distributing them via npm/npx. The strategy focuses on performance, size optimization, and ease of use while maintaining compatibility with both browser and Node.js environments.

## 1. WASM Build Pipeline Configuration

### 1.1 Core Tools and Dependencies

#### wasm-pack Configuration
```toml
# Root Cargo.toml additions for WASM support
[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = "0.3"

# For size optimization (avoid wee_alloc due to memory issues)
console_error_panic_hook = { version = "0.1", optional = true }

[lib]
crate-type = ["cdylib", "rlib"]

[profile.release]
# Enable link time optimizations
lto = true
# Optimize for size
opt-level = "z"
# Strip debug symbols
strip = true
# Enable single codegen unit for better optimization
codegen-units = 1
```

#### Build Commands
```bash
# Standard build
wasm-pack build --target web --out-dir pkg

# With SIMD optimization
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir pkg-simd

# Node.js target
wasm-pack build --target nodejs --out-dir pkg-node
```

### 1.2 SIMD Support Strategy

#### Feature Detection and Fallback
```rust
// src/wasm/feature_detection.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct RuntimeFeatures {
    simd_available: bool,
}

#[wasm_bindgen]
impl RuntimeFeatures {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let simd_available = detect_simd_support();
        RuntimeFeatures { simd_available }
    }
    
    #[wasm_bindgen(getter)]
    pub fn simd_available(&self) -> bool {
        self.simd_available
    }
}

#[cfg(target_arch = "wasm32")]
fn detect_simd_support() -> bool {
    // Runtime SIMD detection for WebAssembly
    #[cfg(target_feature = "simd128")]
    return true;
    
    #[cfg(not(target_feature = "simd128"))]
    return false;
}
```

### 1.3 Size Optimization Techniques

1. **Dead Code Elimination**
   - Use `wee_alloc` alternatives (standard allocator recommended)
   - Enable LTO (Link Time Optimization)
   - Use `wasm-opt` for post-processing (15-20% size reduction)

2. **Conditional Compilation**
   ```rust
   #[cfg(target_arch = "wasm32")]
   mod wasm_optimized;
   
   #[cfg(not(target_arch = "wasm32"))]
   mod native;
   ```

3. **Minimal Dependencies**
   - Remove heavy dependencies for WASM builds
   - Use `no_std` where possible for core algorithms
   - Implement custom lightweight alternatives

## 2. NPX Package Structure and Distribution

### 2.1 Package Directory Structure
```
ruv-swarm/
├── Cargo.toml (workspace configuration)
├── crates/
│   ├── ruv-swarm-core/           # Shared logic (no_std compatible)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── activation.rs     # Core activation functions
│   │       ├── network.rs        # Basic network structures
│   │       └── math.rs           # SIMD-optimized math operations
│   ├── ruv-swarm-wasm/          # WASM-specific bindings
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── bindings.rs      # wasm-bindgen exports
│   │       ├── memory.rs        # WASM memory management
│   │       └── interop.rs       # JS interoperability
│   └── ruv-swarm-native/        # Native CLI bindings
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
├── npm/
│   ├── package.json
│   ├── README.md
│   ├── bin/
│   │   └── ruv-swarm.js        # npx entry point
│   ├── lib/
│   │   ├── index.js            # Main library exports
│   │   ├── loader.js           # WASM loading logic
│   │   └── features.js         # Feature detection
│   └── wasm/
│       ├── ruv_swarm_bg.wasm   # Standard WASM binary
│       ├── ruv_swarm_bg.js     # Generated bindings
│       └── ruv_swarm_simd.wasm # SIMD-optimized binary
└── examples/
    ├── browser/
    └── node/
```

### 2.2 package.json Configuration
```json
{
  "name": "@ruv/swarm",
  "version": "0.1.0",
  "description": "High-performance neural network swarm orchestration in WebAssembly",
  "main": "lib/index.js",
  "module": "lib/index.js",
  "types": "lib/index.d.ts",
  "bin": {
    "ruv-swarm": "./bin/ruv-swarm.js"
  },
  "files": [
    "bin/",
    "lib/",
    "wasm/",
    "README.md"
  ],
  "scripts": {
    "test": "node test/test.js",
    "build": "node scripts/build.js"
  },
  "engines": {
    "node": ">=14.0.0"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/ruvnet/ruv-FANN.git"
  },
  "keywords": [
    "neural-network",
    "wasm",
    "webassembly",
    "machine-learning",
    "swarm",
    "ai"
  ],
  "author": "rUv Contributors",
  "license": "MIT OR Apache-2.0",
  "dependencies": {},
  "devDependencies": {
    "wasm-pack": "^0.12.0"
  }
}
```

### 2.3 NPX Entry Point Implementation
```javascript
#!/usr/bin/env node
// npm/bin/ruv-swarm.js

const { RuvSwarm } = require('../lib');
const path = require('path');
const fs = require('fs');

async function main() {
    const args = process.argv.slice(2);
    
    // Initialize WASM module
    const swarm = await RuvSwarm.initialize({
        wasmPath: path.join(__dirname, '..', 'wasm'),
        useSIMD: RuvSwarm.detectSIMDSupport()
    });
    
    // Parse commands
    const command = args[0] || 'help';
    
    switch (command) {
        case 'train':
            await handleTrain(swarm, args.slice(1));
            break;
        case 'predict':
            await handlePredict(swarm, args.slice(1));
            break;
        case 'benchmark':
            await handleBenchmark(swarm);
            break;
        case 'help':
        default:
            showHelp();
    }
}

function showHelp() {
    console.log(`
ruv-swarm - High-performance neural network orchestration

Usage: npx @ruv/swarm <command> [options]

Commands:
  train <data-file>     Train a neural network
  predict <model-file>  Make predictions
  benchmark            Run performance benchmarks
  help                 Show this help message

Options:
  --model <type>       Model type (mlp, lstm, transformer)
  --epochs <n>         Number of training epochs
  --batch-size <n>     Batch size for training
  --simd              Force SIMD optimization (if available)
`);
}

main().catch(console.error);
```

### 2.4 WASM Loading Strategy
```javascript
// npm/lib/loader.js

class WASMLoader {
    constructor(options = {}) {
        this.useSIMD = options.useSIMD && this.detectSIMDSupport();
        this.wasmPath = options.wasmPath || './wasm';
        this.cache = new Map();
    }
    
    detectSIMDSupport() {
        try {
            // WebAssembly SIMD feature detection
            if (typeof WebAssembly !== 'undefined' && WebAssembly.validate) {
                // Test SIMD instruction
                const simdTest = new Uint8Array([
                    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
                    0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
                    0x41, 0x00, 0xfd, 0x0f, 0x26, 0x0b
                ]);
                return WebAssembly.validate(simdTest);
            }
        } catch (e) {
            console.warn('SIMD detection failed:', e);
        }
        return false;
    }
    
    async loadModule() {
        const moduleFile = this.useSIMD ? 'ruv_swarm_simd.wasm' : 'ruv_swarm_bg.wasm';
        const wasmPath = `${this.wasmPath}/${moduleFile}`;
        
        if (this.cache.has(wasmPath)) {
            return this.cache.get(wasmPath);
        }
        
        let wasmModule;
        
        if (typeof window !== 'undefined') {
            // Browser environment
            const response = await fetch(wasmPath);
            const buffer = await response.arrayBuffer();
            wasmModule = await WebAssembly.instantiate(buffer);
        } else {
            // Node.js environment
            const fs = require('fs').promises;
            const path = require('path');
            const wasmBuffer = await fs.readFile(path.resolve(wasmPath));
            wasmModule = await WebAssembly.instantiate(wasmBuffer);
        }
        
        this.cache.set(wasmPath, wasmModule);
        return wasmModule;
    }
}

module.exports = { WASMLoader };
```

## 3. JavaScript/TypeScript API Design

### 3.1 Core API Interface
```typescript
// npm/lib/index.d.ts

export interface NetworkConfig {
    layers: number[];
    activationFunction?: ActivationFunction;
    learningRate?: number;
    momentum?: number;
}

export interface TrainingOptions {
    epochs: number;
    batchSize?: number;
    validationSplit?: number;
    earlyStop?: boolean;
    callbacks?: TrainingCallback[];
}

export interface PredictionResult {
    outputs: Float32Array;
    confidence?: number;
    timeTaken: number;
}

export class RuvSwarm {
    static initialize(options?: InitOptions): Promise<RuvSwarm>;
    static detectSIMDSupport(): boolean;
    
    createNetwork(config: NetworkConfig): Network;
    train(network: Network, data: TrainingData, options: TrainingOptions): Promise<TrainingResult>;
    predict(network: Network, inputs: Float32Array): PredictionResult;
    benchmark(): BenchmarkResults;
}

export class Network {
    readonly id: string;
    readonly config: NetworkConfig;
    
    save(): Uint8Array;
    load(data: Uint8Array): void;
    getWeights(): Float32Array;
    setWeights(weights: Float32Array): void;
}
```

### 3.2 High-Level Wrapper API
```javascript
// npm/lib/index.js

const { WASMLoader } = require('./loader');
const { RuvSwarmWASM } = require('../wasm/ruv_swarm_bg.js');

class RuvSwarm {
    constructor(wasmInstance) {
        this.wasm = wasmInstance;
        this.networks = new Map();
    }
    
    static async initialize(options = {}) {
        const loader = new WASMLoader(options);
        const wasmModule = await loader.loadModule();
        const wasmInstance = new RuvSwarmWASM(wasmModule);
        return new RuvSwarm(wasmInstance);
    }
    
    createNetwork(config) {
        const networkId = this.wasm.create_network(
            config.layers,
            config.activationFunction || 'sigmoid',
            config.learningRate || 0.7,
            config.momentum || 0.1
        );
        
        const network = new Network(networkId, config, this.wasm);
        this.networks.set(networkId, network);
        return network;
    }
    
    async train(network, data, options) {
        const startTime = performance.now();
        
        // Convert JS data to WASM format
        const wasmData = this.wasm.create_training_data(
            data.inputs,
            data.outputs
        );
        
        // Perform training
        const result = await this.wasm.train_network(
            network.id,
            wasmData,
            options.epochs,
            options.batchSize || 32
        );
        
        const endTime = performance.now();
        
        return {
            finalError: result.error,
            epochs: result.epochs,
            timeTaken: endTime - startTime
        };
    }
    
    predict(network, inputs) {
        const startTime = performance.now();
        const outputs = this.wasm.predict(network.id, inputs);
        const endTime = performance.now();
        
        return {
            outputs,
            timeTaken: endTime - startTime
        };
    }
}

module.exports = { RuvSwarm };
```

## 4. Integration with ruv-FANN Components

### 4.1 Suitable Components for WASM

#### Core Components (High Priority)
1. **Activation Functions** - Pure mathematical operations, ideal for WASM
2. **Network Forward Pass** - Computational intensive, benefits from SIMD
3. **Basic Training Algorithms** - Backpropagation, gradient descent
4. **Layer Operations** - Matrix multiplications, perfect for optimization

#### Data Processing (Medium Priority)
1. **Preprocessing Functions** - Normalization, scaling
2. **Validation Utilities** - Data validation and sanitization
3. **Feature Engineering** - Basic transformations

#### Components to Exclude
1. **File I/O Operations** - Handle in JavaScript layer
2. **Complex Serialization** - Use JavaScript for flexibility
3. **Plotting/Visualization** - Better suited for JS libraries
4. **Async/Parallel Operations** - Use Web Workers instead

### 4.2 Neural Network Operations in WASM

#### Optimized Matrix Operations
```rust
// crates/ruv-swarm-core/src/math.rs

use core::arch::wasm32::*;

#[cfg(target_feature = "simd128")]
pub fn matrix_multiply_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut sum = f32x4_splat(0.0);
                let mut l = 0;
                
                // SIMD loop
                while l + 4 <= k {
                    let a_vec = v128_load(&a[i * k + l] as *const f32 as *const v128);
                    let b_vec = v128_load(&b[l * n + j] as *const f32 as *const v128);
                    sum = f32x4_add(sum, f32x4_mul(a_vec, b_vec));
                    l += 4;
                }
                
                // Horizontal sum
                let sum_scalar = f32x4_extract_lane::<0>(sum) +
                                f32x4_extract_lane::<1>(sum) +
                                f32x4_extract_lane::<2>(sum) +
                                f32x4_extract_lane::<3>(sum);
                
                // Handle remaining elements
                while l < k {
                    sum_scalar += a[i * k + l] * b[l * n + j];
                    l += 1;
                }
                
                c[i * n + j] = sum_scalar;
            }
        }
    }
}

#[cfg(not(target_feature = "simd128"))]
pub fn matrix_multiply_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
```

## 5. Performance Benchmarks and Limitations

### 5.1 Expected Performance Characteristics

#### WASM vs Native Performance
- **Computation-heavy operations**: 80-95% of native performance
- **SIMD-optimized operations**: 85-98% of native performance
- **Memory-intensive operations**: 60-80% of native performance
- **Small frequent calls**: 40-60% of native performance (due to JS-WASM boundary)

#### Size Considerations
- Base WASM module: ~100-200KB (after optimization)
- With neural network models: ~300-500KB
- SIMD variant: +10-15% size increase
- Gzipped size: 40-60% of original

### 5.2 Limitations and Workarounds

1. **Memory Limitations**
   - WASM initial memory: 16MB default
   - Maximum memory: 4GB (32-bit addressing)
   - Solution: Implement streaming/chunking for large datasets

2. **Thread Limitations**
   - No direct thread spawning in WASM
   - Solution: Use Web Workers for parallelism

3. **File System Access**
   - No direct file system access in browser
   - Solution: Use IndexedDB or File API

4. **Debugging Challenges**
   - Limited debugging tools for WASM
   - Solution: Comprehensive logging and error reporting

## 6. Example Usage Patterns

### 6.1 Browser Usage
```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import { RuvSwarm } from 'https://unpkg.com/@ruv/swarm/lib/index.js';
        
        async function runExample() {
            // Initialize WASM module
            const swarm = await RuvSwarm.initialize({
                useSIMD: true
            });
            
            // Create network
            const network = swarm.createNetwork({
                layers: [2, 3, 1],
                activationFunction: 'sigmoid',
                learningRate: 0.7
            });
            
            // Training data (XOR problem)
            const trainingData = {
                inputs: new Float32Array([
                    0, 0,
                    0, 1,
                    1, 0,
                    1, 1
                ]),
                outputs: new Float32Array([0, 1, 1, 0])
            };
            
            // Train network
            const result = await swarm.train(network, trainingData, {
                epochs: 1000,
                batchSize: 4
            });
            
            console.log('Training completed:', result);
            
            // Make predictions
            const prediction = swarm.predict(network, new Float32Array([0, 1]));
            console.log('Prediction:', prediction);
        }
        
        runExample();
    </script>
</head>
<body>
    <h1>RuvSwarm WASM Example</h1>
    <div id="results"></div>
</body>
</html>
```

### 6.2 Node.js Usage
```javascript
// example.js
const { RuvSwarm } = require('@ruv/swarm');

async function main() {
    // Initialize with auto-detection
    const swarm = await RuvSwarm.initialize();
    
    // Create a more complex network
    const network = swarm.createNetwork({
        layers: [10, 20, 15, 5],
        activationFunction: 'relu',
        learningRate: 0.01
    });
    
    // Load training data
    const fs = require('fs');
    const csvData = fs.readFileSync('data.csv', 'utf8');
    const { inputs, outputs } = parseCSV(csvData);
    
    // Train with callbacks
    const result = await swarm.train(network, { inputs, outputs }, {
        epochs: 100,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: [
            {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss=${logs.loss.toFixed(4)}`);
                }
            }
        ]
    });
    
    // Save model
    const modelData = network.save();
    fs.writeFileSync('model.bin', modelData);
    
    // Run benchmarks
    const benchmarks = swarm.benchmark();
    console.log('Performance benchmarks:', benchmarks);
}

main().catch(console.error);
```

### 6.3 NPX One-Liner Examples
```bash
# Quick training
npx @ruv/swarm train data.csv --model mlp --epochs 100

# Prediction
npx @ruv/swarm predict model.bin --input "[1.0, 2.0, 3.0]"

# Benchmark
npx @ruv/swarm benchmark --simd

# Interactive mode
npx @ruv/swarm interactive
```

## 7. Development Workflow

### 7.1 Build Process
```bash
# 1. Build Rust WASM modules
cd crates/ruv-swarm-wasm
wasm-pack build --target web --out-dir ../../npm/wasm

# 2. Build SIMD variant
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir ../../npm/wasm-simd

# 3. Copy and rename SIMD files
cp ../../npm/wasm-simd/ruv_swarm_bg.wasm ../../npm/wasm/ruv_swarm_simd.wasm

# 4. Run optimization
wasm-opt -Oz ../../npm/wasm/*.wasm -o ../../npm/wasm/

# 5. Generate TypeScript definitions
cd ../../npm
npx tsc --declaration --emitDeclarationOnly

# 6. Run tests
npm test

# 7. Publish to npm
npm publish
```

### 7.2 Testing Strategy
1. **Unit Tests**: Test WASM functions in isolation
2. **Integration Tests**: Test JS-WASM interaction
3. **Performance Tests**: Compare with native implementation
4. **Compatibility Tests**: Test across browsers and Node versions

## 8. Future Enhancements

1. **WebGPU Support**: Add GPU acceleration when available
2. **Streaming API**: Support for large datasets
3. **Model Zoo**: Pre-trained models in WASM format
4. **Progressive Enhancement**: Automatic fallback to asm.js
5. **Worker Pool**: Automatic parallelization using Web Workers
6. **Model Quantization**: 8-bit and 16-bit model support

## Conclusion

This WASM compilation strategy provides a comprehensive approach to bringing ruv-FANN's neural network capabilities to the web. By leveraging modern WASM features like SIMD, implementing smart loading strategies, and providing intuitive JavaScript APIs, we can deliver near-native performance in both browser and Node.js environments while maintaining ease of use through npm/npx distribution.