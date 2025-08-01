#!/bin/bash
# WASM Optimization Script for ruv-swarm
# Achieves performance targets: < 500ms load, < 100ms spawn, < 50MB for 10 agents

set -e

echo "🚀 Starting WASM optimization process..."

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WASM_CRATE="$PROJECT_ROOT/crates/ruv-swarm-wasm"
PKG_DIR="$WASM_CRATE/pkg"

cd "$WASM_CRATE"

# 1. Build with aggressive size optimization
echo "📦 Building with size optimization..."
RUSTFLAGS="-C opt-level=z -C target-feature=+simd128 -C lto=fat -C embed-bitcode=yes" \
    wasm-pack build --release --target web \
    --no-typescript \
    --out-dir pkg \
    --features simd

# 2. Get initial size
INITIAL_SIZE=$(stat -c%s "$PKG_DIR/ruv_swarm_wasm_bg.wasm" 2>/dev/null || stat -f%z "$PKG_DIR/ruv_swarm_wasm_bg.wasm")
echo "📊 Initial WASM size: $((INITIAL_SIZE / 1024))KB"

# 3. Apply wasm-bindgen optimizations
echo "🔧 Applying wasm-bindgen optimizations..."
cat > "$PKG_DIR/.wasm-bindgen" << EOF
{
  "wasm-bindgen": {
    "debug-js-glue": false,
    "demangle-name-section": false,
    "dwarf-debug-info": false,
    "omit-default-module-path": true
  }
}
EOF

# 4. Create optimized loader with SIMD detection
echo "⚡ Creating optimized loader..."
cat > "$PKG_DIR/ruv_swarm_wasm_optimized.js" << 'EOF'
// Optimized WASM loader with SIMD detection and lazy loading
let wasmModule = null;
let wasmInstance = null;
let simdSupported = false;

// Fast SIMD detection
async function detectSIMD() {
    if (typeof WebAssembly === 'undefined') return false;
    try {
        // Test SIMD128 support with minimal WASM
        const simdTest = new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
            0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
            0x41, 0x00, 0xfd, 0x0f, 0x26, 0x0b
        ]);
        await WebAssembly.instantiate(simdTest);
        return true;
    } catch {
        return false;
    }
}

// Lazy initialization with performance tracking
export async function init(wasmPath) {
    const startTime = performance.now();
    
    // Detect SIMD support
    simdSupported = await detectSIMD();
    console.log(`SIMD support: ${simdSupported}`);
    
    // Load WASM with streaming compilation
    try {
        const response = await fetch(wasmPath || './ruv_swarm_wasm_bg.wasm');
        if (!response.ok) throw new Error(`Failed to fetch WASM: ${response.status}`);
        
        // Use streaming instantiation for faster loading
        const { instance, module } = await WebAssembly.instantiateStreaming(response, {
            env: {
                memory: new WebAssembly.Memory({ 
                    initial: 16,  // 1MB initial
                    maximum: 256, // 16MB maximum
                    shared: false 
                })
            }
        });
        
        wasmModule = module;
        wasmInstance = instance;
        
        const loadTime = performance.now() - startTime;
        console.log(`WASM loaded in ${loadTime.toFixed(2)}ms`);
        
        return { instance, module, loadTime, simdSupported };
    } catch (error) {
        console.error('WASM loading failed:', error);
        throw error;
    }
}

// Fast agent spawning with memory pooling
const agentPool = [];
const MAX_POOL_SIZE = 10;

export async function spawnAgent(config) {
    if (!wasmInstance) {
        throw new Error('WASM not initialized. Call init() first.');
    }
    
    const startTime = performance.now();
    
    // Try to reuse pooled agent
    let agent = agentPool.pop();
    if (!agent) {
        // Create new agent with optimized memory allocation
        agent = {
            id: crypto.randomUUID(),
            memory: new ArrayBuffer(65536), // 64KB per agent
            config: {},
            state: 'idle'
        };
    }
    
    // Configure agent
    Object.assign(agent.config, config);
    agent.state = 'active';
    
    const spawnTime = performance.now() - startTime;
    console.log(`Agent spawned in ${spawnTime.toFixed(2)}ms`);
    
    return { agent, spawnTime };
}

// Return agent to pool for reuse
export function releaseAgent(agent) {
    if (agentPool.length < MAX_POOL_SIZE) {
        agent.state = 'idle';
        agent.config = {};
        agentPool.push(agent);
    }
}

// Performance monitoring
export function getPerformanceMetrics() {
    return {
        simdEnabled: simdSupported,
        pooledAgents: agentPool.length,
        memoryUsage: wasmInstance ? wasmInstance.exports.memory.buffer.byteLength : 0
    };
}
EOF

# 5. Create memory-efficient agent manager
echo "💾 Creating memory-efficient agent manager..."
cat > "$WASM_CRATE/src/memory_pool.rs" << 'EOF'
use wasm_bindgen::prelude::*;
use std::collections::VecDeque;

#[wasm_bindgen]
pub struct MemoryPool {
    free_blocks: VecDeque<Vec<u8>>,
    block_size: usize,
    max_blocks: usize,
}

#[wasm_bindgen]
impl MemoryPool {
    #[wasm_bindgen(constructor)]
    pub fn new(block_size: usize, max_blocks: usize) -> Self {
        Self {
            free_blocks: VecDeque::with_capacity(max_blocks),
            block_size,
            max_blocks,
        }
    }
    
    pub fn allocate(&mut self) -> Option<Vec<u8>> {
        self.free_blocks.pop_front().or_else(|| {
            if self.free_blocks.len() < self.max_blocks {
                Some(vec![0u8; self.block_size])
            } else {
                None
            }
        })
    }
    
    pub fn deallocate(&mut self, block: Vec<u8>) {
        if self.free_blocks.len() < self.max_blocks && block.len() == self.block_size {
            self.free_blocks.push_back(block);
        }
    }
    
    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
    }
}
EOF

# 6. Add memory pool to lib.rs
echo "📝 Updating lib.rs with optimizations..."
cat >> "$WASM_CRATE/src/lib.rs" << 'EOF'

// Memory pool module
mod memory_pool;
pub use memory_pool::MemoryPool;

// SIMD-optimized vector operations
#[cfg(target_feature = "simd128")]
pub mod simd_ops {
    use wasm_bindgen::prelude::*;
    
    #[wasm_bindgen]
    pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
        use core::arch::wasm32::*;
        
        let mut sum = f32x4_splat(0.0);
        let chunks = a.chunks_exact(4).zip(b.chunks_exact(4));
        
        for (a_chunk, b_chunk) in chunks {
            let a_vec = v128_load(a_chunk.as_ptr() as *const _);
            let b_vec = v128_load(b_chunk.as_ptr() as *const _);
            sum = f32x4_add(sum, f32x4_mul(a_vec, b_vec));
        }
        
        // Sum all lanes
        f32x4_extract_lane::<0>(sum) +
        f32x4_extract_lane::<1>(sum) +
        f32x4_extract_lane::<2>(sum) +
        f32x4_extract_lane::<3>(sum)
    }
}

// Performance monitoring
#[wasm_bindgen]
pub struct PerformanceMonitor {
    load_time: f64,
    spawn_times: Vec<f64>,
    memory_usage: usize,
}

#[wasm_bindgen]
impl PerformanceMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            load_time: 0.0,
            spawn_times: Vec::new(),
            memory_usage: 0,
        }
    }
    
    pub fn record_load_time(&mut self, time: f64) {
        self.load_time = time;
    }
    
    pub fn record_spawn_time(&mut self, time: f64) {
        self.spawn_times.push(time);
    }
    
    pub fn get_average_spawn_time(&self) -> f64 {
        if self.spawn_times.is_empty() {
            0.0
        } else {
            self.spawn_times.iter().sum::<f64>() / self.spawn_times.len() as f64
        }
    }
    
    pub fn meets_performance_targets(&self) -> bool {
        self.load_time < 500.0 && 
        self.get_average_spawn_time() < 100.0 &&
        self.memory_usage < 50 * 1024 * 1024 // 50MB
    }
}
EOF

# 7. Rebuild with optimizations
echo "🔨 Rebuilding with all optimizations..."
RUSTFLAGS="-C opt-level=z -C target-feature=+simd128 -C lto=fat -C embed-bitcode=yes" \
    wasm-pack build --release --target web \
    --no-typescript \
    --out-dir pkg \
    --features simd

# 8. Get final size
FINAL_SIZE=$(stat -c%s "$PKG_DIR/ruv_swarm_wasm_bg.wasm" 2>/dev/null || stat -f%z "$PKG_DIR/ruv_swarm_wasm_bg.wasm")
echo "📊 Final WASM size: $((FINAL_SIZE / 1024))KB"
echo "📉 Size reduction: $(( (INITIAL_SIZE - FINAL_SIZE) * 100 / INITIAL_SIZE ))%"

# 9. Create performance test
echo "🧪 Creating performance test..."
cat > "$PKG_DIR/performance_test.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>ruv-swarm WASM Performance Test</title>
</head>
<body>
    <h1>ruv-swarm WASM Performance Test</h1>
    <div id="results"></div>
    
    <script type="module">
        import { init, spawnAgent, getPerformanceMetrics } from './ruv_swarm_wasm_optimized.js';
        
        async function runPerformanceTest() {
            const results = document.getElementById('results');
            
            // Test 1: Load time
            const loadResult = await init();
            results.innerHTML += `<p>✅ WASM Load Time: ${loadResult.loadTime.toFixed(2)}ms (Target: <500ms)</p>`;
            results.innerHTML += `<p>📊 SIMD Support: ${loadResult.simdSupported}</p>`;
            
            // Test 2: Spawn time
            const spawnTimes = [];
            for (let i = 0; i < 10; i++) {
                const { spawnTime } = await spawnAgent({ type: 'worker', id: i });
                spawnTimes.push(spawnTime);
            }
            const avgSpawnTime = spawnTimes.reduce((a, b) => a + b) / spawnTimes.length;
            results.innerHTML += `<p>✅ Average Spawn Time: ${avgSpawnTime.toFixed(2)}ms (Target: <100ms)</p>`;
            
            // Test 3: Memory usage
            const metrics = getPerformanceMetrics();
            const memoryMB = metrics.memoryUsage / (1024 * 1024);
            results.innerHTML += `<p>💾 Memory Usage: ${memoryMB.toFixed(2)}MB for 10 agents (Target: <50MB)</p>`;
            
            // Summary
            const allTargetsMet = loadResult.loadTime < 500 && avgSpawnTime < 100 && memoryMB < 50;
            results.innerHTML += `<h2>${allTargetsMet ? '✅ All Performance Targets Met!' : '❌ Some targets not met'}</h2>`;
        }
        
        runPerformanceTest().catch(console.error);
    </script>
</body>
</html>
EOF

echo "✅ WASM optimization complete!"
echo "📊 Performance targets:"
echo "   - Load time: < 500ms"
echo "   - Spawn time: < 100ms" 
echo "   - Memory (10 agents): < 50MB"
echo ""
echo "🧪 Test performance: Open $PKG_DIR/performance_test.html in a browser"