#!/bin/bash

# Build script for ruv-swarm WASM with neural network integration
# Agent 2: Neural Network Specialist

set -e

echo "🧠 Building ruv-swarm WASM with neural network integration..."

# Change to the WASM crate directory
cd "$(dirname "$0")/../crates/ruv-swarm-wasm"

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Please install it:"
    echo "   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg/

# Build with optimizations for neural networks
echo "🔨 Building WASM module..."
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build \
    --target web \
    --out-dir pkg \
    --release \
    -- \
    --features "neural"

# Post-process the generated files
echo "📦 Post-processing WASM files..."

# Create a wrapper that exports all neural functionality
cat > pkg/neural-wrapper.js << 'EOF'
// Neural network wrapper for easy JavaScript usage
import init, * as wasm from './ruv_swarm_wasm.js';

let initialized = false;
let wasmModule = null;

export async function initializeNeural() {
    if (!initialized) {
        await init();
        wasmModule = wasm;
        initialized = true;
    }
    return wasmModule;
}

// High-level API exports
export { 
    WasmNeuralNetwork,
    WasmTrainer,
    AgentNeuralNetworkManager,
    ActivationFunctionManager,
    WasmCascadeTrainer
} from './ruv_swarm_wasm.js';

export const NeuralNetwork = {
    create: async (config) => {
        await initializeNeural();
        return new wasm.WasmNeuralNetwork(config);
    },
    
    createTrainer: async (config) => {
        await initializeNeural();
        return new wasm.WasmTrainer(config);
    },
    
    createAgentManager: async () => {
        await initializeNeural();
        return new wasm.AgentNeuralNetworkManager();
    }
};

export default NeuralNetwork;
EOF

# Copy to npm package
echo "📋 Copying to npm package..."
cp -r pkg/* ../../npm/wasm/

# Optimize WASM file size
if command -v wasm-opt &> /dev/null; then
    echo "🎯 Optimizing WASM file size..."
    wasm-opt -O3 -o pkg/ruv_swarm_wasm_bg_optimized.wasm pkg/ruv_swarm_wasm_bg.wasm
    mv pkg/ruv_swarm_wasm_bg_optimized.wasm pkg/ruv_swarm_wasm_bg.wasm
else
    echo "⚠️  wasm-opt not found. Skipping optimization."
fi

# Generate size report
echo ""
echo "📊 Build complete! Size report:"
ls -lh pkg/*.wasm | awk '{print "   WASM size: " $5 " (" $9 ")"}'

# Run basic validation
echo ""
echo "✅ Validating neural network exports..."
node -e "
const pkg = require('./pkg/package.json');
console.log('   Package name:', pkg.name);
console.log('   Package version:', pkg.version);

// Check if TypeScript definitions exist
const fs = require('fs');
const hasDts = fs.existsSync('./pkg/ruv_swarm_wasm.d.ts');
console.log('   TypeScript definitions:', hasDts ? '✓' : '✗');

// Check neural exports
const dts = fs.readFileSync('./pkg/ruv_swarm_wasm.d.ts', 'utf8');
const neuralExports = [
    'WasmNeuralNetwork',
    'WasmTrainer', 
    'AgentNeuralNetworkManager',
    'ActivationFunctionManager',
    'WasmCascadeTrainer'
];

console.log('   Neural network exports:');
neuralExports.forEach(exp => {
    const found = dts.includes(exp);
    console.log('     ' + exp + ':', found ? '✓' : '✗');
});
"

echo ""
echo "🎉 Neural network WASM build complete!"
echo ""
echo "📚 Next steps:"
echo "   1. Test the build: cd ../../npm && npm test"
echo "   2. Run examples: node examples/neural-example.js"
echo "   3. Check performance: npm run benchmark"