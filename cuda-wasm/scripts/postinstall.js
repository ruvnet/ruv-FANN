#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🚀 CUDA-Rust-WASM Post-install Setup');

// Check if we're in development (npm link) or production
const isDev = process.env.npm_lifecycle_event === 'install' &&
              process.cwd().includes('cuda-rust-wasm');

if (isDev) {
  console.log('📋 Development mode detected, skipping post-install');
  process.exit(0);
}

// Check for WebAssembly support
try {
  new WebAssembly.Module(new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]));
  console.log('✅ WebAssembly support detected');
} catch (e) {
  console.warn('⚠️  WebAssembly not supported in this environment');
  console.warn('   Some features may not be available');
}

// Check for native dependencies
const optionalDeps = [
  { name: 'node-gyp', install: 'npm install -g node-gyp' },
  { name: 'wasm-pack', install: 'cargo install wasm-pack' },
  { name: 'wasm-opt', install: 'npm install -g wasm-opt' },
];

console.log('📋 Checking optional dependencies...');
optionalDeps.forEach(dep => {
  try {
    execSync(`which ${dep.name}`, { stdio: 'ignore' });
    console.log(`  ✅ ${dep.name} found`);
  } catch (e) {
    console.log(`  ⚠️  ${dep.name} not found`);
    console.log(`     Install with: ${dep.install}`);
  }
});

// Create example directory if it doesn't exist
const exampleDir = path.join(process.cwd(), 'cuda-examples');
if (!fs.existsSync(exampleDir)) {
  console.log('📁 Creating example directory...');
  fs.mkdirSync(exampleDir, { recursive: true });

  // Create a simple example
  const exampleContent = `// Example: Vector Addition
const { transpileCuda } = require('cuda-rust-wasm');

const cudaCode = \`
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
\`;

async function main() {
  try {
    const result = await transpileCuda(cudaCode, { 
      target: 'wasm',
      optimize: true 
    });
    
    console.log('✅ Transpilation successful!');
    console.log('Generated code length:', result.code.length);
    
    if (result.wasmBinary) {
      console.log('WASM binary size:', result.wasmBinary.length, 'bytes');
    }
  } catch (error) {
    console.error('❌ Transpilation failed:', error.message);
  }
}

main();
`;

  fs.writeFileSync(
    path.join(exampleDir, 'vector_add.js'),
    exampleContent,
  );

  console.log('✅ Example created at cuda-examples/vector_add.js');
}

console.log('\n🎉 CUDA-Rust-WASM setup complete!');
console.log('📚 Documentation: https://github.com/vibecast/cuda-rust-wasm');
console.log('🚀 Get started with: npx cuda-rust-wasm --help\n');
