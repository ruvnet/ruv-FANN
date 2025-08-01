const { transpileCuda, Runtime } = require('cuda-rust-wasm');
const fs = require('fs').promises;
const path = require('path');

async function main() {
  console.log('🚀 CUDA-Rust-WASM Basic Vector Operations Example\n');

  try {
    // Initialize runtime
    console.log('📋 Initializing runtime...');
    const runtime = new Runtime();

    // Load and transpile kernels
    console.log('📂 Loading CUDA kernels...');
    const vectorAddCode = await fs.readFile(
      path.join(__dirname, '../kernels/vector_add.cu'),
      'utf8',
    );

    console.log('🔄 Transpiling CUDA to WebAssembly...');
    const transpiled = await transpileCuda(vectorAddCode, {
      target: 'wasm',
      optimize: true,
      profile: true,
    });

    console.log('✅ Transpilation successful!');
    console.log(`   Generated code size: ${transpiled.code.length} bytes`);
    if (transpiled.wasmBinary) {
      console.log(`   WASM binary size: ${transpiled.wasmBinary.length} bytes`);
    }

    // Compile kernel
    console.log('\n🔨 Compiling kernel...');
    const kernel = await runtime.compileKernel(transpiled.code, 'vectorAdd');

    // Set up test data
    const n = 1024 * 1024; // 1M elements
    const size = n * 4; // float32 = 4 bytes

    console.log(`\n📊 Setting up test data (${n} elements)...`);
    const hostA = new Float32Array(n);
    const hostB = new Float32Array(n);
    const hostC = new Float32Array(n);

    // Initialize with random values
    for (let i = 0; i < n; i++) {
      hostA[i] = Math.random() * 100;
      hostB[i] = Math.random() * 100;
    }

    // Allocate device memory
    console.log('💾 Allocating device memory...');
    const deviceA = await runtime.allocate(size);
    const deviceB = await runtime.allocate(size);
    const deviceC = await runtime.allocate(size);

    // Copy data to device
    console.log('📤 Copying data to device...');
    const copyStart = performance.now();
    await deviceA.copyFrom(hostA.buffer);
    await deviceB.copyFrom(hostB.buffer);
    const copyTime = performance.now() - copyStart;

    // Configure kernel launch
    const blockSize = 256;
    const gridSize = Math.ceil(n / blockSize);

    console.log(`\n🚀 Launching kernel (grid: ${gridSize}, block: ${blockSize})...`);
    kernel.setBlockDim(blockSize);
    kernel.setGridDim(gridSize);
    kernel.setBuffer(0, deviceA);
    kernel.setBuffer(1, deviceB);
    kernel.setBuffer(2, deviceC);
    kernel.setArg(3, n);

    // Launch kernel with timing
    const kernelStart = performance.now();
    await kernel.launch();
    const kernelTime = performance.now() - kernelStart;

    // Copy results back
    console.log('📥 Copying results back...');
    const copyBackStart = performance.now();
    await deviceC.copyTo(hostC.buffer);
    const copyBackTime = performance.now() - copyBackStart;

    // Verify results
    console.log('\n🔍 Verifying results...');
    let errors = 0;
    const tolerance = 1e-5;

    for (let i = 0; i < Math.min(n, 1000); i++) {
      const expected = hostA[i] + hostB[i];
      const actual = hostC[i];

      if (Math.abs(actual - expected) > tolerance) {
        if (errors < 10) {
          console.log(`   Error at index ${i}: expected ${expected}, got ${actual}`);
        }
        errors++;
      }
    }

    if (errors === 0) {
      console.log('✅ All results verified correct!');
    } else {
      console.log(`❌ Found ${errors} errors`);
    }

    // Performance summary
    console.log('\n📊 Performance Summary:');
    console.log(`   Data transfer to device: ${copyTime.toFixed(2)}ms`);
    console.log(`   Kernel execution: ${kernelTime.toFixed(2)}ms`);
    console.log(`   Data transfer from device: ${copyBackTime.toFixed(2)}ms`);
    console.log(`   Total time: ${(copyTime + kernelTime + copyBackTime).toFixed(2)}ms`);

    // Calculate throughput
    const totalElements = n;
    const totalTime = (copyTime + kernelTime + copyBackTime) / 1000; // Convert to seconds
    const throughput = totalElements / totalTime / 1e9; // GOPS

    console.log(`   Throughput: ${throughput.toFixed(2)} GFLOPS`);
    console.log(`   Effective bandwidth: ${(3 * size / totalTime / 1e9).toFixed(2)} GB/s`);

    // Show profiling data if available
    if (transpiled.profile) {
      console.log('\n📈 Profiling Data:');
      console.log(`   Parse time: ${transpiled.profile.parseTime.toFixed(2)}ms`);
      console.log(`   Transpile time: ${transpiled.profile.transpileTime.toFixed(2)}ms`);
      console.log(`   Optimize time: ${transpiled.profile.optimizeTime.toFixed(2)}ms`);
      console.log(`   Total transpile time: ${transpiled.profile.totalTime.toFixed(2)}ms`);
    }

    // Clean up
    console.log('\n🧹 Cleaning up...');
    await runtime.synchronize();

    console.log('✅ Example completed successfully!');

  } catch (error) {
    console.error('❌ Error:', error.message);
    if (error.stack) {
      console.error('Stack trace:', error.stack);
    }
    process.exit(1);
  }
}

// Run the example
main().catch(console.error);
