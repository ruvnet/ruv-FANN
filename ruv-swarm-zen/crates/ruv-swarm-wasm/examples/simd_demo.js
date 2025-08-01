/**
 * SIMD Performance Demo for ruv-swarm WASM
 *
 * This script demonstrates the SIMD-accelerated neural network operations
 * and benchmarks performance improvements over scalar operations.
 */

// Import the WASM module (assuming it's compiled and available)
import init, {
  WasmNeuralNetwork,
  SimdVectorOps,
  SimdMatrixOps,
  SimdBenchmark,
  get_features,
  detect_simd_capabilities,
  run_simd_verification_suite,
  simd_performance_report,
  validate_simd_implementation,
  ActivationFunction,
} from '../pkg/ruv_swarm_wasm.js';

async function runSimdDemo() {
  // Initialize the WASM module
  await init();

  console.log('=== RUV-SWARM SIMD Performance Demo ===\n');

  // Check feature support
  console.log('1. Feature Detection:');
  const features = JSON.parse(get_features());
  console.log('Available features:', features);

  const simdCaps = JSON.parse(detect_simd_capabilities());
  console.log('SIMD capabilities:', simdCaps);
  console.log('');

  // Validate SIMD implementation
  console.log('2. SIMD Implementation Validation:');
  const isValid = validate_simd_implementation();
  console.log(`SIMD implementation valid: ${isValid}`);
  console.log('');

  // Run verification suite
  console.log('3. SIMD Verification Suite:');
  const verificationResults = run_simd_verification_suite();
  console.log(verificationResults);
  console.log('');

  // Performance benchmarks
  console.log('4. Performance Benchmarks:');

  // Small vectors (good for mobile/embedded)
  console.log('Small vectors (1000 elements, 100 iterations):');
  const smallReport = JSON.parse(simd_performance_report(1000, 100));
  console.log(JSON.stringify(smallReport, null, 2));
  console.log('');

  // Medium vectors (typical neural network layers)
  console.log('Medium vectors (10000 elements, 50 iterations):');
  const mediumReport = JSON.parse(simd_performance_report(10000, 50));
  console.log(JSON.stringify(mediumReport, null, 2));
  console.log('');

  // Large vectors (demanding applications)
  console.log('Large vectors (100000 elements, 10 iterations):');
  const largeReport = JSON.parse(simd_performance_report(100000, 10));
  console.log(JSON.stringify(largeReport, null, 2));
  console.log('');

  // Neural network performance comparison
  console.log('5. Neural Network Performance Comparison:');
  await runNeuralNetworkBenchmark();
  console.log('');

  // Individual operation tests
  console.log('6. Individual SIMD Operations:');
  testSimdOperations();
  console.log('');

  console.log('=== Demo Complete ===');
}

async function runNeuralNetworkBenchmark() {
  const layers = [784, 256, 128, 10]; // MNIST-like network
  const iterations = 100;

  // Create neural network
  const network = new WasmNeuralNetwork(layers, ActivationFunction.ReLU);
  network.randomize_weights(-1.0, 1.0);

  // Generate test input
  const testInput = Array.from({length: 784}, () => Math.random() * 2 - 1);

  // Benchmark neural network inference
  console.log(`Benchmarking neural network [${layers.join(', ')}] with ${iterations} iterations:`);

  const startTime = performance.now();
  for (let i = 0; i < iterations; i++) {
    const output = network.run(testInput);
    // Prevent optimization from removing the computation
    if (i === 0) {
      console.log(`First output sample: [${output.slice(0, 3).map(x => x.toFixed(4)).join(', ')}...]`);
    }
  }
  const endTime = performance.now();

  const totalTime = endTime - startTime;
  const avgTime = totalTime / iterations;
  const throughput = 1000 / avgTime; // inferences per second

  console.log(`Total time: ${totalTime.toFixed(2)}ms`);
  console.log(`Average time per inference: ${avgTime.toFixed(4)}ms`);
  console.log(`Throughput: ${throughput.toFixed(1)} inferences/second`);
}

function testSimdOperations() {
  const vectorOps = new SimdVectorOps();
  const matrixOps = new SimdMatrixOps();

  // Test vector operations
  console.log('Vector Operations:');

  const vecA = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
  const vecB = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

  const dotProduct = vectorOps.dot_product(vecA, vecB);
  console.log(`Dot product: ${dotProduct} (expected: 240)`);

  const vectorSum = vectorOps.vector_add(vecA, vecB);
  console.log(`Vector addition: [${vectorSum.slice(0, 4).join(', ')}...] (expected: [3, 5, 7, 9...])`);

  const scaledVector = vectorOps.vector_scale(vecA, 2.0);
  console.log(`Scaled vector: [${scaledVector.slice(0, 4).join(', ')}...] (expected: [2, 4, 6, 8...])`);

  // Test activation functions
  const testVec = [-2.0, -1.0, 0.0, 1.0, 2.0];
  const reluResult = vectorOps.apply_activation(testVec, 'relu');
  console.log(`ReLU activation: [${reluResult.join(', ')}] (expected: [0, 0, 0, 1, 2])`);

  const sigmoidResult = vectorOps.apply_activation(testVec, 'sigmoid');
  console.log(`Sigmoid activation: [${sigmoidResult.map(x => x.toFixed(3)).join(', ')}]`);

  // Test matrix operations
  console.log('\nMatrix Operations:');

  const matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
  const vector = [1.0, 2.0, 3.0];

  const matVecResult = matrixOps.matrix_vector_multiply(matrix, vector, 2, 3);
  console.log(`Matrix-vector multiply: [${matVecResult.join(', ')}] (expected: [14, 32])`);

  // Test matrix-matrix multiplication
  const matrixA = [1.0, 2.0, 3.0, 4.0]; // 2x2
  const matrixB = [5.0, 6.0, 7.0, 8.0]; // 2x2
  const matMulResult = matrixOps.matrix_multiply(matrixA, matrixB, 2, 2, 2);
  console.log(`Matrix-matrix multiply: [${matMulResult.join(', ')}] (expected: [19, 22, 43, 50])`);
}

// Memory usage monitoring
function checkMemoryUsage() {
  if (performance.memory) {
    const memory = performance.memory;
    console.log('\nMemory Usage:');
    console.log(`Used JS Heap: ${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`Total JS Heap: ${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`JS Heap Limit: ${(memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`);
  }
}

// Browser compatibility check
function checkBrowserSupport() {
  console.log('\nBrowser Support:');
  console.log(`WebAssembly: ${typeof WebAssembly !== 'undefined'}`);
  console.log(`WebAssembly SIMD: ${typeof WebAssembly !== 'undefined' && WebAssembly.validate ? 'Needs testing' : 'Unknown'}`);
  console.log(`Performance API: ${typeof performance !== 'undefined'}`);
  console.log(`High-resolution time: ${typeof performance.now !== 'undefined'}`);
}

// Error handling wrapper
async function main() {
  try {
    checkBrowserSupport();
    await runSimdDemo();
    checkMemoryUsage();
  } catch (error) {
    console.error('Error running SIMD demo:', error);
    console.error('Stack trace:', error.stack);
  }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { main, runSimdDemo };
} else if (typeof window !== 'undefined') {
  window.runSimdDemo = runSimdDemo;
  window.main = main;
}

// Auto-run if this is the main module
if (typeof require !== 'undefined' && require.main === module) {
  main();
}
