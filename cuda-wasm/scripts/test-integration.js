#!/usr/bin/env node

/**
 * Integration test suite for @cuda-wasm/core
 * Tests the complete pipeline from CUDA code to execution
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Test colors
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(colors[color] + message + colors.reset);
}

class IntegrationTester {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  addTest(name, testFn) {
    this.tests.push({ name, testFn });
  }

  async runTests() {
    log('\n🧪 Running CUDA-WASM Integration Tests\n', 'cyan');

    for (const test of this.tests) {
      try {
        log(`Running: ${test.name}`, 'blue');
        await test.testFn();
        log(`✅ PASSED: ${test.name}`, 'green');
        this.passed++;
      } catch (error) {
        log(`❌ FAILED: ${test.name}`, 'red');
        log(`   Error: ${error.message}`, 'red');
        this.failed++;
      }
    }

    log(`\n📊 Test Results:`, 'cyan');
    log(`   ✅ Passed: ${this.passed}`, 'green');
    log(`   ❌ Failed: ${this.failed}`, this.failed > 0 ? 'red' : 'green');
    log(`   📈 Success Rate: ${((this.passed / this.tests.length) * 100).toFixed(1)}%`,
      this.failed === 0 ? 'green' : 'yellow');

    return this.failed === 0;
  }

  async runCommand(command, args = []) {
    return new Promise((resolve, reject) => {
      const proc = spawn(command, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        shell: process.platform === 'win32',
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      proc.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      proc.on('close', (code) => {
        if (code === 0) {
          resolve({ stdout, stderr });
        } else {
          reject(new Error(`Command failed with code ${code}: ${stderr}`));
        }
      });
    });
  }
}

// Sample CUDA kernels for testing
const testKernels = {
  vectorAdd: `
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
`,

  matrixMultiply: `
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
`,

  reduction: `
__global__ void reduce(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
`,
};

async function main() {
  const tester = new IntegrationTester();

  // Test 1: Package installation
  tester.addTest('Package can be imported', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      if (!cudaWasm.transpileCuda) {
        throw new Error('Main function not exported');
      }
      if (!cudaWasm.getVersion) {
        throw new Error('Version function not exported');
      }
    } catch (error) {
      throw new Error(`Failed to import package: ${error.message}`);
    }
  });

  // Test 2: CLI availability
  tester.addTest('CLI tool is accessible', async () => {
    try {
      const result = await tester.runCommand('node', ['cli/index.js', '--help']);
      if (!result.stdout.includes('cuda-wasm')) {
        throw new Error('CLI help does not contain expected content');
      }
    } catch (error) {
      throw new Error(`CLI not accessible: ${error.message}`);
    }
  });

  // Test 3: Version command
  tester.addTest('Version command works', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      const version = cudaWasm.getVersion();

      if (!version.version || !version.features) {
        throw new Error('Version info incomplete');
      }
    } catch (error) {
      throw new Error(`Version command failed: ${error.message}`);
    }
  });

  // Test 4: Basic transpilation
  tester.addTest('Basic CUDA transpilation', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      const result = await cudaWasm.transpileCuda(testKernels.vectorAdd, {
        target: 'wasm',
        optimize: false,
      });

      if (!result.code || result.code.length === 0) {
        throw new Error('No transpiled code generated');
      }
    } catch (error) {
      throw new Error(`Transpilation failed: ${error.message}`);
    }
  });

  // Test 5: Kernel analysis
  tester.addTest('Kernel analysis functionality', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      const analysis = await cudaWasm.analyzeKernel(testKernels.vectorAdd);

      if (typeof analysis.threadUtilization !== 'number') {
        throw new Error('Analysis missing thread utilization');
      }

      if (!Array.isArray(analysis.suggestions)) {
        throw new Error('Analysis missing suggestions array');
      }
    } catch (error) {
      throw new Error(`Kernel analysis failed: ${error.message}`);
    }
  });

  // Test 6: Code validation
  tester.addTest('CUDA code validation', async () => {
    try {
      const cudaWasm = require('../dist/index.js');

      const validResult = cudaWasm.validateCudaCode(testKernels.vectorAdd);
      if (!validResult.isValid && validResult.errors.length > 0) {
        throw new Error('Valid CUDA code marked as invalid');
      }

      const invalidResult = cudaWasm.validateCudaCode('invalid code');
      if (invalidResult.warnings.length === 0) {
        throw new Error('Invalid code should generate warnings');
      }
    } catch (error) {
      throw new Error(`Code validation failed: ${error.message}`);
    }
  });

  // Test 7: Kernel parsing
  tester.addTest('CUDA kernel parsing', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      const kernels = cudaWasm.parseCudaKernels(testKernels.vectorAdd);

      if (!Array.isArray(kernels) || kernels.length === 0) {
        throw new Error('No kernels parsed from valid CUDA code');
      }

      if (kernels[0].name !== 'vectorAdd') {
        throw new Error('Incorrect kernel name parsed');
      }
    } catch (error) {
      throw new Error(`Kernel parsing failed: ${error.message}`);
    }
  });

  // Test 8: Benchmark functionality
  tester.addTest('Benchmark functionality', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      const benchmark = await cudaWasm.benchmark(testKernels.vectorAdd, {
        iterations: 5,
        warmupIterations: 2,
      });

      if (typeof benchmark.avgTime !== 'number' || benchmark.avgTime <= 0) {
        throw new Error('Invalid benchmark timing');
      }

      if (typeof benchmark.throughput !== 'number') {
        throw new Error('Missing throughput measurement');
      }
    } catch (error) {
      throw new Error(`Benchmark failed: ${error.message}`);
    }
  });

  // Test 9: WebGPU availability check
  tester.addTest('WebGPU availability check', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      const available = cudaWasm.isWebGPUAvailable();

      // In Node.js environment, this should return false
      if (available !== false) {
        log('   Note: WebGPU availability check behavior may vary by environment', 'yellow');
      }
    } catch (error) {
      throw new Error(`WebGPU availability check failed: ${error.message}`);
    }
  });

  // Test 10: Complex kernel transpilation
  tester.addTest('Complex kernel transpilation', async () => {
    try {
      const cudaWasm = require('../dist/index.js');
      const result = await cudaWasm.transpileCuda(testKernels.reduction, {
        target: 'wasm',
        optimize: true,
        profile: true,
      });

      if (!result.code) {
        throw new Error('No code generated for complex kernel');
      }

      if (!result.profile) {
        throw new Error('Profiling data not generated');
      }

      if (typeof result.profile.totalTime !== 'number') {
        throw new Error('Invalid profiling data');
      }
    } catch (error) {
      throw new Error(`Complex kernel transpilation failed: ${error.message}`);
    }
  });

  // Test 11: File operations test
  tester.addTest('CLI file operations', async () => {
    try {
      // Create a temporary CUDA file
      const tempFile = path.join(__dirname, 'temp_kernel.cu');
      fs.writeFileSync(tempFile, testKernels.vectorAdd);

      try {
        // Test CLI transpilation with file input
        const result = await tester.runCommand('node', [
          'cli/index.js',
          'transpile',
          tempFile,
          '--output',
          'temp_output.wasm',
        ]);

        if (!result.stdout.includes('Transpilation complete') &&
            !result.stdout.includes('Success')) {
          throw new Error('CLI transpilation did not complete successfully');
        }
      } finally {
        // Cleanup
        if (fs.existsSync(tempFile)) {
          fs.unlinkSync(tempFile);
        }
        const outputFile = path.join(__dirname, 'temp_output.wasm');
        if (fs.existsSync(outputFile)) {
          fs.unlinkSync(outputFile);
        }
      }
    } catch (error) {
      throw new Error(`CLI file operations failed: ${error.message}`);
    }
  });

  // Test 12: TypeScript definitions
  tester.addTest('TypeScript definitions available', async () => {
    try {
      const typesFile = path.join(__dirname, '../dist/index.d.ts');

      if (!fs.existsSync(typesFile)) {
        throw new Error('TypeScript definitions file not found');
      }

      const typesContent = fs.readFileSync(typesFile, 'utf8');

      if (!typesContent.includes('TranspileOptions') ||
          !typesContent.includes('transpileCuda')) {
        throw new Error('TypeScript definitions incomplete');
      }
    } catch (error) {
      throw new Error(`TypeScript definitions test failed: ${error.message}`);
    }
  });

  // Run all tests
  const success = await tester.runTests();

  if (success) {
    log('\n🎉 All integration tests passed!', 'green');
    process.exit(0);
  } else {
    log('\n💥 Some integration tests failed!', 'red');
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  main().catch((error) => {
    log(`\n💥 Integration test runner failed: ${error.message}`, 'red');
    process.exit(1);
  });
}

module.exports = { IntegrationTester, testKernels };
