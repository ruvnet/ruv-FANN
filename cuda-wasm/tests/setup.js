/**
 * Jest test setup for @cuda-wasm/core
 */

// Global test timeout
jest.setTimeout(30000);

// Mock WebGPU for Node.js environment
global.navigator = {
  gpu: undefined, // WebGPU not available in Node.js
};

global.WebAssembly = global.WebAssembly || {
  Module: function() {},
  instantiate: function() {
    return Promise.resolve({
      instance: {
        exports: {},
      },
    });
  },
};

// Setup test utilities
global.testUtils = {
  sampleCudaCode: {
    vectorAdd: `
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}`,

    matrixMul: `
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
}`,

    invalidCode: 'not valid cuda code',
  },
};

// Cleanup after tests
afterEach(() => {
  // Clean up any test artifacts
});

console.log('🧪 @cuda-wasm/core test environment initialized');
