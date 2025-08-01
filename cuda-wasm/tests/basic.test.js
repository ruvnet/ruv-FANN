/**
 * Basic functionality tests for @cuda-wasm/core
 */

const cudaWasm = require('../dist/index.js');

describe('@cuda-wasm/core Basic Tests', () => {
  test('module exports all required functions', () => {
    expect(typeof cudaWasm.transpileCuda).toBe('function');
    expect(typeof cudaWasm.analyzeKernel).toBe('function');
    expect(typeof cudaWasm.benchmark).toBe('function');
    expect(typeof cudaWasm.getVersion).toBe('function');
    expect(typeof cudaWasm.validateCudaCode).toBe('function');
    expect(typeof cudaWasm.parseCudaKernels).toBe('function');
    expect(typeof cudaWasm.isWebGPUAvailable).toBe('function');
    expect(typeof cudaWasm.configure).toBe('function');
  });

  test('version information is available', () => {
    const version = cudaWasm.getVersion();
    expect(version).toHaveProperty('version');
    expect(version).toHaveProperty('features');
    expect(Array.isArray(version.features)).toBe(true);
    expect(version.version).toMatch(/^\d+\.\d+\.\d+/);
  });

  test('WebGPU availability check works', () => {
    const available = cudaWasm.isWebGPUAvailable();
    expect(typeof available).toBe('boolean');
    // In Node.js environment, should be false
    expect(available).toBe(false);
  });

  test('CUDA code validation works', () => {
    const validResult = cudaWasm.validateCudaCode(global.testUtils.sampleCudaCode.vectorAdd);
    expect(validResult).toHaveProperty('isValid');
    expect(validResult).toHaveProperty('errors');
    expect(validResult).toHaveProperty('warnings');
    expect(Array.isArray(validResult.errors)).toBe(true);
    expect(Array.isArray(validResult.warnings)).toBe(true);
  });

  test('kernel parsing works', () => {
    const kernels = cudaWasm.parseCudaKernels(global.testUtils.sampleCudaCode.vectorAdd);
    expect(Array.isArray(kernels)).toBe(true);
    if (kernels.length > 0) {
      expect(kernels[0]).toHaveProperty('name');
      expect(kernels[0]).toHaveProperty('parameters');
    }
  });

  test('error classes are available', () => {
    expect(typeof cudaWasm.TranspilationError).toBe('function');
    expect(typeof cudaWasm.WebGPUError).toBe('function');
    expect(typeof cudaWasm.KernelExecutionError).toBe('function');
  });

  test('module configuration works', () => {
    expect(() => {
      cudaWasm.configure({ debug: true });
    }).not.toThrow();
  });
});
