// Example: Vector Addition
const { transpileCuda } = require('cuda-rust-wasm');

const cudaCode = `
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
`;

async function main() {
  try {
    const result = await transpileCuda(cudaCode, {
      target: 'wasm',
      optimize: true,
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
