// CUDA-Rust-WASM Interactive Demo
class CudaRustWasmDemo {
  constructor() {
    this.runtime = null;
    this.currentKernel = null;
    this.isWebGPUSupported = false;
    this.kernelTemplates = {
      vector_add: `__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}`,
      matrix_mul: `__global__ void matrixMul(float* A, float* B, float* C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < N/16; ++m) {
        As[ty][tx] = A[row * N + m * 16 + tx];
        Bs[ty][tx] = B[(m * 16 + ty) * N + col];
        __syncthreads();
        
        for (int k = 0; k < 16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}`,
      reduction: `__global__ void reduce(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}`,
      convolution: `__global__ void convolution(float* input, float* kernel, float* output, 
                         int width, int height, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int radius = kernelSize / 2;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int inputRow = row + ky;
                int inputCol = col + kx;
                
                if (inputRow >= 0 && inputRow < height && 
                    inputCol >= 0 && inputCol < width) {
                    int inputIdx = inputRow * width + inputCol;
                    int kernelIdx = (ky + radius) * kernelSize + (kx + radius);
                    sum += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}`,
      custom: `// Write your custom CUDA kernel here
__global__ void customKernel(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Your code here
        output[tid] = input[tid] * 2.0f;
    }
}`,
    };

    this.init();
  }

  async init() {
    // Check system capabilities
    await this.checkSystemCapabilities();

    // Initialize event handlers
    this.initEventHandlers();

    // Load default kernel
    this.loadKernel('vector_add');

    // Initialize runtime
    await this.initRuntime();

    this.log('Demo initialized successfully!');
  }

  async checkSystemCapabilities() {
    // Check WebAssembly support
    if (typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function') {
      this.updateStatus('wasm-status', 'supported');
      this.log('✅ WebAssembly supported');
    } else {
      this.updateStatus('wasm-status', 'unsupported');
      this.log('❌ WebAssembly not supported');
    }

    // Check WebGPU support
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.isWebGPUSupported = true;
          this.updateStatus('webgpu-status', 'supported');
          this.log('✅ WebGPU supported');
        } else {
          this.updateStatus('webgpu-status', 'unsupported');
          this.log('❌ WebGPU adapter not available');
        }
      } catch (error) {
        this.updateStatus('webgpu-status', 'error');
        this.log('❌ WebGPU error:', error.message);
      }
    } else {
      this.updateStatus('webgpu-status', 'unsupported');
      this.log('❌ WebGPU not supported');
    }

    // Performance status
    this.updateStatus('perf-status', 'ready');
  }

  updateStatus(elementId, status) {
    const element = document.getElementById(elementId);
    element.className = 'w-4 h-4 rounded-full';

    switch (status) {
      case 'supported':
      case 'ready':
        element.className += ' bg-green-500';
        break;
      case 'unsupported':
        element.className += ' bg-red-500';
        break;
      case 'error':
        element.className += ' bg-yellow-500';
        break;
      default:
        element.className += ' bg-gray-500';
    }
  }

  initEventHandlers() {
    // Kernel selection
    document.getElementById('kernel-select').addEventListener('change', (e) => {
      this.loadKernel(e.target.value);
    });

    // Transpile button
    document.getElementById('transpile-btn').addEventListener('click', () => {
      this.transpileCode();
    });

    // Analyze button
    document.getElementById('analyze-btn').addEventListener('click', () => {
      this.analyzeCode();
    });

    // Output tabs
    document.getElementById('tab-wasm').addEventListener('click', () => {
      this.switchTab('wasm');
    });
    document.getElementById('tab-webgpu').addEventListener('click', () => {
      this.switchTab('webgpu');
    });
    document.getElementById('tab-analysis').addEventListener('click', () => {
      this.switchTab('analysis');
    });

    // Execution buttons
    document.getElementById('run-kernel').addEventListener('click', () => {
      this.runKernel();
    });
    document.getElementById('benchmark-kernel').addEventListener('click', () => {
      this.benchmarkKernel();
    });
  }

  loadKernel(kernelName) {
    const codeEditor = document.getElementById('cuda-code');
    codeEditor.value = this.kernelTemplates[kernelName] || '';
    this.highlightCode();
  }

  highlightCode() {
    // Simple syntax highlighting would go here
    // For now, just update the display
  }

  async initRuntime() {
    try {
      if (typeof CudaRustWasm !== 'undefined') {
        this.runtime = new CudaRustWasm.Runtime();
        this.log('✅ Runtime initialized');
      } else {
        this.log('❌ CudaRustWasm library not loaded');
      }
    } catch (error) {
      this.log('❌ Runtime initialization failed:', error.message);
    }
  }

  async transpileCode() {
    const code = document.getElementById('cuda-code').value;
    const transpileBtn = document.getElementById('transpile-btn');

    if (!code.trim()) {
      this.log('❌ No code to transpile');
      return;
    }

    try {
      // Show loading state
      transpileBtn.innerHTML = '<div class="spinner inline-block mr-2"></div>Transpiling...';
      transpileBtn.disabled = true;

      this.log('🔄 Transpiling CUDA code...');

      // Simulate transpilation (replace with actual API call)
      const result = await this.simulateTranspilation(code);

      this.displayTranspiledCode(result);
      this.updatePerformanceMetrics(result.profile);

      this.log('✅ Transpilation completed successfully');

    } catch (error) {
      this.log('❌ Transpilation failed:', error.message);
      this.displayError(error.message);
    } finally {
      // Restore button state
      transpileBtn.innerHTML = 'Transpile to WebAssembly';
      transpileBtn.disabled = false;
    }
  }

  async simulateTranspilation(code) {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Mock transpilation result
    return {
      wasm: `// Transpiled WebAssembly (text format)
(module
  (func $vectorAdd (param $a i32) (param $b i32) (param $c i32) (param $n i32)
    (local $i i32)
    (local $tid i32)
    
    ;; Calculate thread ID
    local.get $tid
    i32.const 256
    i32.mul
    local.get $i
    i32.add
    local.set $tid
    
    ;; Bounds check
    local.get $tid
    local.get $n
    i32.lt_s
    if
      ;; c[tid] = a[tid] + b[tid]
      local.get $c
      local.get $tid
      i32.const 4
      i32.mul
      i32.add
      
      local.get $a
      local.get $tid
      i32.const 4
      i32.mul
      i32.add
      f32.load
      
      local.get $b
      local.get $tid
      i32.const 4
      i32.mul
      i32.add
      f32.load
      
      f32.add
      f32.store
    end
  )
  (export "vectorAdd" (func $vectorAdd))
)`,
      webgpu: `// Transpiled WebGPU Compute Shader
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= arrayLength(&a)) {
        return;
    }
    
    c[tid] = a[tid] + b[tid];
}`,
      analysis: {
        memoryPattern: 'coalesced',
        threadUtilization: 95,
        sharedMemoryUsage: 0,
        registerUsage: 4,
        occupancy: 0.875,
        suggestions: [
          'Memory access pattern is optimal for this kernel',
          'Consider using grid-stride loop for better scalability',
          'Thread utilization is excellent',
        ],
      },
      profile: {
        parseTime: 12.5,
        transpileTime: 45.2,
        optimizeTime: 8.7,
        totalTime: 66.4,
        binarySize: 2048,
        estimatedPerformance: {
          executionTime: 0.35,
          throughput: 2.8,
          memoryBandwidth: 156.7,
        },
      },
    };
  }

  async analyzeCode() {
    const code = document.getElementById('cuda-code').value;
    const analyzeBtn = document.getElementById('analyze-btn');

    if (!code.trim()) {
      this.log('❌ No code to analyze');
      return;
    }

    try {
      analyzeBtn.innerHTML = '<div class="spinner inline-block mr-2"></div>Analyzing...';
      analyzeBtn.disabled = true;

      this.log('🔍 Analyzing CUDA kernel...');

      const result = await this.simulateAnalysis(code);
      this.displayAnalysis(result);

      this.log('✅ Analysis completed');

    } catch (error) {
      this.log('❌ Analysis failed:', error.message);
    } finally {
      analyzeBtn.innerHTML = 'Analyze Performance';
      analyzeBtn.disabled = false;
    }
  }

  async simulateAnalysis(code) {
    await new Promise(resolve => setTimeout(resolve, 800));

    return {
      memoryPattern: 'coalesced',
      threadUtilization: 95,
      sharedMemoryUsage: 0,
      registerUsage: 4,
      occupancy: 0.875,
      suggestions: [
        'Memory access pattern is optimal',
        'Consider using grid-stride loop',
        'Thread utilization is excellent',
      ],
      bottlenecks: [
        { type: 'memory', severity: 'low', description: 'Memory bandwidth limited' },
      ],
    };
  }

  displayTranspiledCode(result) {
    this.currentTranspilation = result;
    this.switchTab('wasm');
  }

  displayAnalysis(analysis) {
    this.currentAnalysis = analysis;
    this.switchTab('analysis');
  }

  switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('[id^="tab-"]').forEach(btn => {
      btn.className = 'px-3 py-1 bg-gray-600 text-white rounded';
    });
    document.getElementById(`tab-${tab}`).className = 'px-3 py-1 bg-blue-500 text-white rounded';

    // Update content
    const outputArea = document.getElementById('output-area');

    switch (tab) {
      case 'wasm':
        if (this.currentTranspilation) {
          outputArea.innerHTML = `<pre><code class="language-wasm">${this.escapeHtml(this.currentTranspilation.wasm)}</code></pre>`;
        } else {
          outputArea.innerHTML = '<div class="text-gray-400 text-center pt-20">Transpile code to see WebAssembly output</div>';
        }
        break;
      case 'webgpu':
        if (this.currentTranspilation) {
          outputArea.innerHTML = `<pre><code class="language-glsl">${this.escapeHtml(this.currentTranspilation.webgpu)}</code></pre>`;
        } else {
          outputArea.innerHTML = '<div class="text-gray-400 text-center pt-20">Transpile code to see WebGPU shader</div>';
        }
        break;
      case 'analysis':
        if (this.currentAnalysis) {
          outputArea.innerHTML = this.formatAnalysis(this.currentAnalysis);
        } else {
          outputArea.innerHTML = '<div class="text-gray-400 text-center pt-20">Analyze code to see performance insights</div>';
        }
        break;
    }
  }

  formatAnalysis(analysis) {
    return `
            <div class="space-y-4">
                <div>
                    <h3 class="text-lg font-semibold mb-2">Performance Metrics</h3>
                    <div class="grid grid-cols-2 gap-4 text-sm">
                        <div>Memory Pattern: <span class="text-green-400">${analysis.memoryPattern}</span></div>
                        <div>Thread Utilization: <span class="text-blue-400">${analysis.threadUtilization}%</span></div>
                        <div>Shared Memory: <span class="text-purple-400">${analysis.sharedMemoryUsage} bytes</span></div>
                        <div>Register Usage: <span class="text-yellow-400">${analysis.registerUsage} per thread</span></div>
                    </div>
                </div>
                
                <div>
                    <h3 class="text-lg font-semibold mb-2">Optimization Suggestions</h3>
                    <ul class="text-sm space-y-1">
                        ${analysis.suggestions.map(s => `<li class="text-green-400">• ${s}</li>`).join('')}
                    </ul>
                </div>
                
                <div>
                    <h3 class="text-lg font-semibold mb-2">Occupancy</h3>
                    <div class="text-sm">
                        <div class="text-blue-400">Theoretical: ${(analysis.occupancy * 100).toFixed(1)}%</div>
                        <div class="w-full bg-gray-600 rounded-full h-2 mt-1">
                            <div class="bg-blue-400 h-2 rounded-full" style="width: ${analysis.occupancy * 100}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
  }

  updatePerformanceMetrics(profile) {
    if (!profile) return;

    document.getElementById('exec-time').textContent = `${profile.estimatedPerformance.executionTime.toFixed(2)} ms`;
    document.getElementById('throughput').textContent = `${profile.estimatedPerformance.throughput.toFixed(1)} GOPS`;
    document.getElementById('memory-usage').textContent = `${(profile.binarySize / 1024).toFixed(1)} KB`;
    document.getElementById('efficiency').textContent = `${(profile.estimatedPerformance.throughput / 10 * 100).toFixed(0)}%`;
  }

  async runKernel() {
    const runBtn = document.getElementById('run-kernel');

    try {
      runBtn.innerHTML = '<div class="spinner inline-block mr-2"></div>Running...';
      runBtn.disabled = true;

      this.log('🚀 Running kernel...');

      const config = this.getExecutionConfig();
      const result = await this.simulateKernelExecution(config);

      this.log(`✅ Kernel executed successfully in ${result.executionTime.toFixed(2)}ms`);
      this.log(`📊 Processed ${result.elementsProcessed} elements`);

    } catch (error) {
      this.log('❌ Kernel execution failed:', error.message);
    } finally {
      runBtn.innerHTML = 'Run Kernel';
      runBtn.disabled = false;
    }
  }

  async benchmarkKernel() {
    const benchBtn = document.getElementById('benchmark-kernel');

    try {
      benchBtn.innerHTML = '<div class="spinner inline-block mr-2"></div>Benchmarking...';
      benchBtn.disabled = true;

      const config = this.getExecutionConfig();
      this.log(`🏃 Running benchmark (${config.iterations} iterations)...`);

      const results = await this.simulateBenchmark(config);

      this.log(`✅ Benchmark completed:`);
      this.log(`   Average: ${results.avgTime.toFixed(2)}ms`);
      this.log(`   Min: ${results.minTime.toFixed(2)}ms`);
      this.log(`   Max: ${results.maxTime.toFixed(2)}ms`);
      this.log(`   Throughput: ${results.throughput.toFixed(1)} GOPS`);

      this.updatePerformanceMetrics({
        estimatedPerformance: {
          executionTime: results.avgTime,
          throughput: results.throughput,
          memoryBandwidth: results.bandwidth,
        },
      });

    } catch (error) {
      this.log('❌ Benchmark failed:', error.message);
    } finally {
      benchBtn.innerHTML = 'Run Benchmark';
      benchBtn.disabled = false;
    }
  }

  getExecutionConfig() {
    return {
      dataSize: parseInt(document.getElementById('data-size').value),
      blockSize: parseInt(document.getElementById('block-size').value),
      iterations: parseInt(document.getElementById('iterations').value),
    };
  }

  async simulateKernelExecution(config) {
    await new Promise(resolve => setTimeout(resolve, 500));

    return {
      executionTime: 0.5 + Math.random() * 2,
      elementsProcessed: config.dataSize,
      memoryTransferred: config.dataSize * 3 * 4, // 3 arrays * 4 bytes
      success: true,
    };
  }

  async simulateBenchmark(config) {
    const times = [];
    const baseTime = 0.5 + Math.random() * 0.5;

    for (let i = 0; i < config.iterations; i++) {
      await new Promise(resolve => setTimeout(resolve, 10));
      times.push(baseTime + (Math.random() - 0.5) * 0.2);
    }

    times.sort((a, b) => a - b);
    const avgTime = times.reduce((a, b) => a + b) / times.length;

    return {
      avgTime,
      minTime: times[0],
      maxTime: times[times.length - 1],
      throughput: config.dataSize / avgTime / 1000,
      bandwidth: config.dataSize * 3 * 4 / avgTime / 1000000,
    };
  }

  displayError(message) {
    const outputArea = document.getElementById('output-area');
    outputArea.innerHTML = `<div class="text-red-400 text-center pt-20">Error: ${this.escapeHtml(message)}</div>`;
  }

  log(message) {
    const logArea = document.getElementById('execution-log');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}\n`;

    logArea.textContent += logEntry;
    logArea.scrollTop = logArea.scrollHeight;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize demo when page loads
document.addEventListener('DOMContentLoaded', () => {
  new CudaRustWasmDemo();
});
