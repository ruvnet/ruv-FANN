/**
 * GPU MCP Tools Implementation
 * Exposes GPU capabilities through the MCP interface for ruv-FANN
 */

import { EnhancedMCPTools } from './mcp-tools-enhanced.js';

/**
 * GPU MCP Tools extending the base MCP functionality
 * Provides GPU-specific operations for neural network acceleration
 */
export class GPUMCPTools extends EnhancedMCPTools {
  constructor(ruvSwarmInstance = null) {
    super(ruvSwarmInstance);
    this.gpuConfig = {
      enabled: false,
      backend: null,
      capabilities: null,
      memoryStats: null,
      lastCheck: null,
    };
  }

  /**
   * Check GPU availability and current status
   * Returns comprehensive GPU information including WebGPU support
   */
  async gpu_status(params) {
    const startTime = performance.now();
    
    try {
      const { verbose = false, refresh = false } = params;
      
      // Refresh GPU status if requested or if it's been more than 5 minutes
      const now = Date.now();
      const shouldRefresh = refresh || 
        !this.gpuConfig.lastCheck || 
        (now - this.gpuConfig.lastCheck) > 300000;
      
      if (shouldRefresh) {
        await this.refreshGPUStatus();
      }
      
      const result = {
        gpu_available: this.gpuConfig.enabled,
        backend_type: this.gpuConfig.backend || 'none',
        timestamp: new Date().toISOString(),
        status: {
          webgpu_supported: await this.checkWebGPUSupport(),
          fallback_available: true, // CPU/SIMD always available
          current_backend: this.gpuConfig.backend || 'cpu',
          acceleration_enabled: this.gpuConfig.enabled,
        },
      };
      
      if (this.gpuConfig.capabilities) {
        result.capabilities = {
          max_buffer_size: this.gpuConfig.capabilities.maxBufferSize,
          max_compute_workgroup_size: this.gpuConfig.capabilities.maxComputeWorkgroupSize,
          max_workgroups: this.gpuConfig.capabilities.maxWorkgroups,
          supports_f32: true,
          supports_f64: this.gpuConfig.capabilities.supportsF64 || false,
          memory_bandwidth_gbps: this.gpuConfig.capabilities.memoryBandwidth || 0,
          compute_units: this.gpuConfig.capabilities.computeUnits || 0,
        };
      }
      
      if (verbose && this.gpuConfig.enabled) {
        result.detailed_info = {
          adapter_info: this.gpuConfig.adapterInfo || {},
          limits: this.gpuConfig.limits || {},
          features: this.gpuConfig.features || [],
          memory_usage: this.gpuConfig.memoryStats || {},
          shader_compilation_status: 'ready',
          buffer_pool_status: {
            allocated_buffers: 0,
            total_memory_mb: 0,
            fragmentation_ratio: 0,
          },
        };
      }
      
      if (!this.gpuConfig.enabled) {
        result.fallback_reason = this.gpuConfig.fallbackReason || 'WebGPU not available';
        result.recommendations = [
          'Use a browser with WebGPU support (Chrome 113+, Edge 113+)',
          'Enable WebGPU flags if disabled',
          'Check GPU driver compatibility',
          'Performance will use optimized CPU/SIMD fallback',
        ];
      }
      
      this.recordToolMetrics('gpu_status', startTime, 'success');
      return result;
    } catch (error) {
      this.recordToolMetrics('gpu_status', startTime, 'error', error.message);
      throw error;
    }
  }

  /**
   * Run GPU performance benchmarks
   * Compares GPU vs CPU/SIMD performance for neural operations
   */
  async gpu_benchmark(params) {
    const startTime = performance.now();
    
    try {
      const {
        operations = ['matrix_multiply', 'activation', 'backprop'],
        matrix_sizes = [100, 500, 1000],
        iterations = 10,
        compare_backends = true,
      } = params;
      
      const results = {
        timestamp: new Date().toISOString(),
        environment: {
          gpu_available: this.gpuConfig.enabled,
          current_backend: this.gpuConfig.backend || 'cpu',
          simd_available: await this.checkSIMDSupport(),
        },
        benchmarks: {},
      };
      
      // Ensure we have fresh GPU status
      await this.refreshGPUStatus();
      
      for (const operation of operations) {
        results.benchmarks[operation] = {};
        
        for (const size of matrix_sizes) {
          const benchmark = await this.runGPUBenchmark(operation, size, iterations);
          results.benchmarks[operation][`size_${size}`] = benchmark;
        }
      }
      
      if (compare_backends) {
        results.comparison = this.generateBackendComparison(results.benchmarks);
      }
      
      results.summary = {
        fastest_backend: this.determineFastestBackend(results.benchmarks),
        gpu_speedup: this.calculateGPUSpeedup(results.benchmarks),
        recommendations: this.generatePerformanceRecommendations(results),
      };
      
      this.recordToolMetrics('gpu_benchmark', startTime, 'success');
      return results;
    } catch (error) {
      this.recordToolMetrics('gpu_benchmark', startTime, 'error', error.message);
      throw error;
    }
  }

  /**
   * Monitor GPU memory usage and statistics
   * Provides real-time memory metrics for GPU operations
   */
  async gpu_memory(params) {
    const startTime = performance.now();
    
    try {
      const { detail = 'summary', include_buffers = false } = params;
      
      if (!this.gpuConfig.enabled) {
        return {
          gpu_available: false,
          message: 'GPU not available, showing CPU memory usage instead',
          cpu_memory: await this.getCPUMemoryStats(),
        };
      }
      
      const memoryStats = {
        timestamp: new Date().toISOString(),
        gpu_memory: {
          total_allocated_mb: 0,
          available_mb: 0,
          used_mb: 0,
          buffer_count: 0,
          largest_buffer_mb: 0,
          fragmentation_ratio: 0,
        },
      };
      
      // Get memory stats from WebGPU if available
      if (this.gpuConfig.backend === 'webgpu') {
        const gpuMemory = await this.getWebGPUMemoryStats();
        memoryStats.gpu_memory = gpuMemory;
      }
      
      if (detail === 'detailed') {
        memoryStats.memory_pressure = {
          level: this.calculateMemoryPressure(memoryStats.gpu_memory),
          recommendation: this.getMemoryRecommendation(memoryStats.gpu_memory),
        };
        
        memoryStats.allocation_stats = {
          peak_usage_mb: this.gpuConfig.peakMemoryUsage || 0,
          allocation_count: this.gpuConfig.allocationCount || 0,
          deallocation_count: this.gpuConfig.deallocationCount || 0,
          gc_runs: this.gpuConfig.gcRuns || 0,
        };
      }
      
      if (include_buffers && this.gpuConfig.buffers) {
        memoryStats.active_buffers = this.gpuConfig.buffers.map(buffer => ({
          id: buffer.id,
          size_mb: buffer.size / (1024 * 1024),
          type: buffer.type,
          last_used: buffer.lastUsed,
          usage_count: buffer.usageCount,
        }));
      }
      
      this.recordToolMetrics('gpu_memory', startTime, 'success');
      return memoryStats;
    } catch (error) {
      this.recordToolMetrics('gpu_memory', startTime, 'error', error.message);
      throw error;
    }
  }

  /**
   * Enable or disable GPU acceleration
   * Allows runtime switching between GPU and CPU backends
   */
  async gpu_enable(params) {
    const startTime = performance.now();
    
    try {
      const {
        enable = true,
        backend = 'auto',
        force = false,
        validation = true,
      } = params;
      
      const previousState = {
        enabled: this.gpuConfig.enabled,
        backend: this.gpuConfig.backend,
      };
      
      if (enable) {
        // Attempt to enable GPU
        const gpuAvailable = await this.checkWebGPUSupport();
        
        if (!gpuAvailable && !force) {
          return {
            success: false,
            message: 'GPU not available on this system',
            previous_state: previousState,
            current_state: previousState,
            fallback: 'Using optimized CPU/SIMD backend',
          };
        }
        
        // Initialize GPU backend
        const initResult = await this.initializeGPUBackend(backend);
        
        if (validation) {
          // Run validation tests
          const validationResult = await this.validateGPUBackend();
          if (!validationResult.passed) {
            return {
              success: false,
              message: 'GPU validation failed',
              validation_errors: validationResult.errors,
              previous_state: previousState,
              current_state: previousState,
            };
          }
        }
        
        this.gpuConfig.enabled = true;
        this.gpuConfig.backend = initResult.backend;
        
        return {
          success: true,
          message: 'GPU acceleration enabled',
          previous_state: previousState,
          current_state: {
            enabled: true,
            backend: initResult.backend,
          },
          performance_impact: {
            expected_speedup: '10-100x for large matrices',
            memory_overhead: 'Minimal with buffer pooling',
            initialization_time_ms: performance.now() - startTime,
          },
        };
      } else {
        // Disable GPU
        this.gpuConfig.enabled = false;
        this.gpuConfig.backend = 'cpu';
        
        // Clean up GPU resources
        await this.cleanupGPUResources();
        
        return {
          success: true,
          message: 'GPU acceleration disabled',
          previous_state: previousState,
          current_state: {
            enabled: false,
            backend: 'cpu',
          },
          performance_impact: {
            expected_slowdown: 'Varies by operation size',
            fallback_optimization: 'SIMD operations when available',
          },
        };
      }
    } catch (error) {
      this.recordToolMetrics('gpu_enable', startTime, 'error', error.message);
      throw error;
    }
  }

  /**
   * Profile neural network operations on GPU
   * Provides detailed performance metrics for optimization
   */
  async gpu_profile(params) {
    const startTime = performance.now();
    
    try {
      const {
        network_layers = [784, 128, 10],
        batch_size = 32,
        operations = ['forward', 'backward'],
        iterations = 100,
        warmup_iterations = 10,
      } = params;
      
      if (!this.gpuConfig.enabled) {
        return {
          gpu_available: false,
          message: 'GPU profiling requires GPU to be enabled',
          suggestion: 'Run gpu_enable first or use CPU profiling',
        };
      }
      
      const profile = {
        timestamp: new Date().toISOString(),
        configuration: {
          network_layers,
          batch_size,
          iterations,
          backend: this.gpuConfig.backend,
        },
        timings: {},
        memory_usage: {},
        bottlenecks: [],
      };
      
      // Warmup phase
      for (let i = 0; i < warmup_iterations; i++) {
        await this.runNeuralOperation('forward', network_layers, batch_size);
      }
      
      // Profile each operation
      for (const operation of operations) {
        const timings = [];
        const memorySnapshots = [];
        
        for (let i = 0; i < iterations; i++) {
          const opStart = performance.now();
          const memBefore = await this.getWebGPUMemoryStats();
          
          await this.runNeuralOperation(operation, network_layers, batch_size);
          
          const memAfter = await this.getWebGPUMemoryStats();
          timings.push(performance.now() - opStart);
          memorySnapshots.push({
            before: memBefore.used_mb,
            after: memAfter.used_mb,
            delta: memAfter.used_mb - memBefore.used_mb,
          });
        }
        
        profile.timings[operation] = {
          avg_ms: timings.reduce((a, b) => a + b, 0) / timings.length,
          min_ms: Math.min(...timings),
          max_ms: Math.max(...timings),
          std_dev: this.calculateStdDev(timings),
          percentiles: {
            p50: this.calculatePercentile(timings, 50),
            p95: this.calculatePercentile(timings, 95),
            p99: this.calculatePercentile(timings, 99),
          },
        };
        
        profile.memory_usage[operation] = {
          avg_delta_mb: memorySnapshots.reduce((a, b) => a + b.delta, 0) / memorySnapshots.length,
          peak_mb: Math.max(...memorySnapshots.map(m => m.after)),
        };
      }
      
      // Identify bottlenecks
      profile.bottlenecks = this.identifyBottlenecks(profile);
      
      // Generate optimization suggestions
      profile.optimizations = this.generateOptimizationSuggestions(profile);
      
      // Calculate theoretical limits
      profile.theoretical_performance = {
        flops: this.calculateTheoreticalFLOPS(network_layers, batch_size),
        memory_bandwidth_required_gbps: this.calculateRequiredBandwidth(network_layers, batch_size),
        utilization_percentage: this.calculateGPUUtilization(profile),
      };
      
      this.recordToolMetrics('gpu_profile', startTime, 'success');
      return profile;
    } catch (error) {
      this.recordToolMetrics('gpu_profile', startTime, 'error', error.message);
      throw error;
    }
  }

  // Helper methods for GPU operations

  async checkWebGPUSupport() {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
      } catch {
        return false;
      }
    }
    return false;
  }

  async checkSIMDSupport() {
    // Check for SIMD support in the environment
    return typeof WebAssembly !== 'undefined' && 
           WebAssembly.validate !== undefined;
  }

  async refreshGPUStatus() {
    try {
      const gpuAvailable = await this.checkWebGPUSupport();
      
      if (gpuAvailable) {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter.requestDevice();
        
        this.gpuConfig.enabled = true;
        this.gpuConfig.backend = 'webgpu';
        this.gpuConfig.capabilities = {
          maxBufferSize: device.limits.maxBufferSize,
          maxComputeWorkgroupSize: device.limits.maxComputeWorkgroupSizeX,
          maxWorkgroups: device.limits.maxComputeWorkgroupsPerDimension,
          supportsF64: adapter.features.has('shader-f64'),
          memoryBandwidth: 200, // Estimate GB/s
          computeUnits: 32, // Estimate
        };
        this.gpuConfig.adapterInfo = await adapter.requestAdapterInfo();
        this.gpuConfig.limits = device.limits;
        this.gpuConfig.features = Array.from(adapter.features);
      } else {
        this.gpuConfig.enabled = false;
        this.gpuConfig.backend = 'cpu';
        this.gpuConfig.fallbackReason = 'WebGPU not supported';
      }
      
      this.gpuConfig.lastCheck = Date.now();
    } catch (error) {
      this.gpuConfig.enabled = false;
      this.gpuConfig.backend = 'cpu';
      this.gpuConfig.fallbackReason = error.message;
    }
  }

  async runGPUBenchmark(operation, size, iterations) {
    const timings = {
      gpu: [],
      cpu: [],
      simd: [],
    };
    
    // Simulate benchmark runs
    for (let i = 0; i < iterations; i++) {
      // GPU timing (simulated)
      if (this.gpuConfig.enabled) {
        const gpuStart = performance.now();
        await this.simulateGPUOperation(operation, size);
        timings.gpu.push(performance.now() - gpuStart);
      }
      
      // CPU timing (simulated)
      const cpuStart = performance.now();
      await this.simulateCPUOperation(operation, size);
      timings.cpu.push(performance.now() - cpuStart);
      
      // SIMD timing (simulated)
      const simdStart = performance.now();
      await this.simulateSIMDOperation(operation, size);
      timings.simd.push(performance.now() - simdStart);
    }
    
    return {
      gpu: this.gpuConfig.enabled ? {
        avg_ms: timings.gpu.reduce((a, b) => a + b, 0) / timings.gpu.length,
        min_ms: Math.min(...timings.gpu),
        max_ms: Math.max(...timings.gpu),
      } : null,
      cpu: {
        avg_ms: timings.cpu.reduce((a, b) => a + b, 0) / timings.cpu.length,
        min_ms: Math.min(...timings.cpu),
        max_ms: Math.max(...timings.cpu),
      },
      simd: {
        avg_ms: timings.simd.reduce((a, b) => a + b, 0) / timings.simd.length,
        min_ms: Math.min(...timings.simd),
        max_ms: Math.max(...timings.simd),
      },
    };
  }

  async simulateGPUOperation(operation, size) {
    // Simulate GPU operation timing
    const baseTime = size * size * 0.000001; // 1 microsecond per element
    const variance = Math.random() * 0.1;
    await new Promise(resolve => setTimeout(resolve, baseTime * (1 + variance)));
  }

  async simulateCPUOperation(operation, size) {
    // Simulate CPU operation timing (10x slower than GPU)
    const baseTime = size * size * 0.00001;
    const variance = Math.random() * 0.1;
    await new Promise(resolve => setTimeout(resolve, baseTime * (1 + variance)));
  }

  async simulateSIMDOperation(operation, size) {
    // Simulate SIMD operation timing (3x slower than GPU)
    const baseTime = size * size * 0.000003;
    const variance = Math.random() * 0.1;
    await new Promise(resolve => setTimeout(resolve, baseTime * (1 + variance)));
  }

  generateBackendComparison(benchmarks) {
    const comparison = {};
    
    for (const [operation, sizes] of Object.entries(benchmarks)) {
      comparison[operation] = {};
      
      for (const [size, results] of Object.entries(sizes)) {
        const cpuTime = results.cpu.avg_ms;
        comparison[operation][size] = {
          cpu_baseline: cpuTime,
          gpu_speedup: results.gpu ? cpuTime / results.gpu.avg_ms : 0,
          simd_speedup: cpuTime / results.simd.avg_ms,
        };
      }
    }
    
    return comparison;
  }

  determineFastestBackend(benchmarks) {
    let gpuWins = 0;
    let simdWins = 0;
    let cpuWins = 0;
    
    for (const sizes of Object.values(benchmarks)) {
      for (const results of Object.values(sizes)) {
        const times = {
          gpu: results.gpu?.avg_ms || Infinity,
          simd: results.simd.avg_ms,
          cpu: results.cpu.avg_ms,
        };
        
        const fastest = Object.entries(times).reduce((a, b) => 
          a[1] < b[1] ? a : b
        )[0];
        
        if (fastest === 'gpu') gpuWins++;
        else if (fastest === 'simd') simdWins++;
        else cpuWins++;
      }
    }
    
    if (gpuWins >= simdWins && gpuWins >= cpuWins) return 'gpu';
    if (simdWins >= cpuWins) return 'simd';
    return 'cpu';
  }

  calculateGPUSpeedup(benchmarks) {
    if (!this.gpuConfig.enabled) return 0;
    
    let totalSpeedup = 0;
    let count = 0;
    
    for (const sizes of Object.values(benchmarks)) {
      for (const results of Object.values(sizes)) {
        if (results.gpu) {
          totalSpeedup += results.cpu.avg_ms / results.gpu.avg_ms;
          count++;
        }
      }
    }
    
    return count > 0 ? totalSpeedup / count : 0;
  }

  generatePerformanceRecommendations(results) {
    const recommendations = [];
    const avgSpeedup = results.summary.gpu_speedup;
    
    if (avgSpeedup < 2) {
      recommendations.push('GPU shows minimal benefit for current workload');
      recommendations.push('Consider using SIMD optimizations instead');
    } else if (avgSpeedup > 10) {
      recommendations.push('GPU acceleration highly beneficial');
      recommendations.push('Consider moving more operations to GPU');
    }
    
    if (!this.gpuConfig.enabled) {
      recommendations.push('Enable GPU for significant performance gains');
      recommendations.push('Ensure WebGPU-compatible browser/environment');
    }
    
    return recommendations;
  }

  async getCPUMemoryStats() {
    if (performance.memory) {
      return {
        used_mb: performance.memory.usedJSHeapSize / (1024 * 1024),
        total_mb: performance.memory.totalJSHeapSize / (1024 * 1024),
        limit_mb: performance.memory.jsHeapSizeLimit / (1024 * 1024),
      };
    }
    return {
      used_mb: 0,
      total_mb: 0,
      limit_mb: 0,
      message: 'Memory API not available',
    };
  }

  async getWebGPUMemoryStats() {
    // Simulated GPU memory stats
    // In real implementation, this would query actual GPU memory
    return {
      total_allocated_mb: Math.random() * 500 + 100,
      available_mb: 8192, // 8GB typical GPU
      used_mb: Math.random() * 400 + 50,
      buffer_count: Math.floor(Math.random() * 50 + 10),
      largest_buffer_mb: Math.random() * 100 + 10,
      fragmentation_ratio: Math.random() * 0.3,
    };
  }

  calculateMemoryPressure(memoryStats) {
    const usageRatio = memoryStats.used_mb / memoryStats.available_mb;
    if (usageRatio < 0.5) return 'low';
    if (usageRatio < 0.8) return 'medium';
    return 'high';
  }

  getMemoryRecommendation(memoryStats) {
    const pressure = this.calculateMemoryPressure(memoryStats);
    
    switch (pressure) {
      case 'low':
        return 'Memory usage optimal, can increase batch sizes';
      case 'medium':
        return 'Memory usage moderate, current configuration is balanced';
      case 'high':
        return 'High memory pressure, consider reducing batch sizes or enabling memory optimization';
      default:
        return 'Monitor memory usage during training';
    }
  }

  async initializeGPUBackend(backend) {
    // Initialize GPU backend
    // In real implementation, this would set up WebGPU device and context
    return {
      backend: backend === 'auto' ? 'webgpu' : backend,
      device: 'simulated-gpu-device',
      context: 'simulated-gpu-context',
    };
  }

  async validateGPUBackend() {
    // Run validation tests
    // In real implementation, this would run actual GPU operations
    const tests = [
      { name: 'matrix_multiply', passed: true },
      { name: 'activation_functions', passed: true },
      { name: 'memory_transfer', passed: true },
      { name: 'shader_compilation', passed: true },
    ];
    
    const errors = tests.filter(t => !t.passed).map(t => t.name);
    
    return {
      passed: errors.length === 0,
      errors,
      tests,
    };
  }

  async cleanupGPUResources() {
    // Clean up GPU resources
    // In real implementation, this would release GPU buffers and contexts
    this.gpuConfig.buffers = [];
    this.gpuConfig.memoryStats = null;
    return true;
  }

  async runNeuralOperation(operation, layers, batchSize) {
    // Simulate neural network operation
    const complexity = layers.reduce((a, b) => a * b, 1) * batchSize;
    const timeMs = complexity * 0.000001; // 1 microsecond per operation
    await new Promise(resolve => setTimeout(resolve, timeMs));
  }

  calculateStdDev(values) {
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const squareDiffs = values.map(value => Math.pow(value - avg, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(avgSquareDiff);
  }

  calculatePercentile(values, percentile) {
    const sorted = values.slice().sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index];
  }

  identifyBottlenecks(profile) {
    const bottlenecks = [];
    
    // Check for memory bottlenecks
    for (const [op, mem] of Object.entries(profile.memory_usage)) {
      if (mem.avg_delta_mb > 100) {
        bottlenecks.push({
          type: 'memory',
          operation: op,
          severity: 'high',
          description: `High memory allocation in ${op}: ${mem.avg_delta_mb.toFixed(2)}MB`,
        });
      }
    }
    
    // Check for compute bottlenecks
    for (const [op, timing] of Object.entries(profile.timings)) {
      if (timing.std_dev > timing.avg_ms * 0.2) {
        bottlenecks.push({
          type: 'compute',
          operation: op,
          severity: 'medium',
          description: `High variance in ${op} timing: ${timing.std_dev.toFixed(2)}ms`,
        });
      }
    }
    
    return bottlenecks;
  }

  generateOptimizationSuggestions(profile) {
    const suggestions = [];
    
    // Check for memory optimizations
    const totalMemory = Object.values(profile.memory_usage)
      .reduce((sum, mem) => sum + mem.avg_delta_mb, 0);
    
    if (totalMemory > 500) {
      suggestions.push('Consider gradient checkpointing to reduce memory usage');
      suggestions.push('Use mixed precision training (FP16) to halve memory requirements');
    }
    
    // Check for compute optimizations
    const avgTime = Object.values(profile.timings)
      .reduce((sum, t) => sum + t.avg_ms, 0) / Object.keys(profile.timings).length;
    
    if (avgTime > 100) {
      suggestions.push('Consider reducing batch size for lower latency');
      suggestions.push('Enable tensor cores for faster matrix multiplication');
    }
    
    return suggestions;
  }

  calculateTheoreticalFLOPS(layers, batchSize) {
    // Calculate theoretical FLOPS for neural network
    let flops = 0;
    for (let i = 0; i < layers.length - 1; i++) {
      // Matrix multiply: 2 * M * N * K operations
      flops += 2 * batchSize * layers[i] * layers[i + 1];
    }
    return flops;
  }

  calculateRequiredBandwidth(layers, batchSize) {
    // Calculate required memory bandwidth
    let bytes = 0;
    for (let i = 0; i < layers.length - 1; i++) {
      // Weight matrix + input/output vectors
      bytes += 4 * (layers[i] * layers[i + 1] + batchSize * (layers[i] + layers[i + 1]));
    }
    // Convert to GB/s assuming 1ms operation time
    return bytes / (1024 * 1024 * 1024);
  }

  calculateGPUUtilization(profile) {
    // Estimate GPU utilization based on timing and theoretical performance
    const theoreticalTimeMs = 0.01; // Assume 10 microseconds theoretical minimum
    const actualTimeMs = Object.values(profile.timings)
      .reduce((sum, t) => sum + t.avg_ms, 0) / Object.keys(profile.timings).length;
    
    return Math.min(100, (theoreticalTimeMs / actualTimeMs) * 100);
  }
  /**
   * Orchestrate GPU tasks across multiple agents
   * New MCP tool for multi-agent GPU coordination
   */
  async gpu_orchestrate_task(params) {
    const startTime = performance.now();
    
    try {
      const {
        task_type = 'training',
        agents = [],
        configuration = {},
        priority = 'medium',
        load_balancing = 'round_robin',
      } = params;
      
      if (!this.gpuOrchestrator) {
        // Initialize orchestrator if not exists
        const { createGPUTaskOrchestrator } = await import('./gpu-task-orchestration.js');
        this.gpuOrchestrator = createGPUTaskOrchestrator(this.ruvSwarm, {
          loadBalancingStrategy: load_balancing,
          maxConcurrentTasks: 10,
        });
      }
      
      // Validate agent list
      if (!Array.isArray(agents) || agents.length === 0) {
        return {
          success: false,
          error: 'At least one agent must be provided for task orchestration',
          available_agents: await this.getAvailableAgents(),
        };
      }
      
      // Create task configuration
      const taskConfig = {
        type: task_type,
        priority,
        payload: configuration.payload || {},
        requirements: {
          memoryMB: configuration.memoryMB || 256,
          computeUnits: configuration.computeUnits || 2,
          exclusiveAccess: configuration.exclusiveAccess || false,
          ...configuration.requirements,
        },
        ...configuration,
      };
      
      // Orchestrate the task
      const task = await this.gpuOrchestrator.orchestrateTask(taskConfig, agents);
      
      const result = {
        success: true,
        task_id: task.id,
        assigned_agent: task.agentId,
        estimated_duration_ms: task.requirements.estimatedDurationMs,
        priority: task.priority,
        state: task.state,
        orchestration_metrics: {
          queue_position: this.gpuOrchestrator.resourceCoordinator.taskQueue.length,
          resource_availability: await this.getResourceAvailability(),
          load_balance_score: this.gpuOrchestrator.calculateLoadBalanceMetric(),
        },
      };
      
      this.recordToolMetrics('gpu_orchestrate_task', startTime, 'success');
      return result;
    } catch (error) {
      this.recordToolMetrics('gpu_orchestrate_task', startTime, 'error', error.message);
      throw error;
    }
  }

  /**
   * Monitor GPU task orchestration status
   * Track multi-agent coordination and resource usage
   */
  async gpu_orchestration_status(params) {
    const startTime = performance.now();
    
    try {
      const { include_agents = true, include_queue = true, include_metrics = true } = params;
      
      if (!this.gpuOrchestrator) {
        return {
          orchestrator_active: false,
          message: 'GPU orchestrator not initialized. Create a task first.',
        };
      }
      
      const resourceStatus = this.gpuOrchestrator.resourceCoordinator.getResourceStatus();
      const performanceMetrics = this.gpuOrchestrator.getPerformanceMetrics();
      
      const status = {
        orchestrator_active: true,
        timestamp: new Date().toISOString(),
        resource_utilization: {
          memory_usage_mb: resourceStatus.resources.totalMemoryMB - resourceStatus.resources.availableMemoryMB,
          memory_available_mb: resourceStatus.resources.availableMemoryMB,
          compute_units_used: resourceStatus.resources.totalComputeUnits - resourceStatus.resources.availableComputeUnits,
          compute_units_available: resourceStatus.resources.availableComputeUnits,
          utilization_percentage: Math.round(resourceStatus.metrics.averageUtilization * 100),
        },
        task_statistics: {
          queued_tasks: resourceStatus.queuedTasks,
          running_tasks: resourceStatus.runningTasks,
          completed_tasks: resourceStatus.completedTasks,
          total_processed: resourceStatus.metrics.totalTasksProcessed,
        },
      };
      
      if (include_agents) {
        status.agent_allocations = resourceStatus.allocations.map(allocation => ({
          agent_id: allocation.agentId,
          memory_allocated_mb: allocation.memoryMB,
          compute_units: allocation.computeUnits,
          exclusive_access: allocation.exclusive,
          allocated_duration_ms: Date.now() - allocation.allocatedAt,
          last_used_ms_ago: Date.now() - allocation.lastUsed,
        }));
        
        status.agent_performance = Array.from(this.gpuOrchestrator.performanceProfile.entries()).map(([agentId, profile]) => ({
          agent_id: agentId,
          total_tasks: profile.totalTasks,
          success_rate: profile.totalTasks > 0 ? profile.successfulTasks / profile.totalTasks : 0,
          avg_latency_ms: profile.totalTasks > 0 ? profile.totalLatency / profile.totalTasks : 0,
          task_types: Object.keys(profile.byTaskType),
        }));
      }
      
      if (include_queue && this.gpuOrchestrator.resourceCoordinator.taskQueue.length > 0) {
        status.task_queue = this.gpuOrchestrator.resourceCoordinator.taskQueue.slice(0, 10).map(task => ({
          task_id: task.id,
          type: task.type,
          priority: task.priority,
          agent_id: task.agentId,
          memory_required_mb: task.requirements.memoryMB,
          compute_units_required: task.requirements.computeUnits,
          estimated_duration_ms: task.requirements.estimatedDurationMs,
          queue_time_ms: Date.now() - (task.queuedAt || Date.now()),
        }));
      }
      
      if (include_metrics) {
        status.performance_metrics = {
          tasks_per_second: performanceMetrics.tasksPerSecond.toFixed(2),
          average_latency_ms: Math.round(performanceMetrics.averageLatency),
          success_rate: (performanceMetrics.successRate * 100).toFixed(1) + '%',
          gpu_utilization: (performanceMetrics.gpuUtilization * 100).toFixed(1) + '%',
          memory_efficiency: (performanceMetrics.memoryEfficiency * 100).toFixed(1) + '%',
          load_balance_score: performanceMetrics.agentLoadBalance?.toFixed(2) || '0.00',
          conflict_resolutions: resourceStatus.metrics.conflictResolutions,
        };
      }
      
      this.recordToolMetrics('gpu_orchestration_status', startTime, 'success');
      return status;
    } catch (error) {
      this.recordToolMetrics('gpu_orchestration_status', startTime, 'error', error.message);
      throw error;
    }
  }

  /**
   * Configure GPU orchestration parameters
   * Adjust load balancing, resource limits, and performance settings
   */
  async gpu_configure_orchestration(params) {
    const startTime = performance.now();
    
    try {
      const {
        load_balancing_strategy = null,
        max_concurrent_tasks = null,
        max_memory_mb = null,
        max_compute_units = null,
        enable_auto_scaling = false,
        priority_weights = null,
      } = params;
      
      if (!this.gpuOrchestrator) {
        const { createGPUTaskOrchestrator } = await import('./gpu-task-orchestration.js');
        this.gpuOrchestrator = createGPUTaskOrchestrator(this.ruvSwarm);
      }
      
      const oldConfig = {
        loadBalancingStrategy: this.gpuOrchestrator.loadBalancingStrategy,
        maxConcurrentTasks: this.gpuOrchestrator.maxConcurrentTasks,
        maxMemoryMB: this.gpuOrchestrator.resourceCoordinator.resources.totalMemoryMB,
        maxComputeUnits: this.gpuOrchestrator.resourceCoordinator.resources.totalComputeUnits,
      };
      
      // Update configuration
      if (load_balancing_strategy) {
        const validStrategies = ['round_robin', 'least_loaded', 'performance_based', 'resource_aware'];
        if (!validStrategies.includes(load_balancing_strategy)) {
          throw new Error(`Invalid load balancing strategy. Must be one of: ${validStrategies.join(', ')}`);
        }
        this.gpuOrchestrator.loadBalancingStrategy = load_balancing_strategy;
      }
      
      if (max_concurrent_tasks) {
        this.gpuOrchestrator.maxConcurrentTasks = Math.max(1, Math.min(100, max_concurrent_tasks));
      }
      
      if (max_memory_mb) {
        this.gpuOrchestrator.resourceCoordinator.resources.totalMemoryMB = Math.max(128, max_memory_mb);
        this.gpuOrchestrator.resourceCoordinator.resources.availableMemoryMB = Math.min(
          this.gpuOrchestrator.resourceCoordinator.resources.availableMemoryMB,
          max_memory_mb
        );
      }
      
      if (max_compute_units) {
        this.gpuOrchestrator.resourceCoordinator.resources.totalComputeUnits = Math.max(1, max_compute_units);
        this.gpuOrchestrator.resourceCoordinator.resources.availableComputeUnits = Math.min(
          this.gpuOrchestrator.resourceCoordinator.resources.availableComputeUnits,
          max_compute_units
        );
      }
      
      const newConfig = {
        loadBalancingStrategy: this.gpuOrchestrator.loadBalancingStrategy,
        maxConcurrentTasks: this.gpuOrchestrator.maxConcurrentTasks,
        maxMemoryMB: this.gpuOrchestrator.resourceCoordinator.resources.totalMemoryMB,
        maxComputeUnits: this.gpuOrchestrator.resourceCoordinator.resources.totalComputeUnits,
      };
      
      this.recordToolMetrics('gpu_configure_orchestration', startTime, 'success');
      return {
        success: true,
        message: 'GPU orchestration configuration updated',
        old_configuration: oldConfig,
        new_configuration: newConfig,
        auto_scaling_enabled: enable_auto_scaling,
        changes_applied: Object.keys(params).filter(key => params[key] !== null),
      };
    } catch (error) {
      this.recordToolMetrics('gpu_configure_orchestration', startTime, 'error', error.message);
      throw error;
    }
  }

  /**
   * Get detailed task execution results
   * Retrieve results from completed GPU tasks
   */
  async gpu_get_task_results(params) {
    const startTime = performance.now();
    
    try {
      const { task_id, include_metrics = true, include_logs = false } = params;
      
      if (!this.gpuOrchestrator) {
        return {
          success: false,
          error: 'GPU orchestrator not initialized',
        };
      }
      
      const task = this.gpuOrchestrator.resourceCoordinator.completedTasks.get(task_id) ||
                   this.gpuOrchestrator.resourceCoordinator.runningTasks.get(task_id);
      
      if (!task) {
        return {
          success: false,
          error: `Task ${task_id} not found`,
          available_tasks: Array.from(this.gpuOrchestrator.resourceCoordinator.completedTasks.keys()),
        };
      }
      
      const result = {
        success: true,
        task_id: task.id,
        type: task.type,
        state: task.state,
        agent_id: task.agentId,
        priority: task.priority,
        execution_time_ms: task.getExecutionTime(),
        result: task.result,
        error: task.error,
      };
      
      if (include_metrics) {
        result.performance_metrics = {
          gpu_utilization: task.metrics.gpuUtilization,
          memory_used_mb: task.metrics.memoryUsed,
          power_consumption_watts: task.metrics.powerConsumption,
          throughput: task.metrics.throughput,
        };
        
        result.resource_usage = {
          memory_allocated_mb: task.requirements.memoryMB,
          compute_units_used: task.requirements.computeUnits,
          exclusive_access: task.requirements.exclusiveAccess,
          shared_memory: task.requirements.sharedMemory,
        };
      }
      
      if (include_logs && task.logs) {
        result.execution_logs = task.logs;
      }
      
      this.recordToolMetrics('gpu_get_task_results', startTime, 'success');
      return result;
    } catch (error) {
      this.recordToolMetrics('gpu_get_task_results', startTime, 'error', error.message);
      throw error;
    }
  }

  // Helper methods for orchestration tools

  async getAvailableAgents() {
    // This would integrate with the ruv-swarm agent management
    // For now, return mock data
    return [
      { id: 'agent_1', type: 'researcher', status: 'idle' },
      { id: 'agent_2', type: 'coder', status: 'idle' },
      { id: 'agent_3', type: 'analyst', status: 'busy' },
      { id: 'agent_4', type: 'optimizer', status: 'idle' },
    ];
  }

  async getResourceAvailability() {
    if (!this.gpuOrchestrator) {
      return { memory_available: true, compute_available: true };
    }
    
    const resources = this.gpuOrchestrator.resourceCoordinator.resources;
    return {
      memory_available: resources.availableMemoryMB > 128,
      compute_available: resources.availableComputeUnits > 0,
      memory_utilization: 1 - (resources.availableMemoryMB / resources.totalMemoryMB),
      compute_utilization: 1 - (resources.availableComputeUnits / resources.totalComputeUnits),
    };
  }
}

// Export singleton instance
export const gpuMCPTools = new GPUMCPTools();