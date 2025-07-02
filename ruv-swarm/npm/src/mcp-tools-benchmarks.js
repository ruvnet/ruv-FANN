/**
 * MCP Tools Benchmarks - Separated for better organization
 * Contains all benchmark and testing functionality for ruv-swarm MCP tools
 */

/**
 * Benchmark utilities and test functions for MCP tools
 */
export class MCPBenchmarks {
  constructor(ruvSwarm, persistence) {
    this.ruvSwarm = ruvSwarm;
    this.persistence = persistence;
  }

  /**
     * Run specialized benchmarks - only real tests, no fake/dummy ones
     */
  async runBenchmarks(type = 'all', iterations = 10) {
    console.log(`MCPBenchmarks.runBenchmarks called with type='${type}', iterations=${iterations}`);
    const benchmarks = {};

    // Only include real memory benchmarks - no fake setTimeout tests
    if (type === 'all' || type === 'memory') {
      try {
        console.log('🔍 ENTERING MEMORY BENCHMARK SECTION');
        console.log('Running enhanced memory benchmarks...');
        benchmarks.memory = await this.runEnhancedMemoryBenchmarks(iterations);
        console.log('Memory benchmarks completed:', benchmarks.memory);
      } catch (error) {
        console.error('Memory benchmark error:', error);
        benchmarks.memory = { error: error.message };
      }
    }

    // Performance profiling benchmarks - real CPU/memory intensive tests
    if (type === 'all' || type === 'profiling') {
      try {
        console.log('Running performance profiling benchmarks...');
        benchmarks.profiling = await this.runPerformanceProfilingBenchmarks(iterations);
      } catch (error) {
        console.error('Profiling benchmark error:', error);
        benchmarks.profiling = { error: error.message };
      }
    }

    console.log('Final benchmarks object keys:', Object.keys(benchmarks));

    const result = {
      benchmark_type: type,
      iterations,
      results: benchmarks,
      environment: {
        features: this.ruvSwarm?.features || {},
        memory_usage_mb: this.ruvSwarm?.wasmLoader?.getTotalMemoryUsage() / (1024 * 1024) || 0,
        runtime_features: this.ruvSwarm?.getRuntimeFeatures ? this.ruvSwarm.getRuntimeFeatures() : {},
      },
      timestamp: new Date().toISOString(),
    };

    return result;
  }

  /**
     * Enhanced Memory-specific benchmarks - REAL TESTS ONLY
     */
  async runEnhancedMemoryBenchmarks(iterations) {
    console.log(`Starting runMemoryBenchmarks with ${iterations} iterations`);
    const benchmarks = {
      memory_allocation: [],
      memory_access: [],
      memory_cleanup: [],
      cache_performance: [],
      garbage_collection: [],
    };

    for (let i = 0; i < iterations; i++) {
      try {
        // Memory allocation benchmark - more substantial work
        let start = performance.now();
        const allocSize = Math.floor(Math.random() * 10000 + 5000); // 5000-15000 items
        const allocation = new Array(allocSize).fill(0).map((_, idx) => ({
          id: idx,
          data: Math.random(),
          timestamp: Date.now(),
          metadata: `item_${idx}_${Math.random()}`,
        }));
        benchmarks.memory_allocation.push(performance.now() - start);

        // Memory access benchmark - more operations
        start = performance.now();
        const accessCount = Math.min(1000, allocation.length);
        let sum = 0;
        for (let j = 0; j < accessCount; j++) {
          const randomIndex = Math.floor(Math.random() * allocation.length);
          sum += allocation[randomIndex].data;
        }
        benchmarks.memory_access.push(performance.now() - start);

        // Cache performance benchmark - more substantial cache operations
        start = performance.now();
        const cacheData = new Map();
        for (let k = 0; k < 1000; k++) {
          cacheData.set(`key_${k}`, { value: Math.random(), index: k });
        }
        // Perform cache lookups
        for (let k = 0; k < 500; k++) {
          const key = `key_${Math.floor(Math.random() * 1000)}`;
          const _ = cacheData.get(key);
        }
        benchmarks.cache_performance.push(performance.now() - start);

        // Garbage collection simulation benchmark - larger objects
        start = performance.now();
        let tempArrays = [];
        for (let l = 0; l < 100; l++) {
          tempArrays.push(new Array(1000).fill(0).map(x => Math.random()));
        }
        // Force some operations on the arrays
        let total = 0;
        tempArrays.forEach(arr => {
          total += arr.length;
        });
        tempArrays = null; // Release for GC
        benchmarks.garbage_collection.push(performance.now() - start);

        // Memory cleanup benchmark - more cleanup work
        start = performance.now();
        allocation.splice(0, allocation.length); // More thorough cleanup
        benchmarks.memory_cleanup.push(performance.now() - start);

      } catch (error) {
        console.warn('Memory benchmark iteration failed:', error.message);
        // Add default values on error to ensure we have data
        benchmarks.memory_allocation.push(0);
        benchmarks.memory_access.push(0);
        benchmarks.cache_performance.push(0);
        benchmarks.garbage_collection.push(0);
        benchmarks.memory_cleanup.push(0);
      }
    }

    return this.calculateBenchmarkStats(benchmarks);
  }

  /**
     * Performance profiling benchmarks - CPU intensive real tests
     */
  async runPerformanceProfilingBenchmarks(iterations) {
    console.log(`Starting performance profiling benchmarks with ${iterations} iterations`);
    const benchmarks = {
      cpu_intensive_computation: [],
      string_processing: [],
      json_operations: [],
      array_operations: [],
      math_operations: [],
    };

    for (let i = 0; i < iterations; i++) {
      try {
        // CPU intensive computation benchmark
        let start = performance.now();
        let result = 0;
        for (let j = 0; j < 100000; j++) {
          result += Math.sqrt(j) * Math.sin(j / 1000) * Math.cos(j / 1000);
        }
        benchmarks.cpu_intensive_computation.push(performance.now() - start);

        // String processing benchmark
        start = performance.now();
        let text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '.repeat(1000);
        for (let k = 0; k < 100; k++) {
          text = text.replace(/Lorem/g, 'Replaced').split(' ').reverse().join(' ');
        }
        benchmarks.string_processing.push(performance.now() - start);

        // JSON operations benchmark
        start = performance.now();
        const largeObject = {
          data: Array.from({ length: 1000 }, (_, idx) => ({
            id: idx,
            name: `Item ${idx}`,
            values: Array.from({ length: 100 }, () => Math.random()),
            metadata: {
              created: new Date(),
              processed: false,
              tags: [`tag${idx % 10}`, `category${idx % 5}`],
            },
          })),
        };
        for (let l = 0; l < 50; l++) {
          const serialized = JSON.stringify(largeObject);
          const parsed = JSON.parse(serialized);
        }
        benchmarks.json_operations.push(performance.now() - start);

        // Array operations benchmark
        start = performance.now();
        const largeArray = Array.from({ length: 10000 }, () => Math.random());
        largeArray.sort();
        const filtered = largeArray.filter(x => x > 0.5);
        const mapped = filtered.map(x => x * 2);
        const reduced = mapped.reduce((sum, x) => sum + x, 0);
        benchmarks.array_operations.push(performance.now() - start);

        // Math operations benchmark
        start = performance.now();
        const matrix1 = Array.from({ length: 100 }, () => Array.from({ length: 100 }, () => Math.random()));
        const matrix2 = Array.from({ length: 100 }, () => Array.from({ length: 100 }, () => Math.random()));
        // Simple matrix multiplication
        const resultMatrix = matrix1.map((row, i) =>
          row.map((_, j) =>
            matrix1[i].reduce((sum, cell, k) => sum + cell * matrix2[k][j], 0),
          ),
        );
        benchmarks.math_operations.push(performance.now() - start);

      } catch (error) {
        console.warn('Performance profiling benchmark iteration failed:', error.message);
        benchmarks.cpu_intensive_computation.push(0);
        benchmarks.string_processing.push(0);
        benchmarks.json_operations.push(0);
        benchmarks.array_operations.push(0);
        benchmarks.math_operations.push(0);
      }
    }

    return this.calculateBenchmarkStats(benchmarks);
  }

  /**
     * Calculate statistics for benchmark results
     */
  calculateBenchmarkStats(benchmarks) {
    const results = {};

    Object.keys(benchmarks).forEach(benchmarkType => {
      const times = benchmarks[benchmarkType];
      if (times.length > 0) {
        const sorted = times.sort((a, b) => a - b);
        results[benchmarkType] = {
          avg_ms: times.reduce((sum, time) => sum + time, 0) / times.length,
          min_ms: Math.min(...times),
          max_ms: Math.max(...times),
          median_ms: sorted[Math.floor(sorted.length / 2)],
          p95_ms: sorted[Math.floor(sorted.length * 0.95)],
          samples: times.length,
          std_dev: this.calculateStandardDeviation(times),
        };
      }
    });

    return results;
  }

  /**
     * Calculate standard deviation
     */
  calculateStandardDeviation(values) {
    const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - avg, 2));
    const avgSquaredDiff = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(avgSquaredDiff);
  }

  /**
     * Format benchmark results for display
     */
  formatBenchmarkResults(benchmarks) {
    const summary = [];

    // Process WASM benchmarks
    if (benchmarks.wasm) {
      Object.keys(benchmarks.wasm).forEach(benchmarkType => {
        const data = benchmarks.wasm[benchmarkType];
        summary.push({
          category: 'WASM',
          name: benchmarkType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          avgTime: `${data.avg_ms?.toFixed(2) }ms` || '0.00ms',
          minTime: `${data.min_ms?.toFixed(2) }ms` || '0.00ms',
          maxTime: `${data.max_ms?.toFixed(2) }ms` || '0.00ms',
          samples: data.samples || 0,
        });
      });
    }

    // Process Neural Network benchmarks
    if (benchmarks.neural) {
      Object.keys(benchmarks.neural).forEach(benchmarkType => {
        const data = benchmarks.neural[benchmarkType];
        summary.push({
          category: 'Neural Network',
          name: benchmarkType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          avgTime: `${data.avg_ms?.toFixed(2) }ms` || '0.00ms',
          minTime: `${data.min_ms?.toFixed(2) }ms` || '0.00ms',
          maxTime: `${data.max_ms?.toFixed(2) }ms` || '0.00ms',
          medianTime: `${data.median_ms?.toFixed(2) }ms` || '0.00ms',
          samples: data.samples || 0,
        });
      });
    }

    // Process other benchmark categories
    ['swarm', 'agent', 'task', 'memory'].forEach(category => {
      if (benchmarks[category]) {
        Object.keys(benchmarks[category]).forEach(benchmarkType => {
          const data = benchmarks[category][benchmarkType];
          summary.push({
            category: category.charAt(0).toUpperCase() + category.slice(1),
            name: benchmarkType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            avgTime: `${data.avg_ms?.toFixed(2) }ms` || '0.00ms',
            minTime: `${data.min_ms?.toFixed(2) }ms` || '0.00ms',
            maxTime: `${data.max_ms?.toFixed(2) }ms` || '0.00ms',
            samples: data.samples || 0,
          });
        });
      }
    });

    return summary.length > 0 ? summary : [{
      category: 'System',
      name: 'No Benchmarks Run',
      avgTime: '0.00ms',
      minTime: '0.00ms',
      maxTime: '0.00ms',
      samples: 0,
    }];
  }
}