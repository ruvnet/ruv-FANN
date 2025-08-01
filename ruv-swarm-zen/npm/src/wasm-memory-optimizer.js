/**
 * WASM Memory Optimizer
 *
 * Advanced memory management and allocation optimization for WASM modules
 * with progressive loading, memory pooling, and garbage collection strategies.
 */

class WasmMemoryPool {
  constructor(initialSize = 16 * 1024 * 1024) { // 16MB initial
    this.pools = new Map();
    this.allocations = new Map();
    this.totalAllocated = 0;
    this.maxMemory = 512 * 1024 * 1024; // 512MB max
    this.initialSize = initialSize;
    this.allocationCounter = 0;
    this.gcThreshold = 0.8; // GC when 80% full
    this.compressionEnabled = true;
  }

  /**
   * Get or create memory pool for specific module
   */
  getPool(moduleId, requiredSize = this.initialSize) {
    if (!this.pools.has(moduleId)) {
      const poolSize = Math.max(requiredSize, this.initialSize);
      const memory = new WebAssembly.Memory({
        initial: Math.ceil(poolSize / (64 * 1024)), // Pages are 64KB
        maximum: Math.ceil(this.maxMemory / (64 * 1024)),
        shared: false,
      });

      this.pools.set(moduleId, {
        memory,
        allocated: 0,
        maxSize: poolSize,
        freeBlocks: [],
        allocations: new Map(),
      });

      console.log(`🧠 Created memory pool for ${moduleId}: ${poolSize / 1024 / 1024}MB`);
    }

    return this.pools.get(moduleId);
  }

  /**
   * Allocate memory with alignment and tracking
   */
  allocate(moduleId, size, alignment = 16) {
    const pool = this.getPool(moduleId, size * 2);
    const alignedSize = Math.ceil(size / alignment) * alignment;

    // Try to reuse free blocks first
    const freeBlock = this.findFreeBlock(pool, alignedSize);
    if (freeBlock) {
      this.allocationCounter++;
      const allocation = {
        id: this.allocationCounter,
        moduleId,
        offset: freeBlock.offset,
        size: alignedSize,
        timestamp: Date.now(),
      };

      pool.allocations.set(allocation.id, allocation);
      this.allocations.set(allocation.id, allocation);

      return {
        id: allocation.id,
        offset: freeBlock.offset,
        ptr: pool.memory.buffer.slice(freeBlock.offset, freeBlock.offset + alignedSize),
      };
    }

    // Allocate new memory
    const currentSize = pool.memory.buffer.byteLength;
    const newOffset = pool.allocated;

    if (newOffset + alignedSize > currentSize) {
      // Need to grow memory
      const requiredPages = Math.ceil((newOffset + alignedSize - currentSize) / (64 * 1024));
      try {
        pool.memory.grow(requiredPages);
        console.log(`📈 Grew memory for ${moduleId} by ${requiredPages} pages`);
      } catch (error) {
        console.error(`❌ Failed to grow memory for ${moduleId}:`, error);
        // Try garbage collection
        this.garbageCollect(moduleId);
        return this.allocate(moduleId, size, alignment); // Retry after GC
      }
    }

    this.allocationCounter++;
    const allocation = {
      id: this.allocationCounter,
      moduleId,
      offset: newOffset,
      size: alignedSize,
      timestamp: Date.now(),
    };

    pool.allocated = newOffset + alignedSize;
    pool.allocations.set(allocation.id, allocation);
    this.allocations.set(allocation.id, allocation);
    this.totalAllocated += alignedSize;

    // Check if GC is needed
    if (this.getMemoryUtilization() > this.gcThreshold) {
      setTimeout(() => this.garbageCollectAll(), 100);
    }

    return {
      id: allocation.id,
      offset: newOffset,
      ptr: pool.memory.buffer.slice(newOffset, newOffset + alignedSize),
    };
  }

  /**
   * Find suitable free block
   */
  findFreeBlock(pool, size) {
    for (let i = 0; i < pool.freeBlocks.length; i++) {
      const block = pool.freeBlocks[i];
      if (block.size >= size) {
        // Remove from free blocks or split if larger
        if (block.size > size + 64) { // Worth splitting
          const remaining = {
            offset: block.offset + size,
            size: block.size - size,
          };
          pool.freeBlocks[i] = remaining;
        } else {
          pool.freeBlocks.splice(i, 1);
        }

        return {
          offset: block.offset,
          size: block.size,
        };
      }
    }
    return null;
  }

  /**
   * Deallocate memory and add to free blocks
   */
  deallocate(allocationId) {
    const allocation = this.allocations.get(allocationId);
    if (!allocation) {
      console.warn(`⚠️ Allocation ${allocationId} not found`);
      return false;
    }

    const pool = this.pools.get(allocation.moduleId);
    if (!pool) {
      console.warn(`⚠️ Pool for ${allocation.moduleId} not found`);
      return false;
    }

    // Add to free blocks
    pool.freeBlocks.push({
      offset: allocation.offset,
      size: allocation.size,
    });

    // Merge adjacent free blocks
    this.mergeFreeBlocks(pool);

    // Remove from allocations
    pool.allocations.delete(allocationId);
    this.allocations.delete(allocationId);
    this.totalAllocated -= allocation.size;

    console.log(`🗑️ Deallocated ${allocation.size} bytes for ${allocation.moduleId}`);
    return true;
  }

  /**
   * Merge adjacent free blocks to reduce fragmentation
   */
  mergeFreeBlocks(pool) {
    pool.freeBlocks.sort((a, b) => a.offset - b.offset);

    for (let i = 0; i < pool.freeBlocks.length - 1; i++) {
      const current = pool.freeBlocks[i];
      const next = pool.freeBlocks[i + 1];

      if (current.offset + current.size === next.offset) {
        // Merge blocks
        current.size += next.size;
        pool.freeBlocks.splice(i + 1, 1);
        i--; // Check again with merged block
      }
    }
  }

  /**
   * Garbage collect unused allocations
   */
  garbageCollect(moduleId) {
    const pool = this.pools.get(moduleId);
    if (!pool) {
      return;
    }

    const now = Date.now();
    const maxAge = 300000; // 5 minutes
    const freedAllocations = [];

    for (const [id, allocation] of pool.allocations) {
      if (now - allocation.timestamp > maxAge) {
        freedAllocations.push(id);
      }
    }

    for (const id of freedAllocations) {
      this.deallocate(id);
    }

    console.log(`🧹 GC for ${moduleId}: freed ${freedAllocations.length} allocations`);
  }

  /**
   * Garbage collect all pools
   */
  garbageCollectAll() {
    for (const moduleId of this.pools.keys()) {
      this.garbageCollect(moduleId);
    }
  }

  /**
   * Get memory utilization ratio
   */
  getMemoryUtilization() {
    return this.totalAllocated / this.maxMemory;
  }

  /**
   * Get detailed memory statistics
   */
  getMemoryStats() {
    const poolStats = {};

    for (const [moduleId, pool] of this.pools) {
      poolStats[moduleId] = {
        allocated: pool.allocated,
        bufferSize: pool.memory.buffer.byteLength,
        freeBlocks: pool.freeBlocks.length,
        activeAllocations: pool.allocations.size,
        utilization: pool.allocated / pool.memory.buffer.byteLength,
      };
    }

    return {
      totalAllocated: this.totalAllocated,
      maxMemory: this.maxMemory,
      globalUtilization: this.getMemoryUtilization(),
      pools: poolStats,
      allocationCount: this.allocationCounter,
    };
  }

  /**
   * Optimize memory layout by compacting allocations
   */
  compactMemory(moduleId) {
    const pool = this.pools.get(moduleId);
    if (!pool) {
      return;
    }

    // Sort allocations by offset
    const allocations = Array.from(pool.allocations.values())
      .sort((a, b) => a.offset - b.offset);

    let newOffset = 0;
    const moves = [];

    for (const allocation of allocations) {
      if (allocation.offset !== newOffset) {
        moves.push({
          from: allocation.offset,
          to: newOffset,
          size: allocation.size,
        });
        allocation.offset = newOffset;
      }
      newOffset += allocation.size;
    }

    // Perform memory moves
    const buffer = new Uint8Array(pool.memory.buffer);
    for (const move of moves) {
      const src = buffer.subarray(move.from, move.from + move.size);
      buffer.set(src, move.to);
    }

    // Update pool state
    pool.allocated = newOffset;
    pool.freeBlocks = newOffset < pool.memory.buffer.byteLength ?
      [{ offset: newOffset, size: pool.memory.buffer.byteLength - newOffset }] : [];

    console.log(`🗜️ Compacted ${moduleId}: ${moves.length} moves, freed ${pool.memory.buffer.byteLength - newOffset} bytes`);
  }
}

/**
 * Progressive WASM Module Loader with Memory Optimization
 */
class ProgressiveWasmLoader {
  constructor() {
    this.memoryPool = new WasmMemoryPool();
    this.loadedModules = new Map();
    this.loadingQueues = new Map();
    this.priorityLevels = {
      'critical': 1,
      'high': 2,
      'medium': 3,
      'low': 4,
    };
    this.loadingStrategies = {
      'eager': this.loadAllModules.bind(this),
      'lazy': this.loadOnDemand.bind(this),
      'progressive': this.loadProgressively.bind(this),
    };
  }

  /**
   * Register module for progressive loading
   */
  registerModule(config) {
    const {
      id,
      url,
      size,
      priority = 'medium',
      dependencies = [],
      features = [],
      preload = false,
    } = config;

    const module = {
      id,
      url,
      size,
      priority,
      dependencies,
      features,
      preload,
      loaded: false,
      loading: false,
      instance: null,
      memoryAllocations: new Set(),
    };

    this.loadedModules.set(id, module);

    if (preload) {
      this.queueLoad(id, 'critical');
    }

    console.log(`📋 Registered WASM module: ${id} (${size / 1024}KB, ${priority} priority)`);
  }

  /**
   * Queue module for loading with priority
   */
  queueLoad(moduleId, priority = 'medium') {
    if (!this.loadingQueues.has(priority)) {
      this.loadingQueues.set(priority, []);
    }

    const queue = this.loadingQueues.get(priority);
    if (!queue.includes(moduleId)) {
      queue.push(moduleId);
      this.processLoadingQueue();
    }
  }

  /**
   * Process loading queue by priority
   */
  async processLoadingQueue() {
    for (const priority of Object.keys(this.priorityLevels).sort((a, b) =>
      this.priorityLevels[a] - this.priorityLevels[b])) {

      const queue = this.loadingQueues.get(priority);
      if (!queue || queue.length === 0) {
        continue;
      }

      const moduleId = queue.shift();
      await this.loadModule(moduleId);
    }
  }

  /**
   * Load individual module with memory optimization
   */
  async loadModule(moduleId) {
    const module = this.loadedModules.get(moduleId);
    if (!module) {
      throw new Error(`Module ${moduleId} not registered`);
    }

    if (module.loaded) {
      return module.instance;
    }

    if (module.loading) {
      // Wait for existing load
      while (module.loading) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      return module.instance;
    }

    module.loading = true;

    try {
      console.log(`📦 Loading WASM module: ${moduleId}`);

      // Load dependencies first
      for (const depId of module.dependencies) {
        await this.loadModule(depId);
      }

      // Fetch WASM bytes
      const response = await fetch(module.url);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${module.url}: ${response.status}`);
      }

      const wasmBytes = await response.arrayBuffer();

      // Allocate memory for module
      const memoryAllocation = this.memoryPool.allocate(
        moduleId,
        module.size || wasmBytes.byteLength * 2,
      );

      module.memoryAllocations.add(memoryAllocation.id);

      // Create imports with optimized memory
      const imports = this.createModuleImports(moduleId, memoryAllocation);

      // Compile and instantiate
      const startTime = performance.now();
      const wasmModule = await WebAssembly.compile(wasmBytes);
      const instance = await WebAssembly.instantiate(wasmModule, imports);
      const loadTime = performance.now() - startTime;

      module.instance = {
        module: wasmModule,
        instance,
        exports: instance.exports,
        memory: memoryAllocation,
        loadTime,
      };

      module.loaded = true;
      module.loading = false;

      console.log(`✅ Loaded ${moduleId} in ${loadTime.toFixed(2)}ms`);

      // Optimize memory after loading
      this.optimizeModuleMemory(moduleId);

      return module.instance;

    } catch (error) {
      module.loading = false;
      console.error(`❌ Failed to load ${moduleId}:`, error);
      throw error;
    }
  }

  /**
   * Create optimized imports for module
   */
  createModuleImports(moduleId, memoryAllocation) {
    const pool = this.memoryPool.getPool(moduleId);

    return {
      env: {
        memory: pool.memory,

        // Optimized memory allocation functions
        malloc: (size) => {
          const allocation = this.memoryPool.allocate(moduleId, size);
          return allocation.offset;
        },

        free: (ptr) => {
          // Find allocation by offset and free it
          for (const allocation of this.memoryPool.allocations.values()) {
            if (allocation.moduleId === moduleId && allocation.offset === ptr) {
              this.memoryPool.deallocate(allocation.id);
              break;
            }
          }
        },

        // SIMD-optimized math functions
        simd_add_f32x4: (a, b, result) => {
          // This would call the SIMD implementation
          console.log('SIMD add called');
        },

        // Performance monitoring
        performance_mark: (name) => {
          performance.mark(`${moduleId}_${name}`);
        },
      },

      // WASI support for file operations
      wasi_snapshot_preview1: {
        proc_exit: (code) => {
          console.log(`Module ${moduleId} exited with code ${code}`);
        },
        fd_write: () => 0,
      },
    };
  }

  /**
   * Optimize module memory after loading
   */
  optimizeModuleMemory(moduleId) {
    setTimeout(() => {
      this.memoryPool.compactMemory(moduleId);
    }, 1000); // Delay to allow initial operations
  }

  /**
   * Progressive loading strategy
   */
  async loadProgressively() {
    // Load critical modules first
    const criticalModules = Array.from(this.loadedModules.values())
      .filter(m => m.priority === 'critical' || m.preload)
      .sort((a, b) => this.priorityLevels[a.priority] - this.priorityLevels[b.priority]);

    for (const module of criticalModules) {
      await this.loadModule(module.id);
    }

    // Load remaining modules in background
    const remainingModules = Array.from(this.loadedModules.values())
      .filter(m => !m.loaded && !m.loading)
      .sort((a, b) => this.priorityLevels[a.priority] - this.priorityLevels[b.priority]);

    // Load with delay to prevent blocking
    let delay = 0;
    for (const module of remainingModules) {
      setTimeout(() => this.loadModule(module.id), delay);
      delay += 100; // 100ms between loads
    }
  }

  /**
   * Eager loading strategy
   */
  async loadAllModules() {
    const modules = Array.from(this.loadedModules.values())
      .sort((a, b) => this.priorityLevels[a.priority] - this.priorityLevels[b.priority]);

    await Promise.all(modules.map(m => this.loadModule(m.id)));
  }

  /**
   * Lazy loading strategy
   */
  async loadOnDemand(moduleId) {
    return this.loadModule(moduleId);
  }

  /**
   * Get module by ID
   */
  getModule(moduleId) {
    const module = this.loadedModules.get(moduleId);
    return module?.instance || null;
  }

  /**
   * Unload module and free memory
   */
  unloadModule(moduleId) {
    const module = this.loadedModules.get(moduleId);
    if (!module || !module.loaded) {
      return false;
    }

    // Free all memory allocations
    for (const allocationId of module.memoryAllocations) {
      this.memoryPool.deallocate(allocationId);
    }

    module.memoryAllocations.clear();
    module.instance = null;
    module.loaded = false;

    console.log(`🗑️ Unloaded module: ${moduleId}`);
    return true;
  }

  /**
   * Get comprehensive loader statistics
   */
  getLoaderStats() {
    const modules = Array.from(this.loadedModules.values());
    const loaded = modules.filter(m => m.loaded);
    const loading = modules.filter(m => m.loading);

    return {
      totalModules: modules.length,
      loadedModules: loaded.length,
      loadingModules: loading.length,
      memoryStats: this.memoryPool.getMemoryStats(),
      loadTimes: loaded.map(m => ({
        id: m.id,
        loadTime: m.instance?.loadTime || 0,
      })),
      averageLoadTime: loaded.reduce((acc, m) => acc + (m.instance?.loadTime || 0), 0) / loaded.length,
    };
  }

  /**
   * Optimize all memory pools
   */
  optimizeMemory() {
    this.memoryPool.garbageCollectAll();

    for (const moduleId of this.loadedModules.keys()) {
      if (this.loadedModules.get(moduleId).loaded) {
        this.memoryPool.compactMemory(moduleId);
      }
    }

    console.log('🧹 Memory optimization completed');
  }
}

/**
 * WASM Browser Compatibility Manager
 */
class WasmCompatibilityManager {
  constructor() {
    this.capabilities = null;
    this.fallbacks = new Map();
  }

  /**
   * Detect browser WASM capabilities
   */
  async detectCapabilities() {
    const capabilities = {
      webassembly: typeof WebAssembly !== 'undefined',
      simd: false,
      threads: false,
      exceptions: false,
      memory64: false,
      streaming: false,
    };

    if (!capabilities.webassembly) {
      this.capabilities = capabilities;
      return capabilities;
    }

    // Test SIMD support
    try {
      const simdTest = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, // WASM magic
        0x01, 0x00, 0x00, 0x00, // version
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // type section
        0x03, 0x02, 0x01, 0x00, // function section
        0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x0c, 0x00, 0x0b, // code section with SIMD
      ]);

      await WebAssembly.compile(simdTest);
      capabilities.simd = true;
    } catch (e) {
      capabilities.simd = false;
    }

    // Test streaming compilation
    capabilities.streaming = typeof WebAssembly.compileStreaming === 'function';

    // Test SharedArrayBuffer for threads
    capabilities.threads = typeof SharedArrayBuffer !== 'undefined';

    this.capabilities = capabilities;
    console.log('🔍 WASM capabilities detected:', capabilities);

    return capabilities;
  }

  /**
   * Get capabilities (detect if not already done)
   */
  async getCapabilities() {
    if (!this.capabilities) {
      await this.detectCapabilities();
    }
    return this.capabilities;
  }

  /**
   * Register fallback for feature
   */
  registerFallback(feature, fallbackFn) {
    this.fallbacks.set(feature, fallbackFn);
  }

  /**
   * Check if feature is supported with fallback
   */
  async isSupported(feature) {
    const capabilities = await this.getCapabilities();

    if (capabilities[feature]) {
      return true;
    }

    if (this.fallbacks.has(feature)) {
      console.log(`⚠️ Using fallback for ${feature}`);
      return 'fallback';
    }

    return false;
  }

  /**
   * Load module with compatibility checks
   */
  async loadCompatibleModule(url, features = []) {
    const capabilities = await this.getCapabilities();

    if (!capabilities.webassembly) {
      throw new Error('WebAssembly not supported in this browser');
    }

    // Check required features
    const unsupported = [];
    for (const feature of features) {
      const support = await this.isSupported(feature);
      if (!support) {
        unsupported.push(feature);
      }
    }

    if (unsupported.length > 0) {
      console.warn(`⚠️ Unsupported features: ${unsupported.join(', ')}`);
      // Could load alternative module or disable features
    }

    // Load with appropriate method
    if (capabilities.streaming) {
      return WebAssembly.compileStreaming(fetch(url));
    }
    const response = await fetch(url);
    const bytes = await response.arrayBuffer();
    return WebAssembly.compile(bytes);

  }
}

export {
  WasmMemoryPool,
  ProgressiveWasmLoader,
  WasmCompatibilityManager,
};

export default {
  WasmMemoryPool,
  ProgressiveWasmLoader,
  WasmCompatibilityManager,
};
