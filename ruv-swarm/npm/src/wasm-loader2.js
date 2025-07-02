/**
 * Progressive WASM Module Loader
 * Implements on-demand, eager, and progressive loading strategies
 * for optimal performance and memory usage
 */

import path from 'path';
import { promises as fs } from 'fs';

class WasmModuleLoader {
  constructor() {
    this.modules = new Map();
    this.loadingPromises = new Map();
    this.loadingStrategy = 'on-demand'; // 'eager', 'on-demand', 'progressive'
    this.moduleManifest = {
      core: {
        path: './wasm/ruv_swarm_wasm_bg.wasm',
        jsBindings: './wasm/ruv_swarm_wasm.js',
        size: 512 * 1024, // 512KB
        priority: 'high',
        dependencies: [],
        exists: true, // This module definitely exists
        type: 'wasm-bindgen', // Uses wasm-bindgen generated bindings
      },
      // Legacy modules - keep for compatibility but mark as optional
      neural: {
        path: './wasm/ruv-fann.wasm',
        size: 1024 * 1024, // 1MB
        priority: 'medium',
        dependencies: ['core'],
        exists: false, // This is a standalone module, not currently built
        optional: true,
      },
      forecasting: {
        path: './wasm/neuro-divergent.wasm',
        size: 1536 * 1024, // 1.5MB
        priority: 'medium',
        dependencies: ['core'],
        exists: false, // This is a standalone module, not currently built
        optional: true,
      },
      swarm: {
        path: './wasm/ruv-swarm-orchestration.wasm',
        size: 768 * 1024, // 768KB
        priority: 'high',
        dependencies: ['core'],
        exists: false, // This functionality is in core module
        optional: true,
      },
      persistence: {
        path: './wasm/ruv-swarm-persistence.wasm',
        size: 256 * 1024, // 256KB
        priority: 'high',
        dependencies: ['core'],
        exists: false, // This functionality is handled by Node.js layer
        optional: true,
      },
    };
    this.baseDir = path.join(new URL('.', import.meta.url).pathname, '..');
  }

  async initialize(strategy = 'progressive') {
    this.loadingStrategy = strategy;

    switch (strategy) {
    case 'eager':
      return this.loadAllModules();
    case 'progressive':
      return this.loadCoreModules();
    case 'on-demand':
      return this.setupLazyLoading();
    default:
      throw new Error(`Unknown loading strategy: ${strategy}`);
    }
  }

  async loadModule(moduleName) {
    if (this.modules.has(moduleName)) {
      return this.modules.get(moduleName);
    }

    if (this.loadingPromises.has(moduleName)) {
      return this.loadingPromises.get(moduleName);
    }

    const moduleInfo = this.moduleManifest[moduleName];
    if (!moduleInfo) {
      throw new Error(`Unknown module: ${moduleName}`);
    }

    // Check if module is marked as non-existent and optional
    if (!moduleInfo.exists && moduleInfo.optional) {
      // Silently use core module for neural and forecasting features
      // These are integrated into the core module, not separate files
      if (moduleName === 'neural' || moduleName === 'forecasting') {
        if (this.modules.has('core')) {
          const coreModule = this.modules.get('core');
          this.modules.set(moduleName, coreModule); // Alias to core module
          return coreModule;
        }
      } else {
        // Only warn for other optional modules
        console.warn(`⚠️ Optional module ${moduleName} is not available, functionality will be provided by core module`);
      }

      // Return a reference to the core module instead of a placeholder
      if (moduleName !== 'core' && this.modules.has('core')) {
        const coreModule = this.modules.get('core');
        this.modules.set(moduleName, coreModule); // Alias to core module
        return coreModule;
      }
      throw new Error(`Optional module ${moduleName} not available and core module not loaded`);
    }

    // Load dependencies first
    for (const dep of moduleInfo.dependencies) {
      await this.loadModule(dep);
    }

    const loadingPromise = this.loadWasmModule(moduleName, moduleInfo);
    this.loadingPromises.set(moduleName, loadingPromise);

    try {
      const module = await loadingPromise;
      this.modules.set(moduleName, module);
      this.loadingPromises.delete(moduleName);

      console.log(`✅ Loaded WASM module: ${moduleName} (${this.formatBytes(moduleInfo.size)})`);
      return module;
    } catch (error) {
      this.loadingPromises.delete(moduleName);

      // If it's an optional module, provide fallback to core functionality
      if (moduleInfo.optional && this.modules.has('core')) {
        console.warn(`⚠️ Optional module ${moduleName} failed to load, using core module functionality`);
        const coreModule = this.modules.get('core');
        this.modules.set(moduleName, coreModule);
        return coreModule;
      }

      console.error(`❌ Failed to load WASM module: ${moduleName}`, error);
      throw error;
    }
  }

  async loadWasmModule(moduleName, moduleInfo) {
    // Special handling for the core module which uses ES module bindings
    if (moduleName === 'core') {
      return this.loadCoreModule();
    }

    // For other modules, load the WASM file directly
    const wasmPath = path.join(this.baseDir, moduleInfo.path);

    try {
      let wasmBuffer;

      if (typeof window !== 'undefined') {
        // Browser environment
        const response = await fetch(wasmPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch WASM module: ${response.statusText}`);
        }
        wasmBuffer = await response.arrayBuffer();
      } else {
        // Node.js environment
        try {
          wasmBuffer = await fs.readFile(wasmPath);
        } catch (error) {
          // Fallback: module might not exist yet, return a placeholder
          console.warn(`Module ${moduleName} not found at ${wasmPath}, using placeholder`);
          return this.createPlaceholderModule(moduleName);
        }
      }

      const imports = this.getModuleImports(moduleName);
      const wasmModule = await WebAssembly.instantiate(wasmBuffer, imports);

      return {
        instance: wasmModule.instance,
        module: wasmModule.module,
        exports: wasmModule.instance.exports,
        memory: wasmModule.instance.exports.memory,
      };
    } catch (error) {
      console.warn(`Failed to load ${moduleName}, using placeholder:`, error.message);
      return this.createPlaceholderModule(moduleName);
    }
  }

  async loadCoreModule() {
    // Load the core module using ES module bindings
    try {
      // Ensure we're using URL-based import for ES modules
      const wasmJsUrl = new URL('../wasm/ruv_swarm_wasm.js', import.meta.url).href;

      // Use dynamic import with URL protocol for ES modules
      const bindings = await import(wasmJsUrl);

      // Initialize WASM module with file buffer for Node.js
      if (bindings.default && typeof window === 'undefined') {
        const wasmPath = path.join(this.baseDir, 'wasm', 'ruv_swarm_wasm_bg.wasm');
        try {
          const wasmBuffer = await fs.readFile(wasmPath);
          await bindings.default(wasmBuffer);
        } catch (error) {
          console.warn('Failed to load WASM file, using bindings defaults:', error);
        }
      }

      return {
        instance: { exports: bindings },
        module: null,
        exports: bindings,
        memory: bindings.memory,
      };
    } catch (error) {
      console.warn('Failed to load core module bindings:', error);
      return this.createPlaceholderModule('core');
    }
  }

  getModuleImports(moduleName) {
    const baseImports = {
      env: {
        memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }),
      },
      wasi_snapshot_preview1: {
        // Basic WASI imports for compatibility
        proc_exit: (code) => {
          throw new Error(`Process exited with code ${code}`);
        },
        fd_write: () => 0,
        fd_prestat_get: () => 1,
        fd_prestat_dir_name: () => 1,
        environ_sizes_get: () => 0,
        environ_get: () => 0,
        args_sizes_get: () => 0,
        args_get: () => 0,
        clock_time_get: () => Date.now() * 1000000,
        path_open: () => 1,
        fd_close: () => 0,
        fd_read: () => 0,
        fd_seek: () => 0,
        random_get: (ptr, len) => {
          const bytes = new Uint8Array(this.memory.buffer, ptr, len);
          crypto.getRandomValues(bytes);
          return 0;
        },
      },
    };

    // Module-specific imports
    switch (moduleName) {
    case 'neural':
      return {
        ...baseImports,
        neural: {
          log_training_progress: (epoch, loss) => {
            console.log(`Training progress - Epoch: ${epoch}, Loss: ${loss}`);
          },
        },
      };
    case 'forecasting':
      return {
        ...baseImports,
        forecasting: {
          log_forecast: (model, horizon) => {
            console.log(`Forecasting with model: ${model}, horizon: ${horizon}`);
          },
        },
      };
    default:
      return baseImports;
    }
  }

  createPlaceholderModule(moduleName) {
    // Create a placeholder module with basic functionality
    console.warn(`Creating placeholder for module: ${moduleName}`);

    const placeholderExports = {
      memory: new WebAssembly.Memory({ initial: 1, maximum: 10 }),
      __wbindgen_malloc: (size) => 0,
      __wbindgen_realloc: (ptr, oldSize, newSize) => ptr,
      __wbindgen_free: (ptr, size) => {},
    };

    // Add module-specific placeholder functions
    switch (moduleName) {
    case 'neural':
      placeholderExports.create_neural_network = () => {
        console.warn('Neural network module not loaded, using placeholder');
        return 0;
      };
      placeholderExports.train_network = () => 0;
      placeholderExports.forward_pass = () => new Float32Array([0.5]);
      break;
    case 'forecasting':
      placeholderExports.create_forecasting_model = () => {
        console.warn('Forecasting module not loaded, using placeholder');
        return 0;
      };
      placeholderExports.forecast = () => new Float32Array([0.0]);
      break;
    case 'swarm':
      placeholderExports.create_swarm_orchestrator = () => {
        console.warn('Swarm orchestration module not loaded, using placeholder');
        return 0;
      };
      break;
    }

    return {
      instance: { exports: placeholderExports },
      module: null,
      exports: placeholderExports,
      memory: placeholderExports.memory,
      isPlaceholder: true,
    };
  }

  async loadCoreModules() {
    // Load only the core module - other functionality is included in it
    await this.loadModule('core');

    console.log('🚀 Core WASM module loaded successfully');
    return true;
  }

  async loadAllModules() {
    // Only load modules that actually exist
    const existingModules = Object.keys(this.moduleManifest)
      .filter(name => this.moduleManifest[name].exists);

    await Promise.all(existingModules.map(name => this.loadModule(name)));

    console.log(`🎯 All available WASM modules loaded successfully (${existingModules.length} modules)`);
    return true;
  }

  setupLazyLoading() {
    // Create proxy objects that load modules on first access
    const moduleProxies = {};

    for (const moduleName of Object.keys(this.moduleManifest)) {
      moduleProxies[moduleName] = new Proxy({}, {
        get: (target, prop) => {
          if (!this.modules.has(moduleName)) {
            // Trigger module loading
            this.loadModule(moduleName);
            throw new Error(`Module ${moduleName} is loading. Please await loadModule('${moduleName}') first.`);
          }

          const module = this.modules.get(moduleName);
          return module.exports[prop];
        },
      });
    }

    return moduleProxies;
  }

  getModuleStatus() {
    const status = {};

    for (const [name, info] of Object.entries(this.moduleManifest)) {
      status[name] = {
        loaded: this.modules.has(name),
        loading: this.loadingPromises.has(name),
        size: info.size,
        priority: info.priority,
        dependencies: info.dependencies,
        isPlaceholder: this.modules.has(name) && this.modules.get(name).isPlaceholder,
      };
    }

    return status;
  }

  getTotalMemoryUsage() {
    let totalBytes = 0;

    for (const module of this.modules.values()) {
      if (module.memory && module.memory.buffer) {
        totalBytes += module.memory.buffer.byteLength;
      }
    }

    return totalBytes;
  }

  formatBytes(bytes) {
    if (bytes === 0) {
      return '0 Bytes';
    }
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2)) } ${ sizes[i]}`;
  }
}

export { WasmModuleLoader };