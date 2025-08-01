/* wasm-module-loader.js
 * Universal (ESM + CJS) progressive WASM loader.
 * Author: Bron refactor 2025‑07‑01
 */

import path from 'node:path';
import { promises as fs, existsSync, accessSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { webcrypto as nodeCrypto } from 'node:crypto';

// ────────────────────────────────────────────────────────────────────────────────
// helpers ────────────────────────────────────────────────────────────────────────
const crypto = globalThis.crypto ?? nodeCrypto; // browser | Node
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ────────────────────────────────────────────────────────────────────────────────
class WasmModuleLoader {
  constructor() {
    this.modules = new Map();
    this.loadingPromises = new Map();
    this.loadingStrategy = 'on-demand'; // eager | on-demand | progressive
    this.baseDir = __dirname; // one-liner dirname
    this.wasmCache = new Map(); // Cache compiled WASM modules
    this.cacheTimeout = 3600000; // 1 hour cache timeout
    this.moduleManifest = {
      /* The only compiled artefact today. Others are historical → optional */
      core: {
        path: '../wasm/ruv_swarm_wasm_bg.wasm',
        jsBindings: '../wasm/ruv_swarm_wasm.js',
        size: 512 * 1024,
        priority: 'high',
        dependencies: [],
        exists: true,
        type: 'wasm-bindgen',
      },

      /* legacy / optional stubs */
      neural: { path: '../wasm/ruv-fann.wasm', size: 1024 * 1024, priority: 'medium', dependencies: ['core'], exists: false, optional: true },
      forecasting: { path: '../wasm/neuro-divergent.wasm', size: 1536 * 1024, priority: 'medium', dependencies: ['core'], exists: false, optional: true },
      swarm: { path: '../wasm/ruv-swarm-orchestration.wasm', size: 768 * 1024, priority: 'high', dependencies: ['core'], exists: false, optional: true },
      persistence: { path: '../wasm/ruv-swarm-persistence.wasm', size: 256 * 1024, priority: 'high', dependencies: ['core'], exists: false, optional: true },
    };
  }

  /* ── public API ───────────────────────────────────────────────────────────── */
  async initialize(strategy = 'progressive') {
    this.loadingStrategy = strategy;
    if (strategy === 'eager') {
      return this.#loadAllModules();
    }
    if (strategy === 'progressive') {
      return this.#loadCoreOnly();
    }
    if (strategy === 'on-demand') {
      return this.#setupLazyProxies();
    }
    throw new Error(`Unknown loading strategy: ${strategy}`);
  }

  async loadModule(name) {
    if (this.modules.has(name)) {
      return this.modules.get(name);
    }
    if (this.loadingPromises.has(name)) {
      return this.loadingPromises.get(name);
    }

    const info = this.moduleManifest[name];
    if (!info) {
      throw new Error(`Unknown module: ${name}`);
    }

    /* optional module not built – silently alias to core when possible */
    if (!info.exists && info.optional) {
      await this.loadModule('core');
      const coreMod = this.modules.get('core');
      this.modules.set(name, coreMod);
      return coreMod;
    }

    /* ensure deps first */
    await Promise.all(info.dependencies.map(dep => this.loadModule(dep)));

    const p = (name === 'core')
      ? this.#loadCoreBindings()
      : this.#instantiateRaw(name, info);

    this.loadingPromises.set(name, p);
    try {
      const m = await p;
      this.modules.set(name, m);
      console.error(`✅  Loaded WASM module: ${name} (${this.#fmt(info.size)})`);
      return m;
    } finally {
      this.loadingPromises.delete(name);
    }
  }

  getModuleStatus() {
    const s = {};
    for (const [n, i] of Object.entries(this.moduleManifest)) {
      s[n] = {
        loaded: this.modules.has(n),
        loading: this.loadingPromises.has(n),
        placeholder: this.modules.get(n)?.isPlaceholder ?? false,
        size: i.size,
        priority: i.priority,
        deps: i.dependencies,
      };
    }
    return s;
  }

  /* ── internal ─────────────────────────────────────────────────────────────── */
  async #instantiateRaw(name, info) {
    const wasmPath = path.join(this.baseDir, info.path);
    const cacheKey = `${name}-${info.path}`;

    // Check cache first
    const cached = this.wasmCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp < this.cacheTimeout)) {
      console.error(`✨ Using cached WASM module: ${name}`);
      return cached.module;
    }

    let buffer;
    if (typeof window !== 'undefined') { // browser
      const resp = await fetch(wasmPath);
      if (!resp.ok) {
        throw new Error(`fetch failed: ${resp.statusText}`);
      }
      buffer = await resp.arrayBuffer();
    } else { // Node
      buffer = await fs.readFile(wasmPath).catch(() => null);
      if (!buffer) {
        return this.#placeholder(name);
      }
    }

    const imports = this.#importsFor(name);
    const { instance, module } = await WebAssembly.instantiate(buffer, imports);
    const result = { instance, module, exports: instance.exports, memory: instance.exports.memory };

    // Cache the compiled module
    this.wasmCache.set(cacheKey, {
      module: result,
      timestamp: Date.now(),
    });

    return result;
  }

  async #loadCoreBindings() {
    /* Enhanced WASM loader with context-aware path resolution */
    try {
      // Try multiple path resolution strategies
      const pathCandidates = this.#getWasmPathCandidates();

      for (const pathCandidate of pathCandidates) {
        try {
          const result = await this.#tryLoadFromPath(pathCandidate);
          if (result && !result.isPlaceholder) {
            console.error(`✅ Successfully loaded WASM from: ${pathCandidate.description}`);
            return result;
          }
        } catch (pathError) {
          console.debug(`❌ Failed to load from ${pathCandidate.description}:`, pathError.message);
          continue;
        }
      }

      throw new Error('All WASM loading strategies failed');
    } catch (error) {
      console.error('Failed to load core module via bindings loader:', error);
      console.warn('⚠️ Falling back to placeholder WASM functionality');

      // Log specific import errors for debugging
      if (error.message && error.message.includes('import')) {
        console.error('WASM import error details:', {
          message: error.message,
          stack: error.stack?.split('\n').slice(0, 5).join('\n'),
        });
      }

      return this.#placeholder('core');
    }
  }

  #importsFor(name) {
    const base = {
      env: { memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }) },
      wasi_snapshot_preview1: {
        proc_exit: c => {
          throw new Error(`WASI exit ${c}`);
        },
        fd_write: () => 0,
        /* …minimal stubs… */
        random_get: (ptr, len) => {
          const view = new Uint8Array(base.env.memory.buffer, ptr, len);
          crypto.getRandomValues(view);
          return 0;
        },
      },
    };

    if (name === 'neural') {
      base.neural = { log_training_progress: (e, l) => console.error(`Epoch ${e} → loss ${l}`) };
    } else if (name === 'forecasting') {
      base.forecasting = { log_forecast: (m, h) => console.error(`Forecast ${m}, horizon ${h}`) };
    }
    return base;
  }

  #placeholder(name) {
    console.warn(`⚠️  Using placeholder for missing module '${name}'`);
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 10 });
    return {
      instance: { exports: { memory: mem } },
      module: null,
      exports: { memory: mem },
      memory: mem,
      isPlaceholder: true,
    };
  }

  async #loadCoreOnly() {
    await this.loadModule('core');
  }
  async #loadAllModules() {
    return Promise.all(Object.keys(this.moduleManifest).map(n => this.loadModule(n)));
  }
  #setupLazyProxies() {
    const proxyBag = {};
    for (const n of Object.keys(this.moduleManifest)) {
      proxyBag[n] = new Proxy({}, {
        get: (_, p) => {
          if (!this.modules.has(n)) {
            throw new Error(
              `Module '${n}' not yet loaded; await loader.loadModule('${n}') first`,
            );
          }
          return this.modules.get(n).exports[p];
        },
      });
    }
    return proxyBag;
  }

  #resolvePackageWasmDir() {
    try {
      // Try different approaches to find the package root
      const approaches = [
        // Current working directory approach
        () => {
          const cwd = process.cwd();
          const potentialPaths = [
            path.join(cwd, 'node_modules', 'ruv-swarm', 'wasm'),
            path.join(cwd, '..', 'wasm'),
            path.join(cwd, 'wasm'),
          ];
          return potentialPaths.find(p => {
            try {
              accessSync(p);
              return true;
            } catch {
              return false;
            }
          });
        },

        // Module resolution approach
        () => {
          try {
            // For ES modules, use import.meta.resolve when available
            const moduleDir = path.dirname(path.dirname(__filename));
            return path.join(moduleDir, 'wasm');
          } catch {
            return null;
          }
        },

        // Environment variable approach
        () => process.env.RUV_SWARM_WASM_PATH,
      ];

      for (const approach of approaches) {
        try {
          const result = approach();
          if (result) {
            return result;
          }
        } catch {
          continue;
        }
      }

      return null;
    } catch {
      return null;
    }
  }

  #getWasmPathCandidates() {
    return [
      {
        description: 'Local development (relative to src/)',
        wasmDir: path.join(this.baseDir, '..', 'wasm'),
        loaderPath: path.join(this.baseDir, '..', 'wasm', 'wasm-bindings-loader.mjs'),
        wasmBinary: path.join(this.baseDir, '..', 'wasm', 'ruv_swarm_wasm_bg.wasm'),
        jsBindings: path.join(this.baseDir, '..', 'wasm', 'ruv_swarm_wasm.js'),
      },
      {
        description: 'NPM package installation (adjacent to src/)',
        wasmDir: path.join(this.baseDir, '..', '..', 'wasm'),
        loaderPath: path.join(this.baseDir, '..', '..', 'wasm', 'wasm-bindings-loader.mjs'),
        wasmBinary: path.join(this.baseDir, '..', '..', 'wasm', 'ruv_swarm_wasm_bg.wasm'),
        jsBindings: path.join(this.baseDir, '..', '..', 'wasm', 'ruv_swarm_wasm.js'),
      },
      {
        description: 'Global npm installation',
        wasmDir: this.#resolvePackageWasmDir(),
        get loaderPath() {
          return this.wasmDir ? path.join(this.wasmDir, 'wasm-bindings-loader.mjs') : null;
        },
        get wasmBinary() {
          return this.wasmDir ? path.join(this.wasmDir, 'ruv_swarm_wasm_bg.wasm') : null;
        },
        get jsBindings() {
          return this.wasmDir ? path.join(this.wasmDir, 'ruv_swarm_wasm.js') : null;
        },
      },
      {
        description: 'Bundled WASM (inline)',
        wasmDir: null,
        loaderPath: null,
        wasmBinary: null,
        jsBindings: null,
        inline: true,
      },
    ].filter(candidate => {
      if (candidate.inline) {
        return true;
      }
      try {
        // Check if the wasm directory exists using sync fs access
        accessSync(candidate.wasmDir);
        return true;
      } catch {
        return false;
      }
    });
  }

  async #tryLoadFromPath(pathCandidate) {
    if (pathCandidate.inline) {
      // Use inline/bundled WASM approach
      return this.#loadInlineWasm();
    }

    // Check if required files exist
    await fs.access(pathCandidate.wasmBinary);

    // Try to load the wasm-bindings-loader if it exists
    if (pathCandidate.loaderPath) {
      try {
        await fs.access(pathCandidate.loaderPath);

        const loaderURL = pathToFileURL(pathCandidate.loaderPath).href;
        const loaderModule = await import(loaderURL);
        const bindingsLoader = loaderModule.default;
        const initialized = await bindingsLoader.initialize();

        // Check if it's using placeholder
        if (initialized.isPlaceholder) {
          console.debug('Bindings loader returned placeholder');
          throw new Error('Bindings loader using placeholder');
        }

        this.loadedBindings = initialized;

        // Return a properly structured module object
        return {
          instance: { exports: initialized },
          module: initialized,
          exports: initialized,
          memory: initialized.memory || new WebAssembly.Memory({ initial: 256, maximum: 4096 }),
          getTotalMemoryUsage: initialized.getTotalMemoryUsage || (() => {
            if (initialized.memory && initialized.memory.buffer) {
              return initialized.memory.buffer.byteLength;
            }
            return 256 * 65536;
          }),
          isPlaceholder: false,
        };
      } catch (loaderError) {
        console.debug('Loader error:', loaderError.message);
        console.debug('Falling back to direct WASM loading');
      }
    }

    // Fallback to direct WASM loading
    return this.#loadDirectWasm(pathCandidate.wasmBinary);
  }

  async #loadInlineWasm() {
    // Placeholder for inline WASM - could be base64 encoded or bundled
    console.error('Using inline WASM placeholder');
    throw new Error('Inline WASM not implemented yet');
  }

  async #loadDirectWasm(wasmPath) {
    // For wasm-bindgen generated WASM, we need to use the JS bindings
    const jsBindingsPath = wasmPath.replace('_bg.wasm', '.js');

    try {
      // Check if the JS bindings file exists
      await fs.access(jsBindingsPath);

      // Import the wasm-bindgen generated module
      const bindingsURL = pathToFileURL(jsBindingsPath).href;
      const wasmModule = await import(bindingsURL);

      // Initialize the wasm module (wasm-bindgen handles the instantiation)
      if (typeof wasmModule.default === 'function') {
        // Call default export with the path to the WASM file
        const wasmUrl = pathToFileURL(wasmPath).href;
        await wasmModule.default(wasmUrl);
      } else if (typeof wasmModule.init === 'function') {
        // Some wasm-bindgen versions export an init function
        const wasmUrl = pathToFileURL(wasmPath).href;
        await wasmModule.init(wasmUrl);
      } else if (typeof wasmModule.initSync === 'function') {
        // Sync initialization variant
        const wasmBuffer = await fs.readFile(wasmPath);
        wasmModule.initSync(wasmBuffer);
      }

      // Store the loaded module
      this.loadedWasm = wasmModule;

      // Return a structure that matches our expected format
      return {
        instance: { exports: wasmModule },
        module: wasmModule,
        exports: wasmModule,
      };
    } catch (error) {
      console.error('Failed to load wasm-bindgen module:', error);
      // Fallback to direct WASM loading (won't work for wasm-bindgen but we try)
      const wasmBuffer = await fs.readFile(wasmPath);
      const imports = this.#importsFor('core');
      const { instance, module } = await WebAssembly.instantiate(wasmBuffer, imports);
      this.loadedWasm = { instance, module };
      return { instance, module, exports: instance.exports };
    }
  }

  #createBindingsApi() {
    // If we have loaded bindings, return them directly
    if (this.loadedBindings) {
      return this.loadedBindings;
    }

    // If we have the actual wasm-bindgen module loaded, use it directly
    if (this.loadedWasm && !this.loadedWasm.instance) {
      // This is a wasm-bindgen module, return it directly
      return this.loadedWasm;
    }

    // Create a minimal API that matches what the bindings would provide
    const api = {
      memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }),

      // Neural network functions
      create_neural_network: (layers, neurons_per_layer) => {
        console.error(`Creating neural network with ${layers} layers and ${neurons_per_layer} neurons per layer`);
        return 1;
      },

      train_network: (network_id, data, epochs) => {
        console.error(`Training network ${network_id} for ${epochs} epochs`);
        return true;
      },

      forward_pass: (network_id, input) => {
        console.error(`Forward pass on network ${network_id}`);
        return new Float32Array([0.5, 0.5, 0.5]);
      },

      // Forecasting functions
      create_forecasting_model: (type) => {
        console.error(`Creating forecasting model of type ${type}`);
        return 1;
      },

      forecast: (model_id, data, horizon) => {
        console.error(`Forecasting with model ${model_id} for horizon ${horizon}`);
        return new Float32Array([0.1, 0.2, 0.3]);
      },

      // Swarm functions
      create_swarm: (topology, max_agents) => {
        console.error(`Creating swarm with ${topology} topology and ${max_agents} max agents`);
        return 1;
      },

      spawn_agent: (swarm_id, agent_type) => {
        console.error(`Spawning ${agent_type} agent in swarm ${swarm_id}`);
        return 1;
      },

      // Memory management
      getTotalMemoryUsage: () => {
        return 256 * 65536; // 256 pages * 64KB
      },

      isPlaceholder: true,
    };

    // If we have actual WASM loaded, proxy to it
    if (this.loadedWasm && this.loadedWasm.exports) {
      const target = this.loadedWasm.exports;
      return new Proxy(api, {
        get(obj, prop) {
          if (target && typeof target[prop] !== 'undefined') {
            return target[prop];
          }
          return obj[prop];
        },
      });
    }

    return api;
  }

  #fmt(b) {
    if (!b) {
      return '0 B';
    }
    const k = 1024, i = Math.floor(Math.log(b) / Math.log(k));
    return `${(b / k ** i).toFixed(1)} ${['B', 'KB', 'MB', 'GB'][i]}`;
  }

  getTotalMemoryUsage() {
    let totalBytes = 0;

    for (const [name, module] of this.modules.entries()) {
      if (module && module.memory && module.memory.buffer) {
        totalBytes += module.memory.buffer.byteLength;
      }
    }

    return totalBytes;
  }

  clearCache() {
    const cacheSize = this.wasmCache.size;
    this.wasmCache.clear();
    console.error(`🧹 Cleared WASM cache (${cacheSize} modules`);
  }

  optimizeMemory() {
    // Clear expired cache entries
    const now = Date.now();
    let expired = 0;

    for (const [key, cached] of this.wasmCache.entries()) {
      if (now - cached.timestamp > this.cacheTimeout) {
        this.wasmCache.delete(key);
        expired++;
      }
    }

    if (expired > 0) {
      console.error(`🧹 Removed ${expired} expired WASM cache entries`);
    }

    // Force garbage collection if available
    if (global.gc) {
      global.gc();
      console.error('🧹 Triggered garbage collection');
    }

    return {
      cacheSize: this.wasmCache.size,
      memoryUsage: this.getTotalMemoryUsage(),
      expiredEntries: expired,
    };
  }
}

// ────────────────────────────────────────────────────────────────────────────────
export { WasmModuleLoader };
export default WasmModuleLoader; // ESM default

/* CJS interop --------------------------------------------------------------- */
if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
  module.exports = WasmModuleLoader; // require('./wasm-module-loader')
}
