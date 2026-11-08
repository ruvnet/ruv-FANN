/** Load generated wasm-bindgen code and its matching binary without substitutes. */
import { readFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

export class WasmBindingsLoader {
  constructor(directory = path.dirname(fileURLToPath(import.meta.url))) {
    this.directory = directory;
    this.initialized = false;
    this.isPlaceholder = false;
    this.exports = null;
    this.wasm = null;
    this.memory = null;
    this.loading = null;
  }

  async initialize() {
    if (this.initialized) return this;
    if (!this.loading) {
      this.loading = this.load();
    }
    try {
      return await this.loading;
    } finally {
      this.loading = null;
    }
  }

  async load() {
    const bytes = await readFile(path.join(this.directory, 'ruv_swarm_wasm_bg.wasm'));
    const bindings = await import(pathToFileURL(path.join(this.directory, 'ruv_swarm_wasm.js')).href);
    if (typeof bindings.default !== 'function') {
      throw new Error('Generated wasm-bindgen initialization entry point is missing');
    }
    const instance = await bindings.default({ module_or_path: bytes });
    if (!(instance.memory instanceof WebAssembly.Memory)) {
      throw new Error('Generated WASM module does not expose its actual memory');
    }
    this.exports = bindings;
    this.wasm = instance;
    this.memory = instance.memory;
    for (const [name, value] of Object.entries(bindings)) {
      if (name === 'default' || name.startsWith('__') || name in this) continue;
      this[name] = value;
    }
    this.initialized = true;
    return this;
  }

  getTotalMemoryUsage() {
    return this.memory?.buffer.byteLength ?? 0;
  }
}

export default new WasmBindingsLoader();
