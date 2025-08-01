let wasm;

const heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

function getObject(idx) { return heap[idx]; }

let heap_next = heap.length;

function addHeapObject(obj) {
  if (heap_next === heap.length) heap.push(heap.length + 1);
  const idx = heap_next;
  heap_next = heap[idx];

  heap[idx] = obj;
  return idx;
}

function handleError(f, args) {
  try {
    return f.apply(this, args);
  } catch (e) {
    wasm.__wbindgen_export_0(addHeapObject(e));
  }
}

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available'); } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
  if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
    cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
  }
  return cachedUint8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder('utf-8') : { encode: () => { throw Error('TextEncoder not available'); } } );

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
  ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
  }
  : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
      read: arg.length,
      written: buf.length,
    };
  });

function passStringToWasm0(arg, malloc, realloc) {

  if (realloc === undefined) {
    const buf = cachedTextEncoder.encode(arg);
    const ptr = malloc(buf.length, 1) >>> 0;
    getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
    WASM_VECTOR_LEN = buf.length;
    return ptr;
  }

  let len = arg.length;
  let ptr = malloc(len, 1) >>> 0;

  const mem = getUint8ArrayMemory0();

  let offset = 0;

  for (; offset < len; offset++) {
    const code = arg.charCodeAt(offset);
    if (code > 0x7F) break;
    mem[ptr + offset] = code;
  }

  if (offset !== len) {
    if (offset !== 0) {
      arg = arg.slice(offset);
    }
    ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
    const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
    const ret = encodeString(arg, view);

    offset += ret.written;
    ptr = realloc(ptr, len, offset, 1) >>> 0;
  }

  WASM_VECTOR_LEN = offset;
  return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
  if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
    cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
  }
  return cachedDataViewMemory0;
}

function isLikeNone(x) {
  return x === undefined || x === null;
}

function debugString(val) {
  // primitive types
  const type = typeof val;
  if (type == 'number' || type == 'boolean' || val == null) {
    return  `${val}`;
  }
  if (type == 'string') {
    return `"${val}"`;
  }
  if (type == 'symbol') {
    const description = val.description;
    if (description == null) {
      return 'Symbol';
    } else {
      return `Symbol(${description})`;
    }
  }
  if (type == 'function') {
    const name = val.name;
    if (typeof name == 'string' && name.length > 0) {
      return `Function(${name})`;
    } else {
      return 'Function';
    }
  }
  // objects
  if (Array.isArray(val)) {
    const length = val.length;
    let debug = '[';
    if (length > 0) {
      debug += debugString(val[0]);
    }
    for(let i = 1; i < length; i++) {
      debug += ', ' + debugString(val[i]);
    }
    debug += ']';
    return debug;
  }
  // Test for built-in
  const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
  let className;
  if (builtInMatches && builtInMatches.length > 1) {
    className = builtInMatches[1];
  } else {
    // Failed to match the standard '[object ClassName]'
    return toString.call(val);
  }
  if (className == 'Object') {
    // we're a user defined class or Object
    // JSON.stringify avoids problems with cycles, and is generally much
    // easier than looping through ownProperties of `val`.
    try {
      return 'Object(' + JSON.stringify(val) + ')';
    } catch (_) {
      return 'Object';
    }
  }
  // errors
  if (val instanceof Error) {
    return `${val.name}: ${val.message}\n${val.stack}`;
  }
  // TODO we could test for more things here, like `Set`s and `Map`s.
  return className;
}

function dropObject(idx) {
  if (idx < 132) return;
  heap[idx] = heap_next;
  heap_next = idx;
}

function takeObject(idx) {
  const ret = getObject(idx);
  dropObject(idx);
  return ret;
}

function getArrayU8FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function passArray8ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 1, 1) >>> 0;
  getUint8ArrayMemory0().set(arg, ptr / 1);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
  if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
    cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
  }
  return cachedFloat32ArrayMemory0;
}

function passArrayF32ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 4, 4) >>> 0;
  getFloat32ArrayMemory0().set(arg, ptr / 4);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}

function getArrayF32FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
/**
 * SIMD feature detection and runtime capabilities
 * @returns {string}
 */
export function detect_simd_capabilities() {
  let deferred1_0;
  let deferred1_1;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    wasm.detect_simd_capabilities(retptr);
    const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
    const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
    deferred1_0 = r0;
    deferred1_1 = r1;
    return getStringFromWasm0(r0, r1);
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
    wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
  }
}

/**
 * @returns {string}
 */
export function run_simd_verification_suite() {
  let deferred1_0;
  let deferred1_1;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    wasm.run_simd_verification_suite(retptr);
    const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
    const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
    deferred1_0 = r0;
    deferred1_1 = r1;
    return getStringFromWasm0(r0, r1);
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
    wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
  }
}

/**
 * @param {number} size
 * @param {number} iterations
 * @returns {string}
 */
export function simd_performance_report(size, iterations) {
  let deferred1_0;
  let deferred1_1;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    wasm.simd_performance_report(retptr, size, iterations);
    const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
    const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
    deferred1_0 = r0;
    deferred1_1 = r1;
    return getStringFromWasm0(r0, r1);
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
    wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
  }
}

/**
 * Comprehensive SIMD feature validation
 * @returns {boolean}
 */
export function validate_simd_implementation() {
  const ret = wasm.validate_simd_implementation();
  return ret !== 0;
}

let stack_pointer = 128;

function addBorrowedObject(obj) {
  if (stack_pointer == 1) throw new Error('out of js stack');
  heap[--stack_pointer] = obj;
  return stack_pointer;
}
/**
 * @param {any} array
 * @returns {Float32Array}
 */
export function js_array_to_vec_f32(array) {
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    wasm.js_array_to_vec_f32(retptr, addBorrowedObject(array));
    const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
    const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
    const r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
    const r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
    if (r3) {
      throw takeObject(r2);
    }
    const v1 = getArrayF32FromWasm0(r0, r1).slice();
    wasm.__wbindgen_export_1(r0, r1 * 4, 4);
    return v1;
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
    heap[stack_pointer++] = undefined;
  }
}

/**
 * @param {Float32Array} vec
 * @returns {Float32Array}
 */
export function vec_f32_to_js_array(vec) {
  const ptr0 = passArrayF32ToWasm0(vec, wasm.__wbindgen_export_2);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.vec_f32_to_js_array(ptr0, len0);
  return takeObject(ret);
}

/**
 * @returns {bigint}
 */
export function get_wasm_memory_usage() {
  const ret = wasm.get_wasm_memory_usage();
  return BigInt.asUintN(64, ret);
}

/**
 * @param {string} message
 */
export function console_log(message) {
  const ptr0 = passStringToWasm0(message, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
  const len0 = WASM_VECTOR_LEN;
  wasm.console_log(ptr0, len0);
}

/**
 * @param {string} message
 */
export function console_error(message) {
  const ptr0 = passStringToWasm0(message, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
  const len0 = WASM_VECTOR_LEN;
  wasm.console_error(ptr0, len0);
}

/**
 * @param {string} message
 */
export function console_warn(message) {
  const ptr0 = passStringToWasm0(message, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
  const len0 = WASM_VECTOR_LEN;
  wasm.console_warn(ptr0, len0);
}

/**
 * @param {any} error
 * @returns {string}
 */
export function format_js_error(error) {
  let deferred1_0;
  let deferred1_1;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    wasm.format_js_error(retptr, addHeapObject(error));
    const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
    const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
    deferred1_0 = r0;
    deferred1_1 = r1;
    return getStringFromWasm0(r0, r1);
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
    wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
  }
}

export function init() {
  wasm.init();
}

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
  if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
    cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
  }
  return cachedUint32ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 4, 4) >>> 0;
  getUint32ArrayMemory0().set(arg, ptr / 4);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
  if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
    cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
  }
  return cachedFloat64ArrayMemory0;
}

function passArrayF64ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 8, 8) >>> 0;
  getFloat64ArrayMemory0().set(arg, ptr / 8);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}

function getArrayF64FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayJsValueFromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  const mem = getDataViewMemory0();
  const result = [];
  for (let i = ptr; i < ptr + 4 * len; i += 4) {
    result.push(takeObject(mem.getUint32(i, true)));
  }
  return result;
}
/**
 * @param {Uint32Array} layers
 * @param {ActivationFunction} activation
 * @returns {WasmNeuralNetwork}
 */
export function create_neural_network(layers, activation) {
  const ptr0 = passArray32ToWasm0(layers, wasm.__wbindgen_export_2);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.create_neural_network(ptr0, len0, activation);
  return WasmNeuralNetwork.__wrap(ret);
}

/**
 * @param {string} topology
 * @returns {WasmSwarmOrchestrator}
 */
export function create_swarm_orchestrator(topology) {
  const ptr0 = passStringToWasm0(topology, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.create_swarm_orchestrator(ptr0, len0);
  return WasmSwarmOrchestrator.__wrap(ret);
}

/**
 * @param {string} model_type
 * @returns {WasmForecastingModel}
 */
export function create_forecasting_model(model_type) {
  const ptr0 = passStringToWasm0(model_type, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.create_forecasting_model(ptr0, len0);
  return WasmForecastingModel.__wrap(ret);
}

/**
 * @returns {string}
 */
export function get_version() {
  let deferred1_0;
  let deferred1_1;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    wasm.get_version(retptr);
    const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
    const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
    deferred1_0 = r0;
    deferred1_1 = r1;
    return getStringFromWasm0(r0, r1);
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
    wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
  }
}

/**
 * @returns {string}
 */
export function get_features() {
  let deferred1_0;
  let deferred1_1;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    wasm.get_features(retptr);
    const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
    const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
    deferred1_0 = r0;
    deferred1_1 = r1;
    return getStringFromWasm0(r0, r1);
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
    wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
  }
}

/**
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17}
 */
export const ActivationFunction = Object.freeze({
  Linear: 0, '0': 'Linear',
  Sigmoid: 1, '1': 'Sigmoid',
  SymmetricSigmoid: 2, '2': 'SymmetricSigmoid',
  Tanh: 3, '3': 'Tanh',
  ReLU: 4, '4': 'ReLU',
  LeakyReLU: 5, '5': 'LeakyReLU',
  Swish: 6, '6': 'Swish',
  Gaussian: 7, '7': 'Gaussian',
  Elliot: 8, '8': 'Elliot',
  SymmetricElliot: 9, '9': 'SymmetricElliot',
  Sine: 10, '10': 'Sine',
  Cosine: 11, '11': 'Cosine',
  SinSymmetric: 12, '12': 'SinSymmetric',
  CosSymmetric: 13, '13': 'CosSymmetric',
  ThresholdSymmetric: 14, '14': 'ThresholdSymmetric',
  Threshold: 15, '15': 'Threshold',
  StepSymmetric: 16, '16': 'StepSymmetric',
  Step: 17, '17': 'Step',
});

const AgentMemoryPoolFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_agentmemorypool_free(ptr >>> 0, 1));
/**
 * Agent memory pool specifically optimized for neural network agents
 */
export class AgentMemoryPool {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    AgentMemoryPoolFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_agentmemorypool_free(ptr, 0);
  }
  constructor() {
    const ret = wasm.agentmemorypool_new();
    this.__wbg_ptr = ret >>> 0;
    AgentMemoryPoolFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * Allocate memory for an agent based on complexity
     * @param {string} complexity
     * @returns {Uint8Array | undefined}
     */
  allocate_for_agent(complexity) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passStringToWasm0(complexity, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
      const len0 = WASM_VECTOR_LEN;
      wasm.agentmemorypool_allocate_for_agent(retptr, this.__wbg_ptr, ptr0, len0);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      let v2;
      if (r0 !== 0) {
        v2 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export_1(r0, r1 * 1, 1);
      }
      return v2;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * Return agent memory to the appropriate pool
     * @param {Uint8Array} memory
     */
  deallocate_agent_memory(memory) {
    const ptr0 = passArray8ToWasm0(memory, wasm.__wbindgen_export_2);
    const len0 = WASM_VECTOR_LEN;
    wasm.agentmemorypool_deallocate_agent_memory(this.__wbg_ptr, ptr0, len0);
  }
  /**
     * Get total memory usage across all pools
     * @returns {number}
     */
  total_memory_usage_mb() {
    const ret = wasm.agentmemorypool_total_memory_usage_mb(this.__wbg_ptr);
    return ret;
  }
  /**
     * Check if memory usage is within target (< 50MB for 10 agents)
     * @returns {boolean}
     */
  is_within_memory_target() {
    const ret = wasm.agentmemorypool_is_within_memory_target(this.__wbg_ptr);
    return ret !== 0;
  }
}

const MemoryPoolFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_memorypool_free(ptr >>> 0, 1));
/**
 * Memory pool for efficient memory management
 */
export class MemoryPool {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    MemoryPoolFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_memorypool_free(ptr, 0);
  }
  /**
     * Create a new memory pool with specified block size and maximum blocks
     * @param {number} block_size
     * @param {number} max_blocks
     */
  constructor(block_size, max_blocks) {
    const ret = wasm.memorypool_new(block_size, max_blocks);
    this.__wbg_ptr = ret >>> 0;
    MemoryPoolFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * Allocate a memory block from the pool
     * @returns {Uint8Array | undefined}
     */
  allocate() {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.memorypool_allocate(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      let v1;
      if (r0 !== 0) {
        v1 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export_1(r0, r1 * 1, 1);
      }
      return v1;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * Return a memory block to the pool for reuse
     * @param {Uint8Array} block
     */
  deallocate(block) {
    const ptr0 = passArray8ToWasm0(block, wasm.__wbindgen_export_2);
    const len0 = WASM_VECTOR_LEN;
    wasm.memorypool_deallocate(this.__wbg_ptr, ptr0, len0);
  }
  /**
     * Get the number of available blocks in the pool
     * @returns {number}
     */
  available_blocks() {
    const ret = wasm.memorypool_available_blocks(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * Get total memory usage in bytes
     * @returns {number}
     */
  memory_usage() {
    const ret = wasm.memorypool_memory_usage(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * Get pool efficiency metrics
     * @returns {PoolMetrics}
     */
  get_metrics() {
    const ret = wasm.memorypool_get_metrics(this.__wbg_ptr);
    return PoolMetrics.__wrap(ret);
  }
}

const OptimizedAgentFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_optimizedagent_free(ptr >>> 0, 1));

export class OptimizedAgent {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    OptimizedAgentFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_optimizedagent_free(ptr, 0);
  }
}

const OptimizedAgentSpawnerFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_optimizedagentspawner_free(ptr >>> 0, 1));

export class OptimizedAgentSpawner {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    OptimizedAgentSpawnerFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_optimizedagentspawner_free(ptr, 0);
  }
  constructor() {
    const ret = wasm.optimizedagentspawner_new();
    this.__wbg_ptr = ret >>> 0;
    OptimizedAgentSpawnerFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @param {string} agent_type
     * @param {string} complexity
     * @returns {string}
     */
  spawn_agent(agent_type, complexity) {
    let deferred4_0;
    let deferred4_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passStringToWasm0(agent_type, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
      const len0 = WASM_VECTOR_LEN;
      const ptr1 = passStringToWasm0(complexity, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
      const len1 = WASM_VECTOR_LEN;
      wasm.optimizedagentspawner_spawn_agent(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
      const r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
      let ptr3 = r0;
      let len3 = r1;
      if (r3) {
        ptr3 = 0; len3 = 0;
        throw takeObject(r2);
      }
      deferred4_0 = ptr3;
      deferred4_1 = len3;
      return getStringFromWasm0(ptr3, len3);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred4_0, deferred4_1, 1);
    }
  }
  /**
     * @param {string} agent_id
     */
  release_agent(agent_id) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passStringToWasm0(agent_id, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
      const len0 = WASM_VECTOR_LEN;
      wasm.optimizedagentspawner_release_agent(retptr, this.__wbg_ptr, ptr0, len0);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      if (r1) {
        throw takeObject(r0);
      }
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * @returns {string}
     */
  get_performance_report() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.optimizedagentspawner_get_performance_report(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @returns {number}
     */
  get_active_agent_count() {
    const ret = wasm.optimizedagentspawner_get_active_agent_count(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * @returns {boolean}
     */
  is_within_memory_target() {
    const ret = wasm.optimizedagentspawner_is_within_memory_target(this.__wbg_ptr);
    return ret !== 0;
  }
}

const PerformanceMonitorFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_performancemonitor_free(ptr >>> 0, 1));

export class PerformanceMonitor {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    PerformanceMonitorFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_performancemonitor_free(ptr, 0);
  }
  constructor() {
    const ret = wasm.performancemonitor_new();
    this.__wbg_ptr = ret >>> 0;
    PerformanceMonitorFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @param {number} time
     */
  record_load_time(time) {
    wasm.performancemonitor_record_load_time(this.__wbg_ptr, time);
  }
  /**
     * @param {number} time
     */
  record_spawn_time(time) {
    wasm.performancemonitor_record_spawn_time(this.__wbg_ptr, time);
  }
  /**
     * @param {number} bytes
     */
  update_memory_usage(bytes) {
    wasm.performancemonitor_update_memory_usage(this.__wbg_ptr, bytes);
  }
  /**
     * @returns {number}
     */
  get_average_spawn_time() {
    const ret = wasm.performancemonitor_get_average_spawn_time(this.__wbg_ptr);
    return ret;
  }
  /**
     * @returns {number}
     */
  get_memory_usage_mb() {
    const ret = wasm.performancemonitor_get_memory_usage_mb(this.__wbg_ptr);
    return ret;
  }
  /**
     * @returns {boolean}
     */
  meets_performance_targets() {
    const ret = wasm.performancemonitor_meets_performance_targets(this.__wbg_ptr);
    return ret !== 0;
  }
  /**
     * @returns {string}
     */
  get_report() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.performancemonitor_get_report(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
}

const PerformanceTimerFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_performancetimer_free(ptr >>> 0, 1));

export class PerformanceTimer {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    PerformanceTimerFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_performancetimer_free(ptr, 0);
  }
  /**
     * @param {string} name
     */
  constructor(name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.performancetimer_new(ptr0, len0);
    this.__wbg_ptr = ret >>> 0;
    PerformanceTimerFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @returns {number}
     */
  elapsed() {
    const ret = wasm.performancetimer_elapsed(this.__wbg_ptr);
    return ret;
  }
  log() {
    wasm.performancetimer_log(this.__wbg_ptr);
  }
}

const PoolMetricsFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_poolmetrics_free(ptr >>> 0, 1));
/**
 * Pool metrics for monitoring
 */
export class PoolMetrics {

  static __wrap(ptr) {
    ptr = ptr >>> 0;
    const obj = Object.create(PoolMetrics.prototype);
    obj.__wbg_ptr = ptr;
    PoolMetricsFinalization.register(obj, obj.__wbg_ptr, obj);
    return obj;
  }

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    PoolMetricsFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_poolmetrics_free(ptr, 0);
  }
  /**
     * @returns {number}
     */
  get total_blocks() {
    const ret = wasm.__wbg_get_poolmetrics_total_blocks(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * @param {number} arg0
     */
  set total_blocks(arg0) {
    wasm.__wbg_set_poolmetrics_total_blocks(this.__wbg_ptr, arg0);
  }
  /**
     * @returns {number}
     */
  get free_blocks() {
    const ret = wasm.__wbg_get_poolmetrics_free_blocks(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * @param {number} arg0
     */
  set free_blocks(arg0) {
    wasm.__wbg_set_poolmetrics_free_blocks(this.__wbg_ptr, arg0);
  }
  /**
     * @returns {number}
     */
  get block_size() {
    const ret = wasm.__wbg_get_poolmetrics_block_size(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * @param {number} arg0
     */
  set block_size(arg0) {
    wasm.__wbg_set_poolmetrics_block_size(this.__wbg_ptr, arg0);
  }
  /**
     * @returns {number}
     */
  get reuse_count() {
    const ret = wasm.__wbg_get_poolmetrics_reuse_count(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * @param {number} arg0
     */
  set reuse_count(arg0) {
    wasm.__wbg_set_poolmetrics_reuse_count(this.__wbg_ptr, arg0);
  }
  /**
     * @returns {number}
     */
  get memory_usage_mb() {
    const ret = wasm.__wbg_get_poolmetrics_memory_usage_mb(this.__wbg_ptr);
    return ret;
  }
  /**
     * @param {number} arg0
     */
  set memory_usage_mb(arg0) {
    wasm.__wbg_set_poolmetrics_memory_usage_mb(this.__wbg_ptr, arg0);
  }
}

const RuntimeFeaturesFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_runtimefeatures_free(ptr >>> 0, 1));

export class RuntimeFeatures {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    RuntimeFeaturesFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_runtimefeatures_free(ptr, 0);
  }
  constructor() {
    const ret = wasm.runtimefeatures_new();
    this.__wbg_ptr = ret >>> 0;
    RuntimeFeaturesFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @returns {boolean}
     */
  get simd_available() {
    const ret = wasm.runtimefeatures_simd_available(this.__wbg_ptr);
    return ret !== 0;
  }
  /**
     * @returns {boolean}
     */
  get threads_available() {
    const ret = wasm.runtimefeatures_threads_available(this.__wbg_ptr);
    return ret !== 0;
  }
  /**
     * @returns {bigint}
     */
  get memory_limit() {
    const ret = wasm.runtimefeatures_memory_limit(this.__wbg_ptr);
    return BigInt.asUintN(64, ret);
  }
  /**
     * @returns {string}
     */
  get_features_json() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.runtimefeatures_get_features_json(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
}

const SimdBenchmarkFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_simdbenchmark_free(ptr >>> 0, 1));
/**
 * Performance benchmarking utilities
 */
export class SimdBenchmark {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    SimdBenchmarkFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_simdbenchmark_free(ptr, 0);
  }
  constructor() {
    const ret = wasm.simdbenchmark_new();
    this.__wbg_ptr = ret >>> 0;
    SimdBenchmarkFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * Benchmark SIMD vs scalar dot product
     * @param {number} size
     * @param {number} iterations
     * @returns {string}
     */
  benchmark_dot_product(size, iterations) {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.simdbenchmark_benchmark_dot_product(retptr, this.__wbg_ptr, size, iterations);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * Benchmark SIMD vs scalar activation functions
     * @param {number} size
     * @param {number} iterations
     * @param {string} activation
     * @returns {string}
     */
  benchmark_activation(size, iterations, activation) {
    let deferred2_0;
    let deferred2_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passStringToWasm0(activation, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
      const len0 = WASM_VECTOR_LEN;
      wasm.simdbenchmark_benchmark_activation(retptr, this.__wbg_ptr, size, iterations, ptr0, len0);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred2_0 = r0;
      deferred2_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred2_0, deferred2_1, 1);
    }
  }
}

const SimdMatrixOpsFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_simdmatrixops_free(ptr >>> 0, 1));
/**
 * SIMD-accelerated matrix operations
 */
export class SimdMatrixOps {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    SimdMatrixOpsFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_simdmatrixops_free(ptr, 0);
  }
  constructor() {
    const ret = wasm.simdbenchmark_new();
    this.__wbg_ptr = ret >>> 0;
    SimdMatrixOpsFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * SIMD-optimized matrix-vector multiplication
     * @param {Float32Array} matrix
     * @param {Float32Array} vector
     * @param {number} rows
     * @param {number} cols
     * @returns {Float32Array}
     */
  matrix_vector_multiply(matrix, vector, rows, cols) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passArrayF32ToWasm0(matrix, wasm.__wbindgen_export_2);
      const len0 = WASM_VECTOR_LEN;
      const ptr1 = passArrayF32ToWasm0(vector, wasm.__wbindgen_export_2);
      const len1 = WASM_VECTOR_LEN;
      wasm.simdmatrixops_matrix_vector_multiply(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1, rows, cols);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v3 = getArrayF32FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 4, 4);
      return v3;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * SIMD-optimized matrix-matrix multiplication (small matrices)
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @param {number} a_rows
     * @param {number} a_cols
     * @param {number} b_cols
     * @returns {Float32Array}
     */
  matrix_multiply(a, b, a_rows, a_cols, b_cols) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_export_2);
      const len0 = WASM_VECTOR_LEN;
      const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_export_2);
      const len1 = WASM_VECTOR_LEN;
      wasm.simdmatrixops_matrix_multiply(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1, a_rows, a_cols, b_cols);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v3 = getArrayF32FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 4, 4);
      return v3;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
}

const SimdVectorOpsFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_simdvectorops_free(ptr >>> 0, 1));
/**
 * SIMD-accelerated vector operations
 */
export class SimdVectorOps {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    SimdVectorOpsFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_simdvectorops_free(ptr, 0);
  }
  constructor() {
    const ret = wasm.simdbenchmark_new();
    this.__wbg_ptr = ret >>> 0;
    SimdVectorOpsFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * SIMD-optimized vector dot product
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {number}
     */
  dot_product(a, b) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_export_2);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_export_2);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.simdvectorops_dot_product(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    return ret;
  }
  /**
     * SIMD-optimized vector addition
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {Float32Array}
     */
  vector_add(a, b) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_export_2);
      const len0 = WASM_VECTOR_LEN;
      const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_export_2);
      const len1 = WASM_VECTOR_LEN;
      wasm.simdvectorops_vector_add(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v3 = getArrayF32FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 4, 4);
      return v3;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * SIMD-optimized vector scaling
     * @param {Float32Array} vec
     * @param {number} scalar
     * @returns {Float32Array}
     */
  vector_scale(vec, scalar) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passArrayF32ToWasm0(vec, wasm.__wbindgen_export_2);
      const len0 = WASM_VECTOR_LEN;
      wasm.simdvectorops_vector_scale(retptr, this.__wbg_ptr, ptr0, len0, scalar);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v2 = getArrayF32FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 4, 4);
      return v2;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * SIMD-optimized activation function application
     * @param {Float32Array} vec
     * @param {string} activation
     * @returns {Float32Array}
     */
  apply_activation(vec, activation) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passArrayF32ToWasm0(vec, wasm.__wbindgen_export_2);
      const len0 = WASM_VECTOR_LEN;
      const ptr1 = passStringToWasm0(activation, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
      const len1 = WASM_VECTOR_LEN;
      wasm.simdvectorops_apply_activation(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v3 = getArrayF32FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 4, 4);
      return v3;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
}

const WasmAgentFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_wasmagent_free(ptr >>> 0, 1));

export class WasmAgent {

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    WasmAgentFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_wasmagent_free(ptr, 0);
  }
  /**
     * @param {string} id
     * @param {string} agent_type
     */
  constructor(id, agent_type) {
    const ptr0 = passStringToWasm0(id, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(agent_type, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.wasmagent_new(ptr0, len0, ptr1, len1);
    this.__wbg_ptr = ret >>> 0;
    WasmAgentFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @returns {string}
     */
  get id() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmagent_id(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @returns {string}
     */
  get agent_type() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmagent_agent_type(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @returns {string}
     */
  get status() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmagent_status(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @param {string} status
     */
  set_status(status) {
    const ptr0 = passStringToWasm0(status, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    wasm.wasmagent_set_status(this.__wbg_ptr, ptr0, len0);
  }
  /**
     * @param {string} capability
     */
  add_capability(capability) {
    const ptr0 = passStringToWasm0(capability, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    wasm.wasmagent_add_capability(this.__wbg_ptr, ptr0, len0);
  }
  /**
     * @param {string} capability
     * @returns {boolean}
     */
  has_capability(capability) {
    const ptr0 = passStringToWasm0(capability, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wasmagent_has_capability(this.__wbg_ptr, ptr0, len0);
    return ret !== 0;
  }
}

const WasmForecastingModelFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_wasmforecastingmodel_free(ptr >>> 0, 1));

export class WasmForecastingModel {

  static __wrap(ptr) {
    ptr = ptr >>> 0;
    const obj = Object.create(WasmForecastingModel.prototype);
    obj.__wbg_ptr = ptr;
    WasmForecastingModelFinalization.register(obj, obj.__wbg_ptr, obj);
    return obj;
  }

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    WasmForecastingModelFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_wasmforecastingmodel_free(ptr, 0);
  }
  /**
     * @param {string} model_type
     */
  constructor(model_type) {
    const ptr0 = passStringToWasm0(model_type, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.create_forecasting_model(ptr0, len0);
    this.__wbg_ptr = ret >>> 0;
    WasmForecastingModelFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @param {Float64Array} input
     * @returns {Float64Array}
     */
  predict(input) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passArrayF64ToWasm0(input, wasm.__wbindgen_export_2);
      const len0 = WASM_VECTOR_LEN;
      wasm.wasmforecastingmodel_predict(retptr, this.__wbg_ptr, ptr0, len0);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v2 = getArrayF64FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 8, 8);
      return v2;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * @returns {string}
     */
  get_model_type() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmforecastingmodel_get_model_type(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
}

const WasmNeuralNetworkFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_wasmneuralnetwork_free(ptr >>> 0, 1));

export class WasmNeuralNetwork {

  static __wrap(ptr) {
    ptr = ptr >>> 0;
    const obj = Object.create(WasmNeuralNetwork.prototype);
    obj.__wbg_ptr = ptr;
    WasmNeuralNetworkFinalization.register(obj, obj.__wbg_ptr, obj);
    return obj;
  }

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    WasmNeuralNetworkFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_wasmneuralnetwork_free(ptr, 0);
  }
  /**
     * @param {Uint32Array} layers
     * @param {ActivationFunction} activation
     */
  constructor(layers, activation) {
    const ptr0 = passArray32ToWasm0(layers, wasm.__wbindgen_export_2);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wasmneuralnetwork_new(ptr0, len0, activation);
    this.__wbg_ptr = ret >>> 0;
    WasmNeuralNetworkFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @param {number} min
     * @param {number} max
     */
  randomize_weights(min, max) {
    wasm.wasmneuralnetwork_randomize_weights(this.__wbg_ptr, min, max);
  }
  /**
     * @param {Float64Array} weights
     */
  set_weights(weights) {
    const ptr0 = passArrayF64ToWasm0(weights, wasm.__wbindgen_export_2);
    const len0 = WASM_VECTOR_LEN;
    wasm.wasmneuralnetwork_set_weights(this.__wbg_ptr, ptr0, len0);
  }
  /**
     * @returns {Float64Array}
     */
  get_weights() {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmneuralnetwork_get_weights(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v1 = getArrayF64FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 8, 8);
      return v1;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * @param {Float64Array} inputs
     * @returns {Float64Array}
     */
  run(inputs) {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passArrayF64ToWasm0(inputs, wasm.__wbindgen_export_2);
      const len0 = WASM_VECTOR_LEN;
      wasm.wasmneuralnetwork_run(retptr, this.__wbg_ptr, ptr0, len0);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v2 = getArrayF64FromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 8, 8);
      return v2;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
}

const WasmSwarmOrchestratorFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_wasmswarmorchestrator_free(ptr >>> 0, 1));

export class WasmSwarmOrchestrator {

  static __wrap(ptr) {
    ptr = ptr >>> 0;
    const obj = Object.create(WasmSwarmOrchestrator.prototype);
    obj.__wbg_ptr = ptr;
    WasmSwarmOrchestratorFinalization.register(obj, obj.__wbg_ptr, obj);
    return obj;
  }

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    WasmSwarmOrchestratorFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_wasmswarmorchestrator_free(ptr, 0);
  }
  /**
     * @param {string} topology
     */
  constructor(topology) {
    const ptr0 = passStringToWasm0(topology, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.create_swarm_orchestrator(ptr0, len0);
    this.__wbg_ptr = ret >>> 0;
    WasmSwarmOrchestratorFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
     * @param {string} config
     * @returns {string}
     */
  spawn(config) {
    let deferred2_0;
    let deferred2_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      const ptr0 = passStringToWasm0(config, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
      const len0 = WASM_VECTOR_LEN;
      wasm.wasmswarmorchestrator_spawn(retptr, this.__wbg_ptr, ptr0, len0);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred2_0 = r0;
      deferred2_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred2_0, deferred2_1, 1);
    }
  }
  /**
     * @param {string} config
     * @returns {WasmTaskResult}
     */
  orchestrate(config) {
    const ptr0 = passStringToWasm0(config, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wasmswarmorchestrator_orchestrate(this.__wbg_ptr, ptr0, len0);
    return WasmTaskResult.__wrap(ret);
  }
  /**
     * @param {string} agent_id
     */
  add_agent(agent_id) {
    const ptr0 = passStringToWasm0(agent_id, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len0 = WASM_VECTOR_LEN;
    wasm.wasmswarmorchestrator_add_agent(this.__wbg_ptr, ptr0, len0);
  }
  /**
     * @returns {number}
     */
  get_agent_count() {
    const ret = wasm.wasmswarmorchestrator_get_agent_count(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
     * @returns {string}
     */
  get_topology() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmswarmorchestrator_get_topology(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @param {boolean} detailed
     * @returns {string}
     */
  get_status(detailed) {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmswarmorchestrator_get_status(retptr, this.__wbg_ptr, detailed);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
}

const WasmTaskResultFinalization = (typeof FinalizationRegistry === 'undefined')
  ? { register: () => {}, unregister: () => {} }
  : new FinalizationRegistry(ptr => wasm.__wbg_wasmtaskresult_free(ptr >>> 0, 1));

export class WasmTaskResult {

  static __wrap(ptr) {
    ptr = ptr >>> 0;
    const obj = Object.create(WasmTaskResult.prototype);
    obj.__wbg_ptr = ptr;
    WasmTaskResultFinalization.register(obj, obj.__wbg_ptr, obj);
    return obj;
  }

  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    WasmTaskResultFinalization.unregister(this);
    return ptr;
  }

  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_wasmtaskresult_free(ptr, 0);
  }
  /**
     * @returns {string}
     */
  get task_id() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmtaskresult_task_id(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @returns {string}
     */
  get description() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmtaskresult_description(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @returns {string}
     */
  get status() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmtaskresult_status(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
  /**
     * @returns {string[]}
     */
  get assigned_agents() {
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmtaskresult_assigned_agents(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      const v1 = getArrayJsValueFromWasm0(r0, r1).slice();
      wasm.__wbindgen_export_1(r0, r1 * 4, 4);
      return v1;
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
    }
  }
  /**
     * @returns {string}
     */
  get priority() {
    let deferred1_0;
    let deferred1_1;
    try {
      const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.wasmtaskresult_priority(retptr, this.__wbg_ptr);
      const r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
      const r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
      deferred1_0 = r0;
      deferred1_1 = r1;
      return getStringFromWasm0(r0, r1);
    } finally {
      wasm.__wbindgen_add_to_stack_pointer(16);
      wasm.__wbindgen_export_1(deferred1_0, deferred1_1, 1);
    }
  }
}

async function __wbg_load(module, imports) {
  if (typeof Response === 'function' && module instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming === 'function') {
      try {
        return await WebAssembly.instantiateStreaming(module, imports);

      } catch (e) {
        if (module.headers.get('Content-Type') != 'application/wasm') {
          console.warn('`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n', e);

        } else {
          throw e;
        }
      }
    }

    const bytes = await module.arrayBuffer();
    return await WebAssembly.instantiate(bytes, imports);

  } else {
    const instance = await WebAssembly.instantiate(module, imports);

    if (instance instanceof WebAssembly.Instance) {
      return { instance, module };

    } else {
      return instance;
    }
  }
}

function __wbg_get_imports() {
  const imports = {};
  imports.wbg = {};
  imports.wbg.__wbg_buffer_609cc3eee51ed158 = function(arg0) {
    const ret = getObject(arg0).buffer;
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_call_672a4d21634d4a24 = function() { return handleError(function (arg0, arg1) {
    const ret = getObject(arg0).call(getObject(arg1));
    return addHeapObject(ret);
  }, arguments); };
  imports.wbg.__wbg_error_524f506f44df1645 = function(arg0) {
    console.error(getObject(arg0));
  };
  imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
    let deferred0_0;
    let deferred0_1;
    try {
      deferred0_0 = arg0;
      deferred0_1 = arg1;
      console.error(getStringFromWasm0(arg0, arg1));
    } finally {
      wasm.__wbindgen_export_1(deferred0_0, deferred0_1, 1);
    }
  };
  imports.wbg.__wbg_from_2a5d3e218e67aa85 = function(arg0) {
    const ret = Array.from(getObject(arg0));
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_get_67b2ba62fc30de12 = function() { return handleError(function (arg0, arg1) {
    const ret = Reflect.get(getObject(arg0), getObject(arg1));
    return addHeapObject(ret);
  }, arguments); };
  imports.wbg.__wbg_get_b9b93047fe3cf45b = function(arg0, arg1) {
    const ret = getObject(arg0)[arg1 >>> 0];
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_instanceof_Error_4d54113b22d20306 = function(arg0) {
    let result;
    try {
      result = getObject(arg0) instanceof Error;
    } catch (_) {
      result = false;
    }
    const ret = result;
    return ret;
  };
  imports.wbg.__wbg_instanceof_Window_def73ea0955fc569 = function(arg0) {
    let result;
    try {
      result = getObject(arg0) instanceof Window;
    } catch (_) {
      result = false;
    }
    const ret = result;
    return ret;
  };
  imports.wbg.__wbg_length_e2d2a49132c1b256 = function(arg0) {
    const ret = getObject(arg0).length;
    return ret;
  };
  imports.wbg.__wbg_log_c222819a41e063d3 = function(arg0) {
    console.log(getObject(arg0));
  };
  imports.wbg.__wbg_message_97a2af9b89d693a3 = function(arg0) {
    const ret = getObject(arg0).message;
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_new_780abee5c1739fd7 = function(arg0) {
    const ret = new Float32Array(getObject(arg0));
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    const ret = new Error();
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_newnoargs_105ed471475aaf50 = function(arg0, arg1) {
    const ret = new Function(getStringFromWasm0(arg0, arg1));
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_newwithbyteoffsetandlength_e6b7e69acd4c7354 = function(arg0, arg1, arg2) {
    const ret = new Float32Array(getObject(arg0), arg1 >>> 0, arg2 >>> 0);
    return addHeapObject(ret);
  };
  imports.wbg.__wbg_now_807e54c39636c349 = function() {
    const ret = Date.now();
    return ret;
  };
  imports.wbg.__wbg_random_3ad904d98382defe = function() {
    const ret = Math.random();
    return ret;
  };
  imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
    const ret = getObject(arg1).stack;
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
  };
  imports.wbg.__wbg_static_accessor_GLOBAL_88a902d13a557d07 = function() {
    const ret = typeof global === 'undefined' ? null : global;
    return isLikeNone(ret) ? 0 : addHeapObject(ret);
  };
  imports.wbg.__wbg_static_accessor_GLOBAL_THIS_56578be7e9f832b0 = function() {
    const ret = typeof globalThis === 'undefined' ? null : globalThis;
    return isLikeNone(ret) ? 0 : addHeapObject(ret);
  };
  imports.wbg.__wbg_static_accessor_SELF_37c5d418e4bf5819 = function() {
    const ret = typeof self === 'undefined' ? null : self;
    return isLikeNone(ret) ? 0 : addHeapObject(ret);
  };
  imports.wbg.__wbg_static_accessor_WINDOW_5de37043a91a9c40 = function() {
    const ret = typeof window === 'undefined' ? null : window;
    return isLikeNone(ret) ? 0 : addHeapObject(ret);
  };
  imports.wbg.__wbg_warn_4ca3906c248c47c4 = function(arg0) {
    console.warn(getObject(arg0));
  };
  imports.wbg.__wbindgen_debug_string = function(arg0, arg1) {
    const ret = debugString(getObject(arg1));
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
  };
  imports.wbg.__wbindgen_is_undefined = function(arg0) {
    const ret = getObject(arg0) === undefined;
    return ret;
  };
  imports.wbg.__wbindgen_memory = function() {
    const ret = wasm.memory;
    return addHeapObject(ret);
  };
  imports.wbg.__wbindgen_number_get = function(arg0, arg1) {
    const obj = getObject(arg1);
    const ret = typeof(obj) === 'number' ? obj : undefined;
    getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
  };
  imports.wbg.__wbindgen_object_clone_ref = function(arg0) {
    const ret = getObject(arg0);
    return addHeapObject(ret);
  };
  imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
    takeObject(arg0);
  };
  imports.wbg.__wbindgen_string_get = function(arg0, arg1) {
    const obj = getObject(arg1);
    const ret = typeof(obj) === 'string' ? obj : undefined;
    const ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_export_2, wasm.__wbindgen_export_3);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
  };
  imports.wbg.__wbindgen_string_new = function(arg0, arg1) {
    const ret = getStringFromWasm0(arg0, arg1);
    return addHeapObject(ret);
  };
  imports.wbg.__wbindgen_throw = function(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
  };

  return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
  wasm = instance.exports;
  __wbg_init.__wbindgen_wasm_module = module;
  cachedDataViewMemory0 = null;
  cachedFloat32ArrayMemory0 = null;
  cachedFloat64ArrayMemory0 = null;
  cachedUint32ArrayMemory0 = null;
  cachedUint8ArrayMemory0 = null;


  wasm.__wbindgen_start();
  return wasm;
}

function initSync(module) {
  if (wasm !== undefined) return wasm;


  if (typeof module !== 'undefined') {
    if (Object.getPrototypeOf(module) === Object.prototype) {
      ({module} = module);
    } else {
      console.warn('using deprecated parameters for `initSync()`; pass a single object instead');
    }
  }

  const imports = __wbg_get_imports();

  __wbg_init_memory(imports);

  if (!(module instanceof WebAssembly.Module)) {
    module = new WebAssembly.Module(module);
  }

  const instance = new WebAssembly.Instance(module, imports);

  return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
  if (wasm !== undefined) return wasm;


  if (typeof module_or_path !== 'undefined') {
    if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
      ({module_or_path} = module_or_path);
    } else {
      console.warn('using deprecated parameters for the initialization function; pass a single object instead');
    }
  }

  if (typeof module_or_path === 'undefined') {
    module_or_path = new URL('ruv_swarm_wasm_bg.wasm', import.meta.url);
  }
  const imports = __wbg_get_imports();

  if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
    module_or_path = fetch(module_or_path);
  }

  __wbg_init_memory(imports);

  const { instance, module } = await __wbg_load(await module_or_path, imports);

  return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
