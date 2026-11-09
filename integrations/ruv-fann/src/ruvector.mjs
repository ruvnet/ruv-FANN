/** Real @ruvector/wasm storage, with explicit caller-supplied bytes and no fetch path. */
import { checkedArray, checkedInteger } from './encoder.mjs';

export const VECTOR_LIMITS = Object.freeze({ maxDimensions: 65536, maxEntries: 10000,
  maxMetadataBytes: 16384, maxMetadataDepth: 8, maxMetadataNodes: 1024, maxLabelBytes: 256,
  maxPayloadBytes: 64 * 1024 * 1024, maxQueryK: 1000, maxWasmBytes: 16 * 1024 * 1024,
  minCosineNormSquared: 1e-18, maxNormSquared: 1e18 });
const utf8 = new TextEncoder();
let pinnedBytes, initialization;

export function checkedLabel(value, name = 'id') {
  if (typeof value !== 'string' || value.length === 0 || value.length > VECTOR_LIMITS.maxLabelBytes)
    throw new TypeError(`${name} must be a nonempty bounded UTF-8 string`);
  const bytes = utf8.encode(value);
  if (bytes.length > VECTOR_LIMITS.maxLabelBytes || new TextDecoder().decode(bytes) !== value)
    throw new RangeError(`${name} exceeds its UTF-8 byte budget or contains invalid Unicode`);
  return value;
}

/** Returns a detached JSON copy; accessors, toJSON callbacks and cyclic data are rejected. */
export function checkedMetadata(value) {
  if (value === undefined) return { value: undefined, json: undefined, bytes: 0 };
  const seen = new Set(); let nodes = 0, characters = 0;
  const text = value => {
    if (value.length > VECTOR_LIMITS.maxMetadataBytes) throw new RangeError('metadata string budget exceeded');
    characters += utf8.encode(value).length;
    if (characters > VECTOR_LIMITS.maxMetadataBytes) throw new RangeError('metadata byte budget exceeded');
    return value;
  };
  const copy = (item, depth) => {
    if (++nodes > VECTOR_LIMITS.maxMetadataNodes || depth > VECTOR_LIMITS.maxMetadataDepth)
      throw new RangeError('metadata depth/node budget exceeded');
    if (item === null || typeof item === 'boolean') return item;
    if (typeof item === 'string') return text(item);
    if (typeof item === 'number' && Number.isFinite(item)) return item;
    if (typeof item !== 'object' || seen.has(item)) throw new TypeError('metadata must contain only acyclic finite JSON data');
    const array = Array.isArray(item), prototype = Object.getPrototypeOf(item);
    if (!array && prototype !== Object.prototype && prototype !== null)
      throw new TypeError('metadata objects must be plain JSON records');
    seen.add(item);
    let result;
    if (array) {
      if (item.length > VECTOR_LIMITS.maxMetadataNodes) throw new RangeError('metadata array budget exceeded');
      result = [];
      for (let index = 0; index < item.length; index++) {
        const descriptor = Object.getOwnPropertyDescriptor(item, String(index));
        if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError('metadata arrays cannot contain accessors or holes');
        result.push(copy(descriptor.value, depth + 1));
      }
    } else {
      result = Object.create(null);
      for (const key in item) if (Object.hasOwn(item, key)) {
        text(key);
        const descriptor = Object.getOwnPropertyDescriptor(item, key);
        if (!Object.hasOwn(descriptor, 'value')) throw new TypeError('metadata accessors are not supported');
        result[key] = copy(descriptor.value, depth + 1);
      }
    }
    seen.delete(item); return result;
  };
  const json = JSON.stringify(copy(value, 0)), bytes = utf8.encode(json).length;
  if (bytes > VECTOR_LIMITS.maxMetadataBytes) throw new RangeError('metadata serialized byte budget exceeded');
  return { value: JSON.parse(json), json, bytes };
}

export function checkedVector(vector, dimensions, metric) {
  if (checkedArray(vector, 'vector', 'Float32Array') !== dimensions) throw new RangeError('vector dimension mismatch');
  let normSquared = 0;
  for (let index = 0; index < dimensions; index++) {
    const value = vector[index];
    if (!Number.isFinite(value)) throw new RangeError('vector must contain finite f32 values');
    normSquared = Math.fround(normSquared + Math.fround(value * value));
    if (!Number.isFinite(normSquared) || normSquared > VECTOR_LIMITS.maxNormSquared)
      throw new RangeError('vector norm exceeds the safe f32 distance budget');
  }
  if (metric === 'cosine' && normSquared < VECTOR_LIMITS.minCosineNormSquared)
    throw new RangeError('cosine vector has zero or unstable f32 norm');
  return vector;
}

async function initialize(wasmBytes) {
  const length = checkedArray(wasmBytes, 'wasmBytes', 'Uint8Array');
  if (length === 0 || length > VECTOR_LIMITS.maxWasmBytes) throw new RangeError('wasmBytes byte budget exceeded');
  const bytes = new Uint8Array(wasmBytes); // Snapshot before any await; shared views were rejected above.
  if (!WebAssembly.validate(bytes)) throw new TypeError('wasmBytes is not a valid WebAssembly binary');
  if (pinnedBytes) {
    if (bytes.length !== pinnedBytes.length || bytes.some((value, index) => value !== pinnedBytes[index]))
      throw new Error('RuVector module already initialized with different explicit bytes');
    return initialization;
  }
  pinnedBytes = bytes;
  // Requires native ESM/import maps or a bundler preserving import.meta.resolve.
  // A private module URL isolates wasm-bindgen's singleton from unrelated users.
  const url = new URL(import.meta.resolve('@ruvector/wasm'));
  const identity = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
  url.searchParams.set('ruv-fann-instance', identity);
  initialization = import(url.href).then(module => {
    module.initSync({ module: bytes }); return module;
  }).catch(error => { pinnedBytes = undefined; initialization = undefined; throw error; });
  return initialization;
}

function similarity(metric, distance) {
  if (metric === 'cosine') return 1 - distance;
  if (metric === 'dotproduct') return -distance;
  return 1 / (1 + distance);
}

/** A new, owned real VectorDB. Logical payload caps do not claim a total WASM/RSS bound. */
export async function createRuvectorStore(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options) ||
      Object.keys(options).some(key => !['wasmBytes', 'dimensions', 'metric', 'useHnsw', 'maxEntries'].includes(key)))
    throw new TypeError('unknown RuVector store options');
  const { wasmBytes, dimensions, metric = 'cosine', useHnsw = true, maxEntries = 10000 } = options;
  checkedInteger(dimensions, 1, VECTOR_LIMITS.maxDimensions, 'dimensions');
  checkedInteger(maxEntries, 1, VECTOR_LIMITS.maxEntries, 'maxEntries');
  if (!['cosine', 'euclidean', 'dotproduct', 'manhattan'].includes(metric) || typeof useHnsw !== 'boolean')
    throw new TypeError('unsupported metric or HNSW option');
  const module = await initialize(wasmBytes);
  const database = new module.VectorDB(dimensions, metric, useHnsw);
  const entries = new Map(); let closed = false, payloadBytes = 0;
  const live = () => { if (closed) throw new Error('RuVector store is closed'); };
  let indexType;
  try {
    indexType = database.indexType;
    if (!['flat', 'hnsw'].includes(indexType) || database.dimensions !== dimensions || database.len() !== 0)
      throw new Error('RuVector returned an incompatible database');
  } catch (error) { database.free(); throw error; }
  return Object.freeze({ dimensions, metric, indexType,
    get size() { live(); return database.len(); },
    insert(entry) {
      live();
      if (!entry || typeof entry !== 'object') throw new TypeError('entry is required');
      const id = checkedLabel(entry.id);
      if (entries.has(id)) throw new Error('duplicate vector id');
      if (entries.size >= maxEntries) throw new RangeError('vector entry budget exceeded');
      const source = entry.vector;
      checkedVector(source, dimensions, metric);
      const vector = new Float32Array(source);
      const metadata = checkedMetadata(entry.metadata), bytes = dimensions * 4 + utf8.encode(id).length + metadata.bytes;
      if (payloadBytes + bytes > VECTOR_LIMITS.maxPayloadBytes) throw new RangeError('vector/metadata payload budget exceeded');
      live();
      if (entries.has(id)) throw new Error('duplicate vector id');
      if (entries.size >= maxEntries) throw new RangeError('vector entry budget exceeded');
      const inserted = database.insert(vector, id, undefined);
      if (inserted !== id) throw new Error('RuVector returned an unexpected inserted identity');
      entries.set(id, { json: metadata.json, bytes }); payloadBytes += bytes;
      return id;
    },
    search(query) {
      live();
      if (!query || typeof query !== 'object') throw new TypeError('query is required');
      const k = checkedInteger(query.k, 0, VECTOR_LIMITS.maxQueryK, 'k');
      const source = query.vector;
      checkedVector(source, dimensions, metric);
      const vector = new Float32Array(source);
      if (k * dimensions * 4 > VECTOR_LIMITS.maxPayloadBytes) throw new RangeError('query result byte budget exceeded');
      live();
      if (k === 0 || entries.size === 0) return [];
      const handles = database.search(vector, Math.min(k, entries.size));
      try {
        const results = handles.map(handle => {
          const id = handle.id, distance = handle.score, stored = entries.get(id);
          if (!stored || !Number.isFinite(distance)) throw new Error('RuVector returned invalid result evidence');
          const score = similarity(metric, distance);
          if (!Number.isFinite(score)) throw new Error('RuVector distance cannot be represented as a finite similarity');
          return { id, distance, score, similarity: score, ...(stored.json === undefined ? {} : { metadata: JSON.parse(stored.json) }) };
        });
        results.sort((a, b) => b.score - a.score);
        return results;
      } finally { for (const handle of handles) handle.free(); }
    },
    delete(id) {
      live(); checkedLabel(id);
      if (!entries.has(id)) return false;
      const deleted = database.delete(id);
      if (deleted !== true) throw new Error('RuVector deletion disagreed with owned identities');
      payloadBytes -= entries.get(id).bytes; entries.delete(id); return true;
    },
    close() { if (!closed) { closed = true; entries.clear(); payloadBytes = 0; database.free(); } }
  });
}
