/** Compose actual prepared FANN inference with an owned vector store. */
import { createEncoder, checkedInputs, preparedDimensions, checkedInteger } from './encoder.mjs';
import { checkedLabel, checkedMetadata, checkedVector, VECTOR_LIMITS } from './ruvector.mjs';

export function createNeuralIndex(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options) ||
      Object.keys(options).some(key => !['prepared', 'store', 'modelId', 'batchRows', 'maxRows', 'maxEntries'].includes(key)))
    throw new TypeError('unknown neural index options');
  const { prepared, store, modelId, batchRows = 128, maxRows = 4096, maxEntries = 10000 } = options;
  const { inputSize, outputSize } = preparedDimensions(prepared);
  checkedLabel(modelId, 'modelId');
  checkedInteger(maxEntries, 1, VECTOR_LIMITS.maxEntries, 'maxEntries');
  if (!store || store.dimensions !== outputSize || store.size !== 0 || !['flat', 'hnsw'].includes(store.indexType) ||
      !['cosine', 'dotproduct', 'euclidean', 'manhattan'].includes(store.metric) ||
      ['insert', 'search', 'delete', 'close'].some(name => typeof store[name] !== 'function'))
    throw new TypeError('store must be a new empty compatible vector store');
  const encoder = createEncoder(prepared, { batchRows, maxRows });
  const indexType = store.indexType, ids = new Set(); let closed = false;
  const live = () => { if (closed) throw new Error('neural index is closed'); };
  const embedding = input => {
    const { rows } = checkedInputs(input, inputSize, outputSize, 1);
    if (rows !== 1) throw new RangeError('neural index records and queries require exactly one input row');
    const output = encoder.encode(input), vector = new Float32Array(outputSize);
    for (let index = 0; index < outputSize; index++) {
      vector[index] = output[index];
      if (!Number.isFinite(vector[index])) throw new RangeError('FANN output overflows finite f32 vector storage');
    }
    return checkedVector(vector, outputSize, store.metric);
  };
  return Object.freeze({ inputSize, outputSize, modelId, indexType,
    get size() { live(); return store.size; },
    add(entry) {
      live();
      if (!entry || typeof entry !== 'object') throw new TypeError('entry is required');
      const id = checkedLabel(entry.id);
      if (ids.has(id)) throw new Error('duplicate neural index id');
      if (ids.size >= maxEntries) throw new RangeError('neural index entry budget exceeded');
      const metadata = checkedMetadata(entry.metadata);
      const vector = embedding(entry.input);
      live();
      if (ids.has(id)) throw new Error('duplicate neural index id');
      if (ids.size >= maxEntries) throw new RangeError('neural index entry budget exceeded');
      const inserted = store.insert({ id, vector, metadata: metadata.value });
      if (inserted !== id) throw new Error('store returned an unexpected inserted identity');
      ids.add(id); return id;
    },
    search(input, options = {}) {
      live();
      if (!options || typeof options !== 'object' || Array.isArray(options) ||
          Object.keys(options).some(key => key !== 'k')) throw new TypeError('unknown search options');
      const k = checkedInteger(options.k ?? 5, 0, VECTOR_LIMITS.maxQueryK, 'k');
      return store.search({ vector: embedding(input), k });
    },
    delete(id) {
      live(); checkedLabel(id);
      const deleted = store.delete(id);
      if (deleted) ids.delete(id); return deleted;
    },
    close() { if (!closed) { closed = true; ids.clear(); try { encoder.close(); } finally { store.close(); } } }
  });
}
