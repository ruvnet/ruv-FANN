/** Structural interface implemented by real ruv-FANN WasmPreparedNetwork. */
export interface PreparedNetwork {
  readonly input_size: number;
  readonly output_size: number;
  readonly estimated_bytes: number;
  run(input: Float64Array): Float64Array;
  create_batch(maxRows: number): PreparedBatch;
}
export interface PreparedBatch {
  readonly input_size: number;
  readonly output_size: number;
  readonly max_rows: number;
  readonly estimated_bytes: number;
  run_into(input: Float64Array, output: Float64Array, rows: number): void;
  free(): void;
}
export interface Encoder {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly batchRows: number;
  encode(inputs: Float64Array): Float64Array;
  /** Frees owned batch storage, retaining the caller's prepared network. */
  close(): void;
}
export interface EncoderOptions { batchRows?: number; maxRows?: number }
export function createEncoder(prepared: PreparedNetwork, options?: EncoderOptions): Encoder;
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export type Metadata = Record<string, JsonValue>;
export type Metric = 'cosine' | 'euclidean' | 'dotproduct' | 'manhattan';
export interface SearchResult {
  id: string;
  /** Higher values indicate greater similarity; not a probability. */
  score: number;
  similarity: number;
  /** Raw index distance, lower is better. */
  distance: number;
  metadata?: Metadata;
}
export interface VectorStore {
  readonly dimensions: number;
  readonly metric: Metric;
  readonly indexType: 'flat' | 'hnsw';
  readonly size: number;
  insert(entry: { id: string; vector: Float32Array; metadata?: Metadata }): string;
  search(query: { vector: Float32Array; k: number }): SearchResult[];
  delete(id: string): boolean;
  close(): void;
}
export interface NeuralIndexOptions extends EncoderOptions {
  prepared: PreparedNetwork;
  /** Ownership transfers to the returned index; must be empty. */
  store: VectorStore;
  /** Caller-supplied label identifying fixed weights; not an attestation. */
  modelId: string;
  maxEntries?: number;
}
export interface NeuralIndex {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly modelId: string;
  readonly indexType: 'flat' | 'hnsw';
  readonly size: number;
  add(entry: { id: string; input: Float64Array; metadata?: Metadata }): string;
  search(input: Float64Array, options?: { k?: number }): SearchResult[];
  delete(id: string): boolean;
  /** Frees owned encoder/store; the caller still owns the prepared model. */
  close(): void;
}
export function createNeuralIndex(options: NeuralIndexOptions): NeuralIndex;
