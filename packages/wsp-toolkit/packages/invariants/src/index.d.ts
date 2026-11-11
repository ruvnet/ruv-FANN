import type { DenseModel } from 'ruv-fann-wsp';

export interface InvariantCase {
  /** Unique 1..64 ASCII letters, digits, underscore, dot or hyphen. */
  name: string;
  /** Flattened row-major inputs. */
  inputs: number[];
  rows: number;
  /** Inclusive lower output bounds, one value per model output. */
  minimum?: number[];
  /** Inclusive upper output bounds, one value per model output. */
  maximum?: number[];
  /** Flattened row-major expected outputs. */
  expected?: number[];
  /** Absolute tolerance for expected outputs only; default 0. */
  tolerance?: number;
}
export interface InvariantSuite {
  schema: 'ruv-fann.invariants/1';
  cases: InvariantCase[];
}
export interface InvariantOptions { simd?: boolean | 'auto' }
export interface InvariantViolation {
  case: string;
  row: number;
  output: number;
  constraint: 'minimum' | 'maximum' | 'expected';
  actual: number;
  /** Expected value for an expected-output constraint, otherwise its inclusive bound. */
  bound: number;
}
export interface InvariantCheck {
  schema: 'ruv-fann.invariant-check/1';
  modelHash: string;
  /** Hash of canonical suite plus the model's input/output dimensions. */
  suiteHash: string;
  accepted: boolean;
  /** Total scalar comparisons, including comparisons beyond the retained violation cap. */
  checks: number;
  violations: InvariantViolation[];
  violationCount: number;
  truncatedViolations: boolean;
  backend: 'wasm' | 'wasm-simd';
  /** parameterCount * sum(case.rows), with one full-model inference pass per case. */
  work: number;
}
/** Synchronous complete model/suite preflight; no inference, hash or runtime initialization. */
export function estimateInvariantWork(model: DenseModel | string, suite: InvariantSuite): number;
/** Finite-domain evidence only; no promotion authority or proof over arbitrary inputs. */
export function checkModelInvariants(model: DenseModel | string, suite: InvariantSuite, options?: InvariantOptions): Promise<InvariantCheck>;
export interface BooleanInvariant {
  /** 1..6 input bits, most-significant-bit-first row order. */
  inputs: number;
  /** 1..16 binary outputs. */
  outputs: number;
  /** Complete flattened numeric 0/1 truth table, length (2 ** inputs) * outputs. */
  targets: number[];
  /** Absolute tolerance; default 0. */
  tolerance?: number;
}
export function createBooleanInvariant(table: BooleanInvariant): InvariantSuite;
export const INVARIANT_LIMITS: Readonly<{ maxCases: 16; maxRows: 1024; maxElements: 131072; maxWork: 10000000; maxSuiteBytes: number; maxViolations: 128 }>;
