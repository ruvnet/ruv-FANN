import type { PreparedNetwork } from './index.js';
export interface TuningOptions {
  prepared: PreparedNetwork;
  inputs: Float64Array;
  modelId: string;
  baselineBatchRows?: number;
  generations?: number;
  children?: number;
  samples?: number;
  seed?: number;
  timeoutMs?: number;
}
export interface MeasuredVariant {
  variantId: string;
  genome: { batchRows: number };
  score: { primary: number; regressed: boolean; raw: TuningContext & {
    batchRows: number; samplesNs: number[]; iterations: number; meanNs: number;
    medianNs: number; rowsPerSecond: number; maximumAbsoluteError: number;
    cacheHit: boolean; observationId: string;
  } };
}
export interface TuningContext {
  modelId: string;
  inputSize: number;
  outputSize: number;
  totalRows: number;
  precision: 'f64';
  inputSha256: string;
  outputSha256: string;
  workloadSha256: string;
  modelIdScope: string;
  timingScope: string;
}
export interface TuningReceipt {
  schema: 'ruv-fann.batch-tuning/1';
  status: 'passed';
  modelId: string;
  modelIdScope: string;
  context: TuningContext;
  options: Required<Omit<TuningOptions, 'prepared' | 'inputs' | 'modelId'>>;
  engine: { package: '@metaharness/darwin'; version: '0.13.0'; algorithm: 'evolveNumeric' };
  trialBudget: {
    absoluteLimit: number; worstCaseDispatches: number; dispatchCount: number;
    actualMeasurementCount: number; cacheHits: number; concurrency: number;
  };
  trials: MeasuredVariant[];
  baseline: MeasuredVariant;
  winner: MeasuredVariant;
  winnerRejected: { variantId: string; reason: string } | null;
  confirmation: {
    verified: true;
    status: 'passed' | 'not_improved';
    faster: boolean;
    baselineBatchRows: number;
    candidateBatchRows: number;
    meanSpeedRatio: number;
    ci95: [number, number];
    baselineSamplesNs: number[];
    candidateSamplesNs: number[];
    maximumAbsoluteError: number;
  };
  recommendation: { batchRows: number };
  modelCalls: 0;
  sourceApplied: false;
  temporaryArchiveRemoved: true;
  limits: {
    maxNumericBufferBytes: number; numericBufferBytes: number;
    encoderStagingBytes: number; cooperativeDeadline: true;
  };
}
/** Node-only, cooperative tuning in a trusted application; never auto-applies. */
export function tuneBatchRows(options: TuningOptions): Promise<TuningReceipt>;
