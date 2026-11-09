// Node-only timing helpers. Samples describe one host/workload, not new hosts.
import {performance} from 'node:perf_hooks';

export const positive = value => Number.isFinite(value) && value > 0;
export const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
export const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export function deadline(timeoutMs) {
  const end = performance.now() + timeoutMs;
  return () => {
    if (performance.now() > end) throw new Error('Batch tuning cooperative deadline exceeded');
  };
}

export function measure(run, iterations, check) {
  check();
  let checksum = 0;
  const start = performance.now();
  for (let i = 0; i < iterations; i++) { check(); checksum = run(); }
  const ns = (performance.now() - start) * 1e6 / iterations;
  if (!positive(ns) || !Number.isFinite(checksum)) throw new Error('Invalid tuning measurement');
  check();
  return ns;
}

export function calibrate(run, check) {
  let iterations = 1;
  for (let attempt = 0; attempt < 5; attempt++) {
    const ns = measure(run, iterations, check);
    if (ns * iterations >= 1e6 || iterations === 128)
      return Math.max(1, Math.min(128, Math.ceil(1e6 / ns)));
    iterations = Math.min(128, iterations * 8);
  }
  return iterations;
}

/** Paired timing-block bootstrap, fixed RNG, 2000 draws; not model generalization. */
export function pairedRatio(baseline, candidate, check = () => {}) {
  if (baseline.length !== candidate.length || baseline.length < 5 || baseline.length > 20 ||
      [...baseline, ...candidate].some(value => !positive(value))) throw new Error('Invalid paired samples');
  let state = 20261109;
  const ratios = [];
  for (let draw = 0; draw < 2000; draw++) {
    if (draw % 64 === 0) check();
    let a = 0, b = 0;
    for (let i = 0; i < baseline.length; i++) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const index = Math.floor(state / 4294967296 * baseline.length);
      a += baseline[index]; b += candidate[index];
    }
    ratios.push(a / b);
  }
  ratios.sort((a, b) => a - b);
  const result = {meanSpeedRatio: mean(baseline) / mean(candidate), ci95: [ratios[50], ratios[1949]]};
  if (!positive(result.meanSpeedRatio) || !result.ci95.every(positive)) throw new Error('Non-finite confidence evidence');
  return result;
}
