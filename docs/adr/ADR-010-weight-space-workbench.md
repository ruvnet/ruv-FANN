# ADR-010: Bounded weight-space introspection and reproducible evaluation

Status: Implemented
Date: 2026-11-10
Updated: 2026-11-10
Related: ADR-008, ADR-009

## Context

The published WSP package can evaluate hash-bound parameter edits, but an editor
cannot inspect hidden activations or ask which explicitly selected parameters
change a measured loss. The existing examples verify execution and revision
control; they do not establish performance on independent observed datasets.

## Decision

Extend `ruv-fann-wsp` to 0.2.0 while retaining the exact published
`ruv-fann@0.1.2` execution dependency. Keep four operations explicit:

- Inspect canonical model dimensions, activations and parameter counts.
- Trace affine values and activations with actual WASM layer preparations.
- Create a bounded hidden-neuron ablation patch by zeroing outgoing weights.
- Estimate MSE sensitivity for at most 16 named parameter locations using central
  finite differences; do not describe this operation as autograd.

All inputs are strictly snapshotted before asynchronous work. Trace storage and
both inference passes share one aggregate budget; sensitivity includes the
baseline and every positive/negative probe in its aggregate work budget. Reject
unknown fields, accessors, extra array properties, custom iterators, holes,
nonfinite values and numerical overflow. Dispose each prepared model in `finally`.
Hashes bind observations to content; they do not authorize model promotion.
The existing evaluated proposal and revision checks retain that authority.

Add a source-checkout benchmark using provenance-pinned, attributed UCI Iris and
Wine data. Split whole rows into stratified training, validation and untouched test
sets across five predetermined seeds. Fit preprocessing only on training rows;
select 100/200-epoch training budgets and fault-injection neurons only on validation rows.
Compare real-WASM linear and small nonlinear networks with simple baselines and
a pinned external tree implementation. Preserve negative results. Controlled
ablation/repair is an execution check, not evidence of arbitrary model improvement.

The npm package has no Python dependency. Python is isolated benchmark tooling.
The protected release must bind the benchmark report, dependency audit, Node22/24
consumer and unit results, and real Chromium result to the exact archive/source.
Revalidate the evidence before publication, and attach the exact benchmark and
security reports to the GitHub release.

## Acceptance criteria

1. Analytic affine values and loss derivatives match independent numeric oracles;
   scalar/SIMD final traces agree with complete real-WASM inference.
2. Malformed, mutation-prone and over-budget requests reject before inference;
   failures release prepared handles; all returned data is owned.
3. Neuron ablation zeros outgoing connections, preserves topology and produces a
   normal hash-bound patch subject to the existing limits and promotion policy.
4. Public-data benchmarks preserve row/split provenance and training-only
   preprocessing, compare baselines and report every run without selecting on test.
5. One immutable archive passes Node22/24, strict installed declarations, Chromium,
   hostile-input probes and a zero-advisory production dependency audit. The public
   npm archive and GitHub assets must match the sealed bytes.

## Limits

This release does not establish state of the art, agent-task generalization,
physical GPU throughput, automatic online learning, a task-embedding registry or
an LLM optimizer. Per-call budgets do not replace caller concurrency/rate limits.
Finite differences are local numerical estimates and may be unreliable at
activation discontinuities or under poor numerical conditioning. Platform
transcendental functions are compared with tolerances, not bit-identity claims.
