# ADR-007: Immutable prepared inference and explicit conformal calibration

Date: 2026-11-08. Status: accepted implementation direction; validation results belong to the implementation receipts.

## Context

The neural-network API exposes mutable topology and weights. Repacking those fields behind a hidden cache risks stale parameters and layout assumptions. Repeated callers also need a clear owner for temporary neuron values. The forecasting facade now produces real dated forecasts and chronological CV results, providing a place to build an explicit uncertainty baseline without pretending that probabilistic model architectures already exist.

[Current primary-source research](../research-sota-2026-11-08.md) separates packed-weight/workspace lifecycles, serving configuration, foundation model inference, and adaptive time-series calibration. They solve different problems. None provides a speedup or coverage guarantee for a different local implementation merely by sharing a name.

## Decision 1: Preparation creates an immutable owned snapshot

Add `Network::prepare()` and an explicit bounded constructor returning `PreparedNetwork<T>`. Validate execution-relevant topology, weights, activations, and canonical bias; ignore overwritten regular-neuron cached sums/values. Check parameter/neuron/byte arithmetic and copy required parameters into owned contiguous storage. Keep canonical activations and the original connection accumulation order. Use an indexed path for sparse, shuffled, or duplicate edges rather than silently changing their meaning.

Provide `prepared.workspace()` and `prepared.run_into(inputs, outputs, workspace)`. A workspace owns reusable layer buffers and is passed mutably by the caller. Share the immutable prepared object across workers with one workspace per concurrent caller. Neither the original mutable network nor a global cache owns those execution buffers.

The prepared object remains a snapshot after the original is edited, trained, or dropped. Applications prepare again when they want new weights. Check input/output sizes, workspace compatibility, finite values, and intermediate overflow before committing output. Rejected requests preserve the output buffer; workspace scratch need not preserve its previous contents.

### Consequences

- Preparation and workspace creation have explicit allocation and copying costs. Repeated inference should allocate zero times after successful setup.
- The first representation is portable f32/f64 storage. It is not an XNNPACK cache file, KleidiAI microkernel layout, quantized model, or automatic GPU/batch execution path.
- Preserving reduction order favors reproducibility over unvalidated reassociation or fused arithmetic. Future optimized kernels require explicit numerical tolerances and hardware checks.
- An owned snapshot adds memory and can be slower for some workloads. Report preparation, memory, and warm inference separately before making performance claims.

## Decision 2: Calibration is a separate immutable artifact

Add an explicit `ConformalConfig` with UTC `as_of`, horizon, coverages, and resource limits. `ConformalCalibrator::fit` consumes real CV results and partitions absolute errors by `(series, model, horizon)`. Derive each horizon from a complete time-ordered cutoff window. Preserve the supplied timestamp representation; do not generate or infer calendar offsets.

For `n` scores and coverage `c`, select sorted rank `ceil((n + 1) * c)`. Reject a rank above `n`; do not silently clamp it. Require finite values, valid unique keys, complete windows, and `cutoff < target_time <= as_of`. Reject invalid coverage values and excessive derived rows/groups before materializing them.

Application takes point forecasts plus an explicit origin for every series. Require origins at or after calibration `as_of`, future timestamps strictly after origin, and matching schema/models/series/horizon. Return long rows containing the point forecast, symmetric bounds, requested coverage, sample count, origin, and calibration boundary. Overflowing endpoints return an error.

The calibrator does not own or fingerprint model parameters, update itself online, or persist itself in this iteration. The caller must recalibrate after changing the model, preprocessing, forecasting procedure, or relevant data regime. Unsupported quantile/bootstrap/pipeline flags remain explicit errors; they are not aliases for this method.

### Statistical contract

Finite-sample split-conformal marginal coverage depends on exchangeable calibration/future scores under a suitably fixed procedure. Rolling CV with refits or overlapping targets can violate that condition. A chronological boundary prevents target leakage but does not prove exchangeability. Requested coverage is not measured coverage, and separate marginal intervals do not imply simultaneous trajectory or conditional coverage.

AcMCP `2410.13115v2` is dated 2026-02-02; version 3 is dated **2026-11-05**. PID-Expert `2610.02777v1` is dated **2026-10-02**. Their online updates, delayed-feedback treatment, dependence handling, expert aggregation, and theorem conditions are not implemented by this static baseline. They remain future comparative methods.

## Acceptance criteria

| Area | Required evidence |
| --- | --- |
| Prepared correctness | Scalar/prepared parity for both precisions, all canonical activations, unusual widths and sparse/shuffled/duplicate connections |
| Prepared lifecycle | Snapshot survives source mutation/drop; independent worker workspaces agree; malformed source/budget/size/nonfinite requests error without output mutation |
| Prepared resource use | Checked construction arithmetic and limits; zero allocation over repeated warmed calls; separate warm inference from explicitly labeled setup/allocation/destruction timing |
| Calibration arithmetic | Independently checked finite-sample ranks, ties, multiple coverages, insufficient samples, exact numeric coercion, finite residuals and endpoints |
| Calibration isolation | Different series/models/horizons retain separate widths; missing groups and incomplete/duplicate windows error |
| Calibration chronology | No observed target beyond `as_of`; no origin before it; no forecast at/before origin; timestamp precision/timezone metadata preserved; ordinal horizons avoid fixed-millisecond calendar assumptions |
| Evaluation | Later untouched targets score actual coverage/width and point error; disclose sample counts, chronological/refit policy, and temporal dependence limits |

Tests establish implementation behavior, not population coverage, SOTA accuracy, or a universal speedup. Benchmark receipts must identify software/hardware, seeds, shapes, precision, warm-up, sample counts, allocations, and setup amortization.

## Alternatives deferred

Hidden preparation caches require reliable mutation invalidation or a different topology API. ISA-specific microkernels require matching layouts, hardware dispatch, and scalar references. Packrat-style deployment needs an actual batching/threading workload. These are separate follow-up decisions.

Foundation models require checkpoint-specific licensing, an external runtime and data contract, and disclosed evaluation tasks. Pinned Chronos-2-small and synthetic-only Chronos-2 cards declare Apache-2.0 and are candidates for future external comparisons. TimesFM 3.0 downloaded weights have separate non-commercial/non-production terms despite Apache repository source. No external weights or paper code are incorporated by this decision.
