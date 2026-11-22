# ADR 013: Certified quantized weight edits

Date: 2026-11-22
Status: Experimental research implementation

## Problem and outcome

The existing WSP lifecycle binds edits and evaluates regression datasets, but
dataset scores do not prove exact preservation. Exact accumulator preservation
can also be unnecessarily restrictive for a quantized layer. The requested
experiment seeks useful edits with an explicit preservation contract and
independent evidence, without claiming an unverified scientific discovery.

## Decision

Add the isolated `crates/wsp-quant-guard` workspace. Its checked integer
reference model admits patches when every protected accumulator remains in
the original output quantization cell. Keep bias, scales and quantizer fixed.
Synthesize single weight replacements through integer interval intersections.
Require recomputation of all certificate claims before a transactional apply.
Retain immutable baseline outputs and monotonic revision history for rollback.

This specializes known quantization and constrained repair mathematics. The
candidate contribution is a small deployable contract, exact sparse capacity
characterization and measured comparison with a stricter preservation rule.
Publication priority is not established by the bounded literature search.

## Alternatives and constraints

| Option | Advantage | Cost or failure boundary |
|:--|:--|:--|
| Exact accumulator preservation | Simple strongest equality | Full protected rank can force zero edits |
| Protected quantized output cells | More admissible integer edits with exact code preservation | Depends on exact execution semantics and available cell width |
| Unconstrained target editing | More successful target proposals | Can damage protected examples and unseen accuracy |
| General MILP neural repair | Broader search and network coverage | Larger dependency and solving cost; outside this bounded experiment |

The crate remains separate from the existing float model and package release
lanes. No production path, user model, published package or hardware behavior
changes. Protection covers declared points or constant output integer boxes.
The prefix and deterministic downstream must be frozen by an integrating host.

## Control flow

Validate shapes, integer ranges and work budget. Compute original protected
output cells. Synthesize a bounded candidate. Check every product, accumulator
prefix and output cell. Bind baseline, parent, patch and specification hashes.
Recheck the certificate, then atomically apply or reject. Retain history and
advance the epoch for both apply and rollback. Evaluate target improvements
and independent retention separately from certificate validity.

## Evidence and requirements

| Requirement | Executable evidence |
|:--|:--|
| Exact integer arithmetic including negatives and saturation | Rust boundary tests and independent cell oracle |
| Solver agrees with complete declared search class | 1,000 Rust exhaustive fixtures and 400 independent Python fixtures |
| Repeated changes preserve original outputs | 100 distinct patches, 12,900 original output checks, 5,050 prior target checks |
| Rejection is atomic and rollback is exact | Tampering, stale parent, overflow and rollback tests |
| Mathematical arguments are checked independently | 17 Lean declarations and finite Python theory checks |
| Useful capacity is measured beyond a planted example | Fixed split digit classification with three protection sizes and controls |
| Claims remain within measured scope | Prior art, experiment report, source hashes and review record |

## Risks and next gate

Finite anchor preservation does not imply generalization. The digit experiment
demonstrates this directly. There is no comparison showing lower total cost
than retraining. Source hashes do not authenticate the external prefix or
authorize deployment. The Lean lemmas do not verify Rust machine code.

Owner: ruv-FANN maintainers. Next gate: one real quantized runtime with matching
integer semantics, disjoint retention evaluation, independent patch checking,
and a cost matched retraining control. Continue only if that experiment shows
useful retained corrections without crossing its declared regression limits.
