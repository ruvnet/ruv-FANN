# Independent correctness and security review

Review date: 2026 11 22.

## Decision

No high or critical defect was found in the bounded review of the integer
engine, JSONL interface, acceptance harness, or digits experiment. This supports
use as a local research prototype. It does not establish production readiness,
global model safety, cryptographic authentication, or full formal verification
of the Rust implementation.

The reviewer authored the mathematical proof package but did not implement
the Rust engine or digits experiment. The engine review was therefore
independent of its implementation; the proof author should not be described
as an independent reviewer of their own Lean statements.

## Inspected surface

The review covered all five Rust source files, both Rust integration test
files, `scripts/acceptance.py`, `scripts/digits_experiment.py`, the crate
manifest, and the generated experiment evidence. It examined malformed input,
integer products and intermediate sums, quantizer semantics, target scope,
certificate binding, stale updates, rollback, resource limits, experiment
selection, and claims supported by the results.

## Additional dynamic evidence

An independent Python arbitrary precision oracle enumerated every permitted
single coordinate replacement for 350 generated integer boundary cases.
The cases included i32 minimum and maximum values, large positive and negative
inputs, scales 1, 2, 8, and i32 maximum, constant clamps, and empty protected
controls. Twenty four cases had unsafe protected baseline execution and were
excluded before comparison. Among the remaining 326 cases, 215 were feasible.
The Rust solver and oracle agreed on feasibility and the exact deterministic
chosen patch in every case.

Eleven additional invalid CLI requests were rejected without unexpected process
termination. These covered zero and negative scales, ragged weights, invalid
weight bounds, reversed output bounds, empty edits, duplicate edits, unknown
model fields, malformed feature dimensions, unrepresentable edited weights,
and a tampered certificate prefix identity. A request exceeding the 8 MiB line
limit was rejected with the expected byte limit error.

These were targeted review checks, not exhaustive fuzzing over all JSON or
machine states. The repository acceptance scripts provide the maintained
reproduction commands for the broader oracle and workflow evidence.

## Correctness observations

1. The reference quantizer uses Euclidean division with positive scale, which
   agrees with floor division for negative accumulators. Constant and saturated
   clamp cells are handled explicitly.
2. Both model evaluation and synthesis check individual product bounds and
   every ordered accumulation prefix. This prevents final cancellation from
   hiding an earlier i32 overflow.
3. For rectangular input regions, row extrema are exact. Baseline boxes with
   nonconstant outputs are rejected. The solver splits replacement weights by
   sign before forming affine extrema constraints.
4. The solver is complete only for one changed weight with fixed biases and
   quantizers. The source and tests distinguish that from arbitrary repair.
5. Applying a certificate recomputes the candidate and preservation condition.
   No state changes occur before successful checks. Duplicate, unchanged,
   unrepresentable, and stale edits are rejected.
6. Rollback restores prior model bytes but advances the ledger revision and
   head. Earlier patches remain stale inside that ledger, including when a
   rollback restores identical weights.

## Integration limits that must remain explicit

**Target objectives are outside the certificate.** A certificate binds model,
patch, parent, prefix identifier, and protected domains. It proves the declared
preservation check. It does not contain target inputs or target outputs. The
solver checks targets during synthesis, and the experiments independently
replay them. An application that needs a portable certificate of improvement
must bind and independently verify the target objective as a separate contract.

**Prefix identity is asserted by the caller.** Hashing a supplied identifier
does not authenticate the executable upstream network, preprocessing, or
deployment environment. Bind real artifact digests and validate deployed
execution before extending the claim beyond this integer layer.

**Ledger replay protection is local to a live ledger.** Reconstructing a fresh
ledger from the same baseline and protected data reconstructs the same initial
head. Durable replay protection requires authenticated persisted state or an
external revision authority. The current API explicitly delegates persistence
and authentication to its caller.

**Fingerprints are not signatures.** SHA256 binds serialized context but does
not establish who authorized or produced a patch. Empty protection is allowed
as an explicit control and carries zero protected counts; a deployment policy
must reject that control when protection is mandatory.

**Per operation limits do not provide an aggregate service budget.** A JSONL
process accepts up to 4096 requests, and a sequence accepts up to 1024 steps.
Each step can consume substantial bounded verification work. The 64 MiB
rollback history cap limits model retention but does not itself limit total
CPU spent certifying later rejected steps. A hosted or otherwise untrusted
service needs request deadlines, an aggregate work budget, admission control,
and a process memory limit. No network service is introduced by this crate.

## Experiment assessment

The digits experiment uses disjoint index splits for training, calibration,
editing, and testing. Feature selection uses training data. Quantization scale
selection and protected anchor selection use calibration data. Adaptation uses
the edit split. Test labels and predictions are used for reporting; no test
feedback into candidate selection was found.

All three preset anchor sizes and all three methods are reported. The
preservation guarantee covers exact codes for the listed anchors and retained
edit inputs. It does not cover unseen data. The four anchor cell experiment
changes previously unprotected test codes and harms two test labels even while
preserving every declared anchor. That is evidence of the certificate's scope,
not a certificate failure, and must be retained in the result discussion.

The 100 patch witness is constructed synthetic data. It demonstrates scalar
target correction and retention, not 100 learned capabilities. The real digits
experiment is one small linear classifier and one split; it establishes neither
statistical superiority nor performance on large language models.

No adaptation cost advantage over retraining is established. In the recorded
local run, fitting the small ridge model takes about 2.7 milliseconds, while
cumulative cell constrained synthesis takes roughly 11 to 78 milliseconds
depending on the anchor count. Rust timings and Python timings must not be
presented as an algorithmic speed comparison. Report measured workload,
language, host, and scope alongside latency.

## Acceptance and remaining work

Accept this change as a research prototype when the Rust tests, independent
acceptance oracle, frozen digits replay, and Lean audit all pass. Before
production integration, add durable authorization and replay state, explicit
target objective binding where required, aggregate resource governance, and
validation against the actual integer deployment kernel. Before asserting a
practical adaptation advantage, run a predefined evaluation over additional
models and splits and compare full adaptation costs against retraining.

## Coordinator follow up

After review, the CLI gained a 100 million aggregate verification work cap for
each sequence request and a regression test for it. Multiple requests still
require a hosting process deadline and admission policy. The final maintained
Rust suite contains 24 tests. The scoped Ruflo 3.25.6 scan reported zero
findings; it does not substitute for a Rust dependency advisory audit.
