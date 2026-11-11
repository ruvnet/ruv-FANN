# Autonomous loop qualification

The implementation adds four local packages to the original five-package toolkit: `invariants`, `swarm`, `registry`, and `loop`, each version 0.1.0. The selected architecture and control-flow contract are in [loop-design.md](loop-design.md). Existing uncommitted toolkit work and the core WSP API are preserved.

## Run

```sh
cd packages/wsp-toolkit
npm ci --ignore-scripts --no-audit --no-fund
npm run setup:agentdb
npx --no-install playwright install --with-deps chromium
npm run validate
npm run demo:loop
```

Setup explicitly installs the separate locked AgentDB runtime with optional dependencies omitted. The library never auto-installs anything. `validate` requires strict declarations, zero failed/cancelled/skipped/todo tests, the original numerical optimization benchmark and clean audits of both installed dependency trees. It records source-file and both lockfile digests in `.artifacts/validation.json`. CI installs the same fixture explicitly before qualification.

## Requirement-to-evidence trace

| Requirement | Executable evidence | Acceptance |
| --- | --- | --- |
| Actual concurrent proposal agents | `packages/swarm/test/swarm.test.mjs` | Distinct overlapping worker threads, bounded concurrency, real scalar/SIMD WASM; fixed operations only |
| Autonomous proposal/evaluation/selection/promotion | `packages/loop/test/loop.test.mjs`, `test/evolution-scenario.mjs` | Two models repaired, three development-only shortcuts independently rejected, fresh stored AgentDB proposal IDs authorize commits |
| Continuous bounded generations | `loop.test.mjs` | Successive proposals bind revisions 1 and 2; accepted updates reach revision 3 without reusing a stale base |
| Independent output constraints | `packages/invariants/test/invariants.test.mjs` | Exhaustive small Boolean-domain and affine oracles, inclusive bounds, finite outputs, scalar/SIMD agreement |
| Valid criteria before dispatch | Invariant work estimator and loop validation tests | Invalid names, shapes, tolerance use and budgets reject as configuration errors rather than candidate quality |
| Persistent state and revision history | `packages/registry/test/registry.test.mjs` | Actual AgentDB/sql.js create/propose/promote/rollback and close/reopen, preserved hashes and monotonic history |
| Concurrent revision authority | Registry and loop tests | Exactly one competing promotion succeeds; same-hash revision changes still invalidate stale CAS |
| Correct goal tracking under interference | Loop two-module rollback test | External rollback clears cached goal evidence; active state is refreshed before declaring goals met |
| Failed durability acknowledgement | Real-controller flush tests | Controller mutates memory, disk flush fails, run rejects, registry is poisoned; reopen recovers the persisted checkpoint |
| Cancellation and resource bounds | Swarm and loop tests | Per-job deadline/AbortSignal cleanup, no queued work after abort, bounded pending operations, conservative cumulative inference reservations |
| Installed package boundaries | `test/consumer.test.mjs` | Nine tarballs installed in a fresh consumer, strict TypeScript, Worker build, original scenarios and complete evolution/restart/rollback scenario |
| Existing edge compatibility | Original Chromium and workerd tests | Existing browser/Cloudflare operations still run actual WASM; the Node-only loop is not bundled into them |
| Supply-chain qualification | `scripts/validate.mjs` | Root audit and isolated AgentDB installed-tree audit report zero known advisories |

The loop-specific suite contains 47 tests: 14 invariants, 10 swarm, 8 registry and 15 coordinator tests. The integrated workspace contains 94 tests, including the original 47. Final Node 22/24 receipts are saved separately under `docs/evidence/loop-node-*-validation.json`; initial five-package receipts remain historical evidence. Live Hub/ruOS results from the earlier task apply to the original packages and are not presented as a remote run of this loop.

## Demonstrated loop

The development slice for XOR contains one input. A search worker changes a single useful output edge and another changes the output bias; both fit that slice, and both fail the protected full-domain suite. A separate two-edge symbolic proposal passes the exhaustive XOR constraints. The linear task similarly rejects a constant-output shortcut and accepts the useful weight update. The coordinator promotes both passing checkpoints through AgentDB, then the demo reopens the database, verifies exact hashes/revisions, rolls back both to original content at revision 3, and reopens again.

The demo uses five real job threads with concurrency three, reserves 596 parameter-row work units, and makes two promotions in one generation. A separate test requires two generations to reach a goal. Worker-startup and elapsed-time observations are not nanosecond or zero-latency claims. The fixture is synthetic; it does not establish task generalization or useful adaptation on arbitrary models.

## Trust boundaries and review

Proposer → coordinator: only fixed search operations and candidate JSON cross worker boundaries. Worker-reported losses are diagnostic; the coordinator recomputes acceptance with real WASM. Neither held-out datasets nor acceptance rules are sent to workers. Input trees are bounded snapshots; functions, getters, unknown fields and executable payloads reject.

Coordinator → registry: every candidate binds the source model hash, independent dataset/policy hashes and invariant suite hash. A fresh AgentDB evaluation creates an internal proposal ID, and expected-revision CAS controls activation. The adapter serializes operations, flushes acknowledged mutations, detects stale SQLJS writers, and blocks reads/writes after uncertain persistence. Atomic commits already entered are awaited; abort cannot retroactively undo a committed transaction. Injected registry I/O requires its own operational timeouts. Budgets apply per invocation; multiple coordinators need an external shared budget authority.

Review reproduced and fixed malformed holdout-name and invariant preflight behavior, inaccurate same-hash revision reporting, cached-goal false success after a competing rollback, and cancellation during the final state refresh. Regression tests use actual AgentDB mutations. A stronger persistence test forces a failure after the real promotion transaction, verifies the error reaches the loop, and verifies the original disk checkpoint on reopen.

The AgentDB capability is a local trusted dependency, not an identity or authorization boundary against arbitrary code in the same process. Node workers provide separate execution contexts and WASM instances; they are not hardened sandboxes. Finite-domain constraints and SHA-256 bindings are not universal alignment proofs, signatures or publication authorization. Continuous mutation does not imply globally safe composition.

## Live tool and integration scope

The cloud executor is current and available. Live Ruflo and AgentDB runtime connector probes returned internal errors, so no execution or persistence is attributed to those connectors. Platform subagents performed the engineering work; the implementation itself uses real local Node workers and real `agentdb@3.10.0` with sql.js. The fixture omits optional native drivers, embedding/model packages and telemetry integrations. No provider credentials are copied, no model downloaded, and no external agent jobs or public federation messages are started.

RuVector semantic retrieval, MetaHarness/Agentic Flow routing, generated source packages, public npm publication, hardware actuation, cryptographic alignment guarantees and autonomous financial contracting remain outside this implementation. The adapter/capability boundaries leave room for those integrations without pretending that absent runtimes executed them.
