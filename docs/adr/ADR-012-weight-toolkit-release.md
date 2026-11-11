# ADR-012: Core capability facade and exact-archive toolkit publication

Status: Implemented source architecture; publication outcome requires current receipts.  
Date: 2026-11-11. Updated: 2026-11-11.  
Related: [toolkit delivery contract](../../packages/wsp-toolkit/docs/design.md), [loop architecture](../../packages/wsp-toolkit/docs/loop-design.md), [release setup](../constellation-release.md).

## Context

The standalone WSP package provides bounded weight edits and local revisions. Twelve additional packages add search, independent regression suites, finite-domain checks, actual Node workers, caller-owned AgentDB authority, checkpoint transport and package-discovery tools. Applications need a discoverable core entrypoint while existing published dependency contracts and browser inference remain stable.

A local passing checkout cannot establish that a package's public bytes match the reviewed source. New package identities also need a practical bootstrap publisher without silently replacing the already-working WSP publication contract or claiming unconfigured npm trusted publishers.

## Decision

The initial `ruv-fann@0.2.0` facade provided fourteen explicit typed capability subpaths. The current `ruv-fann@0.3.0` generation adds a fifteenth, `ruv-fann/agenticow`, backed by the exact `agenticow@0.3.0` dependency for native copy-on-write weight branches. Preserve the root inference/training entrypoint without eagerly importing Node workers, filesystem tooling or build integrations. Install exact dependencies for the complete facade; keep independently installable satellites for smaller footprints. The complete install includes Cloudflare's esbuild toolchain. AgentDB remains optional and caller-owned.

Retain published WSP/satellite pins to core 0.1.2. This is a finite versioned graph: legacy core depends only on WASM 0.1.2 and never imports the new facade. Browser import maps preserve that resolution with scopes. Do not rewrite old dependency contracts to create a same-version cycle.

The initial release grouped core plus twelve satellites under GitHub tag `ruv-fann-toolkit-v0.2.0`, with satellite versions 0.1.0. The current core-only release uses tag `ruv-fann-v0.3.0`: the protected toolkit lane packs one core archive once, installs its unchanged satellites from pinned published versions, binds qualification to exact source/archive digests and seals the complete resolved registry dependency lock. It requires successful actual installed-consumer checks on Node.js 22/24 and Chromium plus native Agenticow branch isolation, rejected-candidate isolation, explicit promotion, checkpoint rollback and reopen behavior. Publication carries only that exact core archive without rebuilding, then verifies public bytes and fresh installed behavior before reporting completion. A fresh candidate-tree audit includes the new Agenticow dependency graph.

Preserve the existing WSP token publisher. Publish core through the existing protected npm credential. The optional OIDC configuration helper is a future opt-in path requiring independently registered package-specific publishers; helper validation is not proof of a live npm identity exchange. Declared catalog `published` status is metadata, not namespace ownership or registry verification.

## Runtime and authority boundaries

Proposal workers execute fixed shipped operations on development cases. Independent holdouts and finite-domain invariants remain with the coordinator. An accepted detached evaluation is evidence, not authority. A fresh stored proposal and expected-revision CAS authorize the caller's explicit promotion. Rollback increases revision while restoring selected old content. Live-record revision checks and captured capabilities protect asynchronous authority boundaries.

Search objectives are trusted synchronous callbacks. Promise-returning objectives reject without leaving unhandled rejections. Actual AbortSignal state must survive shadowing of instance properties. Hub deadlines cover streamed response bodies. The private test/demo AgentDB fixture selects sql.js before import to make persistence-failure oracles reproducible; the published adapter never changes the host's environment or selects its database backend.

## Evidence and consequences

Qualification must use real numerical oracles, malformed-input regressions, strict declarations, browser/edge/worker scenarios where applicable, actual AgentDB reopen/fault cases and archive mutation failures. Zero tests, skips, todos, dirty lineage, missing source bindings or unsuccessful public installs cannot become a passing release. Current counts and publication outcomes are supplied by the final workflow receipts; no historical evidence file is rewritten to describe a newer generation.

The root import remains browser-compatible, but the complete npm installation grows to include toolkit/build dependencies and nested legacy core copies. Node-only subpaths stay explicit. Tests support finite stated domains, not universal safety or generalization. No physical GPU throughput, hosted Worker deployment, LLM-provider optimization or autonomous package-publication capability is implied by this decision.
