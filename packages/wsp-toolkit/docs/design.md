# WSP toolkit delivery contract

## Outcome and scope

Deliver independent ESM/TypeScript subpackages around published `ruv-fann-wsp@0.2.0` with real WASM validation. Keep the existing WSP API unchanged. Actors are local developers, checkpoint reviewers and bounded patch generators. No generated code execution or implicit model promotion. No claim to edit transformer/safetensors checkpoints or validate arbitrary safety tasks.

## Boundaries and owners

- Coordinator: `packages/lifecycle`, `packages/ruos`, workspace integration, browser/packed E2E, CI, benchmarks, docs.
- Search worker: `packages/search` only; bounded deterministic derivative-free search, truth-table compilation and localized plasticity proposals. Experimental features must be explicit.
- Hub worker: `packages/huggingface` only; immutable-revision dense JSON fetch with SHA-256 and bounded downloads, metadata, tests.
- Cloudflare worker: `packages/cloudflare` only; stateless request handler, WASM Worker build and workerd E2E, no account mutations.

Each worker owns its source, declarations, manifest, README and tests. Shared dependency is exact `ruv-fann-wsp@0.2.0`. All packages start at 0.1.0, use ESM, export `src/index.js` and `src/index.d.ts`, and include an MIT license. No worker edits workspace lockfile or root files.

## Control flow and invariants

Snapshot inputs → bound work → hash base/data/policy → propose candidate → evaluate with real WASM → return evidence → explicitly promote against expected revision. Rejected or stale candidates never change live state. Rollback increments revision and invalidates old proposals. Remote JSON is untrusted data; pin a Hub revision and verify bytes before validating model schema. Requests cannot select arbitrary URLs or shell commands. Never transfer host credentials.

## Acceptance

Unit tests cover malformed data, bounds, stale revisions and regression rejection. E2E composes a two-weight XOR repair, activation trace, sensitivity, neuron ablation and rollback. Independent analytic or exhaustive boolean oracles check numerical outputs. A real Chromium run and Cloudflare workerd run use real WASM. Package archives are installed into a fresh consumer and checked with TypeScript. Run ruOS validation using the observed Work Desktop ID and only trust executor-owned completion metadata. Record live Hub fetch evidence separately from local fixtures and account capability reads. Performance benchmarks compare bounded search with exhaustive candidate evaluation and report timing as observations, not guarantees.

## Architecture decision

Use an isolated npm workspace under `packages/wsp-toolkit` rather than adding experimental dependencies to the repository's native/swarm root workspace. Independent packages allow platform adapters and experiments to evolve without inflating the core runtime or changing protected release lanes. Cloudflare operations are stateless; application owners must supply authentication and durable state before exposing a production mutation API. This task prepares and validates artifacts; it does not publish packages or replace existing deployments.
