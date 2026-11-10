# ADR-009: Bounded weight-space programming over dense WASM models

Status: Implemented
Date: 2026-11-10

## Decision

Add a standalone `ruv-fann-wsp` ESM package over the published `ruv-fann` SDK. Represent changes as schema-versioned parameter replacement patches bound to a canonical SHA-256 baseline. Do not permit topology edits, executable code or unbounded tensors. Normalize omitted biases to zero and negative zero to zero before hashing.

Evaluate baseline and candidate on the same caller-supplied dataset using real WASM predictions. Require explicit finite MSE/regression thresholds and optionally a maximum absolute error. Bind evaluation records to model, dataset and policy hashes. Keep proposal generation provider-neutral through a prompt helper; no external provider credentials or calls are introduced.

Promotion is a separate operation. An in-memory workspace owns its proposals, accepted evaluation state and retained checkpoint history. Caller-created receipts cannot authorize promotion. Revision checks reject stale proposals, and rollback appends a new revision. AgentDB adds a durable counterpart with database transactions, revision/generation checks and MCP mutation policies.

## Constraints and consequences

- Structural validation does not prove behavioral correctness or generalization. Dataset provenance and separation from generation examples are application responsibilities.
- Fingerprints identify content; they are not signatures or authorization tokens.
- Synchronous WASM evaluation is bounded by rows, parameters and work; browser applications may schedule it in a Worker.
- Reusing existing core/WASM npm dependencies avoids changes to the inference runtime or binary release.
- The package supports small dense models. Transformer formats, automated LLM calls, gradient optimization and arbitrary model surgery are outside this release.
- Qualification must exercise real backend outputs, accepted/rejected edits, stale/tampered proposals, rollback, strict installed types, browser loading and public package bytes before release completion.
