# ADR: WSP playground with local revisions and stateless edge corroboration

Date: 2026-11-12

Status: Implemented for review

## Decision

Keep each visitor's model workspace in browser memory. Expose a public,
strictly bounded Cloudflare endpoint that evaluates two known synthetic
scenarios using statically imported WASM. Require local and edge evidence
agreement before enabling explicit local promotion. Export receipts as JSON.

The business goal is a runnable introduction to inspectable neural changes,
with a visible rejected shortcut and exact rollback, suitable for package
adoption and demonstrations. Mobile and desktop share the same implementation.

## Alternatives

Browser-only inference has the lowest operating cost and latency, but does not
demonstrate Cloudflare execution. Shared Durable Object or AgentDB authority
would add authentication, state migrations, retention, tenancy and abuse
controls before they are needed. Native Agenticow and Node proposal workers
cannot simply be bundled into the browser or Worker.

Stateless edge corroboration adds one round trip per evaluation, leaving slider
inference local and responsive. It executes fixed policies and fixtures,
accepts only two bounded weight arrays, and has no promotion or persistence
endpoint. Initial asset delivery uses versioned Worker modules for connector
deployment; switch to Static Assets as traffic grows.

## Invariants

* No slider, search or failed gate changes active weights.
* Candidate model hashes must match the local gate and edge gate.
* Dataset, policy and invariant hashes must agree between runtimes.
* Weight changes invalidate passing evidence.
* Promotion uses the expected workspace revision.
* Rollback restores the original hash and increases the revision.
* Edge requests cannot select code, model structure, policy or datasets.
* All numbers in the visualizations originate from WASM predictions or
  explicitly labeled reference arithmetic; animation is illustrative.

## Acceptance

Scalar and SIMD numerical tests cover rejection, promotion and exact rollback.
API tests cover malformed encodings, bounded input, unknown fields and policy
injection. Browser checks cover the complete flow at desktop and mobile sizes,
export, failed edge requests and invalidation after edits. Record actual
execution evidence in the playground validation report; local Chromium failures
must not be represented as successful automated browser tests.
