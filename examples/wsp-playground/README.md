# WSP Live Lab

An interactive, responsive playground backed by actual ruv-FANN WebAssembly.

## Run locally

Requires Node 22 or newer. From this directory:

```sh
npm ci --ignore-scripts
npm run build
npm run dev
```

Open `http://localhost:4173`. `npm test` runs numerical, API and real Chromium
acceptance tests. Set `CHROMIUM_PATH` to your Chromium executable if different
from `/usr/bin/chromium`. Set `PLAYGROUND_URL` to run the browser acceptance
test against a deployed Worker. Browser screenshots and receipts are written
to `.artifacts/` and are intentionally ignored by Git.

## Experiences

Each model lab opens with a plain language walkthrough. Start it to compare the
two versions, load and reject a shortcut, test a repair, explicitly promote it
and restore the original weights. Steps follow completed operations; a failed
edge check cannot advance the tour. Highlighted controls and a small glossary
explain the next action. Visitors can hide, resume or restart the tour. Starting
over restores original weights only in the current lab and retains its history.

* Robot repair: two obstacle sensors feed two ReLU units and a linear steering
  output. Edit the two output weights, trace the actual WASM activations,
  inspect the decision grid and compare active and candidate behavior.
* Ambient room: a synthetic presence signal removes background noise from
  motion evidence. It illustrates a RuView application, without claiming
  measured WiFi accuracy or operating a real sensor.
* Runtime bench: scalar and SIMD inference on the same tiny network, with an
  independent arithmetic oracle. Thirty timing samples each contain 100
  batches of 256 rows, after warmup. Report median and p95 batch latency,
  throughput and maximum arithmetic error; do not assume SIMD wins.

## Checkable lifecycle

1. Weight changes create a candidate and invalidate previous promotion evidence.
2. Search sees one development case and screens at most 24 candidates.
3. Lifecycle gates evaluate that development case and five separate cases.
4. Invariants evaluate reference outputs and bounds on an 11 by 11 sensor grid.
5. `/api/evaluate` repeats the fixed cases and policy in static WASM on Cloudflare.
   Browser and edge model, dataset, policy and invariant hashes must agree.
6. A button explicitly promotes the accepted local proposal using its expected
   revision. API failures block the button. The edge cannot promote anything.
7. Rollback creates a new revision with the exact original checkpoint hash.
8. Export saves model JSON, hashes, gates, edge evidence and revision history.

All fixtures are public. Separate cases are withheld from search, not secret
from visitors. These are finite-domain checks, not a generalization claim.
Session state is in memory and resets when the page reloads. There is no
shared production model, account, persistence, upload or LLM integration.

## Architecture and security

The browser bundles the focused WSP, search, lifecycle and invariants packages.
The worker statically imports scalar and SIMD WASM modules for edge evaluation.
Both runtimes reuse initialized WASM instances. Browser binaries are served from
the deployed Worker; there is no runtime code or WASM CDN dependency. Google
Fonts is optional and falls back to local fonts.

The public stateless evaluator accepts only a scenario and two bounded pairs of
weights. It does not accept datasets, policies, model graphs, external URLs or
code. Requests are capped at 1024 streamed bytes and two concurrent evaluations
per isolate. CSP, same-origin checks, method restrictions and generic errors
reduce the public surface. A deployment CPU limit bounds per-request compute.
The concurrency cap is not a distributed rate limiter. Account-level rate rules
are the next mitigation if traffic warrants them. No secrets are needed.

The initial build embeds assets in Worker modules to support a reproducible
connector deployment. Every asset request invokes the Worker and counts as a
Worker request. Migrate the files to Workers Static Assets if traffic becomes
material; that avoids per-asset Worker invocations without changing the app.

## Deployment and rollback

`npm run build` emits `dist/manifest.json`, recording each deployable module's
byte count, SHA256 and content type. The Worker entrypoint is `dist/worker.js`.
Upload the six listed modules using Cloudflare's Worker Versions API, then
deploy the created version at 100 percent. The main module is `worker.js`.
Use compatibility date `2026-11-12`, no Node compatibility, and a CPU cap of
100 milliseconds. Never infer deployment success from the local build.

Generated modules are retained on the separate `deploy/wsp-live-lab` branch;
they are not part of the source PR. Their manifest identifies the source commit
and npm lockfile. Rebuild from that source before replacing a deployment.
To roll back the website, redeploy the prior Cloudflare version. Model rollback
in the UI is a separate, local session operation.
