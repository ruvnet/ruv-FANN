# Initial five-package validation evidence

This records the original five-package qualification. The expanded nine-package loop is documented in [loop validation](loop-validation.md); its source-bound receipts are separate from the initial records below.

The toolkit was implemented against `ruvnet/ruv-FANN` base commit `4ae155ad300e92fb38bb234bce514e24da2f3cdc` on branch `feat/wsp-toolkit`. Candidate versions are 0.1.0. This is validation of the new workspace, not a release qualification of every native/swarm component in the repository.

## Requirement-to-evidence trace

| Requirement | Executable evidence | Result |
| --- | --- | --- |
| Preserve core patch, diagnostics and lifecycle behavior | `npm test --prefix packages/wsp` from repository root | 37 passed, zero skips |
| Subpackage contracts, bounds, snapshots and failures | `npm run test:unit` | 42 passed, zero skips |
| End-to-end repair, regression rejection, branching and rollback | `test/integration.test.mjs`, `test/scenario.mjs` | Real scalar/SIMD WASM; XOR MSE 0.5 → 0, ablation MSE 0.25 rejected, rollback revision 3 |
| Browser/edge adaptation | `test/browser.test.mjs` | Real Chromium 141.0.7390.37; both WASM backends |
| Cloudflare compatibility | `packages/cloudflare/test/workerd.test.mjs` | Real `workerd@1.20260730.1`; static scalar/SIMD WASM, auth and HTTP failures |
| Published-package shape | `test/consumer.test.mjs` | All five tarballs installed into fresh consumer; strict types, real WASM scenarios and Worker build |
| Public Hub checkpoint transport | `npm run test:hub:live`, [source pin](huggingface-source.json), [result](evidence/huggingface-live.json) | Live 2,619-byte affine head, 129 parameters; digest pinned; oracle error 1.78e-15 |
| ruOS execution | [host completion summary](evidence/ruos-validation.json), [tested archive hashes](evidence/package-archives.json) | Observed Work Desktop, Node 20.20.2; five archives verified; Worker build; scalar/SIMD lifecycle and live Hub check passed |
| Bounded optimization | `npm run benchmark` | Same selected candidate, 19 versus 32 inference passes (40.625% fewer), zero final MSE |
| Dependency security | `npm run audit`, `npm run validate` | Zero known advisories in full locked tree after pinned Miniflare dependency fixes |
| Repeatable qualification | `.github/workflows/wsp-toolkit.yml` | Node 22.13/24.22 matrix configured; not yet run on GitHub |

The full new-suite count is 47 tests: 42 package tests, one browser test, one packed-consumer test and three integration/CLI tests. `npm run validate` additionally requires a strict declaration check, zero cancelled/skipped/todo tests, benchmark equality and a clean npm audit. It writes `.artifacts/validation.json` with exact source and lockfile hashes. Captured local qualification reports are stored under `docs/evidence` after final execution; these are evidence records, not signatures.

## What the platform evidence establishes

Cloudflare's connected plugin successfully supplied documentation and a read of the authorized account. No Worker or account configuration was changed. Runtime testing is local Cloudflare `workerd`, not a hosted Cloudflare deployment. Miniflare is a development runtime and is not used as an isolation boundary for untrusted code.

Hugging Face's connected identity read succeeded. Its advertised model/dataset/documentation discovery methods returned `UNAVAILABLE`; discovery and the live file fetch used the public Hub HTTP API instead. Both this cloud workspace and ruOS fetched the same immutable revision and file digest. Hub validation explicitly maps a public weight vector to an affine layer; synthetic arithmetic tests do not establish accuracy on the source model's real sensing task. The Hub unit tests and the main integration scenario also contain clearly labeled synthetic fetch fixtures, separate from live evidence.

ruOS's live desktop/fleet reads identified the active Work Desktop. A harmless Node/npm/git probe established its executor route. Candidate tarballs were transferred into a unique temporary directory, verified against the cloud archive SHA-256 and per-package manifest, and installed without lifecycle scripts. Host-owned structured responses reported completion version 1, verified true, status ok and integer exit 0 for the actual validation commands. Local and remote checkpoint hashes and numerical results agreed. The helper and this report do not claim cryptographic attestation or independent proof of remote assertions. Desktop output was short and complete enough to inspect; missing optional completeness fields are not treated as an affirmative completeness attestation.

## Security review and changes

Trust boundaries reviewed: local callback → search, untrusted JSON → WASM, public Hub response → model, HTTP caller → Worker, and remote executor → completion consumer. Model/data/patch limits, cumulative work budgets, pre-await copies, finite metrics, no generated code evaluation, explicit promotion and hash/revision comparisons remain enforced.

Review reproduced and fixed a FIFO input hang in the CLI by opening nonblocking before checking for a regular file, and malformed truncation/timeout flags in the completion checker by requiring absent or false flags. Regression tests exercise both. The Worker defaults to denying operations without its bearer secret, refuses unknown routes/fields, and bounds streamed bytes even with fragmented or misleading content lengths. The Hub loader requires commit and raw digest pins, rejects external redirects, bounds downloads and validates model structure before use.

The full dependency audit initially flagged Miniflare's pinned `sharp@0.35.2` and `undici@7.28.0`. The isolated qualification workspace now overrides those development dependencies to `sharp@0.35.5` and `undici@7.30.1`. Real workerd tests were rerun after the overrides. Production dependencies were already clean. The npm audit result is a point-in-time advisory-feed check, not a general security certification.

## Performance interpretation

The benchmark evaluates the identical sequence of sixteen monotonically improving single-coordinate candidates. Its reference calls the original WSP baseline-plus-candidate evaluator for every proposal (32 inference passes). Search caches its initial baseline, screens each candidate once, and finally performs the original authoritative two-pass evaluation (19 total). Warm-up is excluded; nine repetitions alternate execution order. It requires an identical candidate hash and zero final loss. Local median times are recorded in the qualification JSON and may vary; the deterministic result is the reduced inference count. This small controlled case does not measure task accuracy, arbitrary-model repair success or generalization.

## Remaining boundaries

Packages are prepared and packed, not published to npm. CI is authored but has not executed on GitHub. No hosted Worker was deployed, no Hub model uploaded, and no production checkpoint promoted. Applications must choose appropriate held-out data, provide production authentication/durable state where needed, bound concurrent calls and keep trusted callback execution within their own security boundary. Experimental Boolean compilation, Hebbian proposals and branching remain explicitly scoped to small dense models.
