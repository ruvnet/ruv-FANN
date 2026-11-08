---
name: release-verification
description: "Prepare and verify ruv-FANN Cargo/npm release candidates with frozen source, real artifact provenance, clean consumers, dependency order, and immutable receipts. Use for release readiness, archive contents, version migrations, or candidate validation; it does not authorize publication."
---

# Verify reviewable release candidates

Read `docs/release-process.md`, `docs/release-inventory.json`, `docs/release-artifacts.json` and each selected package manifest. These describe prepared candidates and separately dated registry observations. A version in source or an archive is not proof that it has been published.

This workflow creates and checks local artifacts. Registry publication, remote deployments, account changes and outbound messages require the authorization actually present in the user's task; do not infer them from a request to verify. Do not request permission again if the user has already authorized the exact next action.

## Freeze and inventory

1. Identify selected workspaces/packages, intended versions, compatibility changes and release acceptance gates. Read dirty status and preserve other work.
2. Run the current inventory and inspect actual manifests/locks, feature combinations, MSRVs, licenses, exports and package allowlists.

   ```sh
   python3 scripts/verify-packages.py --inventory
   ```

3. Complete relevant tests, browser/native execution and audits before freezing source. Record commands and their actual outputs. Historical receipts do not cover new source changes.
4. Stop concurrent source writers and Cargo jobs before reusing their target caches. Check available disk; do not delete shared caches without their owner's agreement.
5. Choose a new empty task-owned output directory. Existing immutable receipts and archives must not be overwritten.

## Cargo archive verification

From the selected checkout root, with native/compiler prerequisites already configured:

```sh
python3 scripts/verify-release.py --output "/path/to/new-release-directory"
```

Inspect `--help` and the release guide for scoped workspaces and cache options:

```sh
python3 scripts/verify-release.py --output "/path/to/new-forecast-candidates" --workspace neuro-divergent --reuse-targets
```

The driver snapshots source into an isolated directory, preserves live locks and performs ordinary Cargo archive verification. It never uses `--no-verify` or publishes. Selecting swarm/forecasting includes the root dependency first. Verified upstream archives enter a checksum-checked isolated directory catalog; dependent candidates are built in dependency order. Do not edit a live Cargo home/index or source lock to simulate this staging.

Check the returned process status and read `receipt.json`, command logs, archive contents, hashes and `publication-order.json`. Success requires `verified: true`, zero failing commands, expected identities/license payloads and unchanged live locks. An old archive directory is not evidence for a failed new run.

## Node package verification

Use each distribution's own manifest and lockfile. For swarm, the genuine core/SIMD WASM binaries, generated bindings/types and source-build receipt must match frozen source. A binary hash alone is insufficient. A stale source receipt must fail prepack; rebuild from source instead of editing evidence.

Run from `ruv-swarm/npm/` with build prerequisites already installed:

```sh
npm run validate:package
npm test
npm run test:installation
```

For CUDA, from `cuda-wasm/`:

```sh
npm test
npm run build
npm run test:integration
npm pack --dry-run
```

CUDA integration needs the genuine `cuda2rust` prerequisite. Its Node build checks entry points and does not create compiled GPU/WASM kernels. Swarm package installation needs the real native SQLite addon. Clean consumer checks must exercise actual archive APIs, CJS/ESM/types/CLI, finite inference/forecasts or real compiler output, private-export rejection, validation and the full dependency audit.

## Package the Claude plugin

For changes to this plugin, validate the Claude manifest and every skill's frontmatter/references, run the bundled helper tests, and load the exact plugin through the host's supported `--plugin-dir`/installation flow. Inspect the generated archive for secrets, symlinks, caches, unrelated repository files and missing referenced files. Check the archive bytes, not only the source directory. Rebuild after any source change.

Use an isolated Claude configuration for install/uninstall tests so existing marketplaces/settings are preserved. A parsed manifest establishes format validity; actual host discovery and a bounded invocation establish additional E2E behavior. ruOS command receipts and private Ruflo coordination evidence must correspond to the same tested bytes.

## Acceptance and completion

Keep an [evidence bundle](references/evidence.md) with exact tool versions, source/lock hashes, commands, scope, archive SHA-256/size and consumer results. Separate compiler checks, runtime tests, package validation and audits. Record software-adapter versus physical-GPU execution and local OS versus hosted-matrix execution.

Before any separately authorized publication, recheck registry state, identity ownership, unchanged candidate hashes, authentication availability and dependency order. Root FANN precedes dependent swarm/forecasting; native/portable OpenCV leaves precede consumers; the compiler prerequisite must be available before promising usable CUDA Node installation. Never publish merely because verification passed.

If a command fails, preserve its exact error and failed receipt, fix the underlying issue and use a new output directory for a fresh run. Do not relax tests, fabricate receipts, reuse stale archive success, or remove advisory findings to unblock a release.
