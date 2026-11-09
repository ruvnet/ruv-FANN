# Verify local release candidates

The verifier builds Cargo archives and executes Cargo's normal archive verification. It never publishes, requests registry credentials, uses `--no-verify`, or modifies a live checkout lockfile. Version numbers remain release candidates until authenticated registry publication completes.

Requires Python 3.11+ and modern Cargo supporting `cargo package --workspace`. Use the same Rust toolchain/profile as the preceding acceptance checks. Wait until the selected workspaces and their root dependency are frozen, and stop other Cargo jobs before reusing their target caches.

```sh
source "$HOME/.cargo/env"
source opencv-rust/scripts/native-env.sh /workspace/scratch/debian-native/prefix
python3 scripts/verify-release.py --output /workspace/scratch/release-candidates --reuse-targets
```

The native environment is required for `opencv-sys`: install/bootstrap OpenCV and libclang first using the checked scripts under `opencv-rust/scripts/`. The example prefix is the validated local Debian bootstrap, not a required global installation location. The release driver inherits native linker/pkg-config settings and fixes build jobs to one, development debug information to zero, and incremental compilation off.

## What is verified

The default run covers all 27 candidates: root (1), swarm (14), forecasting (6), CUDA/WASM (1), OpenCV (4), and Candle (1). It copies tracked and non-ignored untracked source files into the output directory, excluding target/node caches, agent state, databases, and generated crate artifacts. Internal symlinks are rebased into that copy and external symlinks are rejected. Cargo commands run only in that isolated snapshot. Existing workspace target caches may be reused; live source and locks are not edited.

The root archive is packaged and verified first. For swarm, forecasting, and Candle, its exact verified archive is extracted into an isolated Cargo directory-source catalog. Existing cached registry payloads are linked into that catalog, with file hashes derived from the reviewed `.crate` bytes. Only the external root dependency is redirected from a checkout path to that verified registry identity. The root's development dependencies therefore do not alter a dependent workspace's reviewed dependency graph.

Cargo's automatic multi-package staging does not support a directory-source upstream. Dependent families are consequently packaged **one candidate at a time in dependency order**, using normal Cargo verification; each successful immutable archive is then added to the isolated catalog before verifying its consumers. The real offline regression covers a root archive and two internally dependent candidates. Independent root/CUDA/OpenCV workspaces retain normal `cargo package --workspace` staging. Source replacement is passed on Cargo's command line, so temporary configuration cannot leak into an archive.

Every existing registry identity (name, version, source, checksum) must remain unchanged. The only new external identity permitted is the verified root archive and its exact SHA-256. No live registry index, Cargo-home file, or source lock is edited. The extracted root payload is hashed before/after verification, and selected live lock hashes must match before/after the complete run.

Every resulting archive is read back, checked for path traversal/special files, checked for forbidden database/runtime/build payloads, and checked for files matching its declared MIT/Apache license expression. Normalized manifests determine the publication dependency order, including renamed and platform-specific dependencies. Unknown publication cycles fail rather than inventing an order.

## Select workspaces and caches

```sh
python3 scripts/verify-release.py --output /workspace/scratch/forecast-candidates --workspace neuro-divergent --reuse-targets
python3 scripts/verify-release.py --output /workspace/scratch/native-candidates --workspace cuda-wasm --workspace opencv-rust --target-root /workspace/scratch/release-cache
```

`--workspace` accepts `.`, `ruv-swarm`, `neuro-divergent`, `cuda-wasm`, `opencv-rust`, and `integrations/candle`, plus the short aliases shown by `--help`. Selecting swarm, forecasting, or Candle automatically selects the root first. Use a new empty output directory for each run so an existing immutable receipt/archive cannot be overwritten. Without a cache option, targets are isolated under the output directory; that uses more disk. The driver does not clean caches or run concurrently with their owners.

## Receipts and publication

The output contains `archives/*.crate`, command logs, a source-file hash inventory, `receipt.json`, `receipt.sha256`, and `publication-order.json`. A successful receipt has `verified: true`, zero command exit codes, archive SHA-256/size/license/member details, exact tool versions/profile settings, unchanged live locks, and the dependency order. Archives and final receipts are read-only. A failed command returns a nonzero process exit and writes a failed receipt; a directory of stale `.crate` files is never treated as verification evidence.

Run the helper regression tests with:

```sh
python3 -m unittest discover -s scripts/tests -p test_verify_release.py -v
```

They include a real offline Cargo fixture proving that an extracted unpublished root candidate and two dependent packages can be built in dependency order, with successful builds and unchanged live locks. Archive verification does not replace full-target tests, browser/native runtime checks, package audits, or a review of what will be published. Publication needs authenticated registry access, unchanged reviewed versions, and the generated dependency order. This verifier intentionally does not contact publish endpoints.
