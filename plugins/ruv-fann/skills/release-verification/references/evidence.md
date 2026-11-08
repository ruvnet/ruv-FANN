# Release evidence bundle

For each candidate, retain:

| Field | Required meaning |
| --- | --- |
| Source identity | Commit plus dirty/source snapshot hash, not just branch name |
| Dependencies | Selected manifest/lock hashes and actual resolved identities |
| Toolchain | Rust/Cargo, Node/npm, Python, WASM/native tools and target/profile flags |
| Commands | Literal argv, working directory, start/end or duration, exit code and log |
| Runtime scope | OS, browser, adapter/device, native library versions and actual API exercised |
| Candidate artifact | Name/version, byte size, SHA-256, contents, licenses and exports |
| Consumers | Actual clean installation plus real APIs, errors and types |
| Audits | Tool/feed date, full graph, flags, findings and maintenance warnings |
| Remaining gates | Unexecuted provider, hardware, OS, registry or publication checks |

A byte hash identifies a payload; it does not prove a reproducible build, provenance signature or remote publication. Never claim a published version based on a prepared archive. Keep credentials and private inputs out of the bundle.

## Cargo receipt checks

The verifier's output includes `archives/*.crate`, logs, source inventory, `receipt.json`, `receipt.sha256` and `publication-order.json`. Check that the receipt passed and accounts for every selected candidate. Validate dependency edges, renamed/platform-specific dependencies and cycles. Read back every archive and inspect paths, normal files, license texts and normalized manifests. Special files, external symlinks, runtime databases and caches must not ship.

Run the verifier's real regression when changing it:

```sh
python3 -m unittest discover -s scripts/tests -p test_verify_release.py -v
```

That suite includes offline Cargo fixtures and can require cached registry/toolchain prerequisites. A test failure still fails the gate; do not silently substitute a mock.

## Changes that invalidate evidence

Production source, features, manifests, locks, build scripts, compiler flags, generated bindings/binaries, export maps and package payload changes can invalidate their matching source/artifact receipt. Follow each package's actual hash inventory. Markdown-only changes may be outside the npm WASM snapshot, but can still change a source archive. Recompute packaging evidence for the bytes actually distributed.

## Coordination

Use private Ruflo task/run records for ownership and acceptance notes when requested. Only close a task after checking its artifacts. ruOS E2E commands must run in a task-owned directory with bounded duration and exact artifact hashes. Do not treat a successfully spawned process, a created team, or a model's claimed result as a verified release.
