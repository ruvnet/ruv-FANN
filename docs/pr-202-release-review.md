# PR #202 release review

Reviewed head: `1ff2043567e92e644133f3c40c1e088c20f5af6a` (2026-11-09).

## Findings and repairs

1. **High: the swarm Rust lock still selected vulnerable OpenSSL 0.10.73.** The actual PR advisory job reports RUSTSEC-2026-0463 and fails. Updated only OpenSSL to 0.10.78 and its required openssl-sys supplier; the local swarm audit now reports zero vulnerabilities. Unmaintained-package warnings remain distinct from vulnerabilities.
2. **High: the supported Node 22 clean install fails.** `npm ci --ignore-scripts --prefix ruv-swarm/npm` reproduces the missing minimatch/brace-expansion/balanced-match error. Simply regenerating the lock introduces vulnerable minimatch 10.2.6 below nyc/test-exclude. Explicitly overriding test-exclude's minimatch in both installation roots produces consistent locks and zero npm audit vulnerabilities.
3. **Medium: the new SQLite Node 24 evidence gate rejects successful tests.** It parses TAP summary lines but lets Node choose its reporter. Both focused and security stages now explicitly use TAP; the workflow contract requires those flags.
4. **Medium: release verification omitted the new public Candle crate.** Added the independent Candle workspace to the ordered verifier with the same verified-root archive overlay as the existing dependent families. The verifier now covers 27 candidates. Actual isolated root and Candle archive verification passed.
5. **Distribution limitation: the new napi package is a local platform-specific build.** Its package contains one host binary, its loader checks platform/architecture, and its source build scripts are not included in the npm payload. It is tested as an extracted local consumer but excluded from this public npm release. Portable prebuilt distribution is separate work.

The review covered dense model validation, forward/reverse indexing and derivatives, optimizer atomic commits and revision binding, allocation bounds, dense tiling arithmetic, Candle strided tensor export, raw Node array copying, WASM training, SQLite source/binary receipts and installation, pooled/async storage, MCP transport changes, and release workflows. Targeted training and boundary tests passed; this is not a claim of exhaustive formal verification.

A rejected host argument before entering the core may retain the preceding successful gradients, while a failed core backward pass invalidates them. This is existing behavior of the new bridge implementations, not used as release proof; callers should step only after a successful backward call.

## Validation performed during review

- 28 targeted Rust release tests: autograd, allocation, dense import and dense kernels; no failures or ignored tests.
- 104 Node security/runtime tests; no failures, skips or todos.
- 18 direct native gateway/training/preflight tests; no failures, skips or todos.
- Extracted native consumer: actual addon, CJS/ESM, ownership, lifecycle and training passed.
- Fresh swarm archive installation: patched SQLite 3.54.1 and durable CLI state passed.
- Extracted core/SIMD training archive: four actual Candle comparison cases passed.
- 103 workflow tests and 55 release/tooling tests passed.
- Seven-graph audits and exact final-SHA CI are required again by the protected publisher; earlier PR receipts do not authorize the revised candidate.

WASM was rebuilt with Rust 1.88.0 and wasm-pack 0.15.1 after the lock change. Updated source/artifact receipts are committed with the candidate. Public publication status belongs to the protected workflow's downloadable receipts, not this pre-publication review.

## Publication

`protected-release.yml` requires a manually supplied exact main SHA, no labeled release blockers, and a successful complete CI run. It uploads the normally verified Cargo archives without repacking them using the [Cargo registry publish protocol](https://doc.rust-lang.org/cargo/reference/registry-web-api.html#publish). npm publication uses the archives exercised by fresh SQLite/training consumers. Both registries are downloaded again and compared byte-for-byte; existing versions with different bytes fail closed. Public npm and Cargo consumers run after publication. Failure receipts preserve partial-publication state rather than claiming success.

Credentials are supplied only to upload steps. Consumer lifecycle tests run without registry publishing credentials. GCP-managed credentials are injected into the repository's encrypted Actions secrets; values are never committed or printed.
