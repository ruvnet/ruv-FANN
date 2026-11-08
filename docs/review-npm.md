# npm package review and implementation

Reviewed on 2026-11-07. No packages were published. This report covers the private root npm workspace, `ruv-swarm/npm`, its private installation fixture, and the CUDA Node distribution.

## Candidates and compatibility

| Package | Candidate | Supported contract |
| --- | --- | --- |
| Root npm workspace | Private | Installs the local swarm workspace and delegates build/tests |
| `ruv-swarm` | **2.0.0** | Local agent/task metadata, SQLite persistence, dense WASM inference, three statistical forecasts, measured local benchmarks, stdio MCP |
| Bundled `ruv-swarm-wasm` | 1.0.6, private | Genuine generated core and SIMD bindings; independent Rust version |
| Historical `@ruv/ruv-swarm-wasm` prototype | 1.0.5, private | Preserved source, excluded from the npm archive |
| `cuda-wasm` | **2.0.0** | Node CJS/ESM/CLI wrapper around an external `cuda2rust` >=0.2.0; generates Rust or WGSL source |

Swarm and the root workspace require **Node >=22.13.0**, matching maintained development-tool engines. CUDA has no npm dependencies and supports **Node >=22.12.0**. Official Node 22.13.0 and22.12.0 binary archive hashes were checked against nodejs.org SHASUMS before testing; Node 24.22.1 is also tested. SQLite uses a real native addon and can require a C++ compiler when no matching prebuild exists.

The previous npm `wasm-pack` development dependency unconditionally downloaded a GitHub executable during installation. Node 22 does not provide Node 24's optional environment-proxy behavior; its direct download failed in this proxy-only environment. It was removed from the install graph. Developers explicitly install `wasm-pack 0.15.1` through Cargo when rebuilding WASM. Normal npm consumers need neither Rust nor wasm-pack. Clean Node 22.13 and24 installs run actual lifecycle scripts and native SQLite successfully.

## Confirmed application fixes

| Boundary | Implemented behavior and meaningful regressions |
| --- | --- |
| Claude invocation | Shell-free argument arrays, prompt after `--`, normal permission checks by default; only exact `secure:false` or an explicit CLI option enables bypass. Omitted/true/false, flag-shaped literal prompts, generated bash helpers and actual recorded child argv are tested. The old Windows batch helper fails clearly because CMD cannot preserve arbitrary literal prompts safely. |
| GitHub and hooks | Argument arrays and standard input carry literal text. Repository/issue identifiers are checked. Coordinator and hooks load as ESM and await initialization. External accounts are not invoked by these tests. |
| SQL update fields | Task/neural column allowlists, explicit camelCase aliases, duplicate/unknown/ownership-field rejection and parameterized values. Native SQLite regressions check hostile update keys cannot alter another record. |
| SQLite lifetime and expiry | Per-connection WeakMap statement caches with 128-entry LRU, reader release in `finally`, observable initialization failure, native file-URL handling. Production expiry queries use `julianday` to compare timezone/millisecond timestamps correctly. Memory IDs use random UUIDs, avoiding same-millisecond replacement. |
| MCP framing and lifetime |1 MiB UTF-8 input/output frames,32 pending handlers,4 MiB bounded output queue, bounded IDs/method names and writer backpressure. Oversized input recovers at the next newline. Stdin EOF drains real pending replies before shutdown. All three shipped CLI variants are tested with actual inference through a pipe ending at EOF. |
| MCP metadata resources |64 swarms,1,000 agents and10,000 tasks per database; quota failures are explicit. Assignment state updates use SQLite transactions and idle-state checks, with WASM rollback on persistence failure. Monitoring samples aggregates, at most100 snapshots, rather than repeated full agent inventories. |
| WASM integrity | Hash and return the same read, removing the reopen replacement window. Package names reject traversal. Missing/corrupt generated bindings fail instead of returning placeholder predictions or unrelated memory. |
| Public facade | Uses actual generated `WasmSwarmOrchestrator`, separate core/SIMD instances, checked shapes/finite values, real memory, dense inference and linear/mean/naive forecasts. Assignment preserves real descriptions/capabilities. Unsupported external execution and dependency scheduling fail clearly. |
| Public CLI/MCP | Replaced random scores, fabricated completed tasks, fictional DAA/training and fake benchmark paths with the actual backend.17 advertised tools use real metadata, inference and measurements. Missing tasks and private helper names return JSON-RPC errors. Unknown argument keys are rejected. |

The 2.0 major communicates deliberate corrections to the 1.x public contract. `assigned` means matching idle agents were reserved; it does not mean an external model performed the work. There is no fabricated ETA, LLM benchmark score, training success or autonomous learning result. `DAAService` explicitly reports unsupported capabilities. The public export map excludes historical prototype deep imports. The CLI, pure inference operations, optional legacy files and mock examples are distinguished in the README.

The independent Rust WebSocket MCP test launches an explicit owned binary with a unique loopback listener and temporary database. It waits for the child's post-bind readiness message and health response; it cannot borrow an unrelated service on a colliding port. Cleanup closes the owned child group. Existing55 assertions and all four legacy suites are retained. They include a JavaScript neural simulation and8 mock API examples, so their success is not evidence of model accuracy or LLM execution.

## Dependency review

All audit results below concern complete lockfile graphs, including development tools. Audit query times are captured with the final receipts; npm does not expose a unique advisory-feed revision timestamp. Root, isolated standalone leaf, CUDA and historical installation-fixture locks are checked separately so workspace discovery cannot silently substitute another graph.

The original active graphs had 7 low and 9 moderate findings. Targeted UUID 11.1.1, NYC 18 and patched minimatch/glob resolutions closed the moderate findings without a blanket audit fix. The remaining seven low records were one real `deepmerge 4.3.1` advisory plus Jest/Rollup ancestors. Newer Jest/resolver majors still depended on the affected upstream 4.x package; incompatible suggested downgrades did not address the intended contract.

The maintained **`@stackline/deepmerge` 1.1.2** fork was inspected as a narrow drop-in replacement for Jest config and the Rollup node resolver. It has zero runtime dependencies, MIT licensing, CJS/ESM entry points and an explicit prototype-key defense. Its registry signature **and provenance attestation were verified** using `npm audit signatures`. Scoped overrides and a development alias select it only where needed. Actual tests reproduce the original target-prototype poisoning failure, verify the patched target/source/nested-array cases and symbol/array compatibility, run Jest preset merging, and bundle a bare package with the real Rollup resolver.

The primary advisory is **CVE-2026-93753 / GHSA-gcqx-w48m-w2h4**: published 2026-09-18T18:31:44Z, updated 2026-10-27T21:34:58Z, affected upstream versions >= 4.2.0 through 4.3.1, no upstream patched version recorded. The fork's 1.1.2 release is dated 2026-11-04. Original 4.3.1 failed the poisoning regression; the maintained replacement passes it and both actual tool integrations. The finding was development-only in this repository; no reachable production deepmerge route was claimed.

npm10 correctly reports the aliased override edges. npm11's `npm ls deepmerge` incorrectly flags the fixed alias 1.1.2 against the original requested 4.x range; clean installation, actual loaded package identity, signed registry evidence, runtime compatibility and audit 0 are checked independently. This diagnostic is not used to waive a vulnerability.

**Final result: zero findings across all four tracked npm locks, including development dependencies.** The historical missing 1.0.6 tarball fixture now locks the genuine local 2.0.0 archive. Its normal clean installation compiled the native SQLite addon, then passed real SQL, inference and forecasting smoke checks. Its former four findings (two high, two moderate) were remediated; no lockfile was excluded or finding suppressed. The dependency worker independently repeated all four audits and confirmed identical lock hashes.

Audit queries began at **2026-11-07T23:39:19.126038+00:00**. npm exposes no distinct feed-revision timestamp.

| Full graph | Findings | SHA-256 of tracked lockfile |
| --- | --- | --- |
| Private root | 0 | `088550774a049975671d2e0cc075b8be515c4c436f1fc543bee4256ebeba09a8` |
| Standalone swarm leaf | 0 | `e80d214dfd4f6cd0068ca5beb43de8b028594fd4c52f133c82565c4c3996adf7` |
| Current archive installation fixture | 0 | `4ce197425ee37207814a6a99f78349dbc0f33602c5eebb4e7699b147dd66a56c` |
| CUDA | 0 | `1cfc5ae406d55538289943a4c417e1d39aa0f7ee620d1bade20920e37886e138` |

## Artifact provenance and release gates

`npm run build` invokes real wasm-pack using argument arrays, release mode, target `wasm32-unknown-unknown`, web bindings and Cargo `--locked`. Core uses ordinary flags; SIMD adds `-C target-feature=+simd128`. Generated bindings and declarations come from the compiled module. Optional legacy FANN/forecast binaries are excluded, and nested generated manifests are private.

`wasm/build-info.json` records each generated binary/binding/type/manifest hash, actual tool versions and separate core/SIMD configurations. The filesystem source snapshot includes **new untracked Rust modules**, shaders, build scripts, Cargo configuration/manifests/locks, JavaScript runtime/bin/type sources, root/leaf npm manifests/locks, validation/build recipes and license texts. A pre/post-build snapshot rejects concurrent source changes. Markdown remains independent. LF checkout policy keeps hashed text consistent across platforms.

Prepack requires a current checkout snapshot and matching artifact bytes. Installed `validate:package` verifies the archive without requiring a Rust checkout. Explicit `files` entries, licenses, exports and runtime entry points are checked. The receipt is local integrity/freshness evidence, not a signed external release attestation. Fresh source binaries, consumer installation, real APIs and types are required; a legacy binary cannot receive fabricated build evidence.

Final archives were packed with their real prepack validators. Earlier 1.0.24 archives and receipts are superseded.

| Artifact | Files | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `ruv-swarm@2.0.0` | 97 | 447622 | `4b6e8773dc64ad5b2ee48bc266c25329bf5b25dafbca1c6a783da1b18fc9806f` |
| `cuda-wasm@2.0.0` | 10 | 12954 | `b5f8e6c9c7e567faf4eadda6fa58350deb11ea6ba8fff502d430377f54cb9337` |

```text
Swarm build-source snapshot
9f5694dacb089f5006d439e7c9ecaf36789a9f44a7b0ee48cd7e09c42dbfa518
Core WASM
05acdbf47181e046a6a8d280d9c28101d1cd006106c785d07ca7e2d8bdbbb709
SIMD WASM
c97c9b6f8265d405e812d4df06ab379f1a20a4e7e80c4e74d386fbe9f1ef952e
Final source-built cuda2rust 0.2.0
e9b48240cd6b9f88be1cbcf03b694564481a0785fd1072fcc72c448e9ee1bbf7
```

The source build used rustc 1.99.1 (58cf8f9c3 2026-10-13) and wasm-pack 0.15.1. Machine-readable receipts and actual archives are `/workspace/scratch/npm-swarm-receipt.json`, `/workspace/scratch/ruv-swarm-2.0.0.tgz`, `/workspace/scratch/cuda-node-receipt.json` and `/workspace/scratch/cuda-wasm-2.0.0.tgz`.

The CUDA 2.0 contract generates source through an external compiler; it does not advertise nonexistent browser/WASM artifacts, shader execution, analysis or benchmark modules. Both supported Node runtimes tested its actual final compiler and installed archive.

## Verification

| Gate | Current evidence |
| --- | --- |
| Clean Node 22.13 and24 root installs | Passed with normal lifecycle scripts; native SQLite3.49.2 |
| Default swarm tests, both Node versions | 55 actual security/API/runtime regressions + 3 actual tooling regressions + 8 explicitly labeled mock examples passed |
| Maintained release-code lint | Passed with zero warnings |
| Full legacy suites on both runtimes | 4/4 suites and 55/55 assertions passed (137.92s Node 22.13;137.86s Node 24, including cold compilation) |
| Fresh core/SIMD source receipt and prepack | Passed after final root and swarm source freezes |
| All four tracked npm lock audits | Zero findings, independently confirmed; actual current-archive fixture clean install passed |
| CUDA22.12/24 clean install,18 wrapper/security regressions, real compiler | Both passed; one additional genuine compiler integration test per runtime |
| Installed swarm archive Node 22.13/24 | Four real API tests, native SQLite/inference/forecast smoke, nine-frame MCP EOF/recovery/error smoke, CJS require and ESM, private-export rejection, validator and audit zero passed on each runtime |
| Installed CUDA archive Node 22.12/24 | Real compiler through CJS/ESM/CLI, validator and audit zero passed on each runtime |
| Installed archive types | Both packages pass TypeScript 5.8.3 strict NodeNext ESM+CJS with negative assertions; swarm uses normal Node 22.13 ambient types |

`quality:check` runs maintained-code lint, default tests, all four legacy suites, full low-severity audit and current-source artifact validation. Historical broad lint/test commands cover additional private prototypes and are not substituted for this defined release gate. The supported project setup now writes exact tool documentation and merges `.mcp.json`; it preserves custom servers and leaves existing permission rules/hooks unchanged. Malformed configuration is rejected without replacement. No external Claude/GitHub actions, registry publishing, commits or pushes were performed.

Linux clean-install and native ABI checks were executed here on both documented runtimes. macOS and Windows remain CI matrix gates; this local result does not claim those hosted jobs ran. All source/build inputs and generated assets are Git-visible, and `git diff --check` passes.

## Primary sources

- [Node child_process](https://nodejs.org/api/child_process.html): shell-free argv and platform semantics.
- [Node 22.13.0 release](https://nodejs.org/en/blog/release/v22.13.0) and [Node 22.12.0 release](https://nodejs.org/en/blog/release/v22.12.0): supported runtime baselines.
- [npm audit](https://docs.npmjs.com/cli/v11/commands/npm-audit/) and [registry signatures](https://docs.npmjs.com/about-registry-signatures): full graphs and signature/attestation verification.
- [npm files and exports](https://docs.npmjs.com/cli/v11/configuring-npm/package-json#files): publication contents.
- [wasm-pack build](https://rustwasm.github.io/docs/wasm-pack/commands/build.html): genuine generated binding targets.
- [Deepmerge advisory](https://github.com/advisories/GHSA-gcqx-w48m-w2h4), [maintained fork](https://github.com/alexandroit/stackline-deepmerge), [1.1.2 release](https://github.com/alexandroit/stackline-deepmerge/releases/tag/v1.1.2): reviewed source and patch provenance.
- [UUID bounds advisory](https://github.com/advisories/GHSA-w5hq-g745-h8pq), [minimatch amplification advisory](https://github.com/advisories/GHSA-g3hv-75wg-m249), [glob traversal advisory](https://github.com/advisories/GHSA-3g3m-2fjr-6pw4): targeted patched resolutions.
- [SQLite date/time functions](https://sqlite.org/lang_datefunc.html): timezone-aware Julian-day comparisons.

- [Claude Code project MCP configuration](https://code.claude.com/docs/en/mcp#project-scope): project `.mcp.json` and separation from permission settings.
