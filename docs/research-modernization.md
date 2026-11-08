# Research and modernization decisions

Reviewed 2026-11-07 UTC against the checked-out source and the primary sources linked below. This is an engineering adoption review, not a reproduced state-of-the-art benchmark. Research results from a different model, dataset, runtime, or machine are not ruv-FANN results. Package implementation and test outcomes are recorded separately in [the modernization report](modernization.md).

## What to improve first

The useful near-term changes are reliable inputs, correct mathematics, and a cheaper measured inference path. Renaming a model or enabling a feature is not evidence that an algorithm is implemented. A successful package inventory is not a successful build.

| Area | Decision for this implementation loop | Evidence required |
| --- | --- | --- |
| Small-network CPU inference | Reuse layer values and offer caller-owned output storage before changing the numerical representation. | Same outputs as the existing forward pass, dimension-error tests, repeated-call behavior, allocation checks, and release-mode benchmarks. |
| CPU parallelism | Profile thread count and batch size for the actual workload; keep a sequential path for small jobs. | Latency and throughput at fixed model shape, input data, CPU, and thread budget. |
| SIMD and packed weights | Keep a scalar reference; verify dispatch and data layout before expanding acceleration. | Scalar-versus-vector comparisons, tails and unusual dimensions, runtime CPU detection, architecture-specific execution. |
| Model and training-data loading | Bound input bytes, decompressed bytes, declared dimensions, and structural relationships. | Oversize headers, compressed expansion, corrupt connection indices, truncated files, and valid round trips. |
| Forecasting | Fix exported functionality and numerical kernels; evaluate models with chronological splits. | Gradient checks, finite-output tests, unambiguous training cutoffs, a naive baseline, and external prediction fixtures. |
| CUDA/WebGPU | Reject unsupported syntax and invalid resource sizes explicitly; validate generated shaders. | Parser negative tests, shader validation, device limits, numerical comparison, and real-device execution before acceleration claims. |
| Agent transport and tooling | Favor bounded queues/messages and explicit process arguments. | Concurrent producers/consumers, overflow/decompression limits, path tests, and shell-metacharacter regression tests. |
| Release tooling | Enumerate every workspace, check each package, and preserve the exact failed command. | Locked dependencies, package contents, tests, audits with feed dates, and explicit hardware/platform gaps. |

## CPU inference: remove avoidable work, then profile

[Packrat, ICML 2025](https://proceedings.mlr.press/v267/bhardwaj25a.html), studies CPU DNN serving configurations that divide work among model instances, threads, and batches. Its results show that allocating every thread to one instance is not universally optimal. The paper's reported gains belong to its evaluated models and serving system.

The inference for this project is to expose a predictable per-network execution path, reduce intermediate allocation, and benchmark thread/batch combinations before adding a scheduler or parallelizing every neuron. These are choices derived from the source and workload, not a direct implementation of Packrat.

The [Rust architecture intrinsics documentation](https://doc.rust-lang.org/stable/std/arch/) requires both the correct architecture and supported CPU features before calling vendor intrinsics. Runtime feature detection permits a portable binary to select a specialized path. Merely compiling an AVX2 function does not make calls safe on an arbitrary x86 CPU.

Adoption requirements for a future packed or vectorized network backend:

1. Freeze a supported network representation, including sparse connections and bias ordering. Publicly mutable network structures need revalidation or cache invalidation before using cached weights.
2. Dispatch once per operation or prepared network where possible. Check shape arithmetic and buffer lengths outside unsafe kernels.
3. Preserve a scalar reference and test non-multiple vector widths, empty/rejected shapes, very small inputs, and extreme finite values. Define tolerances because reductions and fused operations can change rounding.
4. Run on each supported instruction set. A build on x86 does not verify ARM or WASM SIMD.
5. Benchmark conversion and packing costs separately from steady-state inference, then include them in the appropriate end-to-end workload.

[XNNPACK](https://github.com/google/XNNPACK) is a useful external comparison for optimized inference operators across ARM, x86, WebAssembly, and RISC-V. It is not a drop-in equivalent of ruv-FANN's generic, mutable network representation. An adapter would add a foreign interface, packing rules, build targets, and licensing obligations, and should be evaluated after the portable baseline is sound.

Quantization, AMX-specific kernels, and replacing the backend with XNNPACK are deferred. Adoption needs a model format for scales/zero points, calibration or quantization-aware training, accuracy budgets, and hardware-specific measurements. See [NVIDIA's quantization-aware training explanation](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/) for why lower precision involves model-dependent scales and accuracy evaluation. Its TensorRT behavior is not a contract for a future Rust backend.

## Serialization and resource limits

The existing [deserialization ADR](adr/ADR-001-deserialization-safety.md) correctly identifies allocation and decompression as trust boundaries. A file-length limit alone is insufficient: a small header can declare a huge vector, decompression can expand the data, and a syntactically valid model can still contain impossible connection indices.

[Read::take](https://doc.rust-lang.org/std/io/trait.Read.html#method.take) bounds bytes read from a reader. It does not bound all allocations performed by a parser. Where the caller must distinguish exact-limit input from excess data, read at most `limit + 1` with checked/saturating arithmetic and reject an oversized result. Apply the decompressed limit after the decoder as well as any compressed-input limit before it.

The project should keep separate limits for wire bytes, expanded bytes, layers, neurons, connections, and training-data cells. Compute products and sums with checked arithmetic before allocating. Validate a connection's source against the actual preceding layer, and validate model shape before execution. A byte budget is not a hard process-memory ceiling; container/process memory limits remain useful for adversarial workloads.

[RUSTSEC-2025-0141](https://rustsec.org/advisories/RUSTSEC-2025-0141.html), issued January 7, 2026 and modified January 16, 2026, classifies bincode as unmaintained and lists no patched version. This is a maintenance advisory; it does not by itself prove an exploitable deserialization defect in a particular call site.

A migration should choose a maintained format, add an explicit format/version identifier, preserve bounded decoding, and test historical fixtures. Existing bincode readers and writers must agree on integer encoding, trailing-byte handling, and size limits. A bare dependency upgrade is not a compatible format migration. Until replacement, record the maintenance risk and validate the actual decode paths rather than describing an ignored advisory as resolved.

## Forecasting: test the algorithm and the evaluation

[PatchTST, ICLR 2023](https://arxiv.org/abs/2211.14730), combines time-series patches with channel-independent processing. Its architectural choices can reduce attention sequence length and share weights across channels. A generic RNN or an alias named after another architecture does not implement this method.

Current external reference points extend beyond the older list of forecasting architectures:

- [Chronos-2, October 2025](https://arxiv.org/abs/2510.15821), describes pretrained univariate, multivariate, and covariate-aware forecasting using group attention. Its benchmark results are the authors' results, not locally reproduced here.
- [Google's TimesFM repository](https://github.com/google-research/timesfm), checked on the review date, identifies TimesFM 3.0 as the current version and records its August 2026 release with multivariate and covariate support. TimesFM 2.5 is an archived version in that repository.
- [NeuralForecast's cross-validation guide](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/cross_validation.html) explains historical cutoffs, forecast horizons, step sizes, and refitting. Those semantics are a better compatibility target than matching method names alone.

These are evaluation candidates. This implementation loop does not claim to reproduce their pretrained weights, training pipelines, or leaderboard results. A future integration needs a supported checkpoint/runtime, explicit license review of that checkpoint, deterministic input/output mapping, memory and latency budgets, and reproducible evaluation on held-out series.

Before adding another named architecture, establish:

1. **Chronological splits:** training and fitted transforms only see observations available at the cutoff. Known-future covariates must be distinguished from unknown-future values.
2. **Declared evaluation:** dataset revision, series identifiers, frequency, horizon, cutoffs, refit behavior, seeds, missing-value policy, and task metric. Record which pretrained data may overlap evaluation data.
3. **Useful baselines:** seasonal naive and a small exported model, using identical splits. Define behavior when the denominator of a scaled metric is zero.
4. **Both prediction and cost:** accuracy per task, aggregate method, uncertainty where appropriate, wall-clock time, peak memory, and hardware. Include preprocessing and loading in a separately reported end-to-end measurement.
5. **Compatibility fixtures:** small inputs and expected outputs from a pinned external implementation. Test loss functions, gradients, dates, scaling, and error behavior as well as output shape.

The [FEV evaluation library](https://github.com/autogluon/fev) provides a candidate external task representation for future forecasting comparisons. Bringing it into a development evaluation environment does not require making it a runtime Rust dependency.

## WebGPU: validate before advertising acceleration

The [WebGPU Candidate Recommendation Draft of September 15, 2026](https://www.w3.org/TR/2026/CRD-webgpu-20260915/) defines device features, resource limits, validation, and error behavior. Optional capabilities must be negotiated. Buffer sizes and dispatch dimensions need validation against the created device's limits. Shader out-of-bounds behavior must not be relied on to repair an invalid computation.

For generated WGSL, [Naga's validator](https://wgpu.rs/doc/naga/valid/index.html) provides an important static gate after parsing. Static shader validity is separate from numerical correctness, workgroup synchronization, driver behavior, and real-device performance. Compare CPU and GPU outputs on representative and boundary shapes. Report host-to-device upload, dispatch, synchronization, and readback costs, including cold-start pipeline creation separately.

The registry observations below show that the existing `wgpu = "0.19"` dependency is far behind the current stable series. Updating to a new major is an API and backend migration. Treat it as a separate hardware-validated change, with a fallback and rollback plan; a successful CPU build cannot certify it.

## Agent tools and concurrency

The [Node child-process API](https://nodejs.org/api/child_process.html) distinguishes `exec`, which uses a shell, from `execFile`, which avoids a shell by default. Prefer a fixed executable and an argument array for user-controlled paths or identifiers; validate values even when no shell is involved. Add tests containing spaces and shell metacharacters, and avoid placing secrets in command-line arguments.

For bounded queues, the capacity check and reservation/update must form one synchronization operation. Making individual counters atomic does not make a multi-step queue algorithm correct. A mutex-based implementation with clear ownership can be preferable to an unverified lock-free design. [Loom](https://github.com/tokio-rs/loom) is a candidate for future schedule exploration; [Miri](https://github.com/rust-lang/miri) can detect classes of undefined behavior in supported Rust executions. Neither tool proves every schedule or backend safe. This review does not claim those tools were run unless the implementation report records their commands and results.

## Registry observations and release boundaries

The following stable-version metadata was retrieved directly from `https://crates.io/api/v1/crates/<name>` on 2026-11-07 UTC, with TLS verification enabled. These are observations, not instructions to migrate every major version.

| Dependency | Registry `max_stable_version` | Registry `updated_at` |
| --- | --- | --- |
| `wgpu` | `30.0.1` | `2026-08-22T02:53:38.229945Z` |
| `rayon` | `1.12.0` | `2026-04-14T00:56:56.454086Z` |
| `serde` | `1.0.229` | `2026-07-18T23:05:13.266456Z` |
| `rand` | `0.10.3` | `2026-09-20T06:37:47.850624Z` |
| `bincode` | `3.0.0` | `2025-12-16T21:34:14.694603Z` |

A registry's highest version is not evidence of active maintenance, compatibility, security, or fitness for this project's minimum Rust version. In particular, the bincode advisory still requires attention. Resolve and audit each workspace lockfile separately, record the advisory database timestamp, and inspect package contents before publishing. The root lockfile does not cover every subworkspace or npm distribution.

Numerical changes, serialization changes, and newly rejected invalid inputs should appear in release notes. Assign package versions after the intended behavior and package dependencies are verified. A prepared checkout or local package archive is not a published release.

## Documentation and diagrams

The root README now separates installed releases from checkout APIs, maps all five workspaces, gives an explicit-weight core example, and explains feature limits. It removes unsupported benchmark, architecture-count, Python-parity, and universal safety claims.

The [SVG diagrams](assets/) use local vector elements and CSS animation with a `prefers-reduced-motion` fallback. They have a title and description, and the README supplies text explaining their content. The [W3C complex-image guidance](https://www.w3.org/WAI/tutorials/images/complex/) recommends both a short alternative and a textual explanation of the essential information. Animation is decorative: all labels and relationships remain visible when it is disabled.

Documentation validation executed in this review:

- `python3 scripts/verify-packages.py --docs`: passed after the linked modernization report was created. This validates local README links and SVG source; it does not compile Rust examples.
- The sole Rust block in the root README was extracted to a temporary file, compiled with Rust 1.99.1 against the built local `ruv_fann` library, and executed. Its assertion passed and it printed `Output: 0.1`.
- All four SVG source files parsed as XML and passed checks for title/description, reduced-motion styling, and absence of scripts, foreign objects, and external resource URLs.
- Chromium/Playwright rendered both root README diagrams. Computed-style checks with reduced motion enabled disabled the animation on all six inference-diagram elements and the package-map trace. Screenshots were visually checked for layout and legibility.
- `git diff --check` passed for the owned README, research, diagram, and ADR changes. These documentation checks do not replace package tests or hardware validation.
