# SOTA Deep-Research: Adversarially Verified Claims (2025–2026)

**Date:** 2026-08-09
**Method:** 105-agent research workflow — 5 parallel search angles, 15 primary
sources fetched, falsifiable claims extracted, each claim subjected to 3-vote
adversarial verification (2/3 refutations kill a claim). 23 raw claims survived,
merged into the 9 findings below (all high-confidence, unanimous 3-0 votes).
Claims that *failed* verification are listed at the end — they matter as much as
the survivors.

**Consumer:** the ruv-FANN SoA/GEMV rewrite (shipped in `src/soa.rs`, measured
6.5x on MNIST-sized forward; see `docs/research/09` for the kernel bake-off and
PR #193 for the implementation).

## A. Verified findings

### 1. Ecosystem convergence validates the SoA/GEMV rewrite (3-0)
`matrixmultiply`, candle, and tract all structure CPU inference around packed,
cache-blocked, register-tiled BLIS/Goto-style kernels on contiguous panels —
none use per-neuron AoS gather. **Scoping caveat from the verifiers:** BLIS-style
packing pays off for *batched* GEMM, not batch-1; for single-sample forward
passes the right design is exactly the simpler SoA row-major GEMV we shipped,
with packed microkernels reserved for a future batched path.
Sources: docs.rs/matrixmultiply, github.com/bluss/matrixmultiply, sonos/tract
`linalg/README.md`, huggingface/candle.

### 2. Runtime AVX2/AVX-512 dispatch is production-standard and nearly free (3-0)
The `multiversion` crate (or hand-rolled `is_x86_feature_detected!`, as
`matrixmultiply` does) compiles per-target variants and dispatches via a cached
atomic load — negligible vs a ~109K-MAC forward pass. Constraints verified:
requires `std` (wasm32/no_std get compile-time features only); dispatch must
wrap a **whole-layer or whole-network free function**, not the inner loop
(dispatch blocks inlining). AVX-512 target features stable since Rust 1.89.

### 3. `std::simd` cannot anchor a stable-Rust SIMD plan (3-0)
Nightly-only (`portable_simd` #86656); empirically reproduced failing on stable
1.94.1 (E0658). Stable alternatives: `std::arch` intrinsics behind runtime
detection (our existing `src/simd` approach), or the `wide`/`pulp` crates.

### 4. WASM Relaxed SIMD is stable-Rust-usable for NN kernels (3-0)
`relaxed-simd` intrinsics stabilized in Rust 1.82 — including
`f32x4_relaxed_madd` (the GEMV inner-loop primitive: 4 f32 MACs/instruction) and
i8 dot products for quantized inference. Verified compiling on stable 1.94.1
for wasm32. **Caveat:** relaxed-madd rounding is implementation-defined — test
tolerances must allow ~1-ulp divergence across runtimes.

### 5. tract's published relaxed-SIMD numbers + the dual-build deployment rule (3-0)
tract's WASM kernels measure **1.40–1.55x kernel-level / 1.08–1.46x end-to-end**
over plain SIMD128 (maintainer-published, `linalg/WASM_RELAXED_SIMD.md`).
Critically: relaxed SIMD **cannot be feature-detected in-process** — a module
containing `f32x4.relaxed_madd` fails *instantiation* on unsupporting hosts. A
wasm deployment must ship **dual builds** (simd128 baseline + relaxed) selected
at load time (e.g. `wasm-feature-detect`), never runtime-dispatched in one
binary.

### 6. SOTA kernel selection is size-aware, not just ISA-aware (3-0)
tract scores candidate kernels by `scale × m_util × n_util` (tile utilization)
and fits per-CPU analytic cost models — because wide tiles go underutilized on
small layers exactly like `[784,128,64,10]`. A credible 2026 FANN-lineage
library picks tile shapes by utilization for small matrices rather than
hardcoding one kernel.

### 7. Avoid memory64 on wasm (3-0)
First-party engine data (SpiderMonkey): 64-bit wasm runs **10–100%+ slower**
because the 4GB virtual-reservation trick that elides bounds checks is
impossible — every load/store pays an explicit check, which lands directly on
memory-dense GEMV loops. Micro-net weights are KB–MB; stay on wasm32.

### 8. Candle proves the single-codebase native+wasm architecture (3-0)
Default CPU backend = pure-Rust `gemm` + rayon (vendor BLAS strictly opt-in);
wasm32 builds run real models with the same math deps, `simd128` module
structurally parallel to `avx2`. **Caveat:** rayon degrades to single-threaded
on wasm32-unknown-unknown — wasm parity is structural, not performance parity,
without explicit wasm-threads plumbing.

### 9. `matrixmultiply` 0.3.11 already ships AVX-512 + wasm SIMD128 sgemm (3-0)
Published 2026-07-14, small pure-Rust crate (ndarray's GEMM backend). For a
future *batched* path, depend on it or copy its kernel structure rather than
writing AVX-512/simd128 microkernels from scratch.

## B. Claims that did NOT survive verification

- **No evolutionary/self-optimizing harness claims survived** (GEPA,
  AlphaEvolve-style, gradient-free config evolution beating static configs).
  Quantitative expectations for that track remain unvalidated in the public
  record — which independently corroborates our own twice-replicated null
  result in `docs/research/04`–`05`: on a well-tuned config, honest gates
  promote nothing.
- **The widely-cited TF.js "1.7–4.5x WASM-SIMD" baseline was refuted** —
  published small-dense-network speedup numbers are unreliable; trust only
  benchmarks run on the target workload (as `09` did).

## C. Implications for ruv-FANN — done vs next

| Finding | Status in this branch |
|---|---|
| Batch-1 SoA GEMV, not BLIS packing (1) | ✅ shipped (`src/soa.rs`, 6.5x measured) |
| simd128 for wasm32 (4, 8) | ✅ rustflags added; `cargo check` green |
| Own-workload benchmarking over published claims (B) | ✅ practiced throughout (`07`, `09`) |
| Whole-layer runtime AVX2/AVX-512 dispatch via `multiversion` (2) | ▢ next: wrap `GemvCache::run_layer` |
| Batched path via `matrixmultiply` to fix `run_batch`'s zero amortization (1, 9) | ▢ next |
| Relaxed-SIMD dual-build wasm artifacts + load-time selection (4, 5) | ▢ next; requires npm packaging change |
| Size-aware kernel/tile selection (6) | ▢ later, after ≥2 kernels exist |
| memory64 (7) | ✅ correctly avoided (nothing to do) |
| wasm threads (8) | ▢ later; sequential fallback acceptable for micro-nets |

Full per-claim sources and evidence quotes are preserved in the workflow
artifacts; the summary above cites the primary source for each finding.
