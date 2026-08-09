# Candle / WASM Plan for the ruv-FANN GEMV Rewrite

Primary-source analysis: candle source (shallow clone of huggingface/candle, 2026-08-09),
local builds of ruv-fann for wasm32, and a micro-benchmark of GEMV kernel candidates run
natively and under Node 22's wasm engine. All numbers below were measured on this machine
(x86_64 Linux, rustc 1.94.1); anything not measured is labeled *estimate*.

---

## 1. Candle architecture findings (from source)

### CPU matmul path

- `candle-core/src/cpu_backend/mod.rs` (`struct MatMul`, `impl Map2 for MatMul`, ~line
  1403-1530): the default CPU matmul (no mkl/accelerate features) calls the **`gemm`
  crate** (`gemm = "0.19"`, workspace dep in `candle/Cargo.toml` line 63) once per batch
  step, with `Parallelism::Rayon(num_threads)` when `get_num_threads() > 1`, else
  `Parallelism::None`. Threading is delegated entirely to the gemm crate; candle itself
  does no blocking/tiling for matmul.
- Per-call overhead visible in the same function: a fresh `vec![T::zero(); b*m*n]`
  allocation for the destination on **every** matmul call, plus stride/layout analysis
  (`lhs_l.stride()`, `ab_skip`, transpose-detection branches) before the kernel runs.
  Above that, every `Tensor::matmul` goes through storage-enum dispatch and (for
  variables) backprop-op recording in `tensor.rs`/`storage.rs`. For a [784,128,64,10]
  network that is 3 heap allocations + dispatch layers per forward pass, on ops that
  measure 58 ns–8 µs of actual math (see §4). Overhead share at the 10x64 layer is
  material; *estimate*: tens of percent for the smallest layers, based on the per-call
  alloc + dispatch structure, not measured end-to-end.
- Batch-shape folding: `MatMul::f` folds `(b,m,n,k)` into a single bigger GEMM when
  strides allow (`b_skip == 0 && a_skip == m*k` → `(1, b*m, n, k)`), which is how dense
  f32 layers with batched inputs get executed as one gemm call.

### Hand-written SIMD (separate from matmul)

- `candle-core/src/cpu/mod.rs` defines a `trait Cpu` (STEP/EPR, `vec_fma`, `vec_reduce`)
  with per-arch impls: `cpu/avx.rs`, `cpu/neon.rs`, and **`cpu/simd128.rs`** (wasm:
  `core::arch::wasm32`, `v128`, `f32x4_mul/add`, STEP=16 i.e. 4×f32x4 accumulators).
  These feed `cpu/kernels.rs` vec_dot-style kernels used by quantized and conv paths —
  **not** the dense matmul path, which stays on the gemm crate.

### WASM story

- `candle/Cargo.toml` line 63: `gemm = { version = "0.19", features = ["wasm-simd128-enable"] }`
  — candle compiles gemm's simd128 kernels in unconditionally (the feature is a no-op
  off-wasm).
- SIMD128 is enabled per-example via rustflags, e.g.
  `candle-wasm-examples/quant-qwen3/.cargo/config.toml`:
  `rustflags = ['--cfg', 'getrandom_backend="wasm_js"', '-C', 'target-feature=+simd128']`
  — exactly the pattern ruv-FANN already has for the getrandom half.
- Threading: none on wasm. The wasm examples run inference in a single web worker
  (`llama2cWorker.js` etc.); rayon compiles for wasm32-unknown-unknown but degrades to
  the current thread. `get_num_threads()` (`candle-core/src/utils.rs` line 343) just
  reads rayon's count. No wasm-threads/SharedArrayBuffer build is provided.
- `candle-core/Cargo.toml` cfg-gates only `tokenizers` and `candle-ug` off wasm32.

### Minimum dependency footprint as an optional backend

`candle-core`'s **non-optional** deps: gemm, half, float8, byteorder, libc, libm,
memmap2, num-traits, num_cpus, rand, rand_distr, rayon, safetensors, thiserror, yoke,
zerocopy, zip. The gemm crate alone resolves to **51 crates** in a fresh project
(measured via `cargo tree` — gemm-{common,f16,f32,f64,c32,c64}, dyn-stack, bytemuck +
three proc-macro stacks). candle-core would be substantially more (*estimate*: 80+
crates) and drags safetensors/zip/memmap2 that ruv-FANN has no use for.

---

## 2. ruv-FANN wasm32 build status (verified)

Commands run from the repo root, rustc 1.94.1, after `rustup target add
wasm32-unknown-unknown`:

| Command | Result |
|---|---|
| `cargo check --target wasm32-unknown-unknown --no-default-features --features wasm` | **PASS** (24.3 s cold) |
| `cargo check --target wasm32-unknown-unknown` (default features, incl. rayon/parallel, flate2, bincode) | **PASS** |

So the root crate already builds for wasm both ways; the getrandom 0.3 `wasm_js`
backend cfg in `.cargo/config.toml` + the `[target.'cfg(...wasm32...)'] getrandom`
dep in `Cargo.toml` (lines 100-104) resolved the previous blocker.

**SIMD128 is not plumbed anywhere.** `.cargo/config.toml`'s
`[target.wasm32-unknown-unknown]` rustflags contain only the getrandom cfg — no
`-C target-feature=+simd128`. `grep -rn "simd128\|target_feature" src/` finds nothing;
`src/simd/mod.rs` (718 lines) is x86-only (`is_x86_feature_detected!("avx2")`, AVX2
`matmul_avx2`/`matvec_avx2`/`add_bias_avx2` intrinsics) and — per the baseline report —
not wired into the scalar forward path.

---

## 3. Micro-benchmark evidence

Throwaway crate (scratchpad, not in repo): four GEMV kernels over the
[784,128,64,10]-relevant shapes, 20k reps native / 5k reps wasm, checksums identical
across kernels. Kernels:

- **aos**: replica of `Neuron::weighted_sum` — per-neuron `Vec<Connection{from,weight}>`,
  gather via `inputs.get(c.from)` (`src/neuron.rs`).
- **soa naive**: contiguous row-major `Vec<f32>`, `chunks_exact(k)` + single-accumulator dot.
- **soa 4-acc**: same layout, 4 independent accumulators via `chunks_exact(4)`.
- **gemm crate**: `gemm::gemm(m, 1, k, ...)`, `Parallelism::None` (wasm build with
  `wasm-simd128-enable`).

### Native (x86_64, `-O3 + lto`, gemm auto-dispatches AVX2/FMA)

| kernel | 128×784 ns/MAC | 64×128 | 10×64 |
|---|---|---|---|
| aos (ruv-fann style) | 0.707 | 0.814 | 0.492 |
| soa naive | 0.667 | 0.509 | 0.421 |
| soa 4-acc | **0.161** | 0.116 | 0.188 |
| gemm crate | **0.078** | 0.063 | 0.091 |

### wasm32 + simd128, Node 22 (v22.22.2)

| kernel | 128×784 ns/MAC | 64×128 | 10×64 |
|---|---|---|---|
| aos (ruv-fann style) | 0.735 | 0.690 | 0.682 |
| soa naive | 0.684 | 0.527 | 0.473 |
| soa 4-acc | **0.158** | **0.139** | **0.172** |
| gemm crate (simd128) | 0.361 | 0.370 | 0.386 |

Readings:

1. Layout alone (soa naive) buys little; **breaking the FP dependency chain** (4
   accumulators) is where the 4-5x comes from, and it autovectorizes on both targets
   with zero intrinsics and zero deps.
2. Native: gemm crate is a further ~2x over soa 4-acc (0.078 vs 0.161) via AVX2/FMA
   microkernels. The repo's existing `matvec_avx2` should recover most of that gap
   in-house (*estimate*, not yet benchmarked against gemm).
3. **wasm: the roles invert.** soa 4-acc (0.158) beats gemm-with-simd128 (0.361) by
   ~2.3x at these shapes — gemm's packing/blocking machinery costs more than it earns
   on tiny GEMVs under the wasm JIT.
4. The aos baseline here (~0.71 ns/MAC) is faster than the repo's measured 1.17 ns/MAC
   because the replica omits the generic `T: Float`, activation dispatch, and struct
   field write-backs of the real `Neuron::calculate`; real-world gains from the rewrite
   should therefore be **larger** than these ratios suggest.

---

## 4. Decision analysis

### Option A — in-house SoA/GEMV kernels (no new deps)

- **Mechanism**: contiguous row-major weights + multi-accumulator dot → autovectorized
  SIMD (SSE/AVX natively, f32x4 on wasm with `+simd128`); fused activation on the output
  write. Measured 4.4x over the AoS replica natively, 4.7x on wasm; vs the real
  `Neuron::calculate` path, *estimate* 5-8x.
- **wasm**: best-in-class per the benchmark; needs only one rustflags line.
- **Maintenance**: ~100-200 lines of safe stable Rust (chunks_exact, no intrinsics,
  no nightly `std::simd`); the existing `src/simd/mod.rs` AVX2 `matvec_avx2` can be
  wired behind the same trait for native f32 to chase the remaining 2x.
- **Go/no-go: GO** — primary recommendation.

### Option B — adopt the `gemm` crate for the inner product

- **Mechanism**: hand-tuned AVX2/FMA (and NEON/simd128) microkernels with runtime
  dispatch. Measured 9x over AoS natively — the fastest native option.
- **wasm**: compiles and runs (verified), but **loses to the simple in-house kernel by
  2.3x** at FANN-scale shapes.
- **Cost**: +51 crates, 3 proc-macro stacks, f16/c32/c64 kernels ruv-FANN never uses;
  no help for the generic `T: Float` API (f32/f64 only).
- **Go/no-go: NO for now** — revisit as an optional `native-gemm` feature only if
  Option A + AVX2 wiring leaves a measured native gap that matters.

### Option C — optional candle backend feature

- **Mechanism**: same gemm kernels as B, wrapped in Tensor machinery.
- **Cost**: candle-core's non-optional deps (safetensors, zip, memmap2, rayon, yoke…,
  *estimate* 80+ crates), per-op dst allocation + layout/dispatch overhead
  (`cpu_backend/mod.rs` `MatMul::f`) that is proportionally worst at exactly the tiny
  shapes ruv-FANN targets, and API impedance: candle's `Tensor`/`Device`/`DType` vs
  ruv-FANN's generic `T: Float` mutable-network API, no cascade-training analogue.
- **Buys**: a path to GPU (cuda/metal) and to candle-nn layers — none of which the
  GEMV bottleneck needs.
- **Go/no-go: NO** as a compute backend for the forward pass. Candle is worth
  *imitating* (its `.cargo/config.toml` wasm pattern, its 4-accumulator simd128 kernel
  shape in `cpu/simd128.rs`), not depending on.

---

## 5. Implementation sketch (Option A)

### Data layout

```rust
/// SoA forward-pass cache, one per layer transition. Rebuilt (or patched)
/// whenever connections/weights change; owned by Network, not serialized.
struct LayerGemv<T> {
    weights: Vec<T>,   // row-major [num_out_real][num_in_prev_incl_bias]
    out_dim: usize,    // non-bias neurons in this layer
    in_dim: usize,     // prev layer outputs incl. bias neuron(s)
    func: ActivationFunction, // uniform per layer (builder guarantees)
    steepness: T,
}
```

Bias is *not* a separate vector: FANN bias neurons already appear as a
constant-1.0 input in `prev_outputs`, so the bias weight is just the last column —
no change to network semantics, and `Layer::calculate`'s existing uniform-activation
detection (src/layer.rs line 176) carries over as the fast-path guard.

### Files that change

1. **`src/layer.rs`** — `calculate_uniform` gains a GEMV fast path: build/borrow the
   `LayerGemv`, run `gemv_fused(&w, prev, &mut out_buf)`, then scatter `out_buf` into
   `neuron.sum`/`neuron.value` (or, better, keep a flat per-layer output buffer and
   make `get_outputs` read it). Falls back to the existing per-neuron loop when
   connections are sparse/irregular (cascade networks) — detected once at cache build:
   the cache is only valid when every non-bias neuron connects to exactly the full
   previous layer in order, which is what `NetworkBuilder` produces.
2. **`src/network.rs`** — owns `Vec<Option<LayerGemv<T>>>`, invalidated by any
   weight/topology mutation (`set_weight`, cascade growth, training updates can write
   straight into the SoA buffer instead and sync back, phase 2).
3. **`src/simd/mod.rs`** — the portable kernel lives here as the default impl:

```rust
fn gemv_fused<T: Float>(g: &LayerGemv<T>, x: &[T], y: &mut [T]) {
    for (o, row) in y.iter_mut().zip(g.weights.chunks_exact(g.in_dim)) {
        let mut s = [T::zero(); 4];
        for (r, xx) in row.chunks_exact(4).zip(x.chunks_exact(4)) {
            s[0] = s[0] + r[0] * xx[0];  s[1] = s[1] + r[1] * xx[1];
            s[2] = s[2] + r[2] * xx[2];  s[3] = s[3] + r[3] * xx[3];
        }
        // + tail loop over in_dim % 4
        let sum = s[0] + s[1] + s[2] + s[3] /* + tail */;
        *o = apply_activation(g.func, g.steepness, sum); // fused: one pass, no revisit
    }
}
```

   For f32 on x86, route to the already-written `CpuSimdOps::matvec_avx2` (feature
   `simd`, runtime `is_x86_feature_detected!`), then apply activation over `y` — this
   finally wires the dormant AVX2 module into the forward path.
4. **`.cargo/config.toml`** — extend the existing wasm block:

```toml
[target.wasm32-unknown-unknown]
rustflags = ['--cfg', 'getrandom_backend="wasm_js"', '-C', 'target-feature=+simd128']
```

   (candle's exact pattern). No source changes needed for wasm SIMD: the 4-acc loop
   compiles to `f32x4` ops under `+simd128`, verified at 0.158 ns/MAC under Node 22.
   ruv-swarm's npm build scripts must pass the same flag when they build their own
   .wasm artifacts.

### Activation fusion

The activation is applied inside the same output-write loop as the dot product
(`apply_activation` is already variant-dispatched once per layer by
`calculate_uniform`), so each output element is touched exactly once — no separate
activation pass, no re-read of `sum` through `&mut self`.

### Verification plan

- Property test: SoA path vs existing per-neuron path bitwise-comparable per layer
  (same summation order within each 4-lane group differs — assert `approx::relative_eq`
  with tight epsilon instead of bitwise).
- Re-run `benches/neural_network.rs` before/after; target from evidence: >4x on the
  forward pass natively, similar on wasm (*the 5-15x plan estimate is plausible at the
  top end only once AVX2 is wired for f32*).
