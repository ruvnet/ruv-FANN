# 07 — Benchmark Baseline: root `ruv-fann` crate

Measured performance baseline for the root neural-network crate, intended as the
"before" reference for optimization work.

## Environment

| Item | Value |
|---|---|
| Commit (source state) | `c1088ba27a133e02b9f624cc7d525859bb5a9317` (HEAD moved to `b946779` mid-run, diff touched only `.claude-flow/` state files — `src/`, `benches/`, `Cargo.toml` identical) |
| Date | 2026-08-08 (UTC) |
| CPU | Intel(R) Xeon(R) Processor @ 2.10 GHz, 4 vCPUs, 1 thread/core |
| Caches | L1d 4x48 KiB, L2 4x2 MiB, L3 260 MiB (shared/virtualized) |
| RAM | 15 GiB |
| Toolchain | rustc 1.94.1, cargo 1.94.1, `bench` profile (optimized) |
| Criterion | 0.5 (`html_reports` feature), plotters backend |
| Crate features | defaults: `std, serde, parallel, binary, compression, logging, io` |

**Caveat:** this is a shared, noisy container (sibling agents were compiling and
editing concurrently on the same 4 vCPUs). Numbers are indicative, not
lab-grade; treat them as order-of-magnitude anchors and re-measure deltas on
the same box. Criterion flagged occasional "high mild" outliers, consistent
with background load.

## Bench inventory

- **Root crate** (`Cargo.toml` `[[bench]]`): one target, `benches/neural_network.rs`,
  8 criterion groups / 31 individual benchmarks. **All 31 ran; zero compile
  failures, zero panics.**
- **ruv-swarm** (not run, out of scope):
  - Registered targets: `ruv-swarm-core/benches/swarm_benchmarks`,
    `ruv-swarm-transport/benches/transport_benchmarks`,
    `swe-bench-adapter/benches/swe_bench_benchmarks`, plus the `benchmarking`
    workspace member crate.
  - **Orphaned files:** `ruv-swarm/benches/{agent_spawn,message_passing,orchestration,wasm}_bench.rs`
    (1,228 lines) sit under the *virtual* workspace root, which has no
    `[package]` — they are not registered to any crate and cannot be run via
    `cargo bench` as-is.

## Method

Single bounded run, serialized against sibling compile jobs:

```bash
cargo bench --bench neural_network -- --noplot --sample-size 10 --measurement-time 2 --warm-up-time 1
```

- 10 samples/bench, 2 s measurement window, 1 s warmup (the `training`,
  `training_algorithms`, and `cascade_network` groups also hard-code
  `sample_size(10)` in the bench source).
- Total wall time ≈ 3 min run + 40 s compile.
- Tables below report criterion's point estimate (middle value of the
  `[low estimate high]` triple). With n=10 confidence intervals are wide;
  differences under ~10% are noise here.

## Results

### network_creation (`Network::<f32>::new`)

| Layers | Time (est.) |
|---|---|
| [2, 3, 1] | 306 ns |
| [10, 20, 10, 5] | 10.26 µs |
| [100, 50, 25, 10] | 90.0 µs |
| [784, 128, 64, 10] (MNIST-sized) | **2.37 ms** |

~22 ns per connection at MNIST scale — creation cost is dominated by per-neuron
`Vec<Connection>` heap allocations.

### forward_propagation (`Network::run`, single input)

| Network | Time (est.) | Derived throughput |
|---|---|---|
| XOR-sized [2,3,1] | 108 ns | 9.3 M inf/s |
| Small [10,20,10] | 845 ns | 1.18 M inf/s |
| Medium [100,50,25,10] | 7.28 µs | 137 k inf/s |
| MNIST-sized [784,128,64,10] | **127.3 µs** | **7.9 k inf/s** |

MNIST-sized net ≈ 109 k MACs → **~1.17 ns/MAC ≈ 0.86 GMAC/s (~1.7 GFLOP/s)
single-thread** — plain scalar territory; a vectorized GEMV on this CPU class
would be roughly an order of magnitude faster.

### training

| Benchmark | Time (est.) |
|---|---|
| XOR_100_epochs ([2,3,1], 4 samples) | 115.1 µs (≈1.15 µs/epoch) |
| Large_dataset_50_epochs ([2,10,5,1], 100 samples) | 4.34 ms |

### training_algorithms (100 epochs, XOR, [2,4,1])

| Algorithm | Time (est.) |
|---|---|
| IncrementalBackprop | 293.8 µs |
| BatchBackprop | 291.6 µs |
| Adam | 316.9 µs |
| Quickprop | 334.4 µs |
| Rprop | 344.4 µs |
| AdamW | 353.3 µs |

Spread across six very different algorithms is only ~20%, suggesting shared
overhead (allocation, dispatch, network traversal) dominates over the actual
optimizer math at this size.

### batch_processing (`Network::run_batch`, [10,20,10,5])

| Batch size | Total (est.) | Per item |
|---|---|---|
| 1 | 926 ns | 926 ns |
| 10 | 9.57 µs | 957 ns |
| 32 | 29.5 µs | 922 ns |
| 64 | 61.5 µs | 961 ns |
| 128 | 122.0 µs | 953 ns |
| 256 | 238.5 µs | 932 ns |

**Perfectly flat per-item cost — batching yields zero amortization** (see
hotspot #2).

### activation_functions ([10,20,10] forward pass)

| Activation | Time (est.) |
|---|---|
| LeakyReLU | 744 ns |
| ReLU | 770 ns |
| Linear | 790 ns |
| Sigmoid | 803 ns |
| Tanh | 908 ns |

Only ~20% spread between `Linear` and `Tanh`: the pass is bound by memory
traversal and per-neuron dispatch, not activation math.

### weight_operations ([100,50,25,10]) and cascade

| Benchmark | Time (est.) |
|---|---|
| get_weights | 8.36 µs |
| set_weights | 4.09 µs |
| randomize_weights | 18.6 µs |
| cascade_creation | 318 ns |

`get_weights` is 2x `set_weights` for the same traversal — it pushes into an
unreserved `Vec` (`src/network.rs`, `get_weights`), so it pays reallocation
copies.

## README claims vs. measurements

- The headline README table (84.8% SWE-Bench, 3,800 tasks/sec, token/memory
  savings) describes **ruv-swarm agent orchestration**, not this crate — not
  measurable by these benches.
- "<100 ms decisions": consistent — even the MNIST-sized forward pass is
  127 µs, three orders of magnitude under that budget.
- "Blazing performance" (FANN rewrite): the crate is *functional and
  predictable*, but at ~1.7 GFLOP/s single-thread scalar it runs well below
  what this hardware can do for dense f32 layers. There is no measured
  evidence here for a speed advantage over C FANN; the architecture (per-neuron
  connection lists) mirrors classic FANN rather than a BLAS-style layout.

## Hotspot analysis (targets for an optimizer)

1. **`Neuron::calculate` + AoS layout — the core inference path**
   (`src/neuron.rs:99`, called via `Layer::calculate`, `src/layer.rs:175`).
   Each connection does an indexed gather `inputs[connection.from_neuron]`
   with a bounds check, on an array-of-structs `Vec<Connection>` per neuron,
   then a `match` on the activation enum per neuron. Measured ~1.17 ns/MAC —
   scalar, cache-unfriendly, auto-vectorization-hostile. Flattening each
   fully-connected layer to a contiguous weight matrix + GEMV (or wiring the
   existing `src/simd` module, which already has rayon `par_iter` code at
   `src/simd/mod.rs:620`, into the scalar path) is the single biggest win —
   plausibly 5–15x on the Medium/MNIST-sized forward benches.

2. **`Network::run_batch` is a serial loop with no amortization**
   (`src/network.rs:429` — `inputs.iter().map(|i| self.run_unchecked(i))`).
   Per-item cost is identical at batch=1 and batch=256 (926 vs 932 ns).
   Despite the default `parallel` (rayon) feature, the batch path uses
   neither parallelism nor batched GEMM, and `Network::run`
   (`src/network.rs:114`) additionally allocates a fresh `Vec` per layer per
   call via `Layer::get_outputs` (`src/layer.rs:156`) plus the output
   `collect`. For the XOR-sized net (108 ns total) those allocations are a
   large fraction of the entire inference. Reusing ping-pong buffers +
   rayon-chunking batches is a cheap, high-leverage fix.

3. **Construction and weight-export overhead**: `Network::new` costs 2.37 ms
   for a MNIST-sized net (~22 ns/connection, one heap `Vec` per neuron), and
   `get_weights` (`src/network.rs:157`) doubles `set_weights`' cost by growing
   an unreserved `Vec`. Secondary, but `Network::new` sits inside the
   `training/*` benches' timed loops (a fresh net per iteration), so it also
   pollutes training numbers. A flat weight arena fixes this and #1 together.

Anomaly worth noting: all six training algorithms land within 20% of each
other — an optimizer should not chase per-algorithm math until the shared
traversal/allocation overhead above is removed, since it currently masks
algorithmic differences.

## Reproduction

```bash
git checkout c1088ba27a133e02b9f624cc7d525859bb5a9317
cd /home/user/ruv-FANN
cargo bench --bench neural_network -- --noplot --sample-size 10 --measurement-time 2 --warm-up-time 1
# Full-fidelity run (criterion defaults, ~30+ min): drop the flags after `--`
lscpu | grep "Model name"   # record CPU alongside results
```

Raw log for this run: criterion stdout with 31 `time:` triples, 0 errors
(sample-size 10, measurement-time 2 s, warm-up 1 s).
