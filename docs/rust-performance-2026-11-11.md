# Rust CPU optimization evaluation — November 11, 2026

This release improves dense `Network::train` and repeated single-call prepared
inference. The numbers compare identical synthetic workloads on one shared
Ryzen 9 9950X host. They establish local throughput changes, not global
state-of-the-art performance, better predictive accuracy, GPU speedups or
results on another machine.

The largest measured dense training improvement is **4.08×** for f32 and
**3.88×** for f64. Large prepared f32 networks improve by about **2.5×**.
Reordered training has small measured tradeoffs, and four small inference
cases are 6–8% slower, reported below. Preparation, fixture creation and
network destruction are excluded from these timings; allocation performed
inside `Network::train` is included.

## Baseline and source identity

The baseline is source commit
`251e23cd63d182ad805f9e2c24c1468754db92bb`. Its complete 81-file `src/` map,
original Cargo manifest and Cargo.lock are byte-identical to the downloaded
public `ruv-fann@0.2.2` crate. The archive SHA-256 is
`febd1d48f07e4ccc9cf37bbf6a9272c682c73141137ac211f110acd33acbb1fe`,
independently matched to the crates.io sparse-index checksum.

The final training comparison uses combined candidate
`6c74e177ba833a73f8e4b36843e25184dd07b44e`, including activation inlining,
training changes, prepared inference and version 0.2.3. Later prepared-only
changes preserve every file in the recorded training execution source map.
The prepared comparison uses frozen kernel commit
`fe1ec2032c0851e6f6bf698ca01335f7122155d8`. The evidence records file hashes,
complete source maps, dependency graphs, lockfiles, toolchains and binary hashes
separately; these measurement commits are not publication receipts.

## What changed

### Dense hidden-layer backpropagation

Previously, each hidden neuron searched every destination neuron's incoming
connections for its first matching source. Canonical dense rows therefore
repeated the same searches. The new path validates connection order once per
hidden layer and pass, then directly indexes the matching weights. Work scales
with the connections instead of repeatedly searching dense rows.

The destination summation order, multiplication/addition order and first-match
behavior are unchanged. Sparse, reordered, missing and duplicate connections
use the original scan. Topology is public and mutable, so this check runs again
on each pass. Redundant forward-output and next-error clones are removed.

### Prepared single inference

A dense first layer reads the caller's input slice directly. Wide canonical
layers evaluate four independent output rows together, exposing independent
arithmetic without reordering any neuron's reduction or introducing fused
multiply-add. Bias remains the last term. Overflow errors preserve neuron
priority and the caller's output buffer.

Preparation caches dispatch in the immutable snapshot. First-layer tiling
requires at least 64 regular inputs and four outputs; later layers require at
least 64 source values, including any bias, and four outputs. An entire layer
must be canonical. Short or irregular layers retain scalar execution, and a
snapshot with no eligible layer enters its original scalar loops after one
cached branch. Neither inference path allocates after workspace creation.

## Dense training results

Times are median nanoseconds per training pattern, expressed in microseconds.
Speed ratios are the median baseline/candidate ratio across 12 paired rounds;
they need not equal the ratio of the two separately pooled timing medians.
Each round contains 21 measured blocks per case after seven warmup blocks.

| Precision | Dense shape | 0.2.2 baseline, µs | Candidate, µs | Speed ratio | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| f32 | 2 → 3 → 1 | 0.128 | 0.107 | 1.207× | 1.185–1.216 |
| f64 | 2 → 3 → 1 | 0.132 | 0.115 | 1.158× | 1.141–1.163 |
| f32 | 32 → 64 → 32 → 8 | 21.691 | 7.894 | 2.737× | 2.716–2.775 |
| f64 | 32 → 64 → 32 → 8 | 21.485 | 8.140 | 2.633× | 2.618–2.664 |
| f32 | 128 → 128 → 64 → 16 | 150.928 | 36.996 | 4.082× | 4.050–4.107 |
| f64 | 128 → 128 → 64 → 16 | 151.043 | 38.964 | 3.880× | 3.845–3.949 |

The six reordered controls preserve numerical behavior but are not uniformly
faster. Their speed ratios are:

| Precision | Reordered shape | Speed ratio | Paired 95% interval | Point latency change |
| --- | --- | ---: | --- | ---: |
| f32 | 2 → 3 → 1 | 1.194× | 1.179–1.218 | −16.2% |
| f64 | 2 → 3 → 1 | 1.155× | 1.130–1.163 | −13.4% |
| f32 | 32 → 64 → 32 → 8 | 1.044× | 1.032–1.055 | −4.2% |
| f64 | 32 → 64 → 32 → 8 | 0.994× | 0.984–0.995 | +0.65% |
| f32 | 128 → 128 → 64 → 16 | 0.988× | 0.975–0.998 | +1.27% |
| f64 | 128 → 128 → 64 → 16 | 0.989× | 0.983–0.998 | +1.06% |

All 12 case intervals clear the predeclared 5% latency noninferiority boundary:
the speed-ratio lower bound must be at least `1 / 1.05 = 0.952381`.
This is a per-case nominal interval check, not a simultaneous guarantee over
all possible networks. All **6,048** observations match their corresponding
baseline/candidate trained-weight checksums and exact weight-bit signatures.

## Prepared single inference results

The final matrix contains 60 cases: f32/f64, Linear/Sigmoid, nine dense
shapes including the 63/64/65 crossover, and mixed/sparse/reordered controls
at widths 33 and 65. All scalar comparisons and cross-revision output/fixture
signatures agree across 12 balanced rounds: **45,360** raw blocks of at least
2 ms. The comparison also retains the first-layer-only intermediate revision.

| Precision / activation | Dense shape | Baseline, µs | Candidate, µs | Paired speed ratio |
| --- | --- | ---: | ---: | ---: |
| f32 Linear | 100 → 50 → 25 → 10 | 1.778 | 1.469 | 1.211× |
| f32 Sigmoid | 100 → 50 → 25 → 10 | 2.710 | 1.975 | 1.330× |
| f64 Linear | 100 → 50 → 25 → 10 | 1.762 | 1.619 | 1.082× |
| f64 Sigmoid | 100 → 50 → 25 → 10 | 2.894 | 2.277 | 1.235× |
| f32 Linear | 784 → 128 → 64 → 10 | 38.783 | 15.534 | 2.491× |
| f32 Sigmoid | 784 → 128 → 64 → 10 | 42.735 | 16.686 | 2.498× |
| f64 Linear | 784 → 128 → 64 → 10 | 39.100 | 22.984 | 1.702× |
| f64 Sigmoid | 784 → 128 → 64 → 10 | 43.304 | 24.108 | 1.748× |

These paired geometric-mean ratios need not equal separately pooled median
latencies. Across the equally weighted 60 cases, the candidate/baseline latency
ratio is **0.9001**, 95% interval **0.8880–0.9168**. Dense cases have ratio
0.8667; irregular controls 0.9526. The tiny-case aggregate interval includes
parity. Equal case weighting is not a model of any application's traffic.

Four small canonical cases have more than 5% point slowdown and an interval
excluding parity. This is a measured release tradeoff:

| Case | Point latency increase | Candidate/baseline latency ratio, 95% interval |
| --- | ---: | --- |
| f32 Linear, 10 → 4 → 1 | +6.31% | 1.0468–1.0784 |
| f32 Linear, 63 → 17 → 5 | +5.60% | 1.0332–1.0787 |
| f32 Sigmoid, 63 → 17 → 5 | +6.90% | 1.0123–1.1340 |
| f64 Sigmoid, 33 → 17 → 5 | +7.89% | 1.0398–1.1492 |

The scoped release decision accepts reproducible large-network gains and the
improved irregular-control aggregate with these disclosed small-model costs.
It does **not** declare a universal 5% regression bound, or turn the preserved
historical `REVIEW_REQUIRED` observation into a numerical pass. Further tuning
needs a new experiment. The [complete 60-case table](rust-prepared-results-2026-11-11.md)
lists every slower result, intervals and the intermediate comparison.

## Evaluation method and limits

- AMD Ryzen 9 9950X, Linux x86-64; Rust 1.98.1 / LLVM 22.1.8.
- Generic Rust CPU target, optimization level 3, 16 codegen units, LTO off,
  incremental compilation off, no native CPU override or fast-math flags.
- Training uses no optional features; prepared inference enables `std` only.
  Runtime dependency versions match between variants. The training candidate's
  local package version changes from 0.2.2 to 0.2.3; its other lock entries match.
- Deterministic weights and inputs; f32/f64, tiny/medium/large shapes, dense and
  irregular controls. Training times the public `Network::train` API on 16
  patterns, four epochs per measured block for medium/large shapes and 64 for
  the tiny shape. Setup, network clones, destruction and printing are excluded.
- Training runs 12 balanced AB/BA pairs on CPU 15. Its SMT sibling 31 was
  3.81% busy during measurement. All raw observations are retained.
- Prepared inference runs all six three-variant order permutations twice on
  CPU 11. It cycles eight deterministic inputs; calibration, three warmup
  blocks and fixture/workspace creation precede the 21 measured blocks.
  Calls include validation, input selection, black-box barriers and output
  copies. SMT sibling 27 had median 1.40% busy time, maximum 30.34%; host load
  ranged 4.52–14.91. The shared-host caveat applies to every reported gain.
- Confidence intervals use 20,000 seeded paired-round bootstrap resamples.
  Within-process blocks are not treated as independent outer experiments.
  The prepared analyzer uses a geometric mean of paired latency ratios;
  the training analyzer uses a median of paired speed ratios.
- Other project lanes pause builds and tests during timing. Unrelated user
  work, SMT activity, scheduling and frequency variation remain possible.
  Intervals describe the observed runs, not machine or workload diversity.
- The comparison does not measure training quality, time to convergence,
  preparation amortization, cold-start latency, GPU throughput or a competing
  framework. Smaller models and irregular networks need their own measurements.

## Rejected experiments are retained

The first training prototype accelerated dense networks but slowed reordered
f64 training by roughly 58–66%; it was rejected. Hoisting the canonical/fallback
dispatch restored the original irregular scan. A noisy followup did not establish
the regression bound; a separately declared 24-pair, longer-block repeat was
retained alongside it. The final combined-source matrix above is the release
training comparison.

The first inference tile accelerated large f32 networks by approximately 2.5×
but slowed the 33-input Linear case by 22.8%. A second prototype protected that
case but still slowed a 63-input control. They were rejected. The final design
caches immutable dispatch and separates tiled execution from the scalar loops;
it improves the irregular-control aggregate but does not eliminate every small
canonical regression listed above.
Earlier raw data is preserved rather than replaced with the preferred result.

## Correctness and release qualification

Regression tests compare scalar and prepared outputs bitwise for both
precisions, all 21 activations, odd widths, biases, sparse/reordered/duplicate
connections, cancellation and signed zero. Overflow checks cover direct and
deeper layers, output atomicity and workspace recovery. Allocation tests include
the wide tiled path. Training equivalence compares every weight and cached
neuron state, including topology changes between calls. Mutating canonical
dispatch makes the corresponding regressions fail.

The actual browser probe executes wide `[65,65,5]` f32/f64 canonical, mixed and
biasless inference against the independent mutable scalar interpreter. It also
checks direct, deeper and mixed-layer overflow priority, unchanged output and
recovery. CI additionally covers native platforms, MSRV, optional features,
WASM, native integrations, dependency advisories and all 27 Cargo archives.
Passing local checks is not evidence that those remote jobs or publication have
completed; the protected release retains their exact-source receipts.

Publication reuses those CI archives, checks the complete inventory before any
upload, downloads canonical crates.io bytes and exercises a virgin public
root/Candle consumer. The consumer checks inference, input/parameter gradients,
three-step SGD and AdamW, and frozen snapshots against actual Candle CPU f64.
It also safely extracts the verified public root archive and executes its
packaged training-equivalence, wide-inference, overflow and allocation tests;
four nonempty summaries with zero failures, ignored or filtered tests are
required, bound to the public archive checksum and actual command log.
The release verifies its tag and every downloaded GitHub asset. See the
[release protocol](rust-release-protocol.md).

## Reproduce and inspect the evidence

The [raw evidence archive](../benchmarks/rust-0.2.3-evidence.tar.gz) contains an
`INDEX.json` with every included file's checksum and run status, the actual raw
measurements, frozen harnesses, drivers, analyzers, source/dependency identities
and rejected experiments. It is also attached to the
[Rust release](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-rust-v0.2.3).
It excludes build caches and compiled executables. Start with
`public-0.2.2/all-src-git-blob-comparison.json` and
`training-combined-6c74e177/summary.json` and `prepared-v3/REPORT.md`; each run's reproduction instructions
explain the source paths and exact build commands.

For a new workload, run the checked-in examples with explicit release flags,
keep baseline and candidate dependency versions equal, bind one quiet core,
alternate order and retain all samples:

```sh
cargo +1.98.1 run --release --locked --no-default-features \
  --example dense_training_benchmark
RUSTFLAGS='-C opt-level=3 -C codegen-units=16 -C lto=off' \
  cargo +1.98.1 run --release --locked --no-default-features --features std \
  --example prepared_matrix
```

Use the same harness source for both revisions. The older public crate does
not contain these new examples, so reproduce the revision comparison with the
external harness and lockfile included in the evidence, changing only the
source checkout path. Use a new output directory; do not overwrite old samples.
