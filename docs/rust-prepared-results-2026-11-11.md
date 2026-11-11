# Prepared single-inference results — 2026-11-11

The measured change is accepted as a **scoped optimization for large canonical dense networks**, with explicit small-model tradeoffs. It is not a claim of universal 5% noninferiority or a projected publication result. On the measured 784→128→64→10 shapes, f32 inference improved approximately 2.49× and f64 approximately 1.70–1.75× versus the verified public 0.2.2 source. All 60 cases matched scalar output bits across all revisions and rounds.

Four smaller canonical cases had geometric mean latency increases above 5%, with 95% intervals excluding parity: f32 Linear 10→4→1 (+6.31%), f32 Linear 63→17→5 (+5.60%), f32 Sigmoid 63→17→5 (+6.90%), and f64 Sigmoid 33→17→5 (+7.89%). Their interval lower endpoints remain below a 5% slowdown. The full case table and all slower comparisons appear below; aggregate improvements do not remove these tradeoffs.

The equal-case geometric mean candidate/public latency ratio is 0.9001 [0.8880, 0.9168], dense cases 0.8667 [0.8555, 0.8796], and mixed/sparse/reordered controls 0.9526 [0.9367, 0.9790]. Tiny-case aggregate 0.9769 [0.9541, 1.0042] includes parity. Cases receive equal weight, without representing any particular production workload.

## Compared artifacts

| Variant | Source commit | Purpose |
|---|---|---|
| Public baseline | `251e23cd63d182ad805f9e2c24c1468754db92bb` | All 81 library source files, original manifest and lock match the registry-checksummed public `ruv-fann` 0.2.2 archive. |
| Intermediate | `8bf15ebb947719b14d9c926ea40342ba8f15336a` | Dense first-layer copy bypass plus activation inlining; shown for transparency. |
| Candidate | `fe1ec2032c0851e6f6bf698ca01335f7122155d8` | Immutable dispatch cache, separate tiled execution, and conservative width eligibility. |

The candidate's complete library source was compared against integration commit `97948d74`; every source file matched. The public archive SHA-256 is `febd1d48f07e4ccc9cf37bbf6a9272c682c73141137ac211f110acd33acbb1fe`, matching the registry checksum. A fourth public binary would duplicate the primary baseline's library source and dependency specification.

Each variant used the same frozen benchmark source, SHA-256 `449fa9ea6ba04ab753815e67a6b1b46c2041bfb1866233cc71a241378ffef8c1`, and an identical standalone dependency lock, SHA-256 `5a0aadbda8fc7c0c7636024373f98cf3bf9261649c5a50d3e27caec964883c5d`. The standalone harness excludes repository development dependencies and uses `ruv-fann` with default features disabled and only `std` enabled, plus `num-traits`.

## Implementation and correctness

Preparation caches eligibility in private network and layer flags, using the snapshot's immutable topology. A layer qualifies only when all rows are canonical and at least four outputs are available; the first layer needs at least 64 regular inputs, and subsequent layers need a source slice width of at least 64. Scalar inference executes the original scalar loops after one cached network dispatch branch. The larger tiled implementation lives in a separate function with inlining disabled. Mixed, sparse, or reordered layers retain scalar evaluation.

Four independent output accumulators share input loads while preserving each row's multiply/add order. There is no horizontal reassociation or fast-math setting. Both immutable constructors initialize the cache. The focused suite passed 46 tests, including f32/f64 activation equivalence, widths 63/64/65, biasless and mixed topology, a mixed first layer followed by a tiled layer, cancellation/signed zero, error priority/recovery, and zero steady-state allocation. A forced canonical-cache guard mutation failed both float-width topology checks; restoring the source passed the suite.

The benchmark checks every fixture against scalar `Network::run` before timing and also compares fixture/output signatures across revisions and rounds. The eight fixed input samples include ordinary seeded finite values, positive/negative zero, alternating signs, and small nonzero values. Weights and inputs use explicit seeds; network construction does not depend on ambient randomness.

## Measurement and reproduction

The host was an AMD Ryzen 9 9950X with 16 physical cores and two threads per core. Every measured process was pinned to CPU 11; its SMT sibling was CPU 27. No governor or other system setting was changed. Builds ran sequentially before the exclusive agent timing window.

The compiler was Rust 1.98.1 (`48a229ceaefd4985c50990b14116b6d856af0985`, LLVM 22.1.8), generic x86-64 target, optimization level 3, 16 codegen units, LTO disabled, and the normal unwind panic strategy. No target-CPU override or fast math was used. External manifests explicitly set the release profile, and every build used the same flags and frozen lock:

```sh
RUSTFLAGS='-C opt-level=3 -C codegen-units=16 -C lto=off' \
  CARGO_TARGET_DIR=<variant>/target \
  cargo +1.98.1 build --release --locked --offline \
  --manifest-path <variant>/Cargo.toml
```

The repository example provides the dependency-free benchmark body:

```sh
cargo +1.98.1 run --release --no-default-features --features std \
  --example prepared_matrix -- --list
```

Use `--case NAME`, `--blocks N`, and `--block-ms N` to select a case and configure raw blocks. Comparable measurements require the same external release profile, compiler flags, dependency lock, benchmark source and CPU affinity for all revisions. The measurement invocation used the frozen standalone binaries with `taskset -c 11`, the default 21 blocks and a 2 ms minimum per reported block.

All six permutations of the three variants were run twice, giving 12 balanced rounds. Each process exercised 60 cases: f32/f64 × Linear/Sigmoid across nine dense shapes and mixed/sparse/reordered controls at input widths 33 and 65. Each case calibrated iterations adaptively, warmed three blocks, and reported 21 blocks, totaling **45,360 raw timing blocks**. Every reported block lasted at least 2 ms.

Network construction, preparation, workspace/output allocation, scalar comparison, calibration, warmup and printing were excluded from reported block durations. Warm calls include validation, selecting the next input sample, black-box barriers and output copying. Setup latency and batching throughput are outside this measurement's scope.

Each case's per-run result is the median of 21 blocks. Ratios pair revisions within the same round; geometric mean ratios and seeded 20,000-resample percentile 95% intervals resample the 12 paired rounds. The 252 blocks per revision are not treated as independent observations. Per-case intervals are exploratory without multiple-comparison adjustment. Table time columns are independent medians across rounds; the paired ratio need not equal their quotient.

Shared-host interference remained material despite pausing other project agents' heavy commands. One-minute host load ranged 4.52–14.91; CPU 27's busy percentage had median 1.40% and maximum 30.34%. CPU 11 was nearly fully busy with the benchmark itself. Balanced orders and paired intervals reduce drift but cannot exclude scheduler, SMT, frequency, instruction-cache or other interference. These are local CPU measurements, not a portable speed guarantee or a state-of-the-art claim.

Frozen local evidence includes all raw JSONL blocks, output/fixture signatures, timestamps, per-process host and CPU counters, source/tree/toolchain/flag/dependency/binary digests, build/test/mutant logs and reproducible runners. The 115-file evidence manifest, including three final binaries, has SHA-256 `9c8d3c94ebcf073f9a7ba15283dffe43c9d97100d38553b4e45cf5cc15fead7d`. Earlier failed candidates and their results are preserved separately. The measurement's original acceptance-review flags are retained as historical evidence; the integration decision accepts the scoped large-network benefit with the tradeoffs stated here.

## Full case results

Ratios below 1 mean lower candidate latency. The `large` aggregate below refers to the original 33→17→5, 100→50→25→10, and 784→128→64→10 dense shapes; crossover cases at widths 63/64/65 are included in `dense` and `all`.

| Case | Baseline ns | Intermediate ns | Candidate ns | Candidate / baseline (95% CI) | Candidate / intermediate (95% CI) |
|---|---:|---:|---:|---:|---:|
| f32_linear_dense_100x50x25x10 | 1778.45 | 1768.18 | 1469.16 | 0.8257 [0.8201, 0.8319] | 0.8274 [0.8225, 0.8320] |
| f32_linear_dense_10x1 | 13.22 | 12.74 | 12.79 | 0.9780 [0.9552, 1.0131] | 1.0068 [0.9937, 1.0216] |
| f32_linear_dense_10x4x1 | 36.35 | 37.84 | 38.69 | 1.0631 [1.0468, 1.0784] | 1.0224 [1.0023, 1.0388] |
| f32_linear_dense_33x17x5 | 216.85 | 219.27 | 220.69 | 1.0164 [1.0057, 1.0274] | 1.0038 [0.9937, 1.0157] |
| f32_linear_dense_5x7x3 | 56.57 | 54.11 | 53.74 | 0.9532 [0.9463, 0.9634] | 0.9976 [0.9922, 1.0048] |
| f32_linear_dense_63x17x5 | 329.50 | 357.74 | 344.42 | 1.0560 [1.0332, 1.0787] | 0.9703 [0.9532, 0.9898] |
| f32_linear_dense_64x17x5 | 375.07 | 337.00 | 322.07 | 0.8588 [0.8402, 0.8808] | 0.9580 [0.9490, 0.9670] |
| f32_linear_dense_65x17x5 | 369.92 | 351.40 | 326.05 | 0.8811 [0.8710, 0.8942] | 0.9184 [0.9098, 0.9274] |
| f32_linear_dense_784x128x64x10 | 38782.82 | 38296.43 | 15533.56 | 0.4014 [0.3989, 0.4037] | 0.4039 [0.4006, 0.4070] |
| f32_linear_mixed_33x17x5 | 217.15 | 219.63 | 216.07 | 0.9938 [0.9811, 1.0066] | 0.9372 [0.8420, 0.9937] |
| f32_linear_mixed_65x17x5 | 360.29 | 393.10 | 362.90 | 1.0018 [0.9938, 1.0083] | 0.8866 [0.8161, 0.9280] |
| f32_linear_reordered_33x17x5 | 242.40 | 263.38 | 244.15 | 1.0063 [1.0026, 1.0104] | 0.9180 [0.9047, 0.9285] |
| f32_linear_reordered_65x17x5 | 407.59 | 449.06 | 407.64 | 1.0005 [0.9946, 1.0059] | 0.9048 [0.8998, 0.9090] |
| f32_linear_sparse_33x17x5 | 193.36 | 206.62 | 193.46 | 0.9966 [0.9802, 1.0106] | 0.9357 [0.9254, 0.9447] |
| f32_linear_sparse_65x17x5 | 287.58 | 324.82 | 287.12 | 1.0008 [0.9963, 1.0058] | 0.8751 [0.8527, 0.8892] |
| f32_sigmoid_dense_100x50x25x10 | 2709.72 | 2753.94 | 1974.57 | 0.7518 [0.7274, 0.7961] | 0.7544 [0.7144, 0.8120] |
| f32_sigmoid_dense_10x1 | 19.53 | 18.62 | 18.65 | 0.9495 [0.9193, 0.9727] | 0.9950 [0.9804, 1.0107] |
| f32_sigmoid_dense_10x4x1 | 53.89 | 53.06 | 53.38 | 1.0240 [0.9798, 1.1095] | 1.0439 [0.9977, 1.1309] |
| f32_sigmoid_dense_33x17x5 | 338.45 | 331.31 | 327.24 | 0.9936 [0.9584, 1.0580] | 1.0211 [0.9839, 1.0881] |
| f32_sigmoid_dense_5x7x3 | 86.31 | 83.92 | 85.65 | 0.9923 [0.9725, 1.0135] | 1.0283 [1.0133, 1.0465] |
| f32_sigmoid_dense_63x17x5 | 513.44 | 570.01 | 548.01 | 1.0690 [1.0123, 1.1340] | 0.9787 [0.9529, 1.0235] |
| f32_sigmoid_dense_64x17x5 | 553.81 | 553.61 | 388.59 | 0.7089 [0.6582, 0.7707] | 0.7228 [0.6992, 0.7675] |
| f32_sigmoid_dense_65x17x5 | 543.20 | 578.51 | 402.54 | 0.7490 [0.6958, 0.8118] | 0.7199 [0.6932, 0.7654] |
| f32_sigmoid_dense_784x128x64x10 | 42734.56 | 41879.52 | 16685.81 | 0.4004 [0.3911, 0.4160] | 0.4055 [0.3949, 0.4245] |
| f32_sigmoid_mixed_33x17x5 | 335.57 | 397.17 | 336.48 | 1.0421 [0.9965, 1.1277] | 0.9209 [0.8500, 1.0350] |
| f32_sigmoid_mixed_65x17x5 | 518.50 | 609.63 | 522.81 | 1.0386 [0.9998, 1.1093] | 0.9014 [0.8488, 0.9939] |
| f32_sigmoid_reordered_33x17x5 | 424.13 | 499.95 | 426.50 | 1.0351 [1.0047, 1.0881] | 0.8988 [0.8468, 0.9848] |
| f32_sigmoid_reordered_65x17x5 | 648.21 | 738.16 | 651.23 | 1.0114 [1.0019, 1.0219] | 0.8993 [0.8735, 0.9332] |
| f32_sigmoid_sparse_33x17x5 | 331.83 | 384.93 | 334.44 | 1.0466 [1.0011, 1.1325] | 0.9158 [0.8542, 1.0230] |
| f32_sigmoid_sparse_65x17x5 | 496.52 | 572.03 | 498.12 | 1.0321 [0.9944, 1.1001] | 0.9106 [0.8582, 1.0007] |
| f64_linear_dense_100x50x25x10 | 1762.00 | 1710.10 | 1618.65 | 0.9239 [0.9181, 0.9303] | 0.9470 [0.9399, 0.9531] |
| f64_linear_dense_10x1 | 13.62 | 12.66 | 12.92 | 0.9901 [0.9430, 1.0866] | 1.0692 [1.0129, 1.1763] |
| f64_linear_dense_10x4x1 | 36.60 | 36.70 | 34.98 | 0.9633 [0.9540, 0.9766] | 0.9554 [0.9414, 0.9739] |
| f64_linear_dense_33x17x5 | 219.82 | 240.59 | 218.49 | 1.0251 [0.9796, 1.1090] | 0.9359 [0.8947, 1.0090] |
| f64_linear_dense_5x7x3 | 55.80 | 54.57 | 53.69 | 1.0119 [0.9608, 1.1127] | 1.0318 [0.9787, 1.1340] |
| f64_linear_dense_63x17x5 | 334.41 | 358.72 | 342.33 | 1.0491 [1.0055, 1.1226] | 0.9761 [0.9326, 1.0448] |
| f64_linear_dense_64x17x5 | 341.35 | 336.69 | 342.56 | 1.0104 [0.9950, 1.0268] | 1.0347 [1.0184, 1.0562] |
| f64_linear_dense_65x17x5 | 353.90 | 356.02 | 345.74 | 0.9867 [0.9779, 0.9967] | 0.9827 [0.9683, 0.9993] |
| f64_linear_dense_784x128x64x10 | 39099.98 | 38486.38 | 22983.92 | 0.5874 [0.5768, 0.5954] | 0.5960 [0.5912, 0.6003] |
| f64_linear_mixed_33x17x5 | 266.85 | 224.97 | 223.16 | 0.8425 [0.8334, 0.8517] | 0.9929 [0.9802, 1.0071] |
| f64_linear_mixed_65x17x5 | 441.00 | 388.70 | 394.56 | 0.9315 [0.8928, 1.0067] | 1.0617 [1.0152, 1.1539] |
| f64_linear_reordered_33x17x5 | 339.65 | 249.14 | 265.39 | 0.7702 [0.7414, 0.7896] | 1.0215 [0.9212, 1.0817] |
| f64_linear_reordered_65x17x5 | 559.77 | 414.21 | 456.50 | 0.7794 [0.7140, 0.8174] | 1.0588 [0.9725, 1.1079] |
| f64_linear_sparse_33x17x5 | 253.33 | 200.29 | 211.66 | 0.8834 [0.8282, 0.9936] | 1.1206 [1.0484, 1.2594] |
| f64_linear_sparse_65x17x5 | 403.14 | 293.80 | 326.42 | 0.8059 [0.7644, 0.8470] | 1.1311 [1.1044, 1.1725] |
| f64_sigmoid_dense_100x50x25x10 | 2894.48 | 2863.89 | 2276.87 | 0.8096 [0.7826, 0.8609] | 0.8163 [0.7903, 0.8665] |
| f64_sigmoid_dense_10x1 | 21.39 | 19.84 | 20.02 | 0.8940 [0.8172, 0.9445] | 0.9703 [0.8941, 1.0170] |
| f64_sigmoid_dense_10x4x1 | 67.59 | 58.76 | 59.11 | 0.9039 [0.8111, 1.0134] | 1.0292 [0.9140, 1.1624] |
| f64_sigmoid_dense_33x17x5 | 371.85 | 384.76 | 387.66 | 1.0789 [1.0398, 1.1492] | 1.0130 [0.9280, 1.1040] |
| f64_sigmoid_dense_5x7x3 | 101.57 | 97.55 | 99.27 | 1.0130 [0.9573, 1.1116] | 1.0178 [0.9012, 1.1511] |
| f64_sigmoid_dense_63x17x5 | 574.61 | 555.41 | 541.01 | 0.9658 [0.9334, 1.0217] | 0.9969 [0.9669, 1.0509] |
| f64_sigmoid_dense_64x17x5 | 573.30 | 548.35 | 445.80 | 0.7793 [0.7659, 0.7937] | 0.8135 [0.8061, 0.8224] |
| f64_sigmoid_dense_65x17x5 | 587.58 | 589.36 | 443.14 | 0.7665 [0.7102, 0.8415] | 0.7763 [0.7456, 0.8364] |
| f64_sigmoid_dense_784x128x64x10 | 43304.30 | 42354.41 | 24108.25 | 0.5721 [0.5572, 0.5950] | 0.5781 [0.5651, 0.5990] |
| f64_sigmoid_mixed_33x17x5 | 390.60 | 367.47 | 379.79 | 0.9848 [0.9674, 1.0074] | 1.0452 [1.0302, 1.0667] |
| f64_sigmoid_mixed_65x17x5 | 602.74 | 551.31 | 609.66 | 1.0051 [0.9942, 1.0168] | 1.0987 [1.0906, 1.1070] |
| f64_sigmoid_reordered_33x17x5 | 493.96 | 452.26 | 463.68 | 0.9356 [0.9282, 0.9427] | 1.0245 [1.0206, 1.0294] |
| f64_sigmoid_reordered_65x17x5 | 738.50 | 678.13 | 704.97 | 0.9530 [0.9470, 0.9583] | 1.0378 [1.0319, 1.0442] |
| f64_sigmoid_sparse_33x17x5 | 382.50 | 348.65 | 361.26 | 0.9089 [0.8392, 0.9495] | 1.0378 [1.0327, 1.0434] |
| f64_sigmoid_sparse_65x17x5 | 574.43 | 528.02 | 544.81 | 0.9479 [0.9389, 0.9548] | 1.0341 [1.0277, 1.0405] |

## Group geometric means

| Comparison | Group | Cases | Ratio | 95% CI |
|---|---|---:|---:|---:|
| intermediate/baseline | all | 60 | 0.9830 | [0.9717, 0.9933] |
| intermediate/baseline | dense | 36 | 0.9896 | [0.9778, 1.0021] |
| intermediate/baseline | controls | 24 | 0.9732 | [0.9612, 0.9845] |
| intermediate/baseline | tiny | 12 | 0.9639 | [0.9413, 0.9906] |
| intermediate/baseline | large | 12 | 1.0048 | [0.9975, 1.0120] |
| candidate/baseline | all | 60 | 0.9001 | [0.8880, 0.9168] |
| candidate/baseline | dense | 36 | 0.8667 | [0.8555, 0.8796] |
| candidate/baseline | controls | 24 | 0.9526 | [0.9367, 0.9790] |
| candidate/baseline | tiny | 12 | 0.9769 | [0.9541, 1.0042] |
| candidate/baseline | large | 12 | 0.7423 | [0.7298, 0.7585] |
| candidate/intermediate | all | 60 | 0.9157 | [0.8997, 0.9389] |
| candidate/intermediate | dense | 36 | 0.8758 | [0.8611, 0.8936] |
| candidate/intermediate | controls | 24 | 0.9789 | [0.9574, 1.0138] |
| candidate/intermediate | tiny | 12 | 1.0135 | [0.9987, 1.0315] |
| candidate/intermediate | large | 12 | 0.7388 | [0.7239, 0.7586] |

## Regressions requiring review

- candidate/baseline, f32_linear_dense_10x4x1: 6.31% slower; 95% CI [1.0468, 1.0784]. Interval excludes parity.
- candidate/baseline, f32_linear_dense_33x17x5: 1.64% slower; 95% CI [1.0057, 1.0274]. Interval excludes parity.
- candidate/baseline, f32_linear_dense_63x17x5: 5.60% slower; 95% CI [1.0332, 1.0787]. Interval excludes parity.
- candidate/baseline, f32_linear_mixed_65x17x5: 0.18% slower; 95% CI [0.9938, 1.0083]. Interval includes parity.
- candidate/baseline, f32_linear_reordered_33x17x5: 0.63% slower; 95% CI [1.0026, 1.0104]. Interval excludes parity.
- candidate/baseline, f32_linear_reordered_65x17x5: 0.05% slower; 95% CI [0.9946, 1.0059]. Interval includes parity.
- candidate/baseline, f32_linear_sparse_65x17x5: 0.08% slower; 95% CI [0.9963, 1.0058]. Interval includes parity.
- candidate/baseline, f32_sigmoid_dense_10x4x1: 2.40% slower; 95% CI [0.9798, 1.1095]. Interval includes parity.
- candidate/baseline, f32_sigmoid_dense_63x17x5: 6.90% slower; 95% CI [1.0123, 1.1340]. Interval excludes parity.
- candidate/baseline, f32_sigmoid_mixed_33x17x5: 4.21% slower; 95% CI [0.9965, 1.1277]. Interval includes parity.
- candidate/baseline, f32_sigmoid_mixed_65x17x5: 3.86% slower; 95% CI [0.9998, 1.1093]. Interval includes parity.
- candidate/baseline, f32_sigmoid_reordered_33x17x5: 3.51% slower; 95% CI [1.0047, 1.0881]. Interval excludes parity.
- candidate/baseline, f32_sigmoid_reordered_65x17x5: 1.14% slower; 95% CI [1.0019, 1.0219]. Interval excludes parity.
- candidate/baseline, f32_sigmoid_sparse_33x17x5: 4.66% slower; 95% CI [1.0011, 1.1325]. Interval excludes parity.
- candidate/baseline, f32_sigmoid_sparse_65x17x5: 3.21% slower; 95% CI [0.9944, 1.1001]. Interval includes parity.
- candidate/baseline, f64_linear_dense_33x17x5: 2.51% slower; 95% CI [0.9796, 1.1090]. Interval includes parity.
- candidate/baseline, f64_linear_dense_5x7x3: 1.19% slower; 95% CI [0.9608, 1.1127]. Interval includes parity.
- candidate/baseline, f64_linear_dense_63x17x5: 4.91% slower; 95% CI [1.0055, 1.1226]. Interval excludes parity.
- candidate/baseline, f64_linear_dense_64x17x5: 1.04% slower; 95% CI [0.9950, 1.0268]. Interval includes parity.
- candidate/baseline, f64_sigmoid_dense_33x17x5: 7.89% slower; 95% CI [1.0398, 1.1492]. Interval excludes parity.
- candidate/baseline, f64_sigmoid_dense_5x7x3: 1.30% slower; 95% CI [0.9573, 1.1116]. Interval includes parity.
- candidate/baseline, f64_sigmoid_mixed_65x17x5: 0.51% slower; 95% CI [0.9942, 1.0168]. Interval includes parity.
- candidate/intermediate, f32_linear_dense_10x1: 0.68% slower; 95% CI [0.9937, 1.0216]. Interval includes parity.
- candidate/intermediate, f32_linear_dense_10x4x1: 2.24% slower; 95% CI [1.0023, 1.0388]. Interval excludes parity.
- candidate/intermediate, f32_linear_dense_33x17x5: 0.38% slower; 95% CI [0.9937, 1.0157]. Interval includes parity.
- candidate/intermediate, f32_sigmoid_dense_10x4x1: 4.39% slower; 95% CI [0.9977, 1.1309]. Interval includes parity.
- candidate/intermediate, f32_sigmoid_dense_33x17x5: 2.11% slower; 95% CI [0.9839, 1.0881]. Interval includes parity.
- candidate/intermediate, f32_sigmoid_dense_5x7x3: 2.83% slower; 95% CI [1.0133, 1.0465]. Interval excludes parity.
- candidate/intermediate, f64_linear_dense_10x1: 6.92% slower; 95% CI [1.0129, 1.1763]. Interval excludes parity.
- candidate/intermediate, f64_linear_dense_5x7x3: 3.18% slower; 95% CI [0.9787, 1.1340]. Interval includes parity.
- candidate/intermediate, f64_linear_dense_64x17x5: 3.47% slower; 95% CI [1.0184, 1.0562]. Interval excludes parity.
- candidate/intermediate, f64_linear_mixed_65x17x5: 6.17% slower; 95% CI [1.0152, 1.1539]. Interval excludes parity.
- candidate/intermediate, f64_linear_reordered_33x17x5: 2.15% slower; 95% CI [0.9212, 1.0817]. Interval includes parity.
- candidate/intermediate, f64_linear_reordered_65x17x5: 5.88% slower; 95% CI [0.9725, 1.1079]. Interval includes parity.
- candidate/intermediate, f64_linear_sparse_33x17x5: 12.06% slower; 95% CI [1.0484, 1.2594]. Interval excludes parity.
- candidate/intermediate, f64_linear_sparse_65x17x5: 13.11% slower; 95% CI [1.1044, 1.1725]. Interval excludes parity.
- candidate/intermediate, f64_sigmoid_dense_10x4x1: 2.92% slower; 95% CI [0.9140, 1.1624]. Interval includes parity.
- candidate/intermediate, f64_sigmoid_dense_33x17x5: 1.30% slower; 95% CI [0.9280, 1.1040]. Interval includes parity.
- candidate/intermediate, f64_sigmoid_dense_5x7x3: 1.78% slower; 95% CI [0.9012, 1.1511]. Interval includes parity.
- candidate/intermediate, f64_sigmoid_mixed_33x17x5: 4.52% slower; 95% CI [1.0302, 1.0667]. Interval excludes parity.
- candidate/intermediate, f64_sigmoid_mixed_65x17x5: 9.87% slower; 95% CI [1.0906, 1.1070]. Interval excludes parity.
- candidate/intermediate, f64_sigmoid_reordered_33x17x5: 2.45% slower; 95% CI [1.0206, 1.0294]. Interval excludes parity.
- candidate/intermediate, f64_sigmoid_reordered_65x17x5: 3.78% slower; 95% CI [1.0319, 1.0442]. Interval excludes parity.
- candidate/intermediate, f64_sigmoid_sparse_33x17x5: 3.78% slower; 95% CI [1.0327, 1.0434]. Interval excludes parity.
- candidate/intermediate, f64_sigmoid_sparse_65x17x5: 3.41% slower; 95% CI [1.0277, 1.0405]. Interval excludes parity.
