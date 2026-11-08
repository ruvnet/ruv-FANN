# CUDA and OpenCV package review

Reviewed 2026-11-07 on `mission/package-modernization`. This report covers the
Rust package in `cuda-wasm` and all four Cargo manifests in `opencv-rust`.
The npm CUDA deliverable is handled by the separate Node package review.
No registry publication, push or commit was performed by this review.

## Package status

| Package | Checkout before review | Staged candidate | Result and remaining release work |
| --- | --- | --- | --- |
| `cuda-rust-wasm` | 0.1.6; registry inventory reports 0.1.7 | 0.2.0 | Portable default/minimal tests and WASM target checks pass. Native CUDA, HIP, Vulkan and browser execution remain unverified. Review license metadata before release. |
| `opencv-core` | 4.8.1 | 4.9.0 | Native/default/minimal tests and full workspace archive verification pass. It is a single-channel portable matrix subset, with an explicit 1 GiB allocation limit. |
| `opencv-wasm` | 4.8.1 | 4.9.0 | Native tests, WASM build, eight real Chromium checks and workspace archive verification pass. Exported pixels own JavaScript storage. |
| `opencv-sdk` | 4.8.1 | 4.9.0 | C string lifetime test, all-feature compilation and workspace archive verification pass. Header generation no longer nests Cargo or writes into package source. The Python API is intentionally small. |
| `opencv-sys` | 0.1.0; registry inventory reports no package | 0.2.0 | Real C++ OpenCV bridge implemented, restored to workspace and publication. Twelve native integrations, eight ABI layouts, all 95 baseline symbol links and archive verification pass against OpenCV 4.10.0. |

These are local release candidates. Version numbers do not establish release
readiness or upstream OpenCV API compatibility. Internal OpenCV path dependencies
now require version 4.9.0 rather than a previous published matrix implementation.

## Native bridge and browser blockers closed

The second implementation pass replaced the quarantined native scaffold with
actual OpenCV C++ code. All previously declared functions have implementations;
there are no duplicated handwritten/generated Rust declarations. One C header
feeds bindgen and the compiled C++ implementation. Owned point, rectangle,
keypoint and nested-point vectors now have constructors, deletion, bounds-checked
access and copying push/get operations, making feature/calibration APIs usable.

The bridge calls native Mat, image filters/codecs, VideoCapture, HighGUI, SIFT,
cascade classification, camera calibration, solvePnP and SVM. Every C++ entry
point contains exceptions and returns a documented sentinel with a thread-local
message. Raw pointers remain unsafe: live ownership, valid sized buffers,
non-overlap, synchronization and GUI main-thread/display rules are caller
obligations. Native ROI shares OpenCV refcounted storage, survives parent handle
deletion and uses a row stride; clone is independent. This differs deliberately
from the portable Rust matrix's copied ROI.

Constructors validate dimensions, matrix types, input byte counts and a 1 GiB
budget; resize validates output dimensions and overflow. Other OpenCV algorithms
also allocate upstream working memory. Optional CUDA functions call actual
GpuMat/CUDA filters when OpenCV provides them. On the tested CPU-only OpenCV,
`--features cuda` fails explicitly because `cudafilters` is absent. No fallback
pretends to execute CUDA. Physical CUDA, camera capture hardware, positive cascade
detection models and GUI display interaction are not claimed by the CPU tests.

The environment already had OpenCV 4.10.0 runtime libraries but lacked development
headers. With no sudo or scoped package helper, the reproducible Debian 13 helper
extracts matching libraries/headers into a user-owned prefix. It verifies Debian's
InRelease signature with the installed archive keyring, then the compressed index
and each package SHA-256. No system or privilege configuration changes are made.
`scripts/native-env.sh` accepts the chosen prefix; published build scripts use
pkg-config and contain no session-specific absolute paths. Session provenance is
`/workspace/scratch/debian-native/prefix/native-package-provenance.json`.

The first actual browser test exposed another ownership bug: ImageData retained a
view of a temporary Rust vector passed as `Clamped<&[u8]>`. After that vector was
freed, exported red/green pixels became allocator bytes (124/124 instead of
77/149). Export now copies to a JavaScript-owned Uint8ClampedArray before creating
ImageData. The regression retains the result across Rust frees and WASM memory
growth. Chromium verifies grayscale conversion, real canvas roundtrip, ROI,
nearest-neighbor pixels, invalid inputs, compiled flags and the 65536-to-65537
height overflow case (last pixel 233).

The SDK archive blocker was recursive `cargo metadata` inside cbindgen and a
header written below package source. cbindgen now parses `src/lib.rs` directly,
limits the output to the actual C exports and writes to OUT_DIR. Ordinary
`cargo package --workspace --allow-dirty --locked` now compiles **all four**
OpenCV archives with their new local dependency versions, without a custom
registry, publication or `--no-verify`. License payloads match the existing
Apache-2.0 metadata.

## Correctness and security changes

### CUDA memory and raw APIs

- `HostBuffer<T>` and `DeviceBuffer<T>` own initialized `Vec<T>` storage. Typed
  allocations respect `T`'s alignment and initialize through `Default`, including
  types whose valid values are not arbitrary zeroed bytes. Allocation overflow is
  rejected. These are host allocations, not CUDA pinned/device memory.
- Static shared memory owns initialized values and runs their destructors.
  Dynamic shared-memory casts require `bytemuck::Pod`, preventing safe creation
  of invalid references, `bool` values, or nonzero types from arbitrary bytes.
- Unified and raw runtime buffers are initialized before reads. The small CPU
  runtime tracks allocation layouts and really deallocates, rejecting unknown or
  repeated frees instead of leaking every allocation.
- The kernel memory manager retains the original vector, length and capacity
  rather than reconstructing a vector with the requested size as its allocation
  layout. It validates size/alignment/overflow and aligns an interior pointer
  within reserved padding. Releasing it returns the original owned vector.
- A full memory pool previously acquired its statistics lock while holding the
  pool lock, opposite to allocation's lock order. Release now drops the pool lock
  first. A four-thread regression repeatedly exercises that branch.
- Raw backend copies and launches now require `unsafe`. Their public contracts
  specify initialized live host storage, correct argument ABI, ownership,
  non-overlap and device synchronization. Converting a pointer to an integer is
  not treated as pointer validation.
- Native CUDA/HIP launch failures and unimplemented Vulkan dispatch now return
  errors; they no longer report successful execution without dispatching work.
- External WASM runtime module files use exclusive creation with random names,
  owner-only Unix permissions and cleanup on every exit path. Predictable paths
  that could follow pre-existing symlinks are removed. The runtime still needs a
  separately installed `wasmtime`/`wasmer`; the prefix check is not full module
  validation. Unsupported grids, blocks and arguments are rejected before execution.

### CUDA parsing, numerical behavior and discovery

- UTF-8 recovery advances by a complete character rather than slicing at byte
  one. Conservative limits reject source over 1 MiB, nesting over 32, or an
  uninterrupted expression/declaration over 256 non-whitespace characters.
  Comments and quoted strings are skipped by this budget. The existing parser
  still recovers from some unsupported input; parsing success is not a proof of
  full-language support or semantic equivalence.
- Attention validates nonzero dimensions/tiles, finite scaling, multiplication
  overflow and unsupported dropout. Nonzero dropout is rejected. It uses one
  reusable score scratch vector instead of allocating inside each tile/query
  loop. Host scratch metrics no longer describe invented GPU SRAM usage.
- Block dimension validation rejects zero axes and overflow. Multi-head shape
  arithmetic is checked, and diagnostic FLOP counters saturate.
- Default features now enable serialization and portable CPU use. Native
  GPU dependencies, GPU driver discovery and adapter tests are opt in. This
  avoids loading multiple vendor LLVM libraries during ordinary CPU tests.
- Local NC2 discovery no longer contacts an AWS IMDSv1 metadata endpoint or
  infers a Nutanix cluster from an AWS instance. Configuration `Debug` redacts
  passwords and API keys. Remote management/migration stubs remain documented.
- Missing browser `Performance` and logging dependencies are supplied. GPU and
  memory traits use wgpu's `WasmNotSendSync`: native implementations retain thread
  guarantees while browser handles remain local to their browser thread.

### OpenCV data and bindings

- `Mat::from_slice` accepts a sealed list of matching primitive representations
  through `MatElement`. It no longer copies arbitrary generic structs with
  padding or reads past a slice when the requested matrix element width differs.
- Matrix sizes use checked arithmetic, reject negative dimensions, and enforce a
  1 GiB allocation budget with fallible reserve. `total` avoids i32 products.
- Scalar construction now writes scalar values. ROI now copies the selected
  pixels into independent storage and checks rectangle arithmetic. Conversion
  computes `alpha * value + beta` and allocates the destination element width,
  with rounded/saturating integer output. Non-finite coefficients and unsupported
  half-float conversion return errors. Half-float raw bits can be imported as u16.
- Safe byte accessors expose initialized native-endian storage. That byte storage
  must not be assumed aligned for arbitrary typed pointer dereferences.
- Browser import/export handles actual grayscale/RGBA pixels. Nearest-neighbor
  resize maps real pixels using u64 intermediate products to avoid WASM32
  multiplication overflow on tall or wide images. Missing canvas contexts and
  cross-origin failures return errors.
- Blur, Gaussian blur and Canny explicitly return unsupported errors. They no
  longer return an unchanged image under the name of an applied operation.
- WASM uses the standard allocator rather than unused `wee_alloc`. SIMD capability
  reflects the actual `simd128` target feature rather than always returning true.
- `opencv_sdk_version` returns a static NUL-terminated string. Its previous Rust
  string pointer was not a valid C string. PyO3 is upgraded from 0.20 to 0.29 and
  the module initialization signature follows the current bound API.

## Migration notes

`cuda-rust-wasm` 0.2.0 callers need `T: Default` when constructing typed buffers or
static shared memory. Dynamic shared-memory element types need `Pod`. Raw copies
and launches require justified `unsafe` blocks and the documented lifetime/ABI
obligations. GPU users must explicitly enable their selected backend feature.

OpenCV generic callers must add `T: MatElement` to `Mat::from_slice` inputs, or
convert to supported primitives. ROI is an independent copy. Conversion now
transforms values instead of relabeling bytes. Unsupported filters and half-float
conversions return errors. These behavior changes motivate a coordinated 4.9.0
candidate rather than republishing 4.8.1. The version does not claim OpenCV 4.9
feature parity.

## Dependency audit

Cargo audit 0.22.2 used RustSec database commit
`309392b18f6b17022f82c701f300433126901068`, committed
2026-11-06T16:34:32+01:00. The JSON output contains 1422 advisories; its database
timestamp fields are null, so the commit above is recorded directly from Git.

The CUDA baseline contained 16 vulnerability entries, including old `bytes`,
`crossbeam-epoch`, `h2`, `libz-sys`, `quick-xml`, `rustls`, `rustls-webpki`,
`ruzstd`, `slab`, `tracing-subscriber` and `webpki-roots`. Compatible dependency
updates and reqwest 0.12 replace the vulnerable HTTP/TLS tree. Unused dev dependency
`cargo-tarpaulin` and unused build dependency `lalrpop` were removed; running a
coverage CLI does not require shipping it as a library's dependency.

The OpenCV baseline contained three vulnerability entries: crossbeam-epoch and two
PyO3 entries. Compatible updates and PyO3 0.29 remove those entries.

Both final lockfiles have **zero known vulnerability entries**, without ignored
advisories. CUDA still reports unmaintained `bincode` (RUSTSEC-2025-0141) and
`paste` (RUSTSEC-2024-0436); OpenCV still reports `paste`. A bincode wire-format
migration and upstream removal of paste require separate compatibility work.
An audit result is dependency evidence, not a claim that the code is secure.

Final SHA-256:

| Lockfile | SHA-256 |
| --- | --- |
| `cuda-wasm/Cargo.lock` | `ff4bb589d5b39585ac19858640a1a98389dcda0ce3b782d886312aa27f4d11e0` |
| `opencv-rust/Cargo.lock` | `818a9e96facee00e9d00ace47e553d879e64e296929a28d37d91f735810b1b17` |

## Validation

Environment: x86_64 Linux, rustc 1.99.1, Cargo 1.99.1, installed
`wasm32-unknown-unknown`. Commands run from the repository root after sourcing
`$HOME/.cargo/env`. Initial portable builds used `CARGO_BUILD_JOBS=2`; final
all-target, native and archive checks used `CARGO_BUILD_JOBS=1`. Both used
`CARGO_PROFILE_DEV_DEBUG=0`. Native commands also sourced
`scripts/native-env.sh` with the verified prefix described above.

| Command | Result |
| --- | --- |
| `cargo test --manifest-path cuda-wasm/Cargo.toml --lib --tests --locked` | 743 passed: 504 library plus integration/binary tests, including nine new safety regressions |
| `cargo test --manifest-path cuda-wasm/Cargo.toml --all-targets --locked` | 744 tests and 40 real Criterion smoke cases passed, including the vector-add example |
| `cargo check --manifest-path cuda-wasm/Cargo.toml --all-targets --no-default-features --locked` | passed |
| `cargo check --manifest-path cuda-wasm/Cargo.toml --all-targets --features webgpu,nutanix,regression-tests --locked` | passed |
| `cargo test --manifest-path cuda-wasm/Cargo.toml --bench regression_benchmarks --features regression-tests --locked` | two real Criterion smoke cases passed |
| `cargo test --manifest-path cuda-wasm/Cargo.toml --lib --tests --no-default-features --locked` | 730 passed |
| `cargo check --manifest-path cuda-wasm/Cargo.toml --lib --features nutanix --locked` | passed with upgraded HTTP client |
| `cargo check --manifest-path cuda-wasm/Cargo.toml --target wasm32-unknown-unknown --no-default-features --locked` | passed |
| `cargo check --manifest-path cuda-wasm/Cargo.toml --target wasm32-unknown-unknown --locked` | passed |
| `cargo test --manifest-path opencv-rust/Cargo.toml --workspace --all-targets --locked` | 54 passed: 33 portable tests, eight native ABI layouts, twelve native integrations, one 95-symbol link test |
| `cargo test --manifest-path opencv-rust/Cargo.toml -p opencv-core --no-default-features` | 31 passed |
| `cargo check --manifest-path opencv-rust/Cargo.toml --workspace --exclude opencv-sys --all-features --locked` | passed, including Python/WASM SDK features; native CUDA is a separate hardware feature |
| `cargo check --manifest-path opencv-rust/Cargo.toml -p opencv-wasm --target wasm32-unknown-unknown --locked` | passed |
| `cargo test --manifest-path opencv-rust/Cargo.toml -p opencv-sys --locked` | 21 tests passed against actual OpenCV 4.10.0 |
| `cargo package --manifest-path cuda-wasm/Cargo.toml --allow-dirty --locked` | passed archive compilation |
| `cargo package --manifest-path opencv-rust/Cargo.toml -p opencv-core --allow-dirty --locked` | passed archive compilation |
| `cargo package --manifest-path opencv-rust/Cargo.toml --workspace --allow-dirty --locked` | all four archives compiled; no publication or verification bypass |
| `node opencv-rust/tests/browser-smoke.cjs /workspace/scratch/opencv-browser-pkg` | eight actual Chromium/WASM checks passed |
| `cargo package --list --allow-dirty --locked` with each release manifest/package | all four source listings passed; a listing alone is not archive verification |
| `cargo audit --file cuda-wasm/Cargo.lock --no-fetch --json` | zero vulnerabilities; unmaintained warnings above |
| `cargo audit --file opencv-rust/Cargo.lock --no-fetch --json` | zero vulnerabilities; unmaintained warning above |
| `python3 -m unittest discover -s opencv-rust/tests -v` | five release-script security/control-flow tests passed with stub Cargo, no registry operations |
| `git diff --check -- cuda-wasm opencv-rust` | passed |

GPU-dependent test functions return early without an opted-in adapter. They appear
as passes in the test harness; the counts above do not demonstrate GPU execution.
Some existing Nutanix integration fixtures define local fake types rather than
exercise a real remote service. No hardware speedups are inferred from these tests.
Builds still emit existing documentation/dead-code warnings; warnings were not
suppressed to produce a clean-looking result.

Detailed execution logs and audit JSON are in `/workspace/scratch` for this
session (`cuda-default-final.log`, `cuda-minimal-final.log`, `cuda-wasm-target.log`,
`cuda-nutanix-check.log`, `opencv-test-final.log`, `opencv-allfeatures-final.log`,
`opencv-wasm-final.log`, `*-package-final.log`, `audit-*-final.json`).

## Release safeguards and remaining work

The OpenCV publish scripts fail on failed tests/checks/packaging, default to
validation, and require explicit `--publish` for registry writes. They never
execute `.env`, print tokens or run `cargo login`. The simple script delegates to
the same implementation. Tests use a stub executable to verify dry-run behavior,
error propagation and explicit publishing order. The entire workspace is
packaged and verified before any registry mutation, using Cargo's staged local
dependencies. No `--no-verify` workaround is used.

Matching Apache license text is included in every OpenCV package; existing license
metadata is preserved. Native CUDA/HIP accelerator kernels still need compatible
vendor-toolchain and physical-device validation. Browser pixel execution and the
native OpenCV shim are now tested. Upstream OpenCV/OS codec security updates are
separate from Cargo advisory coverage. The SDK Python API and portable WASM filters
are intentionally limited as described in their READMEs.

The family and all four OpenCV package READMEs now explain actual supported
behavior with small examples and test commands, and link the shared animated
package diagram. Historical design documents are not treated as implementation
evidence.

## Primary research and implementation decisions

- [Rust allocation contracts](https://doc.rust-lang.org/std/alloc/index.html) and
  [`copy_nonoverlapping` safety](https://doc.rust-lang.org/std/ptr/fn.copy_nonoverlapping.html):
  preserve exact allocation layouts, initialize values before safe references,
  and make caller-supplied raw pointer obligations explicit.
- [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
  and [best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/):
  host, pinned, managed and device allocations have distinct contracts. A host
  allocation and a no-op dispatch cannot substantiate a CUDA speed claim.
- [WGSL specification](https://www.w3.org/TR/WGSL/) and
  [Naga validation API](https://wgpu.rs/doc/naga/valid/index.html): generated WGSL
  needs structural/type validation and real adapter execution. Existing wgpu
  compilation captures shader/pipeline validation errors. Registry research found
  wgpu 30.0.1 compared with this code's 0.19 API; that migration requires shader,
  device, async and browser compatibility testing rather than a blind major bump.
- [OpenCV Mat reference](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html):
  matrix creation, ROI and conversion must operate on pixels. This implementation
  intentionally documents copied ROI and a single-channel subset rather than
  claiming native OpenCV's complete ownership/stride/API semantics.
- [FlashAttention upstream](https://github.com/Dao-AILab/flash-attention) and
  [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608): fused accelerator
  kernels, hardware-specific scheduling and reported accelerator throughput are
  distinct from a tiled CPU reference. Scratch reuse is implemented here; no
  cross-framework throughput claim or state-of-the-art GPU implementation is made.

The next performance loop should first compare numerical outputs and invalid
input behavior, then measure allocation count, wall-clock latency and peak memory
on named hardware with fixed shapes and seeds. Major GPU API migrations and new
fused kernels should have those gates before a capability is advertised.

### Additional native/browser evidence

Native environment: Debian 13, OpenCV 4.10.0+dfsg-5, GCC C++17 and libclang 19.
The verified prefix helper was executed with `--software-vulkan`; vulkaninfo
reports Mesa 25.0.7 llvmpipe, LLVM 19.1.7, Vulkan 1.4.305 and a CPU device. This is
software compute, not physical GPU evidence. The main mission separately tests
native/browser WGSL compute against reference outputs.

Actual browser build: wasm-pack 0.15.1 with official prebuilt wasm-bindgen
0.2.130, `build --dev --target web --out-name opencv_wasm --out-dir
/workspace/scratch/opencv-browser-pkg`. Chromium 156.0.8078.79 executes the generated
WASM; no JavaScript substitute implements the tested matrix operations.

Additional logs: `native-prefix-setup.log`, `opencv-native-tests.log`,
`opencv-native-workspace.log`, `opencv-native-cuda-gate.log` (expected unsupported
CUDA build failure), `opencv-workspace-package-native.log`,
`opencv-browser-build.log`, `opencv-browser-tests.log`,
`audit-opencv-native-final.json` and `vulkan-software-summary.log` under
`/workspace/scratch`. The final OpenCV lock still has zero known vulnerabilities
and its existing unmaintained paste warning; no advisory was suppressed.

### Benchmark blockers closed

The CUDA all-target check exposed six obsolete benchmark files. Several referenced
nonexistent runtime, pinned-memory and allocation-strategy APIs. The former
WASM/native comparison supplied invented CUDA durations, and the regression code
estimated standard deviation as 5% of one duration. An autodiscovered benchmark
also failed a machine-dependent monitoring-overhead ratio (142.86x against a tiny
empty baseline).

Those files now measure real implemented CPU allocation/copy/pool operations,
KernelFunction dispatch, reference attention, parser/code generation, scalar/SIMD
vectors and monitoring around the same real workload. Criterion supplies measured
distributions and saved-baseline comparison. `wasm_vs_native_benchmarks` is renamed
`cpu_vector_benchmarks`; unsupported CUDA/WASM ratios and ad hoc wall-time test
thresholds are removed. This fixes the target set rather than disabling benchmark
compilation. Default all-target execution passes 744 tests and 40 Criterion smoke
cases; the optional regression target passes two additional smoke cases. No actual
performance comparison or accelerator speedup is claimed.

The native bridge's all-target source also compiles with Rust/Cargo 1.88.0, now the
explicit minimum version for all four OpenCV packages. Log:
`/workspace/scratch/opencv-native-msrv.log`. Additional CUDA logs are
`cuda-alltargets-final.log`, `cuda-minimal-alltargets-check.log`,
`cuda-feature-matrix-check.log` and `cuda-regression-bench-smoke.log`.
