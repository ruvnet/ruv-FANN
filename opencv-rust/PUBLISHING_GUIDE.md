# Validate and release the OpenCV packages

The staged packages are `opencv-core`, `opencv-wasm` and `opencv-sdk` 4.9.0,
plus the real native `opencv-sys` bridge 0.2.0. Nothing in this guide publishes
without an explicit publish command.

## Prerequisites

Use a clean checkout and a current Cargo with workspace packaging support.
Native validation needs OpenCV >= 4.4 development files, libclang, a C++17
compiler and pkg-config. The SDK Python check needs Python development support.
For the user-owned Debian 13 setup, follow `opencv-sys/README.md` and source
`scripts/native-env.sh` with your prefix. The crate build scripts contain no
session-specific absolute paths.

Use Cargo's configured credential provider for authentication. Do not source
executable `.env` files, print tokens, write tokens into scripts, or run an
interactive login from a release script.

## Review the candidates

```bash
cargo test --workspace --all-targets --locked
cargo check --workspace --exclude opencv-sys --all-features --locked
cargo check -p opencv-sys --locked
cargo audit --file Cargo.lock
cargo package --workspace --locked
```

The native package's optional `cuda` feature requires an OpenCV build with
`cudafilters` and compatible hardware. A CPU-only OpenCV installation deliberately
fails that feature's build with an actionable message. It is not part of the
portable all-feature check and does not silently provide CPU stubs.

Package the whole workspace together. Cargo stages its members so it can verify
normalized dependencies before their new versions exist in the public registry.
The SDK header generator reads its exported source directly and writes into
`OUT_DIR`; it does not recursively invoke Cargo or modify packaged source files.
Do not bypass archive verification with `--no-verify`.

Inspect each `.crate` archive. In particular, `opencv-sys` must contain its C++
sources, `bridge.hpp`, `wrapper.h`, build script, tests and Apache license text.
All four archives were compiled by Cargo during this review; the review report
records the exact command and native environment.

Browser checks are separate from native compilation. Build `opencv-wasm` with
wasm-pack, then run `tests/browser-smoke.cjs` in Chromium. It checks actual
pixels, canvas operations, invalid inputs and exported pixel lifetimes.

## Publish explicitly

`bash publish.sh` or `bash publish.sh --dry-run` runs all gates and performs no
registry writes. Only after reviewing candidate contents and results:

```bash
bash publish.sh --publish
```

The script validates and packages every workspace member before publishing any
member. It publishes core, WASM, SDK, then the independent native bridge, keeping
Cargo verification enabled and stopping on failure. `publish-simple.sh` delegates
to the same implementation. Re-check registry state after an interrupted release
before retrying; do not assume an operation failed merely because its response
was interrupted.

The report at `../docs/review-cuda-cv.md` describes API migrations, dependency
warnings, real tested capabilities and hardware limitations. The Rust package
versions do not imply upstream OpenCV feature parity.
