# opencv-sdk

![Package roles](../../docs/assets/package-map.svg)

A small interop layer around this repository's `opencv-core` library. It re-exports Rust matrix/geometry types and provides two C exports:

```c
#include <stdint.h>
int32_t opencv_sdk_init(void);
const uint8_t *opencv_sdk_version(void);
```

Initialization returns zero on success. The version pointer refers to a static, NUL-terminated UTF-8 string; callers must not modify or free it.

The optional `python` feature builds a PyO3 module with `py_init` and an experimental `PyMat` class. The class does not yet provide constructors, NumPy conversion, filters or a complete OpenCV Python API. The `wasm` feature includes the experimental sibling browser binding; `full` enables both.

```bash
cargo test --manifest-path opencv-rust/Cargo.toml -p opencv-sdk
cargo check --manifest-path opencv-rust/Cargo.toml -p opencv-sdk --all-features
```

The separate `opencv-sys` package supplies the real native OpenCV C++ bridge. These two SDK exports initialize and describe only the portable Rust SDK. Its C header is generated from the two export definitions into Cargo OUT_DIR, without recursive Cargo metadata or writes into packaged sources. See the [workspace README](../README.md) and [review report](../../docs/review-cuda-cv.md) for release status.
