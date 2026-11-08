# Native ABI and browser lifetime reference

Read `opencv-rust/opencv-sys/README.md`, `opencv-rust/opencv-sys/src/wrapper.h` and included C++ implementations before changing an ABI. The native bridge exposes a documented subset, not all upstream OpenCV. Baseline linkage includes 95 C symbols in the reviewed checkout.

## Native return conventions

| Function kind | Failure |
| --- | --- |
| Status | `-1` (success `0`) |
| Boolean | `-1` (valid `0`/`1`) |
| Pointer | NULL |
| Floating value | NaN |
| Size | `SIZE_MAX` |

Read the thread-local `cv_last_error()` before another operation replaces its message. Destructors accept NULL; foreign/dangling pointers remain invalid. Catch exceptions at every C++/C boundary. Do not permit Rust unwinding or C++ exceptions to cross the ABI.

## Required test cases

- Import, clone, convert and delete valid matrices with exact pixel checks.
- Native ROI with padded stride, parent deletion, aliasing writes and independent clone.
- Bounds, unsupported types, overflows, insufficient buffers and failure sentinels.
- Geometry/vector allocation and nested ownership using matching destructors.
- Real algorithm results and linking all expected ABI symbols.
- Browser image data retained after source frees and after WASM memory growth.
- Cross-origin/canvas errors, tall-image WASM32 arithmetic and rejected unsupported filters.

## Environment boundaries

OpenCV image codecs, cascade files, displays and video devices depend on the actual system. System library/driver security updates are outside Cargo dependency auditing. Do not enable a camera or GUI merely to complete portable unit tests. Test devices only when part of the user's authorized task.

Cross compilation requires the target's OpenCV installation and correct pkg-config paths. Browser builds use `opencv-wasm`, not the native bridge. SDK Python compilation additionally requires Python development support and covers only its small declared module.
