---
name: opencv
description: "Implement and test opencv-rust portable matrices, browser image operations, or the real native OpenCV C ABI with correct pixel ownership, resource bounds, and error handling. Use for image processing, ROI/stride bugs, WASM canvas lifetimes, native setup, or SDK integration."
---

# Choose the image-processing implementation

Read `opencv-rust/README.md` and the target package README in the selected checkout. Portable Rust, browser WASM, SDK exports and the native OpenCV bridge have distinct APIs and ownership rules. Start with the smallest surface that supplies the requested operation.

| Package | Implemented scope |
| --- | --- |
| `opencv-core` | Portable single-channel matrices, fill, copied ROI, conversion and geometry |
| `opencv-wasm` | Browser grayscale/canvas conversion, copied ROI, nearest-neighbor resize and geometry |
| `opencv-sdk` | Small Rust/C/Python surface; no upstream Python API parity |
| `opencv-sys` | Unsafe C ABI over real installed OpenCV C++ algorithms and handles |

## Establish pixel-level expectations

For a local portable consumer:

```rust
use opencv_core::{Mat, MatType, Rect};

fn main() -> opencv_core::Result<()> {
    let image = Mat::from_slice(&[10u8, 20, 30, 40], 2, 2, MatType::CV_8U)?;
    let crop = image.roi(Rect::new(1, 0, 1, 2))?;
    assert_eq!(crop.data_bytes(), &[20, 40]);
    let mut normalized = Mat::new()?;
    image.convert_to(&mut normalized, MatType::CV_32F, 1.0 / 255.0, 0.0)?;
    assert_eq!(normalized.data_bytes().len(), 16);
    Ok(())
}
```

Use a local path dependency to the selected `opencv-rust/opencv-core`. Confirm actual values, type widths and ROI semantics; matching byte length alone is insufficient for a new conversion implementation.

## Matrix and ownership procedure

1. Record rows, columns, channels, scalar type, stride and expected allocation bytes. Use checked arithmetic before allocating/copying and keep the current 1 GiB construction budget.
2. Portable matrices accept supported sealed primitive types and single channels. ROI copies pixels. Bytes use native endian order. Half-float bits can import as `u16`; scalar half-float conversion is unsupported.
3. Native `opencv-sys` imports copy input pixels, but ROI shares refcounted storage. ROI writes alias its parent and the ROI remains allocated after parent deletion. Use clone for an independent native copy.
4. Delete each owned opaque native handle exactly once with its matching destructor. Borrowed pixels expire on reallocation/deletion. Respect row stride; do not assume ROI memory is contiguous.
5. Keep raw buffers live and nonoverlapping with correct ABI/size. Synchronize foreign handles externally. GUI/display and video/camera access need their actual environment and applicable thread rules.
6. Check the documented error sentinel and read `cv_last_error()` immediately on the same thread. Do not turn a native exception into a successful empty image.

## Browser image contract

Canvas import converts RGBA to grayscale using integer luma `(77R + 150G + 29B + 128) / 256` and ignores alpha. Export expands each grayscale pixel to opaque RGBA. `to_image_data` copies into JavaScript-owned storage so Rust frees and WASM memory growth cannot corrupt/detach the returned image.

Portable `blur`, `gaussian_blur` and `canny` return unsupported errors. Their native bridge counterparts call OpenCV. Cross-origin canvas reads or missing 2D contexts also return errors; never return the input unchanged and call the filter complete.

## Verify the selected scope

From the checkout root, without native OpenCV:

```sh
cargo test --manifest-path opencv-rust/Cargo.toml --workspace --exclude opencv-sys --all-targets --locked
cargo check --manifest-path opencv-rust/Cargo.toml --workspace --exclude opencv-sys --all-features --locked
```

For native coverage, first verify C++17, pkg-config, libclang and OpenCV >=4.4 development files. `pkg-config --modversion opencv4` must work. A Debian 13 user-owned prefix helper exists, but install/bootstrap it only when necessary within the authorized environment:

```sh
python3 opencv-rust/scripts/setup-native-debian.py "/path/to/new-native-prefix"
source opencv-rust/scripts/native-env.sh "/path/to/new-native-prefix"
cargo test --manifest-path opencv-rust/Cargo.toml --workspace --all-targets --locked
```

Ordinary configured system installations need no prefix helper. The plugin's `opencv` verification profile requires native prerequisites; inspect `plan` before choosing it. A portable pass must be labeled portable.

For browser changes, from `opencv-rust/opencv-wasm/` with the existing toolchain:

```sh
wasm-pack build --dev --target web --out-name opencv_wasm --out-dir "/path/to/new-browser-pkg"
```

Then from the checkout root, with Playwright Chromium installed:

```sh
node opencv-rust/tests/browser-smoke.cjs "/path/to/new-browser-pkg"
```

## Acceptance and diagnosis

Require pixel-value tests, bad dimensions/types, crop/stride behavior, conversion widths and ownership/lifetime coverage. Browser changes need real canvas/WASM checks. Native ABI changes need actual C++ linkage and exception/error tests, not Rust declarations alone.

If native linking fails, check pkg-config and target-specific libraries before editing source. `opencv-sys`'s optional `cuda` feature needs upstream `cudafilters` plus real hardware; a CPU-only OpenCV build must reject it. No physical CUDA claim follows from software Vulkan or a compiled symbol. Read [native error and lifecycle details](references/ownership.md) when touching FFI.
