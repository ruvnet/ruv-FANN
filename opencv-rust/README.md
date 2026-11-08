# OpenCV-style Rust packages and a native OpenCV bridge

![Repository packages](../docs/assets/package-map.svg)

This workspace offers two separate implementations: small portable Rust matrix
primitives, and a low-level bridge that calls an installed OpenCV C++ library.
The portable packages are not a complete OpenCV replacement.

| Package | Staged version | Available behavior |
| --- | --- | --- |
| [opencv-core](opencv-core/) | 4.9.0 | Portable single-channel matrices, scalar fill, copied ROI, numeric conversion and geometry |
| [opencv-wasm](opencv-wasm/) | 4.9.0 | Browser grayscale/canvas conversion, copied ROI, nearest-neighbor resize and geometry |
| [opencv-sdk](opencv-sdk/) | 4.9.0 | Rust re-exports, two C exports and an experimental Python module |
| [opencv-sys](opencv-sys/) | 0.2.0 | Real OpenCV C++ matrix, image, video, feature, calibration and SVM calls through an exception-safe C ABI |

## Use a portable local matrix

```toml
[dependencies]
opencv-core = { path = "./opencv-rust/opencv-core" }
```

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

Portable matrix construction validates dimensions, primitive types and allocation
sizes, with a 1 GiB limit. ROI copies pixels. Conversion transforms values and
allocates the destination element width. Half-float raw bits can be imported as
`u16`; scalar half-float conversion is unsupported. Bytes use native endian order.

Native `opencv-sys` supports OpenCV's multichannel matrices and shared, refcounted
ROI through an unsafe raw API. It requires a C++17 compiler, pkg-config, libclang
and OpenCV >= 4.4 development files. Its README explains installation, errors and
ownership. The portable packages do not depend on this native bridge.

## Verify

```bash
# Portable packages; no native OpenCV installation needed:
cargo test --manifest-path opencv-rust/Cargo.toml --workspace --exclude opencv-sys --all-targets --locked
cargo check --manifest-path opencv-rust/Cargo.toml --workspace --exclude opencv-sys --all-features --locked
# With the native prerequisites installed:
cargo test --manifest-path opencv-rust/Cargo.toml --workspace --all-targets --locked
cargo audit --file opencv-rust/Cargo.lock
```

Python compilation needs Python development support. Browser tests build real
WASM and run pixel/canvas/lifetime checks in Chromium; see the browser package
README. Native tests execute against OpenCV 4.10.0. Software Vulkan validation
uses Mesa llvmpipe; it does not demonstrate physical CUDA hardware.

The portable WASM blur, Gaussian blur and Canny functions return unsupported
errors. Their native bridge counterparts call OpenCV's implemented algorithms.
The SDK's small Python API does not provide OpenCV Python feature parity.

`publish.sh` validates by default, propagates failures and uses Cargo's configured
credentials. It never executes `.env` or prints tokens. Registry publishing needs
explicit `--publish`. The [review report](../docs/review-cuda-cv.md) records archive
verification, staged dependency handling, migrations and test evidence.
