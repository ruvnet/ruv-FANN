# opencv-core

![Package roles](../../docs/assets/package-map.svg)

A small Rust library of OpenCV-style matrix and geometry types. It currently supports single-channel matrices; it does not provide the full OpenCV image-processing API.

```rust
use opencv_core::{Mat, MatType, Rect, Scalar, Size};

fn main() -> opencv_core::Result<()> {
    let image = Mat::new_size_with_default(Size::new(4, 3), MatType::CV_8U, Scalar::all(200.0))?;
    let crop = image.roi(Rect::new(1, 1, 2, 2))?;
    assert_eq!(crop.data_bytes(), &[200; 4]);
    Ok(())
}
```

Use a path dependency on this checkout until the reviewed changes have a new verified registry release:

```toml
[dependencies]
opencv-core = { path = "./opencv-rust/opencv-core" }
```

`Mat::from_slice` accepts matching primitive element types through the sealed `MatElement` trait. It rejects mismatched representations, negative dimensions, incorrect lengths and allocations above 1 GiB. Arbitrary structs are excluded because their memory can contain padding or uninitialized bytes. Existing generic callers must add `T: MatElement`.

`new_size_with_default` fills the first scalar channel. `roi` returns an independent copy of the selected rows and columns. `convert_to` applies `alpha * source + beta`, with rounded, saturating integer output. Half-float scalar conversions return an unsupported error; raw half-float bits can be imported as `u16` with `CV_16F`.

`data_bytes` and `data_bytes_mut` expose contiguous, initialized native-endian bytes. Raw pointer consumers are responsible for lifetimes, alignment and valid typed access. A `Vec<u8>` backing store must not be assumed aligned for arbitrary typed pointer dereferences.

The default `serde` feature supports geometry serialization. It can be disabled. No SIMD speedup or binary compatibility with native OpenCV is promised.

```bash
cargo test --manifest-path opencv-rust/Cargo.toml -p opencv-core
cargo test --manifest-path opencv-rust/Cargo.toml -p opencv-core --no-default-features
```

See the [workspace README](../README.md) and [review report](../../docs/review-cuda-cv.md) for package status and license review.
