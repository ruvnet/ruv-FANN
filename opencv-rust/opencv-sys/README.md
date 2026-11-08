# opencv-sys

![Package roles](../../docs/assets/package-map.svg)

A native C ABI bridge to a real OpenCV **4.4 or newer** C++ installation. The
package builds its included C++ source, generates Rust declarations from one C
header, and links the system OpenCV libraries. It does not replace OpenCV with
Rust approximations. The staged package version is `0.2.0`.

## Build

Install a C++17 compiler, pkg-config, OpenCV development files and libclang. On
Debian/Ubuntu the relevant packages are `g++`, `pkg-config`, `libopencv-dev` and
`libclang-dev`. `pkg-config --modversion opencv4` must work.

For a Debian 13 development machine without administrator access, this
repository supplies a user-owned prefix helper. It verifies Debian's signed
package index and archive checksums, extracts files without changing the system,
and reports missing runtime libraries:

```bash
python3 opencv-rust/scripts/setup-native-debian.py /path/to/native-prefix
source opencv-rust/scripts/native-env.sh /path/to/native-prefix
cargo test --manifest-path opencv-rust/Cargo.toml -p opencv-sys --locked
```

Ordinary system installations do not need the prefix helper. Cross compilation
needs the target system's OpenCV installation and pkg-config configuration.
Browser builds should use the separate `opencv-wasm` package.

## Use the native matrix API

```rust
use opencv_sys::*;

// SAFETY: the byte slice is live for the import, the returned handle is owned
// here, and it is deleted exactly once after the last call.
unsafe {
    let pixels = [10u8, 20, 30, 40];
    let image = cv_Mat_from_bytes(pixels.as_ptr(), pixels.len(), 2, 2, 0); // CV_8UC1
    assert!(!image.is_null());
    assert_eq!(cv_Mat_rows(image), 2);
    cv_Mat_delete(image);
}
```

The binding covers matrix allocation/import/clone/copy/conversion/ROI, blur,
Gaussian blur, Canny, resize, color conversion, threshold, image codecs, video
capture, HighGUI, SIFT, cascade classification, camera calibration, solvePnP and
SVM training/prediction. Point, rectangle, keypoint and nested point vectors have
owned constructors, accessors and destructors. It exposes these operations, not
OpenCV's entire upstream API.

## Ownership and errors

- Every opaque handle needs its matching delete function exactly once. Delete
  accepts NULL. Foreign or dangling pointers are invalid. Calls require external
  synchronization; HighGUI also needs a running display and its main-thread rules.
- Imports copy pixels. Native ROI shares OpenCV's refcounted storage with its
  parent and remains allocated after parent deletion. Writes through ROI alias
  the parent. Clone makes an independent copy. This differs from this repository's
  portable `opencv-core` ROI, which copies immediately.
- Pixel pointers are borrowed and can expire on reallocation/deletion. ROI rows
  can have a stride: use `cv_Mat_step`, `cv_Mat_row_data` or `cv_Mat_copy_data`.
  Raw byte buffers must be large enough and must not overlap. Constructors and
  explicit resize sizes are checked against a 1 GiB bridge budget; OpenCV's other
  algorithms also use their own working memory.
- C++ exceptions are caught. Status calls return 0 or -1; boolean calls return
  0/1 or -1; pointer failures return NULL; floating failures return NaN; size
  failures return `SIZE_MAX`. Read `cv_last_error()` immediately on the same thread.
  Its message is thread-local and changes on the next operation.
- Video devices, GUI displays, cascade model files and codec availability depend
  on your OpenCV build and environment. A callable symbol is not proof that the
  corresponding physical device or display exists.

The optional `cuda` feature compiles real `cv::cuda::GpuMat` and CUDA filter
calls. It **fails at build time** when OpenCV lacks `cudafilters`; constructing a
GPU handle also fails when no CUDA device is available. Debian's tested OpenCV
build has no CUDA filters. No CUDA execution is claimed. `contrib` is retained as
an empty compatibility feature; it adds no bindings or capability claim.

Native tests run real pixels through filters, PNG/video reading, SIFT, SVM,
calibration and solvePnP, and check ownership, strides, invalid inputs and error
translation. A separate test links all 95 baseline C entry points. The packaged
archive includes the C++ implementation and header and undergoes Cargo's normal
verification. See the [review report](../../docs/review-cuda-cv.md).
