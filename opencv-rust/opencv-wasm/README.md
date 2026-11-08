# opencv-wasm

![Package roles](../../docs/assets/package-map.svg)

Experimental browser bindings for the single-channel `opencv-core` matrix type. This is a subset of OpenCV-style operations.

Implemented operations include matrix creation, dimensions, clone, copied regions, geometry wrappers and nearest-neighbor resize. Canvas import converts RGBA pixels to grayscale with integer luma weights `(77R + 150G + 29B + 128) / 256`, ignoring alpha. Canvas export expands each grayscale pixel to opaque RGBA.

`blur`, `gaussian_blur` and `canny` are not implemented and return errors. They no longer return an unchanged image that could be mistaken for a processed result.

## Build locally

```bash
rustup target add wasm32-unknown-unknown
cargo check --manifest-path opencv-rust/Cargo.toml -p opencv-wasm --target wasm32-unknown-unknown
# With wasm-pack installed, from this package directory:
wasm-pack build --target web
```

The generated `pkg` directory is a build artifact; do not assume an npm package contains it without inspecting the archive. The standard Rust allocator is used. SIMD capability reports whether this build has the `simd128` target feature; it does not claim every operation is vectorized.

A browser can import the generated module and call `mat_from_canvas`, `resize` and `mat_to_canvas`. Canvas reads can fail for cross-origin images or unavailable 2D contexts. Those failures return JavaScript errors.

Native tests cover pixel mapping arithmetic. The browser smoke test runs actual
WASM in Chromium and checks RGBA conversion, canvas roundtrips, ROI, resize,
invalid input, tall-image WASM32 arithmetic and exported pixel lifetimes.
`to_image_data` copies pixels to JavaScript-owned storage, so later Rust frees or
WASM memory growth cannot corrupt/detach the returned image.

```bash
# From this package directory, after installing wasm-pack:
wasm-pack build --dev --target web --out-name opencv_wasm --out-dir /path/to/pkg
# From the repository root, with Playwright installed:
node opencv-rust/tests/browser-smoke.cjs /path/to/pkg
```

The test uses Playwright-managed Chromium by default. Set `CHROMIUM_BIN` to
explicitly select another executable, for example `/usr/bin/chromium`.
Browser execution is tested separately from native target compilation. See the [workspace README](../README.md) and [review report](../../docs/review-cuda-cv.md).
