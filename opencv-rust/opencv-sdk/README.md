# OpenCV SDK for Rust

[![Crates.io](https://img.shields.io/crates/v/opencv-sdk.svg)](https://crates.io/crates/opencv-sdk)
[![Documentation](https://docs.rs/opencv-sdk/badge.svg)](https://docs.rs/opencv-sdk)
[![License](https://img.shields.io/crates/l/opencv-sdk.svg)](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE)

OpenCV SDK provides a compatibility layer for using OpenCV's Rust implementation with existing C/C++ and Python code. This crate bridges the gap between Rust's safety and performance with legacy codebases that expect OpenCV's traditional API.

## Features

- **FFI Compatibility**: Seamless integration with C/C++ codebases
- **Python Bindings**: Optional PyO3-based Python interface
- **WebAssembly Support**: Optional WASM compilation for browser deployment
- **Type Safety**: Rust's ownership system with familiar OpenCV patterns
- **Zero-Copy Operations**: Efficient data sharing between languages

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
opencv-sdk = "4.8.0"

# Optional features
opencv-sdk = { version = "4.8.0", features = ["python", "wasm"] }
```

## Usage

### Rust API

```rust
use opencv_sdk::prelude::*;
use opencv_core::{Mat, Size, MatType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new matrix
    let mat = Mat::new_size(Size::new(640, 480), MatType::CV_8U)?;
    
    // Use the SDK for interop
    let sdk_mat = opencv_sdk::from_mat(&mat)?;
    
    Ok(())
}
```

### C/C++ Interop

```c
#include <opencv_sdk.h>

int main() {
    // Create matrix using SDK
    cv_Mat* mat = cv_Mat_new();
    cv_Mat_new_size_type(mat, 640, 480, CV_8U);
    
    // Use with existing OpenCV C code
    process_image(mat);
    
    // Cleanup
    cv_Mat_delete(mat);
    return 0;
}
```

### Python Bindings (with `python` feature)

```python
import opencv_sdk

# Create a matrix
mat = opencv_sdk.Mat(640, 480)

# Compatible with numpy
import numpy as np
np_array = mat.to_numpy()

# Process with existing Python OpenCV code
processed = cv2.blur(np_array, (5, 5))
```

## Features

### Default Features
- Core SDK functionality
- C/C++ FFI bindings
- Basic type conversions

### Optional Features
- `python`: Enable Python bindings via PyO3
- `wasm`: Enable WebAssembly support
- `full`: Enable all optional features

## Architecture

The SDK provides three layers of compatibility:

1. **Rust Layer**: Native Rust API using `opencv-core`
2. **FFI Layer**: C-compatible interface for interop
3. **Binding Layer**: Language-specific bindings (Python, WASM)

## Safety

This crate maintains Rust's safety guarantees while providing FFI:
- All FFI functions validate inputs
- Memory is managed automatically where possible
- Clear ownership semantics for shared data

## Examples

See the [examples](https://github.com/ruvnet/ruv-FANN/tree/main/opencv-rust/examples) directory for more usage patterns:
- `c_interop.c`: C/C++ integration example
- `python_binding.py`: Python usage example
- `wasm_demo.html`: WebAssembly in browser

## Contributing

Contributions are welcome! Please see our [contributing guidelines](https://github.com/ruvnet/ruv-FANN/blob/main/CONTRIBUTING.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE) file for details.

## Related Crates

- [`opencv-core`](https://crates.io/crates/opencv-core): Core OpenCV data structures
- [`opencv-wasm`](https://crates.io/crates/opencv-wasm): WebAssembly bindings