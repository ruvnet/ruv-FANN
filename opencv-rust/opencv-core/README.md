# OpenCV Core for Rust

[![Crates.io](https://img.shields.io/crates/v/opencv-core.svg)](https://crates.io/crates/opencv-core)
[![Documentation](https://docs.rs/opencv-core/badge.svg)](https://docs.rs/opencv-core)
[![License](https://img.shields.io/crates/l/opencv-core.svg)](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE)

Pure Rust implementation of OpenCV's core data structures and operations. This crate provides the fundamental building blocks for computer vision applications in Rust.

## Features

- **Pure Rust**: No FFI dependencies, 100% safe Rust
- **Core Data Structures**: Mat, Point, Size, Rect, Scalar, Range
- **Type System**: Compatible with OpenCV's type conventions
- **Memory Safe**: Zero unsafe blocks in public APIs
- **Serialization**: Optional serde support
- **Performance**: Optimized with SIMD where available

## Installation

```toml
[dependencies]
opencv-core = "4.8.0"

# With serialization support
opencv-core = { version = "4.8.0", features = ["serde"] }
```

## Quick Start

```rust
use opencv_core::{Mat, MatType, Size, Point, Rect, Scalar};

fn main() -> Result<(), opencv_core::Error> {
    // Create a new 8-bit grayscale image
    let mat = Mat::new_size(Size::new(640, 480), MatType::CV_8U)?;
    
    // Create with initial value
    let white = Mat::new_size_with_default(
        Size::new(640, 480), 
        MatType::CV_8U, 
        Scalar::all(255.0)
    )?;
    
    // Access image properties
    println!("Width: {}, Height: {}", mat.cols(), mat.rows());
    println!("Channels: {}", mat.channels());
    
    // Create points and rectangles
    let pt1 = Point::new(10, 20);
    let pt2 = Point::new(100, 200);
    let rect = Rect::new(10, 10, 100, 100);
    
    // Work with regions of interest
    let roi = mat.roi(rect)?;
    
    Ok(())
}
```

## Core Types

### Mat - Matrix/Image Container
```rust
// Create matrices of different types
let mat_u8 = Mat::new_size(Size::new(640, 480), MatType::CV_8U)?;
let mat_f32 = Mat::new_size(Size::new(640, 480), MatType::CV_32F)?;
let mat_rgb = Mat::new_size(Size::new(640, 480), MatType::CV_8UC3)?;

// Clone and copy
let cloned = mat_u8.clone()?;
let mut dst = Mat::new();
mat_u8.copy_to(&mut dst)?;
```

### Point - 2D/3D Points
```rust
// 2D points
let pt2i = Point::new(10, 20);           // Point2i
let pt2f = Point2f::new(10.5, 20.5);    // Point2f
let pt2d = Point2d::new(10.5, 20.5);    // Point2d

// 3D points  
let pt3f = Point3f::new(1.0, 2.0, 3.0); // Point3f
let pt3d = Point3d::new(1.0, 2.0, 3.0); // Point3d

// Operations
let distance = pt2i.distance_to(&Point::new(20, 30));
let dot_product = pt2i.dot(&Point::new(1, 1));
```

### Size - Dimensions
```rust
let size = Size::new(640, 480);
let area = size.area();  // 307200
let aspect = size.aspect_ratio();  // 1.333...

// Float sizes
let size_f = Size2f::new(640.0, 480.0);
```

### Rect - Rectangles
```rust
let rect = Rect::new(10, 10, 100, 100);

// Properties
let center = rect.center();
let area = rect.area();
let is_inside = rect.contains(Point::new(50, 50));

// Operations
let intersection = rect1.intersection(&rect2);
let union = rect1.union(&rect2);
```

### Scalar - Multi-channel Values
```rust
// Single value for all channels
let gray = Scalar::all(128.0);

// Individual channel values
let bgr = Scalar::new(255.0, 0.0, 0.0, 0.0); // Blue in BGR

// Arithmetic
let sum = scalar1.add(&scalar2);
let product = scalar1.mul(&scalar2);
```

## Memory Management

The crate includes built-in memory management with allocation tracking:

```rust
use opencv_core::memory;

// Initialize memory system
memory::init_allocators()?;

// Check memory usage
let (bytes, count) = memory::get_memory_usage();
println!("Allocated: {} bytes in {} allocations", bytes, count);
```

## Error Handling

All operations that can fail return `Result<T, opencv_core::Error>`:

```rust
use opencv_core::{Error, Result};

fn process_image() -> Result<()> {
    let mat = Mat::new_size(Size::new(-1, -1), MatType::CV_8U)
        .map_err(|_| Error::InvalidArgument("Invalid size".into()))?;
    Ok(())
}
```

## Performance

- SIMD optimizations via `safe_arch` crate
- Efficient memory layout compatible with OpenCV
- Zero-copy operations where possible
- Lazy allocation strategies

## Compatibility

This crate aims to be compatible with OpenCV 4.x conventions while providing Rust-idiomatic APIs. Types can be converted to/from OpenCV's C++ types when using the FFI layer.

## Contributing

Contributions are welcome! Please see our [contributing guidelines](https://github.com/ruvnet/ruv-FANN/blob/main/CONTRIBUTING.md).

## License

Licensed under Apache License 2.0 - see [LICENSE](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE) for details.