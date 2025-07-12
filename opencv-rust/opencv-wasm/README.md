# OpenCV WASM for Rust

[![Crates.io](https://img.shields.io/crates/v/opencv-wasm.svg)](https://crates.io/crates/opencv-wasm)
[![Documentation](https://docs.rs/opencv-wasm/badge.svg)](https://docs.rs/opencv-wasm)
[![License](https://img.shields.io/crates/l/opencv-wasm.svg)](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE)

WebAssembly bindings for OpenCV in Rust, enabling computer vision in the browser. This crate provides a JavaScript-friendly API for OpenCV's core functionality.

## Features

- **Browser-Ready**: Compiled to WebAssembly for web applications
- **Canvas Integration**: Direct interaction with HTML5 Canvas
- **Type Safety**: Rust's safety with JavaScript interop
- **Small Size**: Optimized WASM output with wee_alloc
- **Image Processing**: Basic filters and transformations

## Installation

### Rust Project

```toml
[dependencies]
opencv-wasm = "4.8.0"
wasm-bindgen = "0.2"
```

### JavaScript/TypeScript

```bash
npm install opencv-wasm
```

## Building

Build the WASM module:

```bash
wasm-pack build --target web --out-dir pkg
```

## Usage

### Rust Side

```rust
use opencv_wasm::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn process_image(input: &WasmMat) -> Result<WasmMat, JsValue> {
    // Apply Gaussian blur
    gaussian_blur(input, 5, 1.0)
}
```

### JavaScript Side

```javascript
import init, { 
    init_opencv_wasm, 
    WasmMat, 
    mat_from_canvas,
    mat_to_canvas,
    gaussian_blur 
} from './pkg/opencv_wasm.js';

async function main() {
    // Initialize WASM module
    await init();
    init_opencv_wasm();
    
    // Get canvas element
    const canvas = document.getElementById('canvas');
    
    // Convert canvas to Mat
    const mat = await mat_from_canvas(canvas);
    
    // Apply blur
    const blurred = await gaussian_blur(mat, 5, 1.0);
    
    // Draw result back to canvas
    await mat_to_canvas(blurred, canvas);
}

main();
```

## API Overview

### Core Types

#### WasmMat
WebAssembly-compatible matrix wrapper:

```javascript
// Create new matrix
const mat = new WasmMat(640, 480);

// Properties
console.log(mat.width, mat.height, mat.channels);

// Operations
const roi = mat.roi(10, 10, 100, 100);
const clone = mat.clone();
```

#### WasmPoint
2D point for browser use:

```javascript
const pt1 = new WasmPoint(10, 20);
const pt2 = new WasmPoint(30, 40);
const distance = pt1.distance_to(pt2);
```

#### WasmSize
Dimensions container:

```javascript
const size = new WasmSize(1920, 1080);
console.log(size.area()); // 2073600
```

### Image Processing Functions

```javascript
// Blur operations
const blurred = blur(src, 5);
const gaussian = gaussian_blur(src, 5, 1.0);

// Edge detection
const edges = canny(src, 100, 200);

// Resize
const resized = resize(src, 320, 240);
```

### Canvas Integration

```javascript
// Load from canvas
const mat = await mat_from_canvas(canvas);

// Save to canvas
await mat_to_canvas(mat, canvas);

// Load from image element
const img = document.getElementById('image');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
ctx.drawImage(img, 0, 0);
const mat = await mat_from_canvas(canvas);
```

## Complete Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>OpenCV WASM Demo</title>
</head>
<body>
    <input type="file" id="fileInput" accept="image/*">
    <canvas id="canvas"></canvas>
    
    <script type="module">
        import init, { 
            init_opencv_wasm,
            mat_from_canvas,
            mat_to_canvas,
            gaussian_blur,
            canny,
            get_version
        } from './pkg/opencv_wasm.js';
        
        async function processImage(file) {
            const img = new Image();
            img.src = URL.createObjectURL(file);
            
            await img.decode();
            
            const canvas = document.getElementById('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            // Convert to Mat
            const mat = await mat_from_canvas(canvas);
            
            // Apply edge detection
            const edges = await canny(mat, 50, 150);
            
            // Display result
            await mat_to_canvas(edges, canvas);
        }
        
        async function init_app() {
            await init();
            init_opencv_wasm();
            
            console.log(get_version());
            
            document.getElementById('fileInput').addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    processImage(e.target.files[0]);
                }
            });
        }
        
        init_app();
    </script>
</body>
</html>
```

## Performance Tips

1. **Reuse Matrices**: Create once, reuse multiple times
2. **Batch Operations**: Process multiple operations before canvas update
3. **Use Web Workers**: Offload processing to background threads
4. **Optimize Size**: Use appropriate matrix types (8-bit vs 32-bit)

## Browser Compatibility

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 79+

WebAssembly SIMD support (when available) provides additional performance.

## Building from Source

```bash
# Install dependencies
cargo install wasm-pack

# Build for web
wasm-pack build --target web

# Build optimized for size
wasm-pack build --target web --release -- --features wee_alloc
```

## Limitations

- Currently supports single-channel and 3-channel 8-bit images
- Some advanced OpenCV features not yet implemented
- Performance varies by browser and hardware

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/ruvnet/ruv-FANN/blob/main/CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](https://github.com/ruvnet/ruv-FANN/blob/main/LICENSE) for details.