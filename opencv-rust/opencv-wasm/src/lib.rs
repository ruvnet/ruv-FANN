//! OpenCV WebAssembly bindings for browser deployment

use wasm_bindgen::prelude::*;
use opencv_core::{Mat, MatType, Size, Point, Rect};
use web_sys::{ImageData, CanvasRenderingContext2d, HtmlCanvasElement};

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Macro for console.log
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub fn init_opencv_wasm() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    console_log!("OpenCV WASM initialized successfully!");
}

/// WebAssembly-compatible Mat wrapper
#[wasm_bindgen]
pub struct WasmMat {
    inner: Mat,
}

#[wasm_bindgen]
impl WasmMat {
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32) -> Result<WasmMat, JsValue> {
        let size = Size::new(width, height);
        let mat = Mat::new_size(size, MatType::CV_8U)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(WasmMat { inner: mat })
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> i32 {
        self.inner.cols()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> i32 {
        self.inner.rows()
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> i32 {
        self.inner.channels()
    }

    #[wasm_bindgen(getter)]
    pub fn empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clone the matrix
    pub fn clone(&self) -> Result<WasmMat, JsValue> {
        let cloned = self.inner.clone()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmMat { inner: cloned })
    }

    /// Get region of interest
    pub fn roi(&self, x: i32, y: i32, width: i32, height: i32) -> Result<WasmMat, JsValue> {
        let rect = Rect::new(x, y, width, height);
        let roi_mat = self.inner.roi(rect)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmMat { inner: roi_mat })
    }

    /// Expand the grayscale image into opaque RGBA canvas pixels.
    pub fn to_image_data(&self) -> Result<ImageData, JsValue> {
        let len = self.inner.total().checked_mul(4)
            .filter(|&len| len <= opencv_core::MAX_MAT_BYTES)
            .ok_or_else(|| JsValue::from_str("RGBA image exceeds allocation limit"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|e| JsValue::from_str(&e.to_string()))?;
        for &gray in self.inner.data_bytes() { data.extend_from_slice(&[gray, gray, gray, 255]); }
        // ImageData retains the supplied typed array. A borrowed WASM-memory
        // view would outlive this Vec and become dangling or detached after
        // allocation/memory growth. Copy into JavaScript-owned storage first.
        let pixels = js_sys::Uint8ClampedArray::new_with_length(len as u32);
        pixels.copy_from(&data);
        ImageData::new_with_js_u8_clamped_array_and_sh(&pixels, self.width() as u32, self.height() as u32)
    }

    /// Convert RGBA canvas pixels to single-channel grayscale.
    ///
    /// Uses integer luma weights (77R + 150G + 29B + 128) / 256; alpha is ignored.
    pub fn from_image_data(image_data: &ImageData) -> Result<WasmMat, JsValue> {
        let width = i32::try_from(image_data.width()).map_err(|_| JsValue::from_str("Image width exceeds i32"))?;
        let height = i32::try_from(image_data.height()).map_err(|_| JsValue::from_str("Image height exceeds i32"))?;
        let mut mat = Mat::new_size(Size::new(width, height), MatType::CV_8U)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rgba = image_data.data();
        for (gray, pixel) in mat.data_bytes_mut().iter_mut().zip(rgba.0.chunks_exact(4)) {
            *gray = ((77 * pixel[0] as u32 + 150 * pixel[1] as u32 + 29 * pixel[2] as u32 + 128) >> 8) as u8;
        }
        Ok(WasmMat { inner: mat })
    }

}

/// WebAssembly-compatible Point wrapper
#[wasm_bindgen]
pub struct WasmPoint {
    inner: Point,
}

#[wasm_bindgen]
impl WasmPoint {
    #[wasm_bindgen(constructor)]
    pub fn new(x: i32, y: i32) -> WasmPoint {
        WasmPoint {
            inner: Point::new(x, y),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> i32 {
        self.inner.x
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> i32 {
        self.inner.y
    }

    pub fn distance_to(&self, other: &WasmPoint) -> f64 {
        self.inner.distance_to(&other.inner)
    }

    pub fn dot(&self, other: &WasmPoint) -> i32 {
        self.inner.dot(&other.inner)
    }
}

/// WebAssembly-compatible Size wrapper
#[wasm_bindgen]
pub struct WasmSize {
    inner: Size,
}

#[wasm_bindgen]
impl WasmSize {
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32) -> WasmSize {
        WasmSize {
            inner: Size::new(width, height),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> i32 {
        self.inner.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> i32 {
        self.inner.height
    }

    pub fn area(&self) -> i32 {
        self.inner.area()
    }
}

// These operations are not implemented. Return errors rather than silently
// returning the source image, which callers could mistake for processed output.
#[wasm_bindgen]
pub fn blur(_src: &WasmMat, _kernel_size: i32) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("Box blur is not implemented in this experimental binding"))
}

#[wasm_bindgen]
pub fn gaussian_blur(_src: &WasmMat, _kernel_size: i32, _sigma: f64) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("Gaussian blur is not implemented in this experimental binding"))
}

#[wasm_bindgen]
pub fn canny(_src: &WasmMat, _threshold1: f64, _threshold2: f64) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("Canny edge detection is not implemented in this experimental binding"))
}

/// Resize a nonempty grayscale image with nearest-neighbor sampling.
#[wasm_bindgen]
pub fn resize(src: &WasmMat, width: i32, height: i32) -> Result<WasmMat, JsValue> {
    if src.empty() || width <= 0 || height <= 0 {
        return Err(JsValue::from_str("Resize requires a nonempty image and positive dimensions"));
    }
    let mut mat = Mat::new_size(Size::new(width, height), MatType::CV_8U)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    for y in 0..height as usize {
        let sy = (y as u64 * src.height() as u64 / height as u64) as usize;
        for x in 0..width as usize {
            let sx = (x as u64 * src.width() as u64 / width as u64) as usize;
            mat.data_bytes_mut()[y * width as usize + x] = src.inner.data_bytes()[sy * src.width() as usize + sx];
        }
    }
    Ok(WasmMat { inner: mat })
}

// Utility functions
#[wasm_bindgen]
pub fn mat_from_canvas(canvas: &HtmlCanvasElement) -> Result<WasmMat, JsValue> {
    let context = canvas.get_context("2d")?
        .ok_or_else(|| JsValue::from_str("Canvas 2D context unavailable"))?
        .dyn_into::<CanvasRenderingContext2d>()?;
    let image = context.get_image_data(0.0, 0.0, canvas.width() as f64, canvas.height() as f64)?;
    WasmMat::from_image_data(&image)
}

#[wasm_bindgen]
pub fn mat_to_canvas(mat: &WasmMat, canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
    let context = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("Canvas 2D context unavailable"))?
        .dyn_into::<CanvasRenderingContext2d>()?;

    let image_data = mat.to_image_data()?;
    context.put_image_data(&image_data, 0.0, 0.0)?;
    
    Ok(())
}

#[wasm_bindgen]
pub fn get_version() -> String {
    format!("OpenCV Rust WASM v{}", env!("CARGO_PKG_VERSION"))
}

#[wasm_bindgen]
pub fn check_capabilities() -> js_sys::Object {
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(&obj, &"simd".into(), &cfg!(target_feature = "simd128").into()).unwrap();
    js_sys::Reflect::set(&obj, &"threads".into(), &false.into()).unwrap();
    // Historical property name retained; this is the Rust package version.
    js_sys::Reflect::set(&obj, &"opencv_version".into(), &env!("CARGO_PKG_VERSION").into()).unwrap();
    obj
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_mat_creation() {
        let mat = WasmMat::new(640, 480).unwrap();
        assert_eq!(mat.width(), 640);
        assert_eq!(mat.height(), 480);
        assert!(!mat.empty());
    }

    #[wasm_bindgen_test]
    fn test_wasm_point_operations() {
        let p1 = WasmPoint::new(10, 20);
        let p2 = WasmPoint::new(30, 40);
        
        assert_eq!(p1.x(), 10);
        assert_eq!(p1.y(), 20);
        
        let distance = p1.distance_to(&p2);
        assert!((distance - 28.284271247461902).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn test_wasm_size() {
        let size = WasmSize::new(800, 600);
        assert_eq!(size.width(), 800);
        assert_eq!(size.height(), 600);
        assert_eq!(size.area(), 480000);
    }

    #[wasm_bindgen_test]
    fn test_mat_clone() {
        let mat = WasmMat::new(100, 100).unwrap();
        let cloned = mat.clone().unwrap();
        
        assert_eq!(mat.width(), cloned.width());
        assert_eq!(mat.height(), cloned.height());
    }

    #[wasm_bindgen_test]
    fn test_roi() {
        let mat = WasmMat::new(640, 480).unwrap();
        let roi = mat.roi(100, 100, 200, 200).unwrap();
        
        assert_eq!(roi.width(), 200);
        assert_eq!(roi.height(), 200);
    }
}
#[cfg(test)]
mod portable_tests {
    use super::*;
    #[test]
    fn resize_samples_source_pixels() {
        let source = WasmMat { inner: Mat::from_slice(&[1u8, 2, 3, 4], 2, 2, MatType::CV_8U).unwrap() };
        let result = resize(&source, 4, 2).unwrap();
        assert_eq!(result.inner.data_bytes(), &[1, 1, 2, 2, 3, 3, 4, 4]);
    }
}
