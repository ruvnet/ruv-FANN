//! OpenCV WebAssembly bindings for browser deployment

use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use opencv_core::{Mat, MatType, Size, Point, Rect};
use web_sys::{ImageData, CanvasRenderingContext2d, HtmlCanvasElement};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

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

    /// Convert to ImageData for canvas rendering
    pub fn to_image_data(&self) -> Result<ImageData, JsValue> {
        let width = self.width() as u32;
        let height = self.height() as u32;
        
        // For now, create a placeholder ImageData
        // In a full implementation, this would convert Mat data to RGBA format
        let data = vec![255u8; (width * height * 4) as usize];
        let clamped = Clamped(&data[..]);
        
        ImageData::new_with_u8_clamped_array_and_sh(clamped, width, height)
    }

    /// Create Mat from ImageData
    pub fn from_image_data(image_data: &ImageData) -> Result<WasmMat, JsValue> {
        let width = image_data.width() as i32;
        let height = image_data.height() as i32;
        
        let size = Size::new(width, height);
        let mat = Mat::new_size(size, MatType::CV_8U)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
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

// Image processing functions as standalone functions instead of module
#[wasm_bindgen]
pub fn blur(src: &WasmMat, kernel_size: i32) -> Result<WasmMat, JsValue> {
    // For now, return a clone - full implementation would call opencv_imgproc::blur
    src.clone()
}

#[wasm_bindgen]
pub fn gaussian_blur(src: &WasmMat, kernel_size: i32, sigma: f64) -> Result<WasmMat, JsValue> {
    // For now, return a clone - full implementation would call opencv_imgproc::gaussian_blur
    src.clone()
}

#[wasm_bindgen]
pub fn canny(src: &WasmMat, threshold1: f64, threshold2: f64) -> Result<WasmMat, JsValue> {
    // For now, return a clone - full implementation would call opencv_imgproc::canny
    src.clone()
}

#[wasm_bindgen]
pub fn resize(src: &WasmMat, width: i32, height: i32) -> Result<WasmMat, JsValue> {
    // Create new Mat with target size
    let size = Size::new(width, height);
    let mat = Mat::new_size(size, MatType::CV_8U)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    Ok(WasmMat { inner: mat })
}

// Utility functions
#[wasm_bindgen]
pub fn mat_from_canvas(canvas: &HtmlCanvasElement) -> Result<WasmMat, JsValue> {
    let width = canvas.width() as i32;
    let height = canvas.height() as i32;
    
    WasmMat::new(width, height)
}

#[wasm_bindgen]
pub fn mat_to_canvas(mat: &WasmMat, canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
    let context = canvas
        .get_context("2d")?
        .unwrap()
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
    js_sys::Reflect::set(&obj, &"simd".into(), &true.into()).unwrap();
    js_sys::Reflect::set(&obj, &"threads".into(), &false.into()).unwrap();
    js_sys::Reflect::set(&obj, &"opencv_version".into(), &"4.8.0".into()).unwrap();
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