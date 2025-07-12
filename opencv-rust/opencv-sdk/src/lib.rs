//! OpenCV SDK - C/C++/Python API compatibility layer

pub use opencv_core::*;

/// SDK version information
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the OpenCV SDK
pub fn init() -> Result<()> {
    opencv_core::init()?;
    Ok(())
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn opencv_sdk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_init, m)?)?;
    m.add_class::<PyMat>()?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_init() -> PyResult<()> {
    init().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyclass]
struct PyMat {
    inner: Mat,
}

/// C API exports
#[no_mangle]
pub extern "C" fn opencv_sdk_init() -> i32 {
    match init() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn opencv_sdk_version() -> *const u8 {
    SDK_VERSION.as_ptr()
}