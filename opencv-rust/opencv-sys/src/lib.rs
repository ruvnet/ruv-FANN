//! Low-level C ABI bridge to a system OpenCV >= 4.4 C++ installation.
//!
//! # Safety
//! All raw calls require live bridge-owned handles and correctly sized readable
//! or writable buffers. Readable input bytes must be initialized, and string
//! pointers must be NUL-terminated. Free each owning handle exactly once with its matching
//! delete function. Borrowed pixel pointers expire on deletion/reallocation and
//! may alias native ROI views; coordinate access and do not create overlapping
//! Rust references. Input and output byte buffers must not overlap. Use
//! `cv_Mat_step`/row access for strided images.
//!
//! C++ exceptions become NULL, -1, SIZE_MAX or NaN depending on the result type.
//! Boolean calls return 0/1 or -1 on error. Status calls return 0 or -1. Inspect
//! `cv_last_error()` immediately on the same thread. No exception crosses C ABI.
//! Deletion accepts NULL; foreign/dangling pointers remain caller errors.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl From<f64> for cv_Scalar {
    fn from(value: f64) -> Self {
        Self {
            val: [value, 0.0, 0.0, 0.0],
        }
    }
}
impl From<[f64; 4]> for cv_Scalar {
    fn from(val: [f64; 4]) -> Self {
        Self { val }
    }
}
