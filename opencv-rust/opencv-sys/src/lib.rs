//! Low-level Rust bindings for OpenCV
//! 
//! This crate provides unsafe, low-level bindings to the OpenCV C++ library.
//! Most users should use the higher-level opencv-* crates instead.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::c_void;
use libc::{c_int, c_double, c_char};

// Core Mat operations
extern "C" {
    pub fn cv_Mat_new() -> *mut cv_Mat;
    pub fn cv_Mat_new_size_type_scalar(rows: c_int, cols: c_int, type_: c_int, scalar: cv_Scalar) -> *mut cv_Mat;
    pub fn cv_Mat_from_data(data: *const c_void, rows: c_int, cols: c_int, type_: c_int) -> *mut cv_Mat;
    pub fn cv_Mat_delete(mat: *mut cv_Mat);
    pub fn cv_Mat_clone(mat: *const cv_Mat) -> *mut cv_Mat;
    pub fn cv_Mat_copyTo(src: *const cv_Mat, dst: *mut cv_Mat) -> c_int;
    pub fn cv_Mat_convertTo(src: *const cv_Mat, dst: *mut cv_Mat, rtype: c_int, alpha: c_double, beta: c_double) -> c_int;
    pub fn cv_Mat_rows(mat: *const cv_Mat) -> c_int;
    pub fn cv_Mat_cols(mat: *const cv_Mat) -> c_int;
    pub fn cv_Mat_type(mat: *const cv_Mat) -> c_int;
    pub fn cv_Mat_channels(mat: *const cv_Mat) -> c_int;
    pub fn cv_Mat_empty(mat: *const cv_Mat) -> c_int;
    pub fn cv_Mat_data(mat: *const cv_Mat) -> *mut u8;
    pub fn cv_Mat_roi(mat: *const cv_Mat, x: c_int, y: c_int, width: c_int, height: c_int) -> *mut cv_Mat;
}

// Image processing functions
extern "C" {
    pub fn cv_blur(src: *const cv_Mat, dst: *mut cv_Mat, ksize_width: c_int, ksize_height: c_int) -> c_int;
    pub fn cv_GaussianBlur(src: *const cv_Mat, dst: *mut cv_Mat, ksize_width: c_int, ksize_height: c_int, sigmaX: c_double, sigmaY: c_double) -> c_int;
    pub fn cv_Canny(image: *const cv_Mat, edges: *mut cv_Mat, threshold1: c_double, threshold2: c_double) -> c_int;
    pub fn cv_resize(src: *const cv_Mat, dst: *mut cv_Mat, dsize_width: c_int, dsize_height: c_int, fx: c_double, fy: c_double, interpolation: c_int) -> c_int;
    pub fn cv_cvtColor(src: *const cv_Mat, dst: *mut cv_Mat, code: c_int) -> c_int;
    pub fn cv_threshold(src: *const cv_Mat, dst: *mut cv_Mat, thresh: c_double, maxval: c_double, type_: c_int) -> c_double;
}

// Image I/O functions
extern "C" {
    pub fn cv_imread(filename: *const c_char, flags: c_int) -> *mut cv_Mat;
    pub fn cv_imwrite(filename: *const c_char, img: *const cv_Mat) -> c_int;
}

// Video I/O functions
extern "C" {
    pub fn cv_VideoCapture_new() -> *mut cv_VideoCapture;
    pub fn cv_VideoCapture_new_index(index: c_int) -> *mut cv_VideoCapture;
    pub fn cv_VideoCapture_new_filename(filename: *const c_char) -> *mut cv_VideoCapture;
    pub fn cv_VideoCapture_delete(cap: *mut cv_VideoCapture);
    pub fn cv_VideoCapture_isOpened(cap: *const cv_VideoCapture) -> c_int;
    pub fn cv_VideoCapture_read(cap: *mut cv_VideoCapture, image: *mut cv_Mat) -> c_int;
    pub fn cv_VideoCapture_release(cap: *mut cv_VideoCapture);
}

// GUI functions
extern "C" {
    pub fn cv_namedWindow(winname: *const c_char, flags: c_int) -> c_int;
    pub fn cv_destroyWindow(winname: *const c_char) -> c_int;
    pub fn cv_destroyAllWindows() -> c_int;
    pub fn cv_imshow(winname: *const c_char, mat: *const cv_Mat) -> c_int;
    pub fn cv_waitKey(delay: c_int) -> c_int;
}

// Feature detection
extern "C" {
    pub fn cv_SIFT_create() -> *mut cv_SIFT;
    pub fn cv_SIFT_delete(detector: *mut cv_SIFT);
    pub fn cv_SIFT_detectAndCompute(detector: *mut cv_SIFT, image: *const cv_Mat, mask: *const cv_Mat, keypoints: *mut cv_KeyPoint_vector, descriptors: *mut cv_Mat) -> c_int;
}

// Object detection
extern "C" {
    pub fn cv_CascadeClassifier_new() -> *mut cv_CascadeClassifier;
    pub fn cv_CascadeClassifier_new_filename(filename: *const c_char) -> *mut cv_CascadeClassifier;
    pub fn cv_CascadeClassifier_delete(classifier: *mut cv_CascadeClassifier);
    pub fn cv_CascadeClassifier_detectMultiScale(classifier: *mut cv_CascadeClassifier, image: *const cv_Mat, objects: *mut cv_Rect_vector) -> c_int;
}

// Calibration functions
extern "C" {
    pub fn cv_calibrateCamera(object_points: *const cv_Point3f_vector_vector, image_points: *const cv_Point2f_vector_vector, image_size: cv_Size, camera_matrix: *mut cv_Mat, dist_coeffs: *mut cv_Mat) -> c_double;
    pub fn cv_solvePnP(object_points: *const cv_Point3f_vector, image_points: *const cv_Point2f_vector, camera_matrix: *const cv_Mat, dist_coeffs: *const cv_Mat, rvec: *mut cv_Mat, tvec: *mut cv_Mat) -> c_int;
}

// Machine Learning
extern "C" {
    pub fn cv_ml_SVM_create() -> *mut cv_ml_SVM;
    pub fn cv_ml_SVM_delete(svm: *mut cv_ml_SVM);
    pub fn cv_ml_SVM_train(svm: *mut cv_ml_SVM, samples: *const cv_Mat, layout: c_int, responses: *const cv_Mat) -> c_int;
    pub fn cv_ml_SVM_predict(svm: *mut cv_ml_SVM, samples: *const cv_Mat, results: *mut cv_Mat) -> c_double;
}

// CUDA functions (when enabled)
#[cfg(feature = "cuda")]
extern "C" {
    pub fn cv_cuda_GpuMat_new() -> *mut cv_cuda_GpuMat;
    pub fn cv_cuda_GpuMat_delete(mat: *mut cv_cuda_GpuMat);
    pub fn cv_cuda_GpuMat_upload(gpu_mat: *mut cv_cuda_GpuMat, mat: *const cv_Mat) -> c_int;
    pub fn cv_cuda_GpuMat_download(gpu_mat: *const cv_cuda_GpuMat, mat: *mut cv_Mat) -> c_int;
    pub fn cv_cuda_blur(src: *const cv_cuda_GpuMat, dst: *mut cv_cuda_GpuMat, ksize_width: c_int, ksize_height: c_int) -> c_int;
    pub fn cv_cuda_GaussianBlur(src: *const cv_cuda_GpuMat, dst: *mut cv_cuda_GpuMat, ksize_width: c_int, ksize_height: c_int, sigmaX: c_double, sigmaY: c_double) -> c_int;
}

// Type definitions for commonly used structures
#[repr(C)]
pub struct cv_Mat {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_VideoCapture {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_SIFT {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_CascadeClassifier {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_ml_SVM {
    _private: [u8; 0],
}

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct cv_cuda_GpuMat {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cv_Point2i {
    pub x: c_int,
    pub y: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cv_Point2f {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cv_Point2d {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cv_Point3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cv_Size {
    pub width: c_int,
    pub height: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cv_Rect {
    pub x: c_int,
    pub y: c_int,
    pub width: c_int,
    pub height: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cv_Scalar {
    pub val: [f64; 4],
}

impl Default for cv_Scalar {
    fn default() -> Self {
        Self { val: [0.0; 4] }
    }
}

impl From<f64> for cv_Scalar {
    fn from(val: f64) -> Self {
        Self { val: [val, 0.0, 0.0, 0.0] }
    }
}

impl From<[f64; 4]> for cv_Scalar {
    fn from(val: [f64; 4]) -> Self {
        Self { val }
    }
}

// Vector types for collections
#[repr(C)]
pub struct cv_KeyPoint_vector {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_Rect_vector {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_Point2f_vector {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_Point3f_vector {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_Point2f_vector_vector {
    _private: [u8; 0],
}

#[repr(C)]
pub struct cv_Point3f_vector_vector {
    _private: [u8; 0],
}