//! Mat - The fundamental OpenCV data structure for images and matrices

use crate::{Error, Result, Size, Point, Rect, Scalar};
use std::fmt;
use std::marker::PhantomData;

/// OpenCV Mat data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum MatType {
    /// 8-bit unsigned integer
    CV_8U = 0,
    /// 8-bit signed integer
    CV_8S = 1,
    /// 16-bit unsigned integer
    CV_16U = 2,
    /// 16-bit signed integer
    CV_16S = 3,
    /// 32-bit signed integer
    CV_32S = 4,
    /// 32-bit floating-point
    CV_32F = 5,
    /// 64-bit floating-point
    CV_64F = 6,
    /// 16-bit floating-point
    CV_16F = 7,
}

impl MatType {
    /// Get the element size in bytes
    pub fn elem_size(&self) -> usize {
        match self {
            MatType::CV_8U | MatType::CV_8S => 1,
            MatType::CV_16U | MatType::CV_16S | MatType::CV_16F => 2,
            MatType::CV_32S | MatType::CV_32F => 4,
            MatType::CV_64F => 8,
        }
    }

    /// Check if the type is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, MatType::CV_32F | MatType::CV_64F | MatType::CV_16F)
    }

    /// Check if the type is signed
    pub fn is_signed(&self) -> bool {
        matches!(self, MatType::CV_8S | MatType::CV_16S | MatType::CV_32S | MatType::CV_32F | MatType::CV_64F | MatType::CV_16F)
    }
}

/// Mat structure - the fundamental data container
pub struct Mat {
    data: Vec<u8>,
    rows: i32,
    cols: i32,
    mat_type: MatType,
    channels: i32,
}

unsafe impl Send for Mat {}
unsafe impl Sync for Mat {}

impl Mat {
    /// Create a new empty Mat
    pub fn new() -> Result<Self> {
        Ok(Mat {
            data: Vec::new(),
            rows: 0,
            cols: 0,
            mat_type: MatType::CV_8U,
            channels: 1,
        })
    }

    /// Create a Mat with specified dimensions and type
    pub fn new_size(size: Size, mat_type: MatType) -> Result<Self> {
        Self::new_size_with_default(size, mat_type, Scalar::default())
    }

    /// Create a Mat with specified dimensions, type, and default value
    pub fn new_size_with_default(size: Size, mat_type: MatType, _default_value: Scalar) -> Result<Self> {
        let total_size = (size.width * size.height) as usize * mat_type.elem_size();
        Ok(Mat {
            data: vec![0u8; total_size],
            rows: size.height,
            cols: size.width,
            mat_type,
            channels: 1,
        })
    }

    /// Create a Mat from existing data
    pub fn from_slice<T>(data: &[T], rows: i32, cols: i32, mat_type: MatType) -> Result<Self> {
        let elem_size = mat_type.elem_size();
        let expected_size = (rows * cols) as usize;
        
        if data.len() != expected_size {
            return Err(Error::InvalidArgument(format!(
                "Data size mismatch: expected {}, got {}",
                expected_size,
                data.len()
            )));
        }

        let byte_size = expected_size * elem_size;
        let mut mat_data = vec![0u8; byte_size];
        
        // Copy data (simplified - assumes matching types)
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mat_data.as_mut_ptr(),
                byte_size,
            );
        }

        Ok(Mat {
            data: mat_data,
            rows,
            cols,
            mat_type,
            channels: 1,
        })
    }

    /// Get the number of rows
    pub fn rows(&self) -> i32 {
        self.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> i32 {
        self.cols
    }

    /// Get the size of the matrix
    pub fn size(&self) -> Size {
        Size::new(self.cols, self.rows)
    }

    /// Get the matrix type
    pub fn mat_type(&self) -> MatType {
        self.mat_type
    }

    /// Get the number of channels
    pub fn channels(&self) -> i32 {
        self.channels
    }

    /// Check if the matrix is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() || self.rows == 0 || self.cols == 0
    }

    /// Get a region of interest (ROI)
    pub fn roi(&self, rect: Rect) -> Result<Mat> {
        if rect.x < 0 || rect.y < 0 || 
           rect.x + rect.width > self.cols || 
           rect.y + rect.height > self.rows {
            return Err(Error::InvalidArgument("ROI out of bounds".into()));
        }

        // For now, create a new Mat with the ROI data (simplified)
        Ok(Mat {
            data: Vec::new(), // Would copy ROI data here
            rows: rect.height,
            cols: rect.width,
            mat_type: self.mat_type,
            channels: self.channels,
        })
    }

    /// Clone the matrix
    pub fn clone(&self) -> Result<Mat> {
        Ok(Mat {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
            mat_type: self.mat_type,
            channels: self.channels,
        })
    }

    /// Copy data to another matrix
    pub fn copy_to(&self, dst: &mut Mat) -> Result<()> {
        dst.data = self.data.clone();
        dst.rows = self.rows;
        dst.cols = self.cols;
        dst.mat_type = self.mat_type;
        dst.channels = self.channels;
        Ok(())
    }

    /// Convert matrix type
    pub fn convert_to(&self, dst: &mut Mat, rtype: MatType, _alpha: f64, _beta: f64) -> Result<()> {
        // Simplified conversion - just change type for now
        dst.data = self.data.clone();
        dst.rows = self.rows;
        dst.cols = self.cols;
        dst.mat_type = rtype;
        dst.channels = self.channels;
        Ok(())
    }

    /// Get raw data pointer
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get mutable raw data pointer
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// Get element size in bytes
    pub fn elem_size(&self) -> usize {
        self.mat_type.elem_size() * self.channels as usize
    }

    /// Get total number of elements
    pub fn total(&self) -> usize {
        (self.rows * self.cols) as usize
    }
}

impl Default for Mat {
    fn default() -> Self {
        Self::new().expect("Failed to create default Mat")
    }
}

impl fmt::Debug for Mat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("type", &self.mat_type)
            .field("channels", &self.channels)
            .field("empty", &self.is_empty())
            .finish()
    }
}

/// Trait for Mat-like objects
pub trait MatTrait {
    /// Get number of rows
    fn rows(&self) -> i32;
    /// Get number of columns
    fn cols(&self) -> i32;
    /// Get size
    fn size(&self) -> Size;
    /// Get matrix type
    fn mat_type(&self) -> MatType;
    /// Get number of channels
    fn channels(&self) -> i32;
    /// Check if empty
    fn is_empty(&self) -> bool;
}

impl MatTrait for Mat {
    fn rows(&self) -> i32 {
        self.rows()
    }

    fn cols(&self) -> i32 {
        self.cols()
    }

    fn size(&self) -> Size {
        self.size()
    }

    fn mat_type(&self) -> MatType {
        self.mat_type()
    }

    fn channels(&self) -> i32 {
        self.channels()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_creation() {
        let mat = Mat::new().unwrap();
        assert!(mat.is_empty());
    }

    #[test]
    fn test_mat_with_size() {
        let size = Size::new(640, 480);
        let mat = Mat::new_size(size, MatType::CV_8U).unwrap();
        assert_eq!(mat.rows(), 480);
        assert_eq!(mat.cols(), 640);
        assert_eq!(mat.mat_type(), MatType::CV_8U);
    }

    #[test]
    fn test_mat_type_properties() {
        assert_eq!(MatType::CV_8U.elem_size(), 1);
        assert_eq!(MatType::CV_32F.elem_size(), 4);
        assert!(MatType::CV_32F.is_float());
        assert!(!MatType::CV_8U.is_float());
        assert!(MatType::CV_8S.is_signed());
        assert!(!MatType::CV_8U.is_signed());
    }
}