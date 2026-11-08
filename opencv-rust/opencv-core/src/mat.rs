//! Mat - The fundamental OpenCV data structure for images and matrices

use crate::{Error, Result, Size, Point, Rect, Scalar};
use std::fmt;


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

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {} impl Sealed for i8 {} impl Sealed for u16 {}
    impl Sealed for i16 {} impl Sealed for i32 {} impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Primitive matrix element with no padding or uninitialized bytes.
///
/// The sealed implementations ensure safe slice imports cannot read arbitrary
/// object representations. Use `u16` for the raw bits of `CV_16F` elements.
pub trait MatElement: sealed::Sealed + Copy {
    /// Whether the element's representation matches the requested matrix type.
    fn matches(mat_type: MatType) -> bool;
    /// Append the native-endian representation of an element.
    fn append_bytes(self, output: &mut Vec<u8>);
}

macro_rules! mat_element {
    ($ty:ty, $($variant:pat_param)|+) => {
        impl MatElement for $ty {
            fn matches(mat_type: MatType) -> bool { matches!(mat_type, $($variant)|+) }
            fn append_bytes(self, output: &mut Vec<u8>) { output.extend_from_slice(&self.to_ne_bytes()); }
        }
    };
}
mat_element!(u8, MatType::CV_8U);
mat_element!(i8, MatType::CV_8S);
mat_element!(u16, MatType::CV_16U | MatType::CV_16F);
mat_element!(i16, MatType::CV_16S);
mat_element!(i32, MatType::CV_32S);
mat_element!(f32, MatType::CV_32F);
mat_element!(f64, MatType::CV_64F);

/// Maximum allocation accepted by the portable matrix API (1 GiB).
pub const MAX_MAT_BYTES: usize = 1024 * 1024 * 1024;

fn byte_len(rows: i32, cols: i32, elem_size: usize) -> Result<usize> {
    if rows < 0 || cols < 0 {
        return Err(Error::InvalidArgument("Matrix dimensions must be nonnegative".into()));
    }
    (rows as usize).checked_mul(cols as usize)
        .and_then(|n| n.checked_mul(elem_size))
        .filter(|&n| n <= MAX_MAT_BYTES)
        .ok_or_else(|| Error::InvalidArgument("Matrix exceeds 1 GiB allocation limit".into()))
}

fn buffer(len: usize) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(|e| Error::Memory(e.to_string()))?;
    Ok(data)
}

fn encode(value: f64, ty: MatType, output: &mut Vec<u8>) -> Result<()> {
    match ty {
        MatType::CV_8U => (value.round() as u8).append_bytes(output),
        MatType::CV_8S => (value.round() as i8).append_bytes(output),
        MatType::CV_16U => (value.round() as u16).append_bytes(output),
        MatType::CV_16S => (value.round() as i16).append_bytes(output),
        MatType::CV_32S => (value.round() as i32).append_bytes(output),
        MatType::CV_32F => (value as f32).append_bytes(output),
        MatType::CV_64F => value.append_bytes(output),
        MatType::CV_16F if value == 0.0 => 0u16.append_bytes(output),
        MatType::CV_16F => return Err(Error::Unsupported("Half-float scalar conversion is not implemented".into())),
    }
    Ok(())
}

fn decode(bytes: &[u8], ty: MatType) -> Result<f64> {
    Ok(match ty {
        MatType::CV_8U => bytes[0] as f64,
        MatType::CV_8S => bytes[0] as i8 as f64,
        MatType::CV_16U => u16::from_ne_bytes(bytes.try_into().unwrap()) as f64,
        MatType::CV_16S => i16::from_ne_bytes(bytes.try_into().unwrap()) as f64,
        MatType::CV_32S => i32::from_ne_bytes(bytes.try_into().unwrap()) as f64,
        MatType::CV_32F => f32::from_ne_bytes(bytes.try_into().unwrap()) as f64,
        MatType::CV_64F => f64::from_ne_bytes(bytes.try_into().unwrap()),
        MatType::CV_16F => return Err(Error::Unsupported("Half-float scalar conversion is not implemented".into())),
    })
}

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

    /// Create a single-channel matrix initialized to the scalar's first value.
    pub fn new_size_with_default(size: Size, mat_type: MatType, default_value: Scalar) -> Result<Self> {
        let len = byte_len(size.height, size.width, mat_type.elem_size())?;
        let mut data = buffer(len)?;
        let mut element = Vec::with_capacity(mat_type.elem_size());
        encode(default_value.val[0], mat_type, &mut element)?;
        data.resize(len, 0);
        if default_value.val[0] != 0.0 {
            for bytes in data.chunks_exact_mut(mat_type.elem_size()) {
                bytes.copy_from_slice(&element);
            }
        }
        Ok(Mat { data, rows: size.height, cols: size.width, mat_type, channels: 1 })
    }

    /// Copy matching primitive elements into a single-channel matrix.
    ///
    /// Rejects negative dimensions, mismatched element representations and
    /// allocations over `MAX_MAT_BYTES`. Arbitrary user-defined structs are not
    /// accepted because their padding may contain uninitialized bytes.
    pub fn from_slice<T: MatElement>(data: &[T], rows: i32, cols: i32, mat_type: MatType) -> Result<Self> {
        let len = byte_len(rows, cols, mat_type.elem_size())?;
        if !T::matches(mat_type) {
            return Err(Error::TypeConversion("Slice element type does not match MatType".into()));
        }
        if data.len() != len / mat_type.elem_size() {
            return Err(Error::InvalidArgument("Slice length does not match matrix dimensions".into()));
        }
        let mut bytes = buffer(len)?;
        for &element in data { element.append_bytes(&mut bytes); }
        Ok(Mat { data: bytes, rows, cols, mat_type, channels: 1 })
    }

    /// Borrow the initialized, contiguous matrix bytes in native endian order.
    pub fn data_bytes(&self) -> &[u8] { &self.data }

    /// Mutably borrow the initialized, contiguous matrix bytes.
    pub fn data_bytes_mut(&mut self) -> &mut [u8] { &mut self.data }

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

    /// Copy a rectangular region into an independent contiguous matrix.
    pub fn roi(&self, rect: Rect) -> Result<Mat> {
        if rect.x < 0 || rect.y < 0 || rect.width < 0 || rect.height < 0 ||
            rect.x.checked_add(rect.width).map_or(true, |x| x > self.cols) ||
            rect.y.checked_add(rect.height).map_or(true, |y| y > self.rows) {
            return Err(Error::InvalidArgument("ROI out of bounds".into()));
        }
        let elem_size = self.elem_size();
        let len = byte_len(rect.height, rect.width, elem_size)?;
        let mut data = buffer(len)?;
        let row_bytes = rect.width as usize * elem_size;
        for row in rect.y..rect.y + rect.height {
            let start = (row as usize * self.cols as usize + rect.x as usize) * elem_size;
            data.extend_from_slice(&self.data[start..start + row_bytes]);
        }
        Ok(Mat { data, rows: rect.height, cols: rect.width, mat_type: self.mat_type, channels: self.channels })
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

    /// Convert each element using `dst = alpha * src + beta`.
    ///
    /// Integer destinations round and saturate. Half-float conversion returns an
    /// explicit unsupported error rather than changing metadata without data.
    pub fn convert_to(&self, dst: &mut Mat, rtype: MatType, alpha: f64, beta: f64) -> Result<()> {
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(Error::InvalidArgument("Conversion coefficients must be finite".into()));
        }
        if self.mat_type == MatType::CV_16F || rtype == MatType::CV_16F {
            return Err(Error::Unsupported("Half-float scalar conversion is not implemented".into()));
        }
        let len = byte_len(self.rows, self.cols, rtype.elem_size() * self.channels as usize)?;
        let mut data = buffer(len)?;
        for bytes in self.data.chunks_exact(self.mat_type.elem_size()) {
            encode(decode(bytes, self.mat_type)? * alpha + beta, rtype, &mut data)?;
        }
        *dst = Mat { data, rows: self.rows, cols: self.cols, mat_type: rtype, channels: self.channels };
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
        self.rows as usize * self.cols as usize
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