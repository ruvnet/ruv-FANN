//! Common traits for OpenCV types

use crate::{Result, Size, MatType};

/// Trait for objects that have dimensions
pub trait Dimensioned {
    /// Get width
    fn width(&self) -> i32;
    /// Get height
    fn height(&self) -> i32;
    /// Get size
    fn size(&self) -> Size;
}

/// Trait for image-like objects
pub trait ImageTrait: Dimensioned {
    /// Get image type
    fn image_type(&self) -> MatType;
    /// Get number of channels
    fn channels(&self) -> i32;
    /// Check if empty
    fn is_empty(&self) -> bool;
}

/// Trait for objects that can be converted to/from arrays
pub trait ArrayConvertible<T, const N: usize> {
    /// Convert to array
    fn to_array(&self) -> [T; N];
    /// Create from array
    fn from_array(arr: [T; N]) -> Self;
}

/// Trait for objects that support arithmetic operations
pub trait Arithmetic: Sized {
    /// Add two objects
    fn add(&self, other: &Self) -> Result<Self>;
    /// Subtract two objects
    fn sub(&self, other: &Self) -> Result<Self>;
    /// Multiply by scalar
    fn mul_scalar(&self, scalar: f64) -> Result<Self>;
    /// Divide by scalar
    fn div_scalar(&self, scalar: f64) -> Result<Self>;
}