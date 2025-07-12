//! Scalar and vector types for OpenCV

use std::fmt;
use serde::{Serialize, Deserialize};

/// 4-element scalar value used throughout OpenCV
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Scalar {
    pub val: [f64; 4],
}

impl Scalar {
    /// Create a new scalar with all elements set to the same value
    pub fn all(v: f64) -> Self {
        Self { val: [v, v, v, v] }
    }

    /// Create a scalar from individual values
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Self { val: [v0, v1, v2, v3] }
    }

    /// Create a scalar for grayscale (single channel)
    pub fn gray(v: f64) -> Self {
        Self::new(v, 0.0, 0.0, 0.0)
    }

    /// Create a scalar for RGB color
    pub fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self::new(b, g, r, 0.0) // OpenCV uses BGR order
    }

    /// Create a scalar for RGBA color
    pub fn rgba(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self::new(b, g, r, a) // OpenCV uses BGRA order
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> f64 {
        if index < 4 {
            self.val[index]
        } else {
            0.0
        }
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: f64) {
        if index < 4 {
            self.val[index] = value;
        }
    }

    /// Check if all elements are zero
    pub fn is_zero(&self) -> bool {
        self.val.iter().all(|&x| x == 0.0)
    }

    /// Get the magnitude (L2 norm) of the scalar
    pub fn magnitude(&self) -> f64 {
        (self.val[0] * self.val[0] + 
         self.val[1] * self.val[1] + 
         self.val[2] * self.val[2] + 
         self.val[3] * self.val[3]).sqrt()
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self { val: [0.0; 4] }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.2}, {:.2}, {:.2}, {:.2}]", 
               self.val[0], self.val[1], self.val[2], self.val[3])
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Self::all(v)
    }
}

impl From<(f64, f64, f64)> for Scalar {
    fn from((v0, v1, v2): (f64, f64, f64)) -> Self {
        Self::new(v0, v1, v2, 0.0)
    }
}

impl From<(f64, f64, f64, f64)> for Scalar {
    fn from((v0, v1, v2, v3): (f64, f64, f64, f64)) -> Self {
        Self::new(v0, v1, v2, v3)
    }
}

impl From<[f64; 4]> for Scalar {
    fn from(val: [f64; 4]) -> Self {
        Self { val }
    }
}

impl From<Scalar> for [f64; 4] {
    fn from(scalar: Scalar) -> Self {
        scalar.val
    }
}


/// Generic N-dimensional vector
#[derive(Debug, Clone, PartialEq)]
pub struct VecN<T, const N: usize> {
    pub val: [T; N],
}

impl<T: Copy + Default, const N: usize> VecN<T, N> {
    /// Create a new vector with all elements set to default
    pub fn new() -> Self {
        Self { val: [T::default(); N] }
    }

    /// Create a vector from an array
    pub fn from_array(val: [T; N]) -> Self {
        Self { val }
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.val.get(index)
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) -> bool {
        if index < N {
            self.val[index] = value;
            true
        } else {
            false
        }
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        N
    }

    /// Check if vector is empty (should never be true for const generic)
    pub fn is_empty(&self) -> bool {
        N == 0
    }
}

impl<T: Copy + Default, const N: usize> Default for VecN<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Display, const N: usize> fmt::Display for VecN<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, val) in self.val.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

/// Common vector type aliases
pub type Vec2b = VecN<u8, 2>;
pub type Vec3b = VecN<u8, 3>;
pub type Vec4b = VecN<u8, 4>;
pub type Vec2s = VecN<i16, 2>;
pub type Vec3s = VecN<i16, 3>;
pub type Vec4s = VecN<i16, 4>;
pub type Vec2w = VecN<u16, 2>;
pub type Vec3w = VecN<u16, 3>;
pub type Vec4w = VecN<u16, 4>;
pub type Vec2i = VecN<i32, 2>;
pub type Vec3i = VecN<i32, 3>;
pub type Vec4i = VecN<i32, 4>;
pub type Vec2f = VecN<f32, 2>;
pub type Vec3f = VecN<f32, 3>;
pub type Vec4f = VecN<f32, 4>;
pub type Vec2d = VecN<f64, 2>;
pub type Vec3d = VecN<f64, 3>;
pub type Vec4d = VecN<f64, 4>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_creation() {
        let s1 = Scalar::all(10.0);
        assert_eq!(s1.val, [10.0, 10.0, 10.0, 10.0]);

        let s2 = Scalar::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(s2.val, [1.0, 2.0, 3.0, 4.0]);

        let s3 = Scalar::gray(128.0);
        assert_eq!(s3.val, [128.0, 0.0, 0.0, 0.0]);

        let s4 = Scalar::rgb(255.0, 128.0, 64.0);
        assert_eq!(s4.val, [64.0, 128.0, 255.0, 0.0]); // BGR order
    }

    #[test]
    fn test_scalar_operations() {
        let mut s = Scalar::new(1.0, 2.0, 3.0, 4.0);
        
        assert_eq!(s.get(0), 1.0);
        assert_eq!(s.get(2), 3.0);
        assert_eq!(s.get(5), 0.0); // Out of bounds

        s.set(1, 10.0);
        assert_eq!(s.get(1), 10.0);

        let magnitude = s.magnitude();
        assert!((magnitude - (1.0 + 100.0 + 9.0 + 16.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_conversions() {
        let s1: Scalar = 5.0.into();
        assert_eq!(s1.val, [5.0, 5.0, 5.0, 5.0]);

        let s2: Scalar = (1.0, 2.0, 3.0).into();
        assert_eq!(s2.val, [1.0, 2.0, 3.0, 0.0]);

        let s3: Scalar = (1.0, 2.0, 3.0, 4.0).into();
        assert_eq!(s3.val, [1.0, 2.0, 3.0, 4.0]);

        let arr: [f64; 4] = s3.into();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vecn() {
        let mut v = Vec3f::new();
        assert_eq!(v.len(), 3);
        
        v.set(0, 1.0);
        v.set(1, 2.0);
        v.set(2, 3.0);
        
        assert_eq!(v.get(0), Some(&1.0));
        assert_eq!(v.get(1), Some(&2.0));
        assert_eq!(v.get(2), Some(&3.0));
        assert_eq!(v.get(3), None);

        let v2 = Vec3f::from_array([4.0, 5.0, 6.0]);
        assert_eq!(v2.val, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scalar_zero() {
        let zero = Scalar::default();
        assert!(zero.is_zero());

        let non_zero = Scalar::new(1.0, 0.0, 0.0, 0.0);
        assert!(!non_zero.is_zero());
    }
}