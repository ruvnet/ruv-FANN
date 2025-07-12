//! Size structures for OpenCV

use std::fmt;
use serde::{Serialize, Deserialize};

/// 2D size with integer dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

impl Size {
    /// Create a new size
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    /// Create a square size
    pub fn square(size: i32) -> Self {
        Self::new(size, size)
    }

    /// Calculate area
    pub fn area(&self) -> i32 {
        self.width * self.height
    }

    /// Check if size is empty (width or height <= 0)
    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// Get aspect ratio (width/height)
    pub fn aspect_ratio(&self) -> f64 {
        if self.height != 0 {
            self.width as f64 / self.height as f64
        } else {
            0.0
        }
    }
}

impl Default for Size {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl From<(i32, i32)> for Size {
    fn from((width, height): (i32, i32)) -> Self {
        Self::new(width, height)
    }
}

impl From<Size> for (i32, i32) {
    fn from(size: Size) -> Self {
        (size.width, size.height)
    }
}

/// 2D size with single-precision floating-point dimensions
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Size2f {
    pub width: f32,
    pub height: f32,
}

impl Size2f {
    /// Create a new size
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Create a square size
    pub fn square(size: f32) -> Self {
        Self::new(size, size)
    }

    /// Calculate area
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Check if size is empty (width or height <= 0)
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// Get aspect ratio (width/height)
    pub fn aspect_ratio(&self) -> f32 {
        if self.height != 0.0 {
            self.width / self.height
        } else {
            0.0
        }
    }
}

impl Default for Size2f {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl fmt::Display for Size2f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}x{:.2}", self.width, self.height)
    }
}

impl From<(f32, f32)> for Size2f {
    fn from((width, height): (f32, f32)) -> Self {
        Self::new(width, height)
    }
}

impl From<Size2f> for (f32, f32) {
    fn from(size: Size2f) -> Self {
        (size.width, size.height)
    }
}

impl From<Size> for Size2f {
    fn from(size: Size) -> Self {
        Self::new(size.width as f32, size.height as f32)
    }
}

/// 2D size with double-precision floating-point dimensions
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Size2d {
    pub width: f64,
    pub height: f64,
}

impl Size2d {
    /// Create a new size
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    /// Create a square size
    pub fn square(size: f64) -> Self {
        Self::new(size, size)
    }

    /// Calculate area
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Check if size is empty (width or height <= 0)
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// Get aspect ratio (width/height)
    pub fn aspect_ratio(&self) -> f64 {
        if self.height != 0.0 {
            self.width / self.height
        } else {
            0.0
        }
    }
}

impl Default for Size2d {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl fmt::Display for Size2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}x{:.6}", self.width, self.height)
    }
}

impl From<(f64, f64)> for Size2d {
    fn from((width, height): (f64, f64)) -> Self {
        Self::new(width, height)
    }
}

impl From<Size2d> for (f64, f64) {
    fn from(size: Size2d) -> Self {
        (size.width, size.height)
    }
}

impl From<Size2f> for Size2d {
    fn from(size: Size2f) -> Self {
        Self::new(size.width as f64, size.height as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_creation() {
        let size = Size::new(640, 480);
        assert_eq!(size.width, 640);
        assert_eq!(size.height, 480);
        assert_eq!(size.area(), 307200);
    }

    #[test]
    fn test_size_square() {
        let size = Size::square(100);
        assert_eq!(size.width, 100);
        assert_eq!(size.height, 100);
        assert_eq!(size.area(), 10000);
    }

    #[test]
    fn test_size_aspect_ratio() {
        let size = Size::new(16, 9);
        assert!((size.aspect_ratio() - 16.0/9.0).abs() < 1e-10);
    }

    #[test]
    fn test_size_empty() {
        let empty_size = Size::new(0, 0);
        assert!(empty_size.is_empty());
        
        let valid_size = Size::new(10, 10);
        assert!(!valid_size.is_empty());
    }

    #[test]
    fn test_size_conversions() {
        let size = Size::new(100, 200);
        let tuple: (i32, i32) = size.into();
        assert_eq!(tuple, (100, 200));
        
        let size2: Size = (300, 400).into();
        assert_eq!(size2.width, 300);
        assert_eq!(size2.height, 400);
    }
}