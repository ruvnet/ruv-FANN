//! Point structures for OpenCV

use std::fmt;
use serde::{Serialize, Deserialize};

/// 2D point with integer coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    /// Create a new point
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Origin point (0, 0)
    pub fn origin() -> Self {
        Self::new(0, 0)
    }

    /// Calculate distance to another point
    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        (dx * dx + dy * dy).sqrt()
    }

    /// Dot product with another point
    pub fn dot(&self, other: &Point) -> i32 {
        self.x * other.x + self.y * other.y
    }
}

impl Default for Point {
    fn default() -> Self {
        Self::origin()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl From<(i32, i32)> for Point {
    fn from((x, y): (i32, i32)) -> Self {
        Self::new(x, y)
    }
}

impl From<Point> for (i32, i32) {
    fn from(point: Point) -> Self {
        (point.x, point.y)
    }
}

/// 2D point with single-precision floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point2f {
    pub x: f32,
    pub y: f32,
}

impl Point2f {
    /// Create a new point
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Origin point (0.0, 0.0)
    pub fn origin() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Calculate distance to another point
    pub fn distance_to(&self, other: &Point2f) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Dot product with another point
    pub fn dot(&self, other: &Point2f) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Normalize the point (treating it as a vector)
    pub fn normalize(&self) -> Point2f {
        let len = (self.x * self.x + self.y * self.y).sqrt();
        if len > 0.0 {
            Point2f::new(self.x / len, self.y / len)
        } else {
            *self
        }
    }
}

impl Default for Point2f {
    fn default() -> Self {
        Self::origin()
    }
}

impl fmt::Display for Point2f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

impl From<(f32, f32)> for Point2f {
    fn from((x, y): (f32, f32)) -> Self {
        Self::new(x, y)
    }
}

impl From<Point2f> for (f32, f32) {
    fn from(point: Point2f) -> Self {
        (point.x, point.y)
    }
}

impl From<Point> for Point2f {
    fn from(point: Point) -> Self {
        Self::new(point.x as f32, point.y as f32)
    }
}

/// 2D point with double-precision floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

impl Point2d {
    /// Create a new point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Origin point (0.0, 0.0)
    pub fn origin() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Calculate distance to another point
    pub fn distance_to(&self, other: &Point2d) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Dot product with another point
    pub fn dot(&self, other: &Point2d) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Normalize the point (treating it as a vector)
    pub fn normalize(&self) -> Point2d {
        let len = (self.x * self.x + self.y * self.y).sqrt();
        if len > 0.0 {
            Point2d::new(self.x / len, self.y / len)
        } else {
            *self
        }
    }
}

impl Default for Point2d {
    fn default() -> Self {
        Self::origin()
    }
}

impl fmt::Display for Point2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6})", self.x, self.y)
    }
}

impl From<(f64, f64)> for Point2d {
    fn from((x, y): (f64, f64)) -> Self {
        Self::new(x, y)
    }
}

impl From<Point2d> for (f64, f64) {
    fn from(point: Point2d) -> Self {
        (point.x, point.y)
    }
}

impl From<Point2f> for Point2d {
    fn from(point: Point2f) -> Self {
        Self::new(point.x as f64, point.y as f64)
    }
}

/// 3D point with single-precision floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3f {
    /// Create a new 3D point
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Origin point (0.0, 0.0, 0.0)
    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Calculate distance to another point
    pub fn distance_to(&self, other: &Point3f) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Dot product with another point
    pub fn dot(&self, other: &Point3f) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product with another point
    pub fn cross(&self, other: &Point3f) -> Point3f {
        Point3f::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Normalize the point (treating it as a vector)
    pub fn normalize(&self) -> Point3f {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            Point3f::new(self.x / len, self.y / len, self.z / len)
        } else {
            *self
        }
    }
}

impl Default for Point3f {
    fn default() -> Self {
        Self::origin()
    }
}

impl fmt::Display for Point3f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

impl From<(f32, f32, f32)> for Point3f {
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self::new(x, y, z)
    }
}

impl From<Point3f> for (f32, f32, f32) {
    fn from(point: Point3f) -> Self {
        (point.x, point.y, point.z)
    }
}

/// 3D point with double-precision floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3d {
    /// Create a new 3D point
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Origin point (0.0, 0.0, 0.0)
    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Calculate distance to another point
    pub fn distance_to(&self, other: &Point3d) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Dot product with another point
    pub fn dot(&self, other: &Point3d) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product with another point
    pub fn cross(&self, other: &Point3d) -> Point3d {
        Point3d::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Normalize the point (treating it as a vector)
    pub fn normalize(&self) -> Point3d {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            Point3d::new(self.x / len, self.y / len, self.z / len)
        } else {
            *self
        }
    }
}

impl Default for Point3d {
    fn default() -> Self {
        Self::origin()
    }
}

impl fmt::Display for Point3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
    }
}

impl From<(f64, f64, f64)> for Point3d {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Self::new(x, y, z)
    }
}

impl From<Point3d> for (f64, f64, f64) {
    fn from(point: Point3d) -> Self {
        (point.x, point.y, point.z)
    }
}

impl From<Point3f> for Point3d {
    fn from(point: Point3f) -> Self {
        Self::new(point.x as f64, point.y as f64, point.z as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::new(10, 20);
        assert_eq!(p.x, 10);
        assert_eq!(p.y, 20);
    }

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0, 0);
        let p2 = Point::new(3, 4);
        assert_eq!(p1.distance_to(&p2), 5.0);
    }

    #[test]
    fn test_point3d_cross_product() {
        let p1 = Point3d::new(1.0, 0.0, 0.0);
        let p2 = Point3d::new(0.0, 1.0, 0.0);
        let cross = p1.cross(&p2);
        assert_eq!(cross, Point3d::new(0.0, 0.0, 1.0));
    }
}