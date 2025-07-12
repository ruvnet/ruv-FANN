//! Rectangle structures for OpenCV

use crate::{Point, Size};
use std::fmt;
use serde::{Serialize, Deserialize};

/// Rectangle with integer coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    /// Create a new rectangle
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    /// Create rectangle from top-left and bottom-right points
    pub fn from_points(tl: Point, br: Point) -> Self {
        Self::new(
            tl.x,
            tl.y,
            br.x - tl.x,
            br.y - tl.y,
        )
    }

    /// Create rectangle from center point and size
    pub fn from_center_size(center: Point, size: Size) -> Self {
        let half_width = size.width / 2;
        let half_height = size.height / 2;
        Self::new(
            center.x - half_width,
            center.y - half_height,
            size.width,
            size.height,
        )
    }

    /// Get top-left corner
    pub fn tl(&self) -> Point {
        Point::new(self.x, self.y)
    }

    /// Get bottom-right corner
    pub fn br(&self) -> Point {
        Point::new(self.x + self.width, self.y + self.height)
    }

    /// Get center point
    pub fn center(&self) -> Point {
        Point::new(
            self.x + self.width / 2,
            self.y + self.height / 2,
        )
    }

    /// Get size
    pub fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    /// Calculate area
    pub fn area(&self) -> i32 {
        self.width * self.height
    }

    /// Check if rectangle is empty (width or height <= 0)
    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// Check if point is inside rectangle
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x &&
        point.y >= self.y &&
        point.x < self.x + self.width &&
        point.y < self.y + self.height
    }

    /// Check if another rectangle is completely inside this one
    pub fn contains_rect(&self, other: &Rect) -> bool {
        self.x <= other.x &&
        self.y <= other.y &&
        self.x + self.width >= other.x + other.width &&
        self.y + self.height >= other.y + other.height
    }

    /// Get intersection with another rectangle
    pub fn intersect(&self, other: &Rect) -> Option<Rect> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x1 < x2 && y1 < y2 {
            Some(Rect::new(x1, y1, x2 - x1, y2 - y1))
        } else {
            None
        }
    }

    /// Get union with another rectangle
    pub fn union(&self, other: &Rect) -> Rect {
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = (self.x + self.width).max(other.x + other.width);
        let y2 = (self.y + self.height).max(other.y + other.height);

        Rect::new(x1, y1, x2 - x1, y2 - y1)
    }

    /// Inflate rectangle by given amounts
    pub fn inflate(&self, dx: i32, dy: i32) -> Rect {
        Rect::new(
            self.x - dx,
            self.y - dy,
            self.width + 2 * dx,
            self.height + 2 * dy,
        )
    }

    /// Translate rectangle by given offset
    pub fn translate(&self, dx: i32, dy: i32) -> Rect {
        Rect::new(
            self.x + dx,
            self.y + dy,
            self.width,
            self.height,
        )
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self::new(0, 0, 0, 0)
    }
}

impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}x{}]", self.x, self.y, self.width, self.height)
    }
}

impl From<(i32, i32, i32, i32)> for Rect {
    fn from((x, y, width, height): (i32, i32, i32, i32)) -> Self {
        Self::new(x, y, width, height)
    }
}

impl From<Rect> for (i32, i32, i32, i32) {
    fn from(rect: Rect) -> Self {
        (rect.x, rect.y, rect.width, rect.height)
    }
}

/// Rectangle with single-precision floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rect2f {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect2f {
    /// Create a new rectangle
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Calculate area
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Check if rectangle is empty
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// Check if point is inside rectangle
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x &&
        y >= self.y &&
        x < self.x + self.width &&
        y < self.y + self.height
    }

    /// Get intersection with another rectangle
    pub fn intersect(&self, other: &Rect2f) -> Option<Rect2f> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x1 < x2 && y1 < y2 {
            Some(Rect2f::new(x1, y1, x2 - x1, y2 - y1))
        } else {
            None
        }
    }
}

impl Default for Rect2f {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl fmt::Display for Rect2f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.2}, {:.2}, {:.2}x{:.2}]", self.x, self.y, self.width, self.height)
    }
}

impl From<Rect> for Rect2f {
    fn from(rect: Rect) -> Self {
        Self::new(
            rect.x as f32,
            rect.y as f32,
            rect.width as f32,
            rect.height as f32,
        )
    }
}

/// Rectangle with double-precision floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rect2d {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect2d {
    /// Create a new rectangle
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    /// Calculate area
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Check if rectangle is empty
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// Check if point is inside rectangle
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x &&
        y >= self.y &&
        x < self.x + self.width &&
        y < self.y + self.height
    }
}

impl Default for Rect2d {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl From<Rect2f> for Rect2d {
    fn from(rect: Rect2f) -> Self {
        Self::new(
            rect.x as f64,
            rect.y as f64,
            rect.width as f64,
            rect.height as f64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_creation() {
        let rect = Rect::new(10, 20, 100, 200);
        assert_eq!(rect.x, 10);
        assert_eq!(rect.y, 20);
        assert_eq!(rect.width, 100);
        assert_eq!(rect.height, 200);
        assert_eq!(rect.area(), 20000);
    }

    #[test]
    fn test_rect_points() {
        let rect = Rect::new(10, 20, 100, 200);
        assert_eq!(rect.tl(), Point::new(10, 20));
        assert_eq!(rect.br(), Point::new(110, 220));
        assert_eq!(rect.center(), Point::new(60, 120));
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(10, 20, 100, 200);
        assert!(rect.contains(Point::new(50, 50)));
        assert!(!rect.contains(Point::new(5, 5)));
        assert!(!rect.contains(Point::new(150, 150)));
    }

    #[test]
    fn test_rect_intersection() {
        let rect1 = Rect::new(0, 0, 100, 100);
        let rect2 = Rect::new(50, 50, 100, 100);
        
        let intersection = rect1.intersect(&rect2).unwrap();
        assert_eq!(intersection, Rect::new(50, 50, 50, 50));
    }

    #[test]
    fn test_rect_union() {
        let rect1 = Rect::new(0, 0, 100, 100);
        let rect2 = Rect::new(50, 50, 100, 100);
        
        let union = rect1.union(&rect2);
        assert_eq!(union, Rect::new(0, 0, 150, 150));
    }

    #[test]
    fn test_rect_inflate() {
        let rect = Rect::new(10, 10, 100, 100);
        let inflated = rect.inflate(5, 5);
        assert_eq!(inflated, Rect::new(5, 5, 110, 110));
    }

    #[test]
    fn test_rect_translate() {
        let rect = Rect::new(10, 10, 100, 100);
        let translated = rect.translate(5, 5);
        assert_eq!(translated, Rect::new(15, 15, 100, 100));
    }
}