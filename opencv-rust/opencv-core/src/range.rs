//! Range structure for OpenCV

use std::fmt;

/// Range structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub start: i32,
    pub end: i32,
}

impl Range {
    /// Create a new range
    pub fn new(start: i32, end: i32) -> Self {
        Self { start, end }
    }

    /// Create an all-inclusive range
    pub fn all() -> Self {
        Self { start: i32::MIN, end: i32::MAX }
    }

    /// Get the size of the range
    pub fn size(&self) -> i32 {
        self.end - self.start
    }

    /// Check if the range is empty
    pub fn empty(&self) -> bool {
        self.start >= self.end
    }
}

impl Default for Range {
    fn default() -> Self {
        Self::all()
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}