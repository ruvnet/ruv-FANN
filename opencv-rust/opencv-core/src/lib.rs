//! OpenCV Core Module in Rust
//! 
//! This module provides the fundamental data structures and operations
//! that form the foundation of the OpenCV library.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod error;
pub mod mat;
pub mod point;
pub mod rect;
pub mod size;
pub mod scalar;
pub mod range;
pub mod types;
pub mod memory;
pub mod traits;

pub use error::{Error, Result};
pub use mat::{Mat, MatTrait, MatType};
pub use point::{Point, Point2f, Point2d, Point3f, Point3d};
pub use rect::{Rect, Rect2f, Rect2d};
pub use size::{Size, Size2f, Size2d};
pub use scalar::{Scalar, VecN};
pub use range::Range;
pub use types::*;

/// OpenCV Core module version
pub const OPENCV_VERSION: &str = "4.8.0";

/// Initialize OpenCV core module
pub fn init() -> Result<()> {
    log::info!("Initializing OpenCV Core module v{}", OPENCV_VERSION);
    memory::init_allocators()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }

    #[test]
    fn test_version() {
        assert_eq!(OPENCV_VERSION, "4.8.0");
    }
}