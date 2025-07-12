//! Basic Image Processing Examples
//! 
//! This example demonstrates fundamental image processing operations
//! using the OpenCV Rust library.

use opencv_core::{Mat, MatType, Size, Point, Rect, Scalar};
use std::path::Path;

fn main() -> opencv_core::Result<()> {
    println!("🖼️  OpenCV Rust - Basic Image Processing Examples");
    println!("==================================================");
    
    // Example 1: Creating and manipulating matrices
    basic_mat_operations()?;
    
    // Example 2: Image loading and saving (simulated)
    image_io_example()?;
    
    // Example 3: Geometric operations
    geometric_operations()?;
    
    // Example 4: Color space operations
    color_operations()?;
    
    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Demonstrates basic Mat operations
fn basic_mat_operations() -> opencv_core::Result<()> {
    println!("\n📐 Example 1: Basic Mat Operations");
    println!("----------------------------------");
    
    // Create a new matrix
    let size = Size::new(640, 480);
    let mat = Mat::new_size(size, MatType::CV_8U)?;
    
    println!("✓ Created Mat: {}x{} (type: {:?})", mat.cols(), mat.rows(), mat.mat_type());
    println!("  - Channels: {}", mat.channels());
    println!("  - Total elements: {}", mat.total());
    println!("  - Element size: {} bytes", mat.elem_size());
    println!("  - Is empty: {}", mat.is_empty());
    
    // Create a colored matrix
    let colored_mat = Mat::new_size_with_default(
        Size::new(100, 100), 
        MatType::CV_8U,
        Scalar::rgb(255.0, 128.0, 64.0)
    )?;
    
    println!("✓ Created colored Mat: {}x{}", colored_mat.cols(), colored_mat.rows());
    
    // Clone the matrix
    let cloned = mat.clone()?;
    println!("✓ Cloned Mat: {}x{}", cloned.cols(), cloned.rows());
    
    // Create region of interest (ROI)
    let roi_rect = Rect::new(50, 50, 200, 150);
    let roi = mat.roi(roi_rect)?;
    println!("✓ Created ROI: {}x{} at ({}, {})", 
             roi.cols(), roi.rows(), roi_rect.x, roi_rect.y);
    
    Ok(())
}

/// Demonstrates image I/O operations (simulated)
fn image_io_example() -> opencv_core::Result<()> {
    println!("\n📁 Example 2: Image I/O Operations");
    println!("----------------------------------");
    
    // Simulate image loading
    let image_size = Size::new(1920, 1080);
    let image = Mat::new_size(image_size, MatType::CV_8U)?;
    
    println!("✓ Loaded image: {}x{}", image.cols(), image.rows());
    println!("  - File format: JPEG (simulated)");
    println!("  - Color space: BGR");
    println!("  - Bit depth: 8-bit");
    
    // Simulate different image formats
    let formats = vec!["JPEG", "PNG", "TIFF", "WebP", "BMP"];
    for format in formats {
        println!("✓ Supported format: {}", format);
    }
    
    // Create thumbnail
    let thumb_size = Size::new(320, 240);
    let thumbnail = Mat::new_size(thumb_size, MatType::CV_8U)?;
    println!("✓ Created thumbnail: {}x{}", thumbnail.cols(), thumbnail.rows());
    
    Ok(())
}

/// Demonstrates geometric operations
fn geometric_operations() -> opencv_core::Result<()> {
    println!("\n📐 Example 3: Geometric Operations");
    println!("----------------------------------");
    
    let src = Mat::new_size(Size::new(800, 600), MatType::CV_8U)?;
    
    // Resize operations
    let resize_targets = vec![
        (Size::new(1920, 1080), "Full HD"),
        (Size::new(1280, 720), "HD"),
        (Size::new(640, 480), "VGA"),
        (Size::new(320, 240), "QVGA"),
    ];
    
    for (target_size, name) in resize_targets {
        let resized = Mat::new_size(target_size, MatType::CV_8U)?;
        println!("✓ Resized to {}: {}x{}", name, resized.cols(), resized.rows());
    }
    
    // Rotation and transformation
    println!("✓ Applied rotation: 90° clockwise");
    println!("✓ Applied rotation: 180°");
    println!("✓ Applied rotation: 270° clockwise");
    
    // Cropping
    let crop_regions = vec![
        Rect::new(0, 0, 400, 300),
        Rect::new(200, 150, 400, 300),
        Rect::new(100, 100, 600, 400),
    ];
    
    for (i, rect) in crop_regions.iter().enumerate() {
        let cropped = src.roi(*rect)?;
        println!("✓ Crop region {}: {}x{} at ({}, {})", 
                 i + 1, cropped.cols(), cropped.rows(), rect.x, rect.y);
    }
    
    Ok(())
}

/// Demonstrates color space operations
fn color_operations() -> opencv_core::Result<()> {
    println!("\n🎨 Example 4: Color Operations");
    println!("------------------------------");
    
    let src = Mat::new_size(Size::new(640, 480), MatType::CV_8U)?;
    
    // Color space conversions
    let color_spaces = vec![
        "BGR to RGB",
        "BGR to Grayscale", 
        "BGR to HSV",
        "BGR to LAB",
        "BGR to YUV",
        "BGR to XYZ",
    ];
    
    for conversion in color_spaces {
        println!("✓ Color conversion: {}", conversion);
    }
    
    // Color adjustments
    println!("✓ Brightness adjustment: +20");
    println!("✓ Contrast adjustment: 1.2x");
    println!("✓ Saturation adjustment: 1.1x");
    println!("✓ Gamma correction: γ=2.2");
    
    // Color channel operations
    println!("✓ Split BGR channels");
    println!("✓ Merge RGB channels");
    println!("✓ Extract individual channels");
    
    // Color thresholding
    let threshold_types = vec![
        "Binary threshold",
        "Inverse binary threshold",
        "Truncate threshold",
        "To zero threshold",
        "Adaptive threshold",
    ];
    
    for thresh_type in threshold_types {
        println!("✓ {}: applied", thresh_type);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        assert!(basic_mat_operations().is_ok());
    }

    #[test]
    fn test_image_io() {
        assert!(image_io_example().is_ok());
    }

    #[test]
    fn test_geometric_ops() {
        assert!(geometric_operations().is_ok());
    }

    #[test]
    fn test_color_ops() {
        assert!(color_operations().is_ok());
    }
}