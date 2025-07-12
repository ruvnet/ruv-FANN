//! Comprehensive integration tests for OpenCV Rust implementation

use opencv_core::{Mat, MatType, Size, Point, Rect};
use opencv_imgproc;
use opencv_ml;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencv_core_functionality() {
        // Test Mat creation and basic operations
        let size = Size::new(640, 480);
        let mat = Mat::new_size(size, MatType::CV_8U).unwrap();
        
        assert_eq!(mat.rows(), 480);
        assert_eq!(mat.cols(), 640);
        assert_eq!(mat.mat_type(), MatType::CV_8U);
        assert!(!mat.is_empty());
        
        // Test Mat cloning
        let cloned = mat.clone().unwrap();
        assert_eq!(cloned.size(), mat.size());
        assert_eq!(cloned.mat_type(), mat.mat_type());
    }

    #[test]
    fn test_point_operations() {
        let p1 = Point::new(10, 20);
        let p2 = Point::new(30, 40);
        
        assert_eq!(p1.distance_to(&p2), 28.284271247461902);
        assert_eq!(p1.dot(&p2), 1100);
        
        let p3f = opencv_core::Point3f::new(1.0, 0.0, 0.0);
        let p4f = opencv_core::Point3f::new(0.0, 1.0, 0.0);
        let cross = p3f.cross(&p4f);
        assert_eq!(cross.z, 1.0);
    }

    #[test]
    fn test_roi_operations() {
        let size = Size::new(640, 480);
        let mat = Mat::new_size(size, MatType::CV_8U).unwrap();
        
        let roi_rect = Rect::new(100, 100, 200, 200);
        let roi = mat.roi(roi_rect).unwrap();
        
        assert_eq!(roi.rows(), 200);
        assert_eq!(roi.cols(), 200);
    }

    #[test] 
    fn test_mat_type_conversions() {
        let size = Size::new(100, 100);
        let src = Mat::new_size(size, MatType::CV_8U).unwrap();
        let mut dst = Mat::new().unwrap();
        
        // Test type conversion
        src.convert_to(&mut dst, MatType::CV_32F, 1.0/255.0, 0.0).unwrap();
        
        assert_eq!(dst.mat_type(), MatType::CV_32F);
        assert_eq!(dst.size(), src.size());
    }

    #[test]
    fn test_memory_safety() {
        // Test that Mat properly manages memory
        {
            let _mat = Mat::new_size(Size::new(1000, 1000), MatType::CV_8U).unwrap();
            // Mat should be automatically dropped here
        }
        // No memory leaks should occur
    }

    #[test]
    fn test_opencv_api_compatibility() {
        // Test that our API matches OpenCV conventions
        let mat = Mat::new().unwrap();
        assert!(mat.is_empty());
        
        let size = Size::new(640, 480);
        let typed_mat = Mat::new_size(size, MatType::CV_8U).unwrap();
        assert_eq!(typed_mat.channels(), 1);
        assert_eq!(typed_mat.total(), 640 * 480);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;
        use std::sync::Arc;
        
        let mat = Arc::new(Mat::new_size(Size::new(100, 100), MatType::CV_8U).unwrap());
        
        let handles: Vec<_> = (0..4).map(|_| {
            let mat_clone = Arc::clone(&mat);
            thread::spawn(move || {
                assert_eq!(mat_clone.rows(), 100);
                assert_eq!(mat_clone.cols(), 100);
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
}