use opencv_core::{Mat, MatType, Rect, Scalar, Size};

#[test]
fn reject_negative_huge_and_mismatched_matrix_inputs() {
    assert!(Mat::new_size(Size::new(-1, 2), MatType::CV_8U).is_err());
    assert!(Mat::new_size(Size::new(i32::MAX, i32::MAX), MatType::CV_64F).is_err());
    assert!(Mat::from_slice(&[1u8, 2], 1, 2, MatType::CV_64F).is_err());
    assert!(Mat::from_slice(&[1f64], 1, 1, MatType::CV_8U).is_err());
    assert!(Mat::from_slice(&[1i16], -1, -1, MatType::CV_16S).is_err());
}

#[test]
fn scalar_fill_and_roi_preserve_actual_pixels() {
    let white =
        Mat::new_size_with_default(Size::new(3, 2), MatType::CV_8U, Scalar::all(255.0)).unwrap();
    assert_eq!(white.data_bytes(), &[255; 6]);
    let source = Mat::from_slice(
        &[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        3,
        4,
        MatType::CV_8U,
    )
    .unwrap();
    let mut crop = source.roi(Rect::new(1, 1, 2, 2)).unwrap();
    assert_eq!(crop.data_bytes(), &[5, 6, 9, 10]);
    crop.data_bytes_mut()[0] = 200;
    assert_eq!(source.data_bytes()[5], 5);
    assert!(source.roi(Rect::new(i32::MAX, 0, i32::MAX, 1)).is_err());
    assert!(source.roi(Rect::new(0, 0, -1, 1)).is_err());
}

#[test]
fn conversion_allocates_destination_width_and_applies_scale() {
    let source = Mat::from_slice(&[0u8, 128, 255], 1, 3, MatType::CV_8U).unwrap();
    let mut converted = Mat::new().unwrap();
    source
        .convert_to(&mut converted, MatType::CV_32F, 1.0 / 255.0, 0.0)
        .unwrap();
    assert_eq!(converted.data_bytes().len(), 12);
    let values: Vec<f32> = converted
        .data_bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(values[0], 0.0);
    assert!((values[1] - 128.0 / 255.0).abs() < 1e-6);
    assert_eq!(values[2], 1.0);
    let source = Mat::from_slice(&[-100f64, 127.6, 1000.0], 1, 3, MatType::CV_64F).unwrap();
    source
        .convert_to(&mut converted, MatType::CV_8U, 1.0, 0.0)
        .unwrap();
    assert_eq!(converted.data_bytes(), &[0, 128, 255]);
}
