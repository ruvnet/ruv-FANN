use opencv_sys::*;
use std::{
    ffi::{CStr, CString},
    ptr,
};

struct Image(*mut cv_Mat);
impl Image {
    unsafe fn owned(pointer: *mut cv_Mat) -> Self {
        assert!(!pointer.is_null(), "{}", error());
        Self(pointer)
    }
    unsafe fn bytes(&self) -> Vec<u8> {
        let len = cv_Mat_total(self.0) * cv_Mat_element_size(self.0);
        let mut bytes = vec![0; len];
        assert_eq!(
            cv_Mat_copy_data(self.0, bytes.as_mut_ptr(), len),
            0,
            "{}",
            error()
        );
        bytes
    }
}
impl Drop for Image {
    fn drop(&mut self) {
        unsafe { cv_Mat_delete(self.0) }
    }
}
unsafe fn error() -> String {
    CStr::from_ptr(cv_last_error())
        .to_string_lossy()
        .into_owned()
}
unsafe fn image(bytes: &[u8], rows: i32, cols: i32, kind: i32) -> Image {
    Image::owned(cv_Mat_from_bytes(
        bytes.as_ptr(),
        bytes.len(),
        rows,
        cols,
        kind,
    ))
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
fn native_allocation_bounds_and_import_ownership() {
    unsafe {
        assert!(CStr::from_ptr(cv_version())
            .to_str()
            .unwrap()
            .starts_with("4."));
        let mut pixels = [1, 2, 3, 4, 5, 6];
        let native = image(&pixels, 2, 3, 0);
        pixels.fill(90);
        assert_eq!(native.bytes(), [1, 2, 3, 4, 5, 6]);
        assert_eq!(cv_Mat_rows(native.0), 2);
        assert_eq!(cv_Mat_cols(native.0), 3);
        assert_eq!(cv_Mat_channels(native.0), 1);
        assert_eq!(cv_Mat_type(native.0), 0);
        assert_eq!(cv_Mat_empty(native.0), 0);
        assert_eq!(cv_Mat_is_continuous(native.0), 1);
        let filled = Image::owned(cv_Mat_new_size_type_scalar(2, 3, 0, cv_Scalar::from(37.0)));
        assert_eq!(filled.bytes(), [37; 6]);
        assert!(cv_Mat_from_bytes(pixels.as_ptr(), 5, 2, 3, 0).is_null());
        assert!(error().contains("length"));
        assert!(cv_Mat_new_size_type_scalar(-1, 2, 0, cv_Scalar::default()).is_null());
        assert!(cv_Mat_new_size_type_scalar(i32::MAX, i32::MAX, 6, cv_Scalar::default()).is_null());
        assert!(cv_Mat_new_size_type_scalar(1, 1, 4096, cv_Scalar::default()).is_null());
        assert!(cv_Mat_row_data(native.0, 2).is_null());
        assert_eq!(cv_Mat_copy_data(native.0, pixels.as_mut_ptr(), 1), -1);
        cv_Mat_delete(ptr::null_mut());
    }
}

#[test]
fn native_roi_shares_storage_and_outlives_parent_while_clone_is_independent() {
    unsafe {
        let parent = image(&(0..12).collect::<Vec<_>>(), 3, 4, 0);
        let roi = Image::owned(cv_Mat_roi(parent.0, 1, 1, 2, 2));
        assert_eq!(cv_Mat_step(roi.0), 4);
        assert_eq!(cv_Mat_is_continuous(roi.0), 0);
        assert_eq!(roi.bytes(), [5, 6, 9, 10]);
        let clone = Image::owned(cv_Mat_clone(roi.0));
        *cv_Mat_row_data(roi.0, 0) = 77;
        assert_eq!(parent.bytes()[5], 77);
        assert_eq!(clone.bytes(), [5, 6, 9, 10]);
        assert!(cv_Mat_roi(parent.0, i32::MAX, 0, i32::MAX, 1).is_null());
        assert!(cv_Mat_roi(parent.0, 0, 0, -1, 1).is_null());
        drop(parent);
        assert_eq!(roi.bytes(), [77, 6, 9, 10]);
        let copied = Image::owned(cv_Mat_new());
        assert_eq!(cv_Mat_copyTo(roi.0, copied.0), 0);
        assert_eq!(copied.bytes(), roi.bytes());
    }
}

#[test]
fn convert_uses_native_numeric_width_and_scale() {
    unsafe {
        let source = image(&[0, 128, 255], 1, 3, 0);
        let output = Image::owned(cv_Mat_new());
        assert_eq!(cv_Mat_convertTo(source.0, output.0, 5, 1.0 / 255.0, 0.0), 0);
        let bytes = output.bytes();
        assert_eq!(bytes.len(), 12);
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        assert!((values[1] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(values[2], 1.0);
        assert_eq!(cv_Mat_convertTo(source.0, output.0, 5, f64::NAN, 0.0), -1);
        assert!(error().contains("finite"));
    }
}

#[test]
fn native_filters_resize_color_and_threshold_produce_pixels() {
    unsafe {
        let source = image(&[100; 25], 5, 5, 0);
        let output = Image::owned(cv_Mat_new());
        assert_eq!(cv_blur(source.0, output.0, 3, 3), 0);
        assert_eq!(output.bytes(), [100; 25]);
        assert_eq!(cv_GaussianBlur(source.0, output.0, 3, 3, 1.0, 1.0), 0);
        assert_eq!(output.bytes(), [100; 25]);
        assert_eq!(cv_Canny(source.0, output.0, 30.0, 70.0), 0);
        assert_eq!(output.bytes(), [0; 25]);
        assert_eq!(cv_blur(source.0, output.0, 0, 3), -1);
        assert_eq!(cv_GaussianBlur(source.0, output.0, 2, 2, 1.0, 1.0), -1);
        assert!(!error().is_empty());
        let source = image(&[1, 2, 3, 4], 2, 2, 0);
        assert_eq!(cv_resize(source.0, output.0, 4, 2, 0.0, 0.0, 0), 0);
        assert_eq!(output.bytes(), [1, 1, 2, 2, 3, 3, 4, 4]);
        assert_eq!(cv_threshold(source.0, output.0, 2.0, 255.0, 0), 2.0);
        assert_eq!(output.bytes(), [0, 0, 255, 255]);
        let bgr = image(&[0, 0, 255, 0, 255, 0], 1, 2, 16);
        assert_eq!(cv_cvtColor(bgr.0, output.0, 6), 0); // COLOR_BGR2GRAY
        assert_eq!(output.bytes(), [76, 150]);
    }
}

#[test]
fn native_png_roundtrip() {
    unsafe {
        let directory = tempfile::tempdir().unwrap();
        let filename = CString::new(directory.path().join("pixels.png").to_str().unwrap()).unwrap();
        let source = image(&[0, 64, 128, 255, 42, 17], 2, 3, 0);
        assert_eq!(cv_imwrite(filename.as_ptr(), source.0), 1, "{}", error());
        let decoded = Image::owned(cv_imread(filename.as_ptr(), 0));
        assert_eq!(decoded.bytes(), source.bytes());
        let capture = cv_VideoCapture_new_filename(filename.as_ptr());
        assert!(!capture.is_null(), "{}", error());
        assert_eq!(cv_VideoCapture_isOpened(capture), 1);
        let frame = Image::owned(cv_Mat_new());
        assert_eq!(cv_VideoCapture_read(capture, frame.0), 1, "{}", error());
        assert_eq!(cv_Mat_rows(frame.0), 2);
        assert_eq!(cv_Mat_cols(frame.0), 3);
        assert_eq!(cv_Mat_channels(frame.0), 3);
        assert_eq!(
            frame.bytes(),
            source
                .bytes()
                .iter()
                .flat_map(|v| [*v; 3])
                .collect::<Vec<_>>()
        );
        cv_VideoCapture_release(capture);
        cv_VideoCapture_delete(capture);
        let invalid = CString::new(directory.path().join("absent.png").to_str().unwrap()).unwrap();
        assert!(cv_imread(invalid.as_ptr(), 0).is_null());
        assert!(!error().is_empty());
    }
}

#[test]
fn sift_runs_real_detector_and_returns_owned_accessible_keypoints() {
    unsafe {
        let pixels: Vec<u8> = (0..128 * 128)
            .map(|i| {
                if ((i / 128) / 16 + (i % 128) / 16) % 2 == 0 {
                    0
                } else {
                    255
                }
            })
            .collect();
        let source = image(&pixels, 128, 128, 0);
        let descriptors = Image::owned(cv_Mat_new());
        let detector = cv_SIFT_create();
        let points = cv_KeyPoint_vector_new();
        assert!(!detector.is_null() && !points.is_null());
        assert_eq!(
            cv_SIFT_detectAndCompute(detector, source.0, ptr::null(), points, descriptors.0),
            0,
            "{}",
            error()
        );
        let count = cv_KeyPoint_vector_len(points);
        assert!(count > 0 && count < 10000, "unexpected keypoints: {count}");
        assert_eq!(cv_Mat_cols(descriptors.0), 128);
        assert_eq!(cv_Mat_rows(descriptors.0) as usize, count);
        let mut point = cv_KeyPoint::default();
        assert_eq!(cv_KeyPoint_vector_get(points, 0, &mut point), 0);
        assert!(point.size > 0.0);
        assert_eq!(cv_KeyPoint_vector_get(points, count, &mut point), -1);
        cv_KeyPoint_vector_delete(points);
        cv_SIFT_delete(detector);
    }
}

#[test]
fn svm_trains_and_predicts_real_labels() {
    unsafe {
        let samples = image(
            &f32_bytes(&[-3., -2., -2., -1., -1., -3., 1., 2., 2., 1., 3., 3.]),
            6,
            2,
            5,
        );
        let labels: Vec<u8> = [0i32, 0, 0, 1, 1, 1]
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect();
        let responses = image(&labels, 6, 1, 4);
        let svm = cv_ml_SVM_create();
        assert!(!svm.is_null());
        assert_eq!(cv_ml_SVM_set_type(svm, 100), 0); // C_SVC
        assert_eq!(cv_ml_SVM_set_kernel(svm, 0), 0); // LINEAR
        assert_eq!(cv_ml_SVM_set_c(svm, 1.0), 0);
        assert_eq!(
            cv_ml_SVM_train(svm, samples.0, 0, responses.0),
            1,
            "{}",
            error()
        ); // ROW_SAMPLE
        let query = image(&f32_bytes(&[-4., -3., 4., 3.]), 2, 2, 5);
        let results = Image::owned(cv_Mat_new());
        assert!(
            cv_ml_SVM_predict(svm, query.0, results.0).is_finite(),
            "{}",
            error()
        );
        let labels: Vec<f32> = results
            .bytes()
            .chunks_exact(4)
            .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(labels, [0., 1.]);
        cv_ml_SVM_delete(svm);
    }
}

#[test]
fn vectors_copy_nested_values_and_check_bounds() {
    unsafe {
        let points = cv_Point2f_vector_new();
        let nested = cv_Point2f_vector_vector_new();
        assert_eq!(
            cv_Point2f_vector_push(points, cv_Point2f { x: 1.0, y: 2.0 }),
            0
        );
        assert_eq!(cv_Point2f_vector_vector_push(nested, points), 0);
        assert_eq!(cv_Point2f_vector_clear(points), 0);
        cv_Point2f_vector_delete(points);
        let copy = cv_Point2f_vector_vector_get(nested, 0);
        assert!(!copy.is_null());
        let mut point = cv_Point2f::default();
        assert_eq!(cv_Point2f_vector_get(copy, 0, &mut point), 0);
        assert_eq!(point.x, 1.0);
        assert_eq!(cv_Point2f_vector_get(copy, 1, &mut point), -1);
        assert!(cv_Point2f_vector_vector_get(nested, 1).is_null());
        cv_Point2f_vector_vector_delete(nested);
        assert_eq!(cv_Point2f_vector_len(copy), 1);
        cv_Point2f_vector_delete(copy);
        let rectangles = cv_Rect_vector_new();
        assert_eq!(
            cv_Rect_vector_push(
                rectangles,
                cv_Rect {
                    x: 1,
                    y: 2,
                    width: 3,
                    height: 4
                }
            ),
            0
        );
        let mut rect = cv_Rect::default();
        assert_eq!(cv_Rect_vector_get(rectangles, 0, &mut rect), 0);
        assert_eq!(rect.width, 3);
        cv_Rect_vector_delete(rectangles);
    }
}

#[test]
fn solve_pnp_recovers_known_camera_translation() {
    unsafe {
        let objects = cv_Point3f_vector_new();
        let images = cv_Point2f_vector_new();
        for [x, y, z] in [
            [-1., -1., 0.],
            [1., -1., 0.],
            [-1., 1., 0.],
            [1., 1., 0.],
            [0., 0., 1.],
            [0.5, 0.5, 2.],
        ] {
            assert_eq!(cv_Point3f_vector_push(objects, cv_Point3f { x, y, z }), 0);
            assert_eq!(
                cv_Point2f_vector_push(
                    images,
                    cv_Point2f {
                        x: 100.0 * x / (z + 5.0),
                        y: 100.0 * y / (z + 5.0)
                    }
                ),
                0
            );
        }
        let camera = image(
            &f64_bytes(&[100., 0., 0., 0., 100., 0., 0., 0., 1.]),
            3,
            3,
            6,
        );
        let rotation = Image::owned(cv_Mat_new());
        let translation = Image::owned(cv_Mat_new());
        assert_eq!(
            cv_solvePnP(
                objects,
                images,
                camera.0,
                ptr::null(),
                rotation.0,
                translation.0
            ),
            1,
            "{}",
            error()
        );
        let xyz: Vec<f64> = translation
            .bytes()
            .chunks_exact(8)
            .map(|x| f64::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        assert!(
            xyz[0].abs() < 1e-4 && xyz[1].abs() < 1e-4 && (xyz[2] - 5.0).abs() < 1e-4,
            "{xyz:?}"
        );
        cv_Point3f_vector_delete(objects);
        cv_Point2f_vector_delete(images);
    }
}

#[test]
fn empty_models_and_null_handles_return_errors_without_unwinding() {
    unsafe {
        assert_eq!(cv_Mat_rows(ptr::null()), -1);
        assert!(error().contains("NULL"));
        let capture = cv_VideoCapture_new();
        assert!(!capture.is_null());
        assert_eq!(cv_VideoCapture_isOpened(capture), 0);
        cv_VideoCapture_release(capture);
        cv_VideoCapture_delete(capture);
        let classifier = cv_CascadeClassifier_new();
        let boxes = cv_Rect_vector_new();
        let source = image(&[0; 100], 10, 10, 0);
        assert_eq!(
            cv_CascadeClassifier_detectMultiScale(classifier, source.0, boxes),
            -1
        );
        assert!(!error().is_empty());
        cv_CascadeClassifier_delete(classifier);
        cv_Rect_vector_delete(boxes);
        let objects = cv_Point3f_vector_vector_new();
        let points = cv_Point2f_vector_vector_new();
        let camera = Image::owned(cv_Mat_new());
        let distortion = Image::owned(cv_Mat_new());
        assert!(cv_calibrateCamera(
            objects,
            points,
            cv_Size {
                width: 10,
                height: 10
            },
            camera.0,
            distortion.0
        )
        .is_nan());
        assert!(!error().is_empty());
        cv_Point3f_vector_vector_delete(objects);
        cv_Point2f_vector_vector_delete(points);
    }
}

#[test]
fn error_messages_are_thread_local() {
    unsafe {
        assert_eq!(cv_Mat_rows(ptr::null()), -1);
        let before = error();
        std::thread::spawn(|| {
            let image = Image::owned(cv_Mat_new());
            assert_eq!(cv_Mat_empty(image.0), 1);
            assert!(error().is_empty());
        })
        .join()
        .unwrap();
        assert_eq!(error(), before);
    }
}

#[test]
fn camera_calibration_recovers_a_known_intrinsic_matrix() {
    unsafe {
        let all_objects = cv_Point3f_vector_vector_new();
        let all_images = cv_Point2f_vector_vector_new();
        for (angle_x, angle_y, depth) in [
            (0.1f32, 0.2f32, 5.0f32),
            (-0.2, 0.1, 6.0),
            (0.15, -0.2, 5.5),
            (0.3, 0.15, 6.5),
        ] {
            let objects = cv_Point3f_vector_new();
            let images = cv_Point2f_vector_new();
            for y in 0..6 {
                for x in 0..7 {
                    let px = (x as f32 - 3.0) * 0.3;
                    let py = (y as f32 - 2.5) * 0.3;
                    let rx = angle_y.cos() * px + angle_y.sin() * angle_x.sin() * py;
                    let ry = angle_x.cos() * py;
                    let rz = -angle_y.sin() * px + angle_y.cos() * angle_x.sin() * py + depth;
                    assert_eq!(
                        cv_Point3f_vector_push(
                            objects,
                            cv_Point3f {
                                x: px,
                                y: py,
                                z: 0.0
                            }
                        ),
                        0
                    );
                    assert_eq!(
                        cv_Point2f_vector_push(
                            images,
                            cv_Point2f {
                                x: 600.0 * rx / rz + 320.0,
                                y: 600.0 * ry / rz + 240.0
                            }
                        ),
                        0
                    );
                }
            }
            assert_eq!(cv_Point3f_vector_vector_push(all_objects, objects), 0);
            assert_eq!(cv_Point2f_vector_vector_push(all_images, images), 0);
            cv_Point3f_vector_delete(objects);
            cv_Point2f_vector_delete(images);
        }
        let camera = Image::owned(cv_Mat_new());
        let distortion = Image::owned(cv_Mat_new());
        let rms = cv_calibrateCamera(
            all_objects,
            all_images,
            cv_Size {
                width: 640,
                height: 480,
            },
            camera.0,
            distortion.0,
        );
        assert!(
            rms.is_finite() && rms < 0.001,
            "calibration rms={rms}: {}",
            error()
        );
        let intrinsic: Vec<f64> = camera
            .bytes()
            .chunks_exact(8)
            .map(|x| f64::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        assert!(
            (intrinsic[0] - 600.0).abs() < 0.1 && (intrinsic[4] - 600.0).abs() < 0.1,
            "{intrinsic:?}"
        );
        cv_Point3f_vector_vector_delete(all_objects);
        cv_Point2f_vector_vector_delete(all_images);
    }
}
