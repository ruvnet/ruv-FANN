use super::{math, timeseries};

#[test]
fn scoring_rejects_empty_nonfinite_and_mismatched_forecasts() {
    assert!(math::mae::<f64>(&[], &[]).is_err());
    assert!(math::mse::<f64>(&[], &[]).is_err());
    assert!(math::rmse::<f64>(&[1.0], &[]).is_err());
    assert!(math::mse(&[f64::NAN], &[0.0]).is_err());
    assert!(math::mae(&[0.0], &[f64::INFINITY]).is_err());
    assert_eq!(math::mae(&[1.0, 3.0], &[2.0, 1.0]).unwrap(), 1.5);
    assert_eq!(math::mse(&[1.0, 3.0], &[2.0, 1.0]).unwrap(), 2.5);
}

#[test]
fn windows_reject_zero_and_overflowing_sizes_without_panicking() {
    let data = [1.0, 2.0, 3.0, 4.0];
    assert!(timeseries::create_sliding_windows(&data, 0, 1).is_empty());
    assert!(timeseries::create_sliding_windows(&data, 2, 0).is_empty());
    assert!(timeseries::create_sliding_windows(&data, usize::MAX, 1).is_empty());
    assert_eq!(
        timeseries::create_sliding_windows(&data, 2, 2),
        vec![vec![1.0, 2.0], vec![3.0, 4.0]]
    );
    assert_eq!(
        timeseries::create_sliding_windows(&data, 2, usize::MAX),
        vec![vec![1.0, 2.0]]
    );
}

#[test]
fn split_rejects_nan_and_keeps_chronological_order() {
    let data = [1.0, 2.0, 3.0, 4.0];
    assert!(timeseries::train_validation_split(&data, f64::NAN).is_err());
    assert!(timeseries::train_validation_split(&data, f64::INFINITY).is_err());
    assert_eq!(
        timeseries::train_validation_split(&data, 0.25).unwrap(),
        (vec![1.0, 2.0, 3.0], vec![4.0])
    );
}

#[test]
fn matrix_multiply_rejects_ragged_rows() {
    assert!(math::matrix_multiply(&[vec![1.0, 2.0], vec![1.0]], &[vec![1.0], vec![2.0]]).is_err());
    assert!(math::matrix_multiply(&[vec![1.0, 2.0]], &[vec![1.0, 2.0], vec![2.0]]).is_err());
    assert_eq!(
        math::matrix_multiply(
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
            &[vec![5.0, 6.0], vec![7.0, 8.0]]
        )
        .unwrap(),
        vec![vec![19.0, 22.0], vec![43.0, 50.0]]
    );
}

#[test]
fn softmax_of_empty_input_is_empty() {
    assert!(math::softmax::<f64>(&[]).is_empty());
}
