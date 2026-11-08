use super::{ActivationFunction, Softmax};
use ndarray::{array, Array1};

#[test]
fn softmax_gradient_matches_dense_jacobian() {
    for size in [1, 2, 8, 64] {
        let input = Array1::from_iter((0..size).map(|i| (i as f64 * 0.37).sin() * 5.0));
        let gradient = Array1::from_iter((0..size).map(|i| (i as f64 * 0.73).cos()));
        let probability = Softmax.forward(&input);
        let actual = Softmax.backward(&input, &gradient);
        for i in 0..size {
            let mut expected = 0.0;
            for j in 0..size {
                let diagonal = if i == j { 1.0 } else { 0.0 };
                expected += gradient[j] * probability[i] * (diagonal - probability[j]);
            }
            assert!((actual[i] - expected).abs() < 1e-12);
        }
        assert!(actual.sum().abs() < 1e-12);
    }
}

#[test]
fn softmax_gradient_matches_finite_differences() {
    let input = array![-2.0, 0.2, 1.4, 3.0];
    let gradient = array![1.5, -0.7, 0.3, 0.2];
    let actual = Softmax.backward(&input, &gradient);
    let epsilon = 1e-5;
    for i in 0..input.len() {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[i] += epsilon;
        minus[i] -= epsilon;
        let numerical = (Softmax.forward(&plus).dot(&gradient)
            - Softmax.forward(&minus).dot(&gradient))
            / (2.0 * epsilon);
        assert!((actual[i] - numerical).abs() < 1e-9);
    }
}

#[test]
fn softmax_handles_empty_and_saturated_inputs() {
    let empty = Array1::zeros(0);
    assert!(Softmax.backward(&empty, &empty).is_empty());
    let gradient = Softmax.backward(&array![1000.0, -1000.0], &array![1.0, -1.0]);
    assert!(gradient
        .iter()
        .all(|value| value.is_finite() && value.abs() < 1e-12));
}

#[test]
#[should_panic(expected = "Softmax gradient dimension mismatch")]
fn softmax_rejects_misaligned_gradient() {
    Softmax.backward(&array![0.0, 1.0], &array![1.0]);
}
