//! Numerical primitives used by attention-style models, without claiming a
//! shipped Transformer forecaster. These call the real activation/layer code.
use ndarray::array;
use neuro_divergent::models::{
    activations::{ActivationFunction, LogSoftmax, Softmax},
    layers::LayerNorm,
};

#[test]
fn stable_softmax_is_normalized_and_invariant_to_a_large_common_shift() {
    let function = Softmax::new();
    let input = array![1., 2., 3.];
    let shifted = array![1001., 1002., 1003.];
    let original = function.forward(&input);
    let actual = function.forward(&shifted);
    assert!((actual.sum() - 1.).abs() < 1e-14);
    for (a, b) in original.iter().zip(actual) {
        assert!((a - b).abs() < 1e-14);
    }
}

#[test]
fn softmax_vector_jacobian_product_matches_finite_differences() {
    let function = Softmax::new();
    let input = array![-0.5, 0.3, 1.2];
    let upstream = array![2., -1., 0.5];
    let gradient = function.backward(&input, &upstream);
    let eps = 1e-6;
    for i in 0..3 {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[i] += eps;
        minus[i] -= eps;
        let numerical = (function.forward(&plus).dot(&upstream)
            - function.forward(&minus).dot(&upstream))
            / (2. * eps);
        assert!((gradient[i] - numerical).abs() < 1e-8);
    }
    assert!(gradient.sum().abs() < 1e-14);
}

#[test]
fn log_softmax_and_softmax_express_the_same_probabilities() {
    let input = array![1000., 1001., 1002.];
    let probabilities = Softmax::new().forward(&input);
    let logs = LogSoftmax::new().forward(&input);
    for (p, log) in probabilities.iter().zip(logs) {
        assert!((p - log.exp()).abs() < 1e-12);
    }
}

#[test]
fn layer_normalization_normalizes_each_sequence_row_independently() {
    let result = LayerNorm::new(3, 1e-8).forward(&array![[1., 2., 3.], [10., 20., 30.]]);
    for row in result.rows() {
        assert!(row.sum().abs() < 1e-10);
        assert!((row.mapv(|v| v * v).sum() / 3. - 1.).abs() < 1e-6);
    }
    let constant = LayerNorm::new(3, 1e-8).forward(&array![[7., 7., 7.]]);
    assert!(constant.iter().all(|v| v.is_finite() && v.abs() < 1e-12));
}

#[test]
fn layer_normalization_is_invariant_to_a_large_common_offset() {
    let normalizer = LayerNorm::new(3, 1e-8);
    let reference = normalizer.forward(&array![[0., 1., 2.]]);
    let offset = normalizer.forward(&array![[1e12, 1e12 + 1., 1e12 + 2.]]);
    for (a, b) in reference.iter().zip(offset.iter()) {
        assert!(b.is_finite());
        assert!(
            (a - b).abs() < 1e-10,
            "centered variance should ignore the common offset"
        );
    }
}
