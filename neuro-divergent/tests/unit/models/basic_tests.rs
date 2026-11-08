//! Dense primitives and model-boundary contracts, using shipped implementations.
use crate::support::*;
use ndarray::array;
use neuro_divergent::models::{layers::Dense, BaseModel, ModelConfig, RNNConfig, TimeSeriesInput};

#[test]
fn dense_forward_matches_hand_computed_matrix_product() {
    let mut layer = Dense::new(2, 2);
    layer.weights = array![[1., 2.], [3., 4.]];
    layer.bias = array![0.5, -0.5];
    assert_eq!(layer.forward(&array![[2., -1.]]), array![[-0.5, -0.5]]);
}

#[test]
fn dense_backward_matches_independent_weight_and_input_gradients() {
    let mut layer = Dense::new(2, 2);
    layer.weights = array![[1., 2.], [3., 4.]];
    let input = array![[2., -1.], [1., 3.]];
    let upstream = array![[0.5, 1.], [-2., 0.25]];
    let input_gradient = layer.backward(&input, &upstream);
    let weight_gradient = layer.grad_weights.clone().unwrap();
    assert_eq!(layer.grad_bias.as_ref().unwrap(), &array![-1.5, 1.25]);
    let eps = 1e-6;
    for row in 0..2 {
        for col in 0..2 {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[[row, col]] += eps;
            minus[[row, col]] -= eps;
            let scalar = |x: &ndarray::Array2<f64>| (&layer.forward(x) * &upstream).sum();
            let numerical = (scalar(&plus) - scalar(&minus)) / (2. * eps);
            assert!((input_gradient[[row, col]] - numerical).abs() < 1e-8);
            let mut plus_layer = layer.clone();
            let mut minus_layer = layer.clone();
            plus_layer.weights[[row, col]] += eps;
            minus_layer.weights[[row, col]] -= eps;
            let plus = (&plus_layer.forward(&input) * &upstream).sum();
            let minus = (&minus_layer.forward(&input) * &upstream).sum();
            let numerical = (plus - minus) / (2. * eps);
            assert!((weight_gradient[[row, col]] - numerical).abs() < 1e-8);
        }
    }
}

#[test]
fn actual_model_exposes_its_configuration_and_untrained_state() {
    let model = rnn(3, 5, 2);
    assert_eq!(model.horizon(), 3);
    assert_eq!(model.input_size(), 1);
    assert_eq!(model.config().model_type(), "RNN");
    assert!(!model.is_trained());
    assert!(model.predict(&TimeSeriesInput::new(vec![0.1; 4])).is_err());
}

#[test]
fn rnn_configuration_rejects_invalid_dimensions_and_learning_rate() {
    let mut config = RNNConfig::<f64>::default_with_horizon(0);
    assert!(config.validate().is_err());
    config.horizon = 2;
    config.hidden_size = 0;
    assert!(config.validate().is_err());
    config.hidden_size = 4;
    config.learning_rate = -0.1;
    assert!(config.validate().is_err());
}

#[test]
fn legacy_models_facade_does_not_report_empty_forecasts_as_success() -> TestResult {
    let legacy = neuro_divergent::models::NeuralForecast::<f64>::new()
        .with_model(Box::new(rnn(2, 4, 2)))
        .with_frequency("D")
        .build()?;
    assert!(legacy.predict().is_err());
    // The supported primary facade is independently exercised with real RNN
    // fit/predict/CV/persistence in the integration target.
    Ok(())
}

#[test]
fn legacy_row_data_requires_explicit_window_mapping_instead_of_empty_samples() {
    use neuro_divergent::models::data::{DataValue, TimeSeriesDataFrame, TimeSeriesSchema};
    let mut raw = TimeSeriesDataFrame::<f64>::new(TimeSeriesSchema::new("unique_id", "ds", "y"));
    for t in 0..4 {
        raw.data.push(std::collections::HashMap::from([
            ("unique_id".into(), DataValue::String("one".into())),
            ("ds".into(), DataValue::Int(START + t * DAY)),
            ("y".into(), DataValue::Float(t as f64)),
        ]));
    }
    assert!(raw.to_dataset().is_err());
}
