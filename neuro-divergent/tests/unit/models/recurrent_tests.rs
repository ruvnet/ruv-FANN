//! Real BPTT, recurrent state, prediction shapes and model state persistence.
use crate::support::*;
use neuro_divergent::models::recurrent::{BasicRecurrentCell, RecurrentLayer};
use neuro_divergent::models::ActivationFunction;
use neuro_divergent::models::{BaseModel, TimeSeriesInput};

#[test]
fn bptt_training_reduces_actual_loss_on_a_deterministic_dataset() -> TestResult {
    let data = model_dataset(12, 6, 3);
    let mut model = rnn(3, 5, 80);
    let metrics = model.fit(&data)?;
    let history = model.training_history().unwrap();
    assert!(history.epoch_losses.len() > 1);
    let first = history.epoch_losses[0];
    let last = *history.epoch_losses.last().unwrap();
    assert!(
        last < first * 0.5,
        "training must learn: first={first}, last={last}"
    );
    assert!(metrics.final_loss.is_finite());
    let validation = model.validate(&data)?;
    assert!(validation.mse.is_finite());
    assert!(validation.mse < 0.03);
    Ok(())
}

#[test]
fn recurrent_cell_state_roundtrip_changes_the_next_transition() -> TestResult {
    let mut cell = BasicRecurrentCell::<f64>::new(1, 3, ActivationFunction::Tanh);
    let initial = cell.get_state();
    assert_eq!(initial, vec![0.; 3]);
    let first = cell.forward_step(&[0.25])?;
    assert_eq!(first.len(), 3);
    assert_eq!(cell.get_state(), first);
    cell.reset_state();
    assert_eq!(cell.get_state(), initial);
    assert_eq!(cell.forward_step(&[0.25])?, first);
    assert!(cell.set_state(vec![0.; 2]).is_err());
    assert!(cell.forward_step(&[]).is_err());
    Ok(())
}

#[test]
fn actual_rnn_batch_predictions_equal_individual_predictions() -> TestResult {
    let mut model = rnn(2, 7, 20);
    model.fit(&model_dataset(8, 6, 2))?;
    let inputs = vec![
        TimeSeriesInput::new(vec![0.1; 6]),
        TimeSeriesInput::new(vec![-0.2; 6]),
    ];
    let batch = model.predict_batch(&inputs)?;
    for (input, output) in inputs.iter().zip(batch) {
        assert_eq!(output.forecasts, model.predict(input)?.forecasts);
        assert_eq!(output.forecasts.len(), 2);
    }
    Ok(())
}

#[test]
fn serialized_rnn_weights_and_training_state_reproduce_forecasts() -> TestResult {
    let mut model = rnn(3, 5, 20);
    model.fit(&model_dataset(8, 6, 3))?;
    let input = TimeSeriesInput::new(vec![0.1; 6]);
    let expected = model.predict(&input)?;
    let bytes = model.save_state()?;
    assert!(!bytes.is_empty());
    let mut restored = rnn(3, 5, 20);
    restored.load_state(&bytes)?;
    assert!(restored.is_trained());
    let output = restored.predict(&input)?;
    for (a, b) in output.forecasts.iter().zip(expected.forecasts) {
        assert!((a - b).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn reset_invalidates_trained_model_predictions_and_history() -> TestResult {
    let mut model = rnn(2, 4, 5);
    model.fit(&model_dataset(8, 4, 2))?;
    assert!(model.is_trained());
    model.reset();
    assert!(!model.is_trained());
    assert!(model.predict(&TimeSeriesInput::new(vec![0.1; 4])).is_err());
    model.fit(&model_dataset(8, 4, 2))?;
    assert!(model
        .predict(&TimeSeriesInput::new(vec![0.1; 4]))?
        .forecasts
        .iter()
        .all(|v| v.is_finite()));
    Ok(())
}

#[test]
fn reserved_gated_models_are_never_silent_vanilla_aliases() {
    use neuro_divergent::models::{
        models::{GRU, LSTM},
        RNNConfig,
    };
    assert!(LSTM::<f64>::new(RNNConfig::default_with_horizon(2)).is_err());
    assert!(GRU::<f64>::new(RNNConfig::default_with_horizon(2)).is_err());
}
