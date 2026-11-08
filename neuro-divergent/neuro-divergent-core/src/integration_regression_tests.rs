use super::*;
use crate::data::TimeSeriesSchema;
use polars::prelude::*;

#[test]
fn bridge_uses_actual_network_dimensions_and_includes_the_final_window() {
    let frame = df! { "unique_id" => ["a", "a", "a"], "ds" => [0_i64, 86_400_000, 172_800_000], "y" => [0.0, 0.0, 1.0] }.unwrap();
    let dataset = TimeSeriesDataFrame::<f64>::from_polars(
        frame,
        TimeSeriesSchema::new("unique_id", "ds", "y"),
    )
    .unwrap()
    .to_dataset()
    .unwrap();
    let mut network = Network::<f64>::new(&[2, 3, 1]);
    let mut bridge =
        TrainingBridge::new(Box::new(ruv_fann::training::IncrementalBackprop::new(0.01)));
    let config = TrainingBridgeConfig {
        max_epochs: 1,
        learning_rate: 0.01,
        target_error: 0.0,
        validation_split: None,
        early_stopping: None,
        batch_size: None,
        shuffle: false,
    };
    assert!(bridge
        .train_network(&mut network, &dataset, &config)
        .unwrap()
        .is_finite());
    let state = bridge.state();
    let state = state.lock().unwrap();
    assert_eq!(state.epoch, 1);
    assert_eq!(state.metrics.training_loss.len(), 1);
    assert!(!state.training_active);
}

#[test]
fn adapter_describes_hidden_architecture_and_cascade_fails_explicitly() {
    let adapter = NetworkAdapter::from_network(Network::<f64>::new(&[2, 5, 3, 1]));
    assert_eq!(adapter.config().hidden_layers, [5, 3]);
    assert_eq!(adapter.config().activations.len(), 3);
    let mut bridge = TrainingBridge::new(Box::new(
        ruv_fann::training::IncrementalBackprop::<f64>::new(0.01),
    ));
    let result = bridge.setup_cascade_training(CascadeConfig {
        max_candidates: 4,
        correlation_threshold: 0.5,
        epochs_per_candidate: 1,
        min_improvement: 0.001,
    });
    assert!(result.unwrap_err().to_string().contains("not implemented"));
}
