//! Materialized by scripts/rune.py with the unmodified library under core/.
//! This executable calls production APIs; it contains no replacement kernels.
use ruv_fann::calibration::{
    CalibrationConfig, CalibrationLimits, CalibrationRecord, ForecastRecord, ResidualCalibrator,
};
use ruv_fann::{
    ActivationFunction as A, AdamW, AdamWConfig, AutogradNetwork, BatchFloat, DenseLayer, Network,
};
use std::fmt::Debug;

const ACTIVATIONS: [A; 21] = [
    A::Linear,
    A::Threshold,
    A::ThresholdSymmetric,
    A::Sigmoid,
    A::SigmoidSymmetric,
    A::Tanh,
    A::Gaussian,
    A::GaussianSymmetric,
    A::Elliot,
    A::ElliotSymmetric,
    A::LinearPiece,
    A::LinearPieceSymmetric,
    A::ReLU,
    A::ReLULeaky,
    A::Sin,
    A::Cos,
    A::SinSymmetric,
    A::CosSymmetric,
    A::Gelu,
    A::Swish,
    A::LeakyRelu,
];

fn inference<T: BatchFloat + Debug>() -> usize {
    let mut checked = 0;
    for activation in ACTIVATIONS {
        for sparse in [false, true] {
            let mut network = Network::<T>::new(&[5, 7, 3]);
            let weights = (0..network.total_connections())
                .map(|i| T::from(((i * 7 % 19) as f64 - 9.) / 23.).unwrap())
                .collect::<Vec<_>>();
            network.set_weights(&weights).unwrap();
            network.set_activation_function_hidden(activation);
            network.set_activation_function_output(activation);
            if sparse {
                for layer in network.layers.iter_mut().skip(1) {
                    for neuron in layer.neurons.iter_mut().filter(|n| !n.is_bias) {
                        neuron.connections.reverse();
                        neuron.connections.remove(1);
                        neuron.connections.push(neuron.connections[0].clone());
                    }
                }
            }
            let prepared = network.prepare().unwrap();
            let bytes = prepared.batch_workspace_bytes(17).unwrap();
            assert!(prepared.batch_workspace(17, bytes - 1).is_err());
            let mut workspace = prepared.batch_workspace(17, bytes).unwrap();
            for rows in [0, 1, 2, 3, 8, 17] {
                let inputs = (0..rows * 5)
                    .map(|i| T::from(((i * 11 % 29) as f64 - 14.) / 17.).unwrap())
                    .collect::<Vec<_>>();
                let expected = inputs
                    .chunks_exact(5)
                    .flat_map(|row| network.run(row).unwrap())
                    .collect::<Vec<_>>();
                let mut outputs = vec![T::nan(); rows * 3];
                prepared
                    .run_batch_into(&inputs, &mut outputs, rows, &mut workspace)
                    .unwrap();
                assert_eq!(
                    outputs, expected,
                    "{activation:?}, sparse={sparse}, rows={rows}"
                );
                if rows > 0 {
                    let mut invalid = inputs;
                    *invalid.last_mut().unwrap() = T::nan();
                    let before = outputs.clone();
                    assert!(prepared
                        .run_batch_into(&invalid, &mut outputs, rows, &mut workspace)
                        .is_err());
                    assert_eq!(
                        outputs, before,
                        "a late invalid row must preserve every output"
                    );
                }
                checked += rows;
            }
        }
    }
    checked
}

fn calibration() {
    let mut records = Vec::new();
    for window in 1..=9 {
        for step in [1, 2] {
            records.push(CalibrationRecord {
                series: "series".into(),
                model: "model".into(),
                cutoff: window * 10,
                timestamp: window * 10 + step,
                actual: 100.,
                prediction: 100. - (window * step) as f64,
            });
        }
    }
    records.reverse();
    let config = CalibrationConfig {
        as_of: 1000_i64,
        horizon: 2,
        coverages: vec![0.5, 0.9],
        limits: CalibrationLimits::default(),
    };
    let engine = ResidualCalibrator::fit(config.clone(), &records).unwrap();
    assert_eq!(engine.groups().count(), 2);
    let forecasts = [1, 2].map(|step| ForecastRecord {
        series: "series".into(),
        model: "model".into(),
        origin: 1000,
        timestamp: 1000 + step,
        prediction: 19. + step as f64,
    });
    let intervals = engine.apply(&forecasts).unwrap();
    assert_eq!(intervals.len(), 4);
    assert_eq!((intervals[1].lower, intervals[1].upper), (11., 29.));
    assert!(intervals
        .iter()
        .all(|row| row.calibration_samples == 9 && row.calibration_as_of == 1000));
    let mut future = config;
    future.as_of = 50;
    assert!(ResidualCalibrator::fit(future, &records).is_err());
    let mut invalid = forecasts;
    invalid[0].origin = 999;
    assert!(engine.apply(&invalid).is_err());

    // Force the exact partial-selection path with a hand-checkable rank.
    let large = (1..=2048)
        .map(|i| CalibrationRecord {
            series: "large".into(),
            model: "model".into(),
            cutoff: i * 2,
            timestamp: i * 2 + 1,
            actual: 0.,
            prediction: -(i as f64),
        })
        .collect::<Vec<_>>();
    let engine = ResidualCalibrator::fit(
        CalibrationConfig {
            as_of: 10_000,
            horizon: 1,
            coverages: vec![0.5, 0.9],
            limits: CalibrationLimits::default(),
        },
        &large,
    )
    .unwrap();
    assert_eq!(engine.groups().next().unwrap().radii, [1025., 1845.]);
}

fn training<T: BatchFloat + Debug>() {
    let weights = [T::zero()];
    let bias = [T::zero()];
    let layer = DenseLayer {
        inputs: 1,
        outputs: 1,
        weights: &weights,
        bias: Some(&bias),
        activation: A::Linear,
        steepness: T::one(),
    };
    let inputs = [-1., 0., 1.].map(|n| T::from(n).unwrap());
    let targets = [-1., 1., 3.].map(|n| T::from(n).unwrap());
    for adamw in [false, true] {
        let mut model = AutogradNetwork::from_dense_layers(&[layer]).unwrap();
        let bytes = model.workspace_bytes(3).unwrap();
        assert!(model.workspace(3, bytes - 1).is_err());
        let mut workspace = model.workspace(3, bytes).unwrap();
        let frozen = model.prepare().unwrap();
        let mut frozen_workspace = frozen.workspace().unwrap();
        let mut optimizer = AdamW::new(
            &model,
            AdamWConfig {
                learning_rate: T::from(0.05).unwrap(),
                weight_decay: T::zero(),
                ..AdamWConfig::default()
            },
            usize::MAX,
        )
        .unwrap();
        let first = model
            .backward_mse(&inputs, &targets, 3, &mut workspace)
            .unwrap();
        assert!((first.to_f64().unwrap() - 11. / 3.).abs() < 1e-6);
        for _ in 0..300 {
            if adamw {
                model.adamw_step(&mut workspace, &mut optimizer).unwrap();
            } else {
                model
                    .sgd_step(&mut workspace, T::from(0.05).unwrap())
                    .unwrap();
            }
            model
                .backward_mse(&inputs, &targets, 3, &mut workspace)
                .unwrap();
        }
        let loss = model
            .backward_mse(&inputs, &targets, 3, &mut workspace)
            .unwrap();
        assert!(loss < T::from(1e-6).unwrap());
        assert_eq!(workspace.parameter_gradients().len(), 2);
        assert_eq!(workspace.input_gradients().len(), 3);
        let mut frozen_output = [T::nan()];
        frozen
            .run_into(&[T::one()], &mut frozen_output, &mut frozen_workspace)
            .unwrap();
        assert_eq!(frozen_output, [T::zero()]);
        let before = model.parameters().to_vec();
        assert!(model.sgd_step(&mut workspace, T::nan()).is_err());
        assert_eq!(model.parameters(), before);
    }
}

fn main() {
    let f32_rows = inference::<f32>();
    let f64_rows = inference::<f64>();
    calibration();
    training::<f32>();
    training::<f64>();
    println!("{{\"status\":\"passed\",\"probe\":\"ruv-fann-core\",\"activations\":21,\"float32_rows\":{f32_rows},\"float64_rows\":{f64_rows},\"dense_sparse_duplicate_connections\":true,\"late_error_atomicity\":true,\"storage_budget_rejection\":true,\"dense_autograd\":true,\"mse_mean_all\":true,\"sgd_adamw_learning\":true,\"frozen_training_isolation\":true,\"calibration_groups\":2,\"intervals\":4,\"partial_selection_samples\":2048,\"past_only_guards\":true}}");
}
