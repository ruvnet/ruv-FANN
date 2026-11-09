use super::{fixture, single};
use ruv_fann::{
    ActivationFunction as A, AdamW, AdamWConfig, AutogradError, AutogradNetwork, DenseLayer,
    PreparationLimits,
};

#[test]
fn shape_budget_and_activation_preflight() {
    let layer = DenseLayer {
        inputs: 1,
        outputs: 1,
        weights: &[0.0],
        bias: None,
        activation: A::Linear,
        steepness: 1.0,
    };
    assert!(AutogradNetwork::<f64>::from_dense_layers(&[]).is_err());
    for bad in [
        DenseLayer { inputs: 0, ..layer },
        DenseLayer {
            weights: &[],
            ..layer
        },
        DenseLayer {
            bias: Some(&[]),
            ..layer
        },
        DenseLayer {
            weights: &[f64::NAN],
            ..layer
        },
        DenseLayer {
            steepness: f64::INFINITY,
            ..layer
        },
    ] {
        assert!(AutogradNetwork::from_dense_layers(&[bad]).is_err());
    }
    assert!(matches!(
        AutogradNetwork::from_dense_layers(&[DenseLayer {
            activation: A::Gelu,
            ..layer
        }]),
        Err(AutogradError::UnsupportedActivation { .. })
    ));
    assert!(matches!(
        AutogradNetwork::from_dense_layers(&[DenseLayer {
            inputs: usize::MAX,
            outputs: 2,
            ..layer
        }]),
        Err(AutogradError::SizeOverflow)
    ));
    assert!(AutogradNetwork::from_dense_layers(&[
        layer,
        DenseLayer {
            inputs: 2,
            weights: &[0.0, 0.0],
            ..layer
        }
    ])
    .is_err());
    for limits in [
        PreparationLimits {
            max_bytes: 0,
            ..PreparationLimits::default()
        },
        PreparationLimits {
            max_connections: 0,
            ..PreparationLimits::default()
        },
        PreparationLimits {
            max_neurons: 0,
            ..PreparationLimits::default()
        },
    ] {
        assert!(matches!(
            AutogradNetwork::from_dense_layers_with_limits(&[layer], limits),
            Err(AutogradError::BudgetExceeded { .. })
        ));
    }
    let network = single(&[1.0], 1);
    let bytes = network.workspace_bytes(3).unwrap();
    assert_eq!(
        network.workspace(3, bytes).unwrap().estimated_bytes(),
        bytes
    );
    assert!(network.workspace(3, bytes - 1).is_err());
    assert!(network.workspace_bytes(0).is_err());
    assert!(network.workspace_bytes(usize::MAX).is_err());
    let per_row = network.training_work_estimate(1).unwrap();
    assert_eq!(per_row, 32 + 8 * 2 + 1);
    assert_eq!(network.training_work_estimate(0).unwrap(), 0);
    assert_eq!(network.training_work_estimate(3).unwrap(), 3 * per_row);
    let boundary = usize::MAX / per_row;
    assert_eq!(
        network.training_work_estimate(boundary).unwrap(),
        boundary * per_row
    );
    assert!(matches!(
        network.training_work_estimate(boundary + 1),
        Err(AutogradError::SizeOverflow)
    ));
    assert_eq!(network.optimizer_work_estimate().unwrap(), 64);
    let adam_bytes = AdamW::required_bytes(&network).unwrap();
    assert_eq!(
        AdamW::new(&network, AdamWConfig::default(), adam_bytes)
            .unwrap()
            .estimated_bytes(),
        adam_bytes
    );
    assert!(AdamW::new(&network, AdamWConfig::default(), adam_bytes - 1).is_err());
}

#[test]
fn stale_gradients_foreign_workspaces_and_invalid_passes_cannot_update() {
    let mut network = single(&[1.0], 1);
    let foreign = single(&[1.0], 1);
    let mut workspace = network.workspace(2, usize::MAX).unwrap();
    assert!(network.sgd_step(&mut workspace, 0.1).is_err());
    assert!(matches!(
        foreign.backward_mse(&[1.0], &[0.0], 1, &mut workspace),
        Err(AutogradError::WorkspaceMismatch)
    ));
    network
        .backward_mse(&[1.0], &[0.0], 1, &mut workspace)
        .unwrap();
    assert!(network.set_parameters(&[f64::NAN]).is_err());
    assert!(network.set_parameters(&[]).is_err());
    assert_eq!(network.parameters(), &[1.0]);
    network.sgd_step(&mut workspace, 0.1).unwrap();
    assert!(matches!(
        network.sgd_step(&mut workspace, 0.1),
        Err(AutogradError::StaleGradients)
    ));
    network
        .backward_mse(&[1.0], &[0.0], 1, &mut workspace)
        .unwrap();
    network.set_parameters(&[1.0]).unwrap();
    assert!(matches!(
        network.sgd_step(&mut workspace, 0.1),
        Err(AutogradError::StaleGradients)
    ));
    network
        .backward_mse(&[1.0], &[0.0], 1, &mut workspace)
        .unwrap();
    let mut output = [17.0];
    assert!(network
        .run_batch_into(&[], &mut output, 1, &mut workspace)
        .is_err());
    assert_eq!(output, [17.0]);
    assert!(workspace.outputs().is_empty() && workspace.parameter_gradients().is_empty());
    for (inputs, targets, rows) in [
        (&[1.0][..], &[f64::NAN][..], 1),
        (&[f64::INFINITY][..], &[0.0][..], 1),
        (&[][..], &[][..], 0),
        (&[1.0][..], &[0.0][..], 3),
    ] {
        assert!(network
            .backward_mse(inputs, targets, rows, &mut workspace)
            .is_err());
        assert!(network.sgd_step(&mut workspace, 0.1).is_err());
    }
}

#[test]
fn overflow_rolls_back_outputs_parameters_moments_and_optimizer_time() {
    let network = single(&[f64::MAX], 1);
    let mut workspace = network.workspace(1, usize::MAX).unwrap();
    let mut output = [17.0];
    assert!(network
        .run_batch_into(&[2.0], &mut output, 1, &mut workspace)
        .is_err());
    assert_eq!(output, [17.0]);
    let mut network = single(&[0.0, 1.0], 2);
    let mut workspace = network.workspace(1, usize::MAX).unwrap();
    network
        .backward_mse(&[1.0], &[-1.0, -1e100], 1, &mut workspace)
        .unwrap();
    assert!(network.sgd_step(&mut workspace, 1e250).is_err());
    assert_eq!(network.parameters(), &[0.0, 1.0]);
    network.sgd_step(&mut workspace, 1e-100).unwrap(); // Failed proposal retained valid gradients.
    let mut network = single(&[0.0, 0.0], 2);
    let mut workspace = network.workspace(1, usize::MAX).unwrap();
    let mut adam = AdamW::new(&network, AdamWConfig::default(), usize::MAX).unwrap();
    network
        .backward_mse(&[1e160], &[-1e-160, -1.0], 1, &mut workspace)
        .unwrap();
    assert!(network.adamw_step(&mut workspace, &mut adam).is_err());
    assert_eq!(network.parameters(), &[0.0, 0.0]);
    assert_eq!(adam.first_moments(), &[0.0, 0.0]);
    assert_eq!(adam.second_moments(), &[0.0, 0.0]);
    assert_eq!(adam.steps(), 0);
    network
        .backward_mse(&[1.0], &[-1.0, -1.0], 1, &mut workspace)
        .unwrap();
    network.adamw_step(&mut workspace, &mut adam).unwrap();
    assert_eq!(adam.steps(), 1);
    let network = single(&[1.0], 1);
    let mut workspace = network.workspace(1, usize::MAX).unwrap();
    assert!(network
        .backward_mse(&[1.0], &[f64::MAX], 1, &mut workspace)
        .is_err());
    assert!(workspace.parameter_gradients().is_empty());
}

#[test]
fn optimizer_configuration_identity_and_revision_validation() {
    let mut network = single(&[1.0], 1);
    let mut workspace = network.workspace(1, usize::MAX).unwrap();
    for config in [
        AdamWConfig {
            learning_rate: 0.0,
            ..AdamWConfig::default()
        },
        AdamWConfig {
            beta1: 1.0,
            ..AdamWConfig::default()
        },
        AdamWConfig {
            beta2: -0.1,
            ..AdamWConfig::default()
        },
        AdamWConfig {
            epsilon: 0.0,
            ..AdamWConfig::default()
        },
        AdamWConfig {
            weight_decay: -0.1,
            ..AdamWConfig::default()
        },
        AdamWConfig {
            learning_rate: f64::NAN,
            ..AdamWConfig::default()
        },
        AdamWConfig {
            epsilon: f64::INFINITY,
            ..AdamWConfig::default()
        },
    ] {
        assert!(AdamW::new(&network, config, usize::MAX).is_err());
    }
    let foreign = single(&[1.0], 1);
    let mut foreign_adam = AdamW::new(&foreign, AdamWConfig::default(), usize::MAX).unwrap();
    network
        .backward_mse(&[1.0], &[0.0], 1, &mut workspace)
        .unwrap();
    assert!(matches!(
        network.adamw_step(&mut workspace, &mut foreign_adam),
        Err(AutogradError::OptimizerMismatch)
    ));
    let mut adam = AdamW::new(&network, AdamWConfig::default(), usize::MAX).unwrap();
    network.sgd_step(&mut workspace, 0.1).unwrap();
    network
        .backward_mse(&[1.0], &[0.0], 1, &mut workspace)
        .unwrap();
    assert!(matches!(
        network.adamw_step(&mut workspace, &mut adam),
        Err(AutogradError::StaleOptimizer)
    ));
    for learning_rate in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(network.sgd_step(&mut workspace, learning_rate).is_err());
    }
}

#[cfg(feature = "serde")]
#[test]
fn shared_json_roundtrip_rejections_and_output_caps() {
    let network = fixture::<f64>(A::Tanh);
    let json = network.to_dense_layers_json().unwrap();
    let restored = AutogradNetwork::<f64>::from_dense_layers_json(&json).unwrap();
    assert_eq!(network.parameters(), restored.parameters());
    let prepared = ruv_fann::PreparedNetwork::<f64>::from_dense_layers_json(&json).unwrap();
    assert_eq!(prepared.num_inputs(), restored.num_inputs());
    for bad in [
        json.replace("dense-layers/1", "dense-layers/2"),
        json.replacen("\"inputs\":2", "\"inputs\":2,\"unknown\":true", 1),
        json.replace("Tanh", "Gelu"),
        json.replacen("\"inputs\":2", "\"inputs\":0", 1),
    ] {
        assert!(AutogradNetwork::<f64>::from_dense_layers_json(&bad).is_err());
    }
    assert!(
        AutogradNetwork::<f64>::from_dense_layers_json(&" ".repeat(4 * 1024 * 1024 + 1)).is_err()
    );
    let weights = vec![1.234567891234567e100; 200_000];
    let large = single(&weights, weights.len());
    assert!(matches!(
        large.to_dense_layers_json(),
        Err(AutogradError::BudgetExceeded { .. })
    ));
    let layer = DenseLayer {
        inputs: 1,
        outputs: 1,
        weights: &[0.0],
        bias: None,
        activation: A::Linear,
        steepness: 1.0,
    };
    let deep = AutogradNetwork::from_dense_layers(&[layer; 65]).unwrap();
    assert!(deep.to_dense_layers_json().is_err());
}
