use ruv_fann::{
    ActivationFunction as A, AdamW, AdamWConfig, AutogradNetwork, BatchFloat, DenseLayer,
};

fn scalar<T: BatchFloat>(value: f64) -> T {
    T::from(value).unwrap()
}
fn single(weights: &[f64], outputs: usize) -> AutogradNetwork<f64> {
    AutogradNetwork::from_dense_layers(&[DenseLayer {
        inputs: 1,
        outputs,
        weights,
        bias: None,
        activation: A::Linear,
        steepness: 1.0,
    }])
    .unwrap()
}
fn fixture<T: BatchFloat>(activation: A) -> AutogradNetwork<T> {
    let a: Vec<T> = [0.3, -0.5, -0.4, 0.2, 0.1, 0.7].map(scalar).to_vec();
    let b: Vec<T> = [0.15, -0.2, 0.4].map(scalar).to_vec();
    let c: Vec<T> = [0.2, -0.3, 0.4, -0.5, 0.6, 0.1].map(scalar).to_vec();
    let d: Vec<T> = [0.5, 0.6].map(scalar).to_vec();
    AutogradNetwork::from_dense_layers(&[
        DenseLayer {
            inputs: 2,
            outputs: 3,
            weights: &a,
            bias: Some(&b),
            activation,
            steepness: scalar(0.8),
        },
        DenseLayer {
            inputs: 3,
            outputs: 2,
            weights: &c,
            bias: Some(&d),
            activation,
            steepness: scalar(1.2),
        },
    ])
    .unwrap() // All caller-owned source vectors are dropped here.
}

#[test]
fn exact_mse_and_reverse_affine_graph() {
    let network = AutogradNetwork::from_dense_layers(&[DenseLayer {
        inputs: 2,
        outputs: 2,
        weights: &[1.0, 2.0, 3.0, 4.0],
        bias: Some(&[0.5, -0.5]),
        activation: A::Linear,
        steepness: 1.0,
    }])
    .unwrap();
    let mut workspace = network.workspace(2, usize::MAX).unwrap();
    assert!(workspace.outputs().is_empty());
    assert!(workspace.parameter_gradients().is_empty());
    let loss = network
        .backward_mse(
            &[1.0, -1.0, 0.5, 2.0],
            &[0.5, -1.0, 4.0, 8.0],
            2,
            &mut workspace,
        )
        .unwrap();
    assert_eq!(loss, 0.8125);
    assert_eq!(workspace.outputs(), &[-0.5, -1.5, 5.0, 9.0]);
    assert_eq!(
        workspace.parameter_gradients(),
        &[-0.25, 1.5, 0.0, 1.25, 0.0, 0.25]
    );
    assert_eq!(workspace.input_gradients(), &[-1.25, -2.0, 2.0, 3.0]);
}

fn finite_differences<T: BatchFloat>(epsilon: f64, tolerance: f64) {
    for activation in [A::Linear, A::Sigmoid, A::Tanh, A::SigmoidSymmetric, A::ReLU] {
        let mut network = fixture::<T>(activation);
        let mut workspace = network.workspace(3, usize::MAX).unwrap();
        let inputs = [-0.7, 0.2, 0.1, -0.3, 0.8, 0.9].map(scalar::<T>);
        let targets = [0.1, 0.4, -0.2, 0.8, 0.5, -0.1].map(scalar::<T>);
        network
            .backward_mse(&inputs, &targets, 3, &mut workspace)
            .unwrap();
        let gradients = workspace.parameter_gradients().to_vec();
        let input_gradients = workspace.input_gradients().to_vec();
        let original = network.parameters().to_vec();
        for index in 0..original.len() {
            let mut values = original.clone();
            values[index] = original[index] + scalar(epsilon);
            network.set_parameters(&values).unwrap();
            let plus = network
                .backward_mse(&inputs, &targets, 3, &mut workspace)
                .unwrap();
            values[index] = original[index] - scalar(epsilon);
            network.set_parameters(&values).unwrap();
            let minus = network
                .backward_mse(&inputs, &targets, 3, &mut workspace)
                .unwrap();
            let numeric = (plus.to_f64().unwrap() - minus.to_f64().unwrap()) / (2.0 * epsilon);
            assert!(
                (numeric - gradients[index].to_f64().unwrap()).abs() < tolerance,
                "{activation:?} parameter {index}: numeric={numeric}, analytic={:?}",
                gradients[index].to_f64()
            );
        }
        network.set_parameters(&original).unwrap();
        for index in 0..inputs.len() {
            let mut values = inputs;
            values[index] = inputs[index] + scalar(epsilon);
            let plus = network
                .backward_mse(&values, &targets, 3, &mut workspace)
                .unwrap();
            values[index] = inputs[index] - scalar(epsilon);
            let minus = network
                .backward_mse(&values, &targets, 3, &mut workspace)
                .unwrap();
            let numeric = (plus.to_f64().unwrap() - minus.to_f64().unwrap()) / (2.0 * epsilon);
            assert!(
                (numeric - input_gradients[index].to_f64().unwrap()).abs() < tolerance,
                "{activation:?} input {index}: numeric={numeric}"
            );
        }
    }
}

#[test]
fn finite_difference_parameter_and_input_gradients_f64() {
    finite_differences::<f64>(1e-6, 5e-7);
}
#[test]
fn finite_difference_parameter_and_input_gradients_f32() {
    finite_differences::<f32>(2e-3, 5e-4);
}

#[test]
fn relu_uses_fann_steepness_and_zero_subgradient_conventions() {
    let network = AutogradNetwork::from_dense_layers(&[DenseLayer {
        inputs: 1,
        outputs: 1,
        weights: &[1.0],
        bias: None,
        activation: A::ReLU,
        steepness: 7.0,
    }])
    .unwrap();
    let mut workspace = network.workspace(3, usize::MAX).unwrap();
    network
        .backward_mse(&[-1.0, 0.0, 1.0], &[0.0; 3], 3, &mut workspace)
        .unwrap();
    assert_eq!(workspace.outputs(), &[0.0, 0.0, 1.0]);
    assert_eq!(workspace.input_gradients(), &[0.0, 0.0, 2.0 / 3.0]);
    assert_eq!(workspace.parameter_gradients(), &[2.0 / 3.0]);
}

fn learn_linear<T: BatchFloat>() {
    let mut network = AutogradNetwork::from_dense_layers(&[DenseLayer {
        inputs: 2,
        outputs: 2,
        weights: &[T::zero(); 4],
        bias: Some(&[T::zero(); 2]),
        activation: A::Linear,
        steepness: T::one(),
    }])
    .unwrap();
    let mut workspace = network.workspace(9, usize::MAX).unwrap();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for x in [-1.0, 0.0, 1.0] {
        for y in [-1.0, 0.0, 1.0] {
            inputs.extend([scalar::<T>(x), scalar(y)]);
            targets.extend([scalar::<T>(3.0 * x - 2.0 * y + 0.5), scalar(-x + y - 0.25)]);
        }
    }
    let initial = network
        .backward_mse(&inputs, &targets, 9, &mut workspace)
        .unwrap();
    for _ in 0..1000 {
        network
            .backward_mse(&inputs, &targets, 9, &mut workspace)
            .unwrap();
        network.sgd_step(&mut workspace, scalar(0.05)).unwrap();
    }
    let final_loss = network
        .backward_mse(&inputs, &targets, 9, &mut workspace)
        .unwrap();
    assert!(final_loss < scalar(1e-9));
    assert!(final_loss < initial * scalar(1e-8));
}
#[test]
fn sgd_learns_real_multioutput_regression_f32_f64() {
    learn_linear::<f32>();
    learn_linear::<f64>();
}

#[test]
fn adamw_learns_nonlinear_xor_and_freeze_is_independent() {
    let mut network = AutogradNetwork::from_dense_layers(&[
        DenseLayer {
            inputs: 2,
            outputs: 4,
            weights: &[0.4, -0.8, -0.6, 0.2, 0.7, 0.1, -0.3, -0.5],
            bias: Some(&[0.1, -0.2, 0.3, -0.1]),
            activation: A::Tanh,
            steepness: 1.0,
        },
        DenseLayer {
            inputs: 4,
            outputs: 1,
            weights: &[0.5, -0.4, 0.2, 0.3],
            bias: Some(&[-0.1]),
            activation: A::Linear,
            steepness: 1.0,
        },
    ])
    .unwrap();
    let inputs = [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let targets = [-1.0, 1.0, 1.0, -1.0];
    let frozen = network.prepare().unwrap();
    let mut frozen_workspace = frozen.workspace().unwrap();
    let mut before = [0.0];
    frozen
        .run_into(&inputs[..2], &mut before, &mut frozen_workspace)
        .unwrap();
    let mut workspace = network.workspace(4, usize::MAX).unwrap();
    let initial = network
        .backward_mse(&inputs, &targets, 4, &mut workspace)
        .unwrap();
    let mut optimizer = AdamW::new(
        &network,
        AdamWConfig {
            learning_rate: 0.03,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        },
        usize::MAX,
    )
    .unwrap();
    for _ in 0..1500 {
        network
            .backward_mse(&inputs, &targets, 4, &mut workspace)
            .unwrap();
        network.adamw_step(&mut workspace, &mut optimizer).unwrap();
    }
    let loss = network
        .backward_mse(&inputs, &targets, 4, &mut workspace)
        .unwrap();
    assert!(loss < 1e-8 && loss < initial * 1e-6, "loss={loss}");
    assert_eq!(optimizer.steps(), 1500);
    let mut after = [0.0];
    frozen
        .run_into(&inputs[..2], &mut after, &mut frozen_workspace)
        .unwrap();
    assert_eq!(before, after);
    let trained = network.prepare().unwrap();
    let mut output = [0.0; 4];
    let mut batch = trained.batch_workspace(4, usize::MAX).unwrap();
    trained
        .run_batch_into(&inputs, &mut output, 4, &mut batch)
        .unwrap();
    assert_eq!(&output, workspace.outputs());
}

#[path = "autograd/validation.rs"]
mod validation;
