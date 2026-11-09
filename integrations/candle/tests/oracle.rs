//! Actual Candle CPU is the external oracle; FANN's scalar/batch agreement is exact.
use candle_core::{Device, Tensor, WithDType};
use candle_nn::{Linear, Module};
use ruv_fann::{ActivationFunction as A, BatchFloat, PreparedNetwork};
use ruv_fann_candle::{prepare_f32, prepare_f64, CandleAdapterError, LinearStage};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn values<T: WithDType>(count: usize, seed: usize) -> Vec<T> {
    (0..count)
        .map(|i| T::from_f64(((i * 17 + seed) % 31) as f64 / 64.0 - 0.25))
        .collect()
}

fn activate<T: WithDType>(
    tensor: Tensor,
    stage: &LinearStage<'_, T>,
) -> candle_core::Result<Tensor> {
    let steepness = WithDType::to_f64(stage.steepness);
    match stage.activation {
        A::Linear => tensor.affine(steepness, 0.0),
        A::ReLU => tensor.relu(), // FANN's ReLU intentionally ignores steepness.
        A::Tanh => tensor.affine(steepness, 0.0)?.tanh(),
        A::Sigmoid => candle_nn::ops::sigmoid(&tensor.affine(steepness, 0.0)?),
        _ => unreachable!("This oracle covers explicitly equivalent activations only"),
    }
}

fn compare<T: BatchFloat + WithDType>(
    prepare: fn(&[LinearStage<'_, T>]) -> Result<PreparedNetwork<T>, CandleAdapterError>,
    activation: A,
    first_bias: bool,
    second_bias: bool,
    atol: f64,
    rtol: f64,
) -> TestResult {
    let (inputs, hidden, outputs, rows) = (37, 19, 7, 17);
    let bias = |enabled: bool, count: usize, seed: usize| -> candle_core::Result<Option<Tensor>> {
        enabled
            .then(|| Tensor::from_vec(values::<T>(count, seed), count, &Device::Cpu))
            .transpose()
    };
    let first = Linear::new(
        Tensor::from_vec(
            values::<T>(inputs * hidden, 3),
            (hidden, inputs),
            &Device::Cpu,
        )?,
        bias(first_bias, hidden, 1)?,
    );
    let second = Linear::new(
        Tensor::from_vec(
            values::<T>(hidden * outputs, 7),
            (outputs, hidden),
            &Device::Cpu,
        )?,
        bias(second_bias, outputs, 5)?,
    );
    let stages = [
        LinearStage {
            layer: &first,
            activation,
            steepness: T::from_f64(0.75),
        },
        LinearStage {
            layer: &second,
            activation: A::Linear,
            steepness: T::from_f64(1.0),
        },
    ];
    let input = values::<T>(inputs * rows, 17);
    let mut expected = Tensor::from_vec(input.clone(), (rows, inputs), &Device::Cpu)?;
    for stage in &stages {
        expected = activate(stage.layer.forward(&expected)?, stage)?;
    }
    let expected = expected.flatten_all()?.to_vec1::<T>()?;
    let prepared = prepare(&stages)?;
    assert_eq!(
        (prepared.num_inputs(), prepared.num_outputs()),
        (inputs, outputs)
    );
    let mut scalar = vec![T::zero(); rows * outputs];
    let mut workspace = prepared.workspace()?;
    for (x, y) in input
        .chunks_exact(inputs)
        .zip(scalar.chunks_exact_mut(outputs))
    {
        prepared.run_into(x, y, &mut workspace)?;
    }
    for (index, (&actual, &expected)) in scalar.iter().zip(&expected).enumerate() {
        let actual = WithDType::to_f64(actual);
        let expected = WithDType::to_f64(expected);
        assert!(
            (actual - expected).abs() <= atol + rtol * expected.abs(),
            "Candle mismatch at {index}: {actual} vs {expected}"
        );
    }
    let mut batch = prepared.batch_workspace(rows, prepared.batch_workspace_bytes(rows)?)?;
    for active in [1, 3, 7, 17, 2, 17] {
        let mut actual = vec![T::zero(); active * outputs];
        prepared.run_batch_into(&input[..active * inputs], &mut actual, active, &mut batch)?;
        for (&a, &b) in actual.iter().zip(&scalar) {
            assert!(
                a == b && (a != T::zero() || a.is_sign_negative() == b.is_sign_negative()),
                "FANN batch must retain its scalar operation order"
            );
        }
    }
    Ok(())
}

#[test]
fn f32_actual_candle_two_stage_oracle_all_bias_combinations() -> TestResult {
    for activation in [A::Linear, A::ReLU, A::Sigmoid, A::Tanh] {
        for first in [false, true] {
            for second in [false, true] {
                compare::<f32>(prepare_f32, activation, first, second, 5e-6, 2e-5)?;
            }
        }
    }
    Ok(())
}

#[test]
fn f64_actual_candle_two_stage_oracle_all_bias_combinations() -> TestResult {
    for activation in [A::Linear, A::ReLU, A::Sigmoid, A::Tanh] {
        for first in [false, true] {
            for second in [false, true] {
                compare::<f64>(prepare_f64, activation, first, second, 1e-12, 1e-12)?;
            }
        }
    }
    Ok(())
}

#[test]
fn logical_transpose_and_nonzero_weight_bias_offsets_match_candle() -> TestResult {
    let source = Tensor::new(&[[1f64, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;
    let weight = source.t()?; // logical [2,3], storage is [3,2].
    assert!(!weight.is_contiguous());
    let bias = Tensor::new(&[99f64, 0.5, -0.25, 77.], &Device::Cpu)?.narrow(0, 1, 2)?;
    let layer = Linear::new(weight, Some(bias));
    let prepared = prepare_f64(&[LinearStage {
        layer: &layer,
        activation: A::Linear,
        steepness: 1.,
    }])?;
    let input = [1., 10., 100.];
    let expected = layer
        .forward(&Tensor::new(&[input], &Device::Cpu)?)?
        .to_vec2::<f64>()?;
    let mut output = [0.; 2];
    prepared.run_into(&input, &mut output, &mut prepared.workspace()?)?;
    assert_eq!(output, [531.5, 641.75]);
    assert_eq!(output.as_slice(), expected[0]);

    let source = Tensor::new(
        &[
            [99f32, 99., 99.],
            [1., 3., 5.],
            [2., 4., 6.],
            [77., 77., 77.],
        ],
        &Device::Cpu,
    )?;
    let layer = Linear::new(source.narrow(0, 1, 2)?, None);
    let prepared = prepare_f32(&[LinearStage {
        layer: &layer,
        activation: A::Linear,
        steepness: 1.,
    }])?;
    let expected = layer
        .forward(&Tensor::new(&[[1f32, 10., 100.]], &Device::Cpu)?)?
        .to_vec2::<f32>()?;
    let mut output = [0.; 2];
    prepared.run_into(&[1., 10., 100.], &mut output, &mut prepared.workspace()?)?;
    assert_eq!(output.as_slice(), expected[0]);
    Ok(())
}

#[test]
fn prepared_snapshot_survives_source_variable_update_and_drop() -> TestResult {
    let variable = candle_core::Var::new(&[[1f64, 2.]], &Device::Cpu)?;
    let layer = Linear::new(variable.as_tensor().clone(), None);
    let prepared = prepare_f64(&[LinearStage {
        layer: &layer,
        activation: A::Linear,
        steepness: 1.,
    }])?;
    variable.set(&Tensor::zeros(
        (1, 2),
        candle_core::DType::F64,
        &Device::Cpu,
    )?)?;
    assert_eq!(
        layer
            .forward(&Tensor::new(&[[3f64, 4.]], &Device::Cpu)?)?
            .to_vec2::<f64>()?,
        vec![vec![0.]]
    );
    drop(layer);
    drop(variable);
    let mut output = [0.];
    prepared.run_into(&[3., 4.], &mut output, &mut prepared.workspace()?)?;
    assert_eq!(output, [11.]);
    Ok(())
}
