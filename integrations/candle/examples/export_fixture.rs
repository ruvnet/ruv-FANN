//! Emit a real Candle f64 CPU reference for WASM/napi-rs interoperability tests.
use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::{Linear, Module, Optimizer, ParamsAdamW, SGD};
use ruv_fann::ActivationFunction as A;
use ruv_fann_candle::{export_dense_f64_json, LinearStage};
use serde::Serialize;

#[derive(Serialize)]
struct Fixture {
    oracle: &'static str,
    dense_json: String,
    input: Vec<f64>,
    expected: Vec<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    training_target: Vec<f64>,
    training_loss: f64,
    parameter_order: &'static str,
    parameter_gradients: Vec<f64>,
    input_gradients: Vec<f64>,
    sgd_learning_rate: f64,
    sgd_expected_parameters: Vec<f64>,
    adamw_config: AdamConfig,
    adamw_expected_parameters: Vec<f64>,
    sgd_steps: Vec<TrainingStep>,
    adamw_steps: Vec<TrainingStep>,
}

#[derive(Serialize)]
struct AdamConfig {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
}

#[derive(Serialize)]
struct TrainingStep {
    loss: f64,
    predictions: Vec<f64>,
    parameter_gradients: Vec<f64>,
    input_gradients: Vec<f64>,
    parameters: Vec<f64>,
}

fn flatten(vars: &[Var], gradients: Option<&GradStore>) -> candle_core::Result<Vec<f64>> {
    let mut values = Vec::new();
    for var in vars {
        let tensor = match gradients {
            Some(g) => g.get(var.as_tensor()).expect("Candle parameter gradient"),
            None => var.as_tensor(),
        };
        values.extend(tensor.flatten_all()?.to_vec1::<f64>()?);
    }
    Ok(values)
}

fn training_steps(
    first: &Linear,
    second: &Linear,
    input: &Var,
    target: &Tensor,
    variables: &[Var],
    mut optimizer: impl Optimizer,
) -> candle_core::Result<Vec<TrainingStep>> {
    let mut steps = Vec::new();
    for _ in 0..3 {
        let hidden = first
            .forward(input.as_tensor())?
            .affine(0.75, 0.0)?
            .tanh()?;
        let prediction = second.forward(&hidden)?;
        let loss = candle_nn::loss::mse(&prediction, target)?;
        let gradients = loss.backward()?;
        let mut step = TrainingStep {
            loss: loss.to_scalar::<f64>()?,
            predictions: prediction.flatten_all()?.to_vec1::<f64>()?,
            parameter_gradients: flatten(variables, Some(&gradients))?,
            input_gradients: gradients
                .get(input.as_tensor())
                .expect("Candle input gradient")
                .flatten_all()?
                .to_vec1::<f64>()?,
            parameters: Vec::new(),
        };
        optimizer.step(&gradients)?;
        step.parameters = flatten(variables, None)?;
        steps.push(step);
    }
    Ok(steps)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let first_weight = Var::new(
        &[
            [0.5f64, -0.25, 0.125],
            [-0.75, 0.5, 0.25],
            [0.125, 0.25, -0.5],
            [0.25, -0.125, 0.75],
        ],
        &Device::Cpu,
    )?;
    let first_bias = Var::new(&[0.125f64, -0.25, 0.5, 0.25], &Device::Cpu)?;
    let second_weight = Var::new(
        &[[0.25f64, -0.5, 0.125, 0.75], [-0.125, 0.25, 0.5, -0.25]],
        &Device::Cpu,
    )?;
    let first = Linear::new(
        first_weight.as_tensor().clone(),
        Some(first_bias.as_tensor().clone()),
    );
    let second = Linear::new(second_weight.as_tensor().clone(), None);
    let stages = [
        LinearStage {
            layer: &first,
            activation: A::Tanh,
            steepness: 0.75,
        },
        LinearStage {
            layer: &second,
            activation: A::Linear,
            steepness: 1.0,
        },
    ];
    let input = vec![0.2, -0.7, 0.9];
    let actual = Var::from_vec(input.clone(), (1, 3), &Device::Cpu)?;
    let hidden = first
        .forward(actual.as_tensor())?
        .affine(0.75, 0.0)?
        .tanh()?;
    let prediction = second.forward(&hidden)?;
    let expected = prediction.flatten_all()?.to_vec1::<f64>()?;
    let dense_json = export_dense_f64_json(&stages)?;
    let training_target = vec![0.1, -0.2];
    let target = Tensor::from_vec(training_target.clone(), (1, 2), &Device::Cpu)?;
    let loss = candle_nn::loss::mse(&prediction, &target)?;
    let gradients = loss.backward()?;
    let variables = vec![first_weight, first_bias, second_weight];
    let originals = variables
        .iter()
        .map(|var| var.copy())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let parameter_gradients = flatten(&variables, Some(&gradients))?;
    let input_gradients = gradients
        .get(actual.as_tensor())
        .expect("Candle input gradient")
        .flatten_all()?
        .to_vec1::<f64>()?;
    let sgd_learning_rate = 0.01;
    let sgd_steps = training_steps(
        &first,
        &second,
        &actual,
        &target,
        &variables,
        SGD::new(variables.clone(), sgd_learning_rate)?,
    )?;
    let sgd_expected_parameters = sgd_steps[0].parameters.clone();
    for (var, original) in variables.iter().zip(&originals) {
        var.set(original)?;
    }
    let adamw_config = AdamConfig {
        learning_rate: 0.005,
        beta1: 0.9,
        beta2: 0.99,
        epsilon: 1e-8,
        weight_decay: 0.01,
    };
    let adamw_steps = training_steps(
        &first,
        &second,
        &actual,
        &target,
        &variables,
        candle_nn::AdamW::new(
            variables.clone(),
            ParamsAdamW {
                lr: adamw_config.learning_rate,
                beta1: adamw_config.beta1,
                beta2: adamw_config.beta2,
                eps: adamw_config.epsilon,
                weight_decay: adamw_config.weight_decay,
            },
        )?,
    )?;
    let adamw_expected_parameters = adamw_steps[0].parameters.clone();
    let fixture = Fixture {
        oracle: "candle-core/candle-nn 0.11.1 CPU f64",
        dense_json,
        input,
        expected,
        absolute_tolerance: 1e-12,
        relative_tolerance: 1e-12,
        training_target,
        training_loss: loss.to_scalar::<f64>()?,
        parameter_order: "per layer: weights[out,in] row-major, then optional bias",
        parameter_gradients,
        input_gradients,
        sgd_learning_rate,
        sgd_expected_parameters,
        adamw_config,
        adamw_expected_parameters,
        sgd_steps,
        adamw_steps,
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &fixture)?;
    println!();
    Ok(())
}
