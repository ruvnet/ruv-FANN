//! Actual Candle Var/backward and optimizers are the independent CPU oracle.
use candle_core::{backprop::GradStore, Device, Tensor, Var, WithDType};
use candle_nn::{Linear, Module, Optimizer, ParamsAdamW, SGD};
use ruv_fann::{ActivationFunction as A, AdamW, AdamWConfig, AutogradNetwork, BatchFloat};
use ruv_fann_candle::{
    export_dense_f32_json, export_dense_f64_json, CandleAdapterError, LinearStage,
};
use serde::de::DeserializeOwned;

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Export<T> = fn(&[LinearStage<'_, T>]) -> Result<String, CandleAdapterError>;

struct Graph {
    weights: [Var; 2],
    biases: [Option<Var>; 2],
    layers: [Linear; 2],
    activation: A,
}

fn values<T: WithDType>(input: &[f64]) -> Vec<T> {
    input.iter().map(|&x| T::from_f64(x)).collect()
}

impl Graph {
    fn new<T: WithDType>(
        activation: A,
        first_bias: bool,
        second_bias: bool,
    ) -> candle_core::Result<Self> {
        let weights = [
            Var::from_vec(
                values::<T>(&[
                    0.5, -0.25, 0.125, -0.75, 0.5, 0.25, 0.125, 0.25, -0.5, 0.25, -0.125, 0.75,
                ]),
                (4, 3),
                &Device::Cpu,
            )?,
            Var::from_vec(
                values::<T>(&[0.25, -0.5, 0.125, 0.75, -0.125, 0.25, 0.5, -0.25]),
                (2, 4),
                &Device::Cpu,
            )?,
        ];
        let biases = [
            first_bias
                .then(|| Var::from_vec(values::<T>(&[0.125, -0.25, 0.5, 0.25]), 4, &Device::Cpu))
                .transpose()?,
            second_bias
                .then(|| Var::from_vec(values::<T>(&[0.1, -0.2]), 2, &Device::Cpu))
                .transpose()?,
        ];
        let layers = std::array::from_fn(|i| {
            Linear::new(
                weights[i].as_tensor().clone(),
                biases[i].as_ref().map(|x| x.as_tensor().clone()),
            )
        });
        Ok(Self {
            weights,
            biases,
            layers,
            activation,
        })
    }

    fn stages<T: WithDType>(&self) -> [LinearStage<'_, T>; 2] {
        [
            LinearStage {
                layer: &self.layers[0],
                activation: self.activation,
                steepness: T::from_f64(0.75),
            },
            LinearStage {
                layer: &self.layers[1],
                activation: A::Linear,
                steepness: T::from_f64(1.0),
            },
        ]
    }

    fn parameters(&self) -> Vec<Var> {
        let mut result = Vec::new();
        for i in 0..2 {
            result.push(self.weights[i].clone());
            result.extend(self.biases[i].iter().cloned());
        }
        result
    }

    fn flat<T: WithDType>(&self, gradients: Option<&GradStore>) -> candle_core::Result<Vec<T>> {
        let mut result = Vec::new();
        for parameter in self.parameters() {
            let value = match gradients {
                Some(g) => g
                    .get(parameter.as_tensor())
                    .expect("Candle parameter gradient"),
                None => parameter.as_tensor(),
            };
            result.extend(value.flatten_all()?.to_vec1::<T>()?);
        }
        Ok(result)
    }

    fn forward<T: WithDType>(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.layers[0].forward(input)?;
        let hidden = match self.activation {
            A::Linear => hidden.affine(0.75, 0.0)?,
            A::Sigmoid => candle_nn::ops::sigmoid(&hidden.affine(0.75, 0.0)?)?,
            A::Tanh | A::SigmoidSymmetric => hidden.affine(0.75, 0.0)?.tanh()?,
            A::ReLU => {
                // Candle chooses derivative 1 at zero; FANN chooses 0. Keep
                // this external derivative oracle strictly away from the kink.
                assert!(hidden
                    .flatten_all()?
                    .to_vec1::<T>()?
                    .iter()
                    .all(|&x| WithDType::to_f64(x).abs() > 1e-5));
                hidden.relu()?
            }
            _ => unreachable!(),
        };
        self.layers[1].forward(&hidden)
    }
}

fn close<T: WithDType>(label: &str, actual: &[T], expected: &[T], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (WithDType::to_f64(a), WithDType::to_f64(b));
        assert!(
            (a - b).abs() <= atol + rtol * b.abs(),
            "{label}[{index}]: FANN {a}, Candle {b}"
        );
    }
}

fn compare<T: BatchFloat + WithDType + DeserializeOwned>(
    export: Export<T>,
    activation: A,
    first_bias: bool,
    second_bias: bool,
    optimizer: Option<bool>,
    atol: f64,
    rtol: f64,
) -> TestResult {
    let graph = Graph::new::<T>(activation, first_bias, second_bias)?;
    let input_values = values::<T>(&[0.2, -0.7, 0.9, -0.35, 0.1, 0.4, 0.6, -0.4, -0.2]);
    let target_values = values::<T>(&[0.1, -0.2, 0.3, 0.15, -0.25, 0.4]);
    let input = Var::from_vec(input_values.clone(), (3, 3), &Device::Cpu)?;
    let target = Tensor::from_vec(target_values.clone(), (3, 2), &Device::Cpu)?;
    let mut network = AutogradNetwork::<T>::from_dense_layers_json(&export(&graph.stages::<T>())?)?;
    let mut workspace = network.workspace(3, 64 * 1024 * 1024)?;
    close(
        "initial parameters",
        network.parameters(),
        &graph.flat::<T>(None)?,
        0.,
        0.,
    );
    let mut candle_sgd = SGD::new(graph.parameters(), 0.01)?;
    let mut candle_adam = candle_nn::AdamW::new(
        graph.parameters(),
        ParamsAdamW {
            lr: 0.005,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.01,
        },
    )?;
    let mut adam = AdamW::new(
        &network,
        AdamWConfig {
            learning_rate: T::from_f64(0.005),
            beta1: T::from_f64(0.9),
            beta2: T::from_f64(0.99),
            epsilon: T::from_f64(1e-8),
            weight_decay: T::from_f64(0.01),
        },
        64 * 1024 * 1024,
    )?;
    for _ in 0..if optimizer.is_some() { 3 } else { 1 } {
        let prediction = graph.forward::<T>(input.as_tensor())?;
        let loss = candle_nn::loss::mse(&prediction, &target)?;
        let gradients = loss.backward()?;
        let actual_loss = network.backward_mse(&input_values, &target_values, 3, &mut workspace)?;
        close(
            "MSE loss",
            &[actual_loss],
            &[loss.to_scalar::<T>()?],
            atol,
            rtol,
        );
        close(
            "predictions",
            workspace.outputs(),
            &prediction.flatten_all()?.to_vec1::<T>()?,
            atol,
            rtol,
        );
        close(
            "parameter gradients",
            workspace.parameter_gradients(),
            &graph.flat::<T>(Some(&gradients))?,
            atol,
            rtol,
        );
        close(
            "input gradients",
            workspace.input_gradients(),
            &gradients
                .get(input.as_tensor())
                .expect("Candle input gradient")
                .flatten_all()?
                .to_vec1::<T>()?,
            atol,
            rtol,
        );
        match optimizer {
            Some(false) => {
                candle_sgd.step(&gradients)?;
                network.sgd_step(&mut workspace, T::from_f64(0.01))?;
            }
            Some(true) => {
                candle_adam.step(&gradients)?;
                network.adamw_step(&mut workspace, &mut adam)?;
            }
            None => {}
        }
        close(
            "updated parameters",
            network.parameters(),
            &graph.flat::<T>(None)?,
            atol,
            rtol,
        );
    }
    Ok(())
}

fn all_cases<T: BatchFloat + WithDType + DeserializeOwned>(
    export: Export<T>,
    optimizers: bool,
    atol: f64,
    rtol: f64,
) -> TestResult {
    for activation in [A::Linear, A::Sigmoid, A::Tanh, A::SigmoidSymmetric, A::ReLU] {
        for first in [false, true] {
            for second in [false, true] {
                if optimizers {
                    for adam in [false, true] {
                        compare::<T>(export, activation, first, second, Some(adam), atol, rtol)?;
                    }
                } else {
                    compare::<T>(export, activation, first, second, None, atol, rtol)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn f32_mse_parameter_and_input_gradients_match_actual_candle() -> TestResult {
    all_cases::<f32>(export_dense_f32_json, false, 5e-6, 3e-5)
}

#[test]
fn f64_mse_parameter_and_input_gradients_match_actual_candle() -> TestResult {
    all_cases::<f64>(export_dense_f64_json, false, 1e-12, 1e-12)
}

#[test]
fn f32_three_sgd_and_adamw_steps_match_actual_candle() -> TestResult {
    all_cases::<f32>(export_dense_f32_json, true, 5e-6, 3e-5)
}

#[test]
fn f64_three_sgd_and_adamw_steps_match_actual_candle() -> TestResult {
    all_cases::<f64>(export_dense_f64_json, true, 1e-12, 1e-12)
}
