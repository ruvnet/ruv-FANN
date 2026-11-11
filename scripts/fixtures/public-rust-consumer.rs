//! Compiled in a virgin consumer against exact public crates.io versions.
use candle_core::{Device, Tensor};
use candle_nn::Linear;
use ruv_fann::{ActivationFunction, AdamW, AdamWConfig, AutogradNetwork};
use ruv_fann_candle::{export_dense_f64_json, prepare_f64, LinearStage};
use serde_json::{json, Value};

fn values(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}
fn close(actual: &[f64], expected: &[f64], checks: &mut usize) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 1e-9 * b.abs().max(1.), "{a} != {b}");
        *checks += 1;
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: Value = serde_json::from_slice(&std::fs::read(std::env::args().nth(1).unwrap())?)?;
    assert_eq!(fixture["oracle"], "candle-core/candle-nn 0.11.1 CPU f64");
    let dense: Value = serde_json::from_str(fixture["dense_json"].as_str().unwrap())?;
    let layers = dense["layers"].as_array().unwrap();
    let mut linears = Vec::new();
    for layer in layers {
        let inputs = layer["inputs"].as_u64().unwrap() as usize;
        let outputs = layer["outputs"].as_u64().unwrap() as usize;
        let weight = Tensor::from_vec(values(&layer["weights"]), (outputs, inputs), &Device::Cpu)?;
        let bias = if layer["bias"].is_null() { None } else {
            Some(Tensor::from_vec(values(&layer["bias"]), outputs, &Device::Cpu)?)
        };
        linears.push(Linear::new(weight, bias));
    }
    let stages: Vec<_> = linears.iter().zip(layers).map(|(layer, data)| LinearStage {
        layer, activation: serde_json::from_value::<ActivationFunction>(data["activation"].clone()).unwrap(),
        steepness: data["steepness"].as_f64().unwrap(),
    }).collect();
    let exported = export_dense_f64_json(&stages)?;
    let prepared = prepare_f64(&stages)?;
    let input = values(&fixture["input"]);
    let target = values(&fixture["training_target"]);
    let mut output = vec![0.; target.len()];
    prepared.run_into(&input, &mut output, &mut prepared.workspace()?)?;
    let mut checks = 0;
    close(&output, &values(&fixture["expected"]), &mut checks);
    for kind in ["sgd", "adamw"] {
        let mut model = AutogradNetwork::<f64>::from_dense_layers_json(&exported)?;
        let mut tape = model.workspace(1, 1024 * 1024)?;
        let config = &fixture["adamw_config"];
        let mut optimizer = AdamW::new(&model, AdamWConfig {
            learning_rate: config["learning_rate"].as_f64().unwrap(),
            beta1: config["beta1"].as_f64().unwrap(), beta2: config["beta2"].as_f64().unwrap(),
            epsilon: config["epsilon"].as_f64().unwrap(), weight_decay: config["weight_decay"].as_f64().unwrap(),
        }, 1024 * 1024)?;
        let steps = fixture[format!("{kind}_steps")].as_array().unwrap();
        assert_eq!(steps.len(), 3);
        let frozen = model.prepare()?;
        for step in steps {
            let loss = model.backward_mse(&input, &target, 1, &mut tape)?;
            close(&[loss], &[step["loss"].as_f64().unwrap()], &mut checks);
            close(tape.outputs(), &values(&step["predictions"]), &mut checks);
            close(tape.parameter_gradients(), &values(&step["parameter_gradients"]), &mut checks);
            close(tape.input_gradients(), &values(&step["input_gradients"]), &mut checks);
            if kind == "sgd" { model.sgd_step(&mut tape, fixture["sgd_learning_rate"].as_f64().unwrap())?; }
            else { model.adamw_step(&mut tape, &mut optimizer)?; }
            close(model.parameters(), &values(&step["parameters"]), &mut checks);
        }
        frozen.run_into(&input, &mut output, &mut frozen.workspace()?)?;
        close(&output, &values(&fixture["expected"]), &mut checks);
    }
    println!("{}", json!({"checks": checks, "failures": 0, "skipped": 0, "tests": 3,
        "cases": ["candle-public-inference", "sgd-three-step-gradients", "adamw-three-step-gradients"]}));
    Ok(())
}
