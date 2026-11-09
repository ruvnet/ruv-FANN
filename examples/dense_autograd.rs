//! cargo run --locked --example dense_autograd
use ruv_fann::{ActivationFunction, AdamW, AdamWConfig, AutogradNetwork, DenseLayer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Learn y = 2*x + 1. Weights have shape [outputs, inputs]; bias is optional.
    let mut model = AutogradNetwork::<f64>::from_dense_layers(&[DenseLayer {
        inputs: 1,
        outputs: 1,
        weights: &[0.0],
        bias: Some(&[0.0]),
        activation: ActivationFunction::Linear,
        steepness: 1.0,
    }])?;
    let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let target = [-3.0, -1.0, 1.0, 3.0, 5.0];
    let mut tape = model.workspace(5, 1024 * 1024)?;
    let mut optimizer = AdamW::new(
        &model,
        AdamWConfig {
            learning_rate: 0.05,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        },
        1024 * 1024,
    )?;
    let first = model.backward_mse(&input, &target, 5, &mut tape)?;
    for _ in 0..500 {
        model.adamw_step(&mut tape, &mut optimizer)?;
        model.backward_mse(&input, &target, 5, &mut tape)?;
    }
    let last = model.backward_mse(&input, &target, 5, &mut tape)?;
    assert!(last < 1e-8);
    println!(
        "MSE {first:.6} -> {last:.12}; parameters {:?}",
        model.parameters()
    );

    // Freeze once; subsequent training cannot change this snapshot.
    let frozen = model.prepare()?;
    let mut workspace = frozen.workspace()?;
    let mut output = [0.0];
    frozen.run_into(&[3.0], &mut output, &mut workspace)?;
    assert!((output[0] - 7.0).abs() < 1e-4);
    println!("At x=3, frozen model predicts {}", output[0]);
    Ok(())
}
