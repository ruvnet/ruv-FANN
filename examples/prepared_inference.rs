//! Run with `cargo run --locked --example prepared_inference --no-default-features`.
use ruv_fann::{ActivationFunction, NetworkBuilder};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(2)
        .output_layer_with_activation(1, ActivationFunction::Linear, 1.0)
        .build();
    network.set_weights(&[0.5, -0.25, 0.1])?;
    let prepared = Arc::new(network.prepare()?);
    let worker = {
        let prepared = Arc::clone(&prepared);
        std::thread::spawn(move || -> Result<f32, ruv_fann::PreparedError> {
            let mut workspace = prepared.workspace()?;
            let mut output = [0.0];
            prepared.run_into(&[1.0, 2.0], &mut output, &mut workspace)?;
            Ok(output[0])
        })
    };
    println!(
        "Prepared output: {}",
        worker.join().expect("inference worker panicked")?
    );
    Ok(())
}
