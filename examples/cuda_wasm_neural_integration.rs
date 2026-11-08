//! Run FANN inference with reusable output storage, then an optional WebGPU primitive.
//!
//! `cargo run --example cuda_wasm_neural_integration --features gpu`
//! uses the root crate's compute backend. The separate `cuda-wasm` package
//! translates CUDA source with `cuda2rust`; it does not execute this network.

use ruv_fann::{ActivationFunction, NetworkBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(2)
        .hidden_layer_with_activation(4, ActivationFunction::Tanh, 1.0)
        .output_layer_with_activation(1, ActivationFunction::Linear, 1.0)
        .build();
    let weights: Vec<f32> = (0..network.get_weights().len())
        .map(|i| (i % 7) as f32 * 0.1 - 0.3)
        .collect();
    network.set_weights(&weights)?;
    let mut output = [0.0];
    for input in [[0.5, 0.8], [0.2, 0.9], [-0.5, 0.3]] {
        network.run_into(&input, &mut output)?;
        println!("FANN inference {input:?} -> {output:?}");
    }
    compute_example()?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn compute_example() -> Result<(), Box<dyn std::error::Error>> {
    use ruv_fann::webgpu::{ComputeBackend, WebGPUBackend};
    let backend = WebGPUBackend::<f32>::new()?;
    // Large enough to exercise the GPU dispatch path. Verify every result.
    let (rows, cols) = (129, 79);
    let matrix = vec![0.5; rows * cols];
    let input = vec![2.0; cols];
    let output = backend.matrix_vector_multiply(&matrix, &input, rows, cols)?;
    assert!(output.iter().all(|v| (*v - cols as f32).abs() < 1e-4));
    println!(
        "WebGPU matrix/vector product: {} verified rows",
        output.len()
    );
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn compute_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Enable --features gpu to run the WebGPU matrix/vector example.");
    Ok(())
}
