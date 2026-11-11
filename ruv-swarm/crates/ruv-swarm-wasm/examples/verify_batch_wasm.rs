//! Actual WASM SIMD equivalence probe for both supported native element types.
//! Compile this crate's example with --target wasm32-unknown-unknown and +simd128.
#[cfg(target_arch = "wasm32")]
mod probe {
    use ruv_fann::{ActivationFunction as A, BatchFloat, Network, PreparedError};
    use wasm_bindgen::prelude::*;

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

    fn check<T: BatchFloat + std::fmt::Debug>() -> usize {
        let mut checked = 0;
        for activation in ACTIVATIONS {
            for sparse in [false, true] {
                let mut model = Network::<T>::new(&[3, 5, 2]);
                let weights: Vec<_> = (0..model.total_connections())
                    .map(|i| T::from(((i * 7 % 19) as f64 - 9.) / 23.).unwrap())
                    .collect();
                model.set_weights(&weights).unwrap();
                model.set_activation_function_hidden(activation);
                model.set_activation_function_output(activation);
                if sparse {
                    for layer in model.layers.iter_mut().skip(1) {
                        for neuron in layer.neurons.iter_mut().filter(|n| !n.is_bias) {
                            neuron.connections.reverse();
                            neuron.connections.remove(1);
                            neuron.connections.push(neuron.connections[0].clone());
                        }
                    }
                }
                let prepared = model.prepare().unwrap();
                let mut single = prepared.workspace().unwrap();
                let mut batch = prepared.batch_workspace(17, 4 * 1024 * 1024).unwrap();
                for rows in [0, 1, 2, 3, 4, 5, 8, 15, 17] {
                    let inputs: Vec<_> = (0..rows * 3)
                        .map(|i| T::from(((i * 5 % 19) as f64 - 9.) / 13.).unwrap())
                        .collect();
                    let mut output = vec![T::zero(); rows * 2];
                    prepared
                        .run_batch_into(&inputs, &mut output, rows, &mut batch)
                        .unwrap();
                    let mut expected = vec![T::zero(); 2];
                    for row in 0..rows {
                        prepared
                            .run_into(&inputs[row * 3..row * 3 + 3], &mut expected, &mut single)
                            .unwrap();
                        for column in 0..2 {
                            assert_eq!(output[row * 2 + column].to_f64().unwrap().to_bits(),
                                expected[column].to_f64().unwrap().to_bits(),
                                "SIMD/ordered-scalar mismatch: {activation:?}, sparse={sparse}, rows={rows}");
                        }
                    }
                    checked += rows;
                }
                let mut input = vec![T::zero(); 9];
                input[8] = T::nan();
                let sentinel = T::from(12345.).unwrap();
                let mut output = vec![sentinel; 6];
                assert!(prepared
                    .run_batch_into(&input, &mut output, 3, &mut batch)
                    .is_err());
                assert!(output.iter().all(|value| *value == sentinel));
            }
        }
        checked
    }

    // The browser must execute single-inference tiles with an odd >=64 input
    // width and a neuron tail. The small batching fixtures cannot reach them.
    fn check_wide_single<T: BatchFloat + std::fmt::Debug>() -> (usize, usize) {
        let mut comparisons = 0;
        for topology in 0..3 {
            let shape = [65, 65, 5];
            let mut model = Network::<T>::new(&shape);
            let weights = (0..model.total_connections())
                .map(|i| T::from(((i * 17 % 31) as f64 - 15.) / 128.).unwrap())
                .collect::<Vec<_>>();
            model.set_weights(&weights).unwrap();
            if topology == 2 {
                for layer in model.layers.iter_mut().take(shape.len() - 1) {
                    assert!(layer.neurons.last().unwrap().is_bias);
                    layer.neurons.pop();
                }
                for index in 1..model.layers.len() {
                    let width = model.layers[index - 1].size();
                    for neuron in &mut model.layers[index].neurons {
                        neuron.connections.retain(|edge| edge.from_neuron < width);
                    }
                }
            }
            for layer in model.layers.iter_mut().skip(1) {
                for (index, neuron) in layer.neurons.iter_mut().filter(|n| !n.is_bias).enumerate() {
                    neuron.activation_function = ACTIVATIONS[index % ACTIVATIONS.len()];
                    neuron.activation_steepness =
                        T::from(if index % 2 == 0 { 0.75 } else { -1.25 }).unwrap();
                    if topology == 1 {
                        match index {
                            0 => neuron.connections.clear(),
                            1 => neuron.connections.reverse(),
                            2 => neuron.connections.push(neuron.connections[0].clone()),
                            8 => {
                                neuron.connections.remove(0);
                            }
                            _ => (),
                        }
                    }
                }
            }
            let prepared = model.prepare().unwrap();
            let mut workspace = prepared.workspace().unwrap();
            let mut actual = vec![T::nan(); 5];
            for sample in 0..7 {
                let input = (0..65)
                    .map(|i| T::from((((i + sample) * 7 % 17) as f64 - 8.) / 8.).unwrap())
                    .collect::<Vec<_>>();
                // This oracle runs the mutable scalar interpreter, independently
                // of both the prepared single and prepared batching kernels.
                let expected = model.run(&input).unwrap();
                prepared
                    .run_into(&input, &mut actual, &mut workspace)
                    .unwrap();
                for (&a, &b) in actual.iter().zip(&expected) {
                    assert_eq!(
                        a.to_f64().unwrap().to_bits(),
                        b.to_f64().unwrap().to_bits(),
                        "wide65 single parity: topology={topology}, sample={sample}"
                    );
                    comparisons += 1;
                }
            }
        }
        let mut overflow_cases = 0;
        for mixed in [false, true] {
            let mut model = Network::<T>::new(&[65, 65]);
            model
                .set_weights(&vec![T::zero(); model.total_connections()])
                .unwrap();
            model.set_activation_function_output(A::Linear);
            model.set_activation_steepness_output(T::one());
            model.layers[1].neurons[0].connections[0].weight = T::one();
            model.layers[1].neurons[0].activation_steepness = T::max_value();
            model.layers[1].neurons[1].connections[1].weight = T::max_value();
            if mixed {
                // Disable direct-first dispatch while leaving the first tile
                // canonical; exercise the general layer's wide tile as well.
                model.layers[1].neurons[64].connections.reverse();
            }
            let prepared = model.prepare().unwrap();
            let mut workspace = prepared.workspace().unwrap();
            let mut input = vec![T::zero(); 65];
            input[0] = T::from(2).unwrap();
            input[1] = T::from(2).unwrap();
            let sentinel = T::from(12345).unwrap();
            let mut output = vec![sentinel; 65];
            assert_eq!(
                prepared.run_into(&input, &mut output, &mut workspace),
                Err(PreparedError::NumericOverflow {
                    layer: 1,
                    neuron: 0
                })
            );
            assert!(
                output.iter().all(|&value| value == sentinel),
                "wide overflow mutated output"
            );
            input.fill(T::zero());
            prepared
                .run_into(&input, &mut output, &mut workspace)
                .unwrap();
            assert!(
                output.iter().all(|&value| value == T::zero()),
                "wide overflow recovery failed"
            );
            overflow_cases += 1;
        }
        (comparisons, overflow_cases)
    }

    #[wasm_bindgen]
    pub fn verify_prepared_batches() -> String {
        let f32_rows = check::<f32>();
        let f64_rows = check::<f64>();
        let (wide_f32, overflow_f32) = check_wide_single::<f32>();
        let (wide_f64, overflow_f64) = check_wide_single::<f64>();
        serde_json::json!({"passed":true,"simd128":cfg!(target_feature="simd128"),
            "activations":21,"dense_and_sparse":true,"float32_rows":f32_rows,
            "float64_rows":f64_rows,"bitwise_scalar_equivalence":true,
            "late_error_atomicity":true,"wide_single_input_width":65,
            "wide_single_float32_comparisons":wide_f32,"wide_single_float64_comparisons":wide_f64,
            "wide_single_topologies":["canonical","mixed","biasless"],
            "wide_single_float32_overflow_cases":overflow_f32,
            "wide_single_float64_overflow_cases":overflow_f64})
        .to_string()
    }
}

fn main() {}
