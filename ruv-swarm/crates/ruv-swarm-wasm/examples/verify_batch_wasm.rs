//! Actual WASM SIMD equivalence probe for both supported native element types.
//! Compile this crate's example with --target wasm32-unknown-unknown and +simd128.
#[cfg(target_arch = "wasm32")]
mod probe {
    use ruv_fann::{ActivationFunction as A, BatchFloat, Network};
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

    #[wasm_bindgen]
    pub fn verify_prepared_batches() -> String {
        let f32_rows = check::<f32>();
        let f64_rows = check::<f64>();
        serde_json::json!({"passed":true,"simd128":cfg!(target_feature="simd128"),
            "activations":21,"dense_and_sparse":true,"float32_rows":f32_rows,
            "float64_rows":f64_rows,"bitwise_scalar_equivalence":true,
            "late_error_atomicity":true})
        .to_string()
    }
}

fn main() {}
