use crate::{ActivationFunction as A, Network, NetworkBuilder, NetworkError, Neuron};

#[test]
fn caller_owned_inference_matches_copied_layers_for_sparse_connections() {
    let mut network = NetworkBuilder::<f64>::new()
        .input_layer(3)
        .hidden_layer(4)
        .output_layer(2)
        .connection_rate(0.45)
        .build();
    for (index, layer) in network.layers.iter_mut().enumerate().skip(1) {
        for neuron in &mut layer.neurons {
            neuron.connections.reverse();
            for connection in &mut neuron.connections {
                connection.weight = (connection.from_neuron + index) as f64 * 0.125;
            }
        }
    }
    let mut reference = network.clone();
    let input = [0.25, -0.75, 0.5];
    reference.layers[0].set_inputs(&input).unwrap();
    for index in 1..reference.layers.len() {
        let previous = reference.layers[index - 1].get_outputs();
        reference.layers[index].calculate(&previous);
    }
    let expected = reference.layers.last().unwrap().get_outputs();
    let mut actual = [0.0; 2];
    network.run_into(&input, &mut actual).unwrap();
    assert_eq!(expected, actual);
    assert_eq!(network.run(&input).unwrap(), actual);
}

#[test]
fn caller_owned_inference_rejects_bad_sizes_without_changing_state() {
    let mut network = Network::<f64>::new(&[2, 3, 1]);
    let before: Vec<_> = network
        .layers
        .iter()
        .map(|layer| layer.get_outputs())
        .collect();
    let mut output = [7.0];
    assert!(matches!(
        network.run_into(&[1.0], &mut output),
        Err(NetworkError::InputSizeMismatch { .. })
    ));
    assert_eq!(output, [7.0]);
    assert!(matches!(
        network.run_into(&[1.0, 2.0], &mut []),
        Err(NetworkError::DataSizeMismatch { .. })
    ));
    let after: Vec<_> = network
        .layers
        .iter()
        .map(|layer| layer.get_outputs())
        .collect();
    assert_eq!(before, after);
}

#[test]
fn empty_and_single_layer_builders_do_not_underflow() {
    let mut empty = NetworkBuilder::<f32>::new().build();
    assert!(matches!(empty.run(&[]), Err(NetworkError::NoLayers)));
    let mut single = Network::<f32>::new(&[2]);
    assert_eq!(single.run(&[0.25, -0.5]).unwrap(), [0.25, -0.5]);
}

#[test]
fn all_smooth_scalar_activation_derivatives_match_finite_differences() {
    let functions = [
        A::Linear,
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
        A::LeakyRelu,
        A::Sin,
        A::Cos,
        A::SinSymmetric,
        A::CosSymmetric,
        A::Gelu,
        A::Swish,
    ];
    for activation in functions {
        for x in [-1.7, -0.3, 0.4, 1.8] {
            let mut neuron = Neuron::<f64>::new(activation, 0.8);
            neuron.add_connection(0, 1.0);
            neuron.calculate(&[x]);
            let analytic = neuron.activation_derivative();
            neuron.calculate(&[x + 1e-6]);
            let upper = neuron.value;
            neuron.calculate(&[x - 1e-6]);
            let finite_difference = (upper - neuron.value) / 2e-6;
            assert!(
                (analytic - finite_difference).abs() < 1e-7,
                "{activation:?} at {x}: analytic={analytic}, finite={finite_difference}"
            );
        }
    }
}

#[test]
fn modern_activations_are_not_identity_and_aliases_agree() {
    for (activation, expected) in [(A::Gelu, 0.841192), (A::Swish, 0.731059)] {
        let mut neuron = Neuron::<f64>::new(activation, 1.0);
        neuron.add_connection(0, 1.0);
        neuron.calculate(&[1.0]);
        assert!((neuron.value - expected).abs() < 1e-6);
    }
    let mut leaky = Neuron::<f64>::new(A::LeakyRelu, 1.0);
    leaky.add_connection(0, 1.0);
    leaky.calculate(&[-2.0]);
    assert_eq!(leaky.value, -0.02);
    for (activation, expected) in [(A::Threshold, 0.0), (A::ThresholdSymmetric, -1.0)] {
        let mut neuron = Neuron::<f64>::new(activation, 1.0);
        neuron.add_connection(0, 1.0);
        neuron.calculate(&[-0.5]);
        assert_eq!(neuron.value, expected);
        assert_eq!(neuron.activation_derivative(), 0.0);
    }
}

#[test]
fn gelu_derivative_stays_finite_at_saturated_finite_inputs() {
    let mut neuron = Neuron::<f64>::new(A::Gelu, 1.0);
    neuron.add_connection(0, 1.0);
    neuron.calculate(&[1e150]);
    assert_eq!(neuron.activation_derivative(), 1.0);
    neuron.calculate(&[-1e150]);
    assert_eq!(neuron.activation_derivative(), 0.0);
}

#[cfg(all(feature = "binary", feature = "serde"))]
#[test]
fn binary_validation_uses_previous_layer_not_current_layer_plus_headroom() {
    let mut invalid = Network::<f32>::new(&[1, 8]);
    invalid.layers[1].neurons[0].connections[0].from_neuron = 4;
    assert!(Network::<f32>::from_bytes(&invalid.to_bytes().unwrap()).is_err());
    let valid = Network::<f32>::new(&[1100, 1]);
    assert!(Network::<f32>::from_bytes(&valid.to_bytes().unwrap()).is_ok());
}

#[cfg(all(feature = "binary", feature = "serde"))]
#[test]
fn binary_validation_rejects_nonfinite_weights_and_misplaced_bias() {
    let mut network = Network::<f32>::new(&[1, 1]);
    network.layers[1].neurons[0].connections[0].weight = f32::NAN;
    assert!(Network::<f32>::from_bytes(&network.to_bytes().unwrap()).is_err());
    let mut network = Network::<f32>::new(&[1, 1]);
    network.layers[0].neurons.swap(0, 1);
    assert!(Network::<f32>::from_bytes(&network.to_bytes().unwrap()).is_err());
}

#[cfg(any(feature = "parallel", feature = "simd"))]
mod simd {
    use crate::simd::{CpuSimdOps, ParallelTraining, SimdConfig, SimdMatrixOps};

    fn ops(vectorized: bool) -> CpuSimdOps {
        CpuSimdOps::new(SimdConfig {
            use_avx2: vectorized,
            block_size: 9,
            ..Default::default()
        })
    }

    #[test]
    fn malformed_dimensions_panic_before_unsafe_memory_access() {
        for vectorized in [false, true] {
            let ops = ops(vectorized);
            assert!(std::panic::catch_unwind(|| ops.matmul(
                &[1.0; 8],
                &[1.0; 8],
                &mut [0.0; 7],
                1,
                8,
                1
            ))
            .is_err());
            assert!(std::panic::catch_unwind(|| ops.matvec(
                &[1.0; 7],
                &[1.0; 8],
                &mut [0.0],
                1,
                8
            ))
            .is_err());
            assert!(
                std::panic::catch_unwind(|| ops.add_bias(&mut [0.0; 8], &[1.0; 7], 1, 8)).is_err()
            );
            assert!(std::panic::catch_unwind(|| ops.activation_derivatives(
                &[1.0; 8],
                &mut [0.0; 7],
                crate::ActivationFunction::ReLU
            ))
            .is_err());
            assert!(
                std::panic::catch_unwind(|| ops.matmul(&[], &[], &mut [], usize::MAX, 2, 2))
                    .is_err()
            );
        }
    }

    #[test]
    fn irregular_simd_shapes_match_scalar_including_remainders() {
        for (m, n, k) in [(3, 17, 11), (1, 8, 1), (4, 9, 0)] {
            let a: Vec<_> = (0..m * k).map(|i| (i % 7) as f32 * 0.25).collect();
            let b: Vec<_> = (0..k * n).map(|i| (i % 11) as f32 * 0.5).collect();
            let mut expected = vec![0.0; m * n];
            let mut actual = vec![0.0; m * n];
            ops(false).matmul(&a, &b, &mut expected, m, n, k);
            ops(true).matmul(&a, &b, &mut actual, m, n, k);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn vector_remainders_and_bias_match_scalar() {
        let matrix: Vec<_> = (0..51).map(|i| (i % 9) as f32 * 0.25).collect();
        let input: Vec<_> = (0..17).map(|i| i as f32 * 0.5).collect();
        let mut expected = vec![0.0; 3];
        let mut actual = vec![0.0; 3];
        ops(false).matvec(&matrix, &input, &mut expected, 3, 17);
        ops(true).matvec(&matrix, &input, &mut actual, 3, 17);
        assert_eq!(expected, actual);
        let mut expected = matrix.clone();
        let mut actual = matrix;
        ops(false).add_bias(&mut expected, &input, 3, 17);
        ops(true).add_bias(&mut actual, &input, 3, 17);
        assert_eq!(expected, actual);
    }

    #[test]
    fn simd_gelu_derivatives_share_saturated_scalar_behavior() {
        for vectorized in [false, true] {
            let mut derivatives = [f32::NAN; 2];
            ops(vectorized).activation_derivatives(
                &[1e20, -1e20],
                &mut derivatives,
                crate::ActivationFunction::Gelu,
            );
            assert_eq!(derivatives, [1.0, 0.0]);
        }
    }

    #[test]
    fn gradient_computation_is_an_outer_product() {
        let trainer = ParallelTraining::new();
        let mut gradients = [vec![0.0; 6]];
        trainer.compute_gradients_parallel(
            &[vec![0.0; 6]],
            &[vec![1.0, 2.0, 3.0]],
            &[vec![2.0, 4.0]],
            &mut gradients,
        );
        assert_eq!(gradients[0], [2.0, 4.0, 6.0, 4.0, 8.0, 12.0]);
    }
}

#[cfg(feature = "io")]
mod io {
    use crate::io::{FannReader, TrainingDataReader, TrainingDataStreamReader};
    use std::io::Cursor;

    #[test]
    fn fann_rejects_decoded_dimensions_before_allocation() {
        for sizes in [
            format!("{} 1", usize::MAX),
            "10000000 10000000".into(),
            "0 1".into(),
        ] {
            let text = format!("FANN_FLO:2.1\nnum_layers=2\nlayer_sizes={sizes}\n");
            assert!(FannReader::new()
                .read_network::<f32, _>(&mut text.as_bytes())
                .is_err());
        }
    }

    #[test]
    fn training_headers_and_truncated_rows_are_rejected_without_preallocation() {
        for text in [
            format!("{} 1 1\n", usize::MAX),
            "1 0 0\n".into(),
            "1 1 1\n1\n".into(),
            "1 1 1\nNaN\n0\n".into(),
        ] {
            assert!(TrainingDataReader::new()
                .read_data(&mut text.as_bytes())
                .is_err());
            let mut called = false;
            assert!(TrainingDataStreamReader::new()
                .read_stream(&mut Cursor::new(text.as_bytes()), |_, _| {
                    called = true;
                    Ok(())
                })
                .is_err());
            assert!(!called);
        }
    }

    #[test]
    fn unterminated_text_line_is_bounded_separately_from_the_whole_file() {
        let text = format!("FANN_FLO:2.1\nignored={}", "x".repeat(17 * 1024 * 1024));
        assert!(FannReader::new()
            .read_network_with_limit::<f32, _>(&mut text.as_bytes(), 64 * 1024 * 1024)
            .is_err());
    }

    #[test]
    fn text_file_limit_applies_to_ignored_fields() {
        let text = b"FANN_FLO:2.1\nignored=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nnum_layers=2\nlayer_sizes=1 1\n";
        assert!(FannReader::new()
            .read_network_with_limit::<f32, _>(&mut text.as_slice(), 30)
            .is_err());
    }

    #[cfg(feature = "compression")]
    #[test]
    fn streaming_gzip_does_not_write_beyond_output_limit() {
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;
        let mut gzip = GzEncoder::new(Vec::new(), Compression::default());
        gzip.write_all(&[0u8; 8192]).unwrap();
        let bytes = gzip.finish().unwrap();
        let mut output = Vec::new();
        assert!(
            crate::io::decompress_data_with_limit(&mut bytes.as_slice(), &mut output, 64).is_err()
        );
        assert_eq!(output.len(), 64);
        let mut output = Vec::new();
        crate::io::decompress_data_with_limit(&mut bytes.as_slice(), &mut output, 8192).unwrap();
        assert_eq!(output.len(), 8192);
    }
}

#[test]
fn built_in_losses_never_score_truncated_or_non_finite_samples_as_zero_error() {
    use crate::training::{ErrorFunction, MaeError, MseError, TanhError};
    let losses: [&dyn ErrorFunction<f64>; 3] = [&MseError, &MaeError, &TanhError];
    for loss in losses {
        for (actual, desired) in [
            (&[1.0][..], &[][..]),
            (&[1.0][..], &[1.0, 100.0][..]),
            (&[][..], &[][..]),
            (&[f64::INFINITY][..], &[1.0][..]),
            (&[1.0][..], &[f64::NAN][..]),
        ] {
            assert!(loss.calculate(actual, desired).is_nan());
        }
        assert_eq!(loss.calculate(&[0.5, 1.5], &[0.5, 1.5]), 0.0);
    }
    assert_eq!(MseError.calculate(&[1.0, 3.0], &[0.0, 1.0]), 2.5);
    assert_eq!(MaeError.calculate(&[1.0, 3.0], &[0.0, 1.0]), 1.5);
    assert!(MaeError.derivative(f64::NAN, 1.0).is_nan());
}

#[test]
fn vector_backends_use_every_canonical_activation() {
    use crate::webgpu::{
        backend::{CpuBackend, SimdBackend},
        ComputeBackend,
    };
    let backends: [&dyn ComputeBackend<f64>; 2] = [&CpuBackend::new(), &SimdBackend::new()];
    let functions = [
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
        A::LeakyRelu,
        A::Sin,
        A::Cos,
        A::SinSymmetric,
        A::CosSymmetric,
        A::Gelu,
        A::Swish,
    ];
    let input = [-1000.0, -2.0, 0.0, 0.5, 1000.0];
    for backend in backends {
        for function in functions {
            let actual = backend
                .apply_activation_function(&input, function, 0.7)
                .unwrap();
            let expected: Vec<_> = input
                .iter()
                .map(|&value| function.apply(value, 0.7))
                .collect();
            assert_eq!(actual, expected, "{function:?}");
        }
    }
}

#[test]
fn device_less_memory_helpers_never_claim_upload_or_download_success() {
    use crate::webgpu::{
        backend::{CpuBackend, SimdBackend},
        BufferHandle, ComputeError, MemoryManager,
    };
    let managers: [&dyn MemoryManager<f64>; 2] = [&CpuBackend::new(), &SimdBackend::new()];
    let handle = BufferHandle::new(42);
    for manager in managers {
        assert!(matches!(
            manager.allocate_buffer(8),
            Err(ComputeError::UnsupportedOperation(_))
        ));
        assert!(manager.upload_data(handle, &[1.0]).is_err());
        assert!(manager.download_data(handle).is_err());
        assert!(manager.deallocate_buffer(handle).is_err());
        assert_eq!(manager.memory_usage().available, 0);
    }
    let manager = crate::webgpu::compute_context::GpuMemoryManager::<f64>::new();
    assert!(manager.allocate_buffer(8).is_err());
    assert!(manager.upload_data(handle, &[1.0]).is_err());
    assert!(manager.download_data(handle).is_err());
    assert!(manager.deallocate_buffer(handle).is_err());
}
