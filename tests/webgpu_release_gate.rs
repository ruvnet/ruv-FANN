//! Real compute checks. Requires a native or software Vulkan/WebGPU adapter.
#![cfg(feature = "gpu")]

use ruv_fann::webgpu::{BufferHandle, ComputeBackend, MemoryManager, WebGPUBackend};
use ruv_fann::{ActivationFunction as A, Neuron};

fn backend<T: num_traits::Float + std::fmt::Debug + Send + Sync + 'static>() -> WebGPUBackend<T> {
    WebGPUBackend::new().expect("A compute adapter is required; headless CI may opt in with RUV_FANN_ALLOW_SOFTWARE_GPU=1 and WGPU_BACKEND=vulkan")
}

#[test]
fn real_dispatch_covers_every_batch_and_row() {
    let backend = backend::<f32>();
    let (rows, cols, batches) = (129, 79, 101);
    let matrix: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 13) as f32 * 0.125 - 0.75)
        .collect();
    let vectors: Vec<Vec<f32>> = (0..batches)
        .map(|b| {
            (0..cols)
                .map(|c| ((c + b) % 11) as f32 * 0.25 - 1.0)
                .collect()
        })
        .collect();
    let output = backend
        .batch_matrix_vector_multiply(&matrix, &vectors, rows, cols)
        .unwrap();
    assert_eq!(output.len(), batches);
    for (vector, actual) in vectors.iter().zip(&output) {
        assert_eq!(actual.len(), rows);
        for (r, value) in actual.iter().enumerate() {
            let expected: f32 = matrix[r * cols..(r + 1) * cols]
                .iter()
                .zip(vector)
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (value - expected).abs() < 1e-4,
                "row {r}: {value} vs {expected}"
            );
        }
    }
    let single = backend
        .matrix_vector_multiply(&matrix, &vectors[0], rows, cols)
        .unwrap();
    assert_eq!(single, output[0]);
}

#[test]
fn real_activation_dispatch_matches_all_scalar_variants() {
    let backend = backend::<f32>();
    let inputs: Vec<f32> = (0..1027).map(|i| ((i % 81) as f32 - 40.0) * 0.5).collect();
    let activations = [
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
    for activation in activations {
        let actual = backend
            .apply_activation_function(&inputs, activation, 0.7)
            .unwrap();
        let mut neuron = Neuron::<f32>::new(activation, 0.7);
        neuron.add_connection(0, 1.0);
        for (&input, &value) in inputs.iter().zip(&actual) {
            neuron.calculate(&[input]);
            assert!(
                (value - neuron.value).abs() <= 3e-5 * (1.0 + neuron.value.abs()),
                "{activation:?}({input}): GPU {value} vs scalar {}",
                neuron.value
            );
        }
    }
}

#[test]
fn wide_precision_fallback_and_dimension_errors_are_honest() {
    let backend = backend::<f64>();
    let matrix = vec![1e100; 101 * 101];
    let result = backend
        .matrix_vector_multiply(&matrix, &vec![1.0; 101], 101, 101)
        .unwrap();
    assert!(result
        .iter()
        .all(|v| v.is_finite() && (*v / 1.01e102 - 1.0).abs() < 1e-12));
    assert!(backend
        .matrix_vector_multiply(&[], &[], usize::MAX, 2)
        .is_err());
    assert!(backend
        .batch_matrix_vector_multiply(&[], &[vec![]], usize::MAX, 2)
        .is_err());
    assert!(backend.matrix_vector_multiply(&[], &[], 1, 0).is_err());
}

#[test]
fn managed_device_buffers_round_trip_real_bytes_preserve_tails_and_track_usage() {
    let backend = backend::<f32>();
    let memory = backend.memory_manager();
    let initial = memory.memory_usage();
    let first = memory.allocate_buffer(24).unwrap();
    let second = memory.allocate_buffer(24).unwrap();
    assert_ne!(first, second);
    assert_eq!(memory.memory_usage().total_allocated, 48);
    assert_eq!(memory.memory_usage().buffer_count, 2);
    assert_eq!(memory.memory_usage().available, initial.available - 48);
    assert_eq!(memory.download_data(first).unwrap(), vec![0.0; 6]);
    memory.upload_data(first, &[1.25, -2.5, 3.0, 4.0]).unwrap();
    memory.upload_data(second, &[9.0; 6]).unwrap();
    assert_eq!(
        memory.download_data(first).unwrap(),
        vec![1.25, -2.5, 3.0, 4.0, 0.0, 0.0]
    );
    memory.upload_data(first, &[-7.0]).unwrap();
    assert_eq!(
        memory.download_data(first).unwrap(),
        vec![-7.0, -2.5, 3.0, 4.0, 0.0, 0.0]
    );
    assert_eq!(memory.download_data(second).unwrap(), vec![9.0; 6]);
    assert!(memory.upload_data(first, &[0.0; 7]).is_err());
    assert_eq!(
        memory.download_data(first).unwrap(),
        vec![-7.0, -2.5, 3.0, 4.0, 0.0, 0.0]
    );
    memory.deallocate_buffer(first).unwrap();
    assert!(memory.download_data(first).is_err());
    assert!(memory.upload_data(first, &[1.0]).is_err());
    assert!(memory.deallocate_buffer(first).is_err());
    let replacement = memory.allocate_buffer(24).unwrap();
    assert_ne!(replacement, first);
    assert_ne!(replacement, second);
    memory.deallocate_buffer(replacement).unwrap();
    memory.deallocate_buffer(second).unwrap();
    assert_eq!(memory.memory_usage().total_allocated, 0);
    assert_eq!(memory.memory_usage().buffer_count, 0);
    assert_eq!(memory.memory_usage().available, initial.available);
}

#[test]
fn managed_f64_storage_preserves_ieee_bits_without_f64_shader_support() {
    let backend = backend::<f64>();
    let memory = backend.memory_manager();
    let values = [
        1e200,
        -0.0,
        f64::from_bits(1),
        f64::from_bits(0x7ff8_1234_5678_9abc),
    ];
    let handle = memory.allocate_buffer(values.len() * 8).unwrap();
    memory.upload_data(handle, &values).unwrap();
    let result = memory.download_data(handle).unwrap();
    assert_eq!(
        result
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert!(memory.allocate_buffer(4).is_err());
    memory.deallocate_buffer(handle).unwrap();
}

#[test]
fn managed_handles_reject_invalid_sizes_and_foreign_devices() {
    let first = backend::<f32>();
    let second = backend::<f32>();
    let a = first.memory_manager();
    let b = second.memory_manager();
    assert!(a.allocate_buffer(0).is_err());
    assert!(a.allocate_buffer(5).is_err());
    assert!(a.allocate_buffer(usize::MAX).is_err());
    assert!(a
        .allocate_buffer(first.capabilities().max_buffer_size + 4)
        .is_err());
    let first_handle = a.allocate_buffer(16).unwrap();
    let second_handle = b.allocate_buffer(16).unwrap();
    assert_ne!(first_handle, second_handle);
    let remaining_budget = a.memory_usage().available;
    let budget_handle = a.allocate_buffer(remaining_budget).unwrap();
    assert_eq!(a.memory_usage().available, 0);
    assert!(a.allocate_buffer(4).is_err());
    a.deallocate_buffer(budget_handle).unwrap();
    a.upload_data(first_handle, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(b.upload_data(first_handle, &[8.0]).is_err());
    assert!(b.download_data(first_handle).is_err());
    assert!(b.deallocate_buffer(first_handle).is_err());
    for invalid in [BufferHandle::new(0), BufferHandle::new(u64::MAX)] {
        assert!(a.upload_data(invalid, &[]).is_err());
        assert!(a.download_data(invalid).is_err());
        assert!(a.deallocate_buffer(invalid).is_err());
    }
    assert_eq!(
        a.download_data(first_handle).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    a.deallocate_buffer(first_handle).unwrap();
    b.deallocate_buffer(second_handle).unwrap();
}

#[test]
fn managed_memory_operations_are_safe_across_threads_with_unique_handles() {
    let backend = std::sync::Arc::new(backend::<f32>());
    let mut workers = Vec::new();
    for worker in 0..4 {
        let backend = backend.clone();
        workers.push(std::thread::spawn(move || {
            let mut handles = Vec::new();
            for iteration in 0..8 {
                let values: Vec<f32> = (0..8)
                    .map(|index| (worker * 100 + iteration * 10 + index) as f32)
                    .collect();
                let handle = backend.allocate_buffer(32).unwrap();
                backend.upload_data(handle, &values).unwrap();
                assert_eq!(backend.download_data(handle).unwrap(), values);
                backend.deallocate_buffer(handle).unwrap();
                handles.push(handle);
            }
            handles
        }));
    }
    let handles: Vec<_> = workers
        .into_iter()
        .flat_map(|worker| worker.join().unwrap())
        .collect();
    let unique: std::collections::HashSet<_> = handles.iter().copied().collect();
    assert_eq!(unique.len(), handles.len());
    assert_eq!(backend.memory_usage().total_allocated, 0);
    assert_eq!(backend.memory_usage().buffer_count, 0);
}
