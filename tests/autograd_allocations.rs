//! Unsafe code here only forwards the standard allocator interface; training
//! implementation and arithmetic are safe Rust.
use ruv_fann::{
    ActivationFunction, AdamW, AdamWConfig, AutogradNetwork, BatchFloat, DenseLayer,
    PreparationLimits,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct CountingAllocator;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
fn record() {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout)
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        System.realloc(pointer, layout, size)
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn fixture<T: BatchFloat>() -> AutogradNetwork<T> {
    let weights = [T::from(0.01).unwrap(); 20];
    let bias = [T::from(0.01).unwrap(); 4];
    AutogradNetwork::from_dense_layers(&[
        DenseLayer {
            inputs: 5,
            outputs: 4,
            weights: &weights,
            bias: Some(&bias),
            activation: ActivationFunction::Tanh,
            steepness: T::one(),
        },
        DenseLayer {
            inputs: 4,
            outputs: 3,
            weights: &weights[..12],
            bias: None,
            activation: ActivationFunction::Linear,
            steepness: T::one(),
        },
    ])
    .unwrap()
}
fn repeated<T: BatchFloat>() {
    let mut network = fixture::<T>();
    let mut workspace = network.workspace(17, usize::MAX).unwrap();
    let inputs = [T::from(0.25).unwrap(); 85];
    let targets = [T::from(0.3).unwrap(); 51];
    let mut outputs = [T::zero(); 51];
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    for rows in [17, 1, 3, 7, 16, 5].into_iter().cycle().take(128) {
        network
            .run_batch_into(
                &inputs[..rows * 5],
                &mut outputs[..rows * 3],
                rows,
                &mut workspace,
            )
            .unwrap();
        network
            .backward_mse(
                &inputs[..rows * 5],
                &targets[..rows * 3],
                rows,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(
            workspace.parameter_gradients().len(),
            network.parameter_count()
        );
        assert_eq!(workspace.input_gradients().len(), rows * 5);
        network
            .sgd_step(&mut workspace, T::from(0.001).unwrap())
            .unwrap();
    }
    assert!(network.backward_mse(&[], &[], 1, &mut workspace).is_err());
    assert!(network.sgd_step(&mut workspace, T::one()).is_err());
    TRACK.with(|track| track.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    let mut adam = AdamW::new(&network, AdamWConfig::default(), usize::MAX).unwrap();
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    for rows in [17, 1, 3, 7, 16, 5].into_iter().cycle().take(128) {
        network
            .backward_mse(
                &inputs[..rows * 5],
                &targets[..rows * 3],
                rows,
                &mut workspace,
            )
            .unwrap();
        network.adamw_step(&mut workspace, &mut adam).unwrap();
    }
    TRACK.with(|track| track.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}
#[test]
fn no_hot_allocations_forward_reverse_sgd_adamw_f32() {
    repeated::<f32>();
}
#[test]
fn no_hot_allocations_forward_reverse_sgd_adamw_f64() {
    repeated::<f64>();
}

#[test]
fn failed_preflight_allocates_nothing() {
    let network = fixture::<f64>();
    let layer = DenseLayer {
        inputs: 1,
        outputs: 1,
        weights: &[0.0],
        bias: None,
        activation: ActivationFunction::Linear,
        steepness: 1.0,
    };
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    let model = AutogradNetwork::from_dense_layers_with_limits(
        &[layer],
        PreparationLimits {
            max_bytes: 0,
            ..PreparationLimits::default()
        },
    );
    let workspace = network.workspace(17, 0);
    let overflow = network.workspace_bytes(usize::MAX);
    let optimizer = AdamW::new(&network, AdamWConfig::default(), 0);
    TRACK.with(|track| track.set(false));
    assert!(model.is_err() && workspace.is_err() && overflow.is_err() && optimizer.is_err());
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}
