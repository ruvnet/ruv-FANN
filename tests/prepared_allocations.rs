//! Allocation checks are isolated in this test binary. Unsafe code only forwards
//! the standard allocator interface; the prepared implementation is safe Rust.
use num_traits::Float;
use ruv_fann::{Network, PreparationLimits, PreparedNetwork};
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

fn repeated<T: Float>() {
    let network = Network::<T>::new(&[5, 9, 7, 3]);
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.workspace().unwrap();
    let mut output = [T::zero(); 3];
    let input = [T::from(0.25).unwrap(); 5];
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    for _ in 0..128 {
        prepared
            .run_into(&input, &mut output, &mut workspace)
            .unwrap();
    }
    // Error results also use no dynamically allocated messages.
    assert!(prepared.run_into(&[], &mut output, &mut workspace).is_err());
    TRACK.with(|track| track.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn zero_steady_state_allocations_f32() {
    repeated::<f32>();
}
#[test]
fn zero_steady_state_allocations_f64() {
    repeated::<f64>();
}

#[test]
fn failed_budget_preflight_allocates_nothing() {
    let network = Network::<f32>::new(&[5, 9, 3]);
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    let result = PreparedNetwork::from_network_with_limits(
        &network,
        PreparationLimits {
            max_bytes: 0,
            ..PreparationLimits::default()
        },
    );
    TRACK.with(|track| track.set(false));
    assert!(result.is_err());
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}
