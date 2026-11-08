//! Regressions for the inference API's allocator behavior.

use ruv_fann::Network;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout)
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        System.realloc(pointer, layout, size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn repeated_inference_into_reuses_the_callers_buffer() {
    let mut network = Network::<f32>::new(&[4, 16, 8, 4, 2]);
    let mut output = [0.0; 2];
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    for _ in 0..64 {
        network.run_into(&[0.25; 4], &mut output).unwrap();
    }
    TRACK.with(|track| track.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn returning_inference_allocates_only_its_final_output() {
    let mut network = Network::<f32>::new(&[4, 16, 8, 4, 2]);
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    let output = network.run(&[0.25; 4]).unwrap();
    TRACK.with(|track| track.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 1);
    assert_eq!(output.len(), 2);
}
