//! Per-thread allocation deltas for actual model operations.
//!
//! Counters are monotonic and never reset. The dedicated measuring thread owns
//! all measured inputs, outputs and models; unrelated test/Polars worker activity
//! cannot enter the counters. This measures requested heap bytes, not process RSS.
#[path = "../support/mod.rs"]
mod support;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Default, Debug)]
struct Counts {
    allocated: u64,
    freed: u64,
    calls: u64,
}
thread_local! { static COUNTS: Cell<Counts> = const { Cell::new(Counts { allocated:0, freed:0, calls:0 }) }; }
struct LocalAllocator;
fn record(allocated: usize, freed: usize) {
    let _ = COUNTS.try_with(|cell| {
        let mut counts = cell.get();
        counts.allocated = counts.allocated.saturating_add(allocated as u64);
        counts.freed = counts.freed.saturating_add(freed as u64);
        counts.calls = counts.calls.saturating_add(u64::from(allocated != 0));
        cell.set(counts);
    });
}
// SAFETY: Every pointer/layout is forwarded unchanged to System. The tracker
// uses only nonallocating thread-local Cells and cannot borrow or free payloads.
unsafe impl GlobalAlloc for LocalAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(layout.size(), 0);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(layout.size(), 0);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(0, layout.size());
        unsafe {
            System.dealloc(ptr, layout);
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(ptr, layout, size) };
        if !next.is_null() {
            record(size, layout.size());
        }
        next
    }
}
#[global_allocator]
static ALLOCATOR: LocalAllocator = LocalAllocator;
fn snapshot() -> Counts {
    COUNTS.with(Cell::get)
}
fn net_change(a: Counts, b: Counts) -> i128 {
    (b.allocated - a.allocated) as i128 - (b.freed - a.freed) as i128
}

#[cfg(feature = "models")]
#[test]
fn model_allocations_are_bounded_and_released_after_inference_and_drop() {
    use neuro_divergent::models::{BaseModel, TimeSeriesInput};
    std::thread::spawn(|| {
        // Warm thread-local RNG/model initialization before taking deltas.
        drop(support::rnn(3, 8, 2));
        let before = snapshot();
        for _ in 0..32 {
            drop(support::rnn(3, 8, 2));
        }
        let after = snapshot();
        assert_eq!(
            net_change(before, after),
            0,
            "model creation/drop leaked requested heap bytes"
        );
        assert!(
            after.allocated - before.allocated < 8 * 1024 * 1024,
            "small model construction allocated over 8MiB total"
        );

        let dataset = support::model_dataset(16, 12, 3);
        let mut model = support::rnn(3, 8, 10);
        model.fit(&dataset).unwrap();
        let input = TimeSeriesInput::new(vec![0.1; 12]);
        drop(model.predict(&input).unwrap());
        let before = snapshot();
        for _ in 0..256 {
            let prediction = model.predict(&input).unwrap();
            assert_eq!(prediction.forecasts.len(), 3);
            assert!(prediction.forecasts.iter().all(|x| x.is_finite()));
            drop(prediction);
        }
        let after = snapshot();
        assert_eq!(
            net_change(before, after),
            0,
            "repeated inference retained heap allocations"
        );
        assert!(
            after.allocated - before.allocated < 256 * 256 * 1024,
            "inference exceeded 256KiB requested bytes/call"
        );
        assert!(
            after.calls > before.calls,
            "allocation tracker did not observe model work"
        );

        // Repeated training replaces a fixed-size history/optimizer state.
        // Warm fitting above ensures persistent state already has its final shape.
        let before = snapshot();
        for _ in 0..8 {
            model.fit(&dataset).unwrap();
        }
        assert_eq!(
            net_change(before, snapshot()),
            0,
            "refitting the same small dataset retained additional heap allocations"
        );

        let before = snapshot();
        let models: Vec<_> = (0..16).map(|_| support::rnn(4, 16, 2)).collect();
        let resident = snapshot();
        assert!(net_change(before, resident) > 0);
        assert!(
            net_change(before, resident) < 4 * 1024 * 1024,
            "16 small models retained over 4MiB"
        );
        drop(models);
        assert_eq!(
            net_change(before, snapshot()),
            0,
            "dropping an ensemble retained heap allocations"
        );
    })
    .join()
    .unwrap();
}

#[test]
fn allocator_accounts_for_reallocation_without_global_counter_resets() {
    std::thread::spawn(|| {
        let start = snapshot();
        let mut buffer = Vec::<u8>::with_capacity(128);
        buffer.resize(128, 1);
        buffer.reserve_exact(4096);
        assert!(net_change(start, snapshot()) >= 4224);
        drop(buffer);
        assert_eq!(net_change(start, snapshot()), 0);
    })
    .join()
    .unwrap();
}
