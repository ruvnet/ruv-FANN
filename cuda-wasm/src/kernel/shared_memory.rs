//! Shared memory management for CUDA kernel emulation
//!
//! Emulates CUDA shared memory (`__shared__`) on the CPU, including:
//! - Static shared memory allocation (known size at compile time)
//! - Dynamic (extern) shared memory allocation (size provided at launch)
//! - Bank conflict detection for profiling/debugging
//!
//! In CUDA, shared memory is per-block SRAM accessible by all threads in a
//! block. On the CPU we emulate this with heap-allocated buffers shared among
//! threads in the same block.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of memory banks (matches NVIDIA GPU shared memory banks).
pub const NUM_BANKS: usize = 32;

/// Size of each bank in bytes (4 bytes = 32 bits, matching CUDA).
pub const BANK_WIDTH_BYTES: usize = 4;

/// Static shared memory allocation.
///
/// Represents a fixed-size shared memory buffer that is known at compile time.
/// Analogous to `__shared__ T data[N]` in CUDA.
///
/// # Type Parameters
/// - `T`: Element type (must be `Send + Sync` since shared across threads)
pub struct SharedMemory<T: Send + Sync> {
    data: Vec<T>,
}

impl<T: Send + Sync> SharedMemory<T> {
    /// Allocate initialized shared elements using `T::default()`.
    ///
    /// # Panics
    /// Panics for zero count, layout overflow or failed allocation.
    pub fn new(count: usize) -> Self where T: Default {
        assert!(count > 0, "SharedMemory: count must be > 0");
        let mut data = Vec::new();
        data.resize_with(count, T::default);
        Self { data }
    }
    /// Number of initialized elements.
    pub fn len(&self) -> usize { self.data.len() }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    /// Borrow an element; panics when out of bounds.
    pub fn get(&self, index: usize) -> &T { &self.data[index] }
    /// Mutably borrow an element; panics when out of bounds.
    pub fn get_mut(&mut self, index: usize) -> &mut T { &mut self.data[index] }
    /// Raw pointer valid for this buffer's lifetime.
    pub fn as_ptr(&self) -> *const T { self.data.as_ptr() }
    /// Mutable raw pointer, subject to Rust's aliasing rules.
    pub fn as_mut_ptr(&mut self) -> *mut T { self.data.as_mut_ptr() }
    /// Borrow initialized shared elements.
    pub fn as_slice(&self) -> &[T] { &self.data }
    /// Mutably borrow initialized shared elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }
}

// ---------------------------------------------------------------------------
// Dynamic (extern) shared memory
// ---------------------------------------------------------------------------

/// Dynamic shared memory allocation.
///
/// Represents a shared memory buffer whose size is determined at kernel launch
/// time. Analogous to `extern __shared__ T data[]` in CUDA where the size is
/// passed as a launch parameter.
///
/// The buffer is untyped (byte-level) and callers can reinterpret as needed.
pub struct DynamicSharedMemory {
    /// Raw byte buffer
    ptr: NonNull<u8>,
    /// Size in bytes
    size_bytes: usize,
}

// Safety: Same as SharedMemory - designed for cross-thread sharing.
unsafe impl Send for DynamicSharedMemory {}
unsafe impl Sync for DynamicSharedMemory {}

impl DynamicSharedMemory {
    /// Allocate dynamic shared memory of the given size in bytes.
    ///
    /// # Panics
    /// Panics if `size_bytes` is 0 or allocation fails.
    pub fn new(size_bytes: usize) -> Self {
        assert!(size_bytes > 0, "DynamicSharedMemory: size must be > 0");

        // Align to 16 bytes for SIMD compatibility
        let layout = Layout::from_size_align(size_bytes, 16)
            .expect("DynamicSharedMemory: invalid layout");

        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("DynamicSharedMemory: allocation failed");

        Self { ptr, size_bytes }
    }

    /// Returns the size of the buffer in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Reinterpret the buffer as a typed slice of `T`.
    ///
    /// # Panics
    /// Panics if the buffer size is not a multiple of `size_of::<T>()` or if
    /// the alignment is insufficient.
    pub fn as_typed_slice<T: bytemuck::Pod>(&self) -> &[T] {
        let elem_size = std::mem::size_of::<T>();
        assert!(elem_size > 0, "DynamicSharedMemory: zero-sized type");
        assert!(
            self.size_bytes % elem_size == 0,
            "DynamicSharedMemory: size {} not a multiple of element size {}",
            self.size_bytes,
            elem_size
        );
        assert!(
            self.ptr.as_ptr() as usize % std::mem::align_of::<T>() == 0,
            "DynamicSharedMemory: alignment mismatch for type"
        );

        let count = self.size_bytes / elem_size;
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, count) }
    }

    /// Reinterpret the buffer as a mutable typed slice of `T`.
    ///
    /// # Safety
    /// Caller must ensure exclusive access and correct typing.
    pub fn as_typed_slice_mut<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        let elem_size = std::mem::size_of::<T>();
        assert!(elem_size > 0, "DynamicSharedMemory: zero-sized type");
        assert!(
            self.size_bytes % elem_size == 0,
            "DynamicSharedMemory: size {} not a multiple of element size {}",
            self.size_bytes,
            elem_size
        );
        assert!(
            self.ptr.as_ptr() as usize % std::mem::align_of::<T>() == 0,
            "DynamicSharedMemory: alignment mismatch for type"
        );

        let count = self.size_bytes / elem_size;
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count) }
    }

    /// Get the raw byte pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }

    /// Get a mutable raw byte pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for DynamicSharedMemory {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size_bytes, 16)
            .expect("DynamicSharedMemory::drop: invalid layout");
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

// ---------------------------------------------------------------------------
// Bank conflict detection (profiling)
// ---------------------------------------------------------------------------

/// Tracks shared memory access patterns to detect bank conflicts.
///
/// In CUDA, shared memory is divided into banks. Simultaneous accesses to the
/// same bank by different threads cause serialisation (bank conflicts). This
/// profiler counts such conflicts to help developers optimise access patterns.
pub struct BankConflictDetector {
    /// Total accesses recorded
    total_accesses: AtomicUsize,
    /// Number of bank conflicts detected
    conflict_count: AtomicUsize,
    /// Per-bank access counters for the current "cycle"
    bank_accesses: [AtomicUsize; NUM_BANKS],
}

impl BankConflictDetector {
    /// Create a new bank conflict detector.
    pub fn new() -> Self {
        const INIT: AtomicUsize = AtomicUsize::new(0);
        Self {
            total_accesses: AtomicUsize::new(0),
            conflict_count: AtomicUsize::new(0),
            bank_accesses: [INIT; NUM_BANKS],
        }
    }

    /// Record an access to a shared memory address.
    ///
    /// Computes which bank the byte address maps to and counts conflicts
    /// when multiple threads in the same warp access the same bank in one
    /// cycle (represented by a batch of `record_access` calls between
    /// `begin_cycle` / `end_cycle`).
    ///
    /// # Arguments
    /// * `byte_address` - The byte offset into shared memory
    pub fn record_access(&self, byte_address: usize) {
        let bank = Self::address_to_bank(byte_address);
        let prev = self.bank_accesses[bank].fetch_add(1, Ordering::Relaxed);
        self.total_accesses.fetch_add(1, Ordering::Relaxed);

        // If this bank was already accessed in the current cycle, it is a conflict
        if prev > 0 {
            self.conflict_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Begin a new access cycle (e.g., a new warp instruction).
    /// Resets the per-bank counters.
    pub fn begin_cycle(&self) {
        for bank in &self.bank_accesses {
            bank.store(0, Ordering::Relaxed);
        }
    }

    /// Compute which bank a byte address maps to.
    ///
    /// Bank index = `(byte_address / BANK_WIDTH_BYTES) % NUM_BANKS`
    pub fn address_to_bank(byte_address: usize) -> usize {
        (byte_address / BANK_WIDTH_BYTES) % NUM_BANKS
    }

    /// Get the total number of accesses recorded.
    pub fn total_accesses(&self) -> usize {
        self.total_accesses.load(Ordering::Relaxed)
    }

    /// Get the number of bank conflicts detected.
    pub fn conflict_count(&self) -> usize {
        self.conflict_count.load(Ordering::Relaxed)
    }

    /// Get the conflict rate (conflicts / total accesses).
    /// Returns 0.0 if no accesses have been recorded.
    pub fn conflict_rate(&self) -> f64 {
        let total = self.total_accesses() as f64;
        if total == 0.0 {
            0.0
        } else {
            self.conflict_count() as f64 / total
        }
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.total_accesses.store(0, Ordering::Relaxed);
        self.conflict_count.store(0, Ordering::Relaxed);
        for bank in &self.bank_accesses {
            bank.store(0, Ordering::Relaxed);
        }
    }

    /// Returns a human-readable summary of bank conflict statistics.
    pub fn summary(&self) -> String {
        format!(
            "Bank conflicts: {} / {} accesses ({:.1}% conflict rate)",
            self.conflict_count(),
            self.total_accesses(),
            self.conflict_rate() * 100.0,
        )
    }
}

impl Default for BankConflictDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_shared_memory_new() {
        let smem: SharedMemory<f32> = SharedMemory::new(256);
        assert_eq!(smem.len(), 256);
        assert!(!smem.is_empty());
    }

    #[test]
    fn test_static_shared_memory_read_write() {
        let mut smem: SharedMemory<i32> = SharedMemory::new(16);
        *smem.get_mut(0) = 42;
        *smem.get_mut(15) = 99;
        assert_eq!(*smem.get(0), 42);
        assert_eq!(*smem.get(15), 99);
        // Zeroed elements
        assert_eq!(*smem.get(1), 0);
    }

    #[test]
    fn test_static_shared_memory_slice() {
        let mut smem: SharedMemory<f32> = SharedMemory::new(8);
        {
            let slice = smem.as_mut_slice();
            for (i, val) in slice.iter_mut().enumerate() {
                *val = i as f32 * 2.0;
            }
        }
        let slice = smem.as_slice();
        assert!((slice[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_static_shared_memory_out_of_bounds() {
        let smem: SharedMemory<u32> = SharedMemory::new(16);
        let _ = smem.get(16);
    }

    #[test]
    fn test_dynamic_shared_memory_new() {
        let dsmem = DynamicSharedMemory::new(1024);
        assert_eq!(dsmem.size_bytes(), 1024);
    }

    #[test]
    fn test_dynamic_shared_memory_typed_access() {
        let mut dsmem = DynamicSharedMemory::new(64); // 16 f32s

        {
            let slice: &mut [f32] = dsmem.as_typed_slice_mut();
            assert_eq!(slice.len(), 16);
            slice[0] = 3.14;
            slice[15] = 2.71;
        }

        let slice: &[f32] = dsmem.as_typed_slice();
        assert!((slice[0] - 3.14).abs() < 1e-6);
        assert!((slice[15] - 2.71).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "size must be > 0")]
    fn test_dynamic_shared_memory_zero_size() {
        let _ = DynamicSharedMemory::new(0);
    }

    #[test]
    fn test_bank_address_mapping() {
        // Address 0 -> bank 0
        assert_eq!(BankConflictDetector::address_to_bank(0), 0);
        // Address 4 -> bank 1
        assert_eq!(BankConflictDetector::address_to_bank(4), 1);
        // Address 128 -> bank 0 (128 / 4 = 32 % 32 = 0)
        assert_eq!(BankConflictDetector::address_to_bank(128), 0);
        // Address 132 -> bank 1
        assert_eq!(BankConflictDetector::address_to_bank(132), 1);
    }

    #[test]
    fn test_no_bank_conflicts() {
        let detector = BankConflictDetector::new();
        detector.begin_cycle();

        // Each access goes to a different bank: addresses 0, 4, 8, 12, ...
        for i in 0..32 {
            detector.record_access(i * 4);
        }

        assert_eq!(detector.total_accesses(), 32);
        assert_eq!(detector.conflict_count(), 0);
    }

    #[test]
    fn test_bank_conflicts_detected() {
        let detector = BankConflictDetector::new();
        detector.begin_cycle();

        // Two accesses to the same bank (bank 0): address 0 and address 128
        detector.record_access(0);
        detector.record_access(128);

        assert_eq!(detector.total_accesses(), 2);
        assert_eq!(detector.conflict_count(), 1);
    }

    #[test]
    fn test_bank_conflict_rate() {
        let detector = BankConflictDetector::new();
        detector.begin_cycle();

        // 4 accesses, 2 conflicts (same bank hit 3 times -> 2 conflicts)
        detector.record_access(0);   // bank 0, first
        detector.record_access(128); // bank 0, conflict
        detector.record_access(256); // bank 0, conflict
        detector.record_access(4);   // bank 1, first

        assert_eq!(detector.total_accesses(), 4);
        assert_eq!(detector.conflict_count(), 2);
        assert!((detector.conflict_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bank_conflict_reset() {
        let detector = BankConflictDetector::new();
        detector.begin_cycle();
        detector.record_access(0);
        detector.record_access(128);

        detector.reset();
        assert_eq!(detector.total_accesses(), 0);
        assert_eq!(detector.conflict_count(), 0);
    }

    #[test]
    fn test_bank_conflict_summary() {
        let detector = BankConflictDetector::new();
        let summary = detector.summary();
        assert!(summary.contains("Bank conflicts"));
    }
}
