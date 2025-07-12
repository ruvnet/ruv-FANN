//! Memory management utilities for OpenCV

use crate::{Error, Result};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global memory statistics
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// OpenCV memory allocator
pub struct OpenCVAllocator;

unsafe impl GlobalAlloc for OpenCVAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            ptr
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
            ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            ALLOCATION_COUNT.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Initialize memory allocators
pub fn init_allocators() -> Result<()> {
    // Reset statistics
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    Ok(())
}

/// Get current memory usage
pub fn get_memory_usage() -> (usize, usize) {
    (
        ALLOCATED_BYTES.load(Ordering::Relaxed),
        ALLOCATION_COUNT.load(Ordering::Relaxed),
    )
}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    block_size: usize,
    blocks: Vec<Vec<u8>>,
    free_list: Vec<*mut u8>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(block_size: usize, initial_blocks: usize) -> Self {
        let mut pool = MemoryPool {
            block_size,
            blocks: Vec::new(),
            free_list: Vec::new(),
        };
        
        // Pre-allocate initial blocks
        for _ in 0..initial_blocks {
            pool.allocate_block();
        }
        
        pool
    }
    
    /// Allocate a new block
    fn allocate_block(&mut self) {
        let mut block = vec![0u8; self.block_size];
        let ptr = block.as_mut_ptr();
        self.blocks.push(block);
        self.free_list.push(ptr);
    }
    
    /// Get a memory block from the pool
    pub fn get(&mut self) -> *mut u8 {
        if self.free_list.is_empty() {
            self.allocate_block();
        }
        self.free_list.pop().unwrap()
    }
    
    /// Return a memory block to the pool
    pub fn put(&mut self, ptr: *mut u8) {
        self.free_list.push(ptr);
    }
}

/// Aligned memory allocation
pub fn allocate_aligned(size: usize, alignment: usize) -> Result<*mut u8> {
    let layout = Layout::from_size_align(size, alignment)
        .map_err(|_| Error::Memory("Invalid layout".into()))?;
    
    unsafe {
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            Err(Error::Memory("Allocation failed".into()))
        } else {
            Ok(ptr)
        }
    }
}

/// Free aligned memory
pub unsafe fn deallocate_aligned(ptr: *mut u8, size: usize, alignment: usize) {
    if let Ok(layout) = Layout::from_size_align(size, alignment) {
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let (bytes, count) = get_memory_usage();
        assert!(bytes >= 0);
        assert!(count >= 0);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(1024, 10);
        
        let ptr1 = pool.get();
        let ptr2 = pool.get();
        
        assert_ne!(ptr1, ptr2);
        
        pool.put(ptr1);
        pool.put(ptr2);
    }

    #[test]
    fn test_aligned_allocation() {
        let size = 1024;
        let alignment = 16;
        
        let ptr = allocate_aligned(size, alignment).unwrap();
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % alignment, 0);
        
        unsafe {
            deallocate_aligned(ptr, size, alignment);
        }
    }
}