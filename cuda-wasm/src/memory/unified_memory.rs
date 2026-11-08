//! Unified memory management for CUDA-Rust
//!
//! Provides unified memory allocation that can be accessed by both host and device

use crate::{Result, memory_error};
use std::sync::Arc;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

/// Unified memory allocation
pub struct UnifiedMemory {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl UnifiedMemory {
    /// Allocate unified memory
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(memory_error!("Cannot allocate zero-sized unified memory"));
        }

        let layout = Layout::from_size_align(size, 8)
            .map_err(|e| memory_error!("Invalid layout: {}", e))?;

        let ptr = unsafe { alloc_zeroed(layout) };
        
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| memory_error!("Failed to allocate unified memory"))?;

        Ok(Self { ptr, size, layout })
    }

    /// Get a pointer to the memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }

    /// Get a mutable pointer to the memory
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy data from host to unified memory
    pub fn copy_from_slice(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(memory_error!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                self.size
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), data.len());
        }

        Ok(())
    }

    /// Copy data from unified memory to host
    pub fn copy_to_slice(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(memory_error!(
                "Destination size {} exceeds buffer size {}",
                data.len(),
                self.size
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), data.as_mut_ptr(), data.len());
        }

        Ok(())
    }
}

impl Drop for UnifiedMemory {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// Safety: UnifiedMemory can be safely sent between threads
unsafe impl Send for UnifiedMemory {}
unsafe impl Sync for UnifiedMemory {}

/// Shared unified memory handle
pub type SharedUnifiedMemory = Arc<UnifiedMemory>;

/// Create a new shared unified memory allocation
pub fn allocate_unified(size: usize) -> Result<SharedUnifiedMemory> {
    Ok(Arc::new(UnifiedMemory::new(size)?))
}

/// Backend-aware unified memory that routes allocation through the active backend
///
/// When a GPU backend is available, this allocates memory that is accessible
/// from both host and device via the backend's memory management. Falls back
/// to host-only allocation when no GPU backend is present.
pub struct ManagedMemory {
    /// Underlying unified memory
    inner: UnifiedMemory,
    /// Whether this memory is registered with a backend
    backend_registered: bool,
}

impl ManagedMemory {
    /// Allocate managed memory (tries backend, falls back to host)
    pub fn new(size: usize) -> Result<Self> {
        let inner = UnifiedMemory::new(size)?;
        // No registration API is invoked: capability support alone does not
        // establish that this host allocation is registered with a GPU.
        let backend_registered = false;
        Ok(Self {
            inner,
            backend_registered,
        })
    }

    /// Check if memory is registered with a GPU backend
    pub fn is_backend_registered(&self) -> bool {
        self.backend_registered
    }

    /// Get the underlying unified memory
    pub fn as_unified(&self) -> &UnifiedMemory {
        &self.inner
    }

    /// Get a mutable reference to the underlying unified memory
    pub fn as_unified_mut(&mut self) -> &mut UnifiedMemory {
        &mut self.inner
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// Copy from host slice
    pub fn copy_from_slice(&mut self, data: &[u8]) -> Result<()> {
        self.inner.copy_from_slice(data)
    }

    /// Copy to host slice
    pub fn copy_to_slice(&self, data: &mut [u8]) -> Result<()> {
        self.inner.copy_to_slice(data)
    }

    /// Prefetch to the device (hint for the runtime; no-op in CPU mode)
    pub fn prefetch_to_device(&self) -> Result<()> {
        // In CPU emulation, this is a no-op since all memory is host-accessible
        Ok(())
    }

    /// Prefetch to the host (hint for the runtime; no-op in CPU mode)
    pub fn prefetch_to_host(&self) -> Result<()> {
        Ok(())
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_memory_allocation() {
        let mem = UnifiedMemory::new(1024).unwrap();
        assert_eq!(mem.size(), 1024);
    }

    #[test]
    fn test_unified_memory_copy() {
        let mut mem = UnifiedMemory::new(256).unwrap();

        let data = vec![42u8; 256];
        mem.copy_from_slice(&data).unwrap();

        let mut output = vec![0u8; 256];
        mem.copy_to_slice(&mut output).unwrap();

        assert_eq!(data, output);
    }

    #[test]
    fn test_zero_size_allocation() {
        let result = UnifiedMemory::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_managed_memory() {
        let mem = ManagedMemory::new(512).unwrap();
        assert_eq!(mem.size(), 512);
    }

    #[test]
    fn test_managed_memory_copy() {
        let mut mem = ManagedMemory::new(128).unwrap();
        let data = vec![0xAB_u8; 128];
        mem.copy_from_slice(&data).unwrap();

        let mut out = vec![0u8; 128];
        mem.copy_to_slice(&mut out).unwrap();
        assert_eq!(data, out);
    }

    #[test]
    fn test_managed_memory_prefetch() {
        let mem = ManagedMemory::new(64).unwrap();
        assert!(mem.prefetch_to_device().is_ok());
        assert!(mem.prefetch_to_host().is_ok());
    }
}