//! Device memory allocation and management

use super::HostBuffer;
use crate::runtime::{BackendType, Device};
use crate::{runtime_error, Result};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::Arc;

/// Raw device memory pointer
pub struct DevicePtr {
    raw: *mut u8,
    size: usize,
    backend: BackendType,
}

impl DevicePtr {
    /// Allocate raw device memory
    pub fn allocate(size: usize, device: &Arc<Device>) -> Result<Self> {
        if size == 0 {
            return Err(runtime_error!("Cannot allocate zero-sized buffer"));
        }

        let backend = device.backend();
        let raw = match backend {
            BackendType::Native => {
                // Host-memory allocation serves as the compute target for CPU and
                // fallback backends.  When a native GPU backend is active, the
                // Backend::allocate_memory path handles device-side allocation.
                unsafe {
                    let layout = Layout::from_size_align(size, 8)
                        .map_err(|e| runtime_error!("Invalid layout: {}", e))?;
                    alloc_zeroed(layout)
                }
            }
            BackendType::WebGPU => {
                // Host-memory allocation for the runtime abstraction layer.
                // The high-level WebGPU backend manages its own device-side
                // buffer objects; DevicePtr provides the host-side mirror.
                unsafe {
                    let layout = Layout::from_size_align(size, 8)
                        .map_err(|e| runtime_error!("Invalid layout: {}", e))?;
                    alloc_zeroed(layout)
                }
            }
            BackendType::CPU => {
                // CPU backend uses regular heap allocation
                unsafe {
                    let layout = Layout::from_size_align(size, 8)
                        .map_err(|e| runtime_error!("Invalid layout: {}", e))?;
                    alloc_zeroed(layout)
                }
            }
        };

        if raw.is_null() {
            return Err(runtime_error!(
                "Failed to allocate {} bytes of device memory",
                size
            ));
        }

        Ok(Self { raw, size, backend })
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const u8 {
        self.raw
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.raw
    }

    /// Get allocation size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            match self.backend {
                BackendType::Native => {
                    // Host-side deallocation for the runtime abstraction.
                    // Native GPU backends handle their own device-side frees.
                    unsafe {
                        if let Ok(layout) = Layout::from_size_align(self.size, 8) {
                            dealloc(self.raw, layout);
                        }
                    }
                }
                BackendType::WebGPU => {
                    // Host-side deallocation for the runtime abstraction.
                    // WebGPU device buffers are released by the backend layer.
                    unsafe {
                        if let Ok(layout) = Layout::from_size_align(self.size, 8) {
                            dealloc(self.raw, layout);
                        }
                    }
                }
                BackendType::CPU => unsafe {
                    if let Ok(layout) = Layout::from_size_align(self.size, 8) {
                        dealloc(self.raw, layout);
                    }
                },
            }
        }
    }
}

/// Initialized host mirror for a runtime device.
///
/// GPU backends manage their own device allocations. This buffer provides typed,
/// aligned CPU storage and does not imply a CUDA or WebGPU allocation.
pub struct DeviceBuffer<T> {
    data: HostBuffer<T>,
    device: Arc<Device>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate a host mirror filled with `T::default()`.
    pub fn new(len: usize, device: Arc<Device>) -> Result<Self>
    where
        T: Default,
    {
        Ok(Self {
            data: HostBuffer::new(len)?,
            device,
        })
    }

    /// Get the buffer length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the associated runtime device.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get a pointer to host storage.
    ///
    /// # Safety
    /// Dereferencing must respect the allocation's lifetime and Rust aliasing.
    pub unsafe fn as_ptr(&self) -> *const T {
        self.data.as_slice().as_ptr()
    }

    /// Get a mutable pointer to host storage.
    ///
    /// # Safety
    /// Dereferencing must respect the allocation's lifetime and Rust aliasing.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_slice().as_mut_ptr()
    }

    /// Copy initialized host elements into the mirror.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        self.data.copy_from_slice(data)
    }

    /// Copy initialized mirror elements into host storage.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        self.data.copy_to_slice(data)
    }

    /// Fill the mirror without allocating an intermediate host vector.
    pub fn fill(&mut self, value: T) -> Result<()> {
        self.data.fill(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Device;

    #[test]
    fn test_device_buffer_allocation() {
        let device = Device::get_default().unwrap();
        let buffer = DeviceBuffer::<f32>::new(1024, device).unwrap();
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_host_device_copy() {
        let device = Device::get_default().unwrap();
        let mut buffer = DeviceBuffer::<f32>::new(100, device).unwrap();

        // Create test data
        let host_data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Copy to device
        buffer.copy_from_host(&host_data).unwrap();

        // Copy back
        let mut result = vec![0.0; 100];
        buffer.copy_to_host(&mut result).unwrap();

        // Verify
        assert_eq!(host_data, result);
    }
}
