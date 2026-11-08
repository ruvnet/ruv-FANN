// Included in webgpu_impl; private storage is owned by each backend/device.
#[derive(Default)]
struct BufferStorage {
    buffers: HashMap<BufferHandle, StoredBuffer>,
}

struct StoredBuffer {
    buffer: wgpu::Buffer,
    bytes: usize,
}

// Process-wide IDs prevent another backend's handle or a freed handle from
// accidentally naming a new allocation. IDs are never recycled, even on error.
static NEXT_BUFFER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
const MAX_MANAGED_BUFFERS: usize = 4096;

fn storage_scalar_width<T: 'static>() -> Result<usize, ComputeError> {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Ok(4)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        Ok(8)
    } else {
        Err(ComputeError::UnsupportedOperation(
            "WebGPU buffer transfers support only f32 and f64 IEEE bytes".into(),
        ))
    }
}

fn invalid_buffer(handle: BufferHandle) -> ComputeError {
    ComputeError::MemoryError(format!(
        "Unknown, freed, or foreign buffer handle {}",
        handle.id()
    ))
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> WebGPUBackend<T> {
    fn create_managed_buffer(
        &self,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, ComputeError> {
        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ruv-FANN managed memory"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        let validation = pollster::block_on(self.device.pop_error_scope());
        let allocation = pollster::block_on(self.device.pop_error_scope());
        if let Some(error) = validation.or(allocation) {
            return Err(ComputeError::AllocationError(error.to_string()));
        }
        Ok(buffer)
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> MemoryManager<T> for WebGPUBackend<T> {
    /// Allocate a positive, whole number of T values. Size is in bytes.
    /// The aggregate allocation budget is max_buffer_size, rather than a claim
    /// about physical free VRAM. f64 storage does not enable f64 shaders.
    fn allocate_buffer(&self, size: usize) -> Result<BufferHandle, ComputeError> {
        let width = storage_scalar_width::<T>()?;
        if size == 0 || size % width != 0 {
            return Err(ComputeError::InvalidDimensions(format!(
                "Buffer size must be a positive multiple of {width} bytes"
            )));
        }
        if size > self.capabilities.max_buffer_size
            || size as u64 > self.device.limits().max_buffer_size
        {
            return Err(ComputeError::AllocationError(
                "Buffer exceeds device limit".into(),
            ));
        }
        let mut storage = self
            .managed_buffers
            .lock()
            .map_err(|_| ComputeError::MemoryError("Buffer storage lock is poisoned".into()))?;
        let allocated: usize = storage.buffers.values().map(|buffer| buffer.bytes).sum();
        if size > self.capabilities.max_buffer_size.saturating_sub(allocated)
            || storage.buffers.len() >= MAX_MANAGED_BUFFERS
        {
            return Err(ComputeError::AllocationError(
                "Managed buffer budget exhausted".into(),
            ));
        }
        // fetch_update is the MSRV-compatible spelling of this checked atomic
        // operation; newer Rust versions renamed it to try_update.
        #[allow(deprecated)]
        let id = NEXT_BUFFER_ID
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |id| id.checked_add(1),
            )
            .map_err(|_| ComputeError::AllocationError("Buffer handle space exhausted".into()))?;
        let buffer = self.create_managed_buffer(
            size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )?;
        let handle = BufferHandle::new(id);
        storage.buffers.insert(
            handle,
            StoredBuffer {
                buffer,
                bytes: size,
            },
        );
        Ok(handle)
    }

    /// Write a prefix at offset zero; the remaining allocation is preserved.
    /// All IEEE bit patterns, including NaNs and signed zero, are transferred.
    fn upload_data(&self, handle: BufferHandle, data: &[T]) -> Result<(), ComputeError> {
        use std::any::Any;
        let width = storage_scalar_width::<T>()?;
        let storage = self
            .managed_buffers
            .lock()
            .map_err(|_| ComputeError::MemoryError("Buffer storage lock is poisoned".into()))?;
        let stored = storage
            .buffers
            .get(&handle)
            .ok_or_else(|| invalid_buffer(handle))?;
        let bytes = data
            .len()
            .checked_mul(width)
            .ok_or_else(|| ComputeError::TransferError("Upload byte count overflows".into()))?;
        if bytes > stored.bytes {
            return Err(ComputeError::TransferError(
                "Upload exceeds buffer capacity".into(),
            ));
        }
        if bytes == 0 {
            return Ok(());
        }
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(bytes)
            .map_err(|error| ComputeError::AllocationError(error.to_string()))?;
        // Any validates the concrete type without assuming Float implies Pod or
        // constructing a byte slice from a generic pointer.
        for value in data {
            let value = value as &dyn Any;
            if width == 4 {
                encoded.extend_from_slice(&value.downcast_ref::<f32>().unwrap().to_le_bytes());
            } else {
                encoded.extend_from_slice(&value.downcast_ref::<f64>().unwrap().to_le_bytes());
            }
        }
        self.queue.write_buffer(&stored.buffer, 0, &encoded);
        // Submit queued writes before returning so deallocation cannot leave an
        // unsubmitted transfer referring to a destroyed buffer.
        self.queue.submit(std::iter::empty());
        Ok(())
    }

    /// Read the whole allocation through an actual GPU staging copy.
    fn download_data(&self, handle: BufferHandle) -> Result<Vec<T>, ComputeError> {
        use std::any::Any;
        let width = storage_scalar_width::<T>()?;
        // Keep ownership locked until the copy completes; concurrent writes and
        // destruction of the same allocation cannot race this readback.
        let storage = self
            .managed_buffers
            .lock()
            .map_err(|_| ComputeError::MemoryError("Buffer storage lock is poisoned".into()))?;
        let stored = storage
            .buffers
            .get(&handle)
            .ok_or_else(|| invalid_buffer(handle))?;
        let staging = self.create_managed_buffer(
            stored.bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ruv-FANN managed memory readback"),
            });
        encoder.copy_buffer_to_buffer(&stored.buffer, 0, &staging, 0, stored.bytes as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|error| ComputeError::TransferError(error.to_string()))?
            .map_err(|error| ComputeError::TransferError(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let mut output = Vec::new();
        output
            .try_reserve_exact(stored.bytes / width)
            .map_err(|error| ComputeError::AllocationError(error.to_string()))?;
        for bytes in mapped.chunks_exact(width) {
            let mut value = T::zero();
            let target = &mut value as &mut dyn Any;
            if width == 4 {
                *target.downcast_mut::<f32>().unwrap() =
                    f32::from_le_bytes(bytes.try_into().unwrap());
            } else {
                *target.downcast_mut::<f64>().unwrap() =
                    f64::from_le_bytes(bytes.try_into().unwrap());
            }
            output.push(value);
        }
        drop(mapped);
        staging.unmap();
        Ok(output)
    }

    fn deallocate_buffer(&self, handle: BufferHandle) -> Result<(), ComputeError> {
        let mut storage = self
            .managed_buffers
            .lock()
            .map_err(|_| ComputeError::MemoryError("Buffer storage lock is poisoned".into()))?;
        let stored = storage
            .buffers
            .remove(&handle)
            .ok_or_else(|| invalid_buffer(handle))?;
        stored.buffer.destroy();
        Ok(())
    }

    /// Active managed bytes and remaining aggregate budget. This excludes
    /// transient readback/compute buffers and is not a physical VRAM query.
    fn memory_usage(&self) -> MemoryStats {
        let storage = self
            .managed_buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let allocated = storage.buffers.values().map(|buffer| buffer.bytes).sum();
        MemoryStats {
            total_allocated: allocated,
            available: self.capabilities.max_buffer_size.saturating_sub(allocated),
            buffer_count: storage.buffers.len(),
            fragmentation_ratio: 0.0,
        }
    }
}
