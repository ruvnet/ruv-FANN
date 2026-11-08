//! WebGPU backend implementation using wgpu
//!
//! Provides REAL GPU compute via WebGPU/wgpu with native device, queue, and pipeline
//! management. Buffer handles returned by `allocate_memory` are synthetic pointers
//! that map to real `wgpu::Buffer` objects stored internally, bridging the
//! `BackendTrait` raw-pointer API with wgpu's owned buffer model.

use super::backend_trait::{BackendCapabilities, BackendTrait, MemcpyKind};
use async_trait::async_trait;
use crate::{runtime_error, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Base address for synthetic GPU buffer handles (avoids null / low addresses).
const HANDLE_BASE: usize = 0x1_0000;

/// wgpu requires `copy_buffer_to_buffer` sizes aligned to 4 bytes.
const COPY_ALIGN: u64 = 4;

/// Round `size` up to the next multiple of [`COPY_ALIGN`].
fn aligned(size: usize) -> u64 {
    let s = size as u64;
    (s + COPY_ALIGN - 1) & !(COPY_ALIGN - 1)
}

/// WebGPU backend using wgpu for cross-platform GPU compute.
pub struct WebGPUBackend {
    capabilities: BackendCapabilities,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    /// Compiled compute pipelines keyed by pipeline ID.
    pipelines: Mutex<HashMap<u64, wgpu::ComputePipeline>>,
    /// GPU buffers keyed by synthetic handle address -> (Buffer, requested byte size).
    buffers: Mutex<HashMap<usize, (wgpu::Buffer, usize)>>,
    /// Next pipeline ID counter.
    next_pipeline_id: Mutex<u64>,
    /// Monotonic counter for generating unique buffer handles.
    next_handle: AtomicUsize,
}

impl Default for WebGPUBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WebGPUBackend {
    /// Create a new WebGPU backend. Call [`initialize`] before any GPU operations.
    pub fn new() -> Self {
        Self {
            capabilities: BackendCapabilities {
                name: "WebGPU (wgpu)".to_string(),
                supports_cuda: false,
                supports_opencl: false,
                supports_vulkan: false,
                supports_webgpu: true,
                max_threads: 65535 * 256,
                max_threads_per_block: 256,
                max_blocks_per_grid: 65535,
                max_shared_memory: 16384,
                supports_dynamic_parallelism: false,
                supports_unified_memory: false,
                max_grid_dim: [65535, 65535, 65535],
                max_block_dim: [256, 256, 64],
                warp_size: 32,
            },
            device: None,
            queue: None,
            pipelines: Mutex::new(HashMap::new()),
            buffers: Mutex::new(HashMap::new()),
            next_pipeline_id: Mutex::new(1),
            next_handle: AtomicUsize::new(HANDLE_BASE),
        }
    }

    /// Check if WebGPU is conceptually available on this platform.
    /// Actual adapter availability is verified in [`initialize`].
    pub fn is_available() -> bool {
        true
    }

    /// Encode a pipeline ID as kernel bytes (8 bytes, little-endian).
    fn pipeline_id_to_bytes(id: u64) -> Vec<u8> {
        id.to_le_bytes().to_vec()
    }

    /// Decode kernel bytes back to a pipeline ID.
    fn bytes_to_pipeline_id(bytes: &[u8]) -> Result<u64> {
        if bytes.len() < 8 {
            return Err(runtime_error!("Invalid kernel handle: too short"));
        }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        Ok(u64::from_le_bytes(arr))
    }

    fn device(&self) -> Result<&wgpu::Device> {
        self.device
            .as_ref()
            .ok_or_else(|| runtime_error!("Backend not initialized: call initialize() first"))
    }

    fn queue(&self) -> Result<&wgpu::Queue> {
        self.queue
            .as_ref()
            .ok_or_else(|| runtime_error!("Backend not initialized: call initialize() first"))
    }
}

unsafe impl Send for WebGPUBackend {}
unsafe impl Sync for WebGPUBackend {}

#[async_trait(?Send)]
impl BackendTrait for WebGPUBackend {
    fn name(&self) -> &str {
        "WebGPU (wgpu)"
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    async fn initialize(&mut self) -> Result<()> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| runtime_error!("No WebGPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cuda-wasm"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| runtime_error!("Failed to create wgpu device: {}", e))?;

        self.device = Some(device);
        self.queue = Some(queue);
        Ok(())
    }

    async fn compile_kernel(&self, source: &str) -> Result<Vec<u8>> {
        let device = self.device()?;

        // Use error scopes to capture shader validation failures.
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kernel"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        device.poll(wgpu::Maintain::Wait);
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(runtime_error!("Shader compilation failed: {}", e));
        }

        // Create compute pipeline with auto bind-group layout.
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
        });
        device.poll(wgpu::Maintain::Wait);
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(runtime_error!("Pipeline creation failed: {}", e));
        }

        let mut id_guard = self
            .next_pipeline_id
            .lock()
            .map_err(|e| runtime_error!("Pipeline ID lock poisoned: {}", e))?;
        let id = *id_guard;
        *id_guard += 1;

        self.pipelines
            .lock()
            .map_err(|e| runtime_error!("Pipeline lock poisoned: {}", e))?
            .insert(id, pipeline);

        Ok(Self::pipeline_id_to_bytes(id))
    }

    async unsafe fn launch_kernel(
        &self,
        kernel: &[u8],
        grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        args: &[*const u8],
    ) -> Result<()> {
        // Snapshot arg pointers as usize immediately so the future is Send
        // (raw pointers are !Sync, making &[*const u8] !Send).
        let arg_handles: Vec<usize> = args.iter().map(|p| *p as usize).collect();

        let device = self.device()?;
        let queue = self.queue()?;
        let pipeline_id = Self::bytes_to_pipeline_id(kernel)?;

        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Err(runtime_error!("Grid dimensions must be non-zero"));
        }
        if grid.0 > 65535 || grid.1 > 65535 || grid.2 > 65535 {
            return Err(runtime_error!("Grid dimension exceeds maximum (65535)"));
        }

        let pipelines = self
            .pipelines
            .lock()
            .map_err(|e| runtime_error!("Pipeline lock poisoned: {}", e))?;
        let pipeline = pipelines
            .get(&pipeline_id)
            .ok_or_else(|| runtime_error!("Kernel not found: pipeline ID {}", pipeline_id))?;

        let buffers_guard = self
            .buffers
            .lock()
            .map_err(|e| runtime_error!("Buffer lock poisoned: {}", e))?;

        // Build bind group entries from arg handles.
        let mut entries = Vec::with_capacity(arg_handles.len());
        for (i, &handle) in arg_handles.iter().enumerate() {
            let (buf, _) = buffers_guard
                .get(&handle)
                .ok_or_else(|| runtime_error!("Arg {} buffer handle {:#x} not found", i, handle))?;
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            });
        }

        let bind_group = if !entries.is_empty() {
            let layout = pipeline.get_bind_group_layout(0);
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            }))
        } else {
            None
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            if let Some(bg) = &bind_group {
                pass.set_bind_group(0, bg, &[]);
            }
            pass.dispatch_workgroups(grid.0, grid.1, grid.2);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        Ok(())
    }

    fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Err(runtime_error!("Cannot allocate zero bytes"));
        }
        let device = self.device()?;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: aligned(size),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let handle = self.next_handle.fetch_add(1, Ordering::SeqCst);
        self.buffers
            .lock()
            .map_err(|e| runtime_error!("Buffer lock poisoned: {}", e))?
            .insert(handle, (buffer, size));

        Ok(handle as *mut u8)
    }

    fn free_memory(&self, ptr: *mut u8) -> Result<()> {
        let handle = ptr as usize;
        let (buffer, _) = self
            .buffers
            .lock()
            .map_err(|e| runtime_error!("Buffer lock poisoned: {}", e))?
            .remove(&handle)
            .ok_or_else(|| runtime_error!("Attempted to free untracked handle {:#x}", handle))?;
        drop(buffer);
        Ok(())
    }

    unsafe fn copy_memory(
        &self,
        dst: *mut u8,
        src: *const u8,
        size: usize,
        kind: MemcpyKind,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        match kind {
            MemcpyKind::HostToDevice => {
                let queue = self.queue()?;
                let device = self.device()?;
                let dst_handle = dst as usize;
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|e| runtime_error!("Buffer lock poisoned: {}", e))?;
                let (gpu_buf, buf_size) = buffers
                    .get(&dst_handle)
                    .ok_or_else(|| runtime_error!("Dst buffer handle not found"))?;
                if size > *buf_size {
                    return Err(runtime_error!(
                        "Copy size {} exceeds buffer size {}",
                        size,
                        buf_size
                    ));
                }
                let data = unsafe { std::slice::from_raw_parts(src, size) };
                queue.write_buffer(gpu_buf, 0, data);
                queue.submit(std::iter::empty());
                device.poll(wgpu::Maintain::Wait);
                Ok(())
            }
            MemcpyKind::DeviceToHost => {
                let device = self.device()?;
                let queue = self.queue()?;
                let src_handle = src as usize;
                let copy_size = aligned(size);
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|e| runtime_error!("Buffer lock poisoned: {}", e))?;
                let (gpu_buf, buf_size) = buffers
                    .get(&src_handle)
                    .ok_or_else(|| runtime_error!("Src buffer handle not found"))?;
                if size > *buf_size {
                    return Err(runtime_error!(
                        "Copy size {} exceeds buffer size {}",
                        size,
                        buf_size
                    ));
                }
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging_read"),
                    size: copy_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                encoder.copy_buffer_to_buffer(gpu_buf, 0, &staging, 0, copy_size);
                queue.submit(std::iter::once(encoder.finish()));

                let slice = staging.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).ok();
                });
                device.poll(wgpu::Maintain::Wait);
                rx.recv()
                    .map_err(|_| runtime_error!("Buffer map channel closed"))?
                    .map_err(|e| runtime_error!("Buffer map failed: {:?}", e))?;

                let mapped = slice.get_mapped_range();
                unsafe {
                    std::ptr::copy_nonoverlapping(mapped.as_ptr(), dst, size);
                }
                drop(mapped);
                staging.unmap();
                Ok(())
            }
            MemcpyKind::DeviceToDevice => {
                let device = self.device()?;
                let queue = self.queue()?;
                let src_handle = src as usize;
                let dst_handle = dst as usize;
                let copy_size = aligned(size);
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|e| runtime_error!("Buffer lock poisoned: {}", e))?;
                let (src_buf, _) = buffers
                    .get(&src_handle)
                    .ok_or_else(|| runtime_error!("Src buffer handle not found"))?;
                let (dst_buf, _) = buffers
                    .get(&dst_handle)
                    .ok_or_else(|| runtime_error!("Dst buffer handle not found"))?;
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                encoder.copy_buffer_to_buffer(src_buf, 0, dst_buf, 0, copy_size);
                queue.submit(std::iter::once(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                Ok(())
            }
            MemcpyKind::HostToHost => {
                if dst.is_null() || src.is_null() {
                    return Err(runtime_error!("Null pointer in host memory copy"));
                }
                unsafe { std::ptr::copy_nonoverlapping(src, dst, size) };
                Ok(())
            }
        }
    }

    fn synchronize(&self) -> Result<()> {
        if let Some(device) = &self.device {
            device.poll(wgpu::Maintain::Wait);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Try to create and initialize a backend. Returns None if no GPU adapter found.
    fn try_init_backend() -> Option<WebGPUBackend> {
        if !super::super::gpu_enabled() { return None; }
        let mut backend = WebGPUBackend::new();
        pollster::block_on(backend.initialize()).ok()?;
        Some(backend)
    }

    // ---- Tests that do NOT require a GPU ----

    #[test]
    fn test_backend_creation() {
        let backend = WebGPUBackend::new();
        assert_eq!(backend.name(), "WebGPU (wgpu)");
        assert!(backend.capabilities().supports_webgpu);
    }

    #[test]
    fn test_is_available() {
        assert!(WebGPUBackend::is_available());
    }

    #[test]
    fn test_capabilities() {
        let backend = WebGPUBackend::new();
        let caps = backend.capabilities();
        assert_eq!(caps.warp_size, 32);
        assert!(caps.max_shared_memory > 0);
    }

    #[test]
    fn test_pipeline_id_roundtrip() {
        let id = 12345u64;
        let bytes = WebGPUBackend::pipeline_id_to_bytes(id);
        assert_eq!(bytes.len(), 8);
        assert_eq!(WebGPUBackend::bytes_to_pipeline_id(&bytes).unwrap(), id);
    }

    #[test]
    fn test_pipeline_id_short_fails() {
        assert!(WebGPUBackend::bytes_to_pipeline_id(&[1, 2]).is_err());
    }

    #[test]
    fn test_allocate_zero_fails() {
        let backend = WebGPUBackend::new();
        assert!(backend.allocate_memory(0).is_err());
    }

    #[test]
    fn test_uninitialized_allocate_fails() {
        let backend = WebGPUBackend::new();
        assert!(backend.allocate_memory(1024).is_err());
    }

    #[test]
    fn test_free_untracked_fails() {
        let backend = WebGPUBackend::new();
        let fake = 0xDEAD as *mut u8;
        assert!(backend.free_memory(fake).is_err());
    }

    #[test]
    fn test_copy_zero_noop() {
        let backend = WebGPUBackend::new();
        let a = 1 as *mut u8;
        unsafe { backend
            .copy_memory(a, a, 0, MemcpyKind::DeviceToDevice) }
            .unwrap();
    }

    #[test]
    fn test_host_to_host_copy() {
        let backend = WebGPUBackend::new();
        let src = vec![1u8, 2, 3, 4];
        let mut dst = vec![0u8; 4];
        unsafe { backend
            .copy_memory(dst.as_mut_ptr(), src.as_ptr(), 4, MemcpyKind::HostToHost) }
            .unwrap();
        assert_eq!(dst, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_host_to_host_null_fails() {
        let backend = WebGPUBackend::new();
        let ptr = vec![0u8; 64];
        assert!(unsafe { backend
            .copy_memory(std::ptr::null_mut(), ptr.as_ptr(), 64, MemcpyKind::HostToHost) }
            .is_err());
    }

    #[test]
    fn test_synchronize_uninitialized() {
        let backend = WebGPUBackend::new();
        backend.synchronize().unwrap();
    }

    // ---- Tests that REQUIRE a GPU adapter ----

    #[test]
    fn test_gpu_allocate_and_free() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_allocate_and_free: no GPU adapter");
                return;
            }
        };
        let handle = backend.allocate_memory(1024).unwrap();
        assert!(!handle.is_null());
        assert!(handle as usize >= HANDLE_BASE);
        backend.free_memory(handle).unwrap();
    }

    #[test]
    fn test_gpu_data_roundtrip() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_data_roundtrip: no GPU adapter");
                return;
            }
        };
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let gpu_buf = backend.allocate_memory(256).unwrap();

        unsafe { backend
            .copy_memory(gpu_buf, data.as_ptr(), 256, MemcpyKind::HostToDevice) }
            .unwrap();

        let mut readback = vec![0u8; 256];
        unsafe { backend
            .copy_memory(
                readback.as_mut_ptr(),
                gpu_buf as *const u8,
                256,
                MemcpyKind::DeviceToHost,
            ) }
            .unwrap();

        assert_eq!(readback, data);
        backend.free_memory(gpu_buf).unwrap();
    }

    #[test]
    fn test_gpu_device_to_device_copy() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_device_to_device_copy: no GPU adapter");
                return;
            }
        };
        let data: Vec<u8> = (0..128).map(|i| (i * 2) as u8).collect();
        let buf_a = backend.allocate_memory(128).unwrap();
        let buf_b = backend.allocate_memory(128).unwrap();

        unsafe { backend
            .copy_memory(buf_a, data.as_ptr(), 128, MemcpyKind::HostToDevice) }
            .unwrap();
        unsafe { backend
            .copy_memory(buf_b, buf_a as *const u8, 128, MemcpyKind::DeviceToDevice) }
            .unwrap();

        let mut readback = vec![0u8; 128];
        unsafe { backend
            .copy_memory(
                readback.as_mut_ptr(),
                buf_b as *const u8,
                128,
                MemcpyKind::DeviceToHost,
            ) }
            .unwrap();

        assert_eq!(readback, data);
        backend.free_memory(buf_a).unwrap();
        backend.free_memory(buf_b).unwrap();
    }

    #[test]
    fn test_gpu_synchronize() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_synchronize: no GPU adapter");
                return;
            }
        };
        backend.synchronize().unwrap();
    }

    #[tokio::test]
    async fn test_gpu_compile_valid_wgsl() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_compile_valid_wgsl: no GPU adapter");
                return;
            }
        };
        let kernel = backend
            .compile_kernel("@compute @workgroup_size(64) fn main() {}")
            .await
            .unwrap();
        assert_eq!(kernel.len(), 8);
    }

    #[tokio::test]
    async fn test_gpu_compile_invalid_wgsl() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_compile_invalid_wgsl: no GPU adapter");
                return;
            }
        };
        assert!(backend.compile_kernel("not valid wgsl").await.is_err());
    }

    #[tokio::test]
    async fn test_gpu_launch_missing_kernel() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_launch_missing_kernel: no GPU adapter");
                return;
            }
        };
        let fake = WebGPUBackend::pipeline_id_to_bytes(999);
        assert!(unsafe { backend
            .launch_kernel(&fake, (1, 1, 1), (64, 1, 1), &[]) }
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_gpu_compile_and_launch() {
        let backend = match try_init_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test_gpu_compile_and_launch: no GPU adapter");
                return;
            }
        };
        let kernel = backend
            .compile_kernel("@compute @workgroup_size(64) fn main() {}")
            .await
            .unwrap();
        unsafe { backend
            .launch_kernel(&kernel, (1, 1, 1), (64, 1, 1), &[]) }
            .await
            .unwrap();
    }
}
