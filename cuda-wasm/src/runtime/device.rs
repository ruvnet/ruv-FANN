//! Device abstraction for different backends

use crate::{Result, runtime_error};
use std::sync::Arc;

/// Device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub max_threads_per_block: u32,
    pub max_blocks_per_grid: u32,
    pub warp_size: u32,
    pub compute_capability: (u32, u32),
}

/// Backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Native,
    WebGPU,
    CPU,
}

/// Device abstraction
pub struct Device {
    backend: BackendType,
    properties: DeviceProperties,
    id: usize,
}

impl Device {
    /// Get the default device
    pub fn get_default() -> Result<Arc<Self>> {
        // Detect available backend
        let backend = Self::detect_backend();

        let properties = match backend {
            BackendType::Native => Self::get_native_properties()?,
            BackendType::WebGPU => Self::get_webgpu_properties()?,
            BackendType::CPU => Self::get_cpu_properties(),
        };

        Ok(Arc::new(Self {
            backend,
            properties,
            id: 0,
        }))
    }

    /// Get device by ID
    pub fn get_by_id(id: usize) -> Result<Arc<Self>> {
        // For now, only support device 0
        if id != 0 {
            return Err(runtime_error!("Device {} not found", id));
        }
        Self::get_default()
    }

    /// Get device count
    pub fn count() -> Result<usize> {
        // For now, always return 1
        Ok(1)
    }

    /// Get device properties
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Get backend type
    pub fn backend(&self) -> BackendType {
        self.backend
    }

    /// Get device ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Detect available backend
    fn detect_backend() -> BackendType {
        if !crate::backend::gpu_enabled() { return BackendType::CPU; }
        #[cfg(target_arch = "wasm32")]
        {
            return BackendType::WebGPU;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Try native GPU detection
            if crate::backend::native_gpu::is_cuda_available()
                || crate::backend::native_gpu::is_rocm_available()
            {
                return BackendType::Native;
            }

            // Try WebGPU via wgpu
            if Self::probe_webgpu() {
                return BackendType::WebGPU;
            }

            BackendType::CPU
        }
    }

    /// Probe whether a WebGPU-compatible adapter is available via wgpu.
    #[cfg(not(target_arch = "wasm32"))]
    fn probe_webgpu() -> bool {
        use pollster::FutureExt;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .is_some()
    }

    /// Get native GPU properties by querying the system.
    ///
    /// Tries nvidia-smi for NVIDIA GPUs, then sysfs for AMD GPUs,
    /// falling back to generic properties if neither is available.
    fn get_native_properties() -> Result<DeviceProperties> {
        // Try nvidia-smi first
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let line = stdout.trim();
                let parts: Vec<&str> = line.split(", ").collect();
                if parts.len() >= 2 {
                    let name = parts[0].trim().to_string();
                    let mem_mb: usize = parts[1].trim().parse().unwrap_or(8192);
                    return Ok(DeviceProperties {
                        name,
                        total_memory: mem_mb * 1024 * 1024,
                        max_threads_per_block: 1024,
                        max_blocks_per_grid: 65535,
                        warp_size: 32,
                        compute_capability: (8, 0),
                    });
                }
            }
        }

        // Try reading sysfs for AMD GPUs
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let vendor_path = entry.path().join("device/vendor");
                if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                    if vendor.trim() == "0x1002" {
                        // AMD vendor ID
                        let name = std::fs::read_to_string(
                            entry.path().join("device/product_name"),
                        )
                        .unwrap_or_else(|_| "AMD GPU".to_string());
                        return Ok(DeviceProperties {
                            name: name.trim().to_string(),
                            total_memory: 16 * 1024 * 1024 * 1024,
                            max_threads_per_block: 1024,
                            max_blocks_per_grid: 65535,
                            warp_size: 64,
                            compute_capability: (9, 0),
                        });
                    }
                }
            }
        }

        // Fallback: generic GPU properties
        Ok(DeviceProperties {
            name: "GPU Device (properties unavailable)".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,
            max_threads_per_block: 1024,
            max_blocks_per_grid: 65535,
            warp_size: 32,
            compute_capability: (0, 0),
        })
    }

    /// Get WebGPU device properties by querying a wgpu adapter.
    ///
    /// On non-wasm targets this creates a real wgpu instance and reads
    /// the adapter info and limits.  Falls back to reasonable defaults
    /// when no adapter is found or on wasm32.
    fn get_webgpu_properties() -> Result<DeviceProperties> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use pollster::FutureExt;
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            if let Some(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .block_on()
            {
                let info = adapter.get_info();
                let limits = adapter.limits();
                return Ok(DeviceProperties {
                    name: info.name,
                    total_memory: 0, // WebGPU does not expose total memory
                    max_threads_per_block: limits.max_compute_invocations_per_workgroup,
                    max_blocks_per_grid: limits.max_compute_workgroups_per_dimension,
                    warp_size: 32,
                    compute_capability: (1, 0),
                });
            }
        }
        Ok(DeviceProperties {
            name: "WebGPU Device".to_string(),
            total_memory: 2 * 1024 * 1024 * 1024,
            max_threads_per_block: 256,
            max_blocks_per_grid: 65535,
            warp_size: 32,
            compute_capability: (1, 0),
        })
    }

    /// Get CPU properties by reading system information.
    ///
    /// Reads /proc/cpuinfo for the model name and queries
    /// `available_parallelism` for the thread count.
    fn get_cpu_properties() -> DeviceProperties {
        let name = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|l| l.starts_with("model name"))
                    .map(|l| l.split(':').nth(1).unwrap_or("CPU").trim().to_string())
            })
            .unwrap_or_else(|| "CPU Device".to_string());

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        DeviceProperties {
            name,
            total_memory: 16 * 1024 * 1024 * 1024, // Would need /proc/meminfo
            max_threads_per_block: threads as u32,
            max_blocks_per_grid: 65535,
            warp_size: 1,
            compute_capability: (0, 0),
        }
    }
}
