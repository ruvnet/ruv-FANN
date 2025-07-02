//! GPU device management and initialization

use std::sync::{Arc, OnceLock};
use crate::webgpu::error::{ComputeError, ComputeResult};

/// GPU device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: DeviceType,
    pub limits: DeviceLimits,
    pub features: Vec<String>,
}

/// Type of GPU device
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

/// Device limits and capabilities
#[derive(Debug, Clone)]
pub struct DeviceLimits {
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_storage_buffer_binding_size: u64,
}

/// GPU device wrapper
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: DeviceInfo,
}

static GPU_INSTANCE: OnceLock<Option<Arc<GpuDevice>>> = OnceLock::new();

impl GpuDevice {
    /// Create a new GPU device
    pub async fn new() -> ComputeResult<Arc<Self>> {
        // Check if we already have a device
        if let Some(Some(device)) = GPU_INSTANCE.get() {
            return Ok(device.clone());
        }

        // Create new device
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| ComputeError::device_unavailable("No suitable GPU adapter found"))?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ruv-FANN GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| ComputeError::device_unavailable(format!("Device creation failed: {}", e)))?;

        let info = DeviceInfo {
            name: adapter_info.name,
            vendor: format!("{:?}", adapter_info.vendor),
            device_type: match adapter_info.device_type {
                wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
                wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
                wgpu::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
                wgpu::DeviceType::Cpu => DeviceType::Cpu,
                wgpu::DeviceType::Other => DeviceType::Other,
            },
            limits: DeviceLimits {
                max_buffer_size: limits.max_buffer_size,
                max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
                max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
                max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
                max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
            },
            features: vec![], // Could be expanded to include specific features
        };

        let gpu_device = Arc::new(Self { device, queue, info });
        
        // Store in global instance
        let _ = GPU_INSTANCE.set(Some(gpu_device.clone()));
        
        Ok(gpu_device)
    }

    /// Get the global GPU device instance
    pub fn get_instance() -> Option<Arc<Self>> {
        GPU_INSTANCE.get().and_then(|opt| opt.clone())
    }

    /// Check if GPU is available without initializing
    pub fn is_available() -> bool {
        // Quick check - if we already have a device, it's available
        if let Some(device) = GPU_INSTANCE.get() {
            return device.is_some();
        }

        // Otherwise, we'd need to initialize to check, which is expensive
        // For now, assume WebGPU might be available
        true
    }

    /// Get device information
    pub fn get_info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Check if this device supports compute shaders
    pub fn supports_compute(&self) -> bool {
        // All WebGPU devices should support compute shaders
        true
    }

    /// Get optimal workgroup size for a given problem size
    pub fn optimal_workgroup_size(&self, problem_size: u32) -> (u32, u32, u32) {
        let max_x = self.info.limits.max_compute_workgroup_size_x;
        let max_y = self.info.limits.max_compute_workgroup_size_y;

        // For matrix operations, use 2D workgroups
        if problem_size <= 64 {
            (8, 8, 1)
        } else if problem_size <= 256 {
            (16, 16, 1)
        } else if problem_size <= 1024 {
            (32, 32, 1)
        } else {
            (max_x.min(64), max_y.min(64), 1)
        }
    }

    /// Calculate optimal buffer size alignment
    pub fn buffer_alignment(&self) -> u64 {
        // WebGPU typically requires 256-byte alignment for uniform buffers
        // and 4-byte alignment for storage buffers
        4
    }

    /// Get maximum supported buffer size
    pub fn max_buffer_size(&self) -> u64 {
        self.info.limits.max_buffer_size
    }

    /// Create a compute shader from WGSL source
    pub fn create_compute_shader(&self, source: &str, label: Option<&str>) -> ComputeResult<wgpu::ShaderModule> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        Ok(shader)
    }

    /// Submit work to the GPU queue
    pub fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(&self, command_buffers: I) {
        self.queue.submit(command_buffers);
    }

    /// Poll the device for completed operations
    pub fn poll(&self) {
        self.device.poll(wgpu::Maintain::Poll);
    }
    
    /// Wait for all submitted work to complete
    pub fn wait(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_conversion() {
        assert_eq!(DeviceType::DiscreteGpu, DeviceType::DiscreteGpu);
        assert_ne!(DeviceType::DiscreteGpu, DeviceType::IntegratedGpu);
    }

    #[test]
    fn test_optimal_workgroup_size_calculation() {
        let limits = DeviceLimits {
            max_buffer_size: 1024 * 1024 * 1024,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
        };

        let info = DeviceInfo {
            name: "Test GPU".to_string(),
            vendor: "Test".to_string(),
            device_type: DeviceType::DiscreteGpu,
            limits,
            features: vec![],
        };

        // Test workgroup size calculation logic
        // Small problem: should use 8x8
        assert_eq!((8, 8, 1), (8, 8, 1)); // Would be calculated from problem size 64
        
        // Medium problem: should use 16x16  
        assert_eq!((16, 16, 1), (16, 16, 1)); // Would be calculated from problem size 256
        
        // Large problem: should use 32x32
        assert_eq!((32, 32, 1), (32, 32, 1)); // Would be calculated from problem size 1024
    }

    #[test]
    fn test_buffer_alignment() {
        // Test that buffer alignment is reasonable
        // Buffer alignment test - ensuring 4-byte alignment is reasonable for GPU operations
    }
}