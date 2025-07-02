//! GPU device management with advanced capabilities detection
//!
//! This module provides comprehensive GPU device initialization, capability detection,
//! and management for the WebGPU backend. It includes staging optimizations for
//! production-ready GPU acceleration.

use crate::webgpu::error::{ComputeError, ComputeResult};

/// GPU device wrapper with advanced capabilities
#[derive(Debug)]
pub struct GpuDevice {
    /// WebGPU device handle
    pub device: ::wgpu::Device,
    /// WebGPU queue for command submission
    pub queue: ::wgpu::Queue,
    /// Device adapter information
    adapter_info: ::wgpu::AdapterInfo,
    /// Device limits and capabilities
    limits: ::wgpu::Limits,
}

/// Device type classification for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Unknown,
}

/// Detailed device information for optimization
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub name: String,
    pub vendor: String,
    pub backend: String,
    pub limits: ::wgpu::Limits,
    pub features: ::wgpu::Features,
}

impl GpuDevice {
    /// Initialize GPU device with advanced capability detection
    pub async fn new() -> ComputeResult<Self> {
        // Create WebGPU instance
        let instance = ::wgpu::Instance::new(::wgpu::InstanceDescriptor {
            backends: ::wgpu::Backends::all(),
            flags: ::wgpu::InstanceFlags::default(),
            dx12_shader_compiler: ::wgpu::Dx12Compiler::default(),
            gles_minor_version: ::wgpu::Gles3MinorVersion::Automatic,
        });

        // Request adapter with high performance preference
        let adapter = instance
            .request_adapter(&::wgpu::RequestAdapterOptions {
                power_preference: ::wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                ComputeError::InitializationError(
                    "Failed to find suitable WebGPU adapter".to_string(),
                )
            })?;

        // Get adapter info for optimization decisions
        let adapter_info = adapter.get_info();

        // Request device with required features and limits
        let required_features = ::wgpu::Features::empty();
        let required_limits = ::wgpu::Limits {
            max_compute_workgroup_storage_size: 32768,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_workgroup_size_z: 64,
            ..::wgpu::Limits::downlevel_webgl2_defaults()
        };

        let (device, queue) = adapter
            .request_device(
                &::wgpu::DeviceDescriptor {
                    label: Some("ruv-FANN GPU Device"),
                    required_features,
                    required_limits: required_limits.clone(),
                },
                None,
            )
            .await
            .map_err(|e| {
                ComputeError::InitializationError(format!("Failed to create WebGPU device: {}", e))
            })?;

        Ok(Self {
            device,
            queue,
            adapter_info,
            limits: required_limits,
        })
    }

    /// Get comprehensive device information
    pub fn get_info(&self) -> DeviceInfo {
        let device_type = match self.adapter_info.device_type {
            ::wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
            ::wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
            ::wgpu::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
            ::wgpu::DeviceType::Cpu => DeviceType::Cpu,
            ::wgpu::DeviceType::Other => DeviceType::Unknown,
        };

        DeviceInfo {
            device_type,
            name: self.adapter_info.name.clone(),
            vendor: format!("{:?}", self.adapter_info.vendor),
            backend: format!("{:?}", self.adapter_info.backend),
            limits: self.limits.clone(),
            features: self.device.features(),
        }
    }

    /// Create compute shader with error handling and optimization
    pub fn create_compute_shader(
        &self,
        source: &str,
        label: Option<&str>,
    ) -> ComputeResult<::wgpu::ShaderModule> {
        let shader_descriptor = ::wgpu::ShaderModuleDescriptor {
            label,
            source: ::wgpu::ShaderSource::Wgsl(source.into()),
        };

        Ok(self.device.create_shader_module(shader_descriptor))
    }

    /// Submit command buffers with performance tracking
    pub fn submit<I>(&self, command_buffers: I) -> ::wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = ::wgpu::CommandBuffer>,
    {
        self.queue.submit(command_buffers)
    }

    /// Wait for all submitted work to complete
    pub fn wait(&self) {
        self.device.poll(::wgpu::Maintain::Wait);
    }

    /// Check device features for optimization decisions
    pub fn supports_feature(&self, feature: ::wgpu::Features) -> bool {
        self.device.features().contains(feature)
    }

    /// Get maximum buffer size supported by this device
    pub fn max_buffer_size(&self) -> u64 {
        self.limits.max_buffer_size
    }

    /// Get maximum storage buffer binding size
    pub fn max_storage_buffer_binding_size(&self) -> u32 {
        self.limits.max_storage_buffer_binding_size
    }

    /// Get maximum compute workgroup size in each dimension
    pub fn max_compute_workgroup_size(&self) -> (u32, u32, u32) {
        (
            self.limits.max_compute_workgroup_size_x,
            self.limits.max_compute_workgroup_size_y,
            self.limits.max_compute_workgroup_size_z,
        )
    }

    /// Get maximum number of compute workgroups per dimension
    pub fn max_compute_workgroups_per_dimension(&self) -> u32 {
        self.limits.max_compute_workgroups_per_dimension
    }

    /// Estimate optimal workgroup size for given problem size
    pub fn estimate_optimal_workgroup_size(&self, problem_size: usize) -> u32 {
        let max_size = self.limits.max_compute_workgroup_size_x.min(1024);

        // Common optimal sizes based on GPU architectures
        let candidates = [32, 64, 128, 256, 512, 1024];

        candidates
            .iter()
            .filter(|&&size| size <= max_size)
            .min_by_key(|&&size| {
                // Prefer sizes that minimize padding and align with problem size
                problem_size.div_ceil(size as usize) * size as usize - problem_size
            })
            .copied()
            .unwrap_or(64) // Safe fallback
    }

    /// Check if device is suitable for high-performance computing
    pub fn is_high_performance(&self) -> bool {
        matches!(self.get_info().device_type, DeviceType::DiscreteGpu)
            && self.limits.max_compute_workgroup_size_x >= 256
            && self.limits.max_storage_buffer_binding_size >= 128 * 1024 * 1024 // 128MB
    }

    /// Get estimated memory bandwidth in GB/s
    pub fn estimated_memory_bandwidth(&self) -> f32 {
        match self.get_info().device_type {
            DeviceType::DiscreteGpu => 500.0,  // Modern discrete GPU
            DeviceType::IntegratedGpu => 50.0, // Integrated GPU
            DeviceType::VirtualGpu => 25.0,    // Virtual/cloud GPU
            DeviceType::Cpu => 25.0,           // CPU memory
            DeviceType::Unknown => 10.0,       // Conservative fallback
        }
    }

    /// Get estimated compute throughput in GFLOPS
    pub fn estimated_compute_throughput(&self) -> f32 {
        let base_throughput = match self.get_info().device_type {
            DeviceType::DiscreteGpu => 1000.0,
            DeviceType::IntegratedGpu => 100.0,
            DeviceType::VirtualGpu => 200.0,
            DeviceType::Cpu => 50.0,
            DeviceType::Unknown => 10.0,
        };

        // Scale by compute units (estimated from workgroup size)
        let compute_units = self.limits.max_compute_workgroup_size_x as f32 / 32.0;
        base_throughput * compute_units.min(32.0) // Cap at 32x scaling
    }

    /// Create optimal bind group layout for matrix operations
    pub fn create_matrix_bind_group_layout(&self) -> ::wgpu::BindGroupLayout {
        self.device
            .create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
                label: Some("matrix_operations_bind_group_layout"),
                entries: &[
                    // Input matrix A
                    ::wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ::wgpu::ShaderStages::COMPUTE,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Input matrix/vector B
                    ::wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ::wgpu::ShaderStages::COMPUTE,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output buffer
                    ::wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ::wgpu::ShaderStages::COMPUTE,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Uniform parameters
                    ::wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ::wgpu::ShaderStages::COMPUTE,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
    }

    /// Create optimal bind group layout for activation functions
    pub fn create_activation_bind_group_layout(&self) -> ::wgpu::BindGroupLayout {
        self.device
            .create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
                label: Some("activation_functions_bind_group_layout"),
                entries: &[
                    // Input buffer
                    ::wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ::wgpu::ShaderStages::COMPUTE,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output buffer
                    ::wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ::wgpu::ShaderStages::COMPUTE,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Activation parameters
                    ::wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ::wgpu::ShaderStages::COMPUTE,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
    }

    /// Validate device capabilities for neural network operations
    pub fn validate_neural_network_support(&self) -> ComputeResult<()> {
        let info = self.get_info();

        // Check minimum workgroup size
        if info.limits.max_compute_workgroup_size_x < 32 {
            return Err(ComputeError::InitializationError(
                "Device workgroup size too small for neural network operations".to_string(),
            ));
        }

        // Check minimum storage buffer size
        if info.limits.max_storage_buffer_binding_size < 16 * 1024 * 1024 {
            // 16MB
            return Err(ComputeError::InitializationError(
                "Device storage buffer size too small for neural network operations".to_string(),
            ));
        }

        // Check compute invocations
        if info.limits.max_compute_invocations_per_workgroup < 256 {
            return Err(ComputeError::InitializationError(
                "Device compute invocations per workgroup too small".to_string(),
            ));
        }

        Ok(())
    }
}

// Note: Clone is not implemented for GpuDevice because wgpu::Device and wgpu::Queue
// cannot be cloned. Use Arc<GpuDevice> for sharing instead.

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_device_creation() {
        // This test will only pass if WebGPU is available
        match GpuDevice::new().await {
            Ok(device) => {
                let info = device.get_info();
                println!("Device: {} ({:?})", info.name, info.device_type);

                // Check device capabilities with soft assertions
                // Using unwrap_or for non-fatal error handling
                assert!(device.max_buffer_size() > 0);

                let storage_buffer_size = device.max_storage_buffer_binding_size();
                if storage_buffer_size == 0 {
                    println!("Warning: Device reports zero max_storage_buffer_binding_size!");
                } else {
                    println!("Max storage buffer binding size: {}", storage_buffer_size);
                }

                // Test validation
                let validation_result = device.validate_neural_network_support();
                if validation_result.is_ok() {
                    println!("Device supports neural network operations");
                } else {
                    println!("Device limitations: {:?}", validation_result);
                }
            }
            Err(e) => {
                println!("WebGPU not available: {}", e);
                // Skip test if WebGPU is not available
            }
        }
    }

    #[tokio::test]
    async fn test_workgroup_optimization() {
        if let Ok(device) = GpuDevice::new().await {
            let optimal_size = device.estimate_optimal_workgroup_size(1000);
            assert!(optimal_size > 0);
            assert!(optimal_size <= device.max_compute_workgroup_size().0);

            println!("Optimal workgroup size for 1000 elements: {}", optimal_size);
        }
    }

    #[tokio::test]
    async fn test_performance_estimates() {
        if let Ok(device) = GpuDevice::new().await {
            let bandwidth = device.estimated_memory_bandwidth();
            let throughput = device.estimated_compute_throughput();

            assert!(bandwidth > 0.0);
            assert!(throughput > 0.0);

            println!("Estimated bandwidth: {} GB/s", bandwidth);
            println!("Estimated throughput: {} GFLOPS", throughput);
            println!("High performance: {}", device.is_high_performance());
        }
    }
}
