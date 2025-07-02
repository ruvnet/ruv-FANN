//! WebGPU backend implementation for GPU-accelerated neural network operations

use std::sync::Arc;
use num_traits::Float;
use bytemuck::{Pod, Zeroable};
use crate::webgpu::{
    backend::{ComputeBackend, BackendType, MatrixDims, ActivationFunction, DeviceCapabilities},
    device::GpuDevice,
    memory::GpuMemoryManager,
    error::{ComputeError, ComputeResult},
    shaders,
};

/// WebGPU backend for GPU-accelerated operations
#[derive(Debug)]
pub struct WebGpuBackend {
    device: Arc<GpuDevice>,
    memory_manager: GpuMemoryManager,
    capabilities: DeviceCapabilities,
    compute_pipelines: ComputePipelines,
}

/// Cache of compiled compute pipelines
struct ComputePipelines {
    matrix_multiply: Option<wgpu::ComputePipeline>,
    activation_functions: std::collections::HashMap<String, wgpu::ComputePipeline>,
    vector_operations: Option<wgpu::ComputePipeline>,
}

impl std::fmt::Debug for ComputePipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipelines")
            .field("matrix_multiply", &self.matrix_multiply.is_some())
            .field("activation_functions_count", &self.activation_functions.len())
            .field("vector_operations", &self.vector_operations.is_some())
            .finish()
    }
}

impl ComputePipelines {
    fn new() -> Self {
        Self {
            matrix_multiply: None,
            activation_functions: std::collections::HashMap::new(),
            vector_operations: None,
        }
    }
}

impl WebGpuBackend {
    /// Create a new WebGPU backend
    pub async fn new() -> ComputeResult<Self> {
        let device = GpuDevice::new().await?;
        let memory_manager = GpuMemoryManager::new(device.clone());
        
        let info = device.get_info();
        let capabilities = DeviceCapabilities {
            compute_units: 256, // Estimate for modern GPU
            memory_bandwidth_gbps: match info.device_type {
                crate::webgpu::device::DeviceType::DiscreteGpu => 500.0,
                crate::webgpu::device::DeviceType::IntegratedGpu => 50.0,
                _ => 25.0,
            },
            peak_compute_throughput: 1000.0, // GFLOPS estimate
            supports_f16: false, // Conservative for compatibility
            supports_f64: false, // WebGPU typically doesn't support f64
            max_workgroup_size: (
                info.limits.max_compute_workgroup_size_x,
                info.limits.max_compute_workgroup_size_y,
                info.limits.max_compute_workgroup_size_z,
            ),
            estimated_speedup: 10.0, // 10x speedup over CPU
        };

        Ok(Self {
            device,
            memory_manager,
            capabilities,
            compute_pipelines: ComputePipelines::new(),
        })
    }

    /// Get or create matrix multiply compute pipeline
    async fn get_matrix_multiply_pipeline(&mut self) -> ComputeResult<&wgpu::ComputePipeline> {
        if self.compute_pipelines.matrix_multiply.is_none() {
            let shader_source = include_str!("shaders/matrix_operations.wgsl");
            let shader = self.device.create_compute_shader(&shader_source, Some("matrix_multiply_shader"))?;
            
            let bind_group_layout = self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matrix_multiply_bind_group_layout"),
                entries: &[
                    // Matrix A (input)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Vector B (input)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Result (output)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Dimensions (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout = self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matrix_multiply_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matrix_multiply_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "matrix_vector_multiply",
            });

            self.compute_pipelines.matrix_multiply = Some(pipeline);
        }

        Ok(self.compute_pipelines.matrix_multiply.as_ref().unwrap())
    }

    /// Create a buffer with data
    async fn create_buffer_with_data<T: Pod>(&self, data: &[T], usage: wgpu::BufferUsages) -> ComputeResult<crate::webgpu::memory::GpuBuffer> {
        let size = std::mem::size_of_val(data) as u64;
        let buffer = self.memory_manager.create_storage_buffer(size, None)?;
        
        let data_bytes = bytemuck::cast_slice(data);
        self.memory_manager.write_buffer_data(&buffer, data_bytes)?;
        
        Ok(buffer)
    }

    /// Fused linear layer with activation: output = activation(weights @ input + bias)
    pub async fn fused_linear_activation<T: Float + Send + Sync + Pod + Clone>(
        &self,
        weights: &[T],
        input: &[T],
        bias: &[T],
        dims: MatrixDims,
        activation: ActivationFunction,
    ) -> ComputeResult<Vec<T>> {
        // For now, fall back to separate operations
        // A real implementation would use a custom fused kernel
        let output = <Self as ComputeBackend<T>>::matrix_vector_multiply(self, weights, input, dims).await?;
        let output = <Self as ComputeBackend<T>>::vector_add(self, &output, bias).await?;
        <Self as ComputeBackend<T>>::activation_function(self, &output, activation).await
    }

    /// Read data back from GPU buffer
    async fn read_buffer_data<T: Pod + Clone>(&self, buffer: &crate::webgpu::memory::GpuBuffer, count: usize) -> ComputeResult<Vec<T>> {
        let readback_buffer = self.memory_manager.create_readback_buffer(buffer.size, None)?;
        
        let mut encoder = self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });
        
        self.memory_manager.copy_buffer_to_buffer(&mut encoder, buffer, &readback_buffer, buffer.size)?;
        
        self.device.submit([encoder.finish()]);
        
        // Map the buffer and read the data
        let buffer_slice = readback_buffer.buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        self.device.wait();
        receiver.await.unwrap().map_err(|e| ComputeError::buffer_error(format!("Failed to map buffer: {:?}", e)))?;
        
        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data[..count * std::mem::size_of::<T>()]).to_vec();
        
        drop(data);
        readback_buffer.buffer.unmap();
        
        Ok(result)
    }
}

// Uniform buffer structure for matrix dimensions
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatrixDimsUniform {
    rows: u32,
    cols: u32,
    padding: [u32; 2], // Ensure 16-byte alignment
}

impl<T: Float + Send + Sync + Pod + Clone> ComputeBackend<T> for WebGpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::WebGPU
    }

    fn device_info(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn is_available(&self) -> bool {
        true // If we got this far, WebGPU is available
    }

    async fn initialize(&mut self) -> ComputeResult<()> {
        // Pre-compile common shaders
        self.get_matrix_multiply_pipeline().await?;
        Ok(())
    }

    async fn matrix_vector_multiply(
        &self,
        matrix: &[T],
        vector: &[T],
        matrix_dims: MatrixDims,
    ) -> ComputeResult<Vec<T>> {
        if matrix.len() != matrix_dims.size() {
            return Err(ComputeError::invalid_dimensions(
                format!("{}x{}", matrix_dims.rows, matrix_dims.cols),
                format!("{}", matrix.len()),
            ));
        }
        
        if vector.len() != matrix_dims.cols {
            return Err(ComputeError::invalid_dimensions(
                format!("{}", matrix_dims.cols),
                format!("{}", vector.len()),
            ));
        }

        // Create buffers
        let matrix_buffer = self.create_buffer_with_data(matrix, wgpu::BufferUsages::STORAGE).await?;
        let vector_buffer = self.create_buffer_with_data(vector, wgpu::BufferUsages::STORAGE).await?;
        
        let result_size = matrix_dims.rows * std::mem::size_of::<T>();
        let result_buffer = self.memory_manager.create_storage_buffer(result_size as u64, Some("result_buffer"))?;
        
        // Create uniform buffer for dimensions
        let dims_uniform = MatrixDimsUniform {
            rows: matrix_dims.rows as u32,
            cols: matrix_dims.cols as u32,
            padding: [0, 0],
        };
        let dims_buffer = self.memory_manager.create_uniform_buffer(
            std::mem::size_of::<MatrixDimsUniform>() as u64,
            Some("dims_buffer")
        )?;
        self.memory_manager.write_buffer_data(&dims_buffer, bytemuck::cast_slice(&[dims_uniform]))?;

        // Get pipeline (this is a mutable operation, so we need to handle it carefully)
        // For now, we'll use a simpler approach and create the shader inline
        let shader_source = include_str!("shaders/matrix_operations.wgsl");
        let shader = self.device.create_compute_shader(&shader_source, Some("matrix_multiply_shader"))?;
        
        let bind_group_layout = self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matrix_multiply_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matrix_multiply_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matrix_multiply_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "matrix_vector_multiply",
        });

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matrix_multiply_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vector_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matrix_multiply_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matrix_multiply_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroup size
            let workgroup_size = 64; // Local workgroup size from shader
            let num_workgroups = matrix_dims.rows.div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        self.device.submit([encoder.finish()]);

        // Read result back
        let result = self.read_buffer_data::<T>(&result_buffer, matrix_dims.rows).await?;

        // Return buffers to pool
        self.memory_manager.return_buffer(matrix_buffer);
        self.memory_manager.return_buffer(vector_buffer);
        self.memory_manager.return_buffer(result_buffer);
        self.memory_manager.return_buffer(dims_buffer);

        Ok(result)
    }

    async fn batch_matrix_vector_multiply(
        &self,
        matrix: &[T],
        vectors: &[&[T]],
        matrix_dims: MatrixDims,
    ) -> ComputeResult<Vec<Vec<T>>> {
        // For now, process sequentially
        // A full implementation would batch the operations
        let mut results = Vec::with_capacity(vectors.len());
        
        for vector in vectors {
            let result = self.matrix_vector_multiply(matrix, vector, matrix_dims).await?;
            results.push(result);
        }
        
        Ok(results)
    }

    async fn activation_function(
        &self,
        input: &[T],
        function: ActivationFunction,
    ) -> ComputeResult<Vec<T>> {
        let (shader_source, entry_point) = match function {
            ActivationFunction::ReLU => (include_str!("shaders/activation_functions.wgsl"), "relu"),
            ActivationFunction::Sigmoid => (include_str!("shaders/activation_functions.wgsl"), "sigmoid_fast"),
            ActivationFunction::Tanh => (include_str!("shaders/activation_functions.wgsl"), "tanh_stable"),
            _ => {
                // Fall back to CPU for unsupported activation functions
                let cpu_backend = crate::webgpu::backend::CpuBackend::new();
                return cpu_backend.activation_function(input, function).await;
            }
        };

        // Create buffers
        let input_buffer = self.create_buffer_with_data(input, wgpu::BufferUsages::STORAGE).await?;
        let output_size = input.len() * std::mem::size_of::<T>();
        let output_buffer = self.memory_manager.create_storage_buffer(output_size as u64, Some("activation_output"))?;

        // Create shader and pipeline
        let shader = self.device.create_compute_shader(&shader_source, Some("activation_shader"))?;
        
        let bind_group_layout = self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("activation_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("activation_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("activation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
        });

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("activation_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("activation_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("activation_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroup size
            let workgroup_size = 64;
            let num_workgroups = input.len().div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        self.device.submit([encoder.finish()]);

        // Read result back
        let result = self.read_buffer_data::<T>(&output_buffer, input.len()).await?;

        // Return buffers to pool
        self.memory_manager.return_buffer(input_buffer);
        self.memory_manager.return_buffer(output_buffer);

        Ok(result)
    }

    async fn compute_gradients(
        &self,
        _weights: &[T],
        activations: &[T],
        errors: &[T],
        weight_dims: MatrixDims,
    ) -> ComputeResult<Vec<T>> {
        // Create buffers
        let input_buffer = self.create_buffer_with_data(activations, wgpu::BufferUsages::STORAGE).await?;
        let error_buffer = self.create_buffer_with_data(errors, wgpu::BufferUsages::STORAGE).await?;
        
        let gradient_size = weight_dims.size() * std::mem::size_of::<T>();
        let gradient_buffer = self.memory_manager.create_storage_buffer(gradient_size as u64, Some("weight_gradients"))?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct WeightGradientUniforms {
            input_size: u32,
            output_size: u32,
            batch_size: u32,
            learning_rate: f32,
        }

        let uniforms = WeightGradientUniforms {
            input_size: weight_dims.cols as u32,
            output_size: weight_dims.rows as u32,
            batch_size: 1, // Single sample for now
            learning_rate: 1.0, // Gradient only, no learning rate application here
        };

        let uniforms_buffer = self.memory_manager.create_uniform_buffer(
            std::mem::size_of::<WeightGradientUniforms>() as u64,
            Some("gradient_uniforms")
        )?;
        self.memory_manager.write_buffer_data(&uniforms_buffer, bytemuck::cast_slice(&[uniforms]))?;

        // Load shader
        let shader_source = include_str!("shaders/gradient_computation.wgsl");
        let shader = self.device.create_compute_shader(&shader_source, Some("gradient_computation"))?;
        
        let bind_group_layout = self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gradient_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gradient_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gradient_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "compute_weight_gradients",
        });

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gradient_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: error_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gradient_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniforms_buffer.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gradient_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gradient_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with 2D workgroups for matrix operation
            let workgroup_size = 16; // From shader
            let x_workgroups = weight_dims.cols.div_ceil(workgroup_size);
            let y_workgroups = weight_dims.rows.div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(x_workgroups as u32, y_workgroups as u32, 1);
        }

        self.device.submit([encoder.finish()]);

        // Read result back
        let result = self.read_buffer_data::<T>(&gradient_buffer, weight_dims.size()).await?;

        // Return buffers to pool
        self.memory_manager.return_buffer(input_buffer);
        self.memory_manager.return_buffer(error_buffer);
        self.memory_manager.return_buffer(gradient_buffer);
        self.memory_manager.return_buffer(uniforms_buffer);

        Ok(result)
    }

    async fn update_weights(
        &self,
        weights: &mut [T],
        gradients: &[T],
        learning_rate: T,
    ) -> ComputeResult<()> {
        if weights.len() != gradients.len() {
            return Err(ComputeError::invalid_dimensions(
                format!("{}", weights.len()),
                format!("{}", gradients.len()),
            ));
        }

        // Create buffers
        let weights_buffer = self.create_buffer_with_data(weights, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC).await?;
        let gradients_buffer = self.create_buffer_with_data(gradients, wgpu::BufferUsages::STORAGE).await?;
        
        // Create uniform buffer for optimizer parameters
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct OptimizerUniforms {
            weight_count: u32,
            learning_rate: f32,
            momentum: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            weight_decay: f32,
            optimizer_type: u32,
        }

        let uniforms = OptimizerUniforms {
            weight_count: weights.len() as u32,
            learning_rate: learning_rate.to_f32().unwrap_or(0.01),
            momentum: 0.0, // No momentum for simple SGD
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0, // No weight decay for now
            optimizer_type: 0, // Simple SGD
        };

        let uniforms_buffer = self.memory_manager.create_uniform_buffer(
            std::mem::size_of::<OptimizerUniforms>() as u64,
            Some("optimizer_uniforms")
        )?;
        self.memory_manager.write_buffer_data(&uniforms_buffer, bytemuck::cast_slice(&[uniforms]))?;

        // Create dummy momentum/velocity buffers (not used for simple SGD)
        let dummy_buffer = self.memory_manager.create_storage_buffer(
            (weights.len() * std::mem::size_of::<T>()) as u64,
            Some("dummy_buffer")
        )?;

        // Load shader
        let shader_source = include_str!("shaders/gradient_computation.wgsl");
        let shader = self.device.create_compute_shader(&shader_source, Some("weight_update"))?;
        
        let bind_group_layout = self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("weight_update_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("weight_update_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("weight_update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "simple_weight_update",
        });

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("weight_update_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gradients_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniforms_buffer.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("weight_update_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("weight_update_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroup size
            let workgroup_size = 256; // From shader
            let num_workgroups = weights.len().div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        self.device.submit([encoder.finish()]);

        // Read updated weights back
        let updated_weights = self.read_buffer_data::<T>(&weights_buffer, weights.len()).await?;
        weights.copy_from_slice(&updated_weights);

        // Return buffers to pool
        self.memory_manager.return_buffer(weights_buffer);
        self.memory_manager.return_buffer(gradients_buffer);
        self.memory_manager.return_buffer(uniforms_buffer);
        self.memory_manager.return_buffer(dummy_buffer);

        Ok(())
    }

    async fn vector_add(&self, a: &[T], b: &[T]) -> ComputeResult<Vec<T>> {
        if a.len() != b.len() {
            return Err(ComputeError::invalid_dimensions(
                format!("{}", a.len()),
                format!("{}", b.len()),
            ));
        }

        // Create buffers
        let a_buffer = self.create_buffer_with_data(a, wgpu::BufferUsages::STORAGE).await?;
        let b_buffer = self.create_buffer_with_data(b, wgpu::BufferUsages::STORAGE).await?;
        let result_size = a.len() * std::mem::size_of::<T>();
        let result_buffer = self.memory_manager.create_storage_buffer(result_size as u64, Some("vector_add_result"))?;

        // Create shader and pipeline
        let shader_source = include_str!("shaders/vector_operations.wgsl");
        let shader = self.device.create_compute_shader(&shader_source, Some("vector_add_shader"))?;
        
        // Create uniforms for the vector operation
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct VectorOpUniforms {
            length: u32,
            _padding: [u32; 3],
        }
        
        let uniforms = VectorOpUniforms {
            length: a.len() as u32,
            _padding: [0; 3],
        };
        
        // Create uniform buffer with proper alignment (minimum 256 bytes for wgpu)
        let uniforms_size = std::mem::size_of::<VectorOpUniforms>() as u64;
        let aligned_size = uniforms_size.max(256);
        let uniforms_buffer = self.memory_manager.create_uniform_buffer(
            aligned_size,
            Some("vector_op_uniforms"),
        )?;
        self.memory_manager.write_buffer_data(&uniforms_buffer, bytemuck::cast_slice(&[uniforms]))?;

        let bind_group_layout = self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector_add_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector_add_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vector_add_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "vector_add",
        });

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_add_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniforms_buffer.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vector_add_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vector_add_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroup size
            let workgroup_size = 256; // Must match shader workgroup size
            let num_workgroups = a.len().div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        self.device.submit([encoder.finish()]);

        // Read result back
        let result = self.read_buffer_data::<T>(&result_buffer, a.len()).await?;

        // Return buffers to pool
        self.memory_manager.return_buffer(a_buffer);
        self.memory_manager.return_buffer(b_buffer);
        self.memory_manager.return_buffer(result_buffer);
        self.memory_manager.return_buffer(uniforms_buffer);

        Ok(result)
    }

    async fn vector_scale(&self, vector: &[T], scalar: T) -> ComputeResult<Vec<T>> {
        // Create buffers
        let input_buffer = self.create_buffer_with_data(vector, wgpu::BufferUsages::STORAGE).await?;
        let output_size = vector.len() * std::mem::size_of::<T>();
        let output_buffer = self.memory_manager.create_storage_buffer(output_size as u64, Some("vector_scale_output"))?;

        // Create uniform buffer for scalar
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct ScaleParams {
            scalar: f32,
        }

        let params = ScaleParams {
            scalar: scalar.to_f32().unwrap_or(1.0),
        };

        let params_buffer = self.memory_manager.create_uniform_buffer(
            std::mem::size_of::<ScaleParams>() as u64,
            Some("scale_params")
        )?;
        self.memory_manager.write_buffer_data(&params_buffer, bytemuck::cast_slice(&[params]))?;

        // Create shader and pipeline
        let shader_source = include_str!("shaders/matrix_operations.wgsl");
        let shader = self.device.create_compute_shader(&shader_source, Some("vector_scale_shader"))?;
        
        let bind_group_layout = self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector_scale_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector_scale_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vector_scale_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "elementwise_multiply",
        });

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_scale_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vector_scale_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vector_scale_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroup size
            let workgroup_size = 64;
            let num_workgroups = vector.len().div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        self.device.submit([encoder.finish()]);

        // Read result back
        let result = self.read_buffer_data::<T>(&output_buffer, vector.len()).await?;

        // Return buffers to pool
        self.memory_manager.return_buffer(input_buffer);
        self.memory_manager.return_buffer(output_buffer);
        self.memory_manager.return_buffer(params_buffer);

        Ok(result)
    }

    fn estimate_performance(&self, problem_size: usize) -> f32 {
        // GPU performance is better for larger problems
        let base_performance = 1e12; // 1 teraop/sec for large GPU
        let efficiency = if problem_size > 10000 {
            1.0
        } else if problem_size > 1000 {
            0.5
        } else {
            0.1
        };
        
        (base_performance * efficiency) / (problem_size as f32)
    }

    fn memory_requirements(&self, problem_size: usize) -> u64 {
        // GPU needs more memory for buffers
        (problem_size * 4 * std::mem::size_of::<T>()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_webgpu_backend_creation() {
        // This test will only pass if WebGPU is available
        match WebGpuBackend::new().await {
            Ok(backend) => {
                assert_eq!(<WebGpuBackend as ComputeBackend<f32>>::backend_type(&backend), BackendType::WebGPU);
                assert!(<WebGpuBackend as ComputeBackend<f32>>::is_available(&backend));
            }
            Err(_) => {
                // WebGPU not available in test environment, skip
                println!("WebGPU not available, skipping test");
            }
        }
    }
}