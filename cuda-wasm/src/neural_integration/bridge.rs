//! Bridge implementation connecting CUDA-WASM with ruv-FANN
//!
//! This module provides the core bridge functionality that enables
//! seamless integration between CUDA kernels and ruv-FANN neural networks.

use super::{
    BridgeConfig, BufferHandle, CompiledKernel, DeviceInfo, GpuBackendTrait, GpuDevice,
    NeuralIntegrationError, NeuralOperation, NeuralResult, Precision, BindingType,
};
use crate::backend::backend_trait::BackendTrait;
use crate::runtime::Runtime;
use crate::transpiler::Transpiler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// WebGPU backend implementation for neural operations
pub struct WebGpuBackend {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    adapter_info: Option<wgpu::AdapterInfo>,
    runtime: Arc<Runtime>,
    kernel_cache: Arc<RwLock<HashMap<String, CompiledKernel>>>,
    buffer_pool: Arc<Mutex<BufferPool>>,
    config: BridgeConfig,
}

/// Buffer pool for efficient memory management
struct BufferPool {
    buffers: HashMap<BufferHandle, wgpu::Buffer>,
    free_buffers: Vec<(usize, BufferHandle)>, // size, handle
    next_handle: u64,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            free_buffers: Vec::new(),
            next_handle: 1,
        }
    }
    
    fn get_or_create(&mut self, device: &wgpu::Device, size: usize, usage: wgpu::BufferUsages) -> BufferHandle {
        // Try to reuse existing buffer
        if let Some(pos) = self.free_buffers.iter().position(|(s, _)| *s >= size) {
            let (_, handle) = self.free_buffers.remove(pos);
            return handle;
        }
        
        // Create new buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Neural operation buffer"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        
        let handle = BufferHandle(self.next_handle);
        self.next_handle += 1;
        
        self.buffers.insert(handle, buffer);
        handle
    }
    
    fn return_buffer(&mut self, handle: BufferHandle, size: usize) {
        self.free_buffers.push((size, handle));
    }
    
    fn get_buffer(&self, handle: BufferHandle) -> Option<&wgpu::Buffer> {
        self.buffers.get(&handle)
    }
}

impl WebGpuBackend {
    /// Create a new WebGPU backend
    pub fn new(config: &BridgeConfig) -> NeuralResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(|e| {
            NeuralIntegrationError::GpuInitError(format!("Failed to create runtime: {e}"))
        })?);
        
        let mut backend = Self {
            device: None,
            queue: None,
            adapter_info: None,
            runtime,
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            config: config.clone(),
        };
        
        // A failed adapter initialization must propagate to NeuralBridge,
        // which owns the CPU fallback decision; an empty backend is not a GPU.
        if !crate::backend::gpu_enabled() {
            return Err(NeuralIntegrationError::GpuInitError("GPU support requires a gpu/webgpu feature".into()));
        }
        backend.init_webgpu()?;

        Ok(backend)
    }
    
    /// Initialize WebGPU device and queue
    #[cfg(not(target_arch = "wasm32"))]
    fn init_webgpu(&mut self) -> NeuralResult<()> {
        use pollster::FutureExt;
        
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: match self.config.gpu_device {
                GpuDevice::HighPerformance => wgpu::PowerPreference::HighPerformance,
                GpuDevice::LowPower => wgpu::PowerPreference::LowPower,
                _ => wgpu::PowerPreference::default(),
            },
            compatible_surface: None,
            force_fallback_adapter: false,
        }).block_on().ok_or_else(|| {
            NeuralIntegrationError::GpuInitError("No suitable GPU adapter found".to_string())
        })?;
        
        self.adapter_info = Some(adapter.get_info());
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("Neural Bridge Device"),
            },
            None,
        ).block_on().map_err(|e| {
            NeuralIntegrationError::GpuInitError(format!("Failed to create device: {e}"))
        })?;
        
        self.device = Some(device);
        self.queue = Some(queue);
        
        log::info!("WebGPU initialized successfully");
        Ok(())
    }
    
    #[cfg(target_arch = "wasm32")]
    fn init_webgpu(&mut self) -> NeuralResult<()> {
        // WASM initialization will be handled differently
        log::info!("WASM WebGPU initialization deferred to runtime");
        Ok(())
    }
    
    /// Compile a CUDA kernel to WGSL
    fn compile_kernel(&self, cuda_source: &str, name: &str) -> NeuralResult<CompiledKernel> {
        // Check cache first
        if let Ok(cache) = self.kernel_cache.read() {
            if let Some(kernel) = cache.get(name) {
                return Ok(kernel.clone());
            }
        }
        
        // Transpile CUDA to WGSL using our transpiler
        let wgsl_source = self.transpile_cuda_to_wgsl(cuda_source)?;
        
        let kernel = CompiledKernel {
            name: name.to_string(),
            wgsl_source,
            entry_point: "main".to_string(),
            workgroup_size: [64, 1, 1], // Default workgroup size
            bind_group_layout: vec![
                BindingType::Buffer { read_only: true },  // Input buffer
                BindingType::Buffer { read_only: false }, // Output buffer
            ],
        };
        
        // Cache the kernel
        if let Ok(mut cache) = self.kernel_cache.write() {
            cache.insert(name.to_string(), kernel.clone());
        }
        
        Ok(kernel)
    }
    
    /// Transpile CUDA source to WGSL
    fn transpile_cuda_to_wgsl(&self, cuda_source: &str) -> NeuralResult<String> {
        // Create a transpiler instance
        let transpiler = Transpiler::new();
        
        // Parse the CUDA source
        let ast = crate::parser::CudaParser::new()
            .parse(cuda_source)
            .map_err(|e| NeuralIntegrationError::TranspilationError(e.to_string()))?;
        
        // Transpile to WGSL
        let wgsl = transpiler
            .to_wgsl(ast)
            .map_err(|e| NeuralIntegrationError::TranspilationError(e.to_string()))?;
        
        Ok(wgsl)
    }
}

impl GpuBackendTrait for WebGpuBackend {
    fn initialize(&self) -> NeuralResult<()> {
        if self.device.is_some() && self.queue.is_some() {
            Ok(())
        } else {
            Err(NeuralIntegrationError::GpuInitError("Device not initialized".to_string()))
        }
    }
    
    fn is_available(&self) -> bool {
        self.device.is_some() && self.queue.is_some()
    }
    
    fn get_device_info(&self) -> DeviceInfo {
        if let Some(ref info) = self.adapter_info {
            DeviceInfo {
                name: info.name.clone(),
                vendor: format!("{:?}", info.vendor),
                device_type: format!("{:?}", info.device_type),
                memory_size: 0, // WebGPU doesn't expose this directly
                compute_units: 0, // WebGPU doesn't expose this directly
                max_workgroup_size: 256, // Common default
                supports_f16: false, // Conservative default
                supports_f64: false, // WebGPU doesn't support f64 in shaders
            }
        } else {
            DeviceInfo {
                name: "Unknown".to_string(),
                vendor: "Unknown".to_string(),
                device_type: "Unknown".to_string(),
                memory_size: 0,
                compute_units: 0,
                max_workgroup_size: 64,
                supports_f16: false,
                supports_f64: false,
            }
        }
    }
    
    fn create_buffer(&self, size: usize) -> NeuralResult<BufferHandle> {
        let device = self.device.as_ref().ok_or_else(|| {
            NeuralIntegrationError::GpuInitError("Device not initialized".to_string())
        })?;
        
        let mut pool = self.buffer_pool.lock().unwrap();
        let handle = pool.get_or_create(
            device,
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        
        Ok(handle)
    }
    
    fn execute_kernel(&self, kernel: &CompiledKernel, inputs: &[BufferHandle]) -> NeuralResult<BufferHandle> {
        let device = self.device.as_ref().ok_or_else(|| {
            NeuralIntegrationError::GpuInitError("Device not initialized".to_string())
        })?;
        
        let queue = self.queue.as_ref().ok_or_else(|| {
            NeuralIntegrationError::GpuInitError("Queue not initialized".to_string())
        })?;
        
        // Create compute shader
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} shader", kernel.name)),
            source: wgpu::ShaderSource::Wgsl(kernel.wgsl_source.as_str().into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} bind group layout", kernel.name)),
            entries: &kernel.bind_group_layout.iter().enumerate().map(|(i, binding_type)| {
                wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: match binding_type {
                        BindingType::Buffer { read_only } => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: *read_only },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        BindingType::UniformBuffer => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        BindingType::StorageTexture => wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                    },
                    count: None,
                }
            }).collect::<Vec<_>>(),
        });
        
        // Create compute pipeline
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} pipeline layout", kernel.name)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} pipeline", kernel.name)),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: &kernel.entry_point,
        });
        
        // Get input buffers
        let pool = self.buffer_pool.lock().unwrap();
        let input_buffers: Vec<&wgpu::Buffer> = inputs.iter()
            .map(|handle| pool.get_buffer(*handle))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| NeuralIntegrationError::OperationError("Invalid buffer handle".to_string()))?;
        
        // Create output buffer (same size as first input for simplicity)
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output buffer"),
            size: input_buffers[0].size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create bind group
        let mut bind_group_entries = Vec::new();
        for (i, buffer) in input_buffers.iter().enumerate() {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: input_buffers.len() as u32,
            resource: output_buffer.as_entire_binding(),
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} bind group", kernel.name)),
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });
        
        // Execute the compute pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{} encoder", kernel.name)),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} pass", kernel.name)),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with appropriate workgroup count
            let workgroup_count = (input_buffers[0].size() as u32 / 4) / kernel.workgroup_size[0] + 1;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        queue.submit(std::iter::once(encoder.finish()));
        
        // Return handle to output buffer
        // Note: In a real implementation, we'd need to properly manage the output buffer
        // For now, we'll create a new handle
        drop(pool);
        let mut pool = self.buffer_pool.lock().unwrap();
        let handle = BufferHandle(pool.next_handle);
        pool.next_handle += 1;
        pool.buffers.insert(handle, output_buffer);
        
        Ok(handle)
    }
}

/// Extract WGSL from transpiled Rust code
pub fn extract_wgsl_from_rust(rust_code: &str) -> NeuralResult<CompiledKernel> {
    // This is a simplified implementation
    // In a real implementation, we would parse the Rust code and extract WGSL
    
    // For now, we'll generate basic WGSL for common operations
    let wgsl_source = generate_basic_wgsl(rust_code)?;
    
    Ok(CompiledKernel {
        name: "extracted_kernel".to_string(),
        wgsl_source,
        entry_point: "main".to_string(),
        workgroup_size: [64, 1, 1],
        bind_group_layout: vec![
            BindingType::Buffer { read_only: true },
            BindingType::Buffer { read_only: false },
        ],
    })
}

/// Generate basic WGSL for common operations
fn generate_basic_wgsl(rust_code: &str) -> NeuralResult<String> {
    // Analyze the Rust code to determine the operation type
    if rust_code.contains("matrix_multiply") || rust_code.contains("matmul") {
        Ok(include_str!("../webgpu/shaders/matrix_vector_multiply.wgsl").to_string())
    } else if rust_code.contains("vector_add") || rust_code.contains("add") {
        Ok(r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input_a)) {
        return;
    }
    output[index] = input_a[index] + input_b[index];
}
"#.to_string())
    } else if rust_code.contains("sigmoid") {
        Ok(r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    output[index] = 1.0 / (1.0 + exp(-input[index]));
}
"#.to_string())
    } else {
        // Default: copy operation
        Ok(r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input)) {
        return;
    }
    output[index] = input[index];
}
"#.to_string())
    }
}

/// Execute operation on CPU as fallback
pub fn execute_cpu_fallback<T>(operation: NeuralOperation<T>, inputs: &[T]) -> NeuralResult<Vec<T>>
where
    T: Clone + Send + Sync + 'static + num_traits::Float,
{
    match operation {
        NeuralOperation::VectorAdd { size, _phantom } => {
            if inputs.len() < size * 2 {
                return Err(NeuralIntegrationError::OperationError("Insufficient input data".to_string()));
            }
            
            let mut result = Vec::with_capacity(size);
            for i in 0..size {
                result.push(inputs[i] + inputs[i + size]);
            }
            Ok(result)
        }
        
        NeuralOperation::ActivationFunction { function, size, _phantom } => {
            if inputs.len() < size {
                return Err(NeuralIntegrationError::OperationError("Insufficient input data".to_string()));
            }
            
            let mut result = Vec::with_capacity(size);
            for i in 0..size {
                let value = match function {
                    super::ActivationFunction::Sigmoid => {
                        T::one() / (T::one() + (-inputs[i]).exp())
                    }
                    super::ActivationFunction::ReLU => {
                        if inputs[i] > T::zero() { inputs[i] } else { T::zero() }
                    }
                    super::ActivationFunction::Tanh => inputs[i].tanh(),
                    super::ActivationFunction::LeakyReLU => {
                        if inputs[i] > T::zero() { 
                            inputs[i] 
                        } else { 
                            inputs[i] * T::from(0.01).unwrap_or(T::zero())
                        }
                    }
                    super::ActivationFunction::Swish => {
                        inputs[i] * (T::one() / (T::one() + (-inputs[i]).exp()))
                    }
                    super::ActivationFunction::GELU => {
                        // Approximation of GELU
                        let sqrt_2_pi = T::from(0.7978845608).unwrap_or(T::one());
                        let x = inputs[i];
                        x * T::from(0.5).unwrap_or(T::one()) * 
                        (T::one() + (sqrt_2_pi * (x + T::from(0.044715).unwrap_or(T::zero()) * x * x * x)).tanh())
                    }
                };
                result.push(value);
            }
            Ok(result)
        }
        
        NeuralOperation::MatrixMultiply { a_rows, a_cols, b_cols, _phantom } => {
            if inputs.len() < a_rows * a_cols + a_cols * b_cols {
                return Err(NeuralIntegrationError::OperationError("Insufficient input data for matrix multiplication".to_string()));
            }
            
            let mut result = Vec::with_capacity(a_rows * b_cols);
            let matrix_a = &inputs[0..a_rows * a_cols];
            let matrix_b = &inputs[a_rows * a_cols..];
            
            for i in 0..a_rows {
                for j in 0..b_cols {
                    let mut sum = T::zero();
                    for k in 0..a_cols {
                        sum = sum + matrix_a[i * a_cols + k] * matrix_b[k * b_cols + j];
                    }
                    result.push(sum);
                }
            }
            Ok(result)
        }
        
        NeuralOperation::Convolution { channels, kernel_size, stride, _phantom } => {
            // 1D convolution per channel
            // Input layout: [input_data (channels * input_width), kernel_weights (channels * kernel_size)]
            let kernel_total = channels * kernel_size;
            if inputs.len() < kernel_total {
                return Err(NeuralIntegrationError::OperationError(
                    "Insufficient data for convolution".to_string(),
                ));
            }
            let data_len = inputs.len() - kernel_total;
            if channels == 0 {
                return Err(NeuralIntegrationError::OperationError(
                    "Channels must be > 0".to_string(),
                ));
            }
            let input_per_channel = data_len / channels;

            if input_per_channel < kernel_size {
                return Err(NeuralIntegrationError::OperationError(
                    "Input per channel is smaller than kernel size".to_string(),
                ));
            }

            let output_per_channel = (input_per_channel - kernel_size) / stride + 1;
            let mut result = Vec::with_capacity(output_per_channel * channels);

            for c in 0..channels {
                let data_start = c * input_per_channel;
                let kernel_start = data_len + c * kernel_size;

                for out_idx in 0..output_per_channel {
                    let start = out_idx * stride;
                    let mut sum = T::zero();
                    for k in 0..kernel_size {
                        sum = sum + inputs[data_start + start + k] * inputs[kernel_start + k];
                    }
                    result.push(sum);
                }
            }
            Ok(result)
        }

        NeuralOperation::ForwardPropagation { layer_sizes, _phantom } => {
            // Multi-layer forward propagation: weights * inputs + biases per layer
            // Input layout: [input_data, weights_layer0, biases_layer0, weights_layer1, biases_layer1, ...]
            if layer_sizes.len() < 2 {
                return Err(NeuralIntegrationError::OperationError(
                    "Need at least 2 layer sizes for forward propagation".to_string(),
                ));
            }

            let input_size = layer_sizes[0];
            if inputs.len() < input_size {
                return Err(NeuralIntegrationError::OperationError(
                    "Insufficient input data for forward propagation".to_string(),
                ));
            }

            let mut current = inputs[..input_size].to_vec();
            let mut offset = input_size;

            for layer_idx in 0..layer_sizes.len() - 1 {
                let in_size = layer_sizes[layer_idx];
                let out_size = layer_sizes[layer_idx + 1];
                let weight_count = in_size * out_size;

                if inputs.len() < offset + weight_count + out_size {
                    return Err(NeuralIntegrationError::OperationError(
                        format!("Insufficient data for layer {} forward propagation", layer_idx),
                    ));
                }

                let weights_start = offset;
                let bias_start = offset + weight_count;

                let mut next = Vec::with_capacity(out_size);
                for j in 0..out_size {
                    let mut sum = inputs[bias_start + j]; // bias
                    for i in 0..in_size {
                        sum = sum + current[i] * inputs[weights_start + i * out_size + j];
                    }
                    next.push(sum);
                }

                current = next;
                offset += weight_count + out_size;
            }

            Ok(current)
        }

        NeuralOperation::BackwardPropagation { layer_sizes, _phantom } => {
            // Gradient computation: dL/dx = W^T * dL/dy
            // Input layout: [input_data, weights, output_gradients]
            // Uses first and last layer sizes for a single-step gradient computation
            if layer_sizes.len() < 2 {
                return Err(NeuralIntegrationError::OperationError(
                    "Need at least 2 layer sizes for backward propagation".to_string(),
                ));
            }

            let input_size = layer_sizes[0];
            let output_size = *layer_sizes.last().unwrap();
            let weight_count = input_size * output_size;
            let grad_start = input_size + weight_count;

            if inputs.len() < grad_start + output_size {
                return Err(NeuralIntegrationError::OperationError(
                    "Insufficient data for backward propagation".to_string(),
                ));
            }

            // Compute input gradients: dL/dx = W^T * dL/dy
            let mut result = Vec::with_capacity(input_size);
            for i in 0..input_size {
                let mut sum = T::zero();
                for j in 0..output_size {
                    sum = sum + inputs[input_size + i * output_size + j] * inputs[grad_start + j];
                }
                result.push(sum);
            }
            Ok(result)
        }

        NeuralOperation::Custom { name, .. } => {
            Err(NeuralIntegrationError::OperationError(
                format!("CPU fallback not available for custom kernel: {}", name),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_vector_add() {
        let operation = NeuralOperation::VectorAdd { size: 3, _phantom: std::marker::PhantomData };
        let inputs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = execute_cpu_fallback(operation, &inputs).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }
    
    #[test]
    fn test_cpu_sigmoid() {
        let operation = NeuralOperation::ActivationFunction { 
            function: super::super::ActivationFunction::Sigmoid, 
            size: 3,
            _phantom: std::marker::PhantomData 
        };
        let inputs = vec![0.0f32, 1.0, -1.0];
        let result = execute_cpu_fallback(operation, &inputs).unwrap();
        
        // Check that sigmoid(0) ≈ 0.5
        assert!((result[0] - 0.5).abs() < 1e-6);
        // Check that sigmoid(1) > 0.5
        assert!(result[1] > 0.5);
        // Check that sigmoid(-1) < 0.5
        assert!(result[2] < 0.5);
    }
    
    #[test]
    fn test_wgsl_generation() {
        let rust_code = "fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> { ... }";
        let wgsl = generate_basic_wgsl(rust_code).unwrap();
        assert!(wgsl.contains("vector_add") || wgsl.contains("input_a"));
        assert!(wgsl.contains("@compute"));
    }
}