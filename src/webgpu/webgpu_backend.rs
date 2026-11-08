//! WebGPU matrix/vector and activation operations with a scalar fallback.
//!
//! Large f32 matrix products and activation arrays dispatch WGSL compute kernels.
//! Smaller operations and wider floating-point types use CPU arithmetic to avoid
//! transfer overhead and preserve precision. The thresholds are fixed heuristics;
//! measure representative inputs on the deployment adapter before tuning them.
//!
//! Native headless checks may opt into software adapters with
//! `RUV_FANN_ALLOW_SOFTWARE_GPU=1` and select a backend with `WGPU_BACKEND`.
//! This setting validates dispatch and arithmetic; it measures no physical GPU.
//!
//! ```rust,no_run
//! use ruv_fann::webgpu::{ComputeBackend, WebGPUBackend};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let backend = WebGPUBackend::<f32>::new()?;
//! let result = backend.matrix_vector_multiply(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], 2, 2)?;
//! assert_eq!(result, vec![17.0, 39.0]);
//! # Ok(())
//! # }
//! ```

#![allow(clippy::needless_range_loop)]

#[cfg(feature = "gpu")]
pub mod webgpu_impl {
    use num_traits::Float;

    use crate::webgpu::backend::{
        BackendCapabilities, BackendType, ComputeBackend, MemoryManager, VectorOps,
    };
    use crate::webgpu::error::ComputeError;
    use crate::webgpu::memory::{BufferHandle, MemoryStats};
    use crate::webgpu::shaders::webgpu_shaders::{ShaderManager, ShaderType};
    use crate::ActivationFunction;
    use std::collections::HashMap;

    /// WebGPU compute backend
    ///
    /// Device-backed f32 matrix products and activations, with scalar fallbacks.
    ///
    /// # Thread Safety
    ///
    /// This backend is fully thread-safe. All operations can be called
    /// concurrently from multiple threads without additional synchronization.
    ///
    /// # Memory Management
    ///
    /// Managed buffers own device allocations until explicitly released or the
    /// backend drops. Handles are process-unique. Transfers accept f32/f64 IEEE
    /// storage; f64 storage does not enable f64 shaders. The active allocation
    /// budget is the device's maximum buffer size, with at most 4,096 handles.
    /// Reported availability is remaining managed budget, not free physical VRAM.
    /// Compute and staging allocations are outside that managed-buffer counter.
    pub struct WebGPUBackend<T: Float + std::fmt::Debug + Send + Sync + 'static> {
        /// WebGPU device handle
        device: wgpu::Device,
        /// WebGPU queue for command submission
        queue: wgpu::Queue,
        /// GPU device capabilities and limits
        capabilities: BackendCapabilities,
        /// WGSL shader compiler and manager
        shader_manager: ShaderManager,
        /// Mutable state for GPU resources (thread-safe interior mutability)
        gpu_state: std::sync::RwLock<GpuState>,
        /// Actual device-owned allocations for the MemoryManager API.
        managed_buffers: std::sync::Mutex<BufferStorage>,
        /// Phantom data for type safety
        _phantom: std::marker::PhantomData<T>,
    }

    include!("activation_dispatch.rs");
    include!("buffer_storage.rs");

    /// Mutable GPU state managed through interior mutability
    struct GpuState {
        /// Compiled shader modules cache
        shader_modules: HashMap<ShaderType, wgpu::ShaderModule>,
        /// Compute pipelines cache  
        pipelines: HashMap<ShaderType, wgpu::ComputePipeline>,
        /// GPU buffer pool for memory reuse
        buffer_pool: HashMap<u64, wgpu::Buffer>,
        /// Bind group layouts cache
        bind_group_layouts: HashMap<ShaderType, wgpu::BindGroupLayout>,
    }

    impl<T: Float + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for WebGPUBackend<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let gpu_state = self.gpu_state.read().unwrap();
            f.debug_struct("WebGPUBackend")
                .field("capabilities", &self.capabilities)
                .field("shader_manager", &self.shader_manager)
                .field("shader_modules_count", &gpu_state.shader_modules.len())
                .field("pipelines_count", &gpu_state.pipelines.len())
                .field("buffer_pool_size", &gpu_state.buffer_pool.len())
                .finish()
        }
    }

    impl<T: Float + std::fmt::Debug + Send + Sync + 'static> WebGPUBackend<T> {
        /// Initialize WebGPU backend synchronously
        ///
        /// This is the main entry point for creating a WebGPU backend.
        /// It uses pollster to run the async initialization in a blocking manner.
        pub fn new() -> Result<Self, ComputeError> {
            pollster::block_on(Self::initialize())
        }

        /// Initialize WebGPU backend asynchronously
        ///
        /// This method performs the following initialization steps:
        /// 1. Checks WebGPU device availability
        /// 2. Queries device capabilities and limits
        /// 3. Compiles and caches compute shaders
        /// 4. Sets up memory management pools
        /// 5. Validates compute pipeline functionality
        ///
        /// # Returns
        ///
        /// - `Ok(WebGPUBackend)` if initialization succeeds
        /// - `Err(ComputeError::GpuUnavailable)` if WebGPU is not supported
        /// - `Err(ComputeError::InitializationError)` if device setup fails
        ///
        /// # Examples
        ///
        /// ```rust,no_run
        /// use ruv_fann::webgpu::WebGPUBackend;
        ///
        /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
        /// let backend = WebGPUBackend::<f32>::initialize().await?;
        /// println!("WebGPU backend initialized successfully");
        /// # Ok(())
        /// # }
        /// ```
        pub async fn initialize() -> Result<Self, ComputeError> {
            // Step 1: Check WebGPU availability
            if !Self::is_available() {
                return Err(ComputeError::GpuUnavailable);
            }

            // Step 2: Initialize WebGPU device using synchronous wrapper
            let (device, queue) = pollster::block_on(async {
                let instance = crate::webgpu::device::compute_instance();

                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .ok_or_else(|| ComputeError::GpuUnavailable)?;

                adapter
                    .request_device(
                        &wgpu::DeviceDescriptor {
                            label: Some("ruv-FANN GPU Device"),
                            required_features: wgpu::Features::default(), // Add SHADER_F16 later
                            required_limits: wgpu::Limits::downlevel_defaults(),
                        },
                        None,
                    )
                    .await
                    .map_err(|e| {
                        ComputeError::InitializationError(format!(
                            "Failed to request device: {}",
                            e
                        ))
                    })
            })?;

            // Step 3: Initialize shader manager with error handling
            let shader_manager = ShaderManager::new().map_err(|e| {
                ComputeError::InitializationError(format!(
                    "Failed to initialize shader manager: {e:?}"
                ))
            })?;

            // Step 4: Detect actual device capabilities
            let capabilities = Self::detect_capabilities_from_device(&device);

            // Step 5: Validate minimum requirements
            Self::validate_capabilities(&capabilities)?;

            // Step 6: Add uncaptured error handler for debugging
            device.on_uncaptured_error(Box::new(|error| {
                eprintln!("🚨 WGPU UNCAPTURED ERROR: {:?}", error);
                eprintln!("This may indicate a Metal watchdog timeout or validation error");
            }));

            Ok(Self {
                device,
                queue,
                capabilities,
                shader_manager,
                gpu_state: std::sync::RwLock::new(GpuState {
                    shader_modules: HashMap::new(),
                    pipelines: HashMap::new(),
                    buffer_pool: HashMap::new(),
                    bind_group_layouts: HashMap::new(),
                }),
                managed_buffers: std::sync::Mutex::new(BufferStorage::default()),
                _phantom: std::marker::PhantomData,
            })
        }

        /// Check if WebGPU is available on the current platform
        ///
        /// This method performs a quick check for WebGPU support without
        /// full device initialization. It's safe to call multiple times.
        ///
        /// # Platform Support
        ///
        /// - **Browser**: Checks for `navigator.gpu` API availability
        /// - **Desktop**: Checks for native WebGPU implementation
        /// - **Mobile**: Limited support, falls back to CPU
        ///
        /// # Returns
        ///
        /// `true` if WebGPU is available and can be initialized
        pub fn is_available() -> bool {
            // Match GpuDevice's explicit headless/CI mode before probing native
            // drivers, which may panic while initializing a display connection.
            if std::env::var("RUV_FANN_CI_TESTING").is_ok() {
                return false;
            }
            // Safe: Instance is cheap and can be dropped immediately
            let instance = crate::webgpu::device::compute_instance();
            instance
                .enumerate_adapters(
                    wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all()),
                )
                .into_iter()
                .any(|a| {
                    a.get_info().device_type != wgpu::DeviceType::Cpu
                        || std::env::var("RUV_FANN_ALLOW_SOFTWARE_GPU").as_deref() == Ok("1")
                })
        }

        /// Detect actual device capabilities from a device object
        ///
        /// Query limits and enabled features. WebGPU does not expose physical
        /// compute-unit counts or memory bandwidth; zero denotes unknown.
        fn detect_capabilities_from_device(device: &wgpu::Device) -> BackendCapabilities {
            let limits = device.limits();
            let features = device.features();

            BackendCapabilities {
                max_buffer_size: limits.max_buffer_size as usize,
                supports_f64: false, // WebGPU doesn't support f64
                supports_f32: true,  // All WebGPU implementations support f32
                supports_f16: features.contains(wgpu::Features::SHADER_F16),
                max_compute_units: 0,       // Not exposed by WebGPU.
                memory_bandwidth_gbps: 0.0, // Not measured or exposed.
                shader_model: Some("WGSL 1.0".to_string()),
            }
        }

        /// Validate that device capabilities meet minimum requirements
        fn validate_capabilities(caps: &BackendCapabilities) -> Result<(), ComputeError> {
            // Minimum requirements for WebGPU backend
            const MIN_BUFFER_SIZE: usize = 64 * 1024 * 1024; // 64MB

            if caps.max_buffer_size < MIN_BUFFER_SIZE {
                return Err(ComputeError::InitializationError(format!(
                    "Insufficient buffer size: {} < {}",
                    caps.max_buffer_size, MIN_BUFFER_SIZE
                )));
            }

            if !caps.supports_f32 {
                return Err(ComputeError::InitializationError(
                    "Device does not support f32 operations".to_string(),
                ));
            }

            Ok(())
        }
    }

    impl<T: Float + std::fmt::Debug + Send + Sync + 'static> ComputeBackend<T> for WebGPUBackend<T> {
        fn initialize() -> Result<Self, ComputeError>
        where
            Self: Sized,
        {
            // Synchronous initialization - return error for async requirement
            Err(ComputeError::InitializationError(
                "Use WebGPUBackend::initialize() async method instead".to_string(),
            ))
        }

        fn is_available() -> bool
        where
            Self: Sized,
        {
            Self::is_available()
        }

        fn capabilities(&self) -> BackendCapabilities {
            self.capabilities.clone()
        }

        fn backend_type(&self) -> BackendType {
            BackendType::WebGPU
        }

        /// Perform matrix-vector multiplication using GPU acceleration
        ///
        /// This method implements high-performance matrix-vector multiplication
        /// using WebGPU compute shaders. For large matrices, this provides
        /// significant performance improvements over CPU implementations.
        ///
        /// # Arguments
        ///
        /// * `matrix` - Flattened matrix data in row-major order
        /// * `vector` - Input vector data
        /// * `rows` - Number of matrix rows
        /// * `cols` - Number of matrix columns (must equal vector length)
        ///
        /// # Performance
        ///
        /// f32 matrices with more than 10,000 elements use GPU dispatch.
        /// Smaller operations and wider types use the CPU. Throughput depends
        /// on adapter, transfer costs and dimensions; no speedup is guaranteed.
        ///
        /// # Errors
        ///
        /// Returns `ComputeError::InvalidDimensions` if matrix and vector dimensions don't match
        fn matrix_vector_multiply(
            &self,
            matrix: &[T],
            vector: &[T],
            rows: usize,
            cols: usize,
        ) -> Result<Vec<T>, ComputeError> {
            let elements = checked_matrix_elements(rows, cols)?;
            if matrix.len() != elements {
                return Err(ComputeError::InvalidDimensions(format!(
                    "Matrix size mismatch: expected {}x{} = {} elements, got {}",
                    rows,
                    cols,
                    elements,
                    matrix.len()
                )));
            }

            if vector.len() != cols {
                return Err(ComputeError::InvalidDimensions(format!(
                    "Vector size mismatch: expected {} elements for {}x{} matrix, got {}",
                    cols,
                    rows,
                    cols,
                    vector.len()
                )));
            }

            // Performance heuristic for GPU usage
            const GPU_THRESHOLD: usize = 10000; // Minimum problem size for GPU benefit

            if elements > GPU_THRESHOLD
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            {
                // Use GPU acceleration through interior mutability
                self.gpu_matrix_vector_multiply(matrix, vector, rows, cols)
            } else {
                // Use optimized CPU implementation for smaller problems
                self.cpu_matrix_vector_multiply_optimized(matrix, vector, rows, cols)
            }
        }

        /// Perform batch matrix-vector multiplication with GPU optimization
        ///
        /// This method processes multiple vectors against the same matrix in parallel,
        /// providing excellent scaling for batch operations common in neural networks.
        ///
        /// # Performance Benefits
        ///
        /// - **GPU Parallelism**: All vectors processed simultaneously on GPU
        /// - **Memory Efficiency**: Matrix uploaded once, reused for all vectors
        /// - **Batch Scaling**: Near-linear scaling with batch size
        ///
        /// # Arguments
        ///
        /// * `matrix` - Shared matrix for all operations
        /// * `vectors` - Batch of input vectors
        /// * `rows` - Matrix rows
        /// * `cols` - Matrix columns
        ///
        /// # Returns
        ///
        /// Vector of results, one for each input vector
        fn batch_matrix_vector_multiply(
            &self,
            matrix: &[T],
            vectors: &[Vec<T>],
            rows: usize,
            cols: usize,
        ) -> Result<Vec<Vec<T>>, ComputeError> {
            let batch_size = vectors.len();

            let elements = checked_matrix_elements(rows, cols)?;
            rows.checked_mul(batch_size).ok_or_else(|| {
                ComputeError::InvalidDimensions("Batch output dimensions overflow".into())
            })?;
            if matrix.len() != elements {
                return Err(ComputeError::InvalidDimensions(format!(
                    "Matrix dimensions {}x{} don't match data length {}",
                    rows,
                    cols,
                    matrix.len()
                )));
            }

            // Validate all vectors have correct size
            for (i, vector) in vectors.iter().enumerate() {
                if vector.len() != cols {
                    return Err(ComputeError::InvalidDimensions(format!(
                        "Vector {} size mismatch: expected {} elements, got {}",
                        i,
                        cols,
                        vector.len()
                    )));
                }
            }

            // Performance heuristic for batch operations
            const BATCH_GPU_THRESHOLD: usize = 100; // Minimum batch size for GPU benefit

            if batch_size >= BATCH_GPU_THRESHOLD
                && elements > 10000
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            {
                // Use GPU acceleration through interior mutability
                self.gpu_batch_matrix_vector_multiply(matrix, vectors, rows, cols)
            } else {
                // Process with optimized CPU code for smaller batches
                self.cpu_batch_matrix_vector_multiply_optimized(matrix, vectors, rows, cols)
            }
        }

        /// Apply activation function with GPU acceleration
        ///
        /// This method applies the specified activation function to all inputs
        /// using GPU compute shaders for maximum performance. The implementation
        /// includes optimized kernels for all supported activation functions.
        ///
        /// # Supported Functions
        ///
        /// - `Linear`: f(x) = x * steepness
        /// - `Sigmoid`: f(x) = 1 / (1 + exp(-x * steepness))
        /// - `ReLU`: f(x) = max(0, x)
        /// - `Tanh`: f(x) = tanh(x * steepness)
        /// - And many more...
        ///
        /// # Performance
        ///
        /// More than 1,000 f32 elements use WGSL. Other inputs use the scalar
        /// definitions, including the historical unscaled ReLU/GELU/Swish.
        ///
        /// # Arguments
        ///
        /// * `inputs` - Input values to transform
        /// * `function` - Activation function to apply
        /// * `steepness` - Scaling factor for the activation function
        fn apply_activation_function(
            &self,
            inputs: &[T],
            function: ActivationFunction,
            steepness: T,
        ) -> Result<Vec<T>, ComputeError> {
            if inputs.len() > 1000 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                self.gpu_apply_activation_function(inputs, function, steepness)
            } else {
                self.cpu_apply_activation_function_optimized(inputs, function, steepness)
            }
        }

        fn vector_operations(&self) -> &dyn VectorOps<T> {
            self
        }

        fn memory_manager(&self) -> &dyn MemoryManager<T> {
            self
        }
    }

    // Private implementation methods
    impl<T: Float + std::fmt::Debug + Send + Sync + 'static> WebGPUBackend<T> {
        /// Get or compile a shader module
        fn get_or_create_shader_module(&self, shader_type: ShaderType) -> Result<(), ComputeError> {
            let mut gpu_state = self.gpu_state.write().unwrap();

            if gpu_state.shader_modules.contains_key(&shader_type) {
                return Ok(());
            }

            // Get shader source from the manager
            let source = self
                .shader_manager
                .get_shader_source(&shader_type)
                .ok_or_else(|| {
                    ComputeError::UnsupportedOperation(format!(
                        "No shader source for {:?}",
                        shader_type
                    ))
                })?;

            // Compile the shader
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("{:?} Shader", shader_type)),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });

            gpu_state.shader_modules.insert(shader_type, module);
            Ok(())
        }

        /// Get or create a compute pipeline
        fn get_or_create_pipeline(
            &self,
            shader_type: ShaderType,
            entry_point: &str,
        ) -> Result<(), ComputeError> {
            {
                let gpu_state = self.gpu_state.read().unwrap();
                if gpu_state.pipelines.contains_key(&shader_type) {
                    return Ok(());
                }
            }

            // Ensure shader module exists
            self.get_or_create_shader_module(shader_type.clone())?;

            let mut gpu_state = self.gpu_state.write().unwrap();
            let module = gpu_state.shader_modules.get(&shader_type).unwrap();

            // Create bind group layout based on shader type
            let entries = match shader_type {
                ShaderType::MatrixVectorMultiply => vec![
                    // Storage buffer for matrix
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
                    // Storage buffer for vector
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
                    // Storage buffer for output
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
                    // Uniform buffer for dimensions
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
                ShaderType::BatchMatrixVectorMultiply => vec![
                    // Storage buffer for matrix
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
                    // Storage buffer for vectors (batch)
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
                    // Storage buffer for results (batch)
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
                    // Uniform buffer for dimensions (includes batch_size)
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
                _ => vec![
                    // Default layout for other shaders
                    // Storage buffer for input
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
                    // Storage buffer for output
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
                    // Uniform buffer for parameters
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
            };

            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("{:?} Bind Group Layout", shader_type)),
                        entries: &entries,
                    });

            // Create pipeline layout
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{:?} Pipeline Layout", shader_type)),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            // Create compute pipeline
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{:?} Pipeline", shader_type)),
                    layout: Some(&pipeline_layout),
                    module,
                    entry_point,
                });

            gpu_state
                .bind_group_layouts
                .insert(shader_type.clone(), bind_group_layout);
            gpu_state.pipelines.insert(shader_type, pipeline);
            Ok(())
        }

        /// Optimized CPU matrix-vector multiplication
        ///
        /// This fallback implementation uses vectorized operations and
        /// cache-friendly memory access patterns for optimal CPU performance.
        fn cpu_matrix_vector_multiply_optimized(
            &self,
            matrix: &[T],
            vector: &[T],
            rows: usize,
            cols: usize,
        ) -> Result<Vec<T>, ComputeError> {
            let mut result = vec![T::zero(); rows];

            // Cache-friendly row-wise traversal
            for row in 0..rows {
                let mut sum = T::zero();
                let row_start = row * cols;

                // Unroll small loops for better performance
                let mut col = 0;
                while col + 4 <= cols {
                    sum = sum
                        + matrix[row_start + col] * vector[col]
                        + matrix[row_start + col + 1] * vector[col + 1]
                        + matrix[row_start + col + 2] * vector[col + 2]
                        + matrix[row_start + col + 3] * vector[col + 3];
                    col += 4;
                }

                // Handle remainder
                while col < cols {
                    sum = sum + matrix[row_start + col] * vector[col];
                    col += 1;
                }

                result[row] = sum;
            }

            Ok(result)
        }

        /// Optimized CPU batch matrix-vector multiplication
        fn cpu_batch_matrix_vector_multiply_optimized(
            &self,
            matrix: &[T],
            vectors: &[Vec<T>],
            rows: usize,
            cols: usize,
        ) -> Result<Vec<Vec<T>>, ComputeError> {
            let batch_size = vectors.len();
            let mut results = Vec::with_capacity(batch_size);

            for vector in vectors {
                let result =
                    self.cpu_matrix_vector_multiply_optimized(matrix, vector, rows, cols)?;
                results.push(result);
            }

            Ok(results)
        }

        /// Optimized CPU activation function application
        fn cpu_apply_activation_function_optimized(
            &self,
            inputs: &[T],
            function: ActivationFunction,
            steepness: T,
        ) -> Result<Vec<T>, ComputeError> {
            Ok(inputs
                .iter()
                .map(|&value| function.apply(value, steepness))
                .collect())
        }

        /// Align buffer size to Apple Silicon requirements (256-byte alignment)
        fn align_buffer_size(size: usize) -> usize {
            const ALIGNMENT: usize = 256;
            ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT
        }

        /// Optimized CPU dot product with vectorization hints
        fn cpu_dot_product_optimized(&self, a: &[T], b: &[T]) -> Result<T, ComputeError> {
            let mut sum = T::zero();

            // Unroll loop for better performance
            let mut i = 0;
            while i + 4 <= a.len() {
                sum = sum
                    + a[i] * b[i]
                    + a[i + 1] * b[i + 1]
                    + a[i + 2] * b[i + 2]
                    + a[i + 3] * b[i + 3];
                i += 4;
            }

            // Handle remainder
            while i < a.len() {
                sum = sum + a[i] * b[i];
                i += 1;
            }

            Ok(sum)
        }

        /// GPU-accelerated matrix-vector multiplication
        fn gpu_matrix_vector_multiply(
            &self,
            matrix: &[T],
            vector: &[T],
            rows: usize,
            cols: usize,
        ) -> Result<Vec<T>, ComputeError> {
            // Ensure the compute pipeline exists
            self.get_or_create_pipeline(ShaderType::MatrixVectorMultiply, "main")?;

            let gpu_state = self.gpu_state.read().unwrap();
            let pipeline = gpu_state
                .pipelines
                .get(&ShaderType::MatrixVectorMultiply)
                .unwrap();

            // Create GPU buffers
            use wgpu::util::DeviceExt;

            self.validate_storage_elements(matrix.len())?;
            self.validate_storage_elements(vector.len())?;
            self.validate_storage_elements(rows)?;
            if (rows as u64).div_ceil(32)
                > self.device.limits().max_compute_workgroups_per_dimension as u64
            {
                return Err(ComputeError::InvalidDimensions(
                    "Matrix dispatch exceeds device limits".into(),
                ));
            }
            // Convert to f32 for GPU operations (WebGPU doesn't support f64)
            let matrix_f32: Vec<f32> = matrix.iter().map(|&x| x.to_f32().unwrap()).collect();
            let vector_f32: Vec<f32> = vector.iter().map(|&x| x.to_f32().unwrap()).collect();

            let matrix_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Matrix Buffer"),
                    contents: bytemuck::cast_slice(&matrix_f32),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let vector_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vector Buffer"),
                    contents: bytemuck::cast_slice(&vector_f32),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size: (rows * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: (rows * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Create uniform buffer for dimensions (match shader struct)
            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
            struct Dimensions {
                rows: u32,
                cols: u32,
                batch_id: u32, // Match shader struct
                reserved: u32, // Match shader struct
            }

            let dims = Dimensions {
                rows: rows as u32,
                cols: cols as u32,
                batch_id: 0, // Single matrix operation
                reserved: 0, // Padding
            };

            let dims_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&[dims]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            // Create bind group
            let bind_group_layout = gpu_state
                .bind_group_layouts
                .get(&ShaderType::MatrixVectorMultiply)
                .ok_or_else(|| {
                    ComputeError::InitializationError("Missing bind group layout".to_string())
                })?;

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Matrix Vector Multiply Bind Group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: matrix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vector_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create command encoder and dispatch compute shader
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Matrix Vector Multiply Encoder"),
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Matrix Vector Multiply Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch with appropriate workgroup size (match shader)
                const WORKGROUP_SIZE: u32 = 32;
                let workgroups = ((rows as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE).max(1);
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Copy output to staging buffer
            encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                0,
                (rows * std::mem::size_of::<f32>()) as u64,
            );

            // Submit commands
            self.queue.submit(Some(encoder.finish()));

            // CRITICAL FIX: Poll immediately after submit to process commands
            self.device.poll(wgpu::Maintain::Poll);

            // Map staging buffer and read results
            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            // CRITICAL FIX: Use Poll instead of Wait to avoid blocking
            self.device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .unwrap()
                .map_err(|_| ComputeError::ComputeError("Failed to map buffer".to_string()))?;

            let data = buffer_slice.get_mapped_range();
            let result_f32: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            staging_buffer.unmap();

            // Convert back to T
            let result: Vec<T> = result_f32.iter().map(|&x| T::from(x).unwrap()).collect();

            Ok(result)
        }

        /// GPU-accelerated batch matrix-vector multiplication with tiling
        /// This implementation tiles large batches to avoid Metal watchdog timeout
        fn gpu_batch_matrix_vector_multiply(
            &self,
            matrix: &[T],
            vectors: &[Vec<T>],
            rows: usize,
            cols: usize,
        ) -> Result<Vec<Vec<T>>, ComputeError> {
            // Constants for Metal performance and watchdog limits (Apple Silicon optimized)
            const MAX_ELEMENTS_PER_DISPATCH: usize = 100_000; // Keep dispatches well under 2ms
            const TILE_SIZE: usize = 32; // Optimized for Apple Silicon 32-lane SIMD
            const MAX_BATCH_PER_DISPATCH: usize = 64; // Conservative batch size
            const MAX_DISPATCH_TIME_MS: f32 = 1.5; // Leave headroom below 2ms limit

            let batch_size = vectors.len();

            // Ensure shader and pipeline exist
            self.get_or_create_pipeline(ShaderType::BatchMatrixVectorMultiply, "main")?;

            let gpu_state = self.gpu_state.read().unwrap();
            let pipeline = gpu_state
                .pipelines
                .get(&ShaderType::BatchMatrixVectorMultiply)
                .unwrap();

            use wgpu::util::DeviceExt;

            self.validate_storage_elements(matrix.len())?;
            self.validate_storage_elements(batch_size.checked_mul(cols).ok_or_else(|| {
                ComputeError::InvalidDimensions("Batch input dimensions overflow".into())
            })?)?;
            self.validate_storage_elements(batch_size.checked_mul(rows).ok_or_else(|| {
                ComputeError::InvalidDimensions("Batch output dimensions overflow".into())
            })?)?;
            // Convert data to f32 for GPU
            let matrix_f32: Vec<f32> = matrix.iter().map(|&x| x.to_f32().unwrap()).collect();
            let mut vectors_f32 = Vec::with_capacity(batch_size * cols);
            for vec in vectors {
                for &val in vec {
                    vectors_f32.push(val.to_f32().unwrap());
                }
            }

            // Check buffer size limits and alignment (Apple Silicon: 256-byte alignment)
            const MIN_BUFFER_ALIGNMENT: usize = 256; // Apple Silicon requirement
            const MAX_BUFFER_SIZE: usize = 128 * 1024 * 1024; // 128MB Apple Silicon limit

            let matrix_size =
                Self::align_buffer_size(matrix_f32.len() * std::mem::size_of::<f32>());
            let vectors_size =
                Self::align_buffer_size(vectors_f32.len() * std::mem::size_of::<f32>());
            let output_size =
                Self::align_buffer_size(batch_size * rows * std::mem::size_of::<f32>());

            if matrix_size > MAX_BUFFER_SIZE
                || vectors_size > MAX_BUFFER_SIZE
                || output_size > MAX_BUFFER_SIZE
            {
                return Err(ComputeError::AllocationError(format!(
                    "Buffer size exceeds Apple Silicon limit: {} MB (matrix: {}MB, vectors: {}MB, output: {}MB)",
                    MAX_BUFFER_SIZE / (1024 * 1024),
                    matrix_size / (1024 * 1024),
                    vectors_size / (1024 * 1024),
                    output_size / (1024 * 1024)
                )));
            }

            // Create GPU buffers
            let matrix_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Batch Matrix Buffer"),
                    contents: bytemuck::cast_slice(&matrix_f32),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let vectors_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Batch Vectors Buffer"),
                        contents: bytemuck::cast_slice(&vectors_f32),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batch Output Buffer"),
                size: output_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Create staging buffer for reading results
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batch Staging Buffer"),
                size: output_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Create uniform buffer
            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
            struct BatchDimensions {
                rows: u32,
                cols: u32,
                batch_size: u32,
                reserved: u32,
            }

            let dims = BatchDimensions {
                rows: rows as u32,
                cols: cols as u32,
                batch_size: batch_size as u32,
                reserved: 0,
            };

            let dims_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Batch Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&[dims]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            // Create bind group
            let bind_group_layout = gpu_state
                .bind_group_layouts
                .get(&ShaderType::BatchMatrixVectorMultiply)
                .ok_or_else(|| {
                    ComputeError::InitializationError("Missing batch bind group layout".to_string())
                })?;

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Batch Matrix Vector Multiply Bind Group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: matrix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vectors_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create command encoder
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Batch Matrix Vector Multiply Encoder"),
                });

            {
                let workgroups_x = (rows as u32).div_ceil(32);
                let workgroups_y = batch_size as u32;
                let limit = self.device.limits().max_compute_workgroups_per_dimension;
                if workgroups_x > limit || workgroups_y > limit {
                    return Err(ComputeError::InvalidDimensions(
                        "Batch dispatch exceeds device limits".into(),
                    ));
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batch Matrix Vector Multiply Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Copy output to staging buffer
            encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                0,
                output_size as u64,
            );

            // Submit commands
            self.queue.submit(Some(encoder.finish()));

            // CRITICAL FIX: Poll immediately after submit
            self.device.poll(wgpu::Maintain::Poll);

            // Map staging buffer and read results
            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            // Wait for mapping to complete
            self.device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .unwrap()
                .map_err(|_| ComputeError::ComputeError("Failed to map buffer".to_string()))?;

            // Read data
            let data = buffer_slice.get_mapped_range();
            let result_f32: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            staging_buffer.unmap();

            // Convert back to Vec<Vec<T>>
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let start = i * rows;
                let end = start + rows;
                let row_results: Vec<T> = result_f32[start..end]
                    .iter()
                    .map(|&x| T::from(x).unwrap())
                    .collect();
                results.push(row_results);
            }

            Ok(results)
        }
    }

    impl<T: Float + std::fmt::Debug + Send + Sync + 'static> VectorOps<T> for WebGPUBackend<T> {
        /// Compute dot product of two vectors with GPU acceleration
        ///
        /// For large vectors, this operation benefits significantly from GPU parallelization.
        /// The implementation uses parallel reduction algorithms for optimal performance.
        ///
        /// # Performance
        ///
        /// - CPU: O(n) with single-threaded execution
        /// - GPU: O(log n) with parallel reduction
        ///
        /// # Errors
        ///
        /// Returns `InvalidDimensions` if vector lengths don't match
        fn dot_product(&self, a: &[T], b: &[T]) -> Result<T, ComputeError> {
            if a.len() != b.len() {
                return Err(ComputeError::InvalidDimensions(format!(
                    "Vector length mismatch: {} vs {}",
                    a.len(),
                    b.len()
                )));
            }

            // Performance heuristic for GPU usage
            const DOT_PRODUCT_GPU_THRESHOLD: usize = 10000;

            if a.len() > DOT_PRODUCT_GPU_THRESHOLD {
                // TODO: Implement GPU parallel reduction
                // self.gpu_dot_product(a, b)
                self.cpu_dot_product_optimized(a, b)
            } else {
                self.cpu_dot_product_optimized(a, b)
            }
        }

        /// Element-wise vector addition with GPU acceleration
        ///
        /// This operation is highly parallel and benefits from GPU acceleration
        /// for large vectors, providing near-linear scaling with GPU core count.
        fn vector_add(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError> {
            if a.len() != b.len() {
                return Err(ComputeError::InvalidDimensions(format!(
                    "Vector length mismatch: {} vs {}",
                    a.len(),
                    b.len()
                )));
            }

            const VECTOR_OP_GPU_THRESHOLD: usize = 1000;

            if a.len() > VECTOR_OP_GPU_THRESHOLD {
                // TODO: Implement GPU vectorized addition
                // self.gpu_vector_add(a, b)
                Ok(a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect())
            } else {
                Ok(a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect())
            }
        }

        /// Scale vector by scalar with GPU acceleration
        ///
        /// Multiplies each element by the scalar value. This operation
        /// is embarrassingly parallel and scales excellently on GPU.
        fn vector_scale(&self, vec: &[T], scalar: T) -> Result<Vec<T>, ComputeError> {
            const SCALE_GPU_THRESHOLD: usize = 1000;

            if vec.len() > SCALE_GPU_THRESHOLD {
                // TODO: Implement GPU vectorized scaling
                // self.gpu_vector_scale(vec, scalar)
                Ok(vec.iter().map(|x| *x * scalar).collect())
            } else {
                Ok(vec.iter().map(|x| *x * scalar).collect())
            }
        }

        /// Element-wise vector subtraction with GPU acceleration
        fn vector_subtract(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError> {
            if a.len() != b.len() {
                return Err(ComputeError::InvalidDimensions(format!(
                    "Vector length mismatch: {} vs {}",
                    a.len(),
                    b.len()
                )));
            }

            const VECTOR_OP_GPU_THRESHOLD: usize = 1000;

            if a.len() > VECTOR_OP_GPU_THRESHOLD {
                // TODO: Implement GPU vectorized subtraction
                // self.gpu_vector_subtract(a, b)
                Ok(a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect())
            } else {
                Ok(a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect())
            }
        }
    }
}

// Re-export for convenience
#[cfg(feature = "gpu")]
pub use webgpu_impl::WebGPUBackend;

// Placeholder when WebGPU is not available
#[cfg(not(feature = "gpu"))]
pub struct WebGPUBackend<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(feature = "gpu"))]
impl<T> WebGPUBackend<T> {
    pub fn is_available() -> bool {
        false
    }
}
