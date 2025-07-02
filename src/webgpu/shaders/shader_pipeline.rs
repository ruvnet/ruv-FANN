// High-Performance Shader Pipeline System for ruv-FANN
// Manages compute shader execution with optimal workgroup sizing and memory management
// Designed for 5-50x performance improvements over CPU implementations

use std::sync::Arc;
use wgpu::{Device, Queue, CommandEncoder, ComputePass, Buffer, BindGroup};
use super::shader_manager::{ShaderManager, ShaderType, CompiledShader};

/// High-level interface for executing neural network compute shaders
pub struct ShaderPipeline {
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader_manager: Arc<ShaderManager>,
    command_encoder: Option<CommandEncoder>,
}

/// Configuration for shader execution
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    pub workgroup_count: (u32, u32, u32),
    pub debug_label: Option<String>,
    pub timing_enabled: bool,
}

/// Results from shader execution
#[derive(Debug)]
pub struct ExecutionResults {
    pub execution_time: Option<std::time::Duration>,
    pub memory_transferred: usize,
    pub workgroups_dispatched: u32,
}

/// Buffer configuration for shader operations
#[derive(Debug)]
pub struct ShaderBuffers {
    pub input_buffers: Vec<Arc<Buffer>>,
    pub output_buffers: Vec<Arc<Buffer>>,
    pub uniform_buffers: Vec<Arc<Buffer>>,
}

impl ShaderPipeline {
    /// Create a new shader pipeline
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        shader_manager: Arc<ShaderManager>,
    ) -> Self {
        Self {
            device,
            queue,
            shader_manager,
            command_encoder: None,
        }
    }
    
    /// Begin a new compute pass for batch operations
    pub fn begin_compute_pass(&mut self, label: Option<&str>) {
        self.command_encoder = Some(self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label,
            }
        ));
    }
    
    /// Execute a shader with the given buffers and configuration
    pub fn execute_shader(
        &mut self,
        shader_type: ShaderType,
        buffers: &ShaderBuffers,
        config: &ExecutionConfig,
    ) -> Result<ExecutionResults, ShaderPipelineError> {
        let compiled_shader = self.shader_manager.get_shader(shader_type)
            .map_err(|e| ShaderPipelineError::ShaderError(e.to_string()))?;
        
        let bind_group = self.create_bind_group(&compiled_shader, buffers)?;
        
        let start_time = if config.timing_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        
        // Execute the compute shader
        self.dispatch_compute_shader(&compiled_shader, &bind_group, config)?;
        
        let execution_time = start_time.map(|start| start.elapsed());
        
        Ok(ExecutionResults {
            execution_time,
            memory_transferred: self.calculate_memory_transfer(buffers),
            workgroups_dispatched: config.workgroup_count.0 * config.workgroup_count.1 * config.workgroup_count.2,
        })
    }
    
    /// Submit all queued compute operations
    pub fn submit(&mut self) -> Result<(), ShaderPipelineError> {
        if let Some(encoder) = self.command_encoder.take() {
            self.queue.submit(std::iter::once(encoder.finish()));
        }
        Ok(())
    }
    
    /// Execute matrix-vector multiplication with automatic workgroup sizing
    pub fn execute_matrix_vector_multiply(
        &mut self,
        matrix_buffer: Arc<Buffer>,
        vector_buffer: Arc<Buffer>,
        result_buffer: Arc<Buffer>,
        uniforms_buffer: Arc<Buffer>,
        rows: u32,
        cols: u32,
    ) -> Result<ExecutionResults, ShaderPipelineError> {
        let workgroup_count = self.calculate_1d_workgroup_count(rows, 256);
        
        let buffers = ShaderBuffers {
            input_buffers: vec![matrix_buffer, vector_buffer],
            output_buffers: vec![result_buffer],
            uniform_buffers: vec![uniforms_buffer],
        };
        
        let config = ExecutionConfig {
            workgroup_count: (workgroup_count, 1, 1),
            debug_label: Some("matrix_vector_multiply".to_string()),
            timing_enabled: true,
        };
        
        self.execute_shader(ShaderType::MatrixVectorMultiply, &buffers, &config)
    }
    
    /// Execute batch matrix-vector multiplication with 2D workgroup layout
    pub fn execute_batch_matrix_vector_multiply(
        &mut self,
        matrix_buffer: Arc<Buffer>,
        vectors_buffer: Arc<Buffer>,
        results_buffer: Arc<Buffer>,
        uniforms_buffer: Arc<Buffer>,
        rows: u32,
        cols: u32,
        batch_size: u32,
    ) -> Result<ExecutionResults, ShaderPipelineError> {
        let workgroup_count_x = self.calculate_1d_workgroup_count(rows, 16);
        let workgroup_count_y = self.calculate_1d_workgroup_count(batch_size, 16);
        
        let buffers = ShaderBuffers {
            input_buffers: vec![matrix_buffer, vectors_buffer],
            output_buffers: vec![results_buffer],
            uniform_buffers: vec![uniforms_buffer],
        };
        
        let config = ExecutionConfig {
            workgroup_count: (workgroup_count_x, workgroup_count_y, 1),
            debug_label: Some("batch_matrix_vector_multiply".to_string()),
            timing_enabled: true,
        };
        
        self.execute_shader(ShaderType::BatchMatrixVectorMultiply, &buffers, &config)
    }
    
    /// Execute activation function with automatic vectorization
    pub fn execute_activation_function(
        &mut self,
        shader_type: ShaderType,
        input_buffer: Arc<Buffer>,
        output_buffer: Arc<Buffer>,
        uniforms_buffer: Arc<Buffer>,
        length: u32,
    ) -> Result<ExecutionResults, ShaderPipelineError> {
        // Calculate workgroups for vectorized processing (4 elements per thread)
        let elements_per_thread = 4;
        let threads_needed = (length + elements_per_thread - 1) / elements_per_thread;
        let workgroup_count = self.calculate_1d_workgroup_count(threads_needed, 256);
        
        let buffers = ShaderBuffers {
            input_buffers: vec![input_buffer],
            output_buffers: vec![output_buffer],
            uniform_buffers: vec![uniforms_buffer],
        };
        
        let config = ExecutionConfig {
            workgroup_count: (workgroup_count, 1, 1),
            debug_label: Some(format!("{:?}_activation", shader_type)),
            timing_enabled: true,
        };
        
        self.execute_shader(shader_type, &buffers, &config)
    }
    
    /// Execute gradient computation with optimal workgroup sizing
    pub fn execute_weight_gradient_computation(
        &mut self,
        input_activations: Arc<Buffer>,
        output_errors: Arc<Buffer>,
        weight_gradients: Arc<Buffer>,
        uniforms_buffer: Arc<Buffer>,
        input_size: u32,
        output_size: u32,
    ) -> Result<ExecutionResults, ShaderPipelineError> {
        let workgroup_count_x = self.calculate_1d_workgroup_count(input_size, 16);
        let workgroup_count_y = self.calculate_1d_workgroup_count(output_size, 16);
        
        let buffers = ShaderBuffers {
            input_buffers: vec![input_activations, output_errors],
            output_buffers: vec![weight_gradients],
            uniform_buffers: vec![uniforms_buffer],
        };
        
        let config = ExecutionConfig {
            workgroup_count: (workgroup_count_x, workgroup_count_y, 1),
            debug_label: Some("weight_gradient_computation".to_string()),
            timing_enabled: true,
        };
        
        self.execute_shader(ShaderType::ComputeWeightGradients, &buffers, &config)
    }
    
    /// Execute optimizer update (SGD with momentum or Adam)
    pub fn execute_optimizer_update(
        &mut self,
        optimizer_type: OptimizerType,
        weights_buffer: Arc<Buffer>,
        gradients_buffer: Arc<Buffer>,
        momentum_buffer: Arc<Buffer>,
        velocity_buffer: Option<Arc<Buffer>>, // For Adam
        uniforms_buffer: Arc<Buffer>,
        weight_count: u32,
    ) -> Result<ExecutionResults, ShaderPipelineError> {
        let workgroup_count = self.calculate_1d_workgroup_count(weight_count, 256);
        
        let mut input_buffers = vec![gradients_buffer, momentum_buffer];
        if let Some(velocity) = velocity_buffer {
            input_buffers.push(velocity);
        }
        
        let buffers = ShaderBuffers {
            input_buffers,
            output_buffers: vec![weights_buffer],
            uniform_buffers: vec![uniforms_buffer],
        };
        
        let shader_type = match optimizer_type {
            OptimizerType::SgdMomentum => ShaderType::SgdMomentumUpdate,
            OptimizerType::Adam => ShaderType::AdamUpdate,
        };
        
        let config = ExecutionConfig {
            workgroup_count: (workgroup_count, 1, 1),
            debug_label: Some(format!("{:?}_optimizer_update", optimizer_type)),
            timing_enabled: true,
        };
        
        self.execute_shader(shader_type, &buffers, &config)
    }
    
    /// Create bind group for shader execution
    fn create_bind_group(
        &self,
        compiled_shader: &CompiledShader,
        buffers: &ShaderBuffers,
    ) -> Result<BindGroup, ShaderPipelineError> {
        let mut entries = Vec::new();
        let mut binding_index = 0u32;
        
        // Add input buffers
        for buffer in &buffers.input_buffers {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_index,
                resource: buffer.as_entire_binding(),
            });
            binding_index += 1;
        }
        
        // Add output buffers
        for buffer in &buffers.output_buffers {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_index,
                resource: buffer.as_entire_binding(),
            });
            binding_index += 1;
        }
        
        // Add uniform buffers
        for buffer in &buffers.uniform_buffers {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_index,
                resource: buffer.as_entire_binding(),
            });
            binding_index += 1;
        }
        
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shader_bind_group"),
            layout: &compiled_shader.bind_group_layout,
            entries: &entries,
        }))
    }
    
    /// Dispatch compute shader with proper workgroup sizing
    fn dispatch_compute_shader(
        &mut self,
        compiled_shader: &CompiledShader,
        bind_group: &BindGroup,
        config: &ExecutionConfig,
    ) -> Result<(), ShaderPipelineError> {
        if self.command_encoder.is_none() {
            self.begin_compute_pass(config.debug_label.as_deref());
        }
        
        let encoder = self.command_encoder.as_mut().unwrap();
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: config.debug_label.as_deref(),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&compiled_shader.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            
            compute_pass.dispatch_workgroups(
                config.workgroup_count.0,
                config.workgroup_count.1,
                config.workgroup_count.2,
            );
        }
        
        Ok(())
    }
    
    /// Calculate optimal 1D workgroup count
    fn calculate_1d_workgroup_count(&self, total_threads: u32, threads_per_workgroup: u32) -> u32 {
        (total_threads + threads_per_workgroup - 1) / threads_per_workgroup
    }
    
    /// Calculate memory transfer amount for performance metrics
    fn calculate_memory_transfer(&self, buffers: &ShaderBuffers) -> usize {
        let mut total = 0;
        
        // Estimate based on buffer count (actual sizes would need to be tracked separately)
        total += buffers.input_buffers.len() * 1024; // Rough estimate
        total += buffers.output_buffers.len() * 1024;
        total += buffers.uniform_buffers.len() * 64;
        
        total
    }
}

/// Supported optimizer types for weight updates
#[derive(Debug, Clone, Copy)]
pub enum OptimizerType {
    SgdMomentum,
    Adam,
}

/// Errors that can occur during shader pipeline execution
#[derive(Debug)]
pub enum ShaderPipelineError {
    ShaderError(String),
    BufferError(String),
    BindGroupCreationFailed,
    ComputePassFailed,
    InvalidWorkgroupConfig(String),
    EncoderNotInitialized,
}

impl std::fmt::Display for ShaderPipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShaderError(s) => write!(f, "Shader error: {}", s),
            Self::BufferError(s) => write!(f, "Buffer configuration error: {}", s),
            Self::BindGroupCreationFailed => write!(f, "Bind group creation failed"),
            Self::ComputePassFailed => write!(f, "Compute pass execution failed"),
            Self::InvalidWorkgroupConfig(s) => write!(f, "Invalid workgroup configuration: {}", s),
            Self::EncoderNotInitialized => write!(f, "Command encoder not initialized"),
        }
    }
}

impl std::error::Error for ShaderPipelineError {}

/// Builder pattern for creating shader pipeline executions
pub struct ShaderExecutionBuilder {
    pipeline: Arc<ShaderPipeline>,
    shader_type: Option<ShaderType>,
    buffers: Option<ShaderBuffers>,
    config: Option<ExecutionConfig>,
}

impl ShaderExecutionBuilder {
    pub fn new(pipeline: Arc<ShaderPipeline>) -> Self {
        Self {
            pipeline,
            shader_type: None,
            buffers: None,
            config: None,
        }
    }
    
    pub fn shader_type(mut self, shader_type: ShaderType) -> Self {
        self.shader_type = Some(shader_type);
        self
    }
    
    pub fn buffers(mut self, buffers: ShaderBuffers) -> Self {
        self.buffers = Some(buffers);
        self
    }
    
    pub fn config(mut self, config: ExecutionConfig) -> Self {
        self.config = Some(config);
        self
    }
    
    pub fn workgroup_count(mut self, count: (u32, u32, u32)) -> Self {
        if let Some(ref mut config) = self.config {
            config.workgroup_count = count;
        } else {
            self.config = Some(ExecutionConfig {
                workgroup_count: count,
                debug_label: None,
                timing_enabled: false,
            });
        }
        self
    }
    
    pub fn debug_label(mut self, label: String) -> Self {
        if let Some(ref mut config) = self.config {
            config.debug_label = Some(label);
        } else {
            self.config = Some(ExecutionConfig {
                workgroup_count: (1, 1, 1),
                debug_label: Some(label),
                timing_enabled: false,
            });
        }
        self
    }
    
    pub fn timing_enabled(mut self, enabled: bool) -> Self {
        if let Some(ref mut config) = self.config {
            config.timing_enabled = enabled;
        } else {
            self.config = Some(ExecutionConfig {
                workgroup_count: (1, 1, 1),
                debug_label: None,
                timing_enabled: enabled,
            });
        }
        self
    }
    
    pub fn execute(self) -> Result<ExecutionResults, ShaderPipelineError> {
        let shader_type = self.shader_type.ok_or_else(|| {
            ShaderPipelineError::BufferError("Shader type not specified".to_string())
        })?;
        
        let buffers = self.buffers.ok_or_else(|| {
            ShaderPipelineError::BufferError("Buffers not specified".to_string())
        })?;
        
        let config = self.config.unwrap_or_else(|| ExecutionConfig {
            workgroup_count: (1, 1, 1),
            debug_label: None,
            timing_enabled: false,
        });
        
        // Note: This would need to be modified to work with the mutable reference requirement
        // In practice, you might want to use interior mutability or a different design
        todo!("Implementation requires refactoring for mutable access")
    }
}

/// Performance monitoring for shader pipeline
pub struct ShaderPerformanceMonitor {
    execution_history: Vec<(ShaderType, ExecutionResults)>,
    total_executions: usize,
    total_execution_time: std::time::Duration,
}

impl ShaderPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            execution_history: Vec::new(),
            total_executions: 0,
            total_execution_time: std::time::Duration::ZERO,
        }
    }
    
    pub fn record_execution(&mut self, shader_type: ShaderType, results: ExecutionResults) {
        if let Some(duration) = results.execution_time {
            self.total_execution_time += duration;
        }
        
        self.total_executions += 1;
        self.execution_history.push((shader_type, results));
        
        // Keep only recent history to prevent unbounded growth
        if self.execution_history.len() > 1000 {
            self.execution_history.remove(0);
        }
    }
    
    pub fn average_execution_time(&self, shader_type: ShaderType) -> Option<std::time::Duration> {
        let executions: Vec<_> = self.execution_history.iter()
            .filter(|(st, _)| *st == shader_type)
            .filter_map(|(_, results)| results.execution_time)
            .collect();
        
        if executions.is_empty() {
            return None;
        }
        
        let total: std::time::Duration = executions.iter().sum();
        Some(total / executions.len() as u32)
    }
    
    pub fn performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_executions: self.total_executions,
            total_execution_time: self.total_execution_time,
            average_execution_time: if self.total_executions > 0 {
                Some(self.total_execution_time / self.total_executions as u32)
            } else {
                None
            },
            unique_shader_types: self.execution_history.iter()
                .map(|(st, _)| st.clone())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }
}

#[derive(Debug)]
pub struct PerformanceSummary {
    pub total_executions: usize,
    pub total_execution_time: std::time::Duration,
    pub average_execution_time: Option<std::time::Duration>,
    pub unique_shader_types: usize,
}

impl Default for ShaderPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execution_config_creation() {
        let config = ExecutionConfig {
            workgroup_count: (16, 16, 1),
            debug_label: Some("test_execution".to_string()),
            timing_enabled: true,
        };
        
        assert_eq!(config.workgroup_count, (16, 16, 1));
        assert!(config.timing_enabled);
    }
    
    #[test]
    fn test_performance_monitor() {
        let mut monitor = ShaderPerformanceMonitor::new();
        
        let results = ExecutionResults {
            execution_time: Some(std::time::Duration::from_millis(10)),
            memory_transferred: 1024,
            workgroups_dispatched: 16,
        };
        
        monitor.record_execution(ShaderType::ReLU, results);
        
        let summary = monitor.performance_summary();
        assert_eq!(summary.total_executions, 1);
        assert_eq!(summary.unique_shader_types, 1);
    }
    
    #[test]
    fn test_optimizer_type_variants() {
        let sgd = OptimizerType::SgdMomentum;
        let adam = OptimizerType::Adam;
        
        assert!(matches!(sgd, OptimizerType::SgdMomentum));
        assert!(matches!(adam, OptimizerType::Adam));
    }
}