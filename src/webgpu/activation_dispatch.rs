// Included in webgpu_impl to share its private pipeline/resource state.
fn checked_matrix_elements(rows: usize, cols: usize) -> Result<usize, ComputeError> {
    if rows == 0 || cols == 0 {
        return Err(ComputeError::InvalidDimensions(
            "Matrix dimensions must be positive".into(),
        ));
    }
    rows.checked_mul(cols)
        .ok_or_else(|| ComputeError::InvalidDimensions("Matrix dimensions overflow".into()))
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> WebGPUBackend<T> {
    fn validate_storage_elements(&self, elements: usize) -> Result<(), ComputeError> {
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| ComputeError::AllocationError("Buffer byte count overflows".into()))?;
        let limits = self.device.limits();
        if bytes as u64 > limits.max_buffer_size
            || bytes as u64 > limits.max_storage_buffer_binding_size as u64
            || elements > u32::MAX as usize
        {
            return Err(ComputeError::AllocationError(
                "Storage buffer exceeds device limits".into(),
            ));
        }
        Ok(())
    }

    fn gpu_apply_activation_function(
        &self,
        inputs: &[T],
        function: ActivationFunction,
        steepness: T,
    ) -> Result<Vec<T>, ComputeError> {
        use wgpu::util::DeviceExt;
        self.validate_storage_elements(inputs.len())?;
        let workgroups = (inputs.len() as u32).div_ceil(256);
        if workgroups > self.device.limits().max_compute_workgroups_per_dimension {
            return Err(ComputeError::InvalidDimensions(
                "Activation dispatch exceeds device limits".into(),
            ));
        }
        let (shader, entry) = match function {
            ActivationFunction::Linear => (ShaderType::ActivationLinear, "linear_main"),
            ActivationFunction::Threshold => (ShaderType::ActivationThreshold, "threshold_main"),
            ActivationFunction::ThresholdSymmetric => (
                ShaderType::ActivationThresholdSymmetric,
                "threshold_symmetric_main",
            ),
            ActivationFunction::Sigmoid => (ShaderType::ActivationSigmoid, "sigmoid_main"),
            ActivationFunction::Tanh | ActivationFunction::SigmoidSymmetric => {
                (ShaderType::ActivationTanh, "tanh_main")
            }
            ActivationFunction::ReLU => (ShaderType::ActivationReLU, "relu_main"),
            ActivationFunction::ReLULeaky | ActivationFunction::LeakyRelu => {
                (ShaderType::ActivationLeakyReLU, "leaky_relu_main")
            }
            ActivationFunction::Gaussian => (ShaderType::ActivationGaussian, "gaussian_main"),
            ActivationFunction::GaussianSymmetric => (
                ShaderType::ActivationGaussianSymmetric,
                "gaussian_symmetric_main",
            ),
            ActivationFunction::Elliot => (ShaderType::ActivationElliott, "elliott_main"),
            ActivationFunction::ElliotSymmetric => (
                ShaderType::ActivationElliottSymmetric,
                "elliott_symmetric_main",
            ),
            ActivationFunction::Sin => (ShaderType::ActivationSin, "sin_main"),
            ActivationFunction::Cos => (ShaderType::ActivationCos, "cos_main"),
            ActivationFunction::SinSymmetric => {
                (ShaderType::ActivationSinSymmetric, "sin_symmetric_main")
            }
            ActivationFunction::CosSymmetric => {
                (ShaderType::ActivationCosSymmetric, "cos_symmetric_main")
            }
            ActivationFunction::LinearPiece => {
                (ShaderType::ActivationLinearPiece, "linear_piece_main")
            }
            ActivationFunction::LinearPieceSymmetric => (
                ShaderType::ActivationLinearPieceSymmetric,
                "linear_piece_symmetric_main",
            ),
            ActivationFunction::Gelu => (ShaderType::ActivationGELU, "gelu_main"),
            ActivationFunction::Swish => (ShaderType::ActivationSwish, "swish_main"),
        };
        self.get_or_create_pipeline(shader.clone(), entry)?;
        let values: Vec<f32> = inputs.iter().map(|x| x.to_f32().unwrap()).collect();
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Activation input"),
                contents: bytemuck::cast_slice(&values),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bytes = (values.len() * std::mem::size_of::<f32>()) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Activation output"),
            size: bytes,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Activation readback"),
            size: bytes,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let mut parameters = [0u8; 16];
        parameters[..4].copy_from_slice(&(values.len() as u32).to_ne_bytes());
        parameters[4..8].copy_from_slice(&steepness.to_f32().unwrap().to_ne_bytes());
        parameters[8..12].copy_from_slice(&0.01f32.to_ne_bytes());
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Activation parameters"),
                contents: &parameters,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let state = self.gpu_state.read().unwrap();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Activation group"),
            layout: &state.bind_group_layouts[&shader],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Activation dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Activation pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&state.pipelines[&shader]);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| ComputeError::TransferError(error.to_string()))?
            .map_err(|error| ComputeError::TransferError(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let results = bytemuck::cast_slice::<u8, f32>(&mapped)
            .iter()
            .map(|&value| T::from(value).unwrap())
            .collect();
        drop(mapped);
        readback.unmap();
        Ok(results)
    }
}
