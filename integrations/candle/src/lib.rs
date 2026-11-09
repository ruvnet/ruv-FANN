//! Copy CPU Candle linear tensors into owned ruv-FANN inference parameters.
//!
//! No model, device transfer, dtype conversion or training is started. Candle's
//! weights have logical shape `[outputs, inputs]`; biases must have shape
//! `[outputs]`. Source owners must stop concurrent updates while preparing.
//! After successful preparation, tensors may be changed or dropped independently.

mod error;
mod export;
mod preflight;
pub use error::CandleAdapterError;
pub use export::{export_dense_f32_json, export_dense_f64_json};

use candle_core::{Storage, Tensor, WithDType};
use candle_nn::Linear;
use ruv_fann::{ActivationFunction, BatchFloat, DenseLayer, PreparationLimits, PreparedNetwork};

/// One Candle linear map followed by an explicitly selected FANN activation.
/// This does not inspect or reconstruct a Candle computation/autograd graph.
#[derive(Clone, Copy, Debug)]
pub struct LinearStage<'a, T> {
    pub layer: &'a Linear,
    pub activation: ActivationFunction,
    pub steepness: T,
}

/// Prepare f32 CPU tensors using default limits and conservative copy budgeting.
pub fn prepare_f32(
    stages: &[LinearStage<'_, f32>],
) -> Result<PreparedNetwork<f32>, CandleAdapterError> {
    prepare_f32_with_limits(stages, PreparationLimits::default())
}

/// Prepare f64 CPU tensors using default limits and conservative copy budgeting.
pub fn prepare_f64(
    stages: &[LinearStage<'_, f64>],
) -> Result<PreparedNetwork<f64>, CandleAdapterError> {
    prepare_f64_with_limits(stages, PreparationLimits::default())
}

/// `max_bytes` bounds requested temporary-plus-packed storage conservatively;
/// it is stricter than the core's packed-only budget. See the crate README.
pub fn prepare_f32_with_limits(
    stages: &[LinearStage<'_, f32>],
    limits: PreparationLimits,
) -> Result<PreparedNetwork<f32>, CandleAdapterError> {
    prepare(stages, limits)
}

/// f64 counterpart of [`prepare_f32_with_limits`].
pub fn prepare_f64_with_limits(
    stages: &[LinearStage<'_, f64>],
    limits: PreparationLimits,
) -> Result<PreparedNetwork<f64>, CandleAdapterError> {
    prepare(stages, limits)
}

struct OwnedLayer<T> {
    weights: Vec<T>,
    bias: Option<Vec<T>>,
}

fn prepare<T: BatchFloat + WithDType>(
    stages: &[LinearStage<'_, T>],
    limits: PreparationLimits,
) -> Result<PreparedNetwork<T>, CandleAdapterError> {
    let plan = preflight::validate(stages, limits)?;
    let owned = materialize(stages, &plan)?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(stages.len())
        .map_err(|_| CandleAdapterError::AllocationFailed)?;
    for (index, (stage, values)) in stages.iter().zip(&owned).enumerate() {
        let shape = plan.shapes[index];
        layers.push(DenseLayer {
            inputs: shape.inputs,
            outputs: shape.outputs,
            weights: &values.weights,
            bias: values.bias.as_deref(),
            activation: stage.activation,
            steepness: stage.steepness,
        });
    }
    Ok(PreparedNetwork::from_dense_layers_with_limits(
        &layers, limits,
    )?)
}

fn materialize<T: BatchFloat + WithDType>(
    stages: &[LinearStage<'_, T>],
    plan: &preflight::Plan,
) -> Result<Vec<OwnedLayer<T>>, CandleAdapterError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(stages.len())
        .map_err(|_| CandleAdapterError::AllocationFailed)?;
    for (index, stage) in stages.iter().enumerate() {
        let shape = plan.shapes[index];
        let weights = logical_values(stage.layer.weight(), shape.weights, index, "weight")?;
        let bias = stage
            .layer
            .bias()
            .map(|tensor| logical_values(tensor, shape.outputs, index, "bias"))
            .transpose()?;
        owned.push(OwnedLayer { weights, bias });
    }
    Ok(owned)
}

/// Read one storage guard at a time. Logical strides and nonzero offsets are
/// honored without allocating a contiguous Candle tensor or invoking a device.
fn logical_values<T: BatchFloat + WithDType>(
    tensor: &Tensor,
    expected: usize,
    stage: usize,
    kind: &'static str,
) -> Result<Vec<T>, CandleAdapterError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(expected)
        .map_err(|_| CandleAdapterError::AllocationFailed)?;
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cpu(cpu) = &*storage else {
        return Err(CandleAdapterError::NonCpu {
            stage,
            tensor: kind,
        });
    };
    let source = T::cpu_storage_as_slice(cpu)?;
    if let Some((start, end)) = layout.contiguous_offsets() {
        let values = source
            .get(start..end)
            .filter(|values| values.len() == expected)
            .ok_or(CandleAdapterError::Invalid {
                stage,
                reason: "invalid contiguous tensor layout",
            })?;
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(CandleAdapterError::NonFinite {
                    stage,
                    tensor: kind,
                    index,
                });
            }
            result.push(value);
        }
    } else {
        for (index, offset) in tensor
            .strided_index()
            .take(expected.saturating_add(1))
            .enumerate()
        {
            if index == expected {
                return Err(CandleAdapterError::Invalid {
                    stage,
                    reason: "strided tensor has too many values",
                });
            }
            let &value = source.get(offset).ok_or(CandleAdapterError::Invalid {
                stage,
                reason: "tensor stride exceeds backing storage",
            })?;
            if !value.is_finite() {
                return Err(CandleAdapterError::NonFinite {
                    stage,
                    tensor: kind,
                    index,
                });
            }
            result.push(value);
        }
        if result.len() != expected {
            return Err(CandleAdapterError::Invalid {
                stage,
                reason: "strided tensor value count differs",
            });
        }
    }
    Ok(result)
}
