use crate::{CandleAdapterError as Error, LinearStage};
use candle_core::{Tensor, WithDType};
use ruv_fann::{BatchFloat, PreparationLimits};
use std::mem::size_of;

pub const MAX_STAGES: usize = 64;
#[derive(Clone, Copy, Default)]
pub(crate) struct Shape {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: usize,
}
pub(crate) struct Plan {
    pub shapes: [Shape; MAX_STAGES],
    pub parameters: usize,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::SizeOverflow)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::SizeOverflow)
}
fn budget(resource: &'static str, required: usize, limit: usize) -> Result<(), Error> {
    if required > limit {
        Err(Error::Budget {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
fn tensor<T: WithDType>(value: &Tensor, stage: usize, kind: &'static str) -> Result<(), Error> {
    if !value.device().is_cpu() {
        return Err(Error::NonCpu {
            stage,
            tensor: kind,
        });
    }
    if value.dtype() != T::DTYPE {
        return Err(Error::DType {
            stage,
            tensor: kind,
            expected: T::DTYPE,
            actual: value.dtype(),
        });
    }
    Ok(())
}

/// Only reads metadata. Numerical storage is not read or materialized here.
pub(crate) fn validate<T: BatchFloat + WithDType>(
    stages: &[LinearStage<'_, T>],
    limits: PreparationLimits,
) -> Result<Plan, Error> {
    if stages.is_empty() {
        return Err(Error::Invalid {
            stage: 0,
            reason: "at least one linear stage is required",
        });
    }
    budget("Candle stages", stages.len(), MAX_STAGES)?;
    let mut shapes = [Shape::default(); MAX_STAGES];
    let (mut parameters, mut rows, mut neurons, mut width) = (0, 0, 0, 0);
    for (index, stage) in stages.iter().enumerate() {
        let weight = stage.layer.weight();
        tensor::<T>(weight, index, "weight")?;
        let dims = weight.dims();
        if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
            return Err(Error::Invalid {
                stage: index,
                reason: "weights must have nonzero [outputs, inputs] shape",
            });
        }
        if index > 0 && shapes[index - 1].outputs != dims[1] {
            return Err(Error::Invalid {
                stage: index,
                reason: "adjacent linear dimensions differ",
            });
        }
        if !stage.steepness.is_finite() {
            return Err(Error::Invalid {
                stage: index,
                reason: "activation steepness must be finite",
            });
        }
        let weights = mul(dims[0], dims[1])?;
        let has_bias = stage.layer.bias().is_some();
        if let Some(bias) = stage.layer.bias() {
            tensor::<T>(bias, index, "bias")?;
            if bias.dims() != [dims[0]] {
                return Err(Error::Invalid {
                    stage: index,
                    reason: "bias must have [outputs] shape",
                });
            }
        }
        parameters = add(
            parameters,
            add(weights, if has_bias { dims[0] } else { 0 })?,
        )?;
        rows = add(rows, dims[0])?;
        if index == 0 {
            neurons = add(dims[1], usize::from(has_bias))?;
            width = neurons;
        }
        let next_bias = stages
            .get(index + 1)
            .is_some_and(|next| next.layer.bias().is_some());
        let output_width = add(dims[0], usize::from(next_bias))?;
        neurons = add(neurons, output_width)?;
        width = width.max(output_width);
        budget("Candle neurons", neurons, limits.max_neurons)?;
        budget("Candle connections", parameters, limits.max_connections)?;
        shapes[index] = Shape {
            inputs: dims[1],
            outputs: dims[0],
            weights,
        };
    }
    // Conservative current-layout allowances, not Rust ABI promises: one copy
    // of logical source parameters, final packed values, descriptors, and two
    // workspace buffers. Excludes caller-owned Candle storage/allocator slack.
    let numeric = mul(mul(parameters, size_of::<T>())?, 2)?;
    let scratch = mul(mul(width, size_of::<T>())?, 2)?;
    let descriptors = add(mul(rows, 64)?, mul(stages.len(), 64 + 512)?)?;
    let estimate = add(add(add(numeric, scratch)?, descriptors)?, 1024)?;
    budget(
        "Candle temporary and packed requested bytes",
        estimate,
        limits.max_bytes,
    )?;
    Ok(Plan { shapes, parameters })
}
