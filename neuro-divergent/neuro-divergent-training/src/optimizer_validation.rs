use crate::{TrainingError, TrainingResult};
use num_traits::Float;

pub(super) fn validate_step<T: Float>(
    parameters: &[Vec<T>],
    gradients: &[Vec<T>],
    rate: T,
) -> TrainingResult<()> {
    if !rate.is_finite()
        || rate < T::zero()
        || parameters.len() != gradients.len()
        || parameters
            .iter()
            .zip(gradients)
            .any(|(parameters, gradients)| {
                parameters.len() != gradients.len()
                    || parameters.iter().chain(gradients).any(|v| !v.is_finite())
            })
    {
        return Err(TrainingError::OptimizerError(
            "Invalid learning rate, nested parameter/gradient shape, or non-finite value".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_buffers<T: Float>(
    buffers: &[Vec<T>],
    parameters: &[Vec<T>],
    nonnegative: bool,
) -> TrainingResult<()> {
    if !buffers.is_empty()
        && (buffers.len() != parameters.len()
            || buffers.iter().zip(parameters).any(|(buffer, parameters)| {
                buffer.len() != parameters.len()
                    || buffer
                        .iter()
                        .any(|v| !v.is_finite() || (nonnegative && *v < T::zero()))
            }))
    {
        return Err(TrainingError::OptimizerError(
            "Invalid optimizer buffers or changed parameter shape; reset the optimizer".into(),
        ));
    }
    Ok(())
}

pub(super) fn unit_interval<T: Float>(value: T, inclusive: bool) -> bool {
    value.is_finite()
        && value >= T::zero()
        && if inclusive {
            value <= T::one()
        } else {
            value < T::one()
        }
}

pub(super) fn nonnegative<T: Float>(value: T) -> bool {
    value.is_finite() && value >= T::zero()
}
