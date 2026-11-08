use crate::{TrainingError, TrainingResult};
use num_traits::Float;

pub(super) fn validate_inputs<T: Float>(predictions: &[T], targets: &[T]) -> TrainingResult<()> {
    if predictions.is_empty()
        || targets.is_empty()
        || predictions.iter().chain(targets).any(|v| !v.is_finite())
    {
        return Err(TrainingError::LossError(
            "Loss inputs must be nonempty and finite".into(),
        ));
    }
    Ok(())
}

pub(super) fn checked_loss<T: Float>(loss: T) -> TrainingResult<T> {
    if !loss.is_finite() {
        return Err(TrainingError::LossError(
            "Loss overflowed or has an invalid domain".into(),
        ));
    }
    Ok(loss)
}

pub(super) fn checked_gradients<T: Float>(gradients: Vec<T>) -> TrainingResult<Vec<T>> {
    if gradients.iter().any(|v| !v.is_finite()) {
        return Err(TrainingError::LossError(
            "Loss gradient overflowed or has an invalid domain".into(),
        ));
    }
    Ok(gradients)
}

pub(super) fn positive<T: Float>(value: T) -> TrainingResult<()> {
    if !value.is_finite() || value <= T::zero() {
        return Err(TrainingError::LossError(
            "Loss scale/epsilon must be positive and finite".into(),
        ));
    }
    Ok(())
}

pub(super) fn nonnegative<T: Float>(value: T) -> TrainingResult<()> {
    if !value.is_finite() || value < T::zero() {
        return Err(TrainingError::LossError(
            "Loss parameter/target must be nonnegative and finite".into(),
        ));
    }
    Ok(())
}

pub(super) fn quantile<T: Float>(value: T) -> TrainingResult<()> {
    if !value.is_finite() || value < T::zero() || value > T::one() {
        return Err(TrainingError::LossError(
            "Quantiles must be finite and in [0, 1]".into(),
        ));
    }
    Ok(())
}
