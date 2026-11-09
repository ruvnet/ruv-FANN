//! Reverse-mode differentiation for an owned, fixed sequential dense graph.
//!
//! This is a CPU f32/f64 training graph, rather than a general tensor engine.
//! Parameters and their gradients are ordered by layer: row-major
//! `[outputs, inputs]` weights, followed by that layer's optional bias.
//! Workspaces and optimizers bind to a particular model and parameter revision.
use crate::{ActivationFunction, BatchFloat, PreparationLimits, PreparedError};
use std::mem::size_of;

#[cfg(feature = "serde")]
mod json;
mod model;
mod optim;
mod pass;
mod workspace;

pub use optim::{AdamW, AdamWConfig};
pub use workspace::AutogradWorkspace;

/// Shape, resource, numerical or training-state validation failure.
#[derive(Debug, thiserror::Error)]
pub enum AutogradError {
    #[error("invalid dense layer {layer}: {reason}")]
    InvalidLayer { layer: usize, reason: &'static str },
    #[error("activation {activation:?} at layer {layer} has no supported autograd rule")]
    UnsupportedActivation {
        layer: usize,
        activation: ActivationFunction,
    },
    #[error("{kind} length {actual} does not match {expected}")]
    LengthMismatch {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("batch rows {rows} must be between 1 and {max_rows}")]
    BatchSize { rows: usize, max_rows: usize },
    #[error("workspace belongs to a different model")]
    WorkspaceMismatch,
    #[error("gradients are missing or belong to an earlier parameter revision")]
    StaleGradients,
    #[error("optimizer belongs to a different model")]
    OptimizerMismatch,
    #[error("optimizer state predates this parameter revision; configure a new optimizer")]
    StaleOptimizer,
    #[error("non-finite {kind} at index {index}")]
    NonFinite { kind: &'static str, index: usize },
    #[error("non-finite arithmetic in {stage}, layer {layer}, index {index}")]
    NumericOverflow {
        stage: &'static str,
        layer: usize,
        index: usize,
    },
    #[error("invalid optimizer field {field}")]
    InvalidOptimizerConfig { field: &'static str },
    #[error("size arithmetic overflow")]
    SizeOverflow,
    #[error("{resource} needs {required}, exceeding limit {limit}")]
    BudgetExceeded {
        resource: &'static str,
        required: usize,
        limit: usize,
    },
    #[error("allocation failed")]
    AllocationFailed,
    #[error("model identity space exhausted")]
    IdentityExhausted,
    #[error("parameter revision or optimizer step counter exhausted")]
    RevisionExhausted,
    #[error("invalid dense-layer JSON: {0}")]
    InvalidJson(&'static str),
    #[error(transparent)]
    Preparation(#[from] PreparedError),
}

#[derive(Debug)]
struct Layer<T> {
    inputs: usize,
    outputs: usize,
    weights: usize,
    bias: Option<usize>,
    input_unit: usize,
    output_unit: usize,
    activation: ActivationFunction,
    steepness: T,
}

/// Owned dense parameters with actual reverse-mode affine/activation rules.
/// Cloning a graph is deliberately explicit through export/import or layers;
/// workspaces from a different graph never authorize a parameter update.
#[derive(Debug)]
pub struct AutogradNetwork<T: BatchFloat> {
    layers: Vec<Layer<T>>,
    parameters: Vec<T>,
    inputs: usize,
    outputs: usize,
    width: usize,
    tape_units: usize,
    id: usize,
    revision: u64,
    estimated_bytes: usize,
    limits: PreparationLimits,
}

fn add(a: usize, b: usize) -> Result<usize, AutogradError> {
    a.checked_add(b).ok_or(AutogradError::SizeOverflow)
}
fn mul(a: usize, b: usize) -> Result<usize, AutogradError> {
    a.checked_mul(b).ok_or(AutogradError::SizeOverflow)
}
fn payload<T>(elements: usize) -> Result<usize, AutogradError> {
    mul(elements, size_of::<T>())
}
fn budget(resource: &'static str, required: usize, limit: usize) -> Result<(), AutogradError> {
    if required > limit {
        Err(AutogradError::BudgetExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
fn reserved<T>(len: usize) -> Result<Vec<T>, AutogradError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AutogradError::AllocationFailed)?;
    Ok(values)
}
fn zeros<T: BatchFloat>(len: usize) -> Result<Vec<T>, AutogradError> {
    let mut values = reserved(len)?;
    values.resize(len, T::zero());
    Ok(values)
}
fn finite<T: BatchFloat>(values: &[T], kind: &'static str) -> Result<(), AutogradError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(AutogradError::NonFinite { kind, index })
    } else {
        Ok(())
    }
}
fn length(kind: &'static str, actual: usize, expected: usize) -> Result<(), AutogradError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutogradError::LengthMismatch {
            kind,
            expected,
            actual,
        })
    }
}
fn numerical<T: BatchFloat>(
    value: T,
    stage: &'static str,
    layer: usize,
    index: usize,
) -> Result<T, AutogradError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AutogradError::NumericOverflow {
            stage,
            layer,
            index,
        })
    }
}
