use super::{add, budget, mul, payload, zeros, AutogradError, AutogradNetwork};
use crate::BatchFloat;
use std::mem::size_of;

/// Reusable fixed-capacity tape, adjoints, gradients and proposed parameters.
/// Hot forward, reverse and optimizer operations never grow these buffers.
/// Contents are invalid after a failed pass; model parameters remain unchanged.
#[derive(Debug)]
pub struct AutogradWorkspace<T: BatchFloat> {
    pub(super) tape: Vec<T>,
    pub(super) delta: Vec<T>,
    pub(super) previous_delta: Vec<T>,
    pub(super) gradients: Vec<T>,
    pub(super) input_gradients: Vec<T>,
    pub(super) proposed: Vec<T>,
    pub(super) model_id: usize,
    pub(super) max_rows: usize,
    pub(super) rows: usize,
    pub(super) output_start: usize,
    pub(super) output_width: usize,
    pub(super) input_width: usize,
    pub(super) gradient_revision: Option<u64>,
    estimated_bytes: usize,
}

impl<T: BatchFloat> AutogradWorkspace<T> {
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
    /// Requested object and buffer bytes; model and AdamW buffers are separate.
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }
    /// Last successful pass predictions, row-major. Empty before a pass or
    /// after a failed pass. Successful parameter steps retain those predictions.
    pub fn outputs(&self) -> &[T] {
        &self.tape[self.output_start..self.output_start + self.rows * self.output_width]
    }
    /// Last successful reverse pass gradients; empty after forward or a step.
    /// A model's set_parameters cannot mutate this borrowed workspace; stepping
    /// additionally checks its recorded revision against the model's revision.
    pub fn parameter_gradients(&self) -> &[T] {
        if self.gradient_revision.is_some() {
            &self.gradients
        } else {
            &[]
        }
    }
    pub fn input_gradients(&self) -> &[T] {
        if self.gradient_revision.is_some() {
            &self.input_gradients[..self.rows * self.input_width]
        } else {
            &[]
        }
    }
    pub(super) fn invalidate_pass(&mut self) {
        self.rows = 0;
        self.gradient_revision = None;
    }
}

impl<T: BatchFloat> AutogradNetwork<T> {
    /// Preflight the object and all training scratch before any allocation.
    pub fn workspace_bytes(&self, max_rows: usize) -> Result<usize, AutogradError> {
        if max_rows == 0 {
            return Err(AutogradError::BatchSize {
                rows: 0,
                max_rows: 0,
            });
        }
        let units = add(add(self.tape_units, mul(2, self.width)?)?, self.inputs)?;
        let scalars = add(mul(max_rows, units)?, mul(2, self.parameters.len())?)?;
        add(size_of::<AutogradWorkspace<T>>(), payload::<T>(scalars)?)
    }

    pub fn workspace(
        &self,
        max_rows: usize,
        max_bytes: usize,
    ) -> Result<AutogradWorkspace<T>, AutogradError> {
        let estimated_bytes = self.workspace_bytes(max_rows)?;
        budget("autograd workspace bytes", estimated_bytes, max_bytes)?;
        Ok(AutogradWorkspace {
            tape: zeros(mul(max_rows, self.tape_units)?)?,
            delta: zeros(mul(max_rows, self.width)?)?,
            previous_delta: zeros(mul(max_rows, self.width)?)?,
            gradients: zeros(self.parameters.len())?,
            input_gradients: zeros(mul(max_rows, self.inputs)?)?,
            proposed: zeros(self.parameters.len())?,
            model_id: self.id,
            max_rows,
            rows: 0,
            output_start: self.layers.last().expect("nonempty").output_unit * max_rows,
            output_width: self.outputs,
            input_width: self.inputs,
            gradient_revision: None,
            estimated_bytes,
        })
    }
}
