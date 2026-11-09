use super::{
    add, budget, finite, length, mul, payload, reserved, AutogradError, AutogradNetwork, Layer,
};
use crate::{ActivationFunction, BatchFloat, DenseLayer, PreparationLimits, PreparedNetwork};
use std::{
    mem::size_of,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_MODEL: AtomicUsize = AtomicUsize::new(1);

impl<T: BatchFloat> AutogradNetwork<T> {
    /// Copy borrowed parameters into a trainable sequential graph.
    /// Linear, Sigmoid, Tanh (including SigmoidSymmetric) and ReLU have rules.
    /// ReLU preserves FANN's derivative 0 at zero and ignores steepness.
    pub fn from_dense_layers(layers: &[DenseLayer<'_, T>]) -> Result<Self, AutogradError> {
        Self::from_dense_layers_with_limits(layers, PreparationLimits::default())
    }

    /// Validate every layer and all requested storage before parameter allocation.
    /// The byte limit covers this owned graph, excluding source buffers and
    /// separately budgeted workspaces, optimizers and frozen snapshots.
    pub fn from_dense_layers_with_limits(
        layers: &[DenseLayer<'_, T>],
        limits: PreparationLimits,
    ) -> Result<Self, AutogradError> {
        let first = layers.first().ok_or(AutogradError::InvalidLayer {
            layer: 0,
            reason: "no dense layers",
        })?;
        let mut parameter_count = 0;
        let mut tape_units = first.inputs;
        let mut neurons = first.inputs;
        let mut width = first.inputs;
        for (index, layer) in layers.iter().enumerate() {
            validate_layer(
                layer,
                index,
                layers.get(index.wrapping_sub(1)).map(|x| x.outputs),
            )?;
            let count = add(
                mul(layer.inputs, layer.outputs)?,
                if layer.bias.is_some() {
                    layer.outputs
                } else {
                    0
                },
            )?;
            parameter_count = add(parameter_count, count)?;
            tape_units = add(tape_units, layer.outputs)?;
            neurons = add(
                neurons,
                add(layer.outputs, usize::from(layer.bias.is_some()))?,
            )?;
            width = width.max(layer.outputs);
            budget("neurons", neurons, limits.max_neurons)?;
            budget("connections", parameter_count, limits.max_connections)?;
        }
        let estimated_bytes = add(
            add(size_of::<Self>(), payload::<Layer<T>>(layers.len())?)?,
            payload::<T>(parameter_count)?,
        )?;
        budget("autograd model bytes", estimated_bytes, limits.max_bytes)?;
        // Check values only after cheap shape/storage preflight, before allocation.
        for layer in layers {
            finite(layer.weights, "weights")?;
            if let Some(bias) = layer.bias {
                finite(bias, "bias")?;
            }
            finite(std::slice::from_ref(&layer.steepness), "steepness")?;
        }
        // fetch_update preserves the core's Rust 1.81 API compatibility.
        #[allow(deprecated)]
        let id = NEXT_MODEL
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| AutogradError::IdentityExhausted)?;
        let mut network = Self {
            layers: reserved(layers.len())?,
            parameters: reserved(parameter_count)?,
            inputs: first.inputs,
            outputs: layers.last().expect("nonempty").outputs,
            width,
            tape_units,
            id,
            revision: 0,
            estimated_bytes,
            limits,
        };
        let mut input_unit = 0;
        let mut output_unit = first.inputs;
        for layer in layers {
            let weights = network.parameters.len();
            network.parameters.extend_from_slice(layer.weights);
            let bias = layer.bias.map(|values| {
                let start = network.parameters.len();
                network.parameters.extend_from_slice(values);
                start
            });
            network.layers.push(Layer {
                inputs: layer.inputs,
                outputs: layer.outputs,
                weights,
                bias,
                input_unit,
                output_unit,
                activation: layer.activation,
                steepness: layer.steepness,
            });
            input_unit = output_unit;
            output_unit += layer.outputs; // Preflight proved the sum fits.
        }
        Ok(network)
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs
    }
    pub fn num_outputs(&self) -> usize {
        self.outputs
    }
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    pub fn parameters(&self) -> &[T] {
        &self.parameters
    }
    /// Object and requested owned payload, excluding allocator overhead.
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    /// Checked conservative policy units for a forward/reverse batch, suitable
    /// for a host's per-call work cap. These units are not elapsed CPU time.
    /// Zero rows estimate zero work; execution still requires a nonempty batch.
    pub fn training_work_estimate(&self, rows: usize) -> Result<usize, AutogradError> {
        let parameters = mul(32, self.parameter_count())?;
        let boundaries = mul(8, add(self.inputs, self.outputs)?)?;
        mul(rows, add(add(parameters, boundaries)?, 1)?)
    }

    /// Conservative policy units for an SGD or AdamW parameter update.
    pub fn optimizer_work_estimate(&self) -> Result<usize, AutogradError> {
        mul(64, self.parameter_count())
    }

    /// Replace all parameters atomically. Successful replacement invalidates
    /// previously computed gradients and existing AdamW state by revision.
    pub fn set_parameters(&mut self, values: &[T]) -> Result<(), AutogradError> {
        length("parameters", values.len(), self.parameters.len())?;
        finite(values, "parameters")?;
        let revision = self.next_revision()?;
        self.parameters.copy_from_slice(values);
        self.revision = revision;
        Ok(())
    }

    /// Copy a frozen inference snapshot. Subsequent training cannot modify it.
    pub fn prepare(&self) -> Result<PreparedNetwork<T>, AutogradError> {
        let layers = self.dense_layers()?;
        Ok(PreparedNetwork::from_dense_layers_with_limits(
            &layers,
            self.limits,
        )?)
    }

    pub(super) fn dense_layers(&self) -> Result<Vec<DenseLayer<'_, T>>, AutogradError> {
        let mut result = reserved(self.layers.len())?;
        for layer in &self.layers {
            result.push(DenseLayer {
                inputs: layer.inputs,
                outputs: layer.outputs,
                weights: &self.parameters
                    [layer.weights..layer.weights + layer.inputs * layer.outputs],
                bias: layer
                    .bias
                    .map(|start| &self.parameters[start..start + layer.outputs]),
                activation: layer.activation,
                steepness: layer.steepness,
            });
        }
        Ok(result)
    }

    pub(super) fn next_revision(&self) -> Result<u64, AutogradError> {
        self.revision
            .checked_add(1)
            .ok_or(AutogradError::RevisionExhausted)
    }
}

fn validate_layer<T: BatchFloat>(
    layer: &DenseLayer<'_, T>,
    index: usize,
    previous: Option<usize>,
) -> Result<(), AutogradError> {
    let reason = if layer.inputs == 0 || layer.outputs == 0 {
        Some("dense dimensions must be positive")
    } else if previous.is_some_and(|outputs| outputs != layer.inputs) {
        Some("adjacent dense dimensions disagree")
    } else if layer.weights.len() != mul(layer.inputs, layer.outputs)? {
        Some("dense weight shape must be outputs by inputs")
    } else if layer.bias.is_some_and(|bias| bias.len() != layer.outputs) {
        Some("dense bias length must match outputs")
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(AutogradError::InvalidLayer {
            layer: index,
            reason,
        });
    }
    if !matches!(
        layer.activation,
        ActivationFunction::Linear
            | ActivationFunction::Sigmoid
            | ActivationFunction::Tanh
            | ActivationFunction::SigmoidSymmetric
            | ActivationFunction::ReLU
    ) {
        return Err(AutogradError::UnsupportedActivation {
            layer: index,
            activation: layer.activation,
        });
    }
    Ok(())
}
