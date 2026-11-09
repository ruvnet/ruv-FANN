use super::{finite, length, mul, numerical, AutogradError, AutogradNetwork, AutogradWorkspace};
use crate::{ActivationFunction, BatchFloat};

impl<T: BatchFloat> AutogradNetwork<T> {
    /// Evaluate a row-major batch, committing caller output only after the whole
    /// pass succeeds. The reusable workspace must belong to this exact graph.
    pub fn run_batch_into(
        &self,
        inputs: &[T],
        outputs: &mut [T],
        rows: usize,
        workspace: &mut AutogradWorkspace<T>,
    ) -> Result<(), AutogradError> {
        self.check_workspace(workspace)?;
        workspace.invalidate_pass();
        self.validate_inputs(inputs, rows, workspace)?;
        length("outputs", outputs.len(), mul(rows, self.outputs)?)?;
        self.forward(inputs, rows, workspace)?;
        workspace.rows = rows;
        outputs.copy_from_slice(workspace.outputs());
        Ok(())
    }

    /// Reverse-mode MSE, averaged over every element (`rows * num_outputs`).
    /// Produces parameter and input gradients from the current owned parameters.
    /// Repeated calls replace gradients; there is no implicit accumulation across
    /// batches. Parameters never change until an explicit optimizer step.
    pub fn backward_mse(
        &self,
        inputs: &[T],
        targets: &[T],
        rows: usize,
        workspace: &mut AutogradWorkspace<T>,
    ) -> Result<T, AutogradError> {
        self.check_workspace(workspace)?;
        workspace.invalidate_pass();
        self.validate_inputs(inputs, rows, workspace)?;
        let count = mul(rows, self.outputs)?;
        length("targets", targets.len(), count)?;
        finite(targets, "targets")?;
        self.forward(inputs, rows, workspace)?;
        let denominator = numerical(
            T::from(count).ok_or(AutogradError::SizeOverflow)?,
            "MSE denominator",
            self.layers.len(),
            0,
        )?;
        let two = T::one() + T::one();
        let mut squared_error = T::zero();
        for (index, &target) in targets.iter().enumerate() {
            let prediction = workspace.tape[workspace.output_start + index];
            let difference = numerical(
                prediction - target,
                "MSE difference",
                self.layers.len(),
                index,
            )?;
            squared_error = numerical(
                squared_error + difference * difference,
                "MSE sum",
                self.layers.len(),
                index,
            )?;
            workspace.delta[index] = numerical(
                two * difference / denominator,
                "MSE seed",
                self.layers.len(),
                index,
            )?;
        }
        let loss = numerical(
            squared_error / denominator,
            "MSE mean",
            self.layers.len(),
            0,
        )?;
        workspace.gradients.fill(T::zero());
        for (layer_index, layer) in self.layers.iter().enumerate().rev() {
            let source = layer.input_unit * workspace.max_rows;
            let output = layer.output_unit * workspace.max_rows;
            let previous = &mut workspace.previous_delta[..rows * layer.inputs];
            previous.fill(T::zero());
            for row in 0..rows {
                for neuron in 0..layer.outputs {
                    let target_index = row * layer.outputs + neuron;
                    let value = workspace.tape[output + target_index];
                    let derivative = derivative(layer.activation, value, layer.steepness);
                    let delta = numerical(
                        workspace.delta[target_index] * derivative,
                        "activation adjoint",
                        layer_index,
                        target_index,
                    )?;
                    for input in 0..layer.inputs {
                        let source_index = row * layer.inputs + input;
                        let parameter = layer.weights + neuron * layer.inputs + input;
                        workspace.gradients[parameter] = numerical(
                            workspace.gradients[parameter]
                                + delta * workspace.tape[source + source_index],
                            "weight gradient",
                            layer_index,
                            parameter,
                        )?;
                        previous[source_index] = numerical(
                            previous[source_index] + delta * self.parameters[parameter],
                            "input adjoint",
                            layer_index,
                            source_index,
                        )?;
                    }
                    if let Some(start) = layer.bias {
                        let parameter = start + neuron;
                        workspace.gradients[parameter] = numerical(
                            workspace.gradients[parameter] + delta,
                            "bias gradient",
                            layer_index,
                            parameter,
                        )?;
                    }
                }
            }
            std::mem::swap(&mut workspace.delta, &mut workspace.previous_delta);
        }
        workspace.input_gradients[..rows * self.inputs]
            .copy_from_slice(&workspace.delta[..rows * self.inputs]);
        workspace.rows = rows;
        workspace.gradient_revision = Some(self.revision);
        Ok(loss)
    }

    fn forward(
        &self,
        inputs: &[T],
        rows: usize,
        workspace: &mut AutogradWorkspace<T>,
    ) -> Result<(), AutogradError> {
        workspace.tape[..inputs.len()].copy_from_slice(inputs);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let source = layer.input_unit * workspace.max_rows;
            let output = layer.output_unit * workspace.max_rows;
            for row in 0..rows {
                for neuron in 0..layer.outputs {
                    let mut sum = T::zero();
                    for input in 0..layer.inputs {
                        sum = sum
                            + workspace.tape[source + row * layer.inputs + input]
                                * self.parameters[layer.weights + neuron * layer.inputs + input];
                    }
                    if let Some(start) = layer.bias {
                        sum = sum + self.parameters[start + neuron];
                    }
                    let index = row * layer.outputs + neuron;
                    numerical(sum, "affine forward", layer_index, index)?;
                    workspace.tape[output + index] = numerical(
                        layer.activation.apply(sum, layer.steepness),
                        "activation forward",
                        layer_index,
                        index,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn check_workspace(
        &self,
        workspace: &AutogradWorkspace<T>,
    ) -> Result<(), AutogradError> {
        if workspace.model_id == self.id {
            Ok(())
        } else {
            Err(AutogradError::WorkspaceMismatch)
        }
    }

    pub(super) fn check_gradients(
        &self,
        workspace: &AutogradWorkspace<T>,
    ) -> Result<(), AutogradError> {
        self.check_workspace(workspace)?;
        if workspace.gradient_revision == Some(self.revision) {
            Ok(())
        } else {
            Err(AutogradError::StaleGradients)
        }
    }

    fn validate_inputs(
        &self,
        inputs: &[T],
        rows: usize,
        workspace: &AutogradWorkspace<T>,
    ) -> Result<(), AutogradError> {
        if rows == 0 || rows > workspace.max_rows {
            return Err(AutogradError::BatchSize {
                rows,
                max_rows: workspace.max_rows,
            });
        }
        length("inputs", inputs.len(), mul(rows, self.inputs)?)?;
        finite(inputs, "inputs")
    }
}

fn derivative<T: BatchFloat>(activation: ActivationFunction, value: T, steepness: T) -> T {
    let one = T::one();
    match activation {
        ActivationFunction::Linear => steepness,
        ActivationFunction::Sigmoid => value * (one - value) * steepness,
        ActivationFunction::Tanh | ActivationFunction::SigmoidSymmetric => {
            (one - value * value) * steepness
        }
        ActivationFunction::ReLU => {
            if value > T::zero() {
                one
            } else {
                T::zero()
            }
        }
        _ => unreachable!("constructor validates supported activation rules"),
    }
}
