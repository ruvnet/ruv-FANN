//! Immutable, owned inference parameters with separate reusable caller scratch.
//!
//! Preparation follows the pack-once/reuse-many lifetime pattern used by inference
//! libraries such as XNNPACK; this is not an XNNPACK backend. Dense ordered rows
//! use contiguous scalar dot products, preserving floating-point addition order.

use crate::{ActivationFunction, Network};
use num_traits::Float;
use std::mem::size_of;
use thiserror::Error;
#[path = "prepared_batch.rs"]
mod batch;
pub use batch::{BatchFloat, BatchWorkspace};
#[path = "prepared_dense.rs"]
mod dense;
pub use dense::{DenseLayer, DENSE_LAYERS_SCHEMA};
#[cfg(feature = "serde")]
#[path = "prepared_dense_json.rs"]
mod dense_json;

/// Checked preparation and inference failures. No error requires heap allocation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PreparedError {
    #[error("Invalid network at layer {layer}, neuron {neuron}: {reason}")]
    InvalidNetwork {
        layer: usize,
        neuron: usize,
        reason: &'static str,
    },
    #[error("Prepared allocation size overflow")]
    SizeOverflow,
    #[error("{resource} requires {required}, exceeding limit {limit}")]
    BudgetExceeded {
        resource: &'static str,
        required: usize,
        limit: usize,
    },
    #[error("Prepared inference allocation failed")]
    AllocationFailed,
    #[error("Input size mismatch: expected {expected}, got {actual}")]
    InputSizeMismatch { expected: usize, actual: usize },
    #[error("Output size mismatch: expected {expected}, got {actual}")]
    OutputSizeMismatch { expected: usize, actual: usize },
    #[error("Workspace is too small: need {required} elements per buffer, got {actual}")]
    WorkspaceTooSmall { required: usize, actual: usize },
    #[error("Input {index} is not finite")]
    NonFiniteInput { index: usize },
    #[error("Non-finite sum or activation at layer {layer}, neuron {neuron}")]
    NumericOverflow { layer: usize, neuron: usize },
}

/// Limits apply before allocation. Bytes include packed parameters, indices,
/// descriptors, and one workspace's two value buffers, not peak process RAM.
#[derive(Debug, Clone, Copy)]
pub struct PreparationLimits {
    pub max_bytes: usize,
    pub max_neurons: usize,
    pub max_connections: usize,
}

impl Default for PreparationLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            max_neurons: 1_048_576,
            max_connections: 16_777_216,
        }
    }
}

#[derive(Debug)]
struct Row<T> {
    start: usize,
    len: usize,
    index_start: usize, // usize::MAX denotes a dense, canonical row.
    activation: ActivationFunction,
    steepness: T,
}

#[derive(Debug)]
struct PackedLayer {
    row_start: usize,
    row_count: usize,
    source_width: usize,
    bias: bool,
}

/// An immutable snapshot independent of future source weights, topology or caches.
/// Generic Float arithmetic is used; f32 and f64 are covered by equivalence tests.
/// Share this object across threads, with one workspace per concurrent call.
#[derive(Debug)]
pub struct PreparedNetwork<T: Float> {
    weights: Vec<T>,
    indices: Vec<usize>,
    rows: Vec<Row<T>>,
    layers: Vec<PackedLayer>,
    inputs: usize,
    outputs: usize,
    width: usize,
    input_bias: bool,
    /// Every first-layer row is dense, so single-row inference can read the
    /// caller's inputs in place instead of copying them into the workspace.
    direct_first: bool,
    estimated_bytes: usize,
}

/// Caller-owned scratch. A workspace can serve any prepared network whose
/// required width fits; contents are overwritten and may change on errors.
#[derive(Debug)]
pub struct InferenceWorkspace<T: Float> {
    front: Vec<T>,
    back: Vec<T>,
}

fn add(a: usize, b: usize) -> Result<usize, PreparedError> {
    a.checked_add(b).ok_or(PreparedError::SizeOverflow)
}
fn bytes(count: usize, width: usize) -> Result<usize, PreparedError> {
    count.checked_mul(width).ok_or(PreparedError::SizeOverflow)
}
fn budget(resource: &'static str, required: usize, limit: usize) -> Result<(), PreparedError> {
    if required > limit {
        Err(PreparedError::BudgetExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
fn invalid(layer: usize, neuron: usize, reason: &'static str) -> PreparedError {
    PreparedError::InvalidNetwork {
        layer,
        neuron,
        reason,
    }
}
fn reserved<T>(count: usize) -> Result<Vec<T>, PreparedError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PreparedError::AllocationFailed)?;
    Ok(values)
}

impl<T: Float> Network<T> {
    /// Snapshot validated parameters for immutable inference. Bias values must
    /// be one. Transient regular-neuron sums and values are not copied.
    pub fn prepare(&self) -> Result<PreparedNetwork<T>, PreparedError> {
        PreparedNetwork::from_network_with_limits(self, PreparationLimits::default())
    }
}

impl<T: Float> PreparedNetwork<T> {
    /// Validate shapes, finite parameters and allocation limits before packing.
    /// Biases must be last, unconnected, and have value one; noncanonical bias
    /// state is rejected. Source input-neuron connections are also rejected.
    pub fn from_network_with_limits(
        network: &Network<T>,
        limits: PreparationLimits,
    ) -> Result<Self, PreparedError> {
        if network.layers.is_empty() {
            return Err(invalid(0, 0, "no layers"));
        }
        if !network.connection_rate.is_finite()
            || network.connection_rate < T::zero()
            || network.connection_rate > T::one()
        {
            return Err(invalid(
                0,
                0,
                "connection rate must be finite and in [0, 1]",
            ));
        }
        let (mut neurons, mut connections, mut rows, mut sparse, mut width) = (0, 0, 0, 0, 0);
        for (layer_index, layer) in network.layers.iter().enumerate() {
            if layer.num_regular_neurons() == 0 {
                return Err(invalid(layer_index, 0, "empty layer"));
            }
            neurons = add(neurons, layer.size())?;
            budget("neurons", neurons, limits.max_neurons)?;
            width = width.max(layer.size());
            let source_width = if layer_index == 0 {
                0
            } else {
                network.layers[layer_index - 1].size()
            };
            for (neuron_index, neuron) in layer.neurons.iter().enumerate() {
                if !neuron.activation_steepness.is_finite() {
                    return Err(invalid(layer_index, neuron_index, "non-finite steepness"));
                }
                if neuron.is_bias
                    && (neuron_index + 1 != layer.size()
                        || !neuron.connections.is_empty()
                        || neuron.value != T::one())
                {
                    return Err(invalid(
                        layer_index,
                        neuron_index,
                        "bias must be last, unconnected, and value one",
                    ));
                }
                connections = add(connections, neuron.connections.len())?;
                budget("connections", connections, limits.max_connections)?;
                for connection in &neuron.connections {
                    if connection.from_neuron >= source_width || !connection.weight.is_finite() {
                        return Err(invalid(
                            layer_index,
                            neuron_index,
                            "invalid source index or non-finite weight",
                        ));
                    }
                }
                if layer_index > 0 && !neuron.is_bias {
                    rows = add(rows, 1)?;
                    let dense = neuron.connections.len() == source_width
                        && neuron
                            .connections
                            .iter()
                            .enumerate()
                            .all(|(index, connection)| connection.from_neuron == index);
                    if !dense {
                        sparse = add(sparse, neuron.connections.len())?;
                    }
                }
            }
        }
        let mut estimated = add(size_of::<Self>(), size_of::<InferenceWorkspace<T>>())?;
        for allocation in [
            bytes(connections, size_of::<T>())?,
            bytes(sparse, size_of::<usize>())?,
            bytes(rows, size_of::<Row<T>>())?,
            bytes(network.layers.len() - 1, size_of::<PackedLayer>())?,
            bytes(bytes(width, 2)?, size_of::<T>())?,
        ] {
            estimated = add(estimated, allocation)?;
        }
        budget("snapshot and workspace bytes", estimated, limits.max_bytes)?;
        let mut prepared = Self {
            weights: reserved(connections)?,
            indices: reserved(sparse)?,
            rows: reserved(rows)?,
            layers: reserved(network.layers.len() - 1)?,
            inputs: network.num_inputs(),
            outputs: network.num_outputs(),
            width,
            input_bias: network.layers[0].has_bias(),
            direct_first: false,
            estimated_bytes: estimated,
        };
        for (index, layer) in network.layers.iter().enumerate().skip(1) {
            let source_width = network.layers[index - 1].size();
            let row_start = prepared.rows.len();
            for neuron in layer.neurons.iter().filter(|neuron| !neuron.is_bias) {
                let dense = neuron.connections.len() == source_width
                    && neuron
                        .connections
                        .iter()
                        .enumerate()
                        .all(|(index, connection)| connection.from_neuron == index);
                let index_start = if dense {
                    usize::MAX
                } else {
                    prepared.indices.len()
                };
                let start = prepared.weights.len();
                for connection in &neuron.connections {
                    prepared.weights.push(connection.weight);
                    if !dense {
                        prepared.indices.push(connection.from_neuron);
                    }
                }
                prepared.rows.push(Row {
                    start,
                    len: neuron.connections.len(),
                    index_start,
                    activation: neuron.activation_function,
                    steepness: neuron.activation_steepness,
                });
            }
            prepared.layers.push(PackedLayer {
                row_start,
                row_count: layer.num_regular_neurons(),
                source_width,
                bias: layer.has_bias(),
            });
        }
        prepared.direct_first = prepared.first_layer_is_dense();
        Ok(prepared)
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs
    }
    pub fn num_outputs(&self) -> usize {
        self.outputs
    }
    /// Stored connection count, including duplicate and bias connections.
    pub fn connection_count(&self) -> usize {
        self.weights.len()
    }
    /// Conservative work units per sample: two per connection (multiply/add),
    /// three per computed neuron (initialize/check/activate), two per input and
    /// output (transfer/check), one per bias and one per layer, including input.
    /// This is a budget estimate, not a measured instruction count or time bound;
    /// disconnected activation-heavy graphs still consume work units.
    pub fn inference_work_per_row(&self) -> Result<usize, PreparedError> {
        let biases = self
            .layers
            .iter()
            .try_fold(usize::from(self.input_bias), |sum, layer| {
                add(sum, usize::from(layer.bias))
            })?;
        let layers = add(self.layers.len(), 1)?;
        let arithmetic = add(bytes(self.weights.len(), 2)?, bytes(self.rows.len(), 3)?)?;
        let transfers = bytes(add(self.inputs, self.outputs)?, 2)?;
        add(add(arithmetic, transfers)?, add(biases, layers)?)
    }
    /// Requested snapshot storage, excluding any inference workspace, source
    /// model, allocator bookkeeping and spare allocator capacity.
    pub fn snapshot_bytes(&self) -> usize {
        self.estimated_bytes - size_of::<InferenceWorkspace<T>>() - 2 * self.width * size_of::<T>()
    }
    /// Requested owned storage, including descriptors and one workspace, excluding
    /// allocator bookkeeping, spare allocator capacity and the original network.
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    /// Allocate both scratch buffers once, before repeated inference.
    pub fn workspace(&self) -> Result<InferenceWorkspace<T>, PreparedError> {
        let mut front = reserved(self.width)?;
        let mut back = reserved(self.width)?;
        front.resize(self.width, T::zero());
        back.resize(self.width, T::zero());
        Ok(InferenceWorkspace { front, back })
    }

    /// True when every row of the first layer reads the whole input row (and
    /// the input bias, when present) in canonical order.
    pub(crate) fn first_layer_is_dense(&self) -> bool {
        self.layers.first().is_some_and(|layer| {
            self.rows[layer.row_start..layer.row_start + layer.row_count]
                .iter()
                .all(|row| row.index_start == usize::MAX && row.len == layer.source_width)
        })
    }

    /// Independent output accumulators share source loads without changing a
    /// row's multiply/add order. No horizontal reduction or fused operation.
    #[inline]
    fn run_dense_four(
        &self,
        rows: &[Row<T>],
        inputs: &[T],
        input_bias: bool,
        outputs: &mut [T],
        layer: usize,
        first_neuron: usize,
    ) -> Result<(), PreparedError> {
        let weights = |row: &Row<T>| &self.weights[row.start..row.start + row.len];
        let (a, b, c, d) = (
            weights(&rows[0]),
            weights(&rows[1]),
            weights(&rows[2]),
            weights(&rows[3]),
        );
        let mut sums = [T::zero(); 4];
        for ((((&value, &wa), &wb), &wc), &wd) in inputs.iter().zip(a).zip(b).zip(c).zip(d) {
            sums[0] = sums[0] + value * wa;
            sums[1] = sums[1] + value * wb;
            sums[2] = sums[2] + value * wc;
            sums[3] = sums[3] + value * wd;
        }
        if input_bias {
            for (sum, weights) in sums.iter_mut().zip([a, b, c, d]) {
                *sum = *sum + weights[inputs.len()];
            }
        }
        // Preserve neuron priority, including an earlier activation overflow
        // ahead of a later neuron's non-finite dot product.
        for (lane, (&sum, row)) in sums.iter().zip(rows).enumerate() {
            if !sum.is_finite() {
                return Err(PreparedError::NumericOverflow {
                    layer,
                    neuron: first_neuron + lane,
                });
            }
            let value = row.activation.apply(sum, row.steepness);
            if !value.is_finite() {
                return Err(PreparedError::NumericOverflow {
                    layer,
                    neuron: first_neuron + lane,
                });
            }
            outputs[lane] = value;
        }
        Ok(())
    }

    /// Keep the larger output tile out of the scalar inference function. Only
    /// whole canonical layers with enough source values and rows reach here.
    #[inline(never)]
    fn run_dense_layer(
        &self,
        rows: &[Row<T>],
        inputs: &[T],
        input_bias: bool,
        outputs: &mut [T],
        layer: usize,
    ) -> Result<(), PreparedError> {
        let mut index = 0;
        while index + 4 <= rows.len() {
            self.run_dense_four(
                &rows[index..index + 4],
                inputs,
                input_bias,
                &mut outputs[index..index + 4],
                layer,
                index,
            )?;
            index += 4;
        }
        for (offset, row) in rows[index..].iter().enumerate() {
            let weights = &self.weights[row.start..row.start + row.len];
            let mut sum = inputs
                .iter()
                .zip(weights)
                .fold(T::zero(), |sum, (&value, &weight)| sum + value * weight);
            if input_bias {
                sum = sum + weights[inputs.len()];
            }
            if !sum.is_finite() {
                return Err(PreparedError::NumericOverflow {
                    layer,
                    neuron: index + offset,
                });
            }
            let value = row.activation.apply(sum, row.steepness);
            if !value.is_finite() {
                return Err(PreparedError::NumericOverflow {
                    layer,
                    neuron: index + offset,
                });
            }
            outputs[index + offset] = value;
        }
        Ok(())
    }

    /// No steady-state allocations. Outputs are committed only after all layers
    /// succeed. Non-finite inputs, weighted sums or activation outputs are errors;
    /// stable saturation inside an activation follows the normal scalar API.
    pub fn run_into(
        &self,
        inputs: &[T],
        outputs: &mut [T],
        workspace: &mut InferenceWorkspace<T>,
    ) -> Result<(), PreparedError> {
        if inputs.len() != self.inputs {
            return Err(PreparedError::InputSizeMismatch {
                expected: self.inputs,
                actual: inputs.len(),
            });
        }
        if outputs.len() != self.outputs {
            return Err(PreparedError::OutputSizeMismatch {
                expected: self.outputs,
                actual: outputs.len(),
            });
        }
        let actual = workspace.front.len().min(workspace.back.len());
        if actual < self.width {
            return Err(PreparedError::WorkspaceTooSmall {
                required: self.width,
                actual,
            });
        }
        if let Some(index) = inputs.iter().position(|value| !value.is_finite()) {
            return Err(PreparedError::NonFiniteInput { index });
        }
        let (mut source, mut target) = (&mut workspace.front, &mut workspace.back);
        // Copying the inputs into the workspace and reading them straight back
        // one scalar at a time defeats store-to-load forwarding and dominates
        // the cost of small networks. A dense first layer reads the caller's
        // slice in place instead; the bias source is exactly one, so its term
        // is the bias weight itself, added last as before (bit-identical).
        let mut first = 0;
        if self.direct_first {
            let layer = &self.layers[0];
            if inputs.len() >= 64 && layer.row_count >= 4 {
                self.run_dense_layer(
                    &self.rows[layer.row_start..layer.row_start + layer.row_count],
                    inputs,
                    self.input_bias,
                    target,
                    1,
                )?;
            } else {
                for (index, row) in self.rows[layer.row_start..layer.row_start + layer.row_count]
                    .iter()
                    .enumerate()
                {
                    let weights = &self.weights[row.start..row.start + row.len];
                    let mut sum = inputs
                        .iter()
                        .zip(weights)
                        .fold(T::zero(), |sum, (&value, &weight)| sum + value * weight);
                    if self.input_bias {
                        sum = sum + weights[self.inputs];
                    }
                    if !sum.is_finite() {
                        return Err(PreparedError::NumericOverflow {
                            layer: 1,
                            neuron: index,
                        });
                    }
                    let value = row.activation.apply(sum, row.steepness);
                    if !value.is_finite() {
                        return Err(PreparedError::NumericOverflow {
                            layer: 1,
                            neuron: index,
                        });
                    }
                    target[index] = value;
                }
            }
            if layer.bias {
                target[layer.row_count] = T::one();
            }
            std::mem::swap(&mut source, &mut target);
            first = 1;
        } else {
            source[..self.inputs].copy_from_slice(inputs);
            if self.input_bias {
                source[self.inputs] = T::one();
            }
        }
        for (layer_index, layer) in self.layers.iter().enumerate().skip(first) {
            let previous = &source[..layer.source_width];
            let rows = &self.rows[layer.row_start..layer.row_start + layer.row_count];
            if previous.len() >= 64
                && layer.row_count >= 4
                && rows.iter().all(|row| row.index_start == usize::MAX)
            {
                self.run_dense_layer(rows, previous, false, target, layer_index + 1)?;
            } else {
                for (index, row) in self.rows[layer.row_start..layer.row_start + layer.row_count]
                    .iter()
                    .enumerate()
                {
                    let weights = &self.weights[row.start..row.start + row.len];
                    let sum = if row.index_start == usize::MAX {
                        previous
                            .iter()
                            .zip(weights)
                            .fold(T::zero(), |sum, (&value, &weight)| sum + value * weight)
                    } else {
                        let indices = &self.indices[row.index_start..row.index_start + row.len];
                        indices
                            .iter()
                            .zip(weights)
                            .fold(T::zero(), |sum, (&source, &weight)| {
                                sum + previous[source] * weight
                            })
                    };
                    if !sum.is_finite() {
                        return Err(PreparedError::NumericOverflow {
                            layer: layer_index + 1,
                            neuron: index,
                        });
                    }
                    let value = row.activation.apply(sum, row.steepness);
                    if !value.is_finite() {
                        return Err(PreparedError::NumericOverflow {
                            layer: layer_index + 1,
                            neuron: index,
                        });
                    }
                    target[index] = value;
                }
            }
            if layer.bias {
                target[layer.row_count] = T::one();
            }
            std::mem::swap(&mut source, &mut target);
        }
        outputs.copy_from_slice(&source[..self.outputs]);
        Ok(())
    }
}
