//! Direct packing of Candle-style affine parameters, without a neuron graph.
use super::{
    add, budget, bytes, invalid, reserved, InferenceWorkspace, PackedLayer, PreparationLimits,
    PreparedError, PreparedNetwork, Row,
};
use crate::ActivationFunction;
use num_traits::Float;
use std::mem::size_of;

/// Versioned interchange schema used by the optional JSON constructor.
pub const DENSE_LAYERS_SCHEMA: &str = "ruv-fann.dense-layers/1";

/// Borrowed affine layer: `y = activation(x * weights.t() + bias, steepness)`.
/// Weights are contiguous logical `[outputs, inputs]`, as in Candle's Linear.
/// A present bias is added after input products, even when its entries are zero.
/// Packing copies the parameters; their source may subsequently be dropped.
#[derive(Debug, Clone, Copy)]
pub struct DenseLayer<'a, T: Float> {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: &'a [T],
    pub bias: Option<&'a [T]>,
    pub activation: ActivationFunction,
    pub steepness: T,
}

impl<T: Float> PreparedNetwork<T> {
    /// Prepare dense affine layers without allocating a mutable Network graph.
    /// Preserves the native per-neuron product/addition order; this is not a
    /// general Candle tensor/autograd backend or a reassociating GEMM call.
    pub fn from_dense_layers(layers: &[DenseLayer<'_, T>]) -> Result<Self, PreparedError> {
        Self::from_dense_layers_with_limits(layers, PreparationLimits::default())
    }

    /// Preflight all dimensions, adjacency, storage and finite parameters before
    /// packed allocation. `max_bytes` includes one ordinary inference workspace,
    /// as for from_network_with_limits; caller source buffers are excluded.
    pub fn from_dense_layers_with_limits(
        layers: &[DenseLayer<'_, T>],
        limits: PreparationLimits,
    ) -> Result<Self, PreparedError> {
        let first = layers
            .first()
            .ok_or_else(|| invalid(0, 0, "no dense layers"))?;
        let mut connections = 0usize;
        let mut rows = 0usize;
        let mut width = add(first.inputs, usize::from(first.bias.is_some()))?;
        let mut neurons = width;
        for (index, layer) in layers.iter().enumerate() {
            if layer.inputs == 0 || layer.outputs == 0 {
                return Err(invalid(index + 1, 0, "dense dimensions must be positive"));
            }
            if index > 0 && layers[index - 1].outputs != layer.inputs {
                return Err(invalid(index + 1, 0, "adjacent dense dimensions disagree"));
            }
            if layer.weights.len() != bytes(layer.inputs, layer.outputs)? {
                return Err(invalid(
                    index + 1,
                    0,
                    "dense weight shape must be outputs by inputs",
                ));
            }
            if layer.bias.is_some_and(|bias| bias.len() != layer.outputs) {
                return Err(invalid(
                    index + 1,
                    0,
                    "dense bias length must match outputs",
                ));
            }
            let source_width = add(layer.inputs, usize::from(layer.bias.is_some()))?;
            connections = add(connections, bytes(source_width, layer.outputs)?)?;
            rows = add(rows, layer.outputs)?;
            let next_bias = layers
                .get(index + 1)
                .is_some_and(|next| next.bias.is_some());
            let target_width = add(layer.outputs, usize::from(next_bias))?;
            width = width.max(source_width).max(target_width);
            neurons = add(neurons, target_width)?;
            budget("neurons", neurons, limits.max_neurons)?;
            budget("connections", connections, limits.max_connections)?;
        }
        let mut estimated = add(size_of::<Self>(), size_of::<InferenceWorkspace<T>>())?;
        for allocation in [
            bytes(connections, size_of::<T>())?,
            bytes(rows, size_of::<Row<T>>())?,
            bytes(layers.len(), size_of::<PackedLayer>())?,
            bytes(bytes(width, 2)?, size_of::<T>())?,
        ] {
            estimated = add(estimated, allocation)?;
        }
        budget("snapshot and workspace bytes", estimated, limits.max_bytes)?;
        for (index, layer) in layers.iter().enumerate() {
            if !layer.steepness.is_finite() {
                return Err(invalid(index + 1, 0, "dense steepness must be finite"));
            }
            if layer.weights.iter().any(|x| !x.is_finite())
                || layer
                    .bias
                    .is_some_and(|bias| bias.iter().any(|x| !x.is_finite()))
            {
                return Err(invalid(index + 1, 0, "dense parameters must be finite"));
            }
        }
        let mut prepared = Self {
            weights: reserved(connections)?,
            indices: Vec::new(),
            rows: reserved(rows)?,
            layers: reserved(layers.len())?,
            inputs: first.inputs,
            outputs: layers.last().expect("nonempty layers").outputs,
            width,
            input_bias: first.bias.is_some(),
            direct_first: false,
            single_tiled: false,
            estimated_bytes: estimated,
        };
        for (index, layer) in layers.iter().enumerate() {
            let row_start = prepared.rows.len();
            let source_width = add(layer.inputs, usize::from(layer.bias.is_some()))?;
            for output in 0..layer.outputs {
                let start = prepared.weights.len();
                prepared.weights.extend_from_slice(
                    &layer.weights[output * layer.inputs..(output + 1) * layer.inputs],
                );
                if let Some(bias) = layer.bias {
                    prepared.weights.push(bias[output]);
                }
                prepared.rows.push(Row {
                    start,
                    len: source_width,
                    index_start: usize::MAX,
                    activation: layer.activation,
                    steepness: layer.steepness,
                });
            }
            prepared.layers.push(PackedLayer {
                row_start,
                row_count: layer.outputs,
                source_width,
                bias: layers
                    .get(index + 1)
                    .is_some_and(|next| next.bias.is_some()),
                single_tiled: layer.outputs >= 4
                    && if index == 0 {
                        layer.inputs >= 64
                    } else {
                        source_width >= 64
                    },
            });
        }
        prepared.direct_first = prepared.first_layer_is_dense();
        prepared.single_tiled = prepared.layers.iter().any(|layer| layer.single_tiled);
        Ok(prepared)
    }
}
