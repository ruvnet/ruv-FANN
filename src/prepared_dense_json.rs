//! Bounded versioned dense interchange. Not the existing Network serde format.
use super::{
    budget, invalid, DenseLayer, PreparationLimits, PreparedError, PreparedNetwork,
    DENSE_LAYERS_SCHEMA,
};
use crate::ActivationFunction;
use num_traits::Float;
use serde::Deserialize;

const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;
const MAX_LAYERS: usize = 64;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Stack<T> {
    schema: String,
    layers: Vec<OwnedLayer<T>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedLayer<T> {
    inputs: usize,
    outputs: usize,
    weights: Vec<T>,
    bias: Option<Vec<T>>,
    activation: ActivationFunction,
    steepness: T,
}

impl<T: Float + serde::de::DeserializeOwned> PreparedNetwork<T> {
    /// Import at most 4 MiB UTF-8 JSON and 64 dense layers. The versioned schema
    /// uses `[outputs, inputs]` flat weights and optional per-output bias. Serde's
    /// nesting limit remains enabled; unknown fields and schema versions fail.
    pub fn from_dense_layers_json(json: &str) -> Result<Self, PreparedError> {
        Self::from_dense_layers_json_with_limits(json, PreparationLimits::default())
    }

    /// JSON decoding owns temporary vectors in addition to the packed snapshot;
    /// max_bytes bounds packed/workspace storage, not peak process memory.
    pub fn from_dense_layers_json_with_limits(
        json: &str,
        limits: PreparationLimits,
    ) -> Result<Self, PreparedError> {
        budget("dense JSON bytes", json.len(), MAX_JSON_BYTES)?;
        let stack: Stack<T> =
            serde_json::from_str(json).map_err(|_| invalid(0, 0, "invalid dense-layer JSON"))?;
        if stack.schema != DENSE_LAYERS_SCHEMA {
            return Err(invalid(0, 0, "unsupported dense-layer JSON schema"));
        }
        budget("dense JSON layers", stack.layers.len(), MAX_LAYERS)?;
        let layers = stack
            .layers
            .iter()
            .map(|layer| DenseLayer {
                inputs: layer.inputs,
                outputs: layer.outputs,
                weights: &layer.weights,
                bias: layer.bias.as_deref(),
                activation: layer.activation,
                steepness: layer.steepness,
            })
            .collect::<Vec<_>>();
        Self::from_dense_layers_with_limits(&layers, limits)
    }
}
