//! Bounded shared dense-layer schema. This stores topology and parameters only;
//! it does not checkpoint gradients, optimizer moments or training progress.
use super::{budget, reserved, AutogradError, AutogradNetwork};
use crate::{ActivationFunction, BatchFloat, DenseLayer, PreparationLimits, DENSE_LAYERS_SCHEMA};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

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
#[derive(Serialize)]
struct StackRef<'a, T> {
    schema: &'static str,
    layers: Vec<LayerRef<'a, T>>,
}
#[derive(Serialize)]
struct LayerRef<'a, T> {
    inputs: usize,
    outputs: usize,
    weights: &'a [T],
    bias: Option<&'a [T]>,
    activation: ActivationFunction,
    steepness: T,
}

impl<T: BatchFloat + serde::de::DeserializeOwned> AutogradNetwork<T> {
    /// Read at most 4 MiB UTF-8 JSON and 64 layers using the same versioned
    /// schema as PreparedNetwork. Unknown fields and activation rules fail.
    pub fn from_dense_layers_json(json: &str) -> Result<Self, AutogradError> {
        Self::from_dense_layers_json_with_limits(json, PreparationLimits::default())
    }
    /// The byte limit covers owned model storage; decoded JSON vectors are
    /// temporary additional storage bounded by the separate 4 MiB input cap.
    pub fn from_dense_layers_json_with_limits(
        json: &str,
        limits: PreparationLimits,
    ) -> Result<Self, AutogradError> {
        budget("dense JSON bytes", json.len(), MAX_JSON_BYTES)?;
        let stack: Stack<T> = serde_json::from_str(json)
            .map_err(|_| AutogradError::InvalidJson("malformed document"))?;
        if stack.schema != DENSE_LAYERS_SCHEMA {
            return Err(AutogradError::InvalidJson("unsupported schema"));
        }
        budget("dense JSON layers", stack.layers.len(), MAX_LAYERS)?;
        let mut layers = reserved(stack.layers.len())?;
        for layer in &stack.layers {
            layers.push(DenseLayer {
                inputs: layer.inputs,
                outputs: layer.outputs,
                weights: &layer.weights,
                bias: layer.bias.as_deref(),
                activation: layer.activation,
                steepness: layer.steepness,
            });
        }
        Self::from_dense_layers_with_limits(&layers, limits)
    }
}

impl<T: BatchFloat + Serialize> AutogradNetwork<T> {
    /// Export owned topology and current parameters, without optimizer state.
    /// Serialization stops at the 4 MiB output limit, rather than allocating an
    /// unbounded String and checking its size afterwards.
    pub fn to_dense_layers_json(&self) -> Result<String, AutogradError> {
        budget("dense JSON layers", self.layers.len(), MAX_LAYERS)?;
        let mut layers = reserved(self.layers.len())?;
        for layer in self.dense_layers()? {
            layers.push(LayerRef {
                inputs: layer.inputs,
                outputs: layer.outputs,
                weights: layer.weights,
                bias: layer.bias,
                activation: layer.activation,
                steepness: layer.steepness,
            });
        }
        let mut writer = LimitedWriter {
            bytes: Vec::new(),
            failure: None,
        };
        if serde_json::to_writer(
            &mut writer,
            &StackRef {
                schema: DENSE_LAYERS_SCHEMA,
                layers,
            },
        )
        .is_err()
        {
            return Err(writer
                .failure
                .unwrap_or(AutogradError::InvalidJson("serialization failed")));
        }
        String::from_utf8(writer.bytes)
            .map_err(|_| AutogradError::InvalidJson("serialization was not UTF-8"))
    }
}

struct LimitedWriter {
    bytes: Vec<u8>,
    failure: Option<AutogradError>,
}
impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let required = self.bytes.len().checked_add(bytes.len());
        if required.is_none() || required.is_some_and(|size| size > MAX_JSON_BYTES) {
            self.failure = Some(AutogradError::BudgetExceeded {
                resource: "dense JSON bytes",
                required: required.unwrap_or(usize::MAX),
                limit: MAX_JSON_BYTES,
            });
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "dense JSON byte limit",
            ));
        }
        let required = required.expect("checked above");
        if required > self.bytes.capacity() {
            let doubled = self
                .bytes
                .capacity()
                .max(256)
                .checked_mul(2)
                .unwrap_or(MAX_JSON_BYTES);
            let capacity = doubled.min(MAX_JSON_BYTES).max(required);
            if self
                .bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .is_err()
            {
                self.failure = Some(AutogradError::AllocationFailed);
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "dense JSON allocation",
                ));
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
