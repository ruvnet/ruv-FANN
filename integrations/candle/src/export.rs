use crate::{materialize, preflight, CandleAdapterError as Error, LinearStage};
use candle_core::WithDType;
use ruv_fann::{ActivationFunction, BatchFloat, PreparationLimits, DENSE_LAYERS_SCHEMA};
use serde::Serialize;
use std::io::{self, Write};

const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;

/// Export actual f64 Candle weights to the bounded dense-layer interchange.
/// This precision matches the current WASM and napi-rs prepared bindings.
pub fn export_dense_f64_json(stages: &[LinearStage<'_, f64>]) -> Result<String, Error> {
    export(stages)
}

/// Export f32 values for a f32 Rust parser. Parsing this format as f64 does not
/// preserve f32 arithmetic; no automatic mixed-precision equivalence is claimed.
pub fn export_dense_f32_json(stages: &[LinearStage<'_, f32>]) -> Result<String, Error> {
    export(stages)
}

#[derive(Serialize)]
struct Stack<'a, T> {
    schema: &'static str,
    layers: Vec<Layer<'a, T>>,
}
#[derive(Serialize)]
struct Layer<'a, T> {
    inputs: usize,
    outputs: usize,
    weights: &'a [T],
    bias: Option<&'a [T]>,
    activation: ActivationFunction,
    steepness: T,
}

fn export<T: BatchFloat + WithDType + Serialize>(
    stages: &[LinearStage<'_, T>],
) -> Result<String, Error> {
    let plan = preflight::validate(stages, PreparationLimits::default())?;
    // Every finite parameter needs at least one digit and an array separator
    // or bracket. Reject provably oversized JSON before reading source storage.
    let minimum_bytes = plan.parameters.checked_mul(2).ok_or(Error::SizeOverflow)?;
    if minimum_bytes > MAX_JSON_BYTES {
        return Err(Error::Budget {
            resource: "dense JSON bytes",
            required: minimum_bytes,
            limit: MAX_JSON_BYTES,
        });
    }
    let owned = materialize(stages, &plan)?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(stages.len())
        .map_err(|_| Error::AllocationFailed)?;
    for (index, (stage, data)) in stages.iter().zip(&owned).enumerate() {
        let shape = plan.shapes[index];
        layers.push(Layer {
            inputs: shape.inputs,
            outputs: shape.outputs,
            weights: &data.weights,
            bias: data.bias.as_deref(),
            activation: stage.activation,
            steepness: stage.steepness,
        });
    }
    let mut writer = CappedWriter {
        bytes: Vec::new(),
        limit: MAX_JSON_BYTES,
        overflow: false,
        allocation_failed: false,
    };
    if let Err(error) = serde_json::to_writer(
        &mut writer,
        &Stack {
            schema: DENSE_LAYERS_SCHEMA,
            layers,
        },
    ) {
        return if writer.overflow {
            Err(Error::Budget {
                resource: "dense JSON bytes",
                required: MAX_JSON_BYTES + 1,
                limit: MAX_JSON_BYTES,
            })
        } else if writer.allocation_failed {
            Err(Error::AllocationFailed)
        } else {
            Err(Error::Json(error))
        };
    }
    // serde_json emits UTF-8, without a second materialization or unchecked conversion.
    String::from_utf8(writer.bytes).map_err(|_| Error::Invalid {
        stage: 0,
        reason: "serializer produced invalid UTF-8",
    })
}

struct CappedWriter {
    bytes: Vec<u8>,
    limit: usize,
    overflow: bool,
    allocation_failed: bool,
}
impl Write for CappedWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.overflow = true;
            return Err(io::Error::other("dense JSON byte limit exceeded"));
        }
        let required = self.bytes.len() + input.len();
        if required > self.bytes.capacity() {
            let target = self
                .bytes
                .capacity()
                .max(64)
                .saturating_mul(2)
                .min(self.limit)
                .max(required);
            if self
                .bytes
                .try_reserve_exact(target - self.bytes.len())
                .is_err()
            {
                self.allocation_failed = true;
                return Err(io::Error::other("dense JSON allocation failed"));
            }
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cap_accepts_exact_boundary_and_never_retains_a_truncated_chunk() {
        let mut writer = CappedWriter {
            bytes: Vec::new(),
            limit: 4,
            overflow: false,
            allocation_failed: false,
        };
        writer.write_all(b"1234").unwrap();
        assert!(writer.write_all(b"56").is_err());
        assert_eq!(writer.bytes, b"1234");
        assert!(writer.overflow);
    }
}
