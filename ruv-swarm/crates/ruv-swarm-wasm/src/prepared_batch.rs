//! Shared immutable parameters with bounded, preallocated row-major JS staging.
//! Work units count 2*edges + 3*computed regular neurons + 2*(inputs+outputs)
//! + bias rows + all layers, multiplied by capacity/active rows.
use super::{checked_bytes, inference_error, reserve, ArrayBoundary, BuildResult, MAX_BYTES};
use js_sys::Float64Array;
use ruv_fann::{BatchWorkspace, PreparedError, PreparedNetwork};
use std::{mem::size_of, rc::Rc};
use wasm_bindgen::prelude::*;

pub(super) const MAX_ROWS: usize = 4096;
const MAX_WORK: usize = 100_000_000;

/// A reusable batch shares its model with the source prepared object. All Rust
/// numerical buffers are allocated at creation; JS views are temporary copies.
#[wasm_bindgen]
pub struct WasmPreparedBatch {
    network: Rc<PreparedNetwork<f64>>,
    workspace: BatchWorkspace<f64>,
    inputs: Vec<f64>,
    outputs: Vec<f64>,
    arrays: Rc<ArrayBoundary>,
    estimated: usize,
    work_per_row: usize,
}

#[wasm_bindgen]
impl WasmPreparedBatch {
    #[wasm_bindgen(getter)]
    pub fn input_size(&self) -> usize {
        self.network.num_inputs()
    }

    #[wasm_bindgen(getter)]
    pub fn output_size(&self) -> usize {
        self.network.num_outputs()
    }

    #[wasm_bindgen(getter)]
    pub fn max_rows(&self) -> usize {
        self.workspace.max_rows()
    }

    /// Conservative combined snapshot/workspace/staging/cache storage. Shared
    /// values are counted in full; excludes caller JS arrays and allocator slack.
    #[wasm_bindgen(getter)]
    pub fn estimated_bytes(&self) -> usize {
        self.estimated
    }

    /// Row-major arrays must have exactly rows*input_size / rows*output_size
    /// elements. Zero rows requires valid empty Float64Arrays. Errors preserve
    /// the entire output; input/output aliases are safe. No Rust buffer grows.
    pub fn run_into(
        &mut self,
        inputs: &Float64Array,
        outputs: &Float64Array,
        #[wasm_bindgen(unchecked_param_type = "number")] rows: JsValue,
    ) -> Result<(), JsValue> {
        let rows = checked_rows(rows.as_f64(), 0, self.max_rows()).map_err(JsValue::from_str)?;
        let (input_len, output_len) =
            dimensions(&self.network, rows, self.work_per_row).map_err(JsValue::from_str)?;
        let memory = self.arrays.current_memory_buffer()?;
        let inputs = self.arrays.bounded(inputs, input_len, &memory)?;
        let outputs = self.arrays.bounded(outputs, output_len, &memory)?;
        inputs.copy_to(&mut self.inputs[..input_len]);
        self.evaluate(rows).map_err(inference_error)?;
        outputs.copy_from(&self.outputs[..output_len]);
        Ok(())
    }
}

impl WasmPreparedBatch {
    pub(super) fn new(
        network: Rc<PreparedNetwork<f64>>,
        arrays: Rc<ArrayBoundary>,
        max_rows: usize,
    ) -> BuildResult<Self> {
        if !(1..=MAX_ROWS).contains(&max_rows) {
            return Err("Batch capacity must be an integer in 1..4096");
        }
        let work_per_row = network
            .inference_work_per_row()
            .map_err(|_| "Batch inference work estimate overflow")?;
        let (input_len, output_len) = dimensions(&network, max_rows, work_per_row)?;
        let workspace_bytes = network
            .batch_workspace_bytes(max_rows)
            .map_err(|_| "Batch workspace dimensions overflow or exceed limits")?;
        let staging = checked_bytes(
            input_len
                .checked_add(output_len)
                .ok_or("Batch staging size overflow")?,
            size_of::<f64>(),
        )?;
        // Native workspace_bytes already includes its struct. Count the rest of
        // this struct, shared cache, and both Rc allocations' control words.
        let overhead = size_of::<Self>()
            .checked_sub(size_of::<BatchWorkspace<f64>>())
            .and_then(|n| n.checked_add(size_of::<ArrayBoundary>()))
            .and_then(|n| n.checked_add(4 * size_of::<usize>()))
            .ok_or("Batch metadata size overflow")?;
        let estimated = network
            .snapshot_bytes()
            .checked_add(workspace_bytes)
            .and_then(|n| n.checked_add(staging))
            .and_then(|n| n.checked_add(overhead))
            .ok_or("Batch combined storage overflow")?;
        if estimated > MAX_BYTES {
            return Err("Batch snapshot, workspace, staging and cache exceed 64 MiB");
        }
        let workspace = network
            .batch_workspace(max_rows, workspace_bytes)
            .map_err(|_| "Batch workspace allocation failed")?;
        let mut inputs = reserve(input_len)?;
        inputs.resize(input_len, 0.0);
        let mut outputs = reserve(output_len)?;
        outputs.resize(output_len, 0.0);
        Ok(Self {
            network,
            workspace,
            inputs,
            outputs,
            arrays,
            estimated,
            work_per_row,
        })
    }

    fn evaluate(&mut self, rows: usize) -> Result<(), PreparedError> {
        let input_len = self.input_size() * rows;
        let output_len = self.output_size() * rows;
        self.network.run_batch_into(
            &self.inputs[..input_len],
            &mut self.outputs[..output_len],
            rows,
            &mut self.workspace,
        )
    }
}

pub(super) fn checked_rows(
    value: Option<f64>,
    minimum: usize,
    maximum: usize,
) -> BuildResult<usize> {
    let value = value.ok_or("Batch rows must be a primitive number")?;
    if !value.is_finite()
        || value.fract() != 0.0
        || value < minimum as f64
        || value > maximum as f64
        || value.abs() > 9_007_199_254_740_991.0
    {
        return Err("Batch rows must be a bounded safe integer");
    }
    Ok(value as usize)
}

fn dimensions(
    network: &PreparedNetwork<f64>,
    rows: usize,
    work_per_row: usize,
) -> BuildResult<(usize, usize)> {
    let work = work_per_row
        .checked_mul(rows)
        .ok_or("Batch work size overflow")?;
    if work > MAX_WORK {
        return Err("Batch work exceeds 100,000,000 conservative inference units");
    }
    let inputs = network
        .num_inputs()
        .checked_mul(rows)
        .ok_or("Batch input size overflow")?;
    let outputs = network
        .num_outputs()
        .checked_mul(rows)
        .ok_or("Batch output size overflow")?;
    Ok((inputs, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{network::ActivationFunction as A, prepared::WasmPreparedNetwork};

    fn source(shape: &[usize]) -> WasmPreparedNetwork {
        let weights = shape.windows(2).map(|p| p[0] * p[1]).sum();
        let biases = shape[1..].iter().sum();
        WasmPreparedNetwork::from_dense(shape, &vec![0.0; weights], &vec![0.0; biases], A::Linear)
            .unwrap()
    }

    #[test]
    fn row_validation_rejects_fractional_nonfinite_and_truncating_values() {
        assert_eq!(checked_rows(Some(0.0), 0, MAX_ROWS), Ok(0));
        assert_eq!(checked_rows(Some(4096.0), 1, MAX_ROWS), Ok(4096));
        assert!(checked_rows(None, 1, MAX_ROWS).is_err());
        for value in [
            0.0,
            -1.0,
            1.5,
            f64::NAN,
            f64::INFINITY,
            4097.0,
            4_294_967_296.0,
            9_007_199_254_740_992.0,
        ] {
            assert!(checked_rows(Some(value), 1, MAX_ROWS).is_err());
        }
    }

    #[test]
    fn combined_memory_and_work_limits_precede_large_allocations() {
        let work = source(&[160, 160]);
        assert!(
            WasmPreparedBatch::new(Rc::clone(&work.network), Rc::clone(&work.arrays), 4096)
                .err()
                .unwrap()
                .contains("work")
        );
        let memory = source(&[15000, 1]);
        assert!(WasmPreparedBatch::new(
            Rc::clone(&memory.network),
            Rc::clone(&memory.arrays),
            1024
        )
        .err()
        .unwrap()
        .contains("64 MiB"));
        let small =
            WasmPreparedBatch::new(Rc::clone(&memory.network), Rc::clone(&memory.arrays), 1)
                .unwrap();
        assert!(small.estimated_bytes() <= MAX_BYTES);
        assert!(
            WasmPreparedBatch::new(Rc::clone(&memory.network), Rc::clone(&memory.arrays), 0)
                .is_err()
        );
    }

    #[test]
    fn batch_shares_snapshot_and_reuses_active_staging_after_source_drop() {
        let prepared =
            WasmPreparedNetwork::from_dense(&[2, 1], &[2.0, -1.0], &[0.0], A::Linear).unwrap();
        let mut batch =
            WasmPreparedBatch::new(Rc::clone(&prepared.network), Rc::clone(&prepared.arrays), 5)
                .unwrap();
        let input_pointer = batch.inputs.as_ptr();
        let output_pointer = batch.outputs.as_ptr();
        drop(prepared);
        for rows in [5, 1, 3, 0, 2, 5] {
            for row in 0..rows {
                batch.inputs[row * 2] = row as f64 + 3.0;
                batch.inputs[row * 2 + 1] = 4.0;
            }
            batch.evaluate(rows).unwrap();
            for row in 0..rows {
                assert_eq!(batch.outputs[row], row as f64 * 2.0 + 2.0);
            }
            assert_eq!(input_pointer, batch.inputs.as_ptr());
            assert_eq!(output_pointer, batch.outputs.as_ptr());
        }
    }

    #[test]
    fn later_row_overflow_leaves_all_staged_outputs_and_recovers() {
        let prepared =
            WasmPreparedNetwork::from_dense(&[1, 1], &[f64::MAX], &[0.0], A::Linear).unwrap();
        let mut batch =
            WasmPreparedBatch::new(Rc::clone(&prepared.network), Rc::clone(&prepared.arrays), 5)
                .unwrap();
        batch.inputs.copy_from_slice(&[0.0, 0.25, 0.5, 0.125, 2.0]);
        batch
            .outputs
            .copy_from_slice(&[11.0, 12.0, 13.0, 14.0, 15.0]);
        assert!(batch.evaluate(5).is_err());
        assert_eq!(batch.outputs, [11.0, 12.0, 13.0, 14.0, 15.0]);
        batch.evaluate(3).unwrap();
        assert_eq!(&batch.outputs[..3], &[0.0, f64::MAX * 0.25, f64::MAX * 0.5]);
    }

    #[test]
    fn disconnected_deep_activations_are_included_in_the_work_budget() {
        use ruv_fann::{ActivationFunction as Native, Layer, Network, Neuron};
        let graph = Network {
            layers: (0..30_000)
                .map(|_| Layer {
                    neurons: vec![Neuron::new(Native::Sin, 1.0)],
                })
                .collect(),
            connection_rate: 0.0,
        };
        let prepared = WasmPreparedNetwork::from_native(&graph).unwrap();
        assert_eq!(prepared.network.connection_count(), 0);
        assert_eq!(prepared.network.inference_work_per_row().unwrap(), 120001);
        assert!(WasmPreparedBatch::new(
            Rc::clone(&prepared.network),
            Rc::clone(&prepared.arrays),
            4096
        )
        .err()
        .unwrap()
        .contains("work"));
        let mut small =
            WasmPreparedBatch::new(Rc::clone(&prepared.network), Rc::clone(&prepared.arrays), 1)
                .unwrap();
        small.evaluate(1).unwrap();
        assert_eq!(small.outputs, [0.5]);
    }
}
