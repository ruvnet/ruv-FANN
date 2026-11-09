//! Safe Rust state only. JS memory is copied exclusively by gateway.rs.
#![forbid(unsafe_code)]
use crate::gateway::{invalid, MAX_BYTES, MAX_ROWS};
use napi::Result;
use ruv_fann::{BatchWorkspace, InferenceWorkspace, Network, PreparationLimits, PreparedNetwork};
use std::{mem::size_of, sync::Arc};

const MAX_WORK: usize = 100_000_000;
pub fn size_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Native storage size overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("Native dimensions overflow"))
}
fn zeros(length: usize) -> Result<Vec<f64>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| invalid("Native staging allocation failed"))?;
    result.resize(length, 0.0);
    Ok(result)
}
fn budget(value: usize) -> Result<usize> {
    if value > MAX_BYTES {
        Err(invalid(
            "Combined native model/workspace/staging exceeds 64 MiB",
        ))
    } else {
        Ok(value)
    }
}

pub struct PreparedState {
    pub model: Arc<PreparedNetwork<f64>>,
    pub workspace: InferenceWorkspace<f64>,
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    pub estimated: usize,
}

impl PreparedState {
    pub fn from_network_json(text: &str) -> Result<Self> {
        let graph: Network<f64> = serde_json::from_str(text)
            .map_err(|error| invalid(format!("Invalid native Network<f64> JSON: {error}")))?;
        let staging = size_add(
            mul(size_add(graph.num_inputs(), graph.num_outputs())?, 8)?,
            size_of::<Self>(),
        )?;
        let limit = MAX_BYTES
            .checked_sub(staging)
            .ok_or_else(|| invalid("Native staging exceeds 64 MiB"))?;
        let model = PreparedNetwork::from_network_with_limits(
            &graph,
            PreparationLimits {
                max_bytes: limit,
                ..PreparationLimits::default()
            },
        )
        .map_err(|error| invalid(error.to_string()))?;
        Self::new(model)
    }

    pub fn from_dense_json(text: &str) -> Result<Self> {
        let model = PreparedNetwork::<f64>::from_dense_layers_json(text)
            .map_err(|error| invalid(error.to_string()))?;
        Self::new(model)
    }

    pub fn new(model: PreparedNetwork<f64>) -> Result<Self> {
        let inputs = model.num_inputs();
        let outputs = model.num_outputs();
        let estimated = budget(size_add(
            size_add(model.estimated_bytes(), mul(size_add(inputs, outputs)?, 8)?)?,
            size_of::<Self>() + 2 * size_of::<usize>(),
        )?)?;
        let workspace = model
            .workspace()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            model: Arc::new(model),
            workspace,
            inputs: zeros(inputs)?,
            outputs: zeros(outputs)?,
            estimated,
        })
    }

    pub fn evaluate(&mut self) -> Result<()> {
        self.model
            .run_into(&self.inputs, &mut self.outputs, &mut self.workspace)
            .map_err(|error| invalid(error.to_string()))
    }
}

pub struct BatchState {
    pub model: Arc<PreparedNetwork<f64>>,
    pub workspace: BatchWorkspace<f64>,
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    pub estimated: usize,
    pub capacity: usize,
}

impl BatchState {
    pub fn new(model: Arc<PreparedNetwork<f64>>, rows: usize) -> Result<Self> {
        if !(1..=MAX_ROWS).contains(&rows) {
            return Err(invalid("Batch capacity must be in 1..4096"));
        }
        let work = mul(
            model
                .inference_work_per_row()
                .map_err(|error| invalid(error.to_string()))?,
            rows,
        )?;
        if work > MAX_WORK {
            return Err(invalid("Batch inference work exceeds 100,000,000 units"));
        }
        let inputs = mul(model.num_inputs(), rows)?;
        let outputs = mul(model.num_outputs(), rows)?;
        let workspace_bytes = model
            .batch_workspace_bytes(rows)
            .map_err(|error| invalid(error.to_string()))?;
        let estimated = budget(size_add(
            size_add(
                size_add(model.snapshot_bytes(), workspace_bytes)?,
                mul(size_add(inputs, outputs)?, 8)?,
            )?,
            size_of::<Self>() + 2 * size_of::<usize>(),
        )?)?;
        let workspace = model
            .batch_workspace(rows, workspace_bytes)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            model,
            workspace,
            inputs: zeros(inputs)?,
            outputs: zeros(outputs)?,
            estimated,
            capacity: rows,
        })
    }

    pub fn evaluate(&mut self, rows: usize) -> Result<()> {
        let ni = mul(rows, self.model.num_inputs())?;
        let no = mul(rows, self.model.num_outputs())?;
        self.model
            .run_batch_into(
                &self.inputs[..ni],
                &mut self.outputs[..no],
                rows,
                &mut self.workspace,
            )
            .map_err(|error| invalid(error.to_string()))
    }
}
