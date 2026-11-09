//! Actual shared reverse-mode MSE, SGD and AdamW. Caller arrays are validated
//! intrinsically and copied into fixed Rust staging; live WASM views never leak.
use super::{
    batch::checked_rows, dense_import, reserve, ArrayBoundary, WasmPreparedNetwork, MAX_BYTES,
};
use js_sys::Float64Array;
use ruv_fann::{AdamW, AdamWConfig, AutogradError, AutogradNetwork, AutogradWorkspace};
use std::{mem::size_of, rc::Rc};
use wasm_bindgen::prelude::*;

const MAX_WORK: usize = 100_000_000;

#[wasm_bindgen]
pub struct WasmTrainableNetwork {
    network: AutogradNetwork<f64>,
    workspace: AutogradWorkspace<f64>,
    optimizer: Option<AdamW<f64>>,
    inputs: Vec<f64>,
    targets: Vec<f64>,
    arrays: Rc<ArrayBoundary>,
    estimated: usize,
}

#[wasm_bindgen]
impl WasmTrainableNetwork {
    /// Fixed-capacity f64 training. Combined requested model, workspace, staging
    /// and one AdamW state are capped at 64 MiB, excluding caller JS arrays.
    pub fn from_dense_layers_json(
        #[wasm_bindgen(unchecked_param_type = "string")] json: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] max_rows: JsValue,
    ) -> Result<WasmTrainableNetwork, JsValue> {
        let rows = checked_rows(max_rows.as_f64(), 1, 4096).map_err(JsValue::from_str)?;
        let text = dense_import::json_text(json)?;
        let network =
            AutogradNetwork::from_dense_layers_json_with_limits(&text, dense_import::limits())
                .map_err(training_error)?;
        work(
            network
                .training_work_estimate(rows)
                .map_err(training_error)?,
        )?;
        work(network.optimizer_work_estimate().map_err(training_error)?)?;
        let workspace_bytes = network.workspace_bytes(rows).map_err(training_error)?;
        let optimizer_bytes = AdamW::required_bytes(&network).map_err(training_error)?;
        let input_len = network
            .num_inputs()
            .checked_mul(rows)
            .ok_or_else(storage_error)?;
        let target_len = network
            .num_outputs()
            .checked_mul(rows)
            .ok_or_else(storage_error)?;
        let staging = input_len
            .checked_add(target_len)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(storage_error)?;
        let estimated = network
            .estimated_bytes()
            .checked_add(workspace_bytes)
            .and_then(|n| n.checked_add(optimizer_bytes))
            .and_then(|n| n.checked_add(staging))
            .and_then(|n| {
                n.checked_add(
                    size_of::<Self>() + size_of::<ArrayBoundary>() + 2 * size_of::<usize>(),
                )
            })
            .ok_or_else(storage_error)?;
        if estimated > MAX_BYTES {
            return Err(storage_error());
        }
        let workspace = network
            .workspace(rows, workspace_bytes)
            .map_err(training_error)?;
        let mut inputs = reserve(input_len).map_err(JsValue::from_str)?;
        inputs.resize(input_len, 0.0);
        let mut targets = reserve(target_len).map_err(JsValue::from_str)?;
        targets.resize(target_len, 0.0);
        Ok(Self {
            network,
            workspace,
            optimizer: None,
            inputs,
            targets,
            arrays: Rc::new(ArrayBoundary::new()?),
            estimated,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn input_size(&self) -> usize {
        self.network.num_inputs()
    }
    #[wasm_bindgen(getter)]
    pub fn output_size(&self) -> usize {
        self.network.num_outputs()
    }
    #[wasm_bindgen(getter)]
    pub fn parameter_count(&self) -> usize {
        self.network.parameter_count()
    }
    #[wasm_bindgen(getter)]
    pub fn max_rows(&self) -> usize {
        self.workspace.max_rows()
    }
    #[wasm_bindgen(getter)]
    pub fn estimated_bytes(&self) -> usize {
        self.estimated
    }

    /// Mean squared error over all rows and outputs. Gradients replace the
    /// previous pass and weights change only after an explicit optimizer step.
    pub fn mse_backward(
        &mut self,
        inputs: &Float64Array,
        targets: &Float64Array,
        #[wasm_bindgen(unchecked_param_type = "number")] rows: JsValue,
    ) -> Result<f64, JsValue> {
        let rows = checked_rows(rows.as_f64(), 1, self.max_rows()).map_err(JsValue::from_str)?;
        work(
            self.network
                .training_work_estimate(rows)
                .map_err(training_error)?,
        )?;
        let ni = rows * self.input_size();
        let no = rows * self.output_size();
        let memory = self.arrays.current_memory_buffer()?;
        let inputs = self.arrays.bounded(inputs, ni, &memory)?;
        let targets = self.arrays.bounded(targets, no, &memory)?;
        inputs.copy_to(&mut self.inputs[..ni]);
        targets.copy_to(&mut self.targets[..no]);
        self.network
            .backward_mse(
                &self.inputs[..ni],
                &self.targets[..no],
                rows,
                &mut self.workspace,
            )
            .map_err(training_error)
    }

    /// Owned copies; no returned array aliases live training buffers.
    pub fn parameter_gradients(&self) -> Vec<f64> {
        self.workspace.parameter_gradients().to_vec()
    }
    pub fn input_gradients(&self) -> Vec<f64> {
        self.workspace.input_gradients().to_vec()
    }
    pub fn predictions(&self) -> Vec<f64> {
        self.workspace.outputs().to_vec()
    }

    pub fn sgd_step(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "number")] learning_rate: JsValue,
    ) -> Result<(), JsValue> {
        let rate = scalar(learning_rate)?;
        self.network
            .sgd_step(&mut self.workspace, rate)
            .map_err(training_error)
    }

    /// Reset AdamW's moments. Existing optimizer remains intact on validation or
    /// allocation failure. The temporary old and new states share the byte cap.
    pub fn configure_adamw(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "number")] learning_rate: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] beta1: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] beta2: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] epsilon: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] weight_decay: JsValue,
    ) -> Result<(), JsValue> {
        let config = AdamWConfig {
            learning_rate: scalar(learning_rate)?,
            beta1: scalar(beta1)?,
            beta2: scalar(beta2)?,
            epsilon: scalar(epsilon)?,
            weight_decay: scalar(weight_decay)?,
        };
        let bytes = AdamW::required_bytes(&self.network).map_err(training_error)?;
        if self.optimizer.is_some()
            && self
                .estimated
                .checked_add(bytes)
                .map_or(true, |n| n > MAX_BYTES)
        {
            return Err(storage_error());
        }
        self.optimizer = Some(AdamW::new(&self.network, config, bytes).map_err(training_error)?);
        Ok(())
    }

    pub fn adamw_step(&mut self) -> Result<(), JsValue> {
        let optimizer = self
            .optimizer
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Configure AdamW first"))?;
        self.network
            .adamw_step(&mut self.workspace, optimizer)
            .map_err(training_error)
    }

    /// Freeze independent owned weights for the existing inference/SDK API.
    pub fn prepare(&self) -> Result<WasmPreparedNetwork, JsValue> {
        WasmPreparedNetwork::from_snapshot(self.network.prepare().map_err(training_error)?)
            .map_err(JsValue::from_str)
    }
    /// Topology/weight checkpoint; optimizer moments are not serialized.
    pub fn to_dense_layers_json(&self) -> Result<String, JsValue> {
        self.network.to_dense_layers_json().map_err(training_error)
    }
}

fn scalar(value: JsValue) -> Result<f64, JsValue> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| JsValue::from_str("Optimizer options must be primitive finite numbers"))
}
fn work(units: usize) -> Result<(), JsValue> {
    if units > MAX_WORK {
        Err(JsValue::from_str(
            "Training exceeds 100,000,000 conservative work units",
        ))
    } else {
        Ok(())
    }
}
fn storage_error() -> JsValue {
    JsValue::from_str("Training storage exceeds 64 MiB or overflows")
}
fn training_error(error: AutogradError) -> JsValue {
    // Static text avoids growing WASM memory on hot error paths.
    JsValue::from_str(match error {
        AutogradError::StaleGradients => "Gradients are missing or stale; run mse_backward first",
        AutogradError::StaleOptimizer => "Optimizer is stale; configure AdamW again",
        AutogradError::UnsupportedActivation { .. } => {
            "Training supports Linear, Sigmoid, Tanh, SigmoidSymmetric and ReLU"
        }
        AutogradError::NumericOverflow { .. } | AutogradError::NonFinite { .. } => {
            "Training values and arithmetic must be finite"
        }
        AutogradError::InvalidOptimizerConfig { .. } => "Invalid optimizer options",
        _ => "Invalid training shape, model, workspace or resource limit",
    })
}
