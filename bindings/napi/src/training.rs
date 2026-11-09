//! Actual core reverse-mode MSE and SGD/AdamW; no JS-owned memory is borrowed.
use crate::{
    gateway::{self, invalid, MAX_BYTES, MAX_ROWS},
    live, live_mut,
    state::PreparedState,
    NativePreparedNetwork,
};
use napi::{
    bindgen_prelude::{Float64Array, Unknown},
    Env, Result,
};
use napi_derive::napi;
use ruv_fann::{AdamW, AdamWConfig, AutogradNetwork, AutogradWorkspace};
use std::{cell::RefCell, mem::size_of};

const MAX_WORK: usize = 100_000_000;
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Training storage overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("Training dimensions overflow"))
}
fn bounded(bytes: usize) -> Result<usize> {
    if bytes > MAX_BYTES {
        Err(invalid(
            "Combined training model/workspace/optimizer/staging exceeds 64 MiB",
        ))
    } else {
        Ok(bytes)
    }
}
fn zeros(len: usize) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| invalid("Training staging allocation failed"))?;
    values.resize(len, 0.0);
    Ok(values)
}
fn owned(values: &[f64]) -> Result<Float64Array> {
    bounded(mul(values.len(), 8)?)?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len())
        .map_err(|_| invalid("Gradient output allocation failed"))?;
    copy.extend_from_slice(values);
    Ok(Float64Array::new(copy))
}
fn work(model: &AutogradNetwork<f64>, rows: usize) -> Result<()> {
    if model
        .training_work_estimate(rows)
        .map_err(|error| invalid(error.to_string()))?
        > MAX_WORK
    {
        Err(invalid(
            "Training forward/reverse work exceeds 100,000,000 conservative units",
        ))
    } else {
        Ok(())
    }
}

struct TrainingState {
    model: AutogradNetwork<f64>,
    workspace: AutogradWorkspace<f64>,
    optimizer: Option<AdamW<f64>>,
    inputs: Vec<f64>,
    targets: Vec<f64>,
    capacity: usize,
    base_bytes: usize,
}
impl TrainingState {
    fn new(text: &str, capacity: usize) -> Result<Self> {
        let model = AutogradNetwork::<f64>::from_dense_layers_json(text)
            .map_err(|error| invalid(error.to_string()))?;
        work(&model, capacity)?;
        let ni = mul(model.num_inputs(), capacity)?;
        let no = mul(model.num_outputs(), capacity)?;
        let workspace_bytes = model
            .workspace_bytes(capacity)
            .map_err(|error| invalid(error.to_string()))?;
        let base_bytes = bounded(add(
            add(
                add(model.estimated_bytes(), workspace_bytes)?,
                mul(add(ni, no)?, 8)?,
            )?,
            size_of::<Self>(),
        )?)?;
        let workspace = model
            .workspace(capacity, workspace_bytes)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            model,
            workspace,
            optimizer: None,
            inputs: zeros(ni)?,
            targets: zeros(no)?,
            capacity,
            base_bytes,
        })
    }
    fn optimizer_work(&self) -> Result<()> {
        if self
            .model
            .optimizer_work_estimate()
            .map_err(|error| invalid(error.to_string()))?
            > MAX_WORK
        {
            Err(invalid(
                "Optimizer work exceeds 100,000,000 conservative units",
            ))
        } else {
            Ok(())
        }
    }
}

#[napi]
pub struct NativeTrainableNetwork {
    state: RefCell<Option<TrainingState>>,
}

#[napi]
impl NativeTrainableNetwork {
    #[napi(factory, js_name = "from_dense_layers_json")]
    pub fn from_dense_layers_json(
        env: Env,
        json: Unknown<'_>,
        max_rows: Unknown<'_>,
    ) -> Result<Self> {
        let text = gateway::json_string(env, json)?;
        let capacity = gateway::rows(env, max_rows, 1, MAX_ROWS)?;
        Ok(Self {
            state: RefCell::new(Some(TrainingState::new(&text, capacity)?)),
        })
    }
    #[napi(getter, js_name = "input_size")]
    pub fn input_size(&self) -> Result<u32> {
        Ok(live(&self.state)?.model.num_inputs() as u32)
    }
    #[napi(getter, js_name = "output_size")]
    pub fn output_size(&self) -> Result<u32> {
        Ok(live(&self.state)?.model.num_outputs() as u32)
    }
    #[napi(getter, js_name = "parameter_count")]
    pub fn parameter_count(&self) -> Result<u32> {
        Ok(live(&self.state)?.model.parameter_count() as u32)
    }
    #[napi(getter, js_name = "max_rows")]
    pub fn max_rows(&self) -> Result<u32> {
        Ok(live(&self.state)?.capacity as u32)
    }
    #[napi(getter, js_name = "estimated_bytes")]
    pub fn estimated_bytes(&self) -> Result<u32> {
        let value = live(&self.state)?;
        Ok(add(
            value.base_bytes,
            value.optimizer.as_ref().map_or(0, AdamW::estimated_bytes),
        )? as u32)
    }
    #[napi(js_name = "mse_backward")]
    pub fn mse_backward(
        &self,
        env: Env,
        inputs: Unknown<'_>,
        targets: Unknown<'_>,
        rows: Unknown<'_>,
    ) -> Result<f64> {
        let (ni, no, max) = {
            let value = live(&self.state)?;
            (
                value.model.num_inputs(),
                value.model.num_outputs(),
                value.capacity,
            )
        };
        let rows = gateway::rows(env, rows, 1, max)?;
        let ni = mul(ni, rows)?;
        let no = mul(no, rows)?;
        gateway::validate_array(env, inputs, ni)?;
        gateway::validate_array(env, targets, no)?;
        let mut value = live_mut(&self.state)?;
        work(&value.model, rows)?;
        gateway::copy_input(env, inputs, &mut value.inputs[..ni])?;
        gateway::copy_input(env, targets, &mut value.targets[..no])?;
        let TrainingState {
            model,
            workspace,
            inputs,
            targets,
            ..
        } = &mut *value;
        model
            .backward_mse(&inputs[..ni], &targets[..no], rows, workspace)
            .map_err(|error| invalid(error.to_string()))
    }
    #[napi(js_name = "parameter_gradients")]
    pub fn parameter_gradients(&self) -> Result<Float64Array> {
        owned(live(&self.state)?.workspace.parameter_gradients())
    }
    #[napi(js_name = "input_gradients")]
    pub fn input_gradients(&self) -> Result<Float64Array> {
        owned(live(&self.state)?.workspace.input_gradients())
    }
    #[napi(js_name = "predictions")]
    pub fn predictions(&self) -> Result<Float64Array> {
        owned(live(&self.state)?.workspace.outputs())
    }
    #[napi(js_name = "sgd_step")]
    pub fn sgd_step(&self, env: Env, learning_rate: Unknown<'_>) -> Result<()> {
        let lr = gateway::number(env, learning_rate)?;
        let mut value = live_mut(&self.state)?;
        value.optimizer_work()?;
        let TrainingState {
            model, workspace, ..
        } = &mut *value;
        model
            .sgd_step(workspace, lr)
            .map_err(|error| invalid(error.to_string()))
    }
    #[napi(js_name = "configure_adamw")]
    pub fn configure_adamw(
        &self,
        env: Env,
        learning_rate: Unknown<'_>,
        beta1: Unknown<'_>,
        beta2: Unknown<'_>,
        epsilon: Unknown<'_>,
        weight_decay: Unknown<'_>,
    ) -> Result<()> {
        let config = AdamWConfig {
            learning_rate: gateway::number(env, learning_rate)?,
            beta1: gateway::number(env, beta1)?,
            beta2: gateway::number(env, beta2)?,
            epsilon: gateway::number(env, epsilon)?,
            weight_decay: gateway::number(env, weight_decay)?,
        };
        let mut value = live_mut(&self.state)?;
        value.optimizer_work()?;
        let required =
            AdamW::required_bytes(&value.model).map_err(|error| invalid(error.to_string()))?;
        // Preserve old optimizer on error; account for both allocations during replacement.
        bounded(add(
            add(value.base_bytes, required)?,
            value.optimizer.as_ref().map_or(0, AdamW::estimated_bytes),
        )?)?;
        let optimizer = AdamW::new(&value.model, config, required)
            .map_err(|error| invalid(error.to_string()))?;
        value.optimizer = Some(optimizer);
        Ok(())
    }
    #[napi(js_name = "adamw_step")]
    pub fn adamw_step(&self) -> Result<()> {
        let mut value = live_mut(&self.state)?;
        value.optimizer_work()?;
        let TrainingState {
            model,
            workspace,
            optimizer,
            ..
        } = &mut *value;
        let optimizer = optimizer
            .as_mut()
            .ok_or_else(|| invalid("AdamW is not configured"))?;
        model
            .adamw_step(workspace, optimizer)
            .map_err(|error| invalid(error.to_string()))
    }
    #[napi(js_name = "prepare")]
    pub fn prepare(&self) -> Result<NativePreparedNetwork> {
        let snapshot = live(&self.state)?
            .model
            .prepare()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(NativePreparedNetwork {
            state: RefCell::new(Some(PreparedState::new(snapshot)?)),
        })
    }
    #[napi(js_name = "to_dense_layers_json")]
    pub fn to_dense_layers_json(&self) -> Result<String> {
        let text = live(&self.state)?
            .model
            .to_dense_layers_json()
            .map_err(|error| invalid(error.to_string()))?;
        if text.len() > gateway::MAX_JSON_BYTES {
            return Err(invalid("Training checkpoint exceeds 4 MiB UTF-8"));
        }
        Ok(text)
    }
    #[napi(js_name = "free")]
    pub fn free(&self) -> Result<()> {
        self.state
            .try_borrow_mut()
            .map_err(|_| invalid("Native training object is busy"))?
            .take();
        Ok(())
    }
}
