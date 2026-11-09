//! Node-API 8 classes backed by the actual immutable ruv-FANN prepared core.
mod gateway;
mod state;
mod training;
use gateway::invalid;
use napi::{bindgen_prelude::Unknown, Env, Result};
use napi_derive::napi;
use state::{BatchState, PreparedState};
use std::{
    cell::{Ref, RefCell, RefMut},
    sync::Arc,
};

fn live<T>(state: &RefCell<Option<T>>) -> Result<Ref<'_, T>> {
    let value = state
        .try_borrow()
        .map_err(|_| invalid("Native object is busy"))?;
    Ref::filter_map(value, Option::as_ref).map_err(|_| invalid("Native object is freed"))
}
fn live_mut<T>(state: &RefCell<Option<T>>) -> Result<RefMut<'_, T>> {
    let value = state
        .try_borrow_mut()
        .map_err(|_| invalid("Native object is busy"))?;
    RefMut::filter_map(value, Option::as_mut).map_err(|_| invalid("Native object is freed"))
}

#[napi(module_exports)]
fn register(env: Env) -> Result<()> {
    gateway::register(env)
}

/// All exported instance receivers are shared references. Callback-capable
/// boundary checks run before acquiring the interior mutable inference state.
#[napi]
pub struct NativePreparedNetwork {
    state: RefCell<Option<PreparedState>>,
}

#[napi]
impl NativePreparedNetwork {
    #[napi(factory, js_name = "from_network_json")]
    pub fn from_network_json(env: Env, json: Unknown<'_>) -> Result<Self> {
        let text = gateway::json_string(env, json)?;
        Ok(Self {
            state: RefCell::new(Some(PreparedState::from_network_json(&text)?)),
        })
    }
    #[napi(factory, js_name = "from_dense_layers_json")]
    pub fn from_dense_layers_json(env: Env, json: Unknown<'_>) -> Result<Self> {
        let text = gateway::json_string(env, json)?;
        Ok(Self {
            state: RefCell::new(Some(PreparedState::from_dense_json(&text)?)),
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
    #[napi(getter, js_name = "estimated_bytes")]
    pub fn estimated_bytes(&self) -> Result<u32> {
        Ok(live(&self.state)?.estimated as u32)
    }

    #[napi(js_name = "run")]
    pub fn run<'env>(&self, env: Env, inputs: Unknown<'env>) -> Result<Unknown<'env>> {
        let (ni, no) = {
            let value = live(&self.state)?;
            (value.model.num_inputs(), value.model.num_outputs())
        };
        gateway::validate_array(env, inputs, ni)?;
        let output = gateway::result_array(env, no, inputs)?;
        // No callbacks or JS allocation occurs inside this mutable borrow.
        let mut value = live_mut(&self.state)?;
        gateway::copy_input(env, inputs, &mut value.inputs)?;
        value.evaluate()?;
        gateway::commit_output(env, output, &value.outputs)?;
        Ok(output)
    }
    #[napi(js_name = "run_into")]
    pub fn run_into(&self, env: Env, inputs: Unknown<'_>, outputs: Unknown<'_>) -> Result<()> {
        let (ni, no) = {
            let value = live(&self.state)?;
            (value.model.num_inputs(), value.model.num_outputs())
        };
        gateway::validate_array(env, inputs, ni)?;
        gateway::validate_array(env, outputs, no)?;
        let mut value = live_mut(&self.state)?;
        gateway::copy_input(env, inputs, &mut value.inputs)?;
        value.evaluate()?;
        gateway::commit_output(env, outputs, &value.outputs)
    }
    #[napi(js_name = "create_batch")]
    pub fn create_batch(&self, env: Env, max_rows: Unknown<'_>) -> Result<NativePreparedBatch> {
        let capacity = gateway::rows(env, max_rows, 1, gateway::MAX_ROWS)?;
        let model = Arc::clone(&live(&self.state)?.model);
        Ok(NativePreparedBatch {
            state: RefCell::new(Some(BatchState::new(model, capacity)?)),
        })
    }
    #[napi(js_name = "free")]
    pub fn free(&self) -> Result<()> {
        self.state
            .try_borrow_mut()
            .map_err(|_| invalid("Native object is busy"))?
            .take();
        Ok(())
    }
}

#[napi]
pub struct NativePreparedBatch {
    state: RefCell<Option<BatchState>>,
}

#[napi]
impl NativePreparedBatch {
    #[napi(getter, js_name = "input_size")]
    pub fn input_size(&self) -> Result<u32> {
        Ok(live(&self.state)?.model.num_inputs() as u32)
    }
    #[napi(getter, js_name = "output_size")]
    pub fn output_size(&self) -> Result<u32> {
        Ok(live(&self.state)?.model.num_outputs() as u32)
    }
    #[napi(getter, js_name = "max_rows")]
    pub fn max_rows(&self) -> Result<u32> {
        Ok(live(&self.state)?.capacity as u32)
    }
    #[napi(getter, js_name = "estimated_bytes")]
    pub fn estimated_bytes(&self) -> Result<u32> {
        Ok(live(&self.state)?.estimated as u32)
    }
    #[napi(js_name = "run_into")]
    pub fn run_into(
        &self,
        env: Env,
        inputs: Unknown<'_>,
        outputs: Unknown<'_>,
        rows: Unknown<'_>,
    ) -> Result<()> {
        let (ni, no, max) = {
            let value = live(&self.state)?;
            (
                value.model.num_inputs(),
                value.model.num_outputs(),
                value.capacity,
            )
        };
        let rows = gateway::rows(env, rows, 0, max)?;
        let ni = ni
            .checked_mul(rows)
            .ok_or_else(|| invalid("Input dimensions overflow"))?;
        let no = no
            .checked_mul(rows)
            .ok_or_else(|| invalid("Output dimensions overflow"))?;
        gateway::validate_array(env, inputs, ni)?;
        gateway::validate_array(env, outputs, no)?;
        let mut value = live_mut(&self.state)?;
        gateway::copy_input(env, inputs, &mut value.inputs[..ni])?;
        value.evaluate(rows)?;
        gateway::commit_output(env, outputs, &value.outputs[..no])
    }
    #[napi(js_name = "free")]
    pub fn free(&self) -> Result<()> {
        self.state
            .try_borrow_mut()
            .map_err(|_| invalid("Native object is busy"))?
            .take();
        Ok(())
    }
}
