//! Canonical [outputs, inputs] dense parameters, separate from legacy WASM's
//! input-major layout. The shared versioned schema is also used by Candle/Node.
use super::{WasmPreparedNetwork, MAX_BYTES};
use ruv_fann::{PreparationLimits, PreparedNetwork};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WasmPreparedNetwork {
    pub fn from_dense_layers_json(
        #[wasm_bindgen(unchecked_param_type = "string")] json: JsValue,
    ) -> Result<WasmPreparedNetwork, JsValue> {
        let text = json_text(json)?;
        let network = PreparedNetwork::from_dense_layers_json_with_limits(&text, limits())
            .map_err(|_| JsValue::from_str("Invalid or oversized dense-layer model"))?;
        Self::from_snapshot(network).map_err(JsValue::from_str)
    }
}

pub(super) fn limits() -> PreparationLimits {
    PreparationLimits {
        max_bytes: MAX_BYTES,
        max_connections: super::MAX_PARAMETERS,
        ..PreparationLimits::default()
    }
}

pub(super) fn json_text(json: JsValue) -> Result<String, JsValue> {
    const CAP: usize = 4 * 1024 * 1024;
    if !json.is_string() {
        return Err(JsValue::from_str("Dense JSON must be a primitive string"));
    }
    let text = json.unchecked_into::<js_sys::JsString>();
    if text.length() as usize > CAP {
        return Err(JsValue::from_str("Dense JSON exceeds 4 MiB"));
    }
    let text = text
        .as_string()
        .ok_or_else(|| JsValue::from_str("Dense JSON must be a string"))?;
    if text.len() > CAP {
        return Err(JsValue::from_str("Dense JSON exceeds 4 MiB UTF-8"));
    }
    Ok(text)
}
