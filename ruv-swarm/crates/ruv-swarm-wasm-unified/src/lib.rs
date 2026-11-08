//! Browser bindings shared with `ruv-swarm-wasm`.
//!
//! Version 2 removes the unimplemented core, neural, forecasting and persistence
//! prototype modules. Their source is retained for reference but is not compiled
//! or exported. Use the checked dense network, simple forecasting baselines and
//! local agent state from the base module.

pub use ruv_swarm_wasm::*;
use wasm_bindgen::prelude::*;

/// Version of this facade. `get_version` reports the base implementation version.
#[wasm_bindgen]
pub fn get_unified_version() -> String {
    // Referencing the implementation also ensures its WASM exports are linked.
    let _ = ruv_swarm_wasm::get_version();
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn facade_reports_distinct_implementation_and_facade_versions() {
        assert_eq!(super::get_unified_version(), env!("CARGO_PKG_VERSION"));
        assert_eq!(super::get_version(), ruv_swarm_wasm::get_version());
    }
}
