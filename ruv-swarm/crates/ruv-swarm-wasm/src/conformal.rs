//! Bounded JSON bindings for the shared portable residual calibration engine.
//! This is historical residual calibration, not a temporal coverage guarantee.
use ruv_fann::calibration::{
    CalibrationConfig, CalibrationLimits, CalibrationRecord, ForecastRecord, ResidualCalibrator,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
#[path = "conformal_output.rs"]
mod output;
#[path = "conformal_json.rs"]
mod wire;
use wire::{bounded, name, number, time, INPUT_BYTES, OUTPUT_BYTES};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitInput {
    #[serde(deserialize_with = "time")]
    as_of: i64,
    horizon: usize,
    #[serde(deserialize_with = "levels")]
    coverages: Vec<f64>,
    #[serde(deserialize_with = "records")]
    records: Vec<Observed>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Observed {
    #[serde(deserialize_with = "name")]
    series: String,
    #[serde(deserialize_with = "name")]
    model: String,
    #[serde(deserialize_with = "time")]
    cutoff: i64,
    #[serde(deserialize_with = "time")]
    timestamp: i64,
    #[serde(deserialize_with = "number")]
    actual: f64,
    #[serde(deserialize_with = "number")]
    prediction: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplyInput {
    #[serde(deserialize_with = "forecasts")]
    records: Vec<Predicted>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Predicted {
    #[serde(deserialize_with = "name")]
    series: String,
    #[serde(deserialize_with = "name")]
    model: String,
    #[serde(deserialize_with = "time")]
    origin: i64,
    #[serde(deserialize_with = "time")]
    timestamp: i64,
    #[serde(deserialize_with = "number")]
    prediction: f64,
}
fn levels<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    bounded::<D, f64, 16>(d)
}
fn records<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Observed>, D::Error> {
    bounded::<D, Observed, 10000>(d)
}
fn forecasts<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Predicted>, D::Error> {
    bounded::<D, Predicted, 10000>(d)
}

/// Historical per-series/model/ordinal-horizon intervals, without Polars.
/// All timestamps are safe-integer Unix milliseconds. Input JSON is at most
/// 4 MiB, output at most 8 MiB, records at most 10,000, groups at most 1,000.
/// Caller must recalibrate after changing its model/error distribution.
#[wasm_bindgen]
#[derive(Debug)]
pub struct WasmConformalCalibrator {
    engine: ResidualCalibrator<i64>,
}
#[wasm_bindgen]
impl WasmConformalCalibrator {
    /// Fit `{as_of,horizon,coverages,records:[{series,model,cutoff,timestamp,actual,prediction}]}`.
    /// Every complete window shares the same model set and aligned targets;
    /// `cutoff < timestamp <= as_of`. No insufficient-sample clamping is used.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(unchecked_param_type = "string")] json: JsValue,
    ) -> Result<WasmConformalCalibrator, JsValue> {
        let json = bounded_js_string(json)?;
        Self::from_json(&json).map_err(|e| JsValue::from_str(&e))
    }
    /// Sorted JSON array of group metadata and radii aligned to sorted coverages.
    pub fn groups(&self) -> Result<String, JsValue> {
        self.groups_json().map_err(|e| JsValue::from_str(&e))
    }
    /// Apply `{records:[{series,model,origin,timestamp,prediction}]}` to complete
    /// forecasts with origin >= as_of and target > origin. Returns a JSON array
    /// sorted by series/timestamp/model/coverage. No input or fitted state changes.
    pub fn apply(
        &self,
        #[wasm_bindgen(unchecked_param_type = "string")] json: JsValue,
    ) -> Result<String, JsValue> {
        let json = bounded_js_string(json)?;
        self.apply_json(&json).map_err(|e| JsValue::from_str(&e))
    }
}
// Taking JsValue prevents generated glue from copying unbounded strings into
// WASM first. A JS length cap bounds conversion; parse then checks UTF-8 bytes.
fn bounded_js_string(json: JsValue) -> Result<String, JsValue> {
    if !json.is_string() {
        return Err(JsValue::from_str("Conformal JSON must be a string"));
    }
    let text = json.unchecked_into::<js_sys::JsString>();
    if text.length() as usize > INPUT_BYTES {
        return Err(JsValue::from_str(
            "Conformal JSON exceeds 4 MiB input budget",
        ));
    }
    text.as_string()
        .ok_or_else(|| JsValue::from_str("Conformal JSON must be a string"))
}
impl WasmConformalCalibrator {
    fn from_json(json: &str) -> Result<Self, String> {
        let input: FitInput = wire::parse(json)?;
        let config = CalibrationConfig {
            as_of: input.as_of,
            horizon: input.horizon,
            coverages: input.coverages,
            limits: CalibrationLimits {
                max_records: 10000,
                max_groups: 1000,
                max_output_rows: 100000,
                ..Default::default()
            },
        };
        let records: Vec<_> = input
            .records
            .into_iter()
            .map(|r| CalibrationRecord {
                series: r.series,
                model: r.model,
                cutoff: r.cutoff,
                timestamp: r.timestamp,
                actual: r.actual,
                prediction: r.prediction,
            })
            .collect();
        Ok(Self {
            engine: ResidualCalibrator::fit(config, &records).map_err(|e| e.to_string())?,
        })
    }
    fn groups_json(&self) -> Result<String, String> {
        wire::serialize(&output::Groups(&self.engine))
    }
    fn apply_json(&self, json: &str) -> Result<String, String> {
        let input: ApplyInput = wire::parse(json)?;
        // Conservative escaped-name/numeric/key bound, BEFORE interval allocation.
        let mut projected = 2usize;
        for r in &input.records {
            let row = r
                .series
                .len()
                .checked_add(r.model.len())
                .and_then(|n| n.checked_mul(6))
                .and_then(|n| n.checked_add(512))
                .ok_or("Interval byte budget overflow")?;
            projected = projected
                .checked_add(
                    row.checked_mul(self.engine.config().coverages.len())
                        .ok_or("Interval byte budget overflow")?,
                )
                .ok_or("Interval byte budget overflow")?;
            if projected > OUTPUT_BYTES {
                return Err("Conservative interval JSON byte budget exceeded".into());
            }
        }
        let records: Vec<_> = input
            .records
            .into_iter()
            .map(|r| ForecastRecord {
                series: r.series,
                model: r.model,
                origin: r.origin,
                timestamp: r.timestamp,
                prediction: r.prediction,
            })
            .collect();
        let intervals = self.engine.apply(&records).map_err(|e| e.to_string())?;
        wire::serialize(&output::Intervals(&intervals))
    }
}
#[cfg(test)]
#[path = "conformal_tests.rs"]
mod tests;
