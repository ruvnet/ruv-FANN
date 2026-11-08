use wasm_bindgen::prelude::*;

/// One-step statistical baselines; these models do not train neural networks.
#[wasm_bindgen]
pub struct WasmForecastingModel {
    model_type: String,
}

#[wasm_bindgen]
impl WasmForecastingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_type: &str) -> Result<Self, JsValue> {
        Self::checked_new(model_type).map_err(JsValue::from_str)
    }

    pub fn predict(&self, input: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.predict_checked(input).map_err(JsValue::from_str)
    }

    pub fn get_model_type(&self) -> String {
        self.model_type.clone()
    }
}

impl WasmForecastingModel {
    fn checked_new(model_type: &str) -> Result<Self, &'static str> {
        if !matches!(model_type, "linear" | "mean" | "naive") {
            return Err("Supported forecast baselines are linear, mean and naive");
        }
        Ok(Self {
            model_type: model_type.to_string(),
        })
    }

    fn predict_checked(&self, input: &[f64]) -> Result<Vec<f64>, &'static str> {
        if input.is_empty() || input.len() > 1_000_000 || !input.iter().all(|v| v.is_finite()) {
            return Err("Forecast input must have 1..=1,000,000 finite values");
        }
        let last = input[input.len() - 1];
        let prediction = match self.model_type.as_str() {
            "linear" if input.len() > 1 => last + (last - input[input.len() - 2]),
            "mean" => input.iter().map(|value| value / input.len() as f64).sum(),
            _ => last,
        };
        if !prediction.is_finite() {
            return Err("Forecast overflowed; normalize inputs");
        }
        Ok(vec![prediction])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistical_baselines_return_expected_predictions() {
        for (model, expected) in [("linear", 4.0), ("mean", 2.0), ("naive", 3.0)] {
            let model = WasmForecastingModel::checked_new(model).unwrap();
            assert_eq!(
                model.predict_checked(&[1.0, 2.0, 3.0]).unwrap(),
                vec![expected]
            );
        }
    }

    #[test]
    fn rejects_unsupported_architectures_and_invalid_numbers() {
        assert!(WasmForecastingModel::checked_new("LSTM").is_err());
        let model = WasmForecastingModel::checked_new("linear").unwrap();
        assert!(model.predict_checked(&[]).is_err());
        assert!(model.predict_checked(&[f64::NAN]).is_err());
        assert!(model.predict_checked(&[-f64::MAX, f64::MAX]).is_err());
    }
}
