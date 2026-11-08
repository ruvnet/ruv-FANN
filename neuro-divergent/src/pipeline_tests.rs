use super::*;

// Mock model for testing
#[derive(Debug)]
struct MockModel {
    name: String,
    fitted: bool,
}

impl MockModel {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fitted: false,
        }
    }
}

impl BaseModel<f32> for MockModel {
    fn fit(&mut self, _data: &TimeSeriesDataFrame<f32>) -> NeuroDivergentResult<()> {
        self.fitted = true;
        Ok(())
    }

    fn predict(
        &self,
        _data: &TimeSeriesDataFrame<f32>,
    ) -> NeuroDivergentResult<ForecastDataFrame<f32>> {
        // Return placeholder forecast
        let data = df! {
            "unique_id" => Vec::<String>::new(),
            "ds" => Vec::<i64>::new(),
        }
        .unwrap();

        Ok(ForecastDataFrame::new(
            data,
            vec![self.name.clone()],
            1,
            None,
            TimeSeriesSchema::default(),
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn horizon(&self) -> usize {
        1
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn reset(&mut self) -> NeuroDivergentResult<()> {
        self.fitted = false;
        Ok(())
    }
}

#[test]
fn test_neural_forecast_creation() {
    let model = MockModel::new("test_model");
    let result = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily);
    assert!(result.is_ok());

    let nf = result.unwrap();
    assert_eq!(nf.model_names(), vec!["test_model"]);
    assert_eq!(nf.frequency(), Frequency::Daily);
    assert!(!nf.is_fitted());
}

#[test]
fn test_neural_forecast_builder() {
    let model = MockModel::new("test_model");
    let result = NeuralForecast::builder()
        .with_model(Box::new(model))
        .with_frequency(Frequency::Hourly)
        .with_num_threads(4)
        .build();

    assert!(result.is_ok());
    let nf = result.unwrap();
    assert_eq!(nf.frequency(), Frequency::Hourly);
    assert_eq!(nf.num_threads, Some(4));
}

#[test]
fn test_empty_models_error() {
    let result = NeuralForecast::<f32>::new(vec![], Frequency::Daily);
    assert!(result.is_err());
    assert!(result.unwrap_err().is_config_error());
}

#[test]
fn test_duplicate_model_names() {
    let model1 = MockModel::new("duplicate");
    let model2 = MockModel::new("duplicate");
    let result = NeuralForecast::new(vec![Box::new(model1), Box::new(model2)], Frequency::Daily);
    assert!(result.is_err());
    assert!(result.unwrap_err().is_config_error());
}

#[test]
fn test_prediction_before_fitting() {
    let model = MockModel::new("test_model");
    let nf = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily).unwrap();
    let result = nf.predict();
    assert!(result.is_err());
    assert!(result.unwrap_err().is_prediction_error());
}

#[test]
fn malformed_empty_model_outputs_are_rejected() {
    let model = MockModel::new("fixture");
    let mut pipeline = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily).unwrap();
    let frame = df! {
        "unique_id" => ["a", "a"],
        "ds" => [0_i64, 86_400_000_i64],
        "y" => [1.0, 2.0],
    }
    .unwrap();
    let data = TimeSeriesDataFrame::<f32>::from_polars(
        frame,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    )
    .unwrap();
    pipeline.fit(data.clone()).unwrap();
    assert!(pipeline
        .predict()
        .unwrap_err()
        .to_string()
        .contains("incorrect forecast size"));
    assert!(pipeline
        .predict_on(data.clone())
        .unwrap_err()
        .to_string()
        .contains("incorrect forecast size"));
    let config = CrossValidationConfig::new(1, 1);
    assert!(pipeline
        .cross_validation(data, config)
        .unwrap_err()
        .to_string()
        .contains("incorrect forecast size"));
}
