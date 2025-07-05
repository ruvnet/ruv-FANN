//! Machine learning models for capacity forecasting

use crate::config::*;
use crate::error::*;
use crate::types::*;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use ruv_fann::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Trait for capacity forecasting models
#[async_trait]
pub trait CapacityForecastingModel: Send + Sync {
    /// Train the model with historical data
    async fn train(&mut self, data: &[CapacityDataPoint]) -> ModelResult<()>;

    /// Make predictions for future time periods
    async fn predict(&self, forecast_horizon: usize) -> ModelResult<Vec<f64>>;

    /// Calculate prediction confidence intervals
    async fn predict_with_confidence(
        &self,
        forecast_horizon: usize,
        confidence_level: f64,
    ) -> ModelResult<(Vec<f64>, Vec<f64>, Vec<f64>)>;

    /// Get model performance metrics
    fn get_metrics(&self) -> ModelResult<ModelMetrics>;

    /// Get model name
    fn name(&self) -> &str;

    /// Check if model is trained
    fn is_trained(&self) -> bool;

    /// Save model to file
    async fn save(&self, path: &str) -> ModelResult<()>;

    /// Load model from file
    async fn load(&mut self, path: &str) -> ModelResult<()>;
}

/// Long-term capacity forecasting model using LSTM
pub struct LSTMCapacityModel {
    name: String,
    config: LSTMConfig,
    network: Option<Arc<RwLock<Network>>>,
    training_data: Vec<CapacityDataPoint>,
    is_trained: bool,
    metrics: Option<ModelMetrics>,
    scaler: Option<DataScaler>,
}

/// ARIMA-based capacity forecasting model
pub struct ARIMACapacityModel {
    name: String,
    config: ARIMAConfig,
    is_trained: bool,
    model_params: Option<ARIMAParams>,
    training_data: Vec<CapacityDataPoint>,
    metrics: Option<ModelMetrics>,
}

/// Polynomial regression model for long-term trends
pub struct PolynomialCapacityModel {
    name: String,
    config: PolynomialConfig,
    is_trained: bool,
    coefficients: Option<Array1<f64>>,
    training_data: Vec<CapacityDataPoint>,
    metrics: Option<ModelMetrics>,
}

/// Exponential smoothing model for seasonal patterns
pub struct ExponentialSmoothingModel {
    name: String,
    config: ExponentialSmoothingConfig,
    is_trained: bool,
    state: Option<ExponentialSmoothingState>,
    training_data: Vec<CapacityDataPoint>,
    metrics: Option<ModelMetrics>,
}

/// Neural network model using ruv-FANN
pub struct NeuralForecastModel {
    name: String,
    config: NeuralForecastConfig,
    network: Option<Arc<RwLock<Network>>>,
    training_data: Vec<CapacityDataPoint>,
    is_trained: bool,
    metrics: Option<ModelMetrics>,
    scaler: Option<DataScaler>,
}

/// Ensemble model combining multiple forecasting models
pub struct EnsembleCapacityModel {
    name: String,
    config: EnsembleConfig,
    models: HashMap<String, Box<dyn CapacityForecastingModel>>,
    weights: HashMap<String, f64>,
    is_trained: bool,
    metrics: Option<ModelMetrics>,
}

/// Capacity cliff predictor - main interface for breach prediction
pub struct CapacityCliffPredictor {
    models: HashMap<String, Box<dyn CapacityForecastingModel>>,
    config: CapacityPlanningConfig,
    ensemble_model: Option<EnsembleCapacityModel>,
    breach_threshold: f64,
}

/// Data scaler for normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataScaler {
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
}

/// ARIMA model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMAParams {
    ar_coeffs: Vec<f64>,
    ma_coeffs: Vec<f64>,
    seasonal_ar_coeffs: Vec<f64>,
    seasonal_ma_coeffs: Vec<f64>,
    intercept: f64,
    sigma: f64,
}

/// Exponential smoothing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialSmoothingState {
    level: f64,
    trend: f64,
    seasonal: Vec<f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

impl LSTMCapacityModel {
    /// Create a new LSTM capacity model
    pub fn new(name: String, config: LSTMConfig) -> Self {
        Self {
            name,
            config,
            network: None,
            training_data: Vec::new(),
            is_trained: false,
            metrics: None,
            scaler: None,
        }
    }

    /// Prepare training data for LSTM
    fn prepare_training_data(&self, data: &[CapacityDataPoint]) -> ModelResult<(Array2<f64>, Array2<f64>)> {
        if data.len() < self.config.sequence_length + 1 {
            return Err(ModelError::insufficient_data(
                self.config.sequence_length + 1,
                data.len(),
            ));
        }

        let mut scaler = DataScaler::new();
        let values: Vec<f64> = data.iter().map(|d| d.prb_utilization).collect();
        scaler.fit(&values);

        let scaled_values: Vec<f64> = values.iter().map(|v| scaler.transform(*v)).collect();

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for i in 0..scaled_values.len() - self.config.sequence_length {
            let input_seq = &scaled_values[i..i + self.config.sequence_length];
            let target = scaled_values[i + self.config.sequence_length];

            inputs.push(input_seq.to_vec());
            targets.push(vec![target]);
        }

        let input_array = Array2::from_shape_vec(
            (inputs.len(), self.config.sequence_length),
            inputs.into_iter().flatten().collect(),
        )
        .map_err(|e| ModelError::training_failed(e.to_string()))?;

        let target_array = Array2::from_shape_vec(
            (targets.len(), 1),
            targets.into_iter().flatten().collect(),
        )
        .map_err(|e| ModelError::training_failed(e.to_string()))?;

        Ok((input_array, target_array))
    }

    /// Create neural network architecture
    fn create_network(&self) -> ModelResult<Network> {
        let mut layers = vec![self.config.sequence_length as u32];
        
        // Add hidden layers
        for _ in 0..self.config.layers {
            layers.push(self.config.hidden_size as u32);
        }
        
        // Output layer
        layers.push(1);

        let mut network = Network::new(&layers)
            .map_err(|e| ModelError::training_failed(format!("Failed to create network: {}", e)))?;

        network.set_activation_function_hidden(ActivationFunction::Sigmoid);
        network.set_activation_function_output(ActivationFunction::Linear);

        Ok(network)
    }
}

#[async_trait]
impl CapacityForecastingModel for LSTMCapacityModel {
    async fn train(&mut self, data: &[CapacityDataPoint]) -> ModelResult<()> {
        if data.len() < self.config.sequence_length + 1 {
            return Err(ModelError::insufficient_data(
                self.config.sequence_length + 1,
                data.len(),
            ));
        }

        let (inputs, targets) = self.prepare_training_data(data)?;
        
        let mut network = self.create_network()?;
        
        // Convert to training data format
        let training_data = TrainingData::new(
            inputs.len() as u32,
            self.config.sequence_length as u32,
            1,
        );

        // Train the network
        network.train_on_data(
            &training_data,
            self.config.max_epochs as u32,
            1,
            self.config.target_error as f32,
        ).map_err(|e| ModelError::training_failed(e.to_string()))?;

        self.network = Some(Arc::new(RwLock::new(network)));
        self.training_data = data.to_vec();
        self.is_trained = true;

        // Calculate metrics
        self.calculate_metrics().await?;

        Ok(())
    }

    async fn predict(&self, forecast_horizon: usize) -> ModelResult<Vec<f64>> {
        if !self.is_trained {
            return Err(ModelError::not_trained(self.name.clone()));
        }

        let network = self.network.as_ref().unwrap();
        let network_guard = network.read().await;
        
        let mut predictions = Vec::with_capacity(forecast_horizon);
        
        // Use last sequence from training data as starting point
        let mut current_sequence: Vec<f64> = self.training_data
            .iter()
            .rev()
            .take(self.config.sequence_length)
            .map(|d| d.prb_utilization)
            .collect();
        current_sequence.reverse();

        // Scale input
        let scaler = self.scaler.as_ref().unwrap();
        let mut scaled_sequence: Vec<f64> = current_sequence
            .iter()
            .map(|v| scaler.transform(*v))
            .collect();

        for _ in 0..forecast_horizon {
            let input = scaled_sequence.as_slice();
            let output = network_guard.run(input)
                .map_err(|e| ModelError::prediction_failed(e.to_string()))?;
            
            let scaled_prediction = output[0] as f64;
            let prediction = scaler.inverse_transform(scaled_prediction);
            predictions.push(prediction);

            // Update sequence for next prediction
            scaled_sequence.remove(0);
            scaled_sequence.push(scaled_prediction);
        }

        Ok(predictions)
    }

    async fn predict_with_confidence(
        &self,
        forecast_horizon: usize,
        confidence_level: f64,
    ) -> ModelResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let predictions = self.predict(forecast_horizon).await?;
        
        // Calculate confidence intervals based on historical model performance
        let metrics = self.metrics.as_ref().unwrap();
        let std_error = metrics.rmse;
        let z_score = match confidence_level {
            0.95 => 1.96,
            0.99 => 2.58,
            0.90 => 1.64,
            _ => 1.96, // Default to 95%
        };

        let lower_bounds: Vec<f64> = predictions
            .iter()
            .map(|p| p - z_score * std_error)
            .collect();

        let upper_bounds: Vec<f64> = predictions
            .iter()
            .map(|p| p + z_score * std_error)
            .collect();

        Ok((predictions, lower_bounds, upper_bounds))
    }

    fn get_metrics(&self) -> ModelResult<ModelMetrics> {
        self.metrics.clone().ok_or_else(|| ModelError::not_trained(self.name.clone()))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn save(&self, path: &str) -> ModelResult<()> {
        if !self.is_trained {
            return Err(ModelError::not_trained(self.name.clone()));
        }

        let network = self.network.as_ref().unwrap();
        let network_guard = network.read().await;
        
        network_guard.save(path)
            .map_err(|e| ModelError::training_failed(format!("Failed to save model: {}", e)))?;

        // Save additional model state
        let model_state = LSTMModelState {
            config: self.config.clone(),
            scaler: self.scaler.clone(),
            metrics: self.metrics.clone(),
        };

        let state_path = format!("{}.state", path);
        let state_json = serde_json::to_string(&model_state)
            .map_err(|e| ModelError::training_failed(e.to_string()))?;
        
        std::fs::write(state_path, state_json)
            .map_err(|e| ModelError::training_failed(e.to_string()))?;

        Ok(())
    }

    async fn load(&mut self, path: &str) -> ModelResult<()> {
        let mut network = self.create_network()?;
        network.load(path)
            .map_err(|e| ModelError::training_failed(format!("Failed to load model: {}", e)))?;

        self.network = Some(Arc::new(RwLock::new(network)));

        // Load additional model state
        let state_path = format!("{}.state", path);
        let state_json = std::fs::read_to_string(state_path)
            .map_err(|e| ModelError::training_failed(e.to_string()))?;
        
        let model_state: LSTMModelState = serde_json::from_str(&state_json)
            .map_err(|e| ModelError::training_failed(e.to_string()))?;

        self.config = model_state.config;
        self.scaler = model_state.scaler;
        self.metrics = model_state.metrics;
        self.is_trained = true;

        Ok(())
    }
}

impl LSTMCapacityModel {
    async fn calculate_metrics(&mut self) -> ModelResult<()> {
        if !self.is_trained || self.training_data.is_empty() {
            return Err(ModelError::not_trained(self.name.clone()));
        }

        // Use last 20% of training data for validation
        let validation_size = (self.training_data.len() as f64 * 0.2) as usize;
        let validation_data = &self.training_data[self.training_data.len() - validation_size..];

        let predictions = self.predict(validation_data.len()).await?;
        let actuals: Vec<f64> = validation_data.iter().map(|d| d.prb_utilization).collect();

        let metrics = calculate_regression_metrics(&actuals, &predictions)?;
        self.metrics = Some(metrics);

        Ok(())
    }
}

/// LSTM model state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LSTMModelState {
    config: LSTMConfig,
    scaler: Option<DataScaler>,
    metrics: Option<ModelMetrics>,
}

impl DataScaler {
    fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            std: 1.0,
        }
    }

    fn fit(&mut self, data: &[f64]) {
        if data.is_empty() {
            return;
        }

        self.min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        self.max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        self.mean = data.iter().sum::<f64>() / data.len() as f64;
        
        let variance = data.iter()
            .map(|x| (x - self.mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        self.std = variance.sqrt();
    }

    fn transform(&self, value: f64) -> f64 {
        (value - self.mean) / self.std
    }

    fn inverse_transform(&self, value: f64) -> f64 {
        value * self.std + self.mean
    }
}

impl CapacityCliffPredictor {
    /// Create a new capacity cliff predictor
    pub fn new(config: CapacityPlanningConfig) -> Self {
        Self {
            models: HashMap::new(),
            config,
            ensemble_model: None,
            breach_threshold: 0.8,
        }
    }

    /// Add a forecasting model to the predictor
    pub fn add_model(&mut self, model: Box<dyn CapacityForecastingModel>) {
        let name = model.name().to_string();
        self.models.insert(name, model);
    }

    /// Train all models with historical data
    pub async fn train_models(&mut self, data: &[CapacityDataPoint]) -> ModelResult<()> {
        for model in self.models.values_mut() {
            model.train(data).await?;
        }
        Ok(())
    }

    /// Predict when capacity will breach the threshold
    pub async fn predict_capacity_breach(
        &self,
        threshold: f64,
        cell_id: &str,
    ) -> ModelResult<CapacityBreachPrediction> {
        if self.models.is_empty() {
            return Err(ModelError::not_found("No models available".to_string()));
        }

        let mut best_prediction = None;
        let mut best_confidence = 0.0;

        for (model_name, model) in &self.models {
            if !model.is_trained() {
                continue;
            }

            let predictions = model.predict(self.config.max_forecast_horizon_months).await?;
            
            // Find first breach
            if let Some((breach_month, breach_utilization)) = predictions
                .iter()
                .enumerate()
                .find(|(_, &util)| util >= threshold)
            {
                let metrics = model.get_metrics()?;
                let confidence = 1.0 - (metrics.mape / 100.0); // Convert MAPE to confidence

                if confidence > best_confidence {
                    best_confidence = confidence;
                    best_prediction = Some(CapacityBreachPrediction {
                        id: uuid::Uuid::new_v4(),
                        prediction_timestamp: Utc::now(),
                        cell_id: cell_id.to_string(),
                        threshold,
                        predicted_breach_date: Utc::now() + chrono::Duration::days(breach_month as i64 * 30),
                        confidence,
                        months_ahead: breach_month as f64,
                        predicted_utilization: breach_utilization,
                        model_name: model_name.clone(),
                        accuracy_estimate: self.config.target_accuracy_months,
                    });
                }
            }
        }

        best_prediction.ok_or_else(|| {
            ModelError::prediction_failed("No capacity breach predicted within forecast horizon".to_string())
        })
    }

    /// Get growth trend analysis
    pub async fn analyze_growth_trend(&self, data: &[CapacityDataPoint]) -> ModelResult<GrowthTrendAnalysis> {
        if data.len() < 12 {
            return Err(ModelError::insufficient_data(12, data.len()));
        }

        let values: Vec<f64> = data.iter().map(|d| d.prb_utilization).collect();
        let timestamps: Vec<DateTime<Utc>> = data.iter().map(|d| d.timestamp).collect();

        // Calculate growth rates
        let monthly_growth = calculate_monthly_growth_rate(&values)?;
        let quarterly_growth = calculate_quarterly_growth_rate(&values)?;
        let annual_growth = calculate_annual_growth_rate(&values)?;

        // Determine trend direction
        let trend_direction = if monthly_growth > 0.02 {
            TrendDirection::Increasing
        } else if monthly_growth < -0.02 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        // Calculate volatility
        let volatility = calculate_volatility(&values);

        // Calculate R-squared for trend fit
        let r_squared = calculate_trend_r_squared(&values)?;

        // Detect seasonal patterns
        let seasonal_patterns = detect_seasonal_patterns(&values, &timestamps)?;

        Ok(GrowthTrendAnalysis {
            cell_id: data[0].cell_id.clone(),
            analysis_timestamp: Utc::now(),
            monthly_growth_rate: monthly_growth,
            quarterly_growth_rate: quarterly_growth,
            annual_growth_rate: annual_growth,
            trend_direction,
            seasonal_patterns,
            volatility,
            r_squared,
        })
    }

    /// Generate network expansion recommendations
    pub async fn generate_expansion_recommendations(
        &self,
        predictions: &[CapacityBreachPrediction],
        cell_id: &str,
    ) -> ModelResult<Vec<ExpansionRecommendation>> {
        let mut recommendations = Vec::new();

        for prediction in predictions {
            if prediction.months_ahead <= 6.0 {
                // Urgent capacity expansion needed
                recommendations.push(ExpansionRecommendation {
                    id: uuid::Uuid::new_v4(),
                    cell_id: cell_id.to_string(),
                    timestamp: Utc::now(),
                    recommendation_type: RecommendationType::EquipmentUpgrade,
                    priority: 5,
                    estimated_cost: 75000.0,
                    expected_capacity_increase: 1.5,
                    implementation_timeline_months: 2.0,
                    roi_estimate: 3.0,
                    risk_level: RiskLevel::Low,
                    description: "Urgent equipment upgrade to increase capacity".to_string(),
                });
            } else if prediction.months_ahead <= 12.0 {
                // Medium-term planning
                recommendations.push(ExpansionRecommendation {
                    id: uuid::Uuid::new_v4(),
                    cell_id: cell_id.to_string(),
                    timestamp: Utc::now(),
                    recommendation_type: RecommendationType::SpectrumExpansion,
                    priority: 3,
                    estimated_cost: 120000.0,
                    expected_capacity_increase: 2.0,
                    implementation_timeline_months: 6.0,
                    roi_estimate: 2.5,
                    risk_level: RiskLevel::Medium,
                    description: "Spectrum expansion for medium-term capacity increase".to_string(),
                });
            } else {
                // Long-term strategic planning
                recommendations.push(ExpansionRecommendation {
                    id: uuid::Uuid::new_v4(),
                    cell_id: cell_id.to_string(),
                    timestamp: Utc::now(),
                    recommendation_type: RecommendationType::NewCellSite,
                    priority: 2,
                    estimated_cost: 250000.0,
                    expected_capacity_increase: 3.0,
                    implementation_timeline_months: 12.0,
                    roi_estimate: 2.0,
                    risk_level: RiskLevel::High,
                    description: "New cell site for long-term capacity growth".to_string(),
                });
            }
        }

        Ok(recommendations)
    }

    /// Set breach threshold
    pub fn set_breach_threshold(&mut self, threshold: f64) {
        self.breach_threshold = threshold;
    }

    /// Get current breach threshold
    pub fn get_breach_threshold(&self) -> f64 {
        self.breach_threshold
    }
}

/// Calculate regression metrics
fn calculate_regression_metrics(actual: &[f64], predicted: &[f64]) -> ModelResult<ModelMetrics> {
    if actual.len() != predicted.len() {
        return Err(ModelError::validation_failed(
            "length_mismatch".to_string(),
            actual.len() as f64,
            predicted.len() as f64,
        ));
    }

    let n = actual.len() as f64;
    
    // Mean Absolute Error
    let mae = actual.iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>() / n;

    // Mean Squared Error
    let mse = actual.iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>() / n;

    // Root Mean Squared Error
    let rmse = mse.sqrt();

    // Mean Absolute Percentage Error
    let mape = actual.iter()
        .zip(predicted.iter())
        .map(|(a, p)| if *a != 0.0 { ((a - p) / a).abs() * 100.0 } else { 0.0 })
        .sum::<f64>() / n;

    // R-squared
    let actual_mean = actual.iter().sum::<f64>() / n;
    let ss_tot = actual.iter().map(|a| (a - actual_mean).powi(2)).sum::<f64>();
    let ss_res = actual.iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>();
    let r_squared = 1.0 - (ss_res / ss_tot);

    Ok(ModelMetrics {
        mae,
        mse,
        rmse,
        mape,
        r_squared,
        aic: 0.0, // Would need to implement AIC calculation
        bic: 0.0, // Would need to implement BIC calculation
    })
}

/// Calculate monthly growth rate
fn calculate_monthly_growth_rate(values: &[f64]) -> ModelResult<f64> {
    if values.len() < 2 {
        return Err(ModelError::insufficient_data(2, values.len()));
    }

    let mut total_growth = 0.0;
    let mut count = 0;

    for i in 1..values.len() {
        if values[i - 1] != 0.0 {
            total_growth += (values[i] - values[i - 1]) / values[i - 1];
            count += 1;
        }
    }

    Ok(if count > 0 { total_growth / count as f64 } else { 0.0 })
}

/// Calculate quarterly growth rate
fn calculate_quarterly_growth_rate(values: &[f64]) -> ModelResult<f64> {
    let monthly_growth = calculate_monthly_growth_rate(values)?;
    Ok(monthly_growth * 3.0) // Approximate quarterly growth
}

/// Calculate annual growth rate
fn calculate_annual_growth_rate(values: &[f64]) -> ModelResult<f64> {
    let monthly_growth = calculate_monthly_growth_rate(values)?;
    Ok(monthly_growth * 12.0) // Approximate annual growth
}

/// Calculate volatility
fn calculate_volatility(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Calculate R-squared for trend fit
fn calculate_trend_r_squared(values: &[f64]) -> ModelResult<f64> {
    if values.len() < 2 {
        return Err(ModelError::insufficient_data(2, values.len()));
    }

    let n = values.len() as f64;
    let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
    
    // Calculate linear regression
    let x_mean = x_values.iter().sum::<f64>() / n;
    let y_mean = values.iter().sum::<f64>() / n;
    
    let numerator = x_values.iter()
        .zip(values.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    
    let denominator = x_values.iter()
        .map(|x| (x - x_mean).powi(2))
        .sum::<f64>();
    
    if denominator == 0.0 {
        return Ok(0.0);
    }
    
    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;
    
    // Calculate R-squared
    let predicted: Vec<f64> = x_values.iter()
        .map(|x| slope * x + intercept)
        .collect();
    
    let ss_res = values.iter()
        .zip(predicted.iter())
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>();
    
    let ss_tot = values.iter()
        .map(|y| (y - y_mean).powi(2))
        .sum::<f64>();
    
    Ok(1.0 - (ss_res / ss_tot))
}

/// Detect seasonal patterns in the data
fn detect_seasonal_patterns(
    values: &[f64],
    timestamps: &[DateTime<Utc>],
) -> ModelResult<Vec<SeasonalPattern>> {
    let mut patterns = Vec::new();
    
    if values.len() >= 12 {
        // Check for monthly patterns
        let monthly_pattern = detect_monthly_seasonality(values, timestamps)?;
        if monthly_pattern.strength > 0.1 {
            patterns.push(monthly_pattern);
        }
    }
    
    if values.len() >= 4 {
        // Check for quarterly patterns
        let quarterly_pattern = detect_quarterly_seasonality(values, timestamps)?;
        if quarterly_pattern.strength > 0.1 {
            patterns.push(quarterly_pattern);
        }
    }
    
    Ok(patterns)
}

/// Detect monthly seasonality
fn detect_monthly_seasonality(
    values: &[f64],
    _timestamps: &[DateTime<Utc>],
) -> ModelResult<SeasonalPattern> {
    // Simplified seasonal pattern detection
    // In a real implementation, you would use FFT or autocorrelation
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    let seasonal_strength = if variance > 0.0 {
        // Simplified calculation - in reality would use proper seasonal decomposition
        0.3 // Placeholder value
    } else {
        0.0
    };
    
    Ok(SeasonalPattern {
        pattern_type: SeasonalPatternType::Monthly,
        strength: seasonal_strength,
        phase_offset: 0.0,
        amplitude: variance.sqrt(),
    })
}

/// Detect quarterly seasonality
fn detect_quarterly_seasonality(
    values: &[f64],
    _timestamps: &[DateTime<Utc>],
) -> ModelResult<SeasonalPattern> {
    // Simplified quarterly pattern detection
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    let seasonal_strength = if variance > 0.0 {
        0.2 // Placeholder value
    } else {
        0.0
    };
    
    Ok(SeasonalPattern {
        pattern_type: SeasonalPatternType::Quarterly,
        strength: seasonal_strength,
        phase_offset: 0.0,
        amplitude: variance.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_data() -> Vec<CapacityDataPoint> {
        let mut data = Vec::new();
        let base_time = Utc::now();
        
        for i in 0..24 {
            let utilization = 0.5 + (i as f64) * 0.01 + (i as f64 * 0.1).sin() * 0.05;
            data.push(CapacityDataPoint {
                id: uuid::Uuid::new_v4(),
                timestamp: base_time + chrono::Duration::days(i * 30),
                cell_id: "TEST_CELL".to_string(),
                prb_utilization: utilization,
                total_prb_capacity: 100,
                used_prb_count: (utilization * 100.0) as u32,
                active_users: 50,
                throughput_mbps: 100.0,
                quality_metrics: QualityMetrics::default(),
            });
        }
        
        data
    }

    #[tokio::test]
    async fn test_lstm_model_creation() {
        let config = LSTMConfig::default();
        let model = LSTMCapacityModel::new("test_lstm".to_string(), config);
        
        assert_eq!(model.name(), "test_lstm");
        assert!(!model.is_trained());
    }

    #[tokio::test]
    async fn test_capacity_cliff_predictor() {
        let config = CapacityPlanningConfig::default();
        let mut predictor = CapacityCliffPredictor::new(config);
        
        let lstm_config = LSTMConfig::default();
        let lstm_model = LSTMCapacityModel::new("lstm".to_string(), lstm_config);
        predictor.add_model(Box::new(lstm_model));
        
        assert_eq!(predictor.get_breach_threshold(), 0.8);
        
        predictor.set_breach_threshold(0.75);
        assert_eq!(predictor.get_breach_threshold(), 0.75);
    }

    #[test]
    fn test_data_scaler() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut scaler = DataScaler::new();
        scaler.fit(&data);
        
        let scaled = scaler.transform(3.0);
        let unscaled = scaler.inverse_transform(scaled);
        
        assert!((unscaled - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_regression_metrics() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.1, 1.9, 3.1, 3.9, 5.1];
        
        let metrics = calculate_regression_metrics(&actual, &predicted).unwrap();
        
        assert!(metrics.mae < 0.2);
        assert!(metrics.r_squared > 0.95);
        assert!(metrics.mape < 10.0);
    }

    #[test]
    fn test_growth_rate_calculation() {
        let values = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let growth_rate = calculate_monthly_growth_rate(&values).unwrap();
        
        assert!(growth_rate > 0.08);
        assert!(growth_rate < 0.12);
    }

    #[test]
    fn test_volatility_calculation() {
        let values = vec![1.0, 2.0, 1.5, 3.0, 2.5];
        let volatility = calculate_volatility(&values);
        
        assert!(volatility > 0.0);
    }

    #[test]
    fn test_trend_r_squared() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect linear trend
        let r_squared = calculate_trend_r_squared(&values).unwrap();
        
        assert!((r_squared - 1.0).abs() < 1e-10);
    }
}