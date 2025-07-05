//! Time-series forecasting module for PRB utilization prediction

use std::sync::Arc;
use std::collections::VecDeque;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use anyhow::Result;
use ndarray::{Array1, Array2};
// use smartcore::linear::linear_regression::LinearRegression;
// use smartcore::linalg::basic::matrix::DenseMatrix;
// use smartcore::linalg::basic::vector::DenseVector;
// use smartcore::metrics::mean_absolute_error;
use statrs::distribution::{Normal, Continuous};
use statrs::statistics::{Statistics, Data};

use crate::{PrbUtilization, ForecastingError, config::ForecastingConfig};

/// ARIMA model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArimaParams {
    pub p: usize, // Autoregressive order
    pub d: usize, // Differencing order
    pub q: usize, // Moving average order
}

impl Default for ArimaParams {
    fn default() -> Self {
        Self { p: 2, d: 1, q: 1 }
    }
}

/// Prophet-like model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetParams {
    pub yearly_seasonality: bool,
    pub weekly_seasonality: bool,
    pub daily_seasonality: bool,
    pub seasonality_prior_scale: f64,
    pub changepoint_prior_scale: f64,
}

impl Default for ProphetParams {
    fn default() -> Self {
        Self {
            yearly_seasonality: false,
            weekly_seasonality: true,
            daily_seasonality: true,
            seasonality_prior_scale: 10.0,
            changepoint_prior_scale: 0.05,
        }
    }
}

/// Hybrid forecasting model combining ARIMA and Prophet-like components
#[derive(Debug, Clone)]
pub struct ForecastingModel {
    cell_id: String,
    config: Arc<ForecastingConfig>,
    arima_params: ArimaParams,
    prophet_params: ProphetParams,
    
    // Model state
    historical_data: VecDeque<PrbUtilization>,
    trend_coefficients: Vec<f64>,
    seasonal_components: Vec<f64>,
    residuals: Vec<f64>,
    
    // Model performance
    last_training_time: Option<DateTime<Utc>>,
    model_accuracy: f64,
    is_trained: bool,
}

impl ForecastingModel {
    pub async fn new(config: Arc<ForecastingConfig>, cell_id: &str) -> Result<Self> {
        Ok(Self {
            cell_id: cell_id.to_string(),
            config,
            arima_params: ArimaParams::default(),
            prophet_params: ProphetParams::default(),
            historical_data: VecDeque::new(),
            trend_coefficients: Vec::new(),
            seasonal_components: Vec::new(),
            residuals: Vec::new(),
            last_training_time: None,
            model_accuracy: 0.0,
            is_trained: false,
        })
    }
    
    /// Train the forecasting model on historical data
    pub async fn train(&mut self, data: &[PrbUtilization]) -> Result<()> {
        log::info!("Training forecasting model for cell {}", self.cell_id);
        
        if data.len() < self.config.min_data_points {
            return Err(ForecastingError::InsufficientData(
                format!("Need at least {} data points for training", self.config.min_data_points)
            ).into());
        }
        
        // Store historical data
        self.historical_data = data.iter().cloned().collect();
        
        // Extract time series values
        let values: Vec<f64> = data.iter()
            .map(|d| d.utilization_percentage)
            .collect();
        
        // Decompose time series
        self.decompose_time_series(&values).await?;
        
        // Fit trend model
        self.fit_trend_model(&values).await?;
        
        // Fit seasonal model
        self.fit_seasonal_model(&values).await?;
        
        // Calculate residuals and model performance
        self.calculate_residuals(&values).await?;
        self.evaluate_model_performance().await?;
        
        self.last_training_time = Some(Utc::now());
        self.is_trained = true;
        
        log::info!("Model training completed for cell {} with accuracy: {:.2}%", 
            self.cell_id, self.model_accuracy * 100.0);
        
        Ok(())
    }
    
    /// Predict PRB utilization for the next hour
    pub async fn predict_next_hour(&self, recent_data: &[PrbUtilization]) -> Result<Vec<PrbUtilization>> {
        if !self.is_trained {
            return Err(ForecastingError::ModelTrainingFailed(
                "Model must be trained before making predictions".to_string()
            ).into());
        }
        
        log::debug!("Generating predictions for cell {}", self.cell_id);
        
        let last_timestamp = recent_data.last()
            .map(|d| d.timestamp)
            .unwrap_or_else(|| Utc::now());
        
        let mut predictions = Vec::new();
        let prediction_intervals = self.config.forecast_horizon_minutes / 10; // 10-minute intervals
        
        for i in 1..=prediction_intervals {
            let prediction_time = last_timestamp + Duration::minutes(i as i64 * 10);
            
            // Generate prediction using hybrid model
            let predicted_utilization = self.predict_single_point(i, recent_data).await?;
            
            // Create prediction data point
            let prediction = PrbUtilization {
                timestamp: prediction_time,
                cell_id: self.cell_id.clone(),
                prb_total: 100, // Assume standard PRB count
                prb_used: (predicted_utilization * 100.0 / 100.0) as u32,
                utilization_percentage: predicted_utilization,
                throughput_mbps: self.estimate_throughput(predicted_utilization),
                user_count: self.estimate_user_count(predicted_utilization),
                signal_quality: self.estimate_signal_quality(predicted_utilization),
            };
            
            predictions.push(prediction);
        }
        
        log::debug!("Generated {} predictions for cell {}", predictions.len(), self.cell_id);
        Ok(predictions)
    }
    
    /// Check if model needs retraining
    pub fn needs_retraining(&self) -> bool {
        if let Some(last_training) = self.last_training_time {
            let elapsed = Utc::now().signed_duration_since(last_training);
            elapsed > chrono::Duration::hours(self.config.model_retrain_interval_hours as i64)
        } else {
            true
        }
    }
    
    /// Get model accuracy
    pub fn get_accuracy(&self) -> f64 {
        self.model_accuracy
    }
    
    async fn decompose_time_series(&mut self, values: &[f64]) -> Result<()> {
        // Simple linear trend extraction
        let n = values.len() as f64;
        let sum_x = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = values.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let sum_x2 = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();
        
        // Calculate linear regression coefficients manually
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;
        
        self.trend_coefficients = vec![intercept, slope];
        
        Ok(())
    }
    
    async fn fit_trend_model(&mut self, values: &[f64]) -> Result<()> {
        // Already fitted in decompose_time_series
        Ok(())
    }
    
    async fn fit_seasonal_model(&mut self, values: &[f64]) -> Result<()> {
        // Extract seasonal components
        let seasonal_period = 144; // 24 hours * 6 (10-minute intervals)
        let mut seasonal_avg = vec![0.0; seasonal_period];
        let mut seasonal_counts = vec![0; seasonal_period];
        
        for (i, &value) in values.iter().enumerate() {
            let seasonal_index = i % seasonal_period;
            seasonal_avg[seasonal_index] += value;
            seasonal_counts[seasonal_index] += 1;
        }
        
        // Calculate average for each seasonal component
        for i in 0..seasonal_period {
            if seasonal_counts[i] > 0 {
                seasonal_avg[i] /= seasonal_counts[i] as f64;
            }
        }
        
        self.seasonal_components = seasonal_avg;
        Ok(())
    }
    
    async fn calculate_residuals(&mut self, values: &[f64]) -> Result<()> {
        self.residuals = Vec::new();
        
        for (i, &actual) in values.iter().enumerate() {
            let trend = self.get_trend_value(i);
            let seasonal = self.get_seasonal_value(i);
            let predicted = trend + seasonal;
            let residual = actual - predicted;
            self.residuals.push(residual);
        }
        
        Ok(())
    }
    
    async fn evaluate_model_performance(&mut self) -> Result<()> {
        if self.residuals.is_empty() {
            self.model_accuracy = 0.0;
            return Ok(());
        }
        
        // Calculate R-squared
        let residual_sum_squares: f64 = self.residuals.iter().map(|r| r * r).sum();
        let total_sum_squares: f64 = {
            let mean = self.residuals.iter().sum::<f64>() / self.residuals.len() as f64;
            self.residuals.iter().map(|r| (r - mean) * (r - mean)).sum()
        };
        
        self.model_accuracy = if total_sum_squares > 0.0 {
            1.0 - (residual_sum_squares / total_sum_squares)
        } else {
            0.0
        };
        
        // Ensure accuracy is between 0 and 1
        self.model_accuracy = self.model_accuracy.max(0.0).min(1.0);
        
        Ok(())
    }
    
    async fn predict_single_point(&self, steps_ahead: u32, recent_data: &[PrbUtilization]) -> Result<f64> {
        // Get the last known index
        let last_index = self.historical_data.len() + steps_ahead as usize - 1;
        
        // Calculate trend component
        let trend = self.get_trend_value(last_index);
        
        // Calculate seasonal component
        let seasonal = self.get_seasonal_value(last_index);
        
        // Add some noise based on residual standard deviation
        let noise = if !self.residuals.is_empty() {
            let residual_std = self.calculate_residual_std();
            if residual_std > 0.0 {
                let normal = Normal::new(0.0, residual_std).unwrap();
                // Simple random noise generation instead of sampling
                (rand::random::<f64>() - 0.5) * residual_std * 2.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        let prediction = trend + seasonal + noise;
        
        // Ensure prediction is within reasonable bounds
        Ok(prediction.max(0.0).min(100.0))
    }
    
    fn get_trend_value(&self, index: usize) -> f64 {
        if self.trend_coefficients.len() >= 2 {
            self.trend_coefficients[0] + self.trend_coefficients[1] * index as f64
        } else {
            0.0
        }
    }
    
    fn get_seasonal_value(&self, index: usize) -> f64 {
        if !self.seasonal_components.is_empty() {
            let seasonal_index = index % self.seasonal_components.len();
            self.seasonal_components[seasonal_index]
        } else {
            0.0
        }
    }
    
    fn calculate_residual_std(&self) -> f64 {
        if self.residuals.len() <= 1 {
            return 1.0;
        }
        
        let mean = self.residuals.iter().sum::<f64>() / self.residuals.len() as f64;
        let variance = self.residuals.iter()
            .map(|r| (r - mean) * (r - mean))
            .sum::<f64>() / (self.residuals.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    fn estimate_throughput(&self, utilization: f64) -> f64 {
        // Simple linear relationship between utilization and throughput
        // This would be calibrated based on real network data
        utilization * 2.0 // Mbps per percentage point
    }
    
    fn estimate_user_count(&self, utilization: f64) -> u32 {
        // Simple relationship between utilization and user count
        // This would be calibrated based on real network data
        (utilization * 0.5) as u32
    }
    
    fn estimate_signal_quality(&self, utilization: f64) -> f64 {
        // Assume signal quality decreases slightly with higher utilization
        (1.0 - utilization * 0.001).max(0.5).min(1.0)
    }
}

/// Time series preprocessing utilities
pub struct TimeSeriesPreprocessor;

impl TimeSeriesPreprocessor {
    /// Detect and handle outliers in time series data
    pub fn detect_outliers(data: &[f64], threshold: f64) -> Vec<usize> {
        let mut outliers = Vec::new();
        
        if data.len() < 3 {
            return outliers;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std_dev = {
            let variance = data.iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f64>() / data.len() as f64;
            variance.sqrt()
        };
        
        for (i, &value) in data.iter().enumerate() {
            let z_score = (value - mean).abs() / std_dev;
            if z_score > threshold {
                outliers.push(i);
            }
        }
        
        outliers
    }
    
    /// Smooth time series data using moving average
    pub fn smooth_data(data: &[f64], window_size: usize) -> Vec<f64> {
        if window_size == 0 || window_size > data.len() {
            return data.to_vec();
        }
        
        let mut smoothed = Vec::new();
        
        for i in 0..data.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(data.len());
            
            let window_sum: f64 = data[start..end].iter().sum();
            let window_avg = window_sum / (end - start) as f64;
            
            smoothed.push(window_avg);
        }
        
        smoothed
    }
    
    /// Interpolate missing values in time series
    pub fn interpolate_missing(data: &[Option<f64>]) -> Vec<f64> {
        let mut result = Vec::new();
        let mut last_valid = None;
        let mut next_valid = None;
        
        for (i, &value) in data.iter().enumerate() {
            match value {
                Some(v) => {
                    result.push(v);
                    last_valid = Some((i, v));
                }
                None => {
                    // Find next valid value
                    next_valid = None;
                    for (j, &next_val) in data.iter().enumerate().skip(i + 1) {
                        if let Some(v) = next_val {
                            next_valid = Some((j, v));
                            break;
                        }
                    }
                    
                    // Interpolate
                    let interpolated = match (last_valid, next_valid) {
                        (Some((last_i, last_v)), Some((next_i, next_v))) => {
                            // Linear interpolation
                            let progress = (i - last_i) as f64 / (next_i - last_i) as f64;
                            last_v + progress * (next_v - last_v)
                        }
                        (Some((_, last_v)), None) => last_v, // Forward fill
                        (None, Some((_, next_v))) => next_v, // Backward fill
                        (None, None) => 0.0, // Default value
                    };
                    
                    result.push(interpolated);
                }
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::config::ForecastingConfig;
    
    #[tokio::test]
    async fn test_model_creation() {
        let config = Arc::new(ForecastingConfig::default());
        let model = ForecastingModel::new(config, "test_cell").await.unwrap();
        
        assert_eq!(model.cell_id, "test_cell");
        assert!(!model.is_trained);
        assert!(model.needs_retraining());
    }
    
    #[tokio::test]
    async fn test_insufficient_data_training() {
        let config = Arc::new(ForecastingConfig::default());
        let mut model = ForecastingModel::new(config, "test_cell").await.unwrap();
        
        // Try to train with insufficient data
        let data = vec![
            PrbUtilization::new("test_cell".to_string(), 100, 50, 100.0, 10, 0.8),
            PrbUtilization::new("test_cell".to_string(), 100, 60, 120.0, 12, 0.8),
        ];
        
        let result = model.train(&data).await;
        assert!(result.is_err());
    }
    
    #[test]
    fn test_outlier_detection() {
        let data = vec![1.0, 2.0, 3.0, 100.0, 4.0, 5.0]; // 100.0 is an outlier
        let outliers = TimeSeriesPreprocessor::detect_outliers(&data, 2.0);
        
        assert_eq!(outliers, vec![3]);
    }
    
    #[test]
    fn test_data_smoothing() {
        let data = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let smoothed = TimeSeriesPreprocessor::smooth_data(&data, 3);
        
        assert_eq!(smoothed.len(), data.len());
        // Middle values should be smoothed
        assert!((smoothed[2] - 5.0).abs() < 1.0); // Should be around average of 5,2,8
    }
    
    #[test]
    fn test_interpolation() {
        let data = vec![Some(1.0), None, Some(3.0), None, Some(5.0)];
        let interpolated = TimeSeriesPreprocessor::interpolate_missing(&data);
        
        assert_eq!(interpolated.len(), 5);
        assert_eq!(interpolated[0], 1.0);
        assert_eq!(interpolated[1], 2.0); // Should be interpolated between 1 and 3
        assert_eq!(interpolated[2], 3.0);
        assert_eq!(interpolated[3], 4.0); // Should be interpolated between 3 and 5
        assert_eq!(interpolated[4], 5.0);
    }
}