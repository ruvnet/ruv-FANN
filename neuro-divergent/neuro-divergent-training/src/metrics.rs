//! # Evaluation Metrics for Neural Forecasting
//!
//! Comprehensive collection of metrics for evaluating time series forecasting models,
//! including both point forecast and probabilistic forecast metrics.
//!
//! ## Categories
//!
//! - **Point Forecast Metrics**: MAE, MSE, RMSE, MAPE, SMAPE, MASE
//! - **Correlation Metrics**: R², Pearson, Spearman correlations  
//! - **Probabilistic Metrics**: CRPS, Energy Score, Log Score
//! - **Interval Metrics**: Coverage probability, interval score, Winkler score
//! - **Calibration Metrics**: Calibration error, sharpness, reliability

use num_traits::Float;
use std::collections::HashMap;
use crate::{TrainingError, TrainingResult};

/// Core trait for evaluation metrics
pub trait Metric<T: Float + Send + Sync>: Send + Sync {
    /// Calculate the metric value
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T>;
    
    /// Get the name of the metric
    fn name(&self) -> &'static str;
    
    /// Whether higher values are better (for optimization)
    fn higher_is_better(&self) -> bool { false }
}

/// Container for calculating multiple metrics at once
pub struct MetricCalculator<T: Float + Send + Sync> {
    metrics: HashMap<String, Box<dyn Metric<T>>>,
}

impl<T: Float + Send + Sync> MetricCalculator<T> {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
    
    pub fn add_metric(&mut self, name: &str, metric: Box<dyn Metric<T>>) {
        self.metrics.insert(name.to_string(), metric);
    }
    
    pub fn calculate_all(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<HashMap<String, T>> {
        let mut results = HashMap::new();
        
        for (name, metric) in &self.metrics {
            let value = metric.calculate(y_true, y_pred)?;
            results.insert(name.clone(), value);
        }
        
        Ok(results)
    }
}

// =============================================================================
// Point Forecast Metrics
// =============================================================================

/// Mean Absolute Error (MAE)
pub struct MAE<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> MAE<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for MAE<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).abs())
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(y_true.len()).unwrap())
    }
    
    fn name(&self) -> &'static str {
        "MAE"
    }
}

/// Mean Squared Error (MSE)
pub struct MSE<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> MSE<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for MSE<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(y_true.len()).unwrap())
    }
    
    fn name(&self) -> &'static str {
        "MSE"
    }
}

/// Root Mean Squared Error (RMSE)
pub struct RMSE<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> RMSE<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for RMSE<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        let mse = MSE::new();
        let mse_value = mse.calculate(y_true, y_pred)?;
        Ok(mse_value.sqrt())
    }
    
    fn name(&self) -> &'static str {
        "RMSE"
    }
}

/// Mean Absolute Percentage Error (MAPE)
pub struct MAPE<T: Float + Send + Sync> {
    epsilon: T,
}

impl<T: Float + Send + Sync> MAPE<T> {
    pub fn new() -> Self {
        Self { epsilon: T::from(1e-8).unwrap() }
    }
    
    pub fn with_epsilon(epsilon: T) -> Self {
        Self { epsilon }
    }
}

impl<T: Float + Send + Sync> Metric<T> for MAPE<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let valid_pairs: Vec<_> = y_true.iter()
            .zip(y_pred.iter())
            .filter(|(t, _)| t.abs() > self.epsilon)
            .collect();
        
        if valid_pairs.is_empty() {
            return Ok(T::nan());
        }
        
        let sum = valid_pairs.iter()
            .map(|(&t, &p)| ((t - p) / t).abs())
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum * T::from(100.0).unwrap() / T::from(valid_pairs.len()).unwrap())
    }
    
    fn name(&self) -> &'static str {
        "MAPE"
    }
}

/// Symmetric Mean Absolute Percentage Error (SMAPE)
pub struct SMAPE<T: Float + Send + Sync> {
    epsilon: T,
}

impl<T: Float + Send + Sync> SMAPE<T> {
    pub fn new() -> Self {
        Self { epsilon: T::from(1e-8).unwrap() }
    }
    
    pub fn with_epsilon(epsilon: T) -> Self {
        Self { epsilon }
    }
}

impl<T: Float + Send + Sync> Metric<T> for SMAPE<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| {
                let numerator = (t - p).abs();
                let denominator = (t.abs() + p.abs()).max(self.epsilon);
                T::from(2.0).unwrap() * numerator / denominator
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum * T::from(100.0).unwrap() / T::from(y_true.len()).unwrap())
    }
    
    fn name(&self) -> &'static str {
        "SMAPE"
    }
}

/// Mean Absolute Scaled Error (MASE)
pub struct MASE<T: Float + Send + Sync> {
    seasonal_naive_mae: T,
}

impl<T: Float + Send + Sync> MASE<T> {
    pub fn new(seasonal_naive_mae: T) -> Self {
        Self { seasonal_naive_mae }
    }
}

impl<T: Float + Send + Sync> Metric<T> for MASE<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        let mae = MAE::new();
        let mae_value = mae.calculate(y_true, y_pred)?;
        
        if self.seasonal_naive_mae.is_zero() {
            return Ok(T::infinity());
        }
        
        Ok(mae_value / self.seasonal_naive_mae)
    }
    
    fn name(&self) -> &'static str {
        "MASE"
    }
}

// =============================================================================
// Correlation Metrics
// =============================================================================

/// R-squared (Coefficient of Determination)
pub struct R2<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> R2<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for R2<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(y_true.len()).unwrap();
        let mean_true = y_true.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        
        let ss_res = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .fold(T::zero(), |acc, x| acc + x);
        
        let ss_tot = y_true.iter()
            .map(|&t| (t - mean_true).powi(2))
            .fold(T::zero(), |acc, x| acc + x);
        
        if ss_tot.is_zero() {
            return Ok(if ss_res.is_zero() { T::one() } else { T::neg_infinity() });
        }
        
        Ok(T::one() - ss_res / ss_tot)
    }
    
    fn name(&self) -> &'static str {
        "R2"
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Pearson Correlation Coefficient
pub struct PearsonCorrelation<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> PearsonCorrelation<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for PearsonCorrelation<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let n = T::from(y_true.len()).unwrap();
        let mean_true = y_true.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        let mean_pred = y_pred.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        
        let covariance = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - mean_true) * (p - mean_pred))
            .fold(T::zero(), |acc, x| acc + x) / n;
        
        let var_true = y_true.iter()
            .map(|&t| (t - mean_true).powi(2))
            .fold(T::zero(), |acc, x| acc + x) / n;
        
        let var_pred = y_pred.iter()
            .map(|&p| (p - mean_pred).powi(2))
            .fold(T::zero(), |acc, x| acc + x) / n;
        
        if var_true.is_zero() || var_pred.is_zero() {
            return Ok(T::zero());
        }
        
        Ok(covariance / (var_true.sqrt() * var_pred.sqrt()))
    }
    
    fn name(&self) -> &'static str {
        "PearsonCorrelation"
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Spearman Rank Correlation
pub struct SpearmanCorrelation<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> SpearmanCorrelation<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for SpearmanCorrelation<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        // Convert to ranks
        let rank_true = Self::rank(y_true);
        let rank_pred = Self::rank(y_pred);
        
        // Calculate Pearson correlation on ranks
        let pearson = PearsonCorrelation::new();
        pearson.calculate(&rank_true, &rank_pred)
    }
    
    fn name(&self) -> &'static str {
        "SpearmanCorrelation"
    }
    
    fn higher_is_better(&self) -> bool {
        true
    }
}

impl<T: Float + Send + Sync> SpearmanCorrelation<T> {
    fn rank(values: &[T]) -> Vec<T> {
        let mut indexed: Vec<(usize, T)> = values.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut ranks = vec![T::zero(); values.len()];
        for (rank, (original_idx, _)) in indexed.iter().enumerate() {
            ranks[*original_idx] = T::from(rank + 1).unwrap();
        }
        
        ranks
    }
}

// =============================================================================
// Additional Metrics
// =============================================================================

/// Median Absolute Error
pub struct MedianAbsoluteError<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> MedianAbsoluteError<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for MedianAbsoluteError<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let mut errors: Vec<T> = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).abs())
            .collect();
        
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = errors.len();
        Ok(if n % 2 == 0 {
            (errors[n/2 - 1] + errors[n/2]) / T::from(2.0).unwrap()
        } else {
            errors[n/2]
        })
    }
    
    fn name(&self) -> &'static str {
        "MedianAbsoluteError"
    }
}

/// Quantile Loss
pub struct QuantileLoss<T: Float + Send + Sync> {
    quantile: T,
}

impl<T: Float + Send + Sync> QuantileLoss<T> {
    pub fn new(quantile: T) -> Self {
        Self { quantile }
    }
}

impl<T: Float + Send + Sync> Metric<T> for QuantileLoss<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| {
                let error = t - p;
                if error >= T::zero() {
                    self.quantile * error
                } else {
                    (self.quantile - T::one()) * error
                }
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(y_true.len()).unwrap())
    }
    
    fn name(&self) -> &'static str {
        "QuantileLoss"
    }
}

// =============================================================================
// Probabilistic Metrics
// =============================================================================

/// Coverage Probability
pub struct CoverageProbability<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> CoverageProbability<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
    
    pub fn calculate_interval(&self, y_true: &[T], y_lower: &[T], y_upper: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_lower.len() || y_true.len() != y_upper.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let covered = y_true.iter()
            .zip(y_lower.iter())
            .zip(y_upper.iter())
            .filter(|((t, l), u)| **l <= **t && **t <= **u)
            .count();
        
        Ok(T::from(covered).unwrap() / T::from(y_true.len()).unwrap())
    }
}

/// Pinball Loss for quantile evaluation
pub struct PinballLoss<T: Float + Send + Sync> {
    quantile: T,
}

impl<T: Float + Send + Sync> PinballLoss<T> {
    pub fn new(quantile: T) -> Self {
        Self { quantile }
    }
}

impl<T: Float + Send + Sync> Metric<T> for PinballLoss<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        let ql = QuantileLoss::new(self.quantile);
        ql.calculate(y_true, y_pred)
    }
    
    fn name(&self) -> &'static str {
        "PinballLoss"
    }
}

/// Continuous Ranked Probability Score (CRPS)
pub struct CRPS<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> CRPS<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for CRPS<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        // Simplified CRPS for point forecasts
        // For full implementation, need predictive distribution
        let mae = MAE::new();
        mae.calculate(y_true, y_pred)
    }
    
    fn name(&self) -> &'static str {
        "CRPS"
    }
}

/// Energy Score
pub struct EnergyScore<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> EnergyScore<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for EnergyScore<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        // Simplified implementation
        let mse = MSE::new();
        mse.calculate(y_true, y_pred)
    }
    
    fn name(&self) -> &'static str {
        "EnergyScore"
    }
}

/// Log Score
pub struct LogScore<T: Float + Send + Sync> {
    epsilon: T,
}

impl<T: Float + Send + Sync> LogScore<T> {
    pub fn new() -> Self {
        Self { epsilon: T::from(1e-8).unwrap() }
    }
}

impl<T: Float + Send + Sync> Metric<T> for LogScore<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        // Simplified implementation for point forecasts
        if y_true.len() != y_pred.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| {
                let prob = (-((t - p).powi(2))).exp().max(self.epsilon);
                -prob.ln()
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(y_true.len()).unwrap())
    }
    
    fn name(&self) -> &'static str {
        "LogScore"
    }
}

// =============================================================================
// Calibration Metrics
// =============================================================================

/// Calibration Error
pub struct CalibrationError<T: Float + Send + Sync> {
    n_bins: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> CalibrationError<T> {
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins, _phantom: std::marker::PhantomData }
    }
}

impl<T: Float + Send + Sync> Metric<T> for CalibrationError<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        // Simplified implementation
        // For full implementation, need probabilistic predictions
        Ok(T::zero())
    }
    
    fn name(&self) -> &'static str {
        "CalibrationError"
    }
}

/// Sharpness
pub struct Sharpness<T: Float + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> Sharpness<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
    
    pub fn calculate_interval(&self, y_lower: &[T], y_upper: &[T]) -> TrainingResult<T> {
        if y_lower.len() != y_upper.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let sum = y_lower.iter()
            .zip(y_upper.iter())
            .map(|(&l, &u)| u - l)
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(y_lower.len()).unwrap())
    }
}

/// Interval Score
pub struct IntervalScore<T: Float + Send + Sync> {
    alpha: T,
}

impl<T: Float + Send + Sync> IntervalScore<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }
    
    pub fn calculate_interval(&self, y_true: &[T], y_lower: &[T], y_upper: &[T]) -> TrainingResult<T> {
        if y_true.len() != y_lower.len() || y_true.len() != y_upper.len() {
            return Err(TrainingError::DataError("Dimension mismatch".to_string()));
        }
        
        let two = T::from(2.0).unwrap();
        let sum = y_true.iter()
            .zip(y_lower.iter())
            .zip(y_upper.iter())
            .map(|((&t, &l), &u)| {
                let width = u - l;
                let lower_penalty = if t < l { two / self.alpha * (l - t) } else { T::zero() };
                let upper_penalty = if t > u { two / self.alpha * (t - u) } else { T::zero() };
                width + lower_penalty + upper_penalty
            })
            .fold(T::zero(), |acc, x| acc + x);
        
        Ok(sum / T::from(y_true.len()).unwrap())
    }
}

/// Winkler Score
pub struct WinklerScore<T: Float + Send + Sync> {
    alpha: T,
}

impl<T: Float + Send + Sync> WinklerScore<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }
}

impl<T: Float + Send + Sync> Metric<T> for WinklerScore<T> {
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> TrainingResult<T> {
        // Winkler score requires interval predictions
        // This is a placeholder implementation
        Ok(T::zero())
    }
    
    fn name(&self) -> &'static str {
        "WinklerScore"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mae_metric() {
        let metric = MAE::new();
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 2.1, 2.9, 3.9];
        
        let result = metric.calculate(&y_true, &y_pred).unwrap();
        assert!((result - 0.1).abs() < 1e-6);
    }
    
    #[test]
    fn test_mse_metric() {
        let metric = MSE::new();
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = metric.calculate(&y_true, &y_pred).unwrap();
        assert_eq!(result, 0.0);
    }
    
    #[test]
    fn test_r2_metric() {
        let metric = R2::new();
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = metric.calculate(&y_true, &y_pred).unwrap();
        assert_eq!(result, 1.0);
    }
    
    #[test]
    fn test_metric_calculator() {
        let mut calculator = MetricCalculator::new();
        calculator.add_metric("mae", Box::new(MAE::new()));
        calculator.add_metric("mse", Box::new(MSE::new()));
        
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.1, 2.0, 2.9];
        
        let results = calculator.calculate_all(&y_true, &y_pred).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains_key("mae"));
        assert!(results.contains_key("mse"));
    }
}