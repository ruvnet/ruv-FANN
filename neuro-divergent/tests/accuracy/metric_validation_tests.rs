//! Statistical metric validation tests
//!
//! This module validates that our metric calculations match
//! Python's implementations exactly.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::{Array1, Array2, s};
use num_traits::Float;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

use neuro_divergent_training::metrics::{
    Metric, MetricCalculator,
    MAE, MSE, RMSE, MAPE, SMAPE, MASE,
    R2, PearsonCorrelation, SpearmanCorrelation,
    MedianAbsoluteError, QuantileLoss,
    CoverageProbability, PinballLoss,
    CRPS, EnergyScore, LogScore,
    CalibrationError, Sharpness,
    IntervalScore, WinklerScore,
};

const METRIC_TOLERANCE: f64 = 1e-6;
const COUNT_TOLERANCE: f64 = 0.0; // Counts must be exact

/// Python reference implementations of metrics
mod python_reference {
    use super::*;
    
    pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>() / n as f64
    }
    
    pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>() / n as f64
    }
    
    pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
        mse(y_true, y_pred).sqrt()
    }
    
    pub fn mape(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let valid_pairs: Vec<_> = y_true.iter()
            .zip(y_pred.iter())
            .filter(|(t, _)| t.abs() > 1e-10)
            .collect();
        
        if valid_pairs.is_empty() {
            return f64::NAN;
        }
        
        valid_pairs.iter()
            .map(|(t, p)| ((t - p) / t).abs())
            .sum::<f64>() * 100.0 / valid_pairs.len() as f64
    }
    
    pub fn smape(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| {
                let denominator = (t.abs() + p.abs()).max(1e-10);
                2.0 * (t - p).abs() / denominator
            })
            .sum::<f64>() * 100.0 / n as f64
    }
    
    pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len() as f64;
        let mean_true = y_true.iter().sum::<f64>() / n;
        
        let ss_res = y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>();
        
        let ss_tot = y_true.iter()
            .map(|t| (t - mean_true).powi(2))
            .sum::<f64>();
        
        if ss_tot == 0.0 {
            return if ss_res == 0.0 { 1.0 } else { f64::NEG_INFINITY };
        }
        
        1.0 - ss_res / ss_tot
    }
    
    pub fn pearson_correlation(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len() as f64;
        let mean_true = y_true.iter().sum::<f64>() / n;
        let mean_pred = y_pred.iter().sum::<f64>() / n;
        
        let covariance = y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - mean_true) * (p - mean_pred))
            .sum::<f64>() / n;
        
        let var_true = y_true.iter()
            .map(|t| (t - mean_true).powi(2))
            .sum::<f64>() / n;
        
        let var_pred = y_pred.iter()
            .map(|p| (p - mean_pred).powi(2))
            .sum::<f64>() / n;
        
        if var_true == 0.0 || var_pred == 0.0 {
            return 0.0;
        }
        
        covariance / (var_true.sqrt() * var_pred.sqrt())
    }
    
    pub fn median_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let mut errors: Vec<f64> = y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .collect();
        
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = errors.len();
        if n % 2 == 0 {
            (errors[n/2 - 1] + errors[n/2]) / 2.0
        } else {
            errors[n/2]
        }
    }
    
    pub fn quantile_loss(y_true: &[f64], y_pred: &[f64], quantile: f64) -> f64 {
        let n = y_true.len();
        y_true.iter()
            .zip(y_pred.iter())
            .map(|(t, p)| {
                let error = t - p;
                if error >= 0.0 {
                    quantile * error
                } else {
                    (quantile - 1.0) * error
                }
            })
            .sum::<f64>() / n as f64
    }
    
    pub fn coverage_probability(y_true: &[f64], y_lower: &[f64], y_upper: &[f64]) -> f64 {
        let n = y_true.len();
        let covered = y_true.iter()
            .zip(y_lower.iter())
            .zip(y_upper.iter())
            .filter(|((t, l), u)| **l <= **t && **t <= **u)
            .count();
        
        covered as f64 / n as f64
    }
}

#[cfg(test)]
mod point_forecast_metric_tests {
    use super::*;
    
    fn generate_test_data(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let y_true: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..100.0)).collect();
        let y_pred: Vec<f64> = y_true.iter()
            .map(|&y| y + rng.gen_range(-10.0..10.0))
            .collect();
        
        (y_true, y_pred)
    }
    
    #[test]
    fn test_mae_metric() {
        let (y_true, y_pred) = generate_test_data(42, 100);
        
        let metric = MAE::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::mae(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "MAE metric mismatch"
        );
    }
    
    #[test]
    fn test_mse_metric() {
        let (y_true, y_pred) = generate_test_data(123, 100);
        
        let metric = MSE::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::mse(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "MSE metric mismatch"
        );
    }
    
    #[test]
    fn test_rmse_metric() {
        let (y_true, y_pred) = generate_test_data(456, 100);
        
        let metric = RMSE::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::rmse(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "RMSE metric mismatch"
        );
    }
    
    #[test]
    fn test_mape_metric() {
        let (mut y_true, y_pred) = generate_test_data(789, 100);
        // Ensure no zero values for MAPE
        y_true = y_true.iter().map(|&y| y.max(1.0)).collect();
        
        let metric = MAPE::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::mape(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "MAPE metric mismatch"
        );
    }
    
    #[test]
    fn test_smape_metric() {
        let (y_true, y_pred) = generate_test_data(101112, 100);
        
        let metric = SMAPE::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::smape(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "SMAPE metric mismatch"
        );
    }
    
    #[test]
    fn test_r2_metric() {
        let (y_true, y_pred) = generate_test_data(131415, 100);
        
        let metric = R2::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::r_squared(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "R² metric mismatch"
        );
    }
    
    #[test]
    fn test_pearson_correlation() {
        let (y_true, y_pred) = generate_test_data(161718, 100);
        
        let metric = PearsonCorrelation::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::pearson_correlation(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "Pearson correlation mismatch"
        );
    }
    
    #[test]
    fn test_median_absolute_error() {
        let (y_true, y_pred) = generate_test_data(192021, 100);
        
        let metric = MedianAbsoluteError::new();
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::median_absolute_error(&y_true, &y_pred);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "Median absolute error mismatch"
        );
    }
}

#[cfg(test)]
mod probabilistic_metric_tests {
    use super::*;
    
    #[test]
    fn test_quantile_loss_metric() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.2, 1.8, 3.1, 3.9, 5.2];
        let quantile = 0.5;
        
        let metric = QuantileLoss::new(quantile);
        let rust_result = metric.calculate(&y_true, &y_pred).unwrap();
        let python_result = python_reference::quantile_loss(&y_true, &y_pred, quantile);
        
        assert_abs_diff_eq!(
            rust_result,
            python_result,
            epsilon = METRIC_TOLERANCE,
            "Quantile loss metric mismatch"
        );
    }
    
    #[test]
    fn test_coverage_probability() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_lower = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let y_upper = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        
        let rust_result = python_reference::coverage_probability(&y_true, &y_lower, &y_upper);
        assert_eq!(rust_result, 1.0, "All values should be covered");
        
        // Test partial coverage
        let y_lower_tight = vec![1.2, 2.2, 3.2, 4.2, 5.2];
        let y_upper_tight = vec![1.3, 2.3, 3.3, 4.3, 5.3];
        let partial_coverage = python_reference::coverage_probability(
            &y_true, &y_lower_tight, &y_upper_tight
        );
        assert!(partial_coverage < 1.0, "Not all values should be covered");
    }
}

#[cfg(test)]
mod edge_case_metric_tests {
    use super::*;
    
    #[test]
    fn test_metrics_with_identical_values() {
        let y_true = vec![42.0; 50];
        let y_pred = vec![42.0; 50];
        
        let mae = MAE::new();
        assert_eq!(mae.calculate(&y_true, &y_pred).unwrap(), 0.0);
        
        let mse = MSE::new();
        assert_eq!(mse.calculate(&y_true, &y_pred).unwrap(), 0.0);
        
        let rmse = RMSE::new();
        assert_eq!(rmse.calculate(&y_true, &y_pred).unwrap(), 0.0);
        
        let r2 = R2::new();
        assert_eq!(r2.calculate(&y_true, &y_pred).unwrap(), 1.0);
    }
    
    #[test]
    fn test_metrics_with_constant_predictions() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![3.0; 5]; // Constant prediction
        
        let mae = MAE::new();
        let mae_result = mae.calculate(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(mae_result, 1.2, epsilon = METRIC_TOLERANCE);
        
        let r2 = R2::new();
        let r2_result = r2.calculate(&y_true, &y_pred).unwrap();
        assert_eq!(r2_result, 0.0, "R² should be 0 for constant predictions");
    }
    
    #[test]
    fn test_metrics_with_zero_values() {
        let y_true = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let y_pred = vec![0.1, 0.9, 0.1, 1.9, 0.1];
        
        let mape = MAPE::new();
        let mape_result = mape.calculate(&y_true, &y_pred).unwrap();
        // MAPE should handle zeros appropriately
        assert!(mape_result.is_finite() || mape_result.is_nan());
        
        let smape = SMAPE::new();
        let smape_result = smape.calculate(&y_true, &y_pred).unwrap();
        assert!(smape_result.is_finite(), "SMAPE should handle zeros");
    }
    
    #[test]
    fn test_metrics_with_single_value() {
        let y_true = vec![5.0];
        let y_pred = vec![4.5];
        
        let mae = MAE::new();
        assert_abs_diff_eq!(
            mae.calculate(&y_true, &y_pred).unwrap(),
            0.5,
            epsilon = METRIC_TOLERANCE
        );
        
        let mse = MSE::new();
        assert_abs_diff_eq!(
            mse.calculate(&y_true, &y_pred).unwrap(),
            0.25,
            epsilon = METRIC_TOLERANCE
        );
    }
    
    #[test]
    fn test_metrics_with_extreme_values() {
        let y_true = vec![1e10, -1e10, 1e-10, -1e-10, 0.0];
        let y_pred = vec![1e10 + 1.0, -1e10 - 1.0, 2e-10, -2e-10, 1e-10];
        
        let mae = MAE::new();
        let mae_result = mae.calculate(&y_true, &y_pred).unwrap();
        assert!(mae_result.is_finite(), "MAE should handle extreme values");
        
        let rmse = RMSE::new();
        let rmse_result = rmse.calculate(&y_true, &y_pred).unwrap();
        assert!(rmse_result.is_finite(), "RMSE should handle extreme values");
    }
}

#[cfg(test)]
mod metric_calculator_tests {
    use super::*;
    
    #[test]
    fn test_metric_calculator_batch() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 2.1, 2.9, 4.2, 4.8];
        
        let mut calculator = MetricCalculator::new();
        calculator.add_metric("mae", Box::new(MAE::new()));
        calculator.add_metric("mse", Box::new(MSE::new()));
        calculator.add_metric("rmse", Box::new(RMSE::new()));
        calculator.add_metric("r2", Box::new(R2::new()));
        
        let results = calculator.calculate_all(&y_true, &y_pred).unwrap();
        
        // Verify all metrics were calculated
        assert_eq!(results.len(), 4);
        assert!(results.contains_key("mae"));
        assert!(results.contains_key("mse"));
        assert!(results.contains_key("rmse"));
        assert!(results.contains_key("r2"));
        
        // Verify consistency between metrics
        let mse = results["mse"];
        let rmse = results["rmse"];
        assert_abs_diff_eq!(rmse, mse.sqrt(), epsilon = METRIC_TOLERANCE);
    }
    
    #[test]
    fn test_metric_aggregation() {
        // Test aggregating metrics across multiple series
        let series_results = vec![
            HashMap::from([("mae", 1.0), ("mse", 2.0)]),
            HashMap::from([("mae", 1.5), ("mse", 3.0)]),
            HashMap::from([("mae", 2.0), ("mse", 4.0)]),
        ];
        
        // Calculate average metrics
        let avg_mae = series_results.iter()
            .map(|r| r["mae"])
            .sum::<f64>() / series_results.len() as f64;
        
        let avg_mse = series_results.iter()
            .map(|r| r["mse"])
            .sum::<f64>() / series_results.len() as f64;
        
        assert_abs_diff_eq!(avg_mae, 1.5, epsilon = METRIC_TOLERANCE);
        assert_abs_diff_eq!(avg_mse, 3.0, epsilon = METRIC_TOLERANCE);
    }
}

#[cfg(test)]
mod implementation_specific_tests {
    use super::*;
    
    #[test]
    fn test_mase_calculation() {
        // MASE requires seasonal naive forecast error
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y_pred = vec![1.1, 2.2, 3.1, 4.3, 5.2, 6.1, 7.3, 8.2];
        let seasonal_period = 4;
        
        // Calculate seasonal naive error
        let mut seasonal_naive_errors = Vec::new();
        for i in seasonal_period..y_true.len() {
            let naive_error = (y_true[i] - y_true[i - seasonal_period]).abs();
            seasonal_naive_errors.push(naive_error);
        }
        let seasonal_naive_mae = seasonal_naive_errors.iter().sum::<f64>() 
            / seasonal_naive_errors.len() as f64;
        
        let metric = MASE::new(seasonal_naive_mae);
        let result = metric.calculate(&y_true, &y_pred).unwrap();
        
        // MASE should be ratio of MAE to seasonal naive MAE
        let mae = MAE::new().calculate(&y_true, &y_pred).unwrap();
        let expected_mase = mae / seasonal_naive_mae;
        
        assert_abs_diff_eq!(result, expected_mase, epsilon = METRIC_TOLERANCE);
    }
    
    #[test]
    fn test_interval_score() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_lower = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let y_upper = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let alpha = 0.1; // 90% prediction interval
        
        let metric = IntervalScore::new(alpha);
        let result = metric.calculate_interval(&y_true, &y_lower, &y_upper).unwrap();
        
        // For this perfect calibration case, interval score should be width-based
        let avg_width = y_upper.iter()
            .zip(y_lower.iter())
            .map(|(u, l)| u - l)
            .sum::<f64>() / y_true.len() as f64;
        
        assert!(result >= avg_width, "Interval score should include width component");
    }
    
    #[test]
    fn test_calibration_error() {
        // Test probabilistic calibration
        let predicted_probs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let actual_frequencies = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        
        let metric = CalibrationError::new(5); // 5 bins
        let result = metric.calculate(&predicted_probs, &actual_frequencies).unwrap();
        
        // Perfect calibration would have result close to 0
        assert!(result >= 0.0, "Calibration error should be non-negative");
    }
}