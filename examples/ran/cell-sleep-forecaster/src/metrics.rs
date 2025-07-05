//! Metrics calculation and evaluation module

use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use statrs::statistics::{Statistics, Data};

use crate::{PrbUtilization, SleepWindow, ForecastingMetrics};

/// Comprehensive performance metrics calculator
pub struct MetricsCalculator;

/// Forecast evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastEvaluation {
    pub mape: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub directional_accuracy: f64,
    pub prediction_interval_coverage: f64,
    pub low_traffic_detection_metrics: DetectionMetrics,
}

/// Detection performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetrics {
    pub true_positives: u32,
    pub true_negatives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy: f64,
    pub specificity: f64,
    pub sensitivity: f64,
}

/// Energy savings analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySavingsAnalysis {
    pub total_energy_saved_kwh: f64,
    pub average_savings_per_window: f64,
    pub savings_efficiency_percent: f64,
    pub co2_reduction_kg: f64,
    pub cost_savings_usd: f64,
    pub roi_percent: f64,
    pub payback_period_months: f64,
}

/// Sleep window performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepWindowMetrics {
    pub total_windows_scheduled: u32,
    pub total_windows_executed: u32,
    pub execution_success_rate: f64,
    pub average_duration_minutes: f64,
    pub total_sleep_time_hours: f64,
    pub utilization_during_sleep: Vec<f64>,
    pub unexpected_wakeups: u32,
    pub network_impact_score: f64,
}

/// Real-time performance dashboard metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub timestamp: DateTime<Utc>,
    pub system_status: SystemStatus,
    pub forecasting_performance: ForecastEvaluation,
    pub energy_savings: EnergySavingsAnalysis,
    pub sleep_windows: SleepWindowMetrics,
    pub cell_count: u32,
    pub active_forecasts: u32,
    pub alerts_count: u32,
    pub uptime_hours: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemStatus {
    Optimal,
    Good,
    Warning,
    Critical,
    Offline,
}

impl MetricsCalculator {
    /// Calculate comprehensive forecast evaluation metrics
    pub fn evaluate_forecast(
        actual: &[PrbUtilization],
        predicted: &[PrbUtilization],
        low_traffic_threshold: f64,
    ) -> Result<ForecastEvaluation> {
        if actual.len() != predicted.len() {
            return Err(anyhow::anyhow!("Actual and predicted data must have the same length"));
        }
        
        if actual.is_empty() {
            return Err(anyhow::anyhow!("Cannot evaluate empty dataset"));
        }
        
        let actual_values: Vec<f64> = actual.iter().map(|d| d.utilization_percentage).collect();
        let predicted_values: Vec<f64> = predicted.iter().map(|d| d.utilization_percentage).collect();
        
        // Basic metrics
        let mape = Self::calculate_mape(&actual_values, &predicted_values)?;
        let rmse = Self::calculate_rmse(&actual_values, &predicted_values)?;
        let mae = Self::calculate_mae(&actual_values, &predicted_values)?;
        let r2 = Self::calculate_r2(&actual_values, &predicted_values)?;
        
        // Classification metrics for accuracy assessment
        let accuracy = Self::calculate_accuracy(&actual_values, &predicted_values, 5.0)?; // 5% tolerance
        let directional_accuracy = Self::calculate_directional_accuracy(&actual_values, &predicted_values)?;
        
        // Low traffic detection metrics
        let detection_metrics = Self::calculate_detection_metrics(
            &actual_values,
            &predicted_values,
            low_traffic_threshold,
        )?;
        
        // Prediction interval coverage (simplified)
        let prediction_interval_coverage = Self::calculate_prediction_interval_coverage(
            &actual_values,
            &predicted_values,
        )?;
        
        Ok(ForecastEvaluation {
            mape,
            rmse,
            mae,
            r2,
            accuracy,
            precision: detection_metrics.precision,
            recall: detection_metrics.recall,
            f1_score: detection_metrics.f1_score,
            directional_accuracy,
            prediction_interval_coverage,
            low_traffic_detection_metrics: detection_metrics,
        })
    }
    
    /// Calculate energy savings analysis
    pub fn analyze_energy_savings(
        sleep_windows: &[SleepWindow],
        energy_cost_per_kwh: f64,
        co2_emission_factor_kg_per_kwh: f64,
    ) -> Result<EnergySavingsAnalysis> {
        if sleep_windows.is_empty() {
            return Ok(EnergySavingsAnalysis {
                total_energy_saved_kwh: 0.0,
                average_savings_per_window: 0.0,
                savings_efficiency_percent: 0.0,
                co2_reduction_kg: 0.0,
                cost_savings_usd: 0.0,
                roi_percent: 0.0,
                payback_period_months: 0.0,
            });
        }
        
        let total_energy_saved: f64 = sleep_windows.iter()
            .map(|w| w.energy_savings_kwh)
            .sum();
        
        let average_savings_per_window = total_energy_saved / sleep_windows.len() as f64;
        
        // Calculate efficiency (actual savings vs theoretical maximum)
        let total_duration_hours: f64 = sleep_windows.iter()
            .map(|w| w.duration_minutes as f64 / 60.0)
            .sum();
        
        let theoretical_max_savings = total_duration_hours * 1.0; // Assume 1 kWh per hour base consumption
        let savings_efficiency_percent = if theoretical_max_savings > 0.0 {
            (total_energy_saved / theoretical_max_savings) * 100.0
        } else {
            0.0
        };
        
        // Environmental impact
        let co2_reduction_kg = total_energy_saved * co2_emission_factor_kg_per_kwh;
        
        // Financial impact
        let cost_savings_usd = total_energy_saved * energy_cost_per_kwh;
        
        // ROI calculation (simplified - assume implementation cost)
        let implementation_cost_usd = 10000.0; // Placeholder
        let roi_percent = if implementation_cost_usd > 0.0 {
            ((cost_savings_usd - implementation_cost_usd) / implementation_cost_usd) * 100.0
        } else {
            0.0
        };
        
        // Payback period
        let annual_savings = cost_savings_usd * 365.0 / 30.0; // Extrapolate monthly to annual
        let payback_period_months = if annual_savings > 0.0 {
            (implementation_cost_usd / annual_savings) * 12.0
        } else {
            f64::INFINITY
        };
        
        Ok(EnergySavingsAnalysis {
            total_energy_saved_kwh: total_energy_saved,
            average_savings_per_window,
            savings_efficiency_percent,
            co2_reduction_kg,
            cost_savings_usd,
            roi_percent,
            payback_period_months,
        })
    }
    
    /// Calculate sleep window performance metrics
    pub fn analyze_sleep_window_performance(
        scheduled_windows: &[SleepWindow],
        executed_windows: &[SleepWindow],
        actual_utilization_during_sleep: &[PrbUtilization],
    ) -> Result<SleepWindowMetrics> {
        let total_windows_scheduled = scheduled_windows.len() as u32;
        let total_windows_executed = executed_windows.len() as u32;
        
        let execution_success_rate = if total_windows_scheduled > 0 {
            (total_windows_executed as f64 / total_windows_scheduled as f64) * 100.0
        } else {
            0.0
        };
        
        let average_duration_minutes = if !executed_windows.is_empty() {
            executed_windows.iter()
                .map(|w| w.duration_minutes as f64)
                .sum::<f64>() / executed_windows.len() as f64
        } else {
            0.0
        };
        
        let total_sleep_time_hours = executed_windows.iter()
            .map(|w| w.duration_minutes as f64 / 60.0)
            .sum();
        
        let utilization_during_sleep: Vec<f64> = actual_utilization_during_sleep.iter()
            .map(|d| d.utilization_percentage)
            .collect();
        
        // Count unexpected wakeups (utilization spikes during scheduled sleep)
        let unexpected_wakeups = utilization_during_sleep.iter()
            .filter(|&&util| util > 25.0) // Threshold for unexpected activity
            .count() as u32;
        
        // Network impact score (simplified)
        let avg_utilization_during_sleep = if !utilization_during_sleep.is_empty() {
            utilization_during_sleep.iter().sum::<f64>() / utilization_during_sleep.len() as f64
        } else {
            0.0
        };
        
        let network_impact_score = if avg_utilization_during_sleep < 10.0 {
            1.0 // Minimal impact
        } else if avg_utilization_during_sleep < 20.0 {
            0.8 // Low impact
        } else if avg_utilization_during_sleep < 30.0 {
            0.6 // Moderate impact
        } else {
            0.3 // High impact
        };
        
        Ok(SleepWindowMetrics {
            total_windows_scheduled,
            total_windows_executed,
            execution_success_rate,
            average_duration_minutes,
            total_sleep_time_hours,
            utilization_during_sleep,
            unexpected_wakeups,
            network_impact_score,
        })
    }
    
    /// Generate comprehensive dashboard metrics
    pub fn generate_dashboard_metrics(
        forecasting_metrics: &ForecastingMetrics,
        sleep_windows: &[SleepWindow],
        energy_analysis: &EnergySavingsAnalysis,
        sleep_metrics: &SleepWindowMetrics,
        cell_count: u32,
        active_forecasts: u32,
        alerts_count: u32,
        uptime_hours: f64,
    ) -> Result<DashboardMetrics> {
        // Determine system status based on key metrics
        let system_status = if forecasting_metrics.mape > 15.0 || 
                              forecasting_metrics.low_traffic_detection_rate < 90.0 {
            SystemStatus::Critical
        } else if forecasting_metrics.mape > 10.0 || 
                  forecasting_metrics.low_traffic_detection_rate < 95.0 {
            SystemStatus::Warning
        } else if forecasting_metrics.mape <= 8.0 && 
                  forecasting_metrics.low_traffic_detection_rate >= 97.0 {
            SystemStatus::Optimal
        } else {
            SystemStatus::Good
        };
        
        // Create simplified forecast evaluation for dashboard
        let forecasting_performance = ForecastEvaluation {
            mape: forecasting_metrics.mape,
            rmse: 0.0, // Simplified for dashboard
            mae: 0.0,  // Simplified for dashboard
            r2: forecasting_metrics.r2,
            accuracy: forecasting_metrics.accuracy,
            precision: forecasting_metrics.precision,
            recall: forecasting_metrics.recall,
            f1_score: forecasting_metrics.f1_score,
            directional_accuracy: 0.0, // Simplified for dashboard
            prediction_interval_coverage: 0.0, // Simplified for dashboard
            low_traffic_detection_metrics: DetectionMetrics {
                true_positives: 0,  // Simplified for dashboard
                true_negatives: 0,  // Simplified for dashboard
                false_positives: 0, // Simplified for dashboard
                false_negatives: 0, // Simplified for dashboard
                precision: forecasting_metrics.precision,
                recall: forecasting_metrics.recall,
                f1_score: forecasting_metrics.f1_score,
                accuracy: forecasting_metrics.accuracy,
                specificity: 0.0, // Simplified for dashboard
                sensitivity: forecasting_metrics.recall,
            },
        };
        
        Ok(DashboardMetrics {
            timestamp: Utc::now(),
            system_status,
            forecasting_performance,
            energy_savings: energy_analysis.clone(),
            sleep_windows: sleep_metrics.clone(),
            cell_count,
            active_forecasts,
            alerts_count,
            uptime_hours,
        })
    }
    
    // Helper functions for metric calculations
    
    fn calculate_mape(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        let mut sum_percentage_error = 0.0;
        let mut count = 0;
        
        for (a, p) in actual.iter().zip(predicted.iter()) {
            if *a != 0.0 {
                sum_percentage_error += ((a - p).abs() / a.abs()) * 100.0;
                count += 1;
            }
        }
        
        if count > 0 {
            Ok(sum_percentage_error / count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_rmse(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        let sum_squared_errors: f64 = actual.iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).powi(2))
            .sum();
        
        Ok((sum_squared_errors / actual.len() as f64).sqrt())
    }
    
    fn calculate_mae(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        let sum_absolute_errors: f64 = actual.iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).abs())
            .sum();
        
        Ok(sum_absolute_errors / actual.len() as f64)
    }
    
    fn calculate_r2(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        let actual_mean = actual.mean();
        
        let ss_tot: f64 = actual.iter()
            .map(|a| (a - actual_mean).powi(2))
            .sum();
        
        let ss_res: f64 = actual.iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).powi(2))
            .sum();
        
        if ss_tot > 0.0 {
            Ok(1.0 - (ss_res / ss_tot))
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_accuracy(actual: &[f64], predicted: &[f64], tolerance: f64) -> Result<f64> {
        let correct_predictions = actual.iter()
            .zip(predicted.iter())
            .filter(|(a, p)| (*a - *p).abs() <= tolerance)
            .count();
        
        Ok((correct_predictions as f64 / actual.len() as f64) * 100.0)
    }
    
    fn calculate_directional_accuracy(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        if actual.len() < 2 {
            return Ok(0.0);
        }
        
        let mut correct_directions = 0;
        let mut total_directions = 0;
        
        for i in 1..actual.len() {
            let actual_direction = actual[i] - actual[i-1];
            let predicted_direction = predicted[i] - predicted[i-1];
            
            if (actual_direction >= 0.0 && predicted_direction >= 0.0) ||
               (actual_direction < 0.0 && predicted_direction < 0.0) {
                correct_directions += 1;
            }
            total_directions += 1;
        }
        
        if total_directions > 0 {
            Ok((correct_directions as f64 / total_directions as f64) * 100.0)
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_detection_metrics(
        actual: &[f64],
        predicted: &[f64],
        threshold: f64,
    ) -> Result<DetectionMetrics> {
        let mut tp = 0; // True positives
        let mut tn = 0; // True negatives
        let mut fp = 0; // False positives
        let mut fn_count = 0; // False negatives
        
        for (a, p) in actual.iter().zip(predicted.iter()) {
            let actual_low = *a < threshold;
            let predicted_low = *p < threshold;
            
            match (actual_low, predicted_low) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
            }
        }
        
        let total = tp + tn + fp + fn_count;
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        let accuracy = if total > 0 { (tp + tn) as f64 / total as f64 } else { 0.0 };
        let specificity = if tn + fp > 0 { tn as f64 / (tn + fp) as f64 } else { 0.0 };
        let sensitivity = recall; // Same as recall
        
        Ok(DetectionMetrics {
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives: fn_count,
            precision,
            recall,
            f1_score,
            accuracy,
            specificity,
            sensitivity,
        })
    }
    
    fn calculate_prediction_interval_coverage(
        actual: &[f64],
        predicted: &[f64],
    ) -> Result<f64> {
        // Simplified prediction interval coverage
        // In a real implementation, this would use confidence intervals
        let tolerance = 10.0; // 10% tolerance
        
        let within_interval = actual.iter()
            .zip(predicted.iter())
            .filter(|(a, p)| (*a - *p).abs() <= tolerance)
            .count();
        
        Ok((within_interval as f64 / actual.len() as f64) * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    #[test]
    fn test_mape_calculation() {
        let actual = vec![10.0, 20.0, 30.0, 40.0];
        let predicted = vec![12.0, 18.0, 32.0, 38.0];
        
        let mape = MetricsCalculator::calculate_mape(&actual, &predicted).unwrap();
        assert!(mape > 0.0 && mape < 20.0); // Should be reasonable MAPE
    }
    
    #[test]
    fn test_rmse_calculation() {
        let actual = vec![10.0, 20.0, 30.0, 40.0];
        let predicted = vec![12.0, 18.0, 32.0, 38.0];
        
        let rmse = MetricsCalculator::calculate_rmse(&actual, &predicted).unwrap();
        assert!(rmse > 0.0 && rmse < 10.0); // Should be reasonable RMSE
    }
    
    #[test]
    fn test_detection_metrics() {
        let actual = vec![5.0, 15.0, 25.0, 35.0]; // 5.0 and 15.0 are below threshold 20.0
        let predicted = vec![8.0, 12.0, 28.0, 32.0]; // 8.0 and 12.0 are below threshold 20.0
        
        let metrics = MetricsCalculator::calculate_detection_metrics(&actual, &predicted, 20.0).unwrap();
        
        assert_eq!(metrics.true_positives, 2); // Both correctly identified as low traffic
        assert_eq!(metrics.true_negatives, 2); // Both correctly identified as high traffic
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_eq!(metrics.accuracy, 1.0); // Perfect accuracy in this case
    }
    
    #[test]
    fn test_energy_savings_analysis() {
        let sleep_windows = vec![
            SleepWindow {
                cell_id: "cell1".to_string(),
                start_time: Utc::now(),
                end_time: Utc::now() + Duration::hours(1),
                duration_minutes: 60,
                confidence_score: 0.9,
                predicted_utilization: 5.0,
                energy_savings_kwh: 2.5,
                risk_score: 0.1,
            },
            SleepWindow {
                cell_id: "cell2".to_string(),
                start_time: Utc::now(),
                end_time: Utc::now() + Duration::hours(2),
                duration_minutes: 120,
                confidence_score: 0.8,
                predicted_utilization: 8.0,
                energy_savings_kwh: 4.0,
                risk_score: 0.2,
            },
        ];
        
        let analysis = MetricsCalculator::analyze_energy_savings(&sleep_windows, 0.12, 0.5).unwrap();
        
        assert_eq!(analysis.total_energy_saved_kwh, 6.5);
        assert_eq!(analysis.average_savings_per_window, 3.25);
        assert!(analysis.co2_reduction_kg > 0.0);
        assert!(analysis.cost_savings_usd > 0.0);
    }
    
    #[test]
    fn test_forecast_evaluation() {
        let actual = vec![
            PrbUtilization::new("cell1".to_string(), 100, 10, 50.0, 5, 0.9),
            PrbUtilization::new("cell1".to_string(), 100, 20, 100.0, 10, 0.9),
            PrbUtilization::new("cell1".to_string(), 100, 30, 150.0, 15, 0.9),
        ];
        
        let predicted = vec![
            PrbUtilization::new("cell1".to_string(), 100, 12, 60.0, 6, 0.9),
            PrbUtilization::new("cell1".to_string(), 100, 18, 90.0, 9, 0.9),
            PrbUtilization::new("cell1".to_string(), 100, 32, 160.0, 16, 0.9),
        ];
        
        let evaluation = MetricsCalculator::evaluate_forecast(&actual, &predicted, 25.0).unwrap();
        
        assert!(evaluation.mape >= 0.0);
        assert!(evaluation.rmse >= 0.0);
        assert!(evaluation.mae >= 0.0);
        assert!(evaluation.accuracy >= 0.0);
        assert!(evaluation.low_traffic_detection_metrics.accuracy >= 0.0);
    }
}