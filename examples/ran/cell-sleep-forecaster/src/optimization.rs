//! Sleep window optimization module

use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use argmin::core::{CostFunction, Executor, State, ArgminError};
use argmin::solver::neldermead::NelderMead;

use crate::{PrbUtilization, SleepWindow, ForecastingError, config::ForecastingConfig};

/// Energy consumption model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyModel {
    pub base_power_watts: f64,
    pub active_power_per_prb_watts: f64,
    pub sleep_power_watts: f64,
    pub wakeup_energy_joules: f64,
    pub sleep_transition_time_seconds: f64,
}

impl Default for EnergyModel {
    fn default() -> Self {
        Self {
            base_power_watts: 500.0,
            active_power_per_prb_watts: 2.0,
            sleep_power_watts: 50.0,
            wakeup_energy_joules: 1000.0,
            sleep_transition_time_seconds: 30.0,
        }
    }
}

/// Sleep window optimization engine
pub struct SleepOptimizer {
    config: Arc<ForecastingConfig>,
    energy_model: EnergyModel,
    cell_parameters: HashMap<String, CellParameters>,
}

/// Cell-specific parameters for optimization
#[derive(Debug, Clone)]
struct CellParameters {
    coverage_area_km2: f64,
    neighboring_cells: Vec<String>,
    priority_level: u32,
    max_sleep_duration_minutes: u32,
    min_wakeup_threshold: f64,
}

impl Default for CellParameters {
    fn default() -> Self {
        Self {
            coverage_area_km2: 10.0,
            neighboring_cells: Vec::new(),
            priority_level: 1,
            max_sleep_duration_minutes: 120,
            min_wakeup_threshold: 25.0,
        }
    }
}

impl SleepOptimizer {
    pub fn new(config: Arc<ForecastingConfig>) -> Self {
        Self {
            config,
            energy_model: EnergyModel::default(),
            cell_parameters: HashMap::new(),
        }
    }
    
    /// Identify optimal sleep windows from forecast data
    pub async fn identify_sleep_windows(
        &self,
        cell_id: &str,
        forecast: &[PrbUtilization],
    ) -> Result<Vec<SleepWindow>> {
        log::info!("Identifying sleep windows for cell {}", cell_id);
        
        if forecast.is_empty() {
            return Ok(Vec::new());
        }
        
        // Detect low-traffic periods
        let low_traffic_periods = self.detect_low_traffic_periods(forecast).await?;
        
        // Optimize sleep windows
        let mut sleep_windows = Vec::new();
        
        for period in low_traffic_periods {
            if let Some(optimized_window) = self.optimize_sleep_window(cell_id, &period, forecast).await? {
                sleep_windows.push(optimized_window);
            }
        }
        
        // Post-process windows to avoid conflicts
        let final_windows = self.resolve_window_conflicts(sleep_windows).await?;
        
        log::info!("Found {} optimized sleep windows for cell {}", final_windows.len(), cell_id);
        Ok(final_windows)
    }
    
    /// Calculate energy savings for a sleep window
    pub async fn calculate_energy_savings(
        &self,
        cell_id: &str,
        sleep_window: &SleepWindow,
        forecast: &[PrbUtilization],
    ) -> Result<f64> {
        // Get forecast data for the sleep window period
        let window_forecast = self.get_forecast_for_window(sleep_window, forecast)?;
        
        // Calculate energy consumption without sleep
        let active_energy = self.calculate_active_energy_consumption(&window_forecast)?;
        
        // Calculate energy consumption with sleep
        let sleep_energy = self.calculate_sleep_energy_consumption(sleep_window)?;
        
        // Energy savings = active energy - sleep energy
        let savings = active_energy - sleep_energy;
        
        log::debug!("Energy savings for cell {} window: {:.2} kWh", cell_id, savings);
        Ok(savings.max(0.0))
    }
    
    /// Assess risk of implementing a sleep window
    pub async fn assess_sleep_risk(
        &self,
        cell_id: &str,
        sleep_window: &SleepWindow,
        forecast: &[PrbUtilization],
    ) -> Result<f64> {
        let mut risk_score = 0.0;
        
        // Risk factor 1: Forecast uncertainty
        let forecast_uncertainty = self.calculate_forecast_uncertainty(forecast)?;
        risk_score += forecast_uncertainty * 0.3;
        
        // Risk factor 2: Neighboring cell capacity
        let neighbor_capacity_risk = self.assess_neighbor_capacity_risk(cell_id, sleep_window).await?;
        risk_score += neighbor_capacity_risk * 0.3;
        
        // Risk factor 3: Traffic pattern volatility
        let volatility_risk = self.calculate_traffic_volatility_risk(forecast)?;
        risk_score += volatility_risk * 0.2;
        
        // Risk factor 4: Sleep window duration risk
        let duration_risk = self.calculate_duration_risk(sleep_window)?;
        risk_score += duration_risk * 0.2;
        
        // Normalize risk score to 0-1 range
        let normalized_risk = risk_score.min(1.0).max(0.0);
        
        log::debug!("Risk score for cell {} sleep window: {:.3}", cell_id, normalized_risk);
        Ok(normalized_risk)
    }
    
    async fn detect_low_traffic_periods(&self, forecast: &[PrbUtilization]) -> Result<Vec<TrafficPeriod>> {
        let mut periods = Vec::new();
        let mut current_period_start = None;
        
        for (i, data_point) in forecast.iter().enumerate() {
            let is_low_traffic = data_point.utilization_percentage < self.config.low_traffic_threshold;
            
            match (current_period_start, is_low_traffic) {
                (None, true) => {
                    current_period_start = Some(i);
                }
                (Some(start), false) => {
                    if i - start >= self.config.min_sleep_duration_minutes as usize / 10 {
                        periods.push(TrafficPeriod {
                            start_index: start,
                            end_index: i - 1,
                            avg_utilization: self.calculate_avg_utilization(&forecast[start..i]),
                        });
                    }
                    current_period_start = None;
                }
                _ => {}
            }
        }
        
        // Handle period that extends to the end
        if let Some(start) = current_period_start {
            if forecast.len() - start >= self.config.min_sleep_duration_minutes as usize / 10 {
                periods.push(TrafficPeriod {
                    start_index: start,
                    end_index: forecast.len() - 1,
                    avg_utilization: self.calculate_avg_utilization(&forecast[start..]),
                });
            }
        }
        
        Ok(periods)
    }
    
    async fn optimize_sleep_window(
        &self,
        cell_id: &str,
        period: &TrafficPeriod,
        forecast: &[PrbUtilization],
    ) -> Result<Option<SleepWindow>> {
        if period.start_index >= forecast.len() || period.end_index >= forecast.len() {
            return Ok(None);
        }
        
        let start_time = forecast[period.start_index].timestamp;
        let end_time = forecast[period.end_index].timestamp;
        let duration = end_time.signed_duration_since(start_time);
        
        // Create initial sleep window
        let mut sleep_window = SleepWindow {
            cell_id: cell_id.to_string(),
            start_time,
            end_time,
            duration_minutes: duration.num_minutes() as u32,
            confidence_score: 0.0,
            predicted_utilization: period.avg_utilization,
            energy_savings_kwh: 0.0,
            risk_score: 0.0,
        };
        
        // Calculate energy savings
        sleep_window.energy_savings_kwh = self.calculate_energy_savings(cell_id, &sleep_window, forecast).await?;
        
        // Assess risk
        sleep_window.risk_score = self.assess_sleep_risk(cell_id, &sleep_window, forecast).await?;
        
        // Calculate confidence score
        sleep_window.confidence_score = self.calculate_confidence_score(&sleep_window, forecast)?;
        
        // Optimize window boundaries using numerical optimization
        if let Some(optimized) = self.optimize_window_boundaries(sleep_window, forecast).await? {
            Ok(Some(optimized))
        } else {
            Ok(None)
        }
    }
    
    async fn optimize_window_boundaries(
        &self,
        initial_window: SleepWindow,
        forecast: &[PrbUtilization],
    ) -> Result<Option<SleepWindow>> {
        // Define optimization problem
        let problem = SleepWindowOptimizationProblem {
            forecast: forecast.to_vec(),
            initial_window: initial_window.clone(),
            energy_model: self.energy_model.clone(),
            config: self.config.clone(),
        };
        
        // Initial parameters: [start_offset_minutes, duration_minutes]
        let initial_params = vec![0.0, initial_window.duration_minutes as f64];
        
        // Set up Nelder-Mead optimizer
        let solver = NelderMead::new(vec![
            vec![-60.0, initial_window.duration_minutes as f64 * 0.5], // Simplex vertices
            vec![60.0, initial_window.duration_minutes as f64 * 1.5],
        ]);
        
        // Run optimization
        let executor = Executor::new(problem, solver);
        let result = executor.configure(|state| {
            state
                .param(initial_params)
                .max_iters(100)
                .target_cost(0.0)
        });
        
        match result.run() {
            Ok(res) => {
                let optimized_params = res.state.get_best_param().unwrap();
                let optimized_window = self.create_optimized_window(
                    &initial_window,
                    optimized_params[0],
                    optimized_params[1],
                    forecast,
                ).await?;
                
                // Only return if optimization improved the window
                if optimized_window.energy_savings_kwh > initial_window.energy_savings_kwh {
                    Ok(Some(optimized_window))
                } else {
                    Ok(Some(initial_window))
                }
            }
            Err(_) => Ok(Some(initial_window)), // Return original if optimization fails
        }
    }
    
    async fn create_optimized_window(
        &self,
        original: &SleepWindow,
        start_offset_minutes: f64,
        duration_minutes: f64,
        forecast: &[PrbUtilization],
    ) -> Result<SleepWindow> {
        let new_start = original.start_time + Duration::minutes(start_offset_minutes as i64);
        let new_end = new_start + Duration::minutes(duration_minutes as i64);
        
        let mut optimized_window = SleepWindow {
            cell_id: original.cell_id.clone(),
            start_time: new_start,
            end_time: new_end,
            duration_minutes: duration_minutes as u32,
            confidence_score: 0.0,
            predicted_utilization: original.predicted_utilization,
            energy_savings_kwh: 0.0,
            risk_score: 0.0,
        };
        
        // Recalculate metrics for optimized window
        optimized_window.energy_savings_kwh = self.calculate_energy_savings(&original.cell_id, &optimized_window, forecast).await?;
        optimized_window.risk_score = self.assess_sleep_risk(&original.cell_id, &optimized_window, forecast).await?;
        optimized_window.confidence_score = self.calculate_confidence_score(&optimized_window, forecast)?;
        
        Ok(optimized_window)
    }
    
    async fn resolve_window_conflicts(&self, windows: Vec<SleepWindow>) -> Result<Vec<SleepWindow>> {
        let mut resolved = Vec::new();
        let mut sorted_windows = windows;
        
        // Sort by start time
        sorted_windows.sort_by(|a, b| a.start_time.cmp(&b.start_time));
        
        for window in sorted_windows {
            let mut conflicts = false;
            
            // Check for conflicts with already resolved windows
            for existing in &resolved {
                if self.windows_overlap(&window, existing) {
                    conflicts = true;
                    break;
                }
            }
            
            if !conflicts {
                resolved.push(window);
            }
        }
        
        Ok(resolved)
    }
    
    fn windows_overlap(&self, window1: &SleepWindow, window2: &SleepWindow) -> bool {
        !(window1.end_time <= window2.start_time || window2.end_time <= window1.start_time)
    }
    
    fn calculate_avg_utilization(&self, data: &[PrbUtilization]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let sum: f64 = data.iter().map(|d| d.utilization_percentage).sum();
        sum / data.len() as f64
    }
    
    fn get_forecast_for_window(&self, window: &SleepWindow, forecast: &[PrbUtilization]) -> Result<Vec<PrbUtilization>> {
        let window_data: Vec<PrbUtilization> = forecast
            .iter()
            .filter(|d| d.timestamp >= window.start_time && d.timestamp <= window.end_time)
            .cloned()
            .collect();
        
        Ok(window_data)
    }
    
    fn calculate_active_energy_consumption(&self, forecast: &[PrbUtilization]) -> Result<f64> {
        let mut total_energy = 0.0;
        
        for data_point in forecast {
            let interval_hours = 10.0 / 60.0; // 10-minute intervals
            let base_energy = self.energy_model.base_power_watts * interval_hours / 1000.0; // kWh
            let prb_energy = self.energy_model.active_power_per_prb_watts * 
                data_point.prb_used as f64 * interval_hours / 1000.0; // kWh
            
            total_energy += base_energy + prb_energy;
        }
        
        Ok(total_energy)
    }
    
    fn calculate_sleep_energy_consumption(&self, window: &SleepWindow) -> Result<f64> {
        let sleep_hours = window.duration_minutes as f64 / 60.0;
        let sleep_energy = self.energy_model.sleep_power_watts * sleep_hours / 1000.0; // kWh
        let wakeup_energy = self.energy_model.wakeup_energy_joules / 3_600_000.0; // Convert J to kWh
        
        Ok(sleep_energy + wakeup_energy)
    }
    
    fn calculate_forecast_uncertainty(&self, forecast: &[PrbUtilization]) -> Result<f64> {
        if forecast.len() < 2 {
            return Ok(0.5); // Default uncertainty
        }
        
        // Calculate coefficient of variation as uncertainty measure
        let values: Vec<f64> = forecast.iter().map(|d| d.utilization_percentage).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        if mean == 0.0 {
            return Ok(0.0);
        }
        
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let coefficient_of_variation = variance.sqrt() / mean;
        Ok(coefficient_of_variation.min(1.0))
    }
    
    async fn assess_neighbor_capacity_risk(&self, cell_id: &str, _window: &SleepWindow) -> Result<f64> {
        // In a real implementation, this would query neighboring cells' capacity
        // For now, return a moderate risk score
        Ok(0.3)
    }
    
    fn calculate_traffic_volatility_risk(&self, forecast: &[PrbUtilization]) -> Result<f64> {
        if forecast.len() < 3 {
            return Ok(0.5);
        }
        
        // Calculate rate of change in utilization
        let mut changes = Vec::new();
        for i in 1..forecast.len() {
            let change = (forecast[i].utilization_percentage - forecast[i-1].utilization_percentage).abs();
            changes.push(change);
        }
        
        let avg_change = changes.iter().sum::<f64>() / changes.len() as f64;
        let volatility_risk = (avg_change / 10.0).min(1.0); // Normalize to 0-1
        
        Ok(volatility_risk)
    }
    
    fn calculate_duration_risk(&self, window: &SleepWindow) -> Result<f64> {
        let max_safe_duration = self.config.max_sleep_duration_minutes as f64;
        let duration_risk = (window.duration_minutes as f64 / max_safe_duration).min(1.0);
        
        Ok(duration_risk)
    }
    
    fn calculate_confidence_score(&self, window: &SleepWindow, forecast: &[PrbUtilization]) -> Result<f64> {
        // Combine multiple factors into confidence score
        let low_utilization_factor = (self.config.low_traffic_threshold - window.predicted_utilization) / 
            self.config.low_traffic_threshold;
        
        let duration_factor = if window.duration_minutes >= self.config.min_sleep_duration_minutes {
            1.0
        } else {
            window.duration_minutes as f64 / self.config.min_sleep_duration_minutes as f64
        };
        
        let risk_factor = 1.0 - window.risk_score;
        
        let confidence = (low_utilization_factor * 0.4 + duration_factor * 0.3 + risk_factor * 0.3)
            .max(0.0).min(1.0);
        
        Ok(confidence)
    }
}

#[derive(Debug, Clone)]
struct TrafficPeriod {
    start_index: usize,
    end_index: usize,
    avg_utilization: f64,
}

/// Optimization problem for sleep window boundaries
struct SleepWindowOptimizationProblem {
    forecast: Vec<PrbUtilization>,
    initial_window: SleepWindow,
    energy_model: EnergyModel,
    config: Arc<ForecastingConfig>,
}

impl CostFunction for SleepWindowOptimizationProblem {
    type Param = Vec<f64>;
    type Output = f64;
    
    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        if params.len() != 2 {
            return Err(anyhow::anyhow!("Expected 2 parameters: start_offset and duration"));
        }
        
        let start_offset_minutes = params[0];
        let duration_minutes = params[1];
        
        // Validate parameters
        if duration_minutes < self.config.min_sleep_duration_minutes as f64 ||
           duration_minutes > self.config.max_sleep_duration_minutes as f64 {
            return Ok(f64::MAX); // Invalid duration
        }
        
        // Calculate new window boundaries
        let new_start = self.initial_window.start_time + Duration::minutes(start_offset_minutes as i64);
        let new_end = new_start + Duration::minutes(duration_minutes as i64);
        
        // Create temporary window for evaluation
        let temp_window = SleepWindow {
            cell_id: self.initial_window.cell_id.clone(),
            start_time: new_start,
            end_time: new_end,
            duration_minutes: duration_minutes as u32,
            confidence_score: 0.0,
            predicted_utilization: self.initial_window.predicted_utilization,
            energy_savings_kwh: 0.0,
            risk_score: 0.0,
        };
        
        // Calculate cost (negative energy savings + risk penalty)
        let window_forecast = self.get_forecast_for_window(&temp_window);
        let energy_savings = self.calculate_energy_savings(&window_forecast);
        let risk_penalty = self.calculate_risk_penalty(&temp_window);
        
        // Cost = negative savings + risk penalty (minimize cost = maximize savings - risk)
        let cost = -energy_savings + risk_penalty;
        
        Ok(cost)
    }
}

impl SleepWindowOptimizationProblem {
    fn get_forecast_for_window(&self, window: &SleepWindow) -> Vec<PrbUtilization> {
        self.forecast
            .iter()
            .filter(|d| d.timestamp >= window.start_time && d.timestamp <= window.end_time)
            .cloned()
            .collect()
    }
    
    fn calculate_energy_savings(&self, forecast: &[PrbUtilization]) -> f64 {
        let mut total_energy = 0.0;
        
        for data_point in forecast {
            let interval_hours = 10.0 / 60.0; // 10-minute intervals
            let base_energy = self.energy_model.base_power_watts * interval_hours / 1000.0; // kWh
            let prb_energy = self.energy_model.active_power_per_prb_watts * 
                data_point.prb_used as f64 * interval_hours / 1000.0; // kWh
            
            total_energy += base_energy + prb_energy;
        }
        
        // Subtract sleep energy consumption
        let sleep_hours = forecast.len() as f64 * 10.0 / 60.0; // Total sleep time in hours
        let sleep_energy = self.energy_model.sleep_power_watts * sleep_hours / 1000.0;
        let wakeup_energy = self.energy_model.wakeup_energy_joules / 3_600_000.0;
        
        total_energy - sleep_energy - wakeup_energy
    }
    
    fn calculate_risk_penalty(&self, window: &SleepWindow) -> f64 {
        // Simple risk penalty based on window characteristics
        let duration_risk = if window.duration_minutes > 60 {
            (window.duration_minutes as f64 - 60.0) / 60.0
        } else {
            0.0
        };
        
        duration_risk * 10.0 // Scale penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ForecastingConfig;
    
    #[tokio::test]
    async fn test_sleep_optimizer_creation() {
        let config = Arc::new(ForecastingConfig::default());
        let optimizer = SleepOptimizer::new(config);
        
        assert_eq!(optimizer.energy_model.base_power_watts, 500.0);
        assert_eq!(optimizer.energy_model.sleep_power_watts, 50.0);
    }
    
    #[tokio::test]
    async fn test_energy_savings_calculation() {
        let config = Arc::new(ForecastingConfig::default());
        let optimizer = SleepOptimizer::new(config);
        
        let window = SleepWindow {
            cell_id: "test_cell".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::hours(1),
            duration_minutes: 60,
            confidence_score: 0.8,
            predicted_utilization: 10.0,
            energy_savings_kwh: 0.0,
            risk_score: 0.2,
        };
        
        let forecast = vec![
            PrbUtilization::new("test_cell".to_string(), 100, 10, 50.0, 5, 0.9),
            PrbUtilization::new("test_cell".to_string(), 100, 15, 60.0, 7, 0.9),
        ];
        
        let savings = optimizer.calculate_energy_savings("test_cell", &window, &forecast).await.unwrap();
        assert!(savings > 0.0);
    }
    
    #[tokio::test]
    async fn test_risk_assessment() {
        let config = Arc::new(ForecastingConfig::default());
        let optimizer = SleepOptimizer::new(config);
        
        let window = SleepWindow {
            cell_id: "test_cell".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::hours(1),
            duration_minutes: 60,
            confidence_score: 0.8,
            predicted_utilization: 10.0,
            energy_savings_kwh: 2.5,
            risk_score: 0.0,
        };
        
        let forecast = vec![
            PrbUtilization::new("test_cell".to_string(), 100, 10, 50.0, 5, 0.9),
            PrbUtilization::new("test_cell".to_string(), 100, 15, 60.0, 7, 0.9),
        ];
        
        let risk = optimizer.assess_sleep_risk("test_cell", &window, &forecast).await.unwrap();
        assert!(risk >= 0.0 && risk <= 1.0);
    }
    
    #[test]
    fn test_window_overlap_detection() {
        let config = Arc::new(ForecastingConfig::default());
        let optimizer = SleepOptimizer::new(config);
        
        let window1 = SleepWindow {
            cell_id: "test_cell".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::hours(1),
            duration_minutes: 60,
            confidence_score: 0.8,
            predicted_utilization: 10.0,
            energy_savings_kwh: 2.5,
            risk_score: 0.2,
        };
        
        let window2 = SleepWindow {
            cell_id: "test_cell".to_string(),
            start_time: Utc::now() + Duration::minutes(30),
            end_time: Utc::now() + Duration::hours(1) + Duration::minutes(30),
            duration_minutes: 60,
            confidence_score: 0.8,
            predicted_utilization: 10.0,
            energy_savings_kwh: 2.5,
            risk_score: 0.2,
        };
        
        assert!(optimizer.windows_overlap(&window1, &window2));
        
        let window3 = SleepWindow {
            cell_id: "test_cell".to_string(),
            start_time: Utc::now() + Duration::hours(2),
            end_time: Utc::now() + Duration::hours(3),
            duration_minutes: 60,
            confidence_score: 0.8,
            predicted_utilization: 10.0,
            energy_savings_kwh: 2.5,
            risk_score: 0.2,
        };
        
        assert!(!optimizer.windows_overlap(&window1, &window3));
    }
}