use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Utility functions for VoLTE QoS forecasting

/// Calculate statistical metrics for time series data
pub fn calculate_statistics(values: &[f64]) -> Statistics {
    if values.is_empty() {
        return Statistics::default();
    }
    
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    let std_dev = variance.sqrt();
    
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median = if sorted_values.len() % 2 == 0 {
        let mid = sorted_values.len() / 2;
        (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2]
    };
    
    let min = sorted_values[0];
    let max = sorted_values[sorted_values.len() - 1];
    
    // Calculate percentiles
    let p25_idx = (0.25 * (sorted_values.len() - 1) as f64) as usize;
    let p75_idx = (0.75 * (sorted_values.len() - 1) as f64) as usize;
    let p95_idx = (0.95 * (sorted_values.len() - 1) as f64) as usize;
    
    Statistics {
        count: values.len(),
        mean,
        median,
        std_dev,
        variance,
        min,
        max,
        p25: sorted_values[p25_idx],
        p75: sorted_values[p75_idx],
        p95: sorted_values[p95_idx],
    }
}

/// Calculate correlation coefficient between two time series
pub fn calculate_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err(Error::InvalidInput("Series must be same length and non-empty".to_string()));
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

/// Detect anomalies in time series using statistical methods
pub fn detect_anomalies(values: &[f64], threshold_std: f64) -> Vec<AnomalyPoint> {
    let stats = calculate_statistics(values);
    let mut anomalies = Vec::new();
    
    for (i, &value) in values.iter().enumerate() {
        let z_score = (value - stats.mean) / stats.std_dev;
        if z_score.abs() > threshold_std {
            anomalies.push(AnomalyPoint {
                index: i,
                value,
                z_score,
                severity: if z_score.abs() > threshold_std * 2.0 {
                    AnomalySeverity::High
                } else {
                    AnomalySeverity::Medium
                },
            });
        }
    }
    
    anomalies
}

/// Calculate moving average
pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
    if window_size == 0 || values.len() < window_size {
        return Vec::new();
    }
    
    let mut result = Vec::new();
    
    for i in window_size - 1..values.len() {
        let sum: f64 = values[i - window_size + 1..=i].iter().sum();
        result.push(sum / window_size as f64);
    }
    
    result
}

/// Calculate exponential moving average
pub fn exponential_moving_average(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(values.len());
    result.push(values[0]);
    
    for i in 1..values.len() {
        let ema = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema);
    }
    
    result
}

/// Validate VoLTE metrics
pub fn validate_volte_metrics(metrics: &crate::types::VoLteMetrics) -> Result<()> {
    // Check cell_id
    if metrics.cell_id.is_empty() {
        return Err(Error::Validation("Cell ID cannot be empty".to_string()));
    }
    
    // Check timestamp is not too old or in future
    let now = Utc::now();
    let max_age = chrono::Duration::hours(24);
    let max_future = chrono::Duration::minutes(5);
    
    if now - metrics.timestamp > max_age {
        return Err(Error::Validation("Metrics timestamp is too old".to_string()));
    }
    
    if metrics.timestamp - now > max_future {
        return Err(Error::Validation("Metrics timestamp is in the future".to_string()));
    }
    
    // Check value ranges
    if !(0.0..=1.0).contains(&metrics.prb_utilization_dl) {
        return Err(Error::Validation("PRB utilization must be between 0 and 1".to_string()));
    }
    
    if metrics.active_volte_users > 1000 {
        return Err(Error::Validation("Active VoLTE users seems too high".to_string()));
    }
    
    if metrics.competing_gbr_traffic_mbps < 0.0 {
        return Err(Error::Validation("Competing GBR traffic cannot be negative".to_string()));
    }
    
    if metrics.current_jitter_ms < 0.0 || metrics.current_jitter_ms > 1000.0 {
        return Err(Error::Validation("Current jitter must be between 0 and 1000ms".to_string()));
    }
    
    if !(0.0..=1.0).contains(&metrics.packet_loss_rate) {
        return Err(Error::Validation("Packet loss rate must be between 0 and 1".to_string()));
    }
    
    if metrics.delay_ms < 0.0 || metrics.delay_ms > 10000.0 {
        return Err(Error::Validation("Delay must be between 0 and 10000ms".to_string()));
    }
    
    Ok(())
}

/// Generate synthetic VoLTE metrics for testing
pub fn generate_synthetic_metrics(
    cell_id: &str,
    start_time: DateTime<Utc>,
    count: usize,
    interval_seconds: i64,
) -> Vec<crate::types::VoLteMetrics> {
    let mut metrics = Vec::with_capacity(count);
    let mut rng = fastrand::Rng::new();
    
    for i in 0..count {
        let timestamp = start_time + chrono::Duration::seconds(i as i64 * interval_seconds);
        
        // Generate realistic patterns
        let time_of_day = (timestamp.timestamp() % 86400) as f64 / 86400.0;
        let daily_pattern = 0.5 + 0.3 * (2.0 * std::f64::consts::PI * time_of_day).sin();
        
        let base_utilization = 0.3 + 0.4 * daily_pattern;
        let noise = (rng.f64() - 0.5) * 0.1;
        
        let prb_utilization_dl = (base_utilization + noise).clamp(0.0, 1.0);
        let active_volte_users = ((20.0 + 30.0 * daily_pattern + rng.f64() * 10.0) as u32).min(100);
        let competing_gbr_traffic_mbps = 50.0 + 100.0 * daily_pattern + rng.f64() * 20.0;
        
        // Jitter correlates with utilization
        let base_jitter = 5.0 + 15.0 * prb_utilization_dl;
        let jitter_noise = (rng.f64() - 0.5) * 4.0;
        let current_jitter_ms = (base_jitter + jitter_noise).max(0.0);
        
        let packet_loss_rate = (0.001 + 0.01 * prb_utilization_dl + rng.f64() * 0.005).clamp(0.0, 0.1);
        let delay_ms = 20.0 + 30.0 * prb_utilization_dl + rng.f64() * 10.0;
        
        metrics.push(crate::types::VoLteMetrics {
            cell_id: cell_id.to_string(),
            timestamp,
            prb_utilization_dl,
            active_volte_users,
            competing_gbr_traffic_mbps,
            current_jitter_ms,
            packet_loss_rate,
            delay_ms,
        });
    }
    
    metrics
}

/// Convert timestamps to time series features
pub fn extract_time_features(timestamp: DateTime<Utc>) -> Vec<f64> {
    let hour = timestamp.hour() as f64 / 24.0;
    let day_of_week = timestamp.weekday().num_days_from_monday() as f64 / 7.0;
    let day_of_month = timestamp.day() as f64 / 31.0;
    let month = timestamp.month() as f64 / 12.0;
    
    // Add cyclical features
    let hour_sin = (2.0 * std::f64::consts::PI * hour).sin();
    let hour_cos = (2.0 * std::f64::consts::PI * hour).cos();
    let day_sin = (2.0 * std::f64::consts::PI * day_of_week).sin();
    let day_cos = (2.0 * std::f64::consts::PI * day_of_week).cos();
    
    vec![hour, day_of_week, day_of_month, month, hour_sin, hour_cos, day_sin, day_cos]
}

/// Performance measurement utilities
pub struct PerformanceTimer {
    start_time: std::time::Instant,
    operation: String,
}

impl PerformanceTimer {
    pub fn new(operation: String) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            operation,
        }
    }
    
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
    
    pub fn finish(self) -> u64 {
        let elapsed = self.elapsed_ms();
        tracing::debug!("Operation '{}' completed in {}ms", self.operation, elapsed);
        elapsed
    }
}

/// Data structures for utility functions

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            variance: 0.0,
            min: 0.0,
            max: 0.0,
            p25: 0.0,
            p75: 0.0,
            p95: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPoint {
    pub index: usize,
    pub value: f64,
    pub z_score: f64,
    pub severity: AnomalySeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
}

/// Configuration validation
pub fn validate_config(config: &crate::config::Config) -> Result<()> {
    config.validate().map_err(|e| Error::Configuration(
        config::ConfigError::Message(e)
    ))?;
    
    // Additional QoS-specific validations
    if config.forecasting.default_forecast_horizon_minutes == 0 {
        return Err(Error::Validation("Default forecast horizon must be > 0".to_string()));
    }
    
    if config.forecasting.min_historical_data_points < 10 {
        return Err(Error::Validation("Minimum historical data points must be >= 10".to_string()));
    }
    
    if config.alerts.jitter_threshold_ms <= 0.0 {
        return Err(Error::Validation("Jitter threshold must be positive".to_string()));
    }
    
    Ok(())
}

/// Metrics aggregation utilities
pub fn aggregate_metrics_by_hour(metrics: &[crate::types::VoLteMetrics]) -> HashMap<String, Vec<Statistics>> {
    let mut hourly_data: HashMap<String, Vec<crate::types::VoLteMetrics>> = HashMap::new();
    
    // Group metrics by hour
    for metric in metrics {
        let hour_key = format!("{}-{:02}", metric.timestamp.date_naive(), metric.timestamp.hour());
        hourly_data.entry(hour_key).or_default().push(metric.clone());
    }
    
    // Calculate statistics for each hour
    let mut result = HashMap::new();
    for (hour, hour_metrics) in hourly_data {
        let jitter_values: Vec<f64> = hour_metrics.iter().map(|m| m.current_jitter_ms).collect();
        let utilization_values: Vec<f64> = hour_metrics.iter().map(|m| m.prb_utilization_dl).collect();
        let delay_values: Vec<f64> = hour_metrics.iter().map(|m| m.delay_ms).collect();
        
        result.insert(hour, vec![
            calculate_statistics(&jitter_values),
            calculate_statistics(&utilization_values),
            calculate_statistics(&delay_values),
        ]);
    }
    
    result
}