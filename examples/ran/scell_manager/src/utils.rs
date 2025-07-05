//! Utility functions for the SCell Manager

use anyhow::Result;
use chrono::{DateTime, Utc};
use log::{debug, info};
use std::collections::HashMap;
use std::path::Path;
use tokio::time::{Duration, Instant};

/// Time-based utilities
pub mod time {
    use super::*;
    
    /// Get current UTC timestamp
    pub fn now_utc() -> DateTime<Utc> {
        Utc::now()
    }
    
    /// Get timestamp from seconds since epoch
    pub fn from_timestamp(seconds: i64) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp(seconds, 0)
    }
    
    /// Convert duration to milliseconds
    pub fn duration_to_ms(duration: Duration) -> f64 {
        duration.as_millis() as f64
    }
    
    /// Parse duration from string (e.g., "30s", "5m", "1h")
    pub fn parse_duration(input: &str) -> Result<Duration> {
        let input = input.trim();
        if input.is_empty() {
            return Err(anyhow::anyhow!("Empty duration string"));
        }
        
        let (number_part, unit_part) = if let Some(pos) = input.chars().position(|c| c.is_alphabetic()) {
            (&input[..pos], &input[pos..])
        } else {
            (input, "s") // Default to seconds
        };
        
        let value: f64 = number_part.parse()
            .map_err(|_| anyhow::anyhow!("Invalid number in duration: {}", number_part))?;
        
        let duration = match unit_part.to_lowercase().as_str() {
            "s" | "sec" | "seconds" => Duration::from_secs_f64(value),
            "m" | "min" | "minutes" => Duration::from_secs_f64(value * 60.0),
            "h" | "hour" | "hours" => Duration::from_secs_f64(value * 3600.0),
            "d" | "day" | "days" => Duration::from_secs_f64(value * 86400.0),
            "ms" | "millis" | "milliseconds" => Duration::from_millis(value as u64),
            _ => return Err(anyhow::anyhow!("Unknown time unit: {}", unit_part)),
        };
        
        Ok(duration)
    }
    
    /// Format duration as human-readable string
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        
        if total_seconds < 60 {
            format!("{}s", total_seconds)
        } else if total_seconds < 3600 {
            let minutes = total_seconds / 60;
            let seconds = total_seconds % 60;
            if seconds == 0 {
                format!("{}m", minutes)
            } else {
                format!("{}m{}s", minutes, seconds)
            }
        } else {
            let hours = total_seconds / 3600;
            let minutes = (total_seconds % 3600) / 60;
            if minutes == 0 {
                format!("{}h", hours)
            } else {
                format!("{}h{}m", hours, minutes)
            }
        }
    }
}

/// Performance measurement utilities
pub mod perf {
    use super::*;
    
    /// Simple performance timer
    #[derive(Debug)]
    pub struct Timer {
        start: Instant,
        name: String,
    }
    
    impl Timer {
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                start: Instant::now(),
                name: name.into(),
            }
        }
        
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        pub fn elapsed_ms(&self) -> f64 {
            time::duration_to_ms(self.elapsed())
        }
        
        pub fn log_elapsed(&self) {
            debug!("{} took {:.2}ms", self.name, self.elapsed_ms());
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            self.log_elapsed();
        }
    }
    
    /// Performance counter for tracking operations
    #[derive(Debug, Default, Clone)]
    pub struct PerfCounter {
        pub count: u64,
        pub total_duration: Duration,
        pub min_duration: Option<Duration>,
        pub max_duration: Option<Duration>,
    }
    
    impl PerfCounter {
        pub fn new() -> Self {
            Self::default()
        }
        
        pub fn record(&mut self, duration: Duration) {
            self.count += 1;
            self.total_duration += duration;
            
            match self.min_duration {
                None => self.min_duration = Some(duration),
                Some(min) if duration < min => self.min_duration = Some(duration),
                _ => {}
            }
            
            match self.max_duration {
                None => self.max_duration = Some(duration),
                Some(max) if duration > max => self.max_duration = Some(duration),
                _ => {}
            }
        }
        
        pub fn average_duration(&self) -> Option<Duration> {
            if self.count > 0 {
                Some(self.total_duration / self.count as u32)
            } else {
                None
            }
        }
        
        pub fn average_duration_ms(&self) -> f64 {
            self.average_duration()
                .map(time::duration_to_ms)
                .unwrap_or(0.0)
        }
        
        pub fn min_duration_ms(&self) -> f64 {
            self.min_duration
                .map(time::duration_to_ms)
                .unwrap_or(0.0)
        }
        
        pub fn max_duration_ms(&self) -> f64 {
            self.max_duration
                .map(time::duration_to_ms)
                .unwrap_or(0.0)
        }
    }
}

/// File system utilities
pub mod fs {
    use super::*;
    use std::fs;
    
    /// Ensure directory exists
    pub fn ensure_dir_exists(path: &Path) -> Result<()> {
        if !path.exists() {
            fs::create_dir_all(path)?;
            info!("Created directory: {:?}", path);
        }
        Ok(())
    }
    
    /// Get file size in bytes
    pub fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// Format file size as human-readable string
    pub fn format_file_size(size_bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = size_bytes as f64;
        let mut unit_idx = 0;
        
        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }
        
        if unit_idx == 0 {
            format!("{} {}", size as u64, UNITS[unit_idx])
        } else {
            format!("{:.1} {}", size, UNITS[unit_idx])
        }
    }
    
    /// Check if file is newer than specified duration
    pub fn is_file_newer_than(path: &Path, duration: Duration) -> Result<bool> {
        let metadata = fs::metadata(path)?;
        let modified = metadata.modified()?;
        let elapsed = modified.elapsed().unwrap_or(Duration::MAX);
        Ok(elapsed < duration)
    }
    
    /// List files in directory with extension filter
    pub fn list_files_with_extension(dir: &Path, extension: &str) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();
        
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext.to_string_lossy().to_lowercase() == extension.to_lowercase() {
                            files.push(path);
                        }
                    }
                }
            }
        }
        
        files.sort();
        Ok(files)
    }
}

/// Mathematical utilities
pub mod math {
    /// Calculate percentile of sorted values
    pub fn percentile(sorted_values: &[f32], percentile: f32) -> f32 {
        if sorted_values.is_empty() {
            return 0.0;
        }
        
        let index = (percentile / 100.0) * (sorted_values.len() - 1) as f32;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;
        
        if lower_index == upper_index {
            sorted_values[lower_index]
        } else {
            let weight = index - lower_index as f32;
            sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight
        }
    }
    
    /// Calculate z-score
    pub fn z_score(value: f32, mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 {
            0.0
        } else {
            (value - mean) / std_dev
        }
    }
    
    /// Normalize value to 0-1 range
    pub fn normalize(value: f32, min: f32, max: f32) -> f32 {
        if max == min {
            0.0
        } else {
            ((value - min) / (max - min)).max(0.0).min(1.0)
        }
    }
    
    /// Calculate moving average
    pub fn moving_average(values: &[f32], window_size: usize) -> Vec<f32> {
        if values.is_empty() || window_size == 0 {
            return vec![];
        }
        
        let mut result = Vec::new();
        let window_size = window_size.min(values.len());
        
        for i in 0..values.len() {
            let start = if i >= window_size - 1 { i - window_size + 1 } else { 0 };
            let end = i + 1;
            let window_sum: f32 = values[start..end].iter().sum();
            let window_len = end - start;
            result.push(window_sum / window_len as f32);
        }
        
        result
    }
    
    /// Calculate exponential moving average
    pub fn exponential_moving_average(values: &[f32], alpha: f32) -> Vec<f32> {
        if values.is_empty() {
            return vec![];
        }
        
        let mut result = Vec::with_capacity(values.len());
        let mut ema = values[0];
        result.push(ema);
        
        for &value in &values[1..] {
            ema = alpha * value + (1.0 - alpha) * ema;
            result.push(ema);
        }
        
        result
    }
}

/// String utilities
pub mod string {
    /// Truncate string to maximum length with ellipsis
    pub fn truncate_with_ellipsis(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else if max_len <= 3 {
            "...".to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }
    
    /// Convert bytes to hex string
    pub fn bytes_to_hex(bytes: &[u8]) -> String {
        bytes.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    }
    
    /// Convert snake_case to camelCase
    pub fn snake_to_camel(snake_str: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;
        
        for ch in snake_str.chars() {
            if ch == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.extend(ch.to_uppercase());
                capitalize_next = false;
            } else {
                result.push(ch);
            }
        }
        
        result
    }
}

/// Validation utilities
pub mod validation {
    use crate::types::UEMetrics;
    
    /// Validate UE metrics
    pub fn validate_ue_metrics(metrics: &UEMetrics) -> Result<()> {
        if metrics.ue_id.is_empty() {
            return Err(anyhow::anyhow!("UE ID cannot be empty"));
        }
        
        if metrics.pcell_throughput_mbps < 0.0 {
            return Err(anyhow::anyhow!("Throughput cannot be negative"));
        }
        
        if metrics.pcell_throughput_mbps > 10000.0 {
            return Err(anyhow::anyhow!("Throughput seems unrealistic: {} Mbps", metrics.pcell_throughput_mbps));
        }
        
        if metrics.pcell_cqi < 0.0 || metrics.pcell_cqi > 15.0 {
            return Err(anyhow::anyhow!("CQI must be between 0 and 15, got: {}", metrics.pcell_cqi));
        }
        
        if metrics.pcell_rsrp < -150.0 || metrics.pcell_rsrp > -30.0 {
            return Err(anyhow::anyhow!("RSRP seems unrealistic: {} dBm", metrics.pcell_rsrp));
        }
        
        if metrics.pcell_sinr < -30.0 || metrics.pcell_sinr > 50.0 {
            return Err(anyhow::anyhow!("SINR seems unrealistic: {} dB", metrics.pcell_sinr));
        }
        
        if metrics.active_bearers < 0 || metrics.active_bearers > 10 {
            return Err(anyhow::anyhow!("Active bearers should be between 0 and 10, got: {}", metrics.active_bearers));
        }
        
        if metrics.buffer_status_report_bytes < 0 {
            return Err(anyhow::anyhow!("Buffer size cannot be negative"));
        }
        
        Ok(())
    }
    
    /// Validate confidence score
    pub fn validate_confidence_score(score: f32) -> Result<()> {
        if score < 0.0 || score > 1.0 {
            return Err(anyhow::anyhow!("Confidence score must be between 0 and 1, got: {}", score));
        }
        Ok(())
    }
    
    /// Validate prediction horizon
    pub fn validate_prediction_horizon(horizon_seconds: i32) -> Result<()> {
        if horizon_seconds <= 0 {
            return Err(anyhow::anyhow!("Prediction horizon must be positive, got: {}", horizon_seconds));
        }
        
        if horizon_seconds > 3600 {
            return Err(anyhow::anyhow!("Prediction horizon too long: {} seconds (max 1 hour)", horizon_seconds));
        }
        
        Ok(())
    }
}

/// Configuration utilities
pub mod config {
    use super::*;
    use std::env;
    
    /// Get environment variable with default value
    pub fn get_env_var(name: &str, default: &str) -> String {
        env::var(name).unwrap_or_else(|_| default.to_string())
    }
    
    /// Get environment variable as integer with default
    pub fn get_env_var_int(name: &str, default: i32) -> i32 {
        env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }
    
    /// Get environment variable as float with default
    pub fn get_env_var_float(name: &str, default: f32) -> f32 {
        env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }
    
    /// Get environment variable as boolean with default
    pub fn get_env_var_bool(name: &str, default: bool) -> bool {
        env::var(name)
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                _ => None,
            })
            .unwrap_or(default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_parsing() {
        assert_eq!(time::parse_duration("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(time::parse_duration("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(time::parse_duration("1h").unwrap(), Duration::from_secs(3600));
        assert_eq!(time::parse_duration("1000ms").unwrap(), Duration::from_millis(1000));
    }
    
    #[test]
    fn test_duration_formatting() {
        assert_eq!(time::format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(time::format_duration(Duration::from_secs(90)), "1m30s");
        assert_eq!(time::format_duration(Duration::from_secs(3660)), "1h1m");
    }
    
    #[test]
    fn test_perf_counter() {
        let mut counter = perf::PerfCounter::new();
        
        counter.record(Duration::from_millis(10));
        counter.record(Duration::from_millis(20));
        counter.record(Duration::from_millis(15));
        
        assert_eq!(counter.count, 3);
        assert_eq!(counter.average_duration_ms(), 15.0);
        assert_eq!(counter.min_duration_ms(), 10.0);
        assert_eq!(counter.max_duration_ms(), 20.0);
    }
    
    #[test]
    fn test_file_size_formatting() {
        assert_eq!(fs::format_file_size(512), "512 B");
        assert_eq!(fs::format_file_size(1024), "1.0 KB");
        assert_eq!(fs::format_file_size(1536), "1.5 KB");
        assert_eq!(fs::format_file_size(1048576), "1.0 MB");
    }
    
    #[test]
    fn test_math_utilities() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(math::percentile(&values, 50.0), 3.0);
        assert_eq!(math::percentile(&values, 0.0), 1.0);
        assert_eq!(math::percentile(&values, 100.0), 5.0);
        
        assert_eq!(math::normalize(2.5, 1.0, 4.0), 0.5);
        assert_eq!(math::normalize(0.0, 1.0, 4.0), 0.0);
        assert_eq!(math::normalize(5.0, 1.0, 4.0), 1.0);
        
        let ma = math::moving_average(&values, 3);
        assert_eq!(ma.len(), 5);
        assert!((ma[2] - 2.0).abs() < 0.001); // (1+2+3)/3 = 2
        assert!((ma[4] - 4.0).abs() < 0.001); // (3+4+5)/3 = 4
    }
    
    #[test]
    fn test_string_utilities() {
        assert_eq!(string::truncate_with_ellipsis("hello world", 8), "hello...");
        assert_eq!(string::truncate_with_ellipsis("short", 10), "short");
        
        assert_eq!(string::snake_to_camel("hello_world"), "helloWorld");
        assert_eq!(string::snake_to_camel("test_case_one"), "testCaseOne");
        
        assert_eq!(string::bytes_to_hex(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }
    
    #[test]
    fn test_validation() {
        use crate::types::UEMetrics;
        
        let mut metrics = UEMetrics::new("test_ue".to_string());
        assert!(validation::validate_ue_metrics(&metrics).is_ok());
        
        metrics.pcell_cqi = 20.0; // Invalid
        assert!(validation::validate_ue_metrics(&metrics).is_err());
        
        assert!(validation::validate_confidence_score(0.5).is_ok());
        assert!(validation::validate_confidence_score(1.5).is_err());
        
        assert!(validation::validate_prediction_horizon(30).is_ok());
        assert!(validation::validate_prediction_horizon(-5).is_err());
        assert!(validation::validate_prediction_horizon(5000).is_err());
    }
}