//! Utility functions and helpers for the handover prediction system

use crate::data::{HandoverDataset, HandoverEvent, UeMetrics, NeighborCell};
use crate::{OptMobError, Result};
use chrono::{DateTime, Utc};
use rand::Rng;
use std::collections::HashMap;
use std::path::Path;

/// Data generation utilities for testing and simulation
pub struct DataGenerator {
    rng: rand::rngs::ThreadRng,
    cell_ids: Vec<String>,
    ue_ids: Vec<String>,
}

impl DataGenerator {
    /// Create a new data generator
    pub fn new() -> Self {
        let cell_ids = (1..=20).map(|i| format!("Cell_{:03}", i)).collect();
        let ue_ids = (1..=100).map(|i| format!("UE_{:03}", i)).collect();
        
        Self {
            rng: rand::thread_rng(),
            cell_ids,
            ue_ids,
        }
    }
    
    /// Generate synthetic UE metrics for testing
    pub fn generate_ue_metrics(
        &mut self,
        ue_id: &str,
        serving_cell_id: &str,
        duration_hours: f64,
        sample_rate_seconds: u64,
    ) -> Vec<UeMetrics> {
        let mut metrics = Vec::new();
        let total_samples = (duration_hours * 3600.0 / sample_rate_seconds as f64) as usize;
        let start_time = Utc::now() - chrono::Duration::hours(duration_hours as i64);
        
        // Base signal parameters for this UE
        let base_rsrp = self.rng.gen_range(-110.0..-70.0);
        let base_sinr = self.rng.gen_range(0.0..20.0);
        let base_speed = self.rng.gen_range(0.0..120.0);
        
        for i in 0..total_samples {
            let timestamp = start_time + chrono::Duration::seconds((i * sample_rate_seconds) as i64);
            
            // Add realistic variations
            let rsrp_variation = self.rng.gen_range(-5.0..5.0);
            let sinr_variation = self.rng.gen_range(-3.0..3.0);
            let speed_variation = self.rng.gen_range(-10.0..10.0);
            
            let serving_rsrp = base_rsrp + rsrp_variation;
            let serving_sinr = base_sinr + sinr_variation;
            let ue_speed = (base_speed + speed_variation).max(0.0);
            
            // Generate neighbor cells
            let neighbor_cells = self.generate_neighbor_cells(serving_rsrp);
            let best_neighbor_rsrp = neighbor_cells.iter()
                .map(|n| n.rsrp)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(serving_rsrp - 10.0);
            
            let mut ue_metrics = UeMetrics::new(ue_id, serving_cell_id)
                .with_timestamp(timestamp)
                .with_rsrp(serving_rsrp)
                .with_sinr(serving_sinr)
                .with_speed(ue_speed)
                .with_neighbor_rsrp(best_neighbor_rsrp)
                .with_neighbor_cells(neighbor_cells);
            
            // Add more realistic parameters
            ue_metrics.serving_rsrq = serving_rsrp + self.rng.gen_range(-5.0..0.0);
            ue_metrics.serving_cqi = (serving_sinr / 2.0).max(1.0).min(15.0);
            ue_metrics.serving_ta = self.rng.gen_range(0.0..10.0);
            ue_metrics.serving_phr = self.rng.gen_range(-10.0..20.0);
            ue_metrics.neighbor_rsrq_best = best_neighbor_rsrp + self.rng.gen_range(-5.0..0.0);
            ue_metrics.neighbor_sinr_best = serving_sinr + self.rng.gen_range(-2.0..5.0);
            
            // Technology and band
            ue_metrics.technology = if self.rng.gen_bool(0.7) { "LTE".to_string() } else { "5G-NSA".to_string() };
            ue_metrics.frequency_band = format!("B{}", self.rng.gen_range(1..=20));
            
            metrics.push(ue_metrics);
        }
        
        metrics
    }
    
    /// Generate realistic neighbor cells
    fn generate_neighbor_cells(&mut self, serving_rsrp: f64) -> Vec<NeighborCell> {
        let neighbor_count = self.rng.gen_range(2..=6);
        let mut neighbors = Vec::new();
        
        for i in 0..neighbor_count {
            let cell_id = self.cell_ids[self.rng.gen_range(0..self.cell_ids.len())].clone();
            
            // Neighbors typically have weaker signal than serving cell
            let rsrp_offset = self.rng.gen_range(-15.0..5.0);
            let rsrp = serving_rsrp + rsrp_offset;
            
            let neighbor = NeighborCell {
                cell_id,
                rsrp,
                rsrq: rsrp + self.rng.gen_range(-5.0..0.0),
                sinr: rsrp + 100.0 + self.rng.gen_range(-10.0..10.0), // Convert to SINR-like value
                distance_km: Some(self.rng.gen_range(0.1..5.0)),
                frequency_band: format!("B{}", self.rng.gen_range(1..=20)),
                technology: if self.rng.gen_bool(0.7) { "LTE".to_string() } else { "5G-NSA".to_string() },
                azimuth_degrees: Some(self.rng.gen_range(0.0..360.0)),
                cell_load_percent: Some(self.rng.gen_range(0.1..0.9)),
                handover_success_rate: Some(self.rng.gen_range(0.85..0.98)),
            };
            
            neighbors.push(neighbor);
        }
        
        neighbors
    }
    
    /// Generate synthetic handover events
    pub fn generate_handover_events(
        &mut self,
        ue_metrics: &[UeMetrics],
        handover_probability: f64,
    ) -> Vec<HandoverEvent> {
        let mut events = Vec::new();
        let mut last_handover_time: HashMap<String, DateTime<Utc>> = HashMap::new();
        
        for metrics in ue_metrics {
            // Check if we should generate a handover event
            let should_handover = self.should_generate_handover(metrics, handover_probability);
            
            // Ensure minimum time between handovers for the same UE
            if let Some(last_time) = last_handover_time.get(&metrics.ue_id) {
                let time_diff = metrics.timestamp.signed_duration_since(*last_time);
                if time_diff.num_seconds() < 30 { // Minimum 30 seconds between handovers
                    continue;
                }
            }
            
            if should_handover {
                let target_cell = self.select_target_cell(metrics);
                
                let mut event = HandoverEvent::new(
                    &metrics.ue_id,
                    &metrics.serving_cell_id,
                    &target_cell,
                    metrics.clone(),
                );
                
                // Determine handover type
                event.handover_type = if metrics.technology.contains("5G") {
                    crate::data::HandoverType::Intra5G
                } else {
                    crate::data::HandoverType::IntraLte
                };
                
                // Simulate success/failure
                event.success = self.rng.gen_bool(0.95); // 95% success rate
                
                if event.success {
                    // Generate post-handover metrics
                    let mut post_metrics = metrics.clone();
                    post_metrics.serving_cell_id = target_cell.clone();
                    post_metrics.serving_rsrp += self.rng.gen_range(3.0..8.0); // Gain from handover
                    post_metrics.serving_sinr += self.rng.gen_range(1.0..5.0);
                    post_metrics.timestamp += chrono::Duration::seconds(self.rng.gen_range(1..5));
                    
                    event = event.with_success(post_metrics);
                } else {
                    event = event.with_failure("Radio link failure".to_string());
                }
                
                event.preparation_time_ms = Some(self.rng.gen_range(50..200));
                event.execution_time_ms = Some(self.rng.gen_range(10..100));
                
                events.push(event);
                last_handover_time.insert(metrics.ue_id.clone(), metrics.timestamp);
            }
        }
        
        events
    }
    
    /// Determine if a handover should be generated based on signal conditions
    fn should_generate_handover(&mut self, metrics: &UeMetrics, base_probability: f64) -> bool {
        let mut probability = base_probability;
        
        // Increase probability for poor serving cell quality
        if metrics.serving_rsrp < -100.0 {
            probability += 0.3;
        }
        if metrics.serving_sinr < 5.0 {
            probability += 0.2;
        }
        
        // Increase probability for strong neighbor
        if metrics.neighbor_rsrp_best > metrics.serving_rsrp + 6.0 {
            probability += 0.4;
        }
        
        // Increase probability for high mobility
        if metrics.ue_speed_kmh > 60.0 {
            probability += 0.2;
        }
        
        // Time-based factors
        let hour = metrics.timestamp.hour();
        if hour >= 8 && hour <= 10 || hour >= 17 && hour <= 19 {
            probability += 0.1; // Busy hours
        }
        
        probability = probability.min(0.8); // Cap at 80%
        
        self.rng.gen_bool(probability)
    }
    
    /// Select target cell for handover
    fn select_target_cell(&mut self, metrics: &UeMetrics) -> String {
        if !metrics.neighbor_cells.is_empty() {
            // Select the best neighbor based on RSRP
            let best_neighbor = metrics.neighbor_cells.iter()
                .max_by(|a, b| a.rsrp.partial_cmp(&b.rsrp).unwrap())
                .unwrap();
            best_neighbor.cell_id.clone()
        } else {
            // Fallback to a random cell
            self.cell_ids[self.rng.gen_range(0..self.cell_ids.len())].clone()
        }
    }
    
    /// Generate a complete synthetic dataset
    pub fn generate_dataset(
        &mut self,
        num_ues: usize,
        duration_hours: f64,
        sample_rate_seconds: u64,
        handover_rate: f64,
    ) -> HandoverDataset {
        let mut dataset = HandoverDataset::new("synthetic", "1.0");
        let mut all_metrics = Vec::new();
        
        // Generate UE IDs
        let ue_ids: Vec<String> = (0..num_ues).map(|i| format!("UE_{:03}", i)).collect();
        
        // Generate metrics for each UE
        for ue_id in &ue_ids {
            let serving_cell = &self.cell_ids[self.rng.gen_range(0..self.cell_ids.len())];
            let metrics = self.generate_ue_metrics(ue_id, serving_cell, duration_hours, sample_rate_seconds);
            all_metrics.extend(metrics);
        }
        
        // Generate handover events
        let handover_events = self.generate_handover_events(&all_metrics, handover_rate);
        
        // Add to dataset
        dataset.add_metrics(all_metrics);
        dataset.add_handover_events(handover_events);
        
        dataset
    }
}

impl Default for DataGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration utilities
pub struct ConfigUtils;

impl ConfigUtils {
    /// Load configuration from file
    pub fn load_config<P: AsRef<Path>>(path: P) -> Result<crate::OptMobConfig> {
        let content = std::fs::read_to_string(path)?;
        let config: crate::OptMobConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_config<P: AsRef<Path>>(config: &crate::OptMobConfig, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Create default configuration file
    pub fn create_default_config<P: AsRef<Path>>(path: P) -> Result<()> {
        let config = crate::OptMobConfig::default();
        Self::save_config(&config, path)
    }
}

/// Performance monitoring utilities
pub struct PerformanceMonitor {
    start_time: std::time::Instant,
    checkpoints: Vec<(String, std::time::Instant)>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            checkpoints: Vec::new(),
        }
    }
    
    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), std::time::Instant::now()));
    }
    
    pub fn report(&self) -> String {
        let mut report = String::new();
        let total_time = self.start_time.elapsed();
        
        report.push_str(&format!("Total execution time: {:.2}ms\n", total_time.as_millis()));
        
        let mut last_time = self.start_time;
        for (name, time) in &self.checkpoints {
            let elapsed = time.duration_since(last_time);
            report.push_str(&format!("  {}: {:.2}ms\n", name, elapsed.as_millis()));
            last_time = *time;
        }
        
        report
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate UE metrics for completeness and ranges
    pub fn validate_metrics(metrics: &UeMetrics) -> Result<()> {
        metrics.validate().map_err(|e| OptMobError::Data(e))
    }
    
    /// Validate dataset quality
    pub fn validate_dataset(dataset: &HandoverDataset) -> Result<DatasetValidationReport> {
        let mut report = DatasetValidationReport::default();
        
        // Check for empty dataset
        if dataset.ue_metrics.is_empty() {
            report.errors.push("Dataset is empty".to_string());
            return Ok(report);
        }
        
        // Validate individual metrics
        for (i, metrics) in dataset.ue_metrics.iter().enumerate() {
            if let Err(e) = Self::validate_metrics(metrics) {
                report.warnings.push(format!("Sample {}: {}", i, e));
                report.invalid_samples += 1;
            }
        }
        
        // Check for temporal ordering
        let mut sorted_by_time = dataset.ue_metrics.clone();
        sorted_by_time.sort_by_key(|m| m.timestamp);
        
        if sorted_by_time != dataset.ue_metrics {
            report.warnings.push("Dataset is not sorted by timestamp".to_string());
        }
        
        // Check handover rate
        let handover_rate = dataset.handover_events.len() as f64 / dataset.ue_metrics.len() as f64;
        if handover_rate > 0.3 {
            report.warnings.push(format!("High handover rate: {:.1}%", handover_rate * 100.0));
        } else if handover_rate < 0.01 {
            report.warnings.push(format!("Low handover rate: {:.3}%", handover_rate * 100.0));
        }
        
        // Check for data gaps
        let time_gaps = Self::find_time_gaps(&dataset.ue_metrics);
        if !time_gaps.is_empty() {
            report.warnings.push(format!("Found {} significant time gaps", time_gaps.len()));
        }
        
        report.total_samples = dataset.ue_metrics.len();
        report.valid_samples = dataset.ue_metrics.len() - report.invalid_samples;
        report.handover_events = dataset.handover_events.len();
        
        Ok(report)
    }
    
    /// Find significant time gaps in the data
    fn find_time_gaps(metrics: &[UeMetrics]) -> Vec<(DateTime<Utc>, DateTime<Utc>)> {
        let mut gaps = Vec::new();
        let max_gap_minutes = 30; // Consider gaps > 30 minutes as significant
        
        for window in metrics.windows(2) {
            let gap = window[1].timestamp.signed_duration_since(window[0].timestamp);
            if gap.num_minutes() > max_gap_minutes {
                gaps.push((window[0].timestamp, window[1].timestamp));
            }
        }
        
        gaps
    }
}

/// Dataset validation report
#[derive(Debug, Default)]
pub struct DatasetValidationReport {
    pub total_samples: usize,
    pub valid_samples: usize,
    pub invalid_samples: usize,
    pub handover_events: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl DatasetValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty() && self.invalid_samples == 0
    }
    
    pub fn summary(&self) -> String {
        format!(
            "Dataset Validation Summary:\n\
             - Total samples: {}\n\
             - Valid samples: {}\n\
             - Invalid samples: {}\n\
             - Handover events: {}\n\
             - Errors: {}\n\
             - Warnings: {}",
            self.total_samples,
            self.valid_samples,
            self.invalid_samples,
            self.handover_events,
            self.errors.len(),
            self.warnings.len()
        )
    }
}

/// Export utilities for data exchange
pub struct ExportUtils;

impl ExportUtils {
    /// Export dataset to CSV format
    pub fn export_to_csv<P: AsRef<Path>>(
        dataset: &HandoverDataset,
        path: P,
    ) -> Result<()> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(path)?;
        
        // Write header
        writeln!(file, "timestamp,ue_id,serving_cell_id,serving_rsrp,serving_sinr,serving_rsrq,ue_speed_kmh,neighbor_rsrp_best,technology,frequency_band")?;
        
        // Write data
        for metrics in &dataset.ue_metrics {
            writeln!(
                file,
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
                metrics.timestamp.format("%Y-%m-%d %H:%M:%S"),
                metrics.ue_id,
                metrics.serving_cell_id,
                metrics.serving_rsrp,
                metrics.serving_sinr,
                metrics.serving_rsrq,
                metrics.ue_speed_kmh,
                metrics.neighbor_rsrp_best,
                metrics.technology,
                metrics.frequency_band
            )?;
        }
        
        Ok(())
    }
    
    /// Export handover events to CSV
    pub fn export_handovers_to_csv<P: AsRef<Path>>(
        events: &[HandoverEvent],
        path: P,
    ) -> Result<()> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(path)?;
        
        // Write header
        writeln!(file, "event_id,ue_id,source_cell_id,target_cell_id,handover_timestamp,handover_type,success,rsrp_delta,sinr_delta")?;
        
        // Write data
        for event in events {
            writeln!(
                file,
                "{},{},{},{},{},{:?},{},{:.2},{:.2}",
                event.event_id,
                event.ue_id,
                event.source_cell_id,
                event.target_cell_id,
                event.handover_timestamp.format("%Y-%m-%d %H:%M:%S"),
                event.handover_type,
                event.success,
                event.rsrp_delta,
                event.sinr_delta
            )?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_generator() {
        let mut generator = DataGenerator::new();
        let metrics = generator.generate_ue_metrics("UE_001", "Cell_001", 1.0, 60);
        
        assert!(!metrics.is_empty());
        assert_eq!(metrics[0].ue_id, "UE_001");
        assert_eq!(metrics[0].serving_cell_id, "Cell_001");
    }
    
    #[test]
    fn test_synthetic_dataset_generation() {
        let mut generator = DataGenerator::new();
        let dataset = generator.generate_dataset(5, 2.0, 60, 0.05);
        
        assert!(!dataset.ue_metrics.is_empty());
        assert!(dataset.handover_events.len() > 0);
    }
    
    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.checkpoint("test_checkpoint");
        
        let report = monitor.report();
        assert!(report.contains("Total execution time"));
        assert!(report.contains("test_checkpoint"));
    }
    
    #[test]
    fn test_config_utils() {
        let config = crate::OptMobConfig::default();
        let temp_file = "/tmp/test_config.json";
        
        // Test save and load
        ConfigUtils::save_config(&config, temp_file).unwrap();
        let loaded_config = ConfigUtils::load_config(temp_file).unwrap();
        
        assert_eq!(config.handover_threshold, loaded_config.handover_threshold);
        
        // Cleanup
        std::fs::remove_file(temp_file).ok();
    }
}