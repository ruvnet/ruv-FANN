//! Data structures and management for handover prediction
//!
//! This module defines the core data structures used throughout the handover
//! prediction system, including UE metrics, handover events, and dataset management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// UE (User Equipment) metrics at a specific point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UeMetrics {
    pub timestamp: DateTime<Utc>,
    pub ue_id: String,
    pub serving_cell_id: String,
    
    // Radio measurements
    pub serving_rsrp: f64,          // Reference Signal Received Power (dBm)
    pub serving_sinr: f64,          // Signal-to-Interference-plus-Noise Ratio (dB)
    pub serving_rsrq: f64,          // Reference Signal Received Quality (dB)
    pub serving_cqi: f64,           // Channel Quality Indicator (0-15)
    pub serving_ta: f64,            // Timing Advance (microseconds)
    pub serving_phr: f64,           // Power Headroom Report (dB)
    
    // Best neighbor cell measurements
    pub neighbor_rsrp_best: f64,    // Best neighbor RSRP (dBm)
    pub neighbor_rsrq_best: f64,    // Best neighbor RSRQ (dB)
    pub neighbor_sinr_best: f64,    // Best neighbor SINR (dB)
    pub best_neighbor_cell_id: Option<String>,
    
    // UE mobility information
    pub ue_speed_kmh: f64,          // UE speed in km/h
    pub ue_bearing_degrees: Option<f64>,  // UE bearing in degrees
    pub ue_altitude_m: Option<f64>,       // UE altitude in meters
    
    // Neighbor cells detailed information
    pub neighbor_cells: Vec<NeighborCell>,
    
    // Traffic and load information
    pub serving_prb_usage: Option<f64>,    // PRB usage percentage
    pub serving_user_count: Option<u32>,   // Active users in serving cell
    pub serving_throughput_mbps: Option<f64>, // Cell throughput
    
    // Additional context
    pub frequency_band: String,
    pub technology: String,         // LTE, 5G-NSA, 5G-SA
    pub measurement_gap: Option<bool>,
}

/// Neighbor cell information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborCell {
    pub cell_id: String,
    pub rsrp: f64,
    pub rsrq: f64,
    pub sinr: f64,
    pub distance_km: Option<f64>,
    pub frequency_band: String,
    pub technology: String,
    pub azimuth_degrees: Option<f64>,
    pub cell_load_percent: Option<f64>,
    pub handover_success_rate: Option<f64>,
}

/// Handover event record for training and validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverEvent {
    pub event_id: Uuid,
    pub ue_id: String,
    pub source_cell_id: String,
    pub target_cell_id: String,
    pub handover_timestamp: DateTime<Utc>,
    pub handover_type: HandoverType,
    pub success: bool,
    pub preparation_time_ms: Option<u64>,
    pub execution_time_ms: Option<u64>,
    pub failure_reason: Option<String>,
    
    // UE metrics just before handover
    pub pre_handover_metrics: UeMetrics,
    
    // UE metrics after handover (if successful)
    pub post_handover_metrics: Option<UeMetrics>,
    
    // Derived features
    pub rsrp_delta: f64,           // Target RSRP - Source RSRP
    pub sinr_delta: f64,           // Target SINR - Source SINR
    pub distance_to_target: Option<f64>,
    pub time_in_source_cell_sec: Option<u64>,
}

/// Types of handover events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HandoverType {
    Intra5G,        // Within 5G network
    InterLte,       // Between LTE cells
    Lte5GNsa,       // LTE to 5G NSA
    FiveGNsaLte,    // 5G NSA to LTE
    IntraLte,       // Within LTE network
    InterRat,       // Between different radio access technologies
    Emergency,      // Emergency handover
}

/// Dataset for training and validation
#[derive(Debug, Clone)]
pub struct HandoverDataset {
    pub ue_metrics: Vec<UeMetrics>,
    pub handover_events: Vec<HandoverEvent>,
    pub metadata: DatasetMetadata,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub total_samples: usize,
    pub positive_samples: usize,    // Samples with handover
    pub negative_samples: usize,    // Samples without handover
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    pub unique_ues: usize,
    pub unique_cells: usize,
    pub geographic_region: Option<String>,
    pub network_operator: Option<String>,
    pub data_sources: Vec<String>,
}

impl UeMetrics {
    /// Create a new UeMetrics instance with required fields
    pub fn new(ue_id: &str, serving_cell_id: &str) -> Self {
        Self {
            timestamp: Utc::now(),
            ue_id: ue_id.to_string(),
            serving_cell_id: serving_cell_id.to_string(),
            serving_rsrp: -100.0,
            serving_sinr: 0.0,
            serving_rsrq: -20.0,
            serving_cqi: 7.0,
            serving_ta: 0.0,
            serving_phr: 0.0,
            neighbor_rsrp_best: -110.0,
            neighbor_rsrq_best: -25.0,
            neighbor_sinr_best: -5.0,
            best_neighbor_cell_id: None,
            ue_speed_kmh: 0.0,
            ue_bearing_degrees: None,
            ue_altitude_m: None,
            neighbor_cells: Vec::new(),
            serving_prb_usage: None,
            serving_user_count: None,
            serving_throughput_mbps: None,
            frequency_band: "B1".to_string(),
            technology: "LTE".to_string(),
            measurement_gap: None,
        }
    }
    
    /// Builder pattern methods
    pub fn with_rsrp(mut self, rsrp: f64) -> Self {
        self.serving_rsrp = rsrp;
        self
    }
    
    pub fn with_sinr(mut self, sinr: f64) -> Self {
        self.serving_sinr = sinr;
        self
    }
    
    pub fn with_speed(mut self, speed_kmh: f64) -> Self {
        self.ue_speed_kmh = speed_kmh;
        self
    }
    
    pub fn with_neighbor_rsrp(mut self, rsrp: f64) -> Self {
        self.neighbor_rsrp_best = rsrp;
        self
    }
    
    pub fn with_neighbor_cells(mut self, cells: Vec<NeighborCell>) -> Self {
        self.neighbor_cells = cells;
        self
    }
    
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }
    
    /// Calculate RSRP difference (neighbor - serving)
    pub fn rsrp_delta(&self) -> f64 {
        self.neighbor_rsrp_best - self.serving_rsrp
    }
    
    /// Calculate SINR difference (neighbor - serving)
    pub fn sinr_delta(&self) -> f64 {
        self.neighbor_sinr_best - self.serving_sinr
    }
    
    /// Get the best neighbor cell
    pub fn best_neighbor(&self) -> Option<&NeighborCell> {
        self.neighbor_cells.iter()
            .max_by(|a, b| a.rsrp.partial_cmp(&b.rsrp).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Check if handover criteria are met (basic A3 event)
    pub fn meets_a3_criteria(&self, hysteresis: f64, time_to_trigger_ms: u64) -> bool {
        // Simplified A3 event: Neighbor becomes offset amount better than serving
        let offset = 3.0; // dB
        self.rsrp_delta() > offset + hysteresis
    }
    
    /// Validate metrics for completeness and ranges
    pub fn validate(&self) -> Result<(), String> {
        // RSRP typically ranges from -140 to -44 dBm
        if self.serving_rsrp < -140.0 || self.serving_rsrp > -44.0 {
            return Err(format!("Invalid serving RSRP: {}", self.serving_rsrp));
        }
        
        // SINR typically ranges from -20 to 30 dB
        if self.serving_sinr < -20.0 || self.serving_sinr > 30.0 {
            return Err(format!("Invalid serving SINR: {}", self.serving_sinr));
        }
        
        // Speed should be non-negative and reasonable (< 500 km/h)
        if self.ue_speed_kmh < 0.0 || self.ue_speed_kmh > 500.0 {
            return Err(format!("Invalid UE speed: {}", self.ue_speed_kmh));
        }
        
        Ok(())
    }
}

impl HandoverEvent {
    /// Create a new handover event
    pub fn new(
        ue_id: &str,
        source_cell_id: &str,
        target_cell_id: &str,
        pre_handover_metrics: UeMetrics,
    ) -> Self {
        let rsrp_delta = pre_handover_metrics.rsrp_delta();
        let sinr_delta = pre_handover_metrics.sinr_delta();
        
        Self {
            event_id: Uuid::new_v4(),
            ue_id: ue_id.to_string(),
            source_cell_id: source_cell_id.to_string(),
            target_cell_id: target_cell_id.to_string(),
            handover_timestamp: Utc::now(),
            handover_type: HandoverType::IntraLte,
            success: true,
            preparation_time_ms: None,
            execution_time_ms: None,
            failure_reason: None,
            pre_handover_metrics,
            post_handover_metrics: None,
            rsrp_delta,
            sinr_delta,
            distance_to_target: None,
            time_in_source_cell_sec: None,
        }
    }
    
    /// Set handover as successful with post-handover metrics
    pub fn with_success(mut self, post_metrics: UeMetrics) -> Self {
        self.success = true;
        self.post_handover_metrics = Some(post_metrics);
        self
    }
    
    /// Set handover as failed with reason
    pub fn with_failure(mut self, reason: String) -> Self {
        self.success = false;
        self.failure_reason = Some(reason);
        self
    }
    
    /// Calculate handover gain (improvement in RSRP)
    pub fn handover_gain(&self) -> Option<f64> {
        self.post_handover_metrics.as_ref().map(|post| {
            post.serving_rsrp - self.pre_handover_metrics.serving_rsrp
        })
    }
}

impl HandoverDataset {
    /// Create a new empty dataset
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            ue_metrics: Vec::new(),
            handover_events: Vec::new(),
            metadata: DatasetMetadata {
                name: name.to_string(),
                version: version.to_string(),
                created_at: Utc::now(),
                total_samples: 0,
                positive_samples: 0,
                negative_samples: 0,
                time_range: (Utc::now(), Utc::now()),
                unique_ues: 0,
                unique_cells: 0,
                geographic_region: None,
                network_operator: None,
                data_sources: Vec::new(),
            },
        }
    }
    
    /// Add UE metrics to the dataset
    pub fn add_metrics(&mut self, metrics: Vec<UeMetrics>) {
        self.ue_metrics.extend(metrics);
        self.update_metadata();
    }
    
    /// Add handover events to the dataset
    pub fn add_handover_events(&mut self, events: Vec<HandoverEvent>) {
        self.handover_events.extend(events);
        self.update_metadata();
    }
    
    /// Update dataset metadata
    fn update_metadata(&mut self) {
        self.metadata.total_samples = self.ue_metrics.len();
        self.metadata.positive_samples = self.handover_events.len();
        self.metadata.negative_samples = self.metadata.total_samples - self.metadata.positive_samples;
        
        // Calculate time range
        if !self.ue_metrics.is_empty() {
            let timestamps: Vec<_> = self.ue_metrics.iter().map(|m| m.timestamp).collect();
            self.metadata.time_range = (
                *timestamps.iter().min().unwrap(),
                *timestamps.iter().max().unwrap(),
            );
        }
        
        // Count unique UEs and cells
        let unique_ues: std::collections::HashSet<_> = 
            self.ue_metrics.iter().map(|m| &m.ue_id).collect();
        let unique_cells: std::collections::HashSet<_> = 
            self.ue_metrics.iter().map(|m| &m.serving_cell_id).collect();
            
        self.metadata.unique_ues = unique_ues.len();
        self.metadata.unique_cells = unique_cells.len();
    }
    
    /// Get metrics for a specific UE
    pub fn get_ue_metrics(&self, ue_id: &str) -> Vec<&UeMetrics> {
        self.ue_metrics.iter()
            .filter(|m| m.ue_id == ue_id)
            .collect()
    }
    
    /// Get handover events for a specific UE
    pub fn get_ue_handovers(&self, ue_id: &str) -> Vec<&HandoverEvent> {
        self.handover_events.iter()
            .filter(|e| e.ue_id == ue_id)
            .collect()
    }
    
    /// Split dataset into training and validation sets
    pub fn split(&self, train_ratio: f64) -> (Self, Self) {
        let split_index = (self.ue_metrics.len() as f64 * train_ratio) as usize;
        
        let (train_metrics, val_metrics) = self.ue_metrics.split_at(split_index);
        let (train_events, val_events) = self.handover_events.split_at(
            (self.handover_events.len() as f64 * train_ratio) as usize
        );
        
        let mut train_dataset = Self::new(
            &format!("{}_train", self.metadata.name),
            &self.metadata.version
        );
        train_dataset.add_metrics(train_metrics.to_vec());
        train_dataset.add_handover_events(train_events.to_vec());
        
        let mut val_dataset = Self::new(
            &format!("{}_val", self.metadata.name),
            &self.metadata.version
        );
        val_dataset.add_metrics(val_metrics.to_vec());
        val_dataset.add_handover_events(val_events.to_vec());
        
        (train_dataset, val_dataset)
    }
    
    /// Get dataset statistics
    pub fn statistics(&self) -> DatasetStatistics {
        DatasetStatistics::from_dataset(self)
    }
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    pub total_samples: usize,
    pub handover_rate: f64,
    pub avg_rsrp: f64,
    pub avg_sinr: f64,
    pub avg_speed: f64,
    pub rsrp_std: f64,
    pub sinr_std: f64,
    pub speed_std: f64,
    pub handover_success_rate: f64,
    pub avg_handover_gain: f64,
    pub cell_distribution: HashMap<String, usize>,
    pub technology_distribution: HashMap<String, usize>,
}

impl DatasetStatistics {
    fn from_dataset(dataset: &HandoverDataset) -> Self {
        let total_samples = dataset.ue_metrics.len();
        let handover_rate = dataset.handover_events.len() as f64 / total_samples as f64;
        
        // Calculate means
        let avg_rsrp = dataset.ue_metrics.iter().map(|m| m.serving_rsrp).sum::<f64>() / total_samples as f64;
        let avg_sinr = dataset.ue_metrics.iter().map(|m| m.serving_sinr).sum::<f64>() / total_samples as f64;
        let avg_speed = dataset.ue_metrics.iter().map(|m| m.ue_speed_kmh).sum::<f64>() / total_samples as f64;
        
        // Calculate standard deviations
        let rsrp_var = dataset.ue_metrics.iter()
            .map(|m| (m.serving_rsrp - avg_rsrp).powi(2))
            .sum::<f64>() / total_samples as f64;
        let rsrp_std = rsrp_var.sqrt();
        
        let sinr_var = dataset.ue_metrics.iter()
            .map(|m| (m.serving_sinr - avg_sinr).powi(2))
            .sum::<f64>() / total_samples as f64;
        let sinr_std = sinr_var.sqrt();
        
        let speed_var = dataset.ue_metrics.iter()
            .map(|m| (m.ue_speed_kmh - avg_speed).powi(2))
            .sum::<f64>() / total_samples as f64;
        let speed_std = speed_var.sqrt();
        
        // Handover statistics
        let successful_handovers = dataset.handover_events.iter().filter(|e| e.success).count();
        let handover_success_rate = successful_handovers as f64 / dataset.handover_events.len() as f64;
        
        let avg_handover_gain = dataset.handover_events.iter()
            .filter_map(|e| e.handover_gain())
            .sum::<f64>() / successful_handovers as f64;
        
        // Distribution calculations
        let mut cell_distribution = HashMap::new();
        let mut technology_distribution = HashMap::new();
        
        for metrics in &dataset.ue_metrics {
            *cell_distribution.entry(metrics.serving_cell_id.clone()).or_insert(0) += 1;
            *technology_distribution.entry(metrics.technology.clone()).or_insert(0) += 1;
        }
        
        Self {
            total_samples,
            handover_rate,
            avg_rsrp,
            avg_sinr,
            avg_speed,
            rsrp_std,
            sinr_std,
            speed_std,
            handover_success_rate,
            avg_handover_gain,
            cell_distribution,
            technology_distribution,
        }
    }
}