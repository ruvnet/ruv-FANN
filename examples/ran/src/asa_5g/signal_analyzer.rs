//! Signal Quality Analysis Module
//! 
//! This module provides advanced signal quality analysis capabilities for 5G ENDC scenarios,
//! including degradation detection, pattern recognition, and stability analysis.

use crate::asa_5g::*;
use crate::types::*;
use crate::{Result, RanError};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

/// Advanced signal quality analyzer
pub struct AdvancedSignalAnalyzer {
    config: Asa5gConfig,
    historical_data: Arc<RwLock<HashMap<String, Vec<SignalQualityMeasurement>>>>,
    degradation_patterns: Arc<RwLock<Vec<DegradationPattern>>>,
    analysis_cache: Arc<RwLock<HashMap<String, CachedAnalysis>>>,
}

/// Signal quality measurement with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualityMeasurement {
    pub timestamp: DateTime<Utc>,
    pub ue_id: UeId,
    pub cell_id: CellId,
    pub lte_rsrp: f64,
    pub lte_sinr: f64,
    pub nr_ssb_rsrp: Option<f64>,
    pub nr_sinr: Option<f64>,
    pub endc_success_rate: f64,
    pub throughput_mbps: f64,
    pub rtt_ms: f64,
    pub packet_loss_rate: f64,
    pub handover_count: u32,
    pub location: Option<Location>,
}

/// Geographic location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
    pub accuracy_meters: f64,
}

/// Degradation pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationPattern {
    pub pattern_id: String,
    pub name: String,
    pub description: String,
    pub conditions: Vec<PatternCondition>,
    pub severity: DegradationSeverity,
    pub typical_duration_minutes: u32,
    pub mitigation_suggestions: Vec<String>,
}

/// Pattern condition for degradation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCondition {
    pub metric: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub duration_minutes: u32,
}

/// Comparison operators for pattern conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotEqual,
}

/// Degradation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Cached analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAnalysis {
    pub timestamp: DateTime<Utc>,
    pub ue_id: UeId,
    pub signal_quality_score: f64,
    pub stability_score: f64,
    pub degradation_indicators: Vec<String>,
    pub trend_analysis: TrendAnalysis,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub rsrp_trend: TrendDirection,
    pub sinr_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub success_rate_trend: TrendDirection,
    pub overall_trend: TrendDirection,
    pub trend_confidence: f64,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Signal quality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualityAnalysis {
    pub timestamp: DateTime<Utc>,
    pub ue_id: UeId,
    pub overall_quality_score: f64,
    pub lte_quality_score: f64,
    pub nr_quality_score: f64,
    pub stability_metrics: StabilityMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub risk_factors: Vec<RiskFactor>,
    pub recommendations: Vec<String>,
}

/// Stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    pub signal_variance: f64,
    pub handover_frequency: f64,
    pub connection_stability: f64,
    pub quality_consistency: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub average_latency: f64,
    pub packet_loss_rate: f64,
    pub service_availability: f64,
}

/// Risk factor identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor_type: RiskFactorType,
    pub severity: RiskSeverity,
    pub description: String,
    pub impact_score: f64,
    pub confidence: f64,
}

/// Types of risk factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskFactorType {
    PoorSignalQuality,
    HighInterference,
    CellCongestion,
    HardwareIssue,
    ConfigurationProblem,
    EnvironmentalFactor,
}

/// Risk severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl AdvancedSignalAnalyzer {
    /// Create a new advanced signal analyzer
    pub fn new(config: Asa5gConfig) -> Self {
        Self {
            config,
            historical_data: Arc::new(RwLock::new(HashMap::new())),
            degradation_patterns: Arc::new(RwLock::new(Self::initialize_degradation_patterns())),
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Initialize standard degradation patterns
    fn initialize_degradation_patterns() -> Vec<DegradationPattern> {
        vec![
            DegradationPattern {
                pattern_id: "RSRP_DEGRADATION".to_string(),
                name: "RSRP Degradation".to_string(),
                description: "Continuous degradation in LTE RSRP signal strength".to_string(),
                conditions: vec![
                    PatternCondition {
                        metric: "lte_rsrp".to_string(),
                        operator: ComparisonOperator::LessThan,
                        threshold: -110.0,
                        duration_minutes: 5,
                    }
                ],
                severity: DegradationSeverity::High,
                typical_duration_minutes: 15,
                mitigation_suggestions: vec![
                    "Consider handover to stronger cell".to_string(),
                    "Increase transmission power".to_string(),
                    "Check for interference sources".to_string(),
                ],
            },
            DegradationPattern {
                pattern_id: "SINR_INSTABILITY".to_string(),
                name: "SINR Instability".to_string(),
                description: "High variance in SINR measurements indicating interference".to_string(),
                conditions: vec![
                    PatternCondition {
                        metric: "lte_sinr_variance".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        threshold: 5.0,
                        duration_minutes: 10,
                    }
                ],
                severity: DegradationSeverity::Medium,
                typical_duration_minutes: 20,
                mitigation_suggestions: vec![
                    "Investigate interference sources".to_string(),
                    "Optimize antenna configuration".to_string(),
                    "Consider frequency refarming".to_string(),
                ],
            },
            DegradationPattern {
                pattern_id: "ENDC_FAILURE_SPIKE".to_string(),
                name: "ENDC Failure Spike".to_string(),
                description: "Sharp increase in ENDC setup failures".to_string(),
                conditions: vec![
                    PatternCondition {
                        metric: "endc_success_rate".to_string(),
                        operator: ComparisonOperator::LessThan,
                        threshold: 0.7,
                        duration_minutes: 5,
                    }
                ],
                severity: DegradationSeverity::Critical,
                typical_duration_minutes: 10,
                mitigation_suggestions: vec![
                    "Check 5G network configuration".to_string(),
                    "Verify inter-RAT parameters".to_string(),
                    "Monitor core network performance".to_string(),
                ],
            },
        ]
    }
    
    /// Add signal quality measurement to historical data
    pub async fn add_measurement(&self, measurement: SignalQualityMeasurement) -> Result<()> {
        let mut data_lock = self.historical_data.write().await;
        let ue_key = measurement.ue_id.0.clone();
        
        data_lock.entry(ue_key).or_insert_with(Vec::new).push(measurement);
        
        // Keep only recent measurements (last 24 hours)
        if let Some(measurements) = data_lock.get_mut(&ue_key) {
            let cutoff_time = Utc::now() - Duration::hours(24);
            measurements.retain(|m| m.timestamp > cutoff_time);
        }
        
        Ok(())
    }
    
    /// Perform comprehensive signal quality analysis
    pub async fn analyze_comprehensive(&self, ue_id: &UeId) -> Result<SignalQualityAnalysis> {
        let data_lock = self.historical_data.read().await;
        let measurements = data_lock.get(&ue_id.0)
            .ok_or_else(|| RanError::DataError("No measurements found for UE".to_string()))?;
        
        if measurements.is_empty() {
            return Err(RanError::DataError("No measurements available".to_string()));
        }
        
        let latest_measurement = measurements.last().unwrap();
        
        // Calculate quality scores
        let lte_quality_score = self.calculate_lte_quality_score(measurements).await?;
        let nr_quality_score = self.calculate_nr_quality_score(measurements).await?;
        let overall_quality_score = self.calculate_overall_quality_score(lte_quality_score, nr_quality_score).await?;
        
        // Calculate stability metrics
        let stability_metrics = self.calculate_stability_metrics(measurements).await?;
        
        // Calculate performance metrics
        let performance_metrics = self.calculate_performance_metrics(measurements).await?;
        
        // Identify risk factors
        let risk_factors = self.identify_risk_factors(measurements).await?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&risk_factors, &stability_metrics).await?;
        
        Ok(SignalQualityAnalysis {
            timestamp: latest_measurement.timestamp,
            ue_id: ue_id.clone(),
            overall_quality_score,
            lte_quality_score,
            nr_quality_score,
            stability_metrics,
            performance_metrics,
            risk_factors,
            recommendations,
        })
    }
    
    /// Calculate LTE signal quality score
    async fn calculate_lte_quality_score(&self, measurements: &[SignalQualityMeasurement]) -> Result<f64> {
        if measurements.is_empty() {
            return Ok(0.0);
        }
        
        let mut score = 0.0;
        let mut count = 0;
        
        for measurement in measurements {
            let rsrp_score = self.normalize_rsrp_score(measurement.lte_rsrp);
            let sinr_score = self.normalize_sinr_score(measurement.lte_sinr);
            
            score += (rsrp_score + sinr_score) / 2.0;
            count += 1;
        }
        
        Ok(score / count as f64)
    }
    
    /// Calculate 5G NR signal quality score
    async fn calculate_nr_quality_score(&self, measurements: &[SignalQualityMeasurement]) -> Result<f64> {
        let mut score = 0.0;
        let mut count = 0;
        
        for measurement in measurements {
            if let Some(nr_rsrp) = measurement.nr_ssb_rsrp {
                let rsrp_score = self.normalize_rsrp_score(nr_rsrp);
                let sinr_score = measurement.nr_sinr.map(|s| self.normalize_sinr_score(s)).unwrap_or(0.5);
                
                score += (rsrp_score + sinr_score) / 2.0;
                count += 1;
            }
        }
        
        if count == 0 {
            Ok(0.0)
        } else {
            Ok(score / count as f64)
        }
    }
    
    /// Calculate overall quality score
    async fn calculate_overall_quality_score(&self, lte_score: f64, nr_score: f64) -> Result<f64> {
        // Weight LTE more heavily if no 5G data is available
        if nr_score == 0.0 {
            Ok(lte_score * 0.8) // Penalize for lack of 5G
        } else {
            Ok((lte_score * 0.4) + (nr_score * 0.6)) // Favor 5G when available
        }
    }
    
    /// Calculate stability metrics
    async fn calculate_stability_metrics(&self, measurements: &[SignalQualityMeasurement]) -> Result<StabilityMetrics> {
        let mut rsrp_values = Vec::new();
        let mut sinr_values = Vec::new();
        let mut throughput_values = Vec::new();
        let mut handover_count = 0;
        
        for measurement in measurements {
            rsrp_values.push(measurement.lte_rsrp);
            sinr_values.push(measurement.lte_sinr);
            throughput_values.push(measurement.throughput_mbps);
            handover_count += measurement.handover_count;
        }
        
        let signal_variance = self.calculate_variance(&rsrp_values) + self.calculate_variance(&sinr_values);
        let handover_frequency = handover_count as f64 / measurements.len() as f64;
        let connection_stability = 1.0 - (signal_variance / 100.0).min(1.0);
        let quality_consistency = 1.0 - (self.calculate_variance(&throughput_values) / 100.0).min(1.0);
        
        Ok(StabilityMetrics {
            signal_variance,
            handover_frequency,
            connection_stability,
            quality_consistency,
        })
    }
    
    /// Calculate performance metrics
    async fn calculate_performance_metrics(&self, measurements: &[SignalQualityMeasurement]) -> Result<PerformanceMetrics> {
        let mut throughput_values = Vec::new();
        let mut latency_values = Vec::new();
        let mut packet_loss_values = Vec::new();
        
        for measurement in measurements {
            throughput_values.push(measurement.throughput_mbps);
            latency_values.push(measurement.rtt_ms);
            packet_loss_values.push(measurement.packet_loss_rate);
        }
        
        let average_throughput = self.calculate_mean(&throughput_values);
        let peak_throughput = throughput_values.iter().cloned().fold(0.0, f64::max);
        let average_latency = self.calculate_mean(&latency_values);
        let packet_loss_rate = self.calculate_mean(&packet_loss_values);
        
        // Calculate service availability based on successful connections
        let successful_connections = measurements.iter()
            .filter(|m| m.endc_success_rate > 0.8)
            .count();
        let service_availability = successful_connections as f64 / measurements.len() as f64;
        
        Ok(PerformanceMetrics {
            average_throughput,
            peak_throughput,
            average_latency,
            packet_loss_rate,
            service_availability,
        })
    }
    
    /// Identify risk factors
    async fn identify_risk_factors(&self, measurements: &[SignalQualityMeasurement]) -> Result<Vec<RiskFactor>> {
        let mut risk_factors = Vec::new();
        
        let latest = measurements.last().unwrap();
        
        // Check for poor signal quality
        if latest.lte_rsrp < -110.0 {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::PoorSignalQuality,
                severity: RiskSeverity::High,
                description: "LTE RSRP below acceptable threshold".to_string(),
                impact_score: 0.8,
                confidence: 0.9,
            });
        }
        
        // Check for high interference
        if latest.lte_sinr < 0.0 {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::HighInterference,
                severity: RiskSeverity::Medium,
                description: "LTE SINR indicates high interference".to_string(),
                impact_score: 0.6,
                confidence: 0.8,
            });
        }
        
        // Check for ENDC setup issues
        if latest.endc_success_rate < 0.8 {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::ConfigurationProblem,
                severity: RiskSeverity::High,
                description: "Low ENDC setup success rate".to_string(),
                impact_score: 0.9,
                confidence: 0.95,
            });
        }
        
        // Check for high packet loss
        if latest.packet_loss_rate > 0.05 {
            risk_factors.push(RiskFactor {
                factor_type: RiskFactorType::CellCongestion,
                severity: RiskSeverity::Medium,
                description: "High packet loss indicating congestion".to_string(),
                impact_score: 0.7,
                confidence: 0.85,
            });
        }
        
        Ok(risk_factors)
    }
    
    /// Generate recommendations based on analysis
    async fn generate_recommendations(&self, risk_factors: &[RiskFactor], stability: &StabilityMetrics) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        for risk_factor in risk_factors {
            match risk_factor.factor_type {
                RiskFactorType::PoorSignalQuality => {
                    recommendations.push("Consider handover to stronger cell".to_string());
                    recommendations.push("Increase transmission power if possible".to_string());
                }
                RiskFactorType::HighInterference => {
                    recommendations.push("Investigate interference sources".to_string());
                    recommendations.push("Optimize antenna configuration".to_string());
                }
                RiskFactorType::CellCongestion => {
                    recommendations.push("Implement load balancing".to_string());
                    recommendations.push("Consider carrier aggregation".to_string());
                }
                RiskFactorType::ConfigurationProblem => {
                    recommendations.push("Check 5G network configuration".to_string());
                    recommendations.push("Verify inter-RAT parameters".to_string());
                }
                _ => {}
            }
        }
        
        // Add stability-based recommendations
        if stability.connection_stability < 0.7 {
            recommendations.push("Monitor signal stability closely".to_string());
        }
        
        if stability.handover_frequency > 0.1 {
            recommendations.push("Optimize handover parameters".to_string());
        }
        
        Ok(recommendations)
    }
    
    /// Normalize RSRP score to 0-1 range
    fn normalize_rsrp_score(&self, rsrp: f64) -> f64 {
        // RSRP typically ranges from -140 to -44 dBm
        ((rsrp + 140.0) / 96.0).clamp(0.0, 1.0)
    }
    
    /// Normalize SINR score to 0-1 range
    fn normalize_sinr_score(&self, sinr: f64) -> f64 {
        // SINR typically ranges from -20 to 30 dB
        ((sinr + 20.0) / 50.0).clamp(0.0, 1.0)
    }
    
    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = self.calculate_mean(values);
        let variance = values.iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance
    }
    
    /// Calculate mean of a vector
    fn calculate_mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[async_trait]
impl SignalAnalyzer for AdvancedSignalAnalyzer {
    async fn analyze_signal_quality(&self, data: &[SignalQuality]) -> Result<SignalQualityFeatures> {
        if data.is_empty() {
            return Err(RanError::DataError("No signal quality data provided".to_string()));
        }
        
        let latest = data.last().unwrap();
        
        // Convert SignalQuality to SignalQualityMeasurement
        let measurements: Vec<SignalQualityMeasurement> = data.iter().map(|sq| {
            SignalQualityMeasurement {
                timestamp: sq.timestamp,
                ue_id: sq.ue_id.clone(),
                cell_id: CellId("default".to_string()), // Would need to be provided
                lte_rsrp: sq.lte_rsrp,
                lte_sinr: sq.lte_sinr,
                nr_ssb_rsrp: sq.nr_ssb_rsrp,
                nr_sinr: None,
                endc_success_rate: sq.endc_setup_success_rate,
                throughput_mbps: 50.0, // Would need to be provided
                rtt_ms: 20.0, // Would need to be provided
                packet_loss_rate: 0.01, // Would need to be provided
                handover_count: 0, // Would need to be provided
                location: None,
            }
        }).collect();
        
        // Calculate features
        let signal_stability_score = self.calculate_stability_score_from_measurements(&measurements).await?;
        let handover_likelihood = self.calculate_handover_likelihood_from_measurements(&measurements).await?;
        let cell_congestion_factor = 0.5; // Would need to be calculated from cell data
        let historical_failure_rate = 1.0 - latest.endc_setup_success_rate;
        
        Ok(SignalQualityFeatures {
            ue_id: latest.ue_id.clone(),
            timestamp: latest.timestamp,
            lte_rsrp: latest.lte_rsrp,
            lte_sinr: latest.lte_sinr,
            nr_ssb_rsrp: latest.nr_ssb_rsrp,
            endc_success_rate: latest.endc_setup_success_rate,
            signal_stability_score,
            handover_likelihood,
            cell_congestion_factor,
            historical_failure_rate,
            hour_of_day: latest.timestamp.hour(),
            day_of_week: latest.timestamp.weekday().num_days_from_monday(),
            is_peak_hour: self.is_peak_hour(latest.timestamp.hour()),
            rsrp_trend_5min: 0.0, // Would need historical data
            sinr_trend_5min: 0.0, // Would need historical data
            success_rate_trend_15min: 0.0, // Would need historical data
            rsrp_variance_1hour: self.calculate_variance(&measurements.iter().map(|m| m.lte_rsrp).collect::<Vec<_>>()),
            sinr_variance_1hour: self.calculate_variance(&measurements.iter().map(|m| m.lte_sinr).collect::<Vec<_>>()),
            success_rate_mean_1hour: self.calculate_mean(&measurements.iter().map(|m| m.endc_success_rate).collect::<Vec<_>>()),
        })
    }
    
    async fn detect_degradation(&self, data: &[SignalQuality]) -> Result<Vec<String>> {
        let mut degradations = Vec::new();
        
        if data.is_empty() {
            return Ok(degradations);
        }
        
        let latest = data.last().unwrap();
        
        // Check for various degradation patterns
        if latest.lte_rsrp < -110.0 {
            degradations.push("LTE RSRP degradation detected".to_string());
        }
        
        if latest.lte_sinr < 0.0 {
            degradations.push("LTE SINR degradation detected".to_string());
        }
        
        if latest.endc_setup_success_rate < 0.8 {
            degradations.push("ENDC setup success rate degradation detected".to_string());
        }
        
        // Check for trends if we have enough data
        if data.len() >= 5 {
            let recent_rsrp: Vec<f64> = data.iter().rev().take(5).map(|sq| sq.lte_rsrp).collect();
            if self.is_degrading_trend(&recent_rsrp) {
                degradations.push("RSRP degrading trend detected".to_string());
            }
            
            let recent_sinr: Vec<f64> = data.iter().rev().take(5).map(|sq| sq.lte_sinr).collect();
            if self.is_degrading_trend(&recent_sinr) {
                degradations.push("SINR degrading trend detected".to_string());
            }
        }
        
        Ok(degradations)
    }
    
    async fn calculate_stability_score(&self, data: &[SignalQuality]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        
        let rsrp_values: Vec<f64> = data.iter().map(|sq| sq.lte_rsrp).collect();
        let sinr_values: Vec<f64> = data.iter().map(|sq| sq.lte_sinr).collect();
        
        let rsrp_variance = self.calculate_variance(&rsrp_values);
        let sinr_variance = self.calculate_variance(&sinr_values);
        
        // Lower variance means higher stability
        let rsrp_stability = 1.0 - (rsrp_variance / 100.0).min(1.0);
        let sinr_stability = 1.0 - (sinr_variance / 100.0).min(1.0);
        
        Ok((rsrp_stability + sinr_stability) / 2.0)
    }
}

impl AdvancedSignalAnalyzer {
    async fn calculate_stability_score_from_measurements(&self, measurements: &[SignalQualityMeasurement]) -> Result<f64> {
        let rsrp_values: Vec<f64> = measurements.iter().map(|m| m.lte_rsrp).collect();
        let sinr_values: Vec<f64> = measurements.iter().map(|m| m.lte_sinr).collect();
        
        let rsrp_variance = self.calculate_variance(&rsrp_values);
        let sinr_variance = self.calculate_variance(&sinr_values);
        
        let rsrp_stability = 1.0 - (rsrp_variance / 100.0).min(1.0);
        let sinr_stability = 1.0 - (sinr_variance / 100.0).min(1.0);
        
        Ok((rsrp_stability + sinr_stability) / 2.0)
    }
    
    async fn calculate_handover_likelihood_from_measurements(&self, measurements: &[SignalQualityMeasurement]) -> Result<f64> {
        if measurements.is_empty() {
            return Ok(0.0);
        }
        
        let latest = measurements.last().unwrap();
        let signal_factor = if latest.lte_rsrp < -110.0 { 0.7 } else { 0.3 };
        let handover_factor = (latest.handover_count as f64 / 10.0).min(1.0);
        
        Ok((signal_factor + handover_factor) / 2.0)
    }
    
    fn is_peak_hour(&self, hour: u32) -> bool {
        (hour >= 8 && hour <= 10) || (hour >= 18 && hour <= 20)
    }
    
    fn is_degrading_trend(&self, values: &[f64]) -> bool {
        if values.len() < 3 {
            return false;
        }
        
        let mut decreasing_count = 0;
        for i in 1..values.len() {
            if values[i] < values[i-1] {
                decreasing_count += 1;
            }
        }
        
        // Consider it degrading if more than 60% of values are decreasing
        decreasing_count as f64 / (values.len() - 1) as f64 > 0.6
    }
}