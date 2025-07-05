//! Core data types and structures for capacity planning

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a capacity utilization measurement at a specific point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityDataPoint {
    /// Unique identifier for the measurement
    pub id: Uuid,
    /// Timestamp of the measurement
    pub timestamp: DateTime<Utc>,
    /// Cell identifier
    pub cell_id: String,
    /// PRB utilization percentage (0.0 to 1.0)
    pub prb_utilization: f64,
    /// Total PRB capacity
    pub total_prb_capacity: u32,
    /// Used PRB count
    pub used_prb_count: u32,
    /// Number of active users
    pub active_users: u32,
    /// Throughput in Mbps
    pub throughput_mbps: f64,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for capacity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Block Error Rate (BLER)
    pub bler: f64,
    /// Reference Signal Received Power (RSRP) in dBm
    pub rsrp_dbm: f64,
    /// Reference Signal Received Quality (RSRQ) in dB
    pub rsrq_db: f64,
    /// Signal-to-Interference-plus-Noise Ratio (SINR) in dB
    pub sinr_db: f64,
    /// Packet loss rate
    pub packet_loss_rate: f64,
    /// Average latency in milliseconds
    pub latency_ms: f64,
}

/// Represents a capacity breach prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityBreachPrediction {
    /// Unique identifier for the prediction
    pub id: Uuid,
    /// Timestamp when prediction was made
    pub prediction_timestamp: DateTime<Utc>,
    /// Cell identifier
    pub cell_id: String,
    /// Threshold that will be breached (e.g., 0.8 for 80%)
    pub threshold: f64,
    /// Predicted date when breach will occur
    pub predicted_breach_date: DateTime<Utc>,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Months ahead prediction
    pub months_ahead: f64,
    /// Predicted utilization at breach
    pub predicted_utilization: f64,
    /// Model used for prediction
    pub model_name: String,
    /// Prediction accuracy estimate
    pub accuracy_estimate: f64,
}

/// Growth trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthTrendAnalysis {
    /// Cell identifier
    pub cell_id: String,
    /// Analysis timestamp
    pub analysis_timestamp: DateTime<Utc>,
    /// Monthly growth rate
    pub monthly_growth_rate: f64,
    /// Quarterly growth rate
    pub quarterly_growth_rate: f64,
    /// Annual growth rate
    pub annual_growth_rate: f64,
    /// Trend direction (Increasing, Decreasing, Stable)
    pub trend_direction: TrendDirection,
    /// Seasonal patterns detected
    pub seasonal_patterns: Vec<SeasonalPattern>,
    /// Volatility measure
    pub volatility: f64,
    /// R-squared value for trend fit
    pub r_squared: f64,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Seasonal pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Pattern type (Daily, Weekly, Monthly, Quarterly)
    pub pattern_type: SeasonalPatternType,
    /// Strength of the pattern (0.0 to 1.0)
    pub strength: f64,
    /// Phase offset
    pub phase_offset: f64,
    /// Amplitude
    pub amplitude: f64,
}

/// Types of seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalPatternType {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

/// Network expansion recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionRecommendation {
    /// Unique identifier
    pub id: Uuid,
    /// Cell identifier
    pub cell_id: String,
    /// Recommendation timestamp
    pub timestamp: DateTime<Utc>,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level (1-5, where 5 is highest)
    pub priority: u8,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Expected capacity increase
    pub expected_capacity_increase: f64,
    /// Timeline for implementation (months)
    pub implementation_timeline_months: f64,
    /// ROI estimate
    pub roi_estimate: f64,
    /// Risk assessment
    pub risk_level: RiskLevel,
    /// Detailed description
    pub description: String,
}

/// Types of expansion recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Add new cell site
    NewCellSite,
    /// Upgrade existing equipment
    EquipmentUpgrade,
    /// Add carriers/spectrum
    SpectrumExpansion,
    /// Improve antenna configuration
    AntennaOptimization,
    /// Deploy small cells
    SmallCellDeployment,
    /// Implement carrier aggregation
    CarrierAggregation,
    /// Network densification
    NetworkDensification,
}

/// Risk level assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Capacity forecasting model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Training parameters
    pub training_params: HashMap<String, serde_json::Value>,
    /// Forecast horizon in months
    pub forecast_horizon: usize,
    /// Minimum data points required
    pub min_data_points: usize,
    /// Model weight in ensemble
    pub ensemble_weight: f64,
}

/// Available model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LSTM,
    ARIMA,
    PolynomialRegression,
    ExponentialSmoothing,
    NeuralForecast,
    EnsembleModel,
}

/// Forecast result from a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Model name
    pub model_name: String,
    /// Forecast timestamp
    pub timestamp: DateTime<Utc>,
    /// Forecasted values
    pub forecasted_values: Vec<f64>,
    /// Prediction intervals (lower bound)
    pub prediction_intervals_lower: Vec<f64>,
    /// Prediction intervals (upper bound)
    pub prediction_intervals_upper: Vec<f64>,
    /// Confidence levels
    pub confidence_levels: Vec<f64>,
    /// Model metrics
    pub model_metrics: ModelMetrics,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// R-squared
    pub r_squared: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
}

/// Strategic investment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestmentAnalysis {
    /// Analysis identifier
    pub id: Uuid,
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Geographic region
    pub region: String,
    /// Total investment required
    pub total_investment: f64,
    /// Investment timeline
    pub investment_timeline: Vec<InvestmentPhase>,
    /// Expected ROI
    pub expected_roi: f64,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Benefits quantification
    pub benefits: InvestmentBenefits,
}

/// Investment phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestmentPhase {
    /// Phase name
    pub name: String,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: DateTime<Utc>,
    /// Investment amount
    pub amount: f64,
    /// Expected outcomes
    pub expected_outcomes: Vec<String>,
}

/// Risk assessment for investment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk score (0.0 to 1.0)
    pub overall_risk: f64,
    /// Technology risk
    pub technology_risk: f64,
    /// Market risk
    pub market_risk: f64,
    /// Regulatory risk
    pub regulatory_risk: f64,
    /// Financial risk
    pub financial_risk: f64,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Investment benefits quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestmentBenefits {
    /// Capacity increase (%)
    pub capacity_increase_percent: f64,
    /// Quality improvement metrics
    pub quality_improvement: QualityImprovement,
    /// Revenue impact
    pub revenue_impact: f64,
    /// Cost savings
    pub cost_savings: f64,
    /// Customer satisfaction improvement
    pub customer_satisfaction_improvement: f64,
}

/// Quality improvement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityImprovement {
    /// Latency reduction (ms)
    pub latency_reduction_ms: f64,
    /// Throughput increase (Mbps)
    pub throughput_increase_mbps: f64,
    /// BLER improvement
    pub bler_improvement: f64,
    /// Coverage improvement (%)
    pub coverage_improvement_percent: f64,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            bler: 0.0,
            rsrp_dbm: -80.0,
            rsrq_db: -10.0,
            sinr_db: 15.0,
            packet_loss_rate: 0.0,
            latency_ms: 10.0,
        }
    }
}

impl CapacityDataPoint {
    /// Create a new capacity data point
    pub fn new(
        cell_id: String,
        prb_utilization: f64,
        total_prb_capacity: u32,
        used_prb_count: u32,
        active_users: u32,
        throughput_mbps: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            cell_id,
            prb_utilization,
            total_prb_capacity,
            used_prb_count,
            active_users,
            throughput_mbps,
            quality_metrics: QualityMetrics::default(),
        }
    }

    /// Check if this data point represents a capacity breach
    pub fn is_breach(&self, threshold: f64) -> bool {
        self.prb_utilization >= threshold
    }

    /// Calculate efficiency (throughput per PRB)
    pub fn efficiency(&self) -> f64 {
        if self.used_prb_count == 0 {
            0.0
        } else {
            self.throughput_mbps / self.used_prb_count as f64
        }
    }
}

impl GrowthTrendAnalysis {
    /// Determine if growth trend is concerning
    pub fn is_concerning(&self) -> bool {
        matches!(self.trend_direction, TrendDirection::Increasing | TrendDirection::Volatile)
            && self.monthly_growth_rate > 0.05 // 5% monthly growth
    }

    /// Get trend strength
    pub fn trend_strength(&self) -> f64 {
        self.r_squared
    }
}

impl ExpansionRecommendation {
    /// Check if recommendation is urgent
    pub fn is_urgent(&self) -> bool {
        self.priority >= 4 && self.implementation_timeline_months <= 6.0
    }

    /// Calculate cost-benefit ratio
    pub fn cost_benefit_ratio(&self) -> f64 {
        if self.estimated_cost == 0.0 {
            f64::INFINITY
        } else {
            self.expected_capacity_increase / self.estimated_cost
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_data_point_creation() {
        let dp = CapacityDataPoint::new(
            "CELL_001".to_string(),
            0.75,
            100,
            75,
            50,
            150.0,
        );
        
        assert_eq!(dp.cell_id, "CELL_001");
        assert_eq!(dp.prb_utilization, 0.75);
        assert!(!dp.is_breach(0.8));
        assert!(dp.is_breach(0.7));
        assert!(dp.efficiency() > 0.0);
    }

    #[test]
    fn test_growth_trend_analysis() {
        let analysis = GrowthTrendAnalysis {
            cell_id: "CELL_001".to_string(),
            analysis_timestamp: Utc::now(),
            monthly_growth_rate: 0.06,
            quarterly_growth_rate: 0.18,
            annual_growth_rate: 0.72,
            trend_direction: TrendDirection::Increasing,
            seasonal_patterns: vec![],
            volatility: 0.1,
            r_squared: 0.85,
        };
        
        assert!(analysis.is_concerning());
        assert_eq!(analysis.trend_strength(), 0.85);
    }

    #[test]
    fn test_expansion_recommendation() {
        let rec = ExpansionRecommendation {
            id: Uuid::new_v4(),
            cell_id: "CELL_001".to_string(),
            timestamp: Utc::now(),
            recommendation_type: RecommendationType::NewCellSite,
            priority: 5,
            estimated_cost: 100000.0,
            expected_capacity_increase: 1.5,
            implementation_timeline_months: 3.0,
            roi_estimate: 2.5,
            risk_level: RiskLevel::Medium,
            description: "Urgent capacity expansion needed".to_string(),
        };
        
        assert!(rec.is_urgent());
        assert_eq!(rec.cost_benefit_ratio(), 1.5 / 100000.0);
    }
}