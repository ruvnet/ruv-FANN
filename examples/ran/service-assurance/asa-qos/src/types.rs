use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// VoLTE quality metrics used for jitter forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoLteMetrics {
    pub cell_id: String,
    pub timestamp: DateTime<Utc>,
    pub prb_utilization_dl: f64,
    pub active_volte_users: u32,
    pub competing_gbr_traffic_mbps: f64,
    pub current_jitter_ms: f64,
    pub packet_loss_rate: f64,
    pub delay_ms: f64,
}

/// Time-series data point for forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub features: Vec<f64>,
}

/// Jitter forecast with confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterForecast {
    pub timestamp: DateTime<Utc>,
    pub predicted_jitter_ms: f64,
    pub confidence: f64,
    pub prediction_interval_lower: f64,
    pub prediction_interval_upper: f64,
}

/// Quality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAnalysis {
    pub cell_id: String,
    pub baseline_jitter_ms: f64,
    pub peak_jitter_ms: f64,
    pub jitter_variability: f64,
    pub contributing_factors: Vec<String>,
    pub quality_impact_score: f64,
    pub quality_trend: QualityTrend,
}

/// Quality trend enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityTrend {
    Improving,
    Stable,
    Degrading,
}

/// Quality recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub expected_improvement_ms: f64,
    pub priority: Priority,
    pub implementation: String,
}

/// Recommendation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    TrafficShaping,
    ResourceAllocation,
    PriorityAdjustment,
}

/// Priority enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Model training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub input_features: Vec<String>,
    pub forecast_horizon_minutes: u32,
    pub training_window_hours: u32,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: u32,
}

/// Model type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Lstm,
    Gru,
    Transformer,
    Arima,
    LinearRegression,
    RandomForest,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_id: String,
    pub accuracy_10ms: f64,
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
    pub r2_score: f64,
    pub training_time_ms: u64,
    pub inference_time_ms: u64,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub jitter_threshold_ms: f64,
    pub quality_degradation_threshold: f64,
    pub prediction_confidence_threshold: f64,
    pub alert_cooldown_minutes: u32,
}

/// Quality alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlert {
    pub alert_id: String,
    pub cell_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub predicted_jitter_ms: f64,
    pub confidence: f64,
    pub recommendations: Vec<QualityRecommendation>,
}

/// Alert type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    JitterThresholdExceeded,
    QualityDegradation,
    PredictionConfidenceLow,
    ServiceAssuranceRisk,
}

/// Alert severity enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Dashboard data for service assurance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub cell_id: String,
    pub current_metrics: VoLteMetrics,
    pub forecast: Vec<JitterForecast>,
    pub analysis: QualityAnalysis,
    pub recommendations: Vec<QualityRecommendation>,
    pub alerts: Vec<QualityAlert>,
    pub model_performance: ModelMetrics,
    pub last_updated: DateTime<Utc>,
}