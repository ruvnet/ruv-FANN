//! Capacity forecasting engine implementation

use crate::config::*;
use crate::error::*;
use crate::models::*;
use crate::types::*;
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Main capacity forecasting engine
pub struct CapacityForecaster {
    config: Config,
    predictor: CapacityCliffPredictor,
    historical_data: Arc<RwLock<HashMap<String, Vec<CapacityDataPoint>>>>,
    forecast_cache: Arc<RwLock<HashMap<String, CachedForecast>>>,
    metrics_collector: MetricsCollector,
}

/// Cached forecast results
#[derive(Debug, Clone)]
struct CachedForecast {
    forecast: ForecastResult,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
}

/// Metrics collector for forecasting performance
pub struct MetricsCollector {
    accuracy_history: Vec<AccuracyMetric>,
    prediction_latency: Vec<LatencyMetric>,
    model_performance: HashMap<String, ModelPerformanceMetric>,
}

/// Accuracy tracking metric
#[derive(Debug, Clone)]
pub struct AccuracyMetric {
    pub timestamp: DateTime<Utc>,
    pub cell_id: String,
    pub model_name: String,
    pub predicted_value: f64,
    pub actual_value: f64,
    pub error: f64,
    pub absolute_error: f64,
    pub percentage_error: f64,
}

/// Latency tracking metric
#[derive(Debug, Clone)]
pub struct LatencyMetric {
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub duration_ms: u64,
    pub success: bool,
}

/// Model performance tracking
#[derive(Debug, Clone)]
pub struct ModelPerformanceMetric {
    pub model_name: String,
    pub total_predictions: u64,
    pub successful_predictions: u64,
    pub average_accuracy: f64,
    pub average_latency_ms: f64,
    pub last_updated: DateTime<Utc>,
}

/// Configuration for forecasting operations
#[derive(Debug, Clone)]
pub struct ForecastConfig {
    pub horizon_months: usize,
    pub confidence_level: f64,
    pub include_confidence_intervals: bool,
    pub enable_caching: bool,
    pub cache_ttl_minutes: u64,
    pub min_data_points: usize,
    pub max_data_age_days: u64,
}

/// Forecasting request
#[derive(Debug, Clone)]
pub struct ForecastRequest {
    pub cell_id: String,
    pub forecast_type: ForecastType,
    pub config: ForecastConfig,
    pub additional_context: Option<HashMap<String, serde_json::Value>>,
}

/// Types of forecasts
#[derive(Debug, Clone)]
pub enum ForecastType {
    /// Predict capacity breach
    CapacityBreach { threshold: f64 },
    /// Forecast utilization values
    UtilizationForecast,
    /// Growth trend analysis
    GrowthTrend,
    /// Seasonal pattern analysis
    SeasonalAnalysis,
    /// Investment planning forecast
    InvestmentPlanning,
}

/// Comprehensive forecast response
#[derive(Debug, Clone)]
pub struct ForecastResponse {
    pub request_id: uuid::Uuid,
    pub cell_id: String,
    pub forecast_type: ForecastType,
    pub timestamp: DateTime<Utc>,
    pub results: ForecastResults,
    pub metadata: ForecastMetadata,
}

/// Forecast results union
#[derive(Debug, Clone)]
pub enum ForecastResults {
    CapacityBreach(CapacityBreachPrediction),
    Utilization(Vec<f64>),
    GrowthTrend(GrowthTrendAnalysis),
    Seasonal(Vec<SeasonalPattern>),
    Investment(InvestmentAnalysis),
}

/// Forecast metadata
#[derive(Debug, Clone)]
pub struct ForecastMetadata {
    pub model_used: String,
    pub confidence: f64,
    pub data_points_used: usize,
    pub forecast_horizon: usize,
    pub processing_time_ms: u64,
    pub data_quality_score: f64,
    pub warnings: Vec<String>,
}

impl CapacityForecaster {
    /// Create a new capacity forecaster
    pub fn new(config: Config) -> Self {
        let capacity_config = config.capacity_planning.clone();
        let predictor = CapacityCliffPredictor::new(capacity_config);
        
        Self {
            config,
            predictor,
            historical_data: Arc::new(RwLock::new(HashMap::new())),
            forecast_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics_collector: MetricsCollector::new(),
        }
    }

    /// Initialize the forecaster with models
    pub async fn initialize(&mut self) -> CapacityResult<()> {
        info!("Initializing capacity forecaster");

        // Initialize models based on configuration
        self.initialize_models().await?;

        info!("Capacity forecaster initialized successfully");
        Ok(())
    }

    /// Add historical data for a cell
    pub async fn add_historical_data(
        &self,
        cell_id: &str,
        data: Vec<CapacityDataPoint>,
    ) -> CapacityResult<()> {
        let mut historical_data = self.historical_data.write().await;
        
        // Validate and sort data
        let mut sorted_data = data;
        sorted_data.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        // Remove duplicates
        sorted_data.dedup_by(|a, b| a.timestamp == b.timestamp);
        
        // Quality checks
        let quality_score = self.calculate_data_quality(&sorted_data)?;
        if quality_score < self.config.capacity_planning.growth_trend.min_r_squared {
            warn!(
                "Low data quality for cell {}: {:.2}",
                cell_id, quality_score
            );
        }
        
        historical_data.insert(cell_id.to_string(), sorted_data);
        
        info!("Added historical data for cell {}", cell_id);
        Ok(())
    }

    /// Generate forecast based on request
    pub async fn forecast(&self, request: ForecastRequest) -> CapacityResult<ForecastResponse> {
        let start_time = std::time::Instant::now();
        let request_id = uuid::Uuid::new_v4();
        
        debug!("Processing forecast request {:?} for cell {}", request_id, request.cell_id);

        // Check cache first
        if request.config.enable_caching {
            if let Some(cached) = self.get_cached_forecast(&request.cell_id, &request.forecast_type).await? {
                debug!("Returning cached forecast for cell {}", request.cell_id);
                return Ok(self.build_response_from_cache(request_id, request, cached, start_time.elapsed().as_millis() as u64));
            }
        }

        // Get historical data
        let historical_data = self.historical_data.read().await;
        let data = historical_data.get(&request.cell_id)
            .ok_or_else(|| DataError::missing_field(format!("No data for cell {}", request.cell_id)))?;

        // Validate data requirements
        self.validate_data_requirements(data, &request.config)?;

        // Train models if needed
        if !self.predictor.models.is_empty() {
            for model in self.predictor.models.values() {
                if !model.is_trained() {
                    return Err(ModelError::not_trained(model.name().to_string()).into());
                }
            }
        }

        // Generate forecast based on type
        let results = match request.forecast_type {
            ForecastType::CapacityBreach { threshold } => {
                let prediction = self.predictor
                    .predict_capacity_breach(threshold, &request.cell_id)
                    .await?;
                ForecastResults::CapacityBreach(prediction)
            }
            ForecastType::UtilizationForecast => {
                let forecasts = self.predict_utilization(&request.cell_id, &request.config).await?;
                ForecastResults::Utilization(forecasts)
            }
            ForecastType::GrowthTrend => {
                let analysis = self.predictor.analyze_growth_trend(data).await?;
                ForecastResults::GrowthTrend(analysis)
            }
            ForecastType::SeasonalAnalysis => {
                let patterns = self.analyze_seasonal_patterns(data).await?;
                ForecastResults::Seasonal(patterns)
            }
            ForecastType::InvestmentPlanning => {
                let analysis = self.generate_investment_analysis(&request.cell_id, data).await?;
                ForecastResults::Investment(analysis)
            }
        };

        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Build response
        let response = ForecastResponse {
            request_id,
            cell_id: request.cell_id.clone(),
            forecast_type: request.forecast_type.clone(),
            timestamp: Utc::now(),
            results,
            metadata: ForecastMetadata {
                model_used: "ensemble".to_string(),
                confidence: 0.85, // TODO: Calculate actual confidence
                data_points_used: data.len(),
                forecast_horizon: request.config.horizon_months,
                processing_time_ms: processing_time,
                data_quality_score: self.calculate_data_quality(data)?,
                warnings: self.generate_warnings(data, &request.config),
            },
        };

        // Cache result if enabled
        if request.config.enable_caching {
            self.cache_forecast(&request.cell_id, &request.forecast_type, &response).await?;
        }

        // Record metrics
        self.record_latency_metric("forecast", processing_time, true).await;

        info!("Generated forecast for cell {} in {}ms", request.cell_id, processing_time);
        Ok(response)
    }

    /// Train all models with historical data
    pub async fn train_models(&mut self, cell_id: &str) -> CapacityResult<()> {
        let historical_data = self.historical_data.read().await;
        let data = historical_data.get(cell_id)
            .ok_or_else(|| DataError::missing_field(format!("No data for cell {}", cell_id)))?;

        if data.len() < self.config.capacity_planning.min_historical_data_points {
            return Err(DataError::insufficient_data(
                self.config.capacity_planning.min_historical_data_points,
                data.len(),
            ).into());
        }

        info!("Training models for cell {} with {} data points", cell_id, data.len());
        
        self.predictor.train_models(data).await?;
        
        info!("Models trained successfully for cell {}", cell_id);
        Ok(())
    }

    /// Get forecast accuracy metrics
    pub async fn get_accuracy_metrics(&self, cell_id: &str) -> CapacityResult<Vec<AccuracyMetric>> {
        let metrics = self.metrics_collector.accuracy_history
            .iter()
            .filter(|m| m.cell_id == cell_id)
            .cloned()
            .collect();
        
        Ok(metrics)
    }

    /// Get model performance summary
    pub async fn get_model_performance(&self) -> CapacityResult<HashMap<String, ModelPerformanceMetric>> {
        Ok(self.metrics_collector.model_performance.clone())
    }

    /// Validate model accuracy against actual data
    pub async fn validate_predictions(
        &mut self,
        cell_id: &str,
        actual_data: &[CapacityDataPoint],
    ) -> CapacityResult<()> {
        // Get previous predictions for this cell
        let cached_forecasts = self.forecast_cache.read().await;
        
        for actual in actual_data {
            // Find corresponding predictions
            // This is a simplified implementation - in practice you'd match by timestamp
            let predicted_value = 0.75; // Placeholder
            
            let accuracy_metric = AccuracyMetric {
                timestamp: Utc::now(),
                cell_id: cell_id.to_string(),
                model_name: "ensemble".to_string(),
                predicted_value,
                actual_value: actual.prb_utilization,
                error: predicted_value - actual.prb_utilization,
                absolute_error: (predicted_value - actual.prb_utilization).abs(),
                percentage_error: ((predicted_value - actual.prb_utilization) / actual.prb_utilization).abs() * 100.0,
            };
            
            self.metrics_collector.accuracy_history.push(accuracy_metric);
        }
        
        Ok(())
    }

    /// Initialize forecasting models
    async fn initialize_models(&mut self) -> CapacityResult<()> {
        // Initialize LSTM model
        let lstm_model = LSTMCapacityModel::new(
            "lstm".to_string(),
            self.config.models.lstm.clone(),
        );
        self.predictor.add_model(Box::new(lstm_model));

        // Initialize Neural Forecast model using ruv-FANN
        let neural_model = NeuralForecastModel::new(
            "neural_forecast".to_string(),
            self.config.models.neural_forecast.clone(),
        );
        self.predictor.add_model(Box::new(neural_model));

        info!("Initialized {} forecasting models", self.predictor.models.len());
        Ok(())
    }

    /// Predict utilization values
    async fn predict_utilization(
        &self,
        cell_id: &str,
        config: &ForecastConfig,
    ) -> CapacityResult<Vec<f64>> {
        // Use the best performing model for prediction
        let mut best_predictions = None;
        let mut best_confidence = 0.0;

        for (model_name, model) in &self.predictor.models {
            if !model.is_trained() {
                continue;
            }

            let predictions = model.predict(config.horizon_months).await?;
            let metrics = model.get_metrics()?;
            let confidence = 1.0 - (metrics.mape / 100.0);

            if confidence > best_confidence {
                best_confidence = confidence;
                best_predictions = Some(predictions);
            }
        }

        best_predictions.ok_or_else(|| {
            ForecastingError::ensemble_prediction_failed("No trained models available".to_string())
        }.into())
    }

    /// Analyze seasonal patterns
    async fn analyze_seasonal_patterns(
        &self,
        data: &[CapacityDataPoint],
    ) -> CapacityResult<Vec<SeasonalPattern>> {
        let values: Vec<f64> = data.iter().map(|d| d.prb_utilization).collect();
        let timestamps: Vec<DateTime<Utc>> = data.iter().map(|d| d.timestamp).collect();

        // Detect various seasonal patterns
        let mut patterns = Vec::new();

        // Monthly seasonality
        if data.len() >= 12 {
            let monthly_pattern = self.detect_monthly_pattern(&values, &timestamps)?;
            if monthly_pattern.strength > 0.1 {
                patterns.push(monthly_pattern);
            }
        }

        // Weekly seasonality (if we have daily data)
        if data.len() >= 7 {
            let weekly_pattern = self.detect_weekly_pattern(&values, &timestamps)?;
            if weekly_pattern.strength > 0.1 {
                patterns.push(weekly_pattern);
            }
        }

        Ok(patterns)
    }

    /// Generate investment analysis
    async fn generate_investment_analysis(
        &self,
        cell_id: &str,
        data: &[CapacityDataPoint],
    ) -> CapacityResult<InvestmentAnalysis> {
        // Predict capacity breaches
        let breach_predictions = vec![
            self.predictor.predict_capacity_breach(0.8, cell_id).await?,
        ];

        // Generate expansion recommendations
        let recommendations = self.predictor
            .generate_expansion_recommendations(&breach_predictions, cell_id)
            .await?;

        // Calculate investment requirements
        let total_investment: f64 = recommendations.iter()
            .map(|r| r.estimated_cost)
            .sum();

        let expected_roi: f64 = recommendations.iter()
            .map(|r| r.roi_estimate)
            .sum::<f64>() / recommendations.len() as f64;

        // Create investment phases
        let mut investment_phases = Vec::new();
        let mut start_date = Utc::now();

        for (i, rec) in recommendations.iter().enumerate() {
            let end_date = start_date + Duration::days((rec.implementation_timeline_months * 30.0) as i64);
            
            investment_phases.push(InvestmentPhase {
                name: format!("Phase {} - {}", i + 1, rec.description),
                start_date,
                end_date,
                amount: rec.estimated_cost,
                expected_outcomes: vec![
                    format!("Capacity increase: {:.1}x", rec.expected_capacity_increase),
                    format!("ROI: {:.1}x", rec.roi_estimate),
                ],
            });
            
            start_date = end_date;
        }

        Ok(InvestmentAnalysis {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            region: cell_id.to_string(),
            total_investment,
            investment_timeline: investment_phases,
            expected_roi,
            risk_assessment: RiskAssessment {
                overall_risk: 0.3,
                technology_risk: 0.2,
                market_risk: 0.3,
                regulatory_risk: 0.1,
                financial_risk: 0.4,
                mitigation_strategies: vec![
                    "Phased implementation to reduce risk".to_string(),
                    "Regular monitoring and adjustment".to_string(),
                ],
            },
            benefits: InvestmentBenefits {
                capacity_increase_percent: 150.0,
                quality_improvement: QualityImprovement {
                    latency_reduction_ms: 5.0,
                    throughput_increase_mbps: 100.0,
                    bler_improvement: 0.5,
                    coverage_improvement_percent: 10.0,
                },
                revenue_impact: total_investment * expected_roi,
                cost_savings: total_investment * 0.2,
                customer_satisfaction_improvement: 15.0,
            },
        })
    }

    /// Calculate data quality score
    fn calculate_data_quality(&self, data: &[CapacityDataPoint]) -> CapacityResult<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mut quality_score = 1.0;

        // Check for missing values (simplified)
        let missing_ratio = 0.0; // Would check for null/invalid values
        quality_score -= missing_ratio * 0.3;

        // Check for outliers
        let values: Vec<f64> = data.iter().map(|d| d.prb_utilization).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        
        let outlier_count = values.iter()
            .filter(|&&v| (v - mean).abs() > 3.0 * std_dev)
            .count();
        let outlier_ratio = outlier_count as f64 / values.len() as f64;
        quality_score -= outlier_ratio * 0.2;

        // Check data consistency
        let consistency_score = self.calculate_consistency_score(data);
        quality_score *= consistency_score;

        Ok(quality_score.max(0.0).min(1.0))
    }

    /// Calculate data consistency score
    fn calculate_consistency_score(&self, data: &[CapacityDataPoint]) -> f64 {
        if data.len() < 2 {
            return 1.0;
        }

        // Check timestamp consistency
        let mut irregular_intervals = 0;
        for i in 1..data.len() {
            let interval = data[i].timestamp - data[i-1].timestamp;
            if interval.num_days() < 25 || interval.num_days() > 35 {
                irregular_intervals += 1;
            }
        }

        let regularity_score = 1.0 - (irregular_intervals as f64 / (data.len() - 1) as f64);
        regularity_score.max(0.0)
    }

    /// Validate data requirements for forecasting
    fn validate_data_requirements(
        &self,
        data: &[CapacityDataPoint],
        config: &ForecastConfig,
    ) -> CapacityResult<()> {
        if data.len() < config.min_data_points {
            return Err(DataError::insufficient_data(config.min_data_points, data.len()).into());
        }

        // Check data age
        let max_age = Duration::days(config.max_data_age_days as i64);
        let oldest_allowed = Utc::now() - max_age;
        
        if let Some(oldest_data) = data.first() {
            if oldest_data.timestamp < oldest_allowed {
                return Err(DataError::range_error(
                    format!("Data too old: oldest data from {}", oldest_data.timestamp)
                ).into());
            }
        }

        Ok(())
    }

    /// Generate warnings based on data analysis
    fn generate_warnings(&self, data: &[CapacityDataPoint], config: &ForecastConfig) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check for data gaps
        if data.len() < config.horizon_months * 2 {
            warnings.push("Insufficient historical data for reliable long-term forecasting".to_string());
        }

        // Check for recent high utilization
        if let Some(latest) = data.last() {
            if latest.prb_utilization > 0.9 {
                warnings.push("Current utilization is very high (>90%)".to_string());
            }
        }

        // Check for rapid growth
        if data.len() >= 6 {
            let recent_growth = (data.last().unwrap().prb_utilization - 
                               data[data.len()-6].prb_utilization) / 6.0;
            if recent_growth > 0.05 {
                warnings.push("Rapid capacity growth detected in recent months".to_string());
            }
        }

        warnings
    }

    /// Detect monthly seasonal patterns
    fn detect_monthly_pattern(
        &self,
        values: &[f64],
        _timestamps: &[DateTime<Utc>],
    ) -> CapacityResult<SeasonalPattern> {
        // Simplified seasonal detection
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        Ok(SeasonalPattern {
            pattern_type: SeasonalPatternType::Monthly,
            strength: 0.25, // Would use proper seasonal decomposition
            phase_offset: 0.0,
            amplitude: variance.sqrt(),
        })
    }

    /// Detect weekly seasonal patterns
    fn detect_weekly_pattern(
        &self,
        values: &[f64],
        _timestamps: &[DateTime<Utc>],
    ) -> CapacityResult<SeasonalPattern> {
        // Simplified weekly pattern detection
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        Ok(SeasonalPattern {
            pattern_type: SeasonalPatternType::Weekly,
            strength: 0.15,
            phase_offset: 0.0,
            amplitude: variance.sqrt() * 0.5,
        })
    }

    /// Get cached forecast if available and valid
    async fn get_cached_forecast(
        &self,
        cell_id: &str,
        forecast_type: &ForecastType,
    ) -> CapacityResult<Option<CachedForecast>> {
        let cache = self.forecast_cache.read().await;
        let cache_key = format!("{}:{:?}", cell_id, forecast_type);
        
        if let Some(cached) = cache.get(&cache_key) {
            if cached.expires_at > Utc::now() {
                return Ok(Some(cached.clone()));
            }
        }
        
        Ok(None)
    }

    /// Cache forecast result
    async fn cache_forecast(
        &self,
        cell_id: &str,
        forecast_type: &ForecastType,
        response: &ForecastResponse,
    ) -> CapacityResult<()> {
        let mut cache = self.forecast_cache.write().await;
        let cache_key = format!("{}:{:?}", cell_id, forecast_type);
        
        let cached_forecast = CachedForecast {
            forecast: ForecastResult {
                model_name: response.metadata.model_used.clone(),
                timestamp: response.timestamp,
                forecasted_values: vec![], // Would extract from response.results
                prediction_intervals_lower: vec![],
                prediction_intervals_upper: vec![],
                confidence_levels: vec![response.metadata.confidence],
                model_metrics: ModelMetrics {
                    mae: 0.0,
                    mse: 0.0,
                    rmse: 0.0,
                    mape: 0.0,
                    r_squared: response.metadata.confidence,
                    aic: 0.0,
                    bic: 0.0,
                },
            },
            created_at: Utc::now(),
            expires_at: Utc::now() + Duration::minutes(30), // Default 30 min TTL
        };
        
        cache.insert(cache_key, cached_forecast);
        Ok(())
    }

    /// Build response from cached forecast
    fn build_response_from_cache(
        &self,
        request_id: uuid::Uuid,
        request: ForecastRequest,
        cached: CachedForecast,
        processing_time: u64,
    ) -> ForecastResponse {
        ForecastResponse {
            request_id,
            cell_id: request.cell_id,
            forecast_type: request.forecast_type,
            timestamp: Utc::now(),
            results: ForecastResults::Utilization(cached.forecast.forecasted_values), // Simplified
            metadata: ForecastMetadata {
                model_used: cached.forecast.model_name,
                confidence: cached.forecast.model_metrics.r_squared,
                data_points_used: 0, // Would track in cache
                forecast_horizon: request.config.horizon_months,
                processing_time_ms: processing_time,
                data_quality_score: 1.0, // Would track in cache
                warnings: vec!["Using cached results".to_string()],
            },
        }
    }

    /// Record latency metric
    async fn record_latency_metric(&self, operation: &str, duration_ms: u64, success: bool) {
        // In a real implementation, this would be thread-safe
        debug!(
            "Recorded latency metric: {} took {}ms (success: {})",
            operation, duration_ms, success
        );
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            accuracy_history: Vec::new(),
            prediction_latency: Vec::new(),
            model_performance: HashMap::new(),
        }
    }
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            horizon_months: 12,
            confidence_level: 0.95,
            include_confidence_intervals: true,
            enable_caching: true,
            cache_ttl_minutes: 30,
            min_data_points: 12,
            max_data_age_days: 365,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Config {
        Config::default()
    }

    fn create_test_data() -> Vec<CapacityDataPoint> {
        let mut data = Vec::new();
        let base_time = Utc::now() - Duration::days(365);
        
        for i in 0..24 {
            let utilization = 0.5 + (i as f64) * 0.01 + (i as f64 * 0.1).sin() * 0.05;
            data.push(CapacityDataPoint {
                id: uuid::Uuid::new_v4(),
                timestamp: base_time + Duration::days(i * 30),
                cell_id: "TEST_CELL".to_string(),
                prb_utilization: utilization,
                total_prb_capacity: 100,
                used_prb_count: (utilization * 100.0) as u32,
                active_users: 50,
                throughput_mbps: 100.0,
                quality_metrics: QualityMetrics::default(),
            });
        }
        
        data
    }

    #[tokio::test]
    async fn test_forecaster_creation() {
        let config = create_test_config();
        let forecaster = CapacityForecaster::new(config);
        
        assert!(forecaster.historical_data.read().await.is_empty());
        assert!(forecaster.forecast_cache.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_add_historical_data() {
        let config = create_test_config();
        let forecaster = CapacityForecaster::new(config);
        let test_data = create_test_data();
        
        let result = forecaster.add_historical_data("TEST_CELL", test_data.clone()).await;
        assert!(result.is_ok());
        
        let historical_data = forecaster.historical_data.read().await;
        assert!(historical_data.contains_key("TEST_CELL"));
        assert_eq!(historical_data["TEST_CELL"].len(), test_data.len());
    }

    #[tokio::test]
    async fn test_data_quality_calculation() {
        let config = create_test_config();
        let forecaster = CapacityForecaster::new(config);
        let test_data = create_test_data();
        
        let quality_score = forecaster.calculate_data_quality(&test_data).unwrap();
        assert!(quality_score > 0.0);
        assert!(quality_score <= 1.0);
    }

    #[tokio::test]
    async fn test_forecast_config_default() {
        let config = ForecastConfig::default();
        
        assert_eq!(config.horizon_months, 12);
        assert_eq!(config.confidence_level, 0.95);
        assert!(config.include_confidence_intervals);
        assert!(config.enable_caching);
        assert_eq!(config.min_data_points, 12);
    }

    #[test]
    fn test_metrics_collector_creation() {
        let metrics = MetricsCollector::new();
        
        assert!(metrics.accuracy_history.is_empty());
        assert!(metrics.prediction_latency.is_empty());
        assert!(metrics.model_performance.is_empty());
    }
}