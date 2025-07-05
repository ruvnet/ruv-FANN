use crate::config::{Config, ForecastingConfig};
use crate::error::{Error, Result};
use crate::models::{ForecastingModel, ModelFactory, ensemble::EnsembleModel};
use crate::types::*;
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use tracing::{info, warn, error};

/// Main jitter forecasting engine
pub struct JitterForecaster {
    config: ForecastingConfig,
    primary_model: Option<Box<dyn ForecastingModel>>,
    ensemble_model: Option<EnsembleModel>,
    feature_processor: FeatureProcessor,
    alert_engine: AlertEngine,
    model_performance_tracker: ModelPerformanceTracker,
}

impl JitterForecaster {
    pub fn new(config: Config) -> Self {
        Self {
            config: config.forecasting,
            primary_model: None,
            ensemble_model: None,
            feature_processor: FeatureProcessor::new(config.forecasting.feature_engineering),
            alert_engine: AlertEngine::new(config.alerts),
            model_performance_tracker: ModelPerformanceTracker::new(),
        }
    }
    
    /// Initialize the forecaster with models
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing VoLTE Jitter Forecaster");
        
        // Create primary model
        self.primary_model = Some(ModelFactory::create_model(ModelType::Lstm)?);
        
        // Create ensemble model if enabled
        if self.config.model_ensemble {
            let mut ensemble = EnsembleModel::new();
            
            for (model_name, weight) in &self.config.model_weights {
                let model_type = match model_name.as_str() {
                    "lstm" => ModelType::Lstm,
                    "gru" => ModelType::Gru,
                    "transformer" => ModelType::Transformer,
                    "arima" => ModelType::Arima,
                    "linear" => ModelType::LinearRegression,
                    "random_forest" => ModelType::RandomForest,
                    _ => continue,
                };
                
                match ModelFactory::create_model(model_type) {
                    Ok(model) => {
                        ensemble.add_model(model_name.clone(), model, *weight);
                        info!("Added {} model to ensemble with weight {}", model_name, weight);
                    }
                    Err(e) => {
                        warn!("Failed to create {} model: {}", model_name, e);
                    }
                }
            }
            
            self.ensemble_model = Some(ensemble);
        }
        
        info!("Jitter Forecaster initialized successfully");
        Ok(())
    }
    
    /// Train the forecasting models with historical data
    pub async fn train(&mut self, historical_data: &[VoLteMetrics]) -> Result<()> {
        if historical_data.len() < self.config.min_historical_data_points {
            return Err(Error::InsufficientData(
                format!("Need at least {} historical data points", 
                       self.config.min_historical_data_points)
            ));
        }
        
        info!("Starting model training with {} data points", historical_data.len());
        
        // Convert to time series format and engineer features
        let time_series_data = self.feature_processor.process_metrics(historical_data)?;
        
        // Train primary model
        if let Some(ref mut model) = self.primary_model {
            info!("Training primary model");
            model.train(&time_series_data).await?;
            
            if let Some(metrics) = model.get_metrics() {
                info!("Primary model training completed. Accuracy: {:.2}%", 
                     metrics.accuracy_10ms * 100.0);
                self.model_performance_tracker.update_metrics("primary", metrics);
            }
        }
        
        // Train ensemble model
        if let Some(ref mut ensemble) = self.ensemble_model {
            info!("Training ensemble model");
            ensemble.train(&time_series_data).await?;
            
            if let Some(metrics) = ensemble.get_metrics() {
                info!("Ensemble model training completed. Accuracy: {:.2}%", 
                     metrics.accuracy_10ms * 100.0);
                self.model_performance_tracker.update_metrics("ensemble", metrics);
            }
        }
        
        Ok(())
    }
    
    /// Generate jitter forecast for a specific cell
    pub async fn forecast_jitter(
        &self,
        cell_id: &str,
        current_metrics: &VoLteMetrics,
        forecast_horizon_minutes: Option<u32>,
    ) -> Result<Vec<JitterForecast>> {
        let horizon = forecast_horizon_minutes
            .unwrap_or(self.config.default_forecast_horizon_minutes)
            .min(self.config.max_forecast_horizon_minutes);
        
        info!("Generating jitter forecast for cell {} with horizon {}min", 
             cell_id, horizon);
        
        // Convert current metrics to time series format
        let time_series_point = self.feature_processor.convert_metrics_to_time_series(current_metrics)?;
        let input_data = vec![time_series_point];
        
        // Get predictions from available models
        let mut forecasts = Vec::new();
        
        // Use ensemble model if available and trained
        if let Some(ref ensemble) = self.ensemble_model {
            if ensemble.is_trained() {
                match ensemble.predict(&input_data, horizon as usize).await {
                    Ok(ensemble_forecasts) => {
                        forecasts = ensemble_forecasts;
                        info!("Using ensemble model predictions");
                    }
                    Err(e) => {
                        warn!("Ensemble prediction failed: {}", e);
                    }
                }
            }
        }
        
        // Fall back to primary model if ensemble failed or unavailable
        if forecasts.is_empty() {
            if let Some(ref model) = self.primary_model {
                if model.is_trained() {
                    forecasts = model.predict(&input_data, horizon as usize).await?;
                    info!("Using primary model predictions");
                } else {
                    return Err(Error::Model("No trained models available".to_string()));
                }
            } else {
                return Err(Error::Model("No models initialized".to_string()));
            }
        }
        
        // Post-process forecasts
        self.post_process_forecasts(&mut forecasts, current_metrics)?;
        
        // Check for alerts
        self.check_and_generate_alerts(cell_id, &forecasts, current_metrics)?;
        
        Ok(forecasts)
    }
    
    /// Generate quality analysis for a cell
    pub async fn analyze_quality(
        &self,
        cell_id: &str,
        recent_metrics: &[VoLteMetrics],
    ) -> Result<QualityAnalysis> {
        if recent_metrics.is_empty() {
            return Err(Error::InsufficientData("No metrics provided for analysis".to_string()));
        }
        
        let jitter_values: Vec<f64> = recent_metrics.iter().map(|m| m.current_jitter_ms).collect();
        
        let baseline_jitter = jitter_values.iter().sum::<f64>() / jitter_values.len() as f64;
        let peak_jitter = jitter_values.iter().fold(0.0, |a, &b| a.max(b));
        
        // Calculate jitter variability (standard deviation)
        let variance = jitter_values.iter()
            .map(|&x| (x - baseline_jitter).powi(2))
            .sum::<f64>() / jitter_values.len() as f64;
        let jitter_variability = variance.sqrt();
        
        // Analyze contributing factors
        let contributing_factors = self.analyze_contributing_factors(recent_metrics);
        
        // Calculate quality impact score (0-1, where 1 is worst impact)
        let quality_impact_score = (peak_jitter / 50.0).min(1.0) + 
                                  (jitter_variability / 20.0).min(0.5);
        
        // Determine quality trend
        let quality_trend = self.determine_quality_trend(&jitter_values);
        
        Ok(QualityAnalysis {
            cell_id: cell_id.to_string(),
            baseline_jitter_ms: baseline_jitter,
            peak_jitter_ms: peak_jitter,
            jitter_variability,
            contributing_factors,
            quality_impact_score,
            quality_trend,
        })
    }
    
    /// Generate quality recommendations based on analysis
    pub async fn generate_recommendations(
        &self,
        quality_analysis: &QualityAnalysis,
        current_metrics: &VoLteMetrics,
    ) -> Result<Vec<QualityRecommendation>> {
        let mut recommendations = Vec::new();
        
        // High jitter recommendations
        if quality_analysis.peak_jitter_ms > 20.0 {
            recommendations.push(QualityRecommendation {
                recommendation_type: RecommendationType::TrafficShaping,
                description: "Implement traffic shaping to prioritize VoLTE traffic".to_string(),
                expected_improvement_ms: quality_analysis.peak_jitter_ms * 0.3,
                priority: Priority::High,
                implementation: "Configure QCI-1 traffic prioritization in packet scheduler".to_string(),
            });
        }
        
        // High utilization recommendations
        if current_metrics.prb_utilization_dl > 0.8 {
            recommendations.push(QualityRecommendation {
                recommendation_type: RecommendationType::ResourceAllocation,
                description: "Increase PRB allocation for VoLTE bearers".to_string(),
                expected_improvement_ms: 5.0,
                priority: Priority::Medium,
                implementation: "Adjust PRB allocation algorithm to reserve resources for VoLTE".to_string(),
            });
        }
        
        // High variability recommendations
        if quality_analysis.jitter_variability > 10.0 {
            recommendations.push(QualityRecommendation {
                recommendation_type: RecommendationType::PriorityAdjustment,
                description: "Stabilize jitter through enhanced packet scheduling".to_string(),
                expected_improvement_ms: quality_analysis.jitter_variability * 0.4,
                priority: Priority::Medium,
                implementation: "Enable jitter buffer optimization and implement stricter QoS policies".to_string(),
            });
        }
        
        Ok(recommendations)
    }
    
    /// Get model performance metrics
    pub fn get_model_metrics(&self) -> HashMap<String, ModelMetrics> {
        self.model_performance_tracker.get_all_metrics()
    }
    
    /// Update model with new data (online learning)
    pub async fn update_models(&mut self, new_data: &[VoLteMetrics]) -> Result<()> {
        info!("Updating models with {} new data points", new_data.len());
        
        let time_series_data = self.feature_processor.process_metrics(new_data)?;
        
        // For now, we'll retrain models periodically
        // In a production system, this could implement incremental learning
        
        Ok(())
    }
    
    // Private helper methods
    
    fn post_process_forecasts(
        &self,
        forecasts: &mut [JitterForecast],
        _current_metrics: &VoLteMetrics,
    ) -> Result<()> {
        for forecast in forecasts.iter_mut() {
            // Ensure non-negative jitter values
            forecast.predicted_jitter_ms = forecast.predicted_jitter_ms.max(0.0);
            forecast.prediction_interval_lower = forecast.prediction_interval_lower.max(0.0);
            
            // Apply reasonable upper bounds
            forecast.predicted_jitter_ms = forecast.predicted_jitter_ms.min(200.0);
            forecast.prediction_interval_upper = forecast.prediction_interval_upper.min(300.0);
            
            // Ensure confidence is in valid range
            forecast.confidence = forecast.confidence.max(0.0).min(1.0);
        }
        
        Ok(())
    }
    
    fn check_and_generate_alerts(
        &self,
        cell_id: &str,
        forecasts: &[JitterForecast],
        current_metrics: &VoLteMetrics,
    ) -> Result<()> {
        for forecast in forecasts {
            let alerts = self.alert_engine.check_forecast_alerts(
                cell_id,
                forecast,
                current_metrics,
            );
            
            for alert in alerts {
                // In a real system, this would send alerts to monitoring systems
                warn!("Quality alert generated: {:?}", alert);
            }
        }
        
        Ok(())
    }
    
    fn analyze_contributing_factors(&self, metrics: &[VoLteMetrics]) -> Vec<String> {
        let mut factors = Vec::new();
        
        // Check utilization patterns
        let avg_utilization = metrics.iter().map(|m| m.prb_utilization_dl).sum::<f64>() 
                             / metrics.len() as f64;
        if avg_utilization > 0.8 {
            factors.push("High PRB utilization".to_string());
        }
        
        // Check user count patterns
        let avg_users = metrics.iter().map(|m| m.active_volte_users).sum::<u32>() 
                       / metrics.len() as u32;
        if avg_users > 50 {
            factors.push("High VoLTE user count".to_string());
        }
        
        // Check competing traffic
        let avg_competing_traffic = metrics.iter().map(|m| m.competing_gbr_traffic_mbps).sum::<f64>() 
                                   / metrics.len() as f64;
        if avg_competing_traffic > 100.0 {
            factors.push("High competing GBR traffic".to_string());
        }
        
        // Check packet loss correlation
        let avg_packet_loss = metrics.iter().map(|m| m.packet_loss_rate).sum::<f64>() 
                             / metrics.len() as f64;
        if avg_packet_loss > 0.01 {
            factors.push("Elevated packet loss rate".to_string());
        }
        
        factors
    }
    
    fn determine_quality_trend(&self, jitter_values: &[f64]) -> QualityTrend {
        if jitter_values.len() < 2 {
            return QualityTrend::Stable;
        }
        
        // Simple trend analysis based on first and last values
        let first_half_avg = jitter_values[..jitter_values.len()/2].iter().sum::<f64>() 
                            / (jitter_values.len()/2) as f64;
        let second_half_avg = jitter_values[jitter_values.len()/2..].iter().sum::<f64>() 
                             / (jitter_values.len() - jitter_values.len()/2) as f64;
        
        let trend_threshold = 2.0; // 2ms threshold for trend detection
        
        if second_half_avg > first_half_avg + trend_threshold {
            QualityTrend::Degrading
        } else if first_half_avg > second_half_avg + trend_threshold {
            QualityTrend::Improving
        } else {
            QualityTrend::Stable
        }
    }
}

/// Feature processing engine for time series data
pub struct FeatureProcessor {
    config: crate::config::FeatureEngineeringConfig,
}

impl FeatureProcessor {
    pub fn new(config: crate::config::FeatureEngineeringConfig) -> Self {
        Self { config }
    }
    
    pub fn process_metrics(&self, metrics: &[VoLteMetrics]) -> Result<Vec<TimeSeriesPoint>> {
        let mut time_series = Vec::new();
        
        for metric in metrics {
            let point = self.convert_metrics_to_time_series(metric)?;
            time_series.push(point);
        }
        
        // Apply feature engineering
        if self.config.enable_lag_features {
            self.add_lag_features(&mut time_series);
        }
        
        if self.config.enable_moving_averages {
            self.add_moving_averages(&mut time_series);
        }
        
        if self.config.enable_seasonal_decomposition {
            self.add_seasonal_features(&mut time_series);
        }
        
        if self.config.enable_fourier_features {
            self.add_fourier_features(&mut time_series);
        }
        
        Ok(time_series)
    }
    
    pub fn convert_metrics_to_time_series(&self, metrics: &VoLteMetrics) -> Result<TimeSeriesPoint> {
        let features = vec![
            metrics.prb_utilization_dl,
            metrics.active_volte_users as f64,
            metrics.competing_gbr_traffic_mbps,
            metrics.packet_loss_rate * 100.0, // Convert to percentage
            metrics.delay_ms,
            (metrics.timestamp.timestamp() % 86400) as f64 / 86400.0, // Time of day feature
        ];
        
        Ok(TimeSeriesPoint {
            timestamp: metrics.timestamp,
            value: metrics.current_jitter_ms,
            features,
        })
    }
    
    // Feature engineering methods (simplified implementations)
    
    fn add_lag_features(&self, _time_series: &mut Vec<TimeSeriesPoint>) {
        // TODO: Implement lag features
    }
    
    fn add_moving_averages(&self, _time_series: &mut Vec<TimeSeriesPoint>) {
        // TODO: Implement moving averages
    }
    
    fn add_seasonal_features(&self, _time_series: &mut Vec<TimeSeriesPoint>) {
        // TODO: Implement seasonal decomposition
    }
    
    fn add_fourier_features(&self, _time_series: &mut Vec<TimeSeriesPoint>) {
        // TODO: Implement Fourier features
    }
}

/// Alert generation engine
pub struct AlertEngine {
    config: AlertConfig,
    recent_alerts: HashMap<String, DateTime<Utc>>,
}

impl AlertEngine {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            recent_alerts: HashMap::new(),
        }
    }
    
    pub fn check_forecast_alerts(
        &mut self,
        cell_id: &str,
        forecast: &JitterForecast,
        current_metrics: &VoLteMetrics,
    ) -> Vec<QualityAlert> {
        let mut alerts = Vec::new();
        
        // Check if we're in cooldown period
        let alert_key = format!("{}_{}", cell_id, "jitter_threshold");
        if let Some(&last_alert_time) = self.recent_alerts.get(&alert_key) {
            let cooldown_duration = Duration::minutes(self.config.alert_cooldown_minutes as i64);
            if Utc::now() - last_alert_time < cooldown_duration {
                return alerts; // Still in cooldown
            }
        }
        
        // Jitter threshold check
        if forecast.predicted_jitter_ms > self.config.jitter_threshold_ms {
            let alert = QualityAlert {
                alert_id: uuid::Uuid::new_v4().to_string(),
                cell_id: cell_id.to_string(),
                alert_type: AlertType::JitterThresholdExceeded,
                severity: if forecast.predicted_jitter_ms > self.config.jitter_threshold_ms * 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                message: format!(
                    "Predicted jitter {:.1}ms exceeds threshold {:.1}ms",
                    forecast.predicted_jitter_ms,
                    self.config.jitter_threshold_ms
                ),
                timestamp: Utc::now(),
                predicted_jitter_ms: forecast.predicted_jitter_ms,
                confidence: forecast.confidence,
                recommendations: vec![], // Would be filled by recommendation engine
            };
            
            alerts.push(alert);
            self.recent_alerts.insert(alert_key, Utc::now());
        }
        
        // Low confidence check
        if forecast.confidence < self.config.prediction_confidence_threshold {
            let alert = QualityAlert {
                alert_id: uuid::Uuid::new_v4().to_string(),
                cell_id: cell_id.to_string(),
                alert_type: AlertType::PredictionConfidenceLow,
                severity: AlertSeverity::Info,
                message: format!(
                    "Low prediction confidence: {:.2} < {:.2}",
                    forecast.confidence,
                    self.config.prediction_confidence_threshold
                ),
                timestamp: Utc::now(),
                predicted_jitter_ms: forecast.predicted_jitter_ms,
                confidence: forecast.confidence,
                recommendations: vec![],
            };
            
            alerts.push(alert);
        }
        
        alerts
    }
}

/// Model performance tracking
pub struct ModelPerformanceTracker {
    metrics: HashMap<String, ModelMetrics>,
}

impl ModelPerformanceTracker {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
    
    pub fn update_metrics(&mut self, model_name: &str, metrics: ModelMetrics) {
        self.metrics.insert(model_name.to_string(), metrics);
    }
    
    pub fn get_metrics(&self, model_name: &str) -> Option<&ModelMetrics> {
        self.metrics.get(model_name)
    }
    
    pub fn get_all_metrics(&self) -> HashMap<String, ModelMetrics> {
        self.metrics.clone()
    }
}