//! Real-time monitoring and alerting module

use std::sync::Arc;
use std::collections::HashMap;
use std::time::SystemTime;
use chrono::{DateTime, Utc};
use tokio::sync::{RwLock, mpsc};
use tokio::time::{interval, Duration};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use prometheus::{Counter, Histogram, Gauge, Registry, Encoder, TextEncoder};
use axum::{
    extract::State,
    http::StatusCode,
    response::Response,
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;

use crate::{ForecastingMetrics, ForecastingError, config::ForecastingConfig};

/// Performance monitoring system
pub struct PerformanceMonitor {
    config: Arc<ForecastingConfig>,
    metrics: Arc<RwLock<MonitoringMetrics>>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    alert_sender: Option<mpsc::UnboundedSender<Alert>>,
    prometheus_registry: Arc<Registry>,
    
    // Prometheus metrics
    forecast_requests_total: Counter,
    forecast_duration_seconds: Histogram,
    forecast_accuracy_gauge: Gauge,
    energy_savings_total: Counter,
    active_sleep_windows: Gauge,
    system_errors_total: Counter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringMetrics {
    pub forecast_requests_count: u64,
    pub avg_forecast_duration_ms: f64,
    pub current_mape: f64,
    pub current_detection_rate: f64,
    pub total_energy_saved_kwh: f64,
    pub active_sleep_windows_count: u32,
    pub error_rate_percent: f64,
    pub system_uptime_seconds: u64,
    pub last_updated: DateTime<Utc>,
}

impl Default for MonitoringMetrics {
    fn default() -> Self {
        Self {
            forecast_requests_count: 0,
            avg_forecast_duration_ms: 0.0,
            current_mape: 0.0,
            current_detection_rate: 0.0,
            total_energy_saved_kwh: 0.0,
            active_sleep_windows_count: 0,
            error_rate_percent: 0.0,
            system_uptime_seconds: 0,
            last_updated: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub category: AlertCategory,
    pub title: String,
    pub description: String,
    pub cell_id: Option<String>,
    pub metric_value: Option<f64>,
    pub threshold: Option<f64>,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCategory {
    Performance,
    Accuracy,
    Energy,
    System,
    Network,
}

impl PerformanceMonitor {
    pub async fn new(config: Arc<ForecastingConfig>) -> Result<Self> {
        let registry = Arc::new(Registry::new());
        
        // Initialize Prometheus metrics
        let forecast_requests_total = Counter::new(
            "cell_sleep_forecaster_requests_total",
            "Total number of forecast requests"
        )?;
        registry.register(Box::new(forecast_requests_total.clone()))?;
        
        let forecast_duration_seconds = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "cell_sleep_forecaster_duration_seconds",
                "Duration of forecast requests in seconds"
            )
        )?;
        registry.register(Box::new(forecast_duration_seconds.clone()))?;
        
        let forecast_accuracy_gauge = Gauge::new(
            "cell_sleep_forecaster_accuracy",
            "Current forecasting accuracy (1 - MAPE/100)"
        )?;
        registry.register(Box::new(forecast_accuracy_gauge.clone()))?;
        
        let energy_savings_total = Counter::new(
            "cell_sleep_energy_savings_kwh_total",
            "Total energy savings in kWh"
        )?;
        registry.register(Box::new(energy_savings_total.clone()))?;
        
        let active_sleep_windows = Gauge::new(
            "cell_sleep_windows_active",
            "Number of currently active sleep windows"
        )?;
        registry.register(Box::new(active_sleep_windows.clone()))?;
        
        let system_errors_total = Counter::new(
            "cell_sleep_system_errors_total",
            "Total number of system errors"
        )?;
        registry.register(Box::new(system_errors_total.clone()))?;
        
        Ok(Self {
            config,
            metrics: Arc::new(RwLock::new(MonitoringMetrics::default())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            alert_sender: None,
            prometheus_registry: registry,
            forecast_requests_total,
            forecast_duration_seconds,
            forecast_accuracy_gauge,
            energy_savings_total,
            active_sleep_windows,
            system_errors_total,
        })
    }
    
    /// Start monitoring system
    pub async fn start(&self) -> Result<()> {
        log::info!("Starting performance monitoring system");
        
        // Start alert processing task  
        let (alert_sender, alert_receiver) = mpsc::unbounded_channel();
        // Note: In a real implementation, we'd store the sender in an Arc<RwLock<Option<...>>>
        
        let config = self.config.clone();
        let alerts = self.alerts.clone();
        
        tokio::spawn(async move {
            Self::process_alerts(alert_receiver, alerts).await;
        });
        
        // Start metrics collection task
        let metrics = self.metrics.clone();
        let prometheus_registry = self.prometheus_registry.clone();
        
        tokio::spawn(async move {
            Self::collect_system_metrics(metrics, prometheus_registry).await;
        });
        
        // Start HTTP server for metrics endpoint
        if self.config.monitoring.enabled {
            let registry = self.prometheus_registry.clone();
            let alerts = self.alerts.clone();
            let metrics = self.metrics.clone();
            let port = self.config.monitoring.metrics_port;
            
            tokio::spawn(async move {
                Self::start_metrics_server(registry, alerts, metrics, port).await;
            });
        }
        
        log::info!("Performance monitoring system started");
        Ok(())
    }
    
    /// Stop monitoring system
    pub async fn stop(&self) -> Result<()> {
        log::info!("Stopping performance monitoring system");
        // In a real implementation, we would stop the background tasks
        Ok(())
    }
    
    /// Record a forecast request
    pub async fn record_forecast_request(&self, cell_id: &str) -> Result<()> {
        let start_time = SystemTime::now();
        
        // Update metrics
        self.forecast_requests_total.inc();
        
        let mut metrics = self.metrics.write().await;
        metrics.forecast_requests_count += 1;
        metrics.last_updated = Utc::now();
        
        log::debug!("Recorded forecast request for cell {}", cell_id);
        Ok(())
    }
    
    /// Record forecast completion with timing
    pub async fn record_forecast_completion(&self, cell_id: &str, duration_ms: f64) -> Result<()> {
        self.forecast_duration_seconds.observe(duration_ms / 1000.0);
        
        let mut metrics = self.metrics.write().await;
        // Update rolling average
        let alpha = 0.1; // Smoothing factor
        metrics.avg_forecast_duration_ms = alpha * duration_ms + (1.0 - alpha) * metrics.avg_forecast_duration_ms;
        
        // Check for performance alerts
        if duration_ms > self.config.monitoring.alert_thresholds.prediction_latency_ms as f64 {
            self.send_alert(Alert {
                id: format!("latency_{}_{}", cell_id, Utc::now().timestamp()),
                timestamp: Utc::now(),
                severity: AlertSeverity::Warning,
                category: AlertCategory::Performance,
                title: "High Forecast Latency".to_string(),
                description: format!("Forecast for cell {} took {:.2}ms (threshold: {}ms)", 
                    cell_id, duration_ms, self.config.monitoring.alert_thresholds.prediction_latency_ms),
                cell_id: Some(cell_id.to_string()),
                metric_value: Some(duration_ms),
                threshold: Some(self.config.monitoring.alert_thresholds.prediction_latency_ms as f64),
                resolved: false,
            }).await?;
        }
        
        log::debug!("Recorded forecast completion for cell {} in {:.2}ms", cell_id, duration_ms);
        Ok(())
    }
    
    /// Update accuracy metrics
    pub async fn update_accuracy_metrics(&self, forecasting_metrics: &ForecastingMetrics) -> Result<()> {
        let accuracy = (100.0 - forecasting_metrics.mape) / 100.0;
        self.forecast_accuracy_gauge.set(accuracy);
        
        let mut metrics = self.metrics.write().await;
        metrics.current_mape = forecasting_metrics.mape;
        metrics.current_detection_rate = forecasting_metrics.low_traffic_detection_rate;
        
        // Check for accuracy alerts
        if forecasting_metrics.mape > self.config.monitoring.alert_thresholds.mape_threshold {
            self.send_alert(Alert {
                id: format!("mape_{}", Utc::now().timestamp()),
                timestamp: Utc::now(),
                severity: AlertSeverity::Critical,
                category: AlertCategory::Accuracy,
                title: "High MAPE Detected".to_string(),
                description: format!("MAPE is {:.2}% (threshold: {:.2}%)", 
                    forecasting_metrics.mape, self.config.monitoring.alert_thresholds.mape_threshold),
                cell_id: None,
                metric_value: Some(forecasting_metrics.mape),
                threshold: Some(self.config.monitoring.alert_thresholds.mape_threshold),
                resolved: false,
            }).await?;
        }
        
        if forecasting_metrics.low_traffic_detection_rate < self.config.monitoring.alert_thresholds.detection_rate_threshold {
            self.send_alert(Alert {
                id: format!("detection_rate_{}", Utc::now().timestamp()),
                timestamp: Utc::now(),
                severity: AlertSeverity::Critical,
                category: AlertCategory::Accuracy,
                title: "Low Detection Rate".to_string(),
                description: format!("Detection rate is {:.2}% (threshold: {:.2}%)", 
                    forecasting_metrics.low_traffic_detection_rate, self.config.monitoring.alert_thresholds.detection_rate_threshold),
                cell_id: None,
                metric_value: Some(forecasting_metrics.low_traffic_detection_rate),
                threshold: Some(self.config.monitoring.alert_thresholds.detection_rate_threshold),
                resolved: false,
            }).await?;
        }
        
        log::debug!("Updated accuracy metrics: MAPE={:.2}%, Detection Rate={:.2}%", 
            forecasting_metrics.mape, forecasting_metrics.low_traffic_detection_rate);
        Ok(())
    }
    
    /// Record energy savings
    pub async fn record_energy_savings(&self, cell_id: &str, savings_kwh: f64) -> Result<()> {
        self.energy_savings_total.inc_by(savings_kwh);
        
        let mut metrics = self.metrics.write().await;
        metrics.total_energy_saved_kwh += savings_kwh;
        
        log::info!("Recorded energy savings for cell {}: {:.2} kWh", cell_id, savings_kwh);
        Ok(())
    }
    
    /// Update active sleep windows count
    pub async fn update_active_sleep_windows(&self, count: u32) -> Result<()> {
        self.active_sleep_windows.set(count as f64);
        
        let mut metrics = self.metrics.write().await;
        metrics.active_sleep_windows_count = count;
        
        log::debug!("Updated active sleep windows count: {}", count);
        Ok(())
    }
    
    /// Record system error
    pub async fn record_error(&self, error: &ForecastingError) -> Result<()> {
        self.system_errors_total.inc();
        
        let mut metrics = self.metrics.write().await;
        let total_requests = metrics.forecast_requests_count;
        if total_requests > 0 {
            metrics.error_rate_percent = (self.system_errors_total.get() / total_requests as f64) * 100.0;
        }
        
        // Send error alert
        let severity = match error {
            ForecastingError::InsufficientData(_) => AlertSeverity::Warning,
            ForecastingError::ModelTrainingFailed(_) => AlertSeverity::Critical,
            ForecastingError::PredictionFailed(_) => AlertSeverity::Critical,
            _ => AlertSeverity::Warning,
        };
        
        self.send_alert(Alert {
            id: format!("error_{}", Utc::now().timestamp()),
            timestamp: Utc::now(),
            severity,
            category: AlertCategory::System,
            title: "System Error".to_string(),
            description: format!("Error occurred: {}", error),
            cell_id: None,
            metric_value: None,
            threshold: None,
            resolved: false,
        }).await?;
        
        log::error!("Recorded system error: {}", error);
        Ok(())
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> Result<MonitoringMetrics> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }
    
    /// Get recent alerts
    pub async fn get_alerts(&self, limit: Option<usize>) -> Result<Vec<Alert>> {
        let alerts = self.alerts.read().await;
        let mut sorted_alerts = alerts.clone();
        sorted_alerts.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        if let Some(limit) = limit {
            sorted_alerts.truncate(limit);
        }
        
        Ok(sorted_alerts)
    }
    
    async fn send_alert(&self, alert: Alert) -> Result<()> {
        // In a real implementation, we'd store the sender in an Arc<RwLock<Option<...>>>
        // For now, just log the alert
        log::warn!("Alert: {} - {}", alert.severity, alert.title);
        Ok(())
    }
    
    async fn process_alerts(
        mut receiver: mpsc::UnboundedReceiver<Alert>,
        alerts: Arc<RwLock<Vec<Alert>>>,
    ) {
        while let Some(alert) = receiver.recv().await {
            log::info!("Processing alert: {} - {}", alert.severity, alert.title);
            
            let mut alerts_store = alerts.write().await;
            alerts_store.push(alert);
            
            // Keep only recent alerts (last 1000)
            if alerts_store.len() > 1000 {
                alerts_store.remove(0);
            }
        }
    }
    
    async fn collect_system_metrics(
        metrics: Arc<RwLock<MonitoringMetrics>>,
        _registry: Arc<Registry>,
    ) {
        let mut interval = interval(Duration::from_secs(60));
        let start_time = SystemTime::now();
        
        loop {
            interval.tick().await;
            
            let uptime = start_time.elapsed().unwrap_or_default().as_secs();
            
            let mut metrics_store = metrics.write().await;
            metrics_store.system_uptime_seconds = uptime;
            metrics_store.last_updated = Utc::now();
        }
    }
    
    async fn start_metrics_server(
        registry: Arc<Registry>,
        alerts: Arc<RwLock<Vec<Alert>>>,
        metrics: Arc<RwLock<MonitoringMetrics>>,
        port: u16,
    ) {
        let app = Router::new()
            .route("/metrics", get(Self::handle_metrics))
            .route("/alerts", get(Self::handle_alerts))
            .route("/health", get(Self::handle_health))
            // .layer(CorsLayer::permissive())  // Commented out due to compatibility issues
            .with_state(MonitoringState {
                registry,
                alerts,
                metrics,
            });
        
        let addr = format!("0.0.0.0:{}", port);
        log::info!("Starting metrics server on {}", addr);
        
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }
    
    async fn handle_metrics(State(state): State<MonitoringState>) -> Result<Response<String>, StatusCode> {
        let encoder = TextEncoder::new();
        let metric_families = state.registry.gather();
        
        match encoder.encode_to_string(&metric_families) {
            Ok(output) => Ok(Response::builder()
                .header("Content-Type", "text/plain; version=0.0.4")
                .body(output)
                .unwrap()),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }
    
    async fn handle_alerts(State(state): State<MonitoringState>) -> Result<axum::Json<Vec<Alert>>, StatusCode> {
        let alerts = state.alerts.read().await;
        let mut recent_alerts = alerts.clone();
        recent_alerts.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        recent_alerts.truncate(50); // Last 50 alerts
        
        Ok(axum::Json(recent_alerts))
    }
    
    async fn handle_health(State(state): State<MonitoringState>) -> Result<axum::Json<MonitoringMetrics>, StatusCode> {
        let metrics = state.metrics.read().await;
        Ok(axum::Json(metrics.clone()))
    }
}

#[derive(Clone)]
struct MonitoringState {
    registry: Arc<Registry>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    metrics: Arc<RwLock<MonitoringMetrics>>,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Critical => write!(f, "CRITICAL"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Info => write!(f, "INFO"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ForecastingConfig;
    
    #[tokio::test]
    async fn test_monitor_creation() {
        let config = Arc::new(ForecastingConfig::default());
        let monitor = PerformanceMonitor::new(config).await.unwrap();
        
        let metrics = monitor.get_metrics().await.unwrap();
        assert_eq!(metrics.forecast_requests_count, 0);
        assert_eq!(metrics.total_energy_saved_kwh, 0.0);
    }
    
    #[tokio::test]
    async fn test_forecast_request_recording() {
        let config = Arc::new(ForecastingConfig::default());
        let monitor = PerformanceMonitor::new(config).await.unwrap();
        
        monitor.record_forecast_request("test_cell").await.unwrap();
        
        let metrics = monitor.get_metrics().await.unwrap();
        assert_eq!(metrics.forecast_requests_count, 1);
    }
    
    #[tokio::test]
    async fn test_energy_savings_recording() {
        let config = Arc::new(ForecastingConfig::default());
        let monitor = PerformanceMonitor::new(config).await.unwrap();
        
        monitor.record_energy_savings("test_cell", 5.5).await.unwrap();
        
        let metrics = monitor.get_metrics().await.unwrap();
        assert_eq!(metrics.total_energy_saved_kwh, 5.5);
    }
    
    #[tokio::test]
    async fn test_alert_generation() {
        let config = Arc::new(ForecastingConfig::default());
        let mut monitor = PerformanceMonitor::new(config).await.unwrap();
        
        // Start monitoring to initialize alert processing
        monitor.start().await.unwrap();
        
        // Record high latency to trigger alert
        monitor.record_forecast_completion("test_cell", 2000.0).await.unwrap();
        
        // Give some time for alert processing
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let alerts = monitor.get_alerts(Some(10)).await.unwrap();
        assert!(!alerts.is_empty());
    }
}