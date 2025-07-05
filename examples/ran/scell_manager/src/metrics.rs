//! Metrics collection and monitoring for the SCell Manager

use anyhow::Result;
use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry,
};
use std::sync::Arc;
use tokio::time::{Duration, Instant};

/// Metrics collector for SCell Manager
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    registry: Arc<Registry>,
    
    // Prediction metrics
    predictions_total: IntCounter,
    predictions_successful: IntCounter,
    predictions_failed: IntCounter,
    prediction_duration: Histogram,
    prediction_confidence: Histogram,
    
    // Model metrics
    models_active: IntGauge,
    model_accuracy: Gauge,
    model_precision: Gauge,
    model_recall: Gauge,
    
    // SCell activation metrics
    scell_activations_recommended: IntCounter,
    scell_activations_confidence_high: IntCounter,
    scell_activations_confidence_medium: IntCounter,
    scell_activations_confidence_low: IntCounter,
    
    // Performance metrics
    cache_hits: IntCounter,
    cache_misses: IntCounter,
    cache_size: IntGauge,
    
    // System metrics
    system_uptime: Gauge,
    memory_usage: Gauge,
    cpu_usage: Gauge,
    
    // Business metrics
    throughput_demand_predicted: Histogram,
    ue_sessions_active: IntGauge,
    
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Result<Self> {
        let registry = Arc::new(Registry::new());
        
        // Prediction metrics
        let predictions_total = IntCounter::with_opts(
            Opts::new("scell_predictions_total", "Total number of predictions made")
        )?;
        
        let predictions_successful = IntCounter::with_opts(
            Opts::new("scell_predictions_successful_total", "Total number of successful predictions")
        )?;
        
        let predictions_failed = IntCounter::with_opts(
            Opts::new("scell_predictions_failed_total", "Total number of failed predictions")
        )?;
        
        let prediction_duration = Histogram::with_opts(
            HistogramOpts::new("scell_prediction_duration_seconds", "Time taken for predictions")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
        )?;
        
        let prediction_confidence = Histogram::with_opts(
            HistogramOpts::new("scell_prediction_confidence", "Confidence scores of predictions")
                .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        )?;
        
        // Model metrics
        let models_active = IntGauge::with_opts(
            Opts::new("scell_models_active", "Number of active models")
        )?;
        
        let model_accuracy = Gauge::with_opts(
            Opts::new("scell_model_accuracy", "Current model accuracy")
        )?;
        
        let model_precision = Gauge::with_opts(
            Opts::new("scell_model_precision", "Current model precision")
        )?;
        
        let model_recall = Gauge::with_opts(
            Opts::new("scell_model_recall", "Current model recall")
        )?;
        
        // SCell activation metrics
        let scell_activations_recommended = IntCounter::with_opts(
            Opts::new("scell_activations_recommended_total", "Total SCell activations recommended")
        )?;
        
        let scell_activations_confidence_high = IntCounter::with_opts(
            Opts::new("scell_activations_confidence_high_total", "SCell activations with high confidence (>0.8)")
        )?;
        
        let scell_activations_confidence_medium = IntCounter::with_opts(
            Opts::new("scell_activations_confidence_medium_total", "SCell activations with medium confidence (0.5-0.8)")
        )?;
        
        let scell_activations_confidence_low = IntCounter::with_opts(
            Opts::new("scell_activations_confidence_low_total", "SCell activations with low confidence (<0.5)")
        )?;
        
        // Performance metrics
        let cache_hits = IntCounter::with_opts(
            Opts::new("scell_cache_hits_total", "Total cache hits")
        )?;
        
        let cache_misses = IntCounter::with_opts(
            Opts::new("scell_cache_misses_total", "Total cache misses")
        )?;
        
        let cache_size = IntGauge::with_opts(
            Opts::new("scell_cache_size", "Current cache size")
        )?;
        
        // System metrics
        let system_uptime = Gauge::with_opts(
            Opts::new("scell_system_uptime_seconds", "System uptime in seconds")
        )?;
        
        let memory_usage = Gauge::with_opts(
            Opts::new("scell_memory_usage_bytes", "Memory usage in bytes")
        )?;
        
        let cpu_usage = Gauge::with_opts(
            Opts::new("scell_cpu_usage_percent", "CPU usage percentage")
        )?;
        
        // Business metrics
        let throughput_demand_predicted = Histogram::with_opts(
            HistogramOpts::new("scell_throughput_demand_predicted_mbps", "Predicted throughput demand in Mbps")
                .buckets(vec![0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
        )?;
        
        let ue_sessions_active = IntGauge::with_opts(
            Opts::new("scell_ue_sessions_active", "Number of active UE sessions")
        )?;
        
        // Register all metrics
        registry.register(Box::new(predictions_total.clone()))?;
        registry.register(Box::new(predictions_successful.clone()))?;
        registry.register(Box::new(predictions_failed.clone()))?;
        registry.register(Box::new(prediction_duration.clone()))?;
        registry.register(Box::new(prediction_confidence.clone()))?;
        registry.register(Box::new(models_active.clone()))?;
        registry.register(Box::new(model_accuracy.clone()))?;
        registry.register(Box::new(model_precision.clone()))?;
        registry.register(Box::new(model_recall.clone()))?;
        registry.register(Box::new(scell_activations_recommended.clone()))?;
        registry.register(Box::new(scell_activations_confidence_high.clone()))?;
        registry.register(Box::new(scell_activations_confidence_medium.clone()))?;
        registry.register(Box::new(scell_activations_confidence_low.clone()))?;
        registry.register(Box::new(cache_hits.clone()))?;
        registry.register(Box::new(cache_misses.clone()))?;
        registry.register(Box::new(cache_size.clone()))?;
        registry.register(Box::new(system_uptime.clone()))?;
        registry.register(Box::new(memory_usage.clone()))?;
        registry.register(Box::new(cpu_usage.clone()))?;
        registry.register(Box::new(throughput_demand_predicted.clone()))?;
        registry.register(Box::new(ue_sessions_active.clone()))?;
        
        Ok(Self {
            registry,
            predictions_total,
            predictions_successful,
            predictions_failed,
            prediction_duration,
            prediction_confidence,
            models_active,
            model_accuracy,
            model_precision,
            model_recall,
            scell_activations_recommended,
            scell_activations_confidence_high,
            scell_activations_confidence_medium,
            scell_activations_confidence_low,
            cache_hits,
            cache_misses,
            cache_size,
            system_uptime,
            memory_usage,
            cpu_usage,
            throughput_demand_predicted,
            ue_sessions_active,
            start_time: Instant::now(),
        })
    }
    
    /// Record a prediction event
    pub fn record_prediction(&self, 
                           duration: Duration, 
                           success: bool, 
                           confidence: f32, 
                           scell_recommended: bool,
                           predicted_throughput: f32) {
        self.predictions_total.inc();
        
        if success {
            self.predictions_successful.inc();
        } else {
            self.predictions_failed.inc();
        }
        
        self.prediction_duration.observe(duration.as_secs_f64());
        self.prediction_confidence.observe(confidence as f64);
        
        if scell_recommended {
            self.scell_activations_recommended.inc();
            
            // Categorize by confidence level
            if confidence > 0.8 {
                self.scell_activations_confidence_high.inc();
            } else if confidence > 0.5 {
                self.scell_activations_confidence_medium.inc();
            } else {
                self.scell_activations_confidence_low.inc();
            }
        }
        
        self.throughput_demand_predicted.observe(predicted_throughput as f64);
    }
    
    /// Record model metrics
    pub fn record_model_metrics(&self, 
                              accuracy: f32, 
                              precision: f32, 
                              recall: f32, 
                              active_models: i64) {
        self.model_accuracy.set(accuracy as f64);
        self.model_precision.set(precision as f64);
        self.model_recall.set(recall as f64);
        self.models_active.set(active_models);
    }
    
    /// Record cache event
    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }
    
    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }
    
    /// Update cache size
    pub fn update_cache_size(&self, size: i64) {
        self.cache_size.set(size);
    }
    
    /// Update system metrics
    pub fn update_system_metrics(&self, memory_bytes: f64, cpu_percent: f64) {
        self.system_uptime.set(self.start_time.elapsed().as_secs_f64());
        self.memory_usage.set(memory_bytes);
        self.cpu_usage.set(cpu_percent);
    }
    
    /// Update active UE sessions
    pub fn update_active_ue_sessions(&self, count: i64) {
        self.ue_sessions_active.set(count);
    }
    
    /// Get Prometheus metrics registry
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
    
    /// Get metrics summary
    pub fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_predictions: self.predictions_total.get(),
            successful_predictions: self.predictions_successful.get(),
            failed_predictions: self.predictions_failed.get(),
            scell_activations_recommended: self.scell_activations_recommended.get(),
            cache_hit_rate: self.calculate_cache_hit_rate(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            active_models: self.models_active.get(),
            active_ue_sessions: self.ue_sessions_active.get(),
        }
    }
    
    /// Calculate cache hit rate
    fn calculate_cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get() as f64;
        let misses = self.cache_misses.get() as f64;
        let total = hits + misses;
        
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
    
    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> Result<String> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        
        Ok(String::from_utf8(buffer)?)
    }
    
    /// Reset all metrics (useful for testing)
    #[cfg(test)]
    pub fn reset(&self) {
        // Note: Prometheus counters cannot be reset, only gauges
        self.models_active.set(0);
        self.model_accuracy.set(0.0);
        self.model_precision.set(0.0);
        self.model_recall.set(0.0);
        self.cache_size.set(0);
        self.system_uptime.set(0.0);
        self.memory_usage.set(0.0);
        self.cpu_usage.set(0.0);
        self.ue_sessions_active.set(0);
    }
}

/// Summary of key metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_predictions: u64,
    pub successful_predictions: u64,
    pub failed_predictions: u64,
    pub scell_activations_recommended: u64,
    pub cache_hit_rate: f64,
    pub uptime_seconds: u64,
    pub active_models: i64,
    pub active_ue_sessions: i64,
}

impl MetricsSummary {
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_predictions > 0 {
            self.successful_predictions as f64 / self.total_predictions as f64
        } else {
            0.0
        }
    }
    
    /// Get SCell activation rate
    pub fn scell_activation_rate(&self) -> f64 {
        if self.total_predictions > 0 {
            self.scell_activations_recommended as f64 / self.total_predictions as f64
        } else {
            0.0
        }
    }
}

/// Metrics aggregator for batch operations
#[derive(Debug, Default)]
pub struct MetricsAggregator {
    prediction_count: u64,
    total_duration: Duration,
    success_count: u64,
    confidence_sum: f64,
    scell_recommendations: u64,
    throughput_sum: f64,
}

impl MetricsAggregator {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add_prediction(&mut self, 
                         duration: Duration, 
                         success: bool, 
                         confidence: f32,
                         scell_recommended: bool,
                         predicted_throughput: f32) {
        self.prediction_count += 1;
        self.total_duration += duration;
        
        if success {
            self.success_count += 1;
        }
        
        self.confidence_sum += confidence as f64;
        
        if scell_recommended {
            self.scell_recommendations += 1;
        }
        
        self.throughput_sum += predicted_throughput as f64;
    }
    
    pub fn flush_to_collector(&self, collector: &MetricsCollector) {
        if self.prediction_count == 0 {
            return;
        }
        
        let avg_duration = self.total_duration / self.prediction_count as u32;
        let avg_confidence = (self.confidence_sum / self.prediction_count as f64) as f32;
        let avg_throughput = (self.throughput_sum / self.prediction_count as f64) as f32;
        
        // Record aggregated metrics
        for _ in 0..self.prediction_count {
            collector.record_prediction(
                avg_duration,
                self.success_count > 0,
                avg_confidence,
                self.scell_recommendations > 0,
                avg_throughput,
            );
        }
    }
    
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;
    
    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert!(collector.is_ok());
    }
    
    #[test]
    fn test_metrics_recording() {
        let collector = MetricsCollector::new().unwrap();
        
        collector.record_prediction(
            Duration::from_millis(10),
            true,
            0.8,
            true,
            150.0,
        );
        
        let summary = collector.get_summary();
        assert_eq!(summary.total_predictions, 1);
        assert_eq!(summary.successful_predictions, 1);
        assert_eq!(summary.scell_activations_recommended, 1);
    }
    
    #[test]
    fn test_cache_hit_rate() {
        let collector = MetricsCollector::new().unwrap();
        
        collector.record_cache_hit();
        collector.record_cache_hit();
        collector.record_cache_miss();
        
        let hit_rate = collector.calculate_cache_hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01);
    }
    
    #[test]
    fn test_metrics_aggregator() {
        let mut aggregator = MetricsAggregator::new();
        
        aggregator.add_prediction(
            Duration::from_millis(5),
            true,
            0.9,
            true,
            200.0,
        );
        
        aggregator.add_prediction(
            Duration::from_millis(15),
            true,
            0.7,
            false,
            50.0,
        );
        
        assert_eq!(aggregator.prediction_count, 2);
        assert_eq!(aggregator.success_count, 2);
        assert_eq!(aggregator.scell_recommendations, 1);
    }
    
    #[test]
    fn test_metrics_summary() {
        let summary = MetricsSummary {
            total_predictions: 100,
            successful_predictions: 95,
            failed_predictions: 5,
            scell_activations_recommended: 30,
            cache_hit_rate: 0.85,
            uptime_seconds: 3600,
            active_models: 2,
            active_ue_sessions: 150,
        };
        
        assert_eq!(summary.success_rate(), 0.95);
        assert_eq!(summary.scell_activation_rate(), 0.30);
    }
}