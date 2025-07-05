//! Metrics collection and reporting for the RAN Intelligence Platform

use crate::{Result, RanError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Metrics collector and aggregator
pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    counters: Arc<RwLock<HashMap<String, u64>>>,
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Record a gauge metric
    pub async fn gauge(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        let mut metrics_lock = self.metrics.write().await;
        metrics_lock.insert(name.to_string(), MetricValue {
            timestamp: Utc::now(),
            value,
            tags,
        });
        Ok(())
    }
    
    /// Increment a counter
    pub async fn counter(&self, name: &str) -> Result<()> {
        let mut counters_lock = self.counters.write().await;
        *counters_lock.entry(name.to_string()).or_insert(0) += 1;
        Ok(())
    }
    
    /// Record a histogram value
    pub async fn histogram(&self, name: &str, value: f64) -> Result<()> {
        let mut histograms_lock = self.histograms.write().await;
        histograms_lock.entry(name.to_string()).or_insert_with(Vec::new).push(value);
        Ok(())
    }
    
    /// Get current metrics snapshot
    pub async fn get_metrics(&self) -> Result<MetricsSnapshot> {
        let metrics_lock = self.metrics.read().await;
        let counters_lock = self.counters.read().await;
        let histograms_lock = self.histograms.read().await;
        
        Ok(MetricsSnapshot {
            timestamp: Utc::now(),
            gauges: metrics_lock.clone(),
            counters: counters_lock.clone(),
            histograms: histograms_lock.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub gauges: HashMap<String, MetricValue>,
    pub counters: HashMap<String, u64>,
    pub histograms: HashMap<String, Vec<f64>>,
}