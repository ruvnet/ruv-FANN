use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub agent_id: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct TelemetryCollector {
    events: Arc<RwLock<Vec<TelemetryEvent>>>,
}

impl TelemetryCollector {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn record_event(&self, event: TelemetryEvent) {
        self.events.write().push(event);
    }

    pub fn get_events(&self) -> Vec<TelemetryEvent> {
        self.events.read().clone()
    }

    pub fn clear_events(&self) {
        self.events.write().clear();
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new()
    }
}
