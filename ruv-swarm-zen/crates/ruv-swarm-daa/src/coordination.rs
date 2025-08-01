//! Coordination modules for DAA agents

use crate::*;
use std::collections::HashMap;

/// Coordination memory for managing agent interactions
pub struct CoordinationMemory {
    pub shared_state: HashMap<String, serde_json::Value>,
    pub agent_locations: HashMap<String, AgentLocation>,
    pub coordination_history: Vec<CoordinationEvent>,
}

/// Agent location in coordination space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLocation {
    pub agent_id: String,
    pub position: [f64; 3], // 3D coordination space
    pub capabilities: Vec<String>,
    pub current_task: Option<String>,
    pub availability: f64,
}

/// Coordination event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: CoordinationEventType,
    pub participants: Vec<String>,
    pub outcome: serde_json::Value,
}

/// Types of coordination events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationEventType {
    TaskAssignment,
    KnowledgeSharing,
    ConflictResolution,
    ResourceAllocation,
    PerformanceEvaluation,
}

impl Default for CoordinationMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordinationMemory {
    pub fn new() -> Self {
        Self {
            shared_state: HashMap::new(),
            agent_locations: HashMap::new(),
            coordination_history: Vec::new(),
        }
    }

    pub async fn store_event(&mut self, event: CoordinationEvent) -> Result<(), DAAError> {
        self.coordination_history.push(event);
        Ok(())
    }

    pub async fn get_recent_events(
        &self,
        limit: usize,
    ) -> Result<Vec<CoordinationEvent>, DAAError> {
        let len = self.coordination_history.len();
        let start = len.saturating_sub(limit);
        Ok(self.coordination_history[start..].to_vec())
    }

    pub async fn get_event_count(&self) -> Result<usize, DAAError> {
        Ok(self.coordination_history.len())
    }
}
