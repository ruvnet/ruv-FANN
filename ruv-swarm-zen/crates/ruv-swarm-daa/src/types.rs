//! Type definitions for DAA WASM compatibility

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Autonomous capability enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AutonomousCapability {
    SelfMonitoring,
    DecisionMaking,
    ResourceOptimization,
    SelfHealing,
    Learning,
    EmergentBehavior,
    Prediction,
    GoalPlanning,
    Coordination,
    MemoryManagement,
    Custom(String),
}

/// Decision context for autonomous agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    pub available_actions: Vec<String>,
    pub constraints: HashMap<String, f64>,
    pub objectives: Vec<String>,
    pub current_state: HashMap<String, serde_json::Value>,
    pub historical_data: Vec<serde_json::Value>,
}

impl Default for DecisionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionContext {
    pub fn new() -> Self {
        Self {
            available_actions: Vec::new(),
            constraints: HashMap::new(),
            objectives: Vec::new(),
            current_state: HashMap::new(),
            historical_data: Vec::new(),
        }
    }
}

/// Adaptation feedback for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationFeedback {
    pub performance_score: f64,
    pub efficiency_score: f64,
    pub learning_rate_adjustment: f64,
    pub timestamp: u64,
}

impl AdaptationFeedback {
    pub fn positive(score: f64) -> Self {
        Self {
            performance_score: score,
            efficiency_score: score,
            learning_rate_adjustment: 1.1, // Increase learning rate
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }

    pub fn negative(score: f64) -> Self {
        Self {
            performance_score: score,
            efficiency_score: score,
            learning_rate_adjustment: 0.9, // Decrease learning rate
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }

    pub fn is_positive(&self) -> bool {
        self.performance_score > 0.5 && self.efficiency_score > 0.5
    }
}

// Priority is defined in main lib.rs, reexport it here
pub use crate::Priority;

/// Agent type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    Researcher,
    Coder,
    Analyst,
    Coordinator,
    Specialist,
}

/// Task request for coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub id: String,
    pub task_type: String,
    pub payload: serde_json::Value,
    pub priority: Priority,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub requirements: Vec<String>,
}

// LearningEngine is defined in learning.rs, reexport it here for coordinator compatibility
pub use crate::learning::LearningEngine;

/// Neural network manager for coordination
#[derive(Debug)]
pub struct NeuralNetworkManager {
    initialized: bool,
}

impl Default for NeuralNetworkManager {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralNetworkManager {
    pub fn new() -> Self {
        Self { initialized: false }
    }

    pub async fn initialize_coordination_patterns(&mut self) -> Result<(), crate::DAAError> {
        self.initialized = true;
        Ok(())
    }
}
