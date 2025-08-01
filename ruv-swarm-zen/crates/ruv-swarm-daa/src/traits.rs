//! Core DAA trait definitions for autonomous agent behaviors

#[cfg(feature = "async")]
use async_trait::async_trait;

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

use crate::{
    types::{AdaptationFeedback, AutonomousCapability, DecisionContext},
    DAAResult as Result,
};

/// Resource allocation structure (moved here to fix compilation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub memory_mb: f64,
    pub cpu_cores: f64,
    pub network_bandwidth: f64,
    pub storage_gb: f64,
    pub priority: f64,
}

/// Core trait for Distributed Autonomous Agents
///
/// This trait defines the fundamental capabilities that make an agent truly autonomous:
/// - Independent decision making
/// - Self-adaptation based on feedback
/// - Resource-aware operation
/// - Emergent behavior capabilities
#[cfg_attr(feature = "async", async_trait)]
pub trait DistributedAutonomousAgent: Send + Sync {
    /// Context type for decision making
    type Context: Send + Sync;

    /// Decision output type
    type Decision: Send + Sync;

    /// Adaptation feedback type
    type Adaptation: Send + Sync;

    /// Make an autonomous decision based on current context
    #[cfg(feature = "async")]
    async fn autonomous_decision(&mut self, context: &Self::Context) -> Result<Self::Decision>;

    /// Make an autonomous decision (sync version for no-std)
    #[cfg(not(feature = "async"))]
    fn autonomous_decision(&mut self, context: &Self::Context) -> Result<Self::Decision>;

    /// Adapt behavior based on feedback from environment or other agents
    #[cfg(feature = "async")]
    async fn self_adapt(&mut self, feedback: &Self::Adaptation) -> Result<()>;

    /// Adapt behavior (sync version for no-std)
    #[cfg(not(feature = "async"))]
    fn self_adapt(&mut self, feedback: &Self::Adaptation) -> Result<()>;

    /// Get autonomous capabilities
    fn autonomous_capabilities(&self) -> &[AutonomousCapability];

    /// Evaluate current autonomy level (0.0 to 1.0)
    fn autonomy_level(&self) -> f64 {
        1.0 // Default: fully autonomous
    }

    /// Check if agent can operate independently
    fn is_autonomous(&self) -> bool {
        self.autonomy_level() > 0.5
    }

    /// Get agent's learning rate for adaptation
    fn learning_rate(&self) -> f64 {
        0.1 // Default learning rate
    }

    /// Update learning rate based on performance
    fn update_learning_rate(&mut self, performance: f64) {
        // Default: no-op, implement in concrete types
        let _ = performance;
    }
}

/// Self-healing capabilities for autonomous agents
#[cfg_attr(feature = "async", async_trait)]
pub trait SelfHealingAgent: Send + Sync {
    /// Error type for self-healing operations
    type Error: Send + Sync;

    /// Diagnose current agent health
    #[cfg(feature = "async")]
    async fn diagnose_health(&self) -> Result<HealthDiagnostic>;

    /// Diagnose health (sync version)
    #[cfg(not(feature = "async"))]
    fn diagnose_health(&self) -> Result<HealthDiagnostic>;

    /// Attempt to heal from detected issues
    #[cfg(feature = "async")]
    async fn self_heal(&mut self, diagnostic: &HealthDiagnostic) -> Result<HealingResult>;

    /// Self-heal (sync version)
    #[cfg(not(feature = "async"))]
    fn self_heal(&mut self, diagnostic: &HealthDiagnostic) -> Result<HealingResult>;

    /// Monitor health continuously
    #[cfg(feature = "async")]
    async fn monitor_health(&mut self) -> Result<()> {
        let diagnostic = self.diagnose_health().await?;
        if diagnostic.needs_healing() {
            self.self_heal(&diagnostic).await?;
        }
        Ok(())
    }

    /// Recovery strategies available to this agent
    fn recovery_strategies(&self) -> &[RecoveryStrategy];

    /// Check if agent can recover from specific error type
    fn can_recover_from(&self, error: &Self::Error) -> bool;
}

/// Resource optimization capabilities
#[cfg_attr(feature = "async", async_trait)]
pub trait ResourceOptimizer: Send + Sync {
    /// Optimize resource allocation
    #[cfg(feature = "async")]
    async fn optimize_resources(&mut self) -> Result<ResourceAllocation>;

    /// Optimize resources (sync version)
    #[cfg(not(feature = "async"))]
    fn optimize_resources(&mut self) -> Result<ResourceAllocation>;

    /// Monitor resource usage
    fn monitor_resources(&self) -> ResourceUsage;

    /// Predict future resource needs
    fn predict_resource_needs(&self, horizon: u32) -> Result<ResourcePrediction>;

    /// Check if resource optimization is needed
    fn needs_optimization(&self) -> bool {
        let usage = self.monitor_resources();
        usage.memory_usage > 0.8 || usage.cpu_usage > 0.9
    }
}

/// Emergent behavior generation
#[cfg_attr(feature = "async", async_trait)]
pub trait EmergentBehavior: Send + Sync {
    /// Emergent state type
    type EmergentState: Send + Sync;

    /// Generate emergent behaviors from local interactions
    #[cfg(feature = "async")]
    async fn generate_emergent_behavior(&mut self) -> Result<Self::EmergentState>;

    /// Generate emergent behavior (sync version)
    #[cfg(not(feature = "async"))]
    fn generate_emergent_behavior(&mut self) -> Result<Self::EmergentState>;

    /// Respond to emergent behaviors from other agents
    #[cfg(feature = "async")]
    async fn respond_to_emergence(&mut self, state: &Self::EmergentState) -> Result<()>;

    /// Respond to emergence (sync version)
    #[cfg(not(feature = "async"))]
    fn respond_to_emergence(&mut self, state: &Self::EmergentState) -> Result<()>;

    /// Check if current state can trigger emergence
    fn can_trigger_emergence(&self) -> bool;

    /// Get emergence probability
    fn emergence_probability(&self) -> f64;
}

/// Cognitive architecture for advanced reasoning
#[cfg_attr(feature = "async", async_trait)]
pub trait CognitiveArchitecture: Send + Sync {
    /// Working memory type
    type WorkingMemory: Send + Sync;

    /// Long-term memory type
    type LongTermMemory: Send + Sync;

    /// Reasoning process type
    type ReasoningProcess: Send + Sync;

    /// Access working memory
    fn working_memory(&mut self) -> &mut Self::WorkingMemory;

    /// Access long-term memory
    fn long_term_memory(&self) -> &Self::LongTermMemory;

    /// Execute reasoning process
    #[cfg(feature = "async")]
    async fn reason(&mut self, problem: &DecisionContext) -> Result<Self::ReasoningProcess>;

    /// Execute reasoning (sync version)
    #[cfg(not(feature = "async"))]
    fn reason(&mut self, problem: &DecisionContext) -> Result<Self::ReasoningProcess>;

    /// Update memories based on experience
    #[cfg(feature = "async")]
    async fn update_memories(&mut self, experience: &AdaptationFeedback) -> Result<()>;

    /// Update memories (sync version)
    #[cfg(not(feature = "async"))]
    fn update_memories(&mut self, experience: &AdaptationFeedback) -> Result<()>;

    /// Get cognitive load (0.0 to 1.0)
    fn cognitive_load(&self) -> f64;

    /// Check if cognitive resources are available
    fn has_cognitive_capacity(&self) -> bool {
        self.cognitive_load() < 0.8
    }
}

/// Health diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDiagnostic {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,

    /// Identified issues
    pub issues: Vec<HealthIssue>,

    /// Recommended actions
    pub recommendations: Vec<String>,

    /// Severity level
    pub severity: Severity,
}

impl HealthDiagnostic {
    /// Check if healing is needed
    pub fn needs_healing(&self) -> bool {
        self.health_score < 0.7 || matches!(self.severity, Severity::Critical | Severity::High)
    }

    /// Get priority issues
    pub fn priority_issues(&self) -> Vec<&HealthIssue> {
        self.issues
            .iter()
            .filter(|issue| matches!(issue.severity, Severity::Critical | Severity::High))
            .collect()
    }
}

/// Health issue description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Issue type
    pub issue_type: IssueType,

    /// Issue description
    pub description: String,

    /// Severity level
    pub severity: Severity,

    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Types of health issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueType {
    /// Memory-related issues
    Memory,
    /// CPU performance issues
    Performance,
    /// Network connectivity issues
    Network,
    /// Logic or algorithm errors
    Logic,
    /// Resource starvation
    Resources,
    /// Communication failures
    Communication,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Low severity - monitoring only
    Low,
    /// Medium severity - should be addressed
    Medium,
    /// High severity - needs immediate attention
    High,
    /// Critical severity - requires immediate action
    Critical,
}

/// Result of a healing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingResult {
    /// Whether healing was successful
    pub success: bool,

    /// Issues that were resolved
    pub resolved_issues: Vec<IssueType>,

    /// Issues that could not be resolved
    pub unresolved_issues: Vec<IssueType>,

    /// New health score after healing
    pub new_health_score: f64,

    /// Actions taken during healing
    pub actions_taken: Vec<String>,
}

/// Recovery strategies for self-healing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Restart the agent
    Restart,
    /// Reset to default state
    Reset,
    /// Rollback to previous state
    Rollback,
    /// Reallocate resources
    Reallocate,
    /// Reconfigure parameters
    Reconfigure,
    /// Request external help
    RequestHelp,
}

/// Current resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage ratio (0.0 to 1.0)
    pub memory_usage: f64,

    /// CPU usage ratio (0.0 to 1.0)
    pub cpu_usage: f64,

    /// Network bandwidth usage
    pub network_usage: f64,

    /// Number of active tasks
    pub active_tasks: usize,

    /// Queue length
    pub queue_length: usize,
}

/// Predicted resource needs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePrediction {
    /// Predicted memory needs
    pub memory_needs: f64,

    /// Predicted CPU needs
    pub cpu_needs: f64,

    /// Predicted network needs
    pub network_needs: f64,

    /// Confidence in prediction (0.0 to 1.0)
    pub confidence: f64,

    /// Time horizon for prediction
    pub time_horizon: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_diagnostic() {
        let diagnostic = HealthDiagnostic {
            health_score: 0.6,
            issues: vec![HealthIssue {
                issue_type: IssueType::Memory,
                description: "High memory usage".to_string(),
                severity: Severity::High,
                suggested_fix: Some("Free unused memory".to_string()),
            }],
            recommendations: vec!["Optimize memory usage".to_string()],
            severity: Severity::High,
        };

        assert!(diagnostic.needs_healing());
        assert_eq!(diagnostic.priority_issues().len(), 1);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
    }
}
