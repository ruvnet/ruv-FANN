//! Core types and data structures for DAA operations

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, string::String, vec::Vec};

#[cfg(feature = "std")]
use std::collections::HashMap;

/// Capabilities that autonomous agents can possess
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AutonomousCapability {
    /// Self-monitoring and diagnosis
    SelfMonitoring,
    /// Autonomous decision making
    DecisionMaking,
    /// Resource optimization
    ResourceOptimization,
    /// Self-healing and recovery
    SelfHealing,
    /// Learning and adaptation
    Learning,
    /// Emergent behavior generation
    EmergentBehavior,
    /// Predictive analysis
    Prediction,
    /// Goal formation and planning
    GoalPlanning,
    /// Communication and coordination
    Coordination,
    /// Memory management
    MemoryManagement,
    /// Custom capability
    Custom(String),
}

impl AutonomousCapability {
    /// Get all standard capabilities
    pub fn standard_capabilities() -> &'static [AutonomousCapability] {
        &[
            AutonomousCapability::SelfMonitoring,
            AutonomousCapability::DecisionMaking,
            AutonomousCapability::ResourceOptimization,
            AutonomousCapability::SelfHealing,
            AutonomousCapability::Learning,
            AutonomousCapability::EmergentBehavior,
            AutonomousCapability::Prediction,
            AutonomousCapability::GoalPlanning,
            AutonomousCapability::Coordination,
            AutonomousCapability::MemoryManagement,
        ]
    }
    
    /// Check if capability requires neural processing
    pub fn requires_neural(&self) -> bool {
        matches!(self, 
            AutonomousCapability::Learning |
            AutonomousCapability::Prediction |
            AutonomousCapability::EmergentBehavior |
            AutonomousCapability::DecisionMaking
        )
    }
    
    /// Get computational complexity (1-10 scale)
    pub fn complexity(&self) -> u8 {
        match self {
            AutonomousCapability::SelfMonitoring => 2,
            AutonomousCapability::DecisionMaking => 8,
            AutonomousCapability::ResourceOptimization => 6,
            AutonomousCapability::SelfHealing => 5,
            AutonomousCapability::Learning => 9,
            AutonomousCapability::EmergentBehavior => 10,
            AutonomousCapability::Prediction => 7,
            AutonomousCapability::GoalPlanning => 8,
            AutonomousCapability::Coordination => 6,
            AutonomousCapability::MemoryManagement => 4,
            AutonomousCapability::Custom(_) => 5, // Default assumption
        }
    }
}

/// Context information for autonomous decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    /// Current environment state
    pub environment_state: EnvironmentState,
    
    /// Available actions
    pub available_actions: Vec<Action>,
    
    /// Current goals
    pub goals: Vec<Goal>,
    
    /// Historical context
    pub history: Vec<HistoricalEvent>,
    
    /// Resource constraints
    pub constraints: ResourceConstraints,
    
    /// Time pressure (0.0 = no pressure, 1.0 = immediate)
    pub time_pressure: f64,
    
    /// Uncertainty level (0.0 = certain, 1.0 = completely uncertain)
    pub uncertainty: f64,
}

impl DecisionContext {
    /// Create a new decision context
    pub fn new() -> Self {
        DecisionContext {
            environment_state: EnvironmentState::default(),
            available_actions: Vec::new(),
            goals: Vec::new(),
            history: Vec::new(),
            constraints: ResourceConstraints::default(),
            time_pressure: 0.0,
            uncertainty: 0.0,
        }
    }
    
    /// Add an available action
    pub fn add_action(&mut self, action: Action) {
        self.available_actions.push(action);
    }
    
    /// Add a goal
    pub fn add_goal(&mut self, goal: Goal) {
        self.goals.push(goal);
    }
    
    /// Check if context requires immediate action
    pub fn requires_immediate_action(&self) -> bool {
        self.time_pressure > 0.8
    }
    
    /// Get decision complexity score
    pub fn complexity_score(&self) -> f64 {
        let action_complexity = self.available_actions.len() as f64 * 0.1;
        let goal_complexity = self.goals.len() as f64 * 0.2;
        let uncertainty_factor = self.uncertainty;
        let time_factor = self.time_pressure;
        
        (action_complexity + goal_complexity + uncertainty_factor + time_factor).min(1.0)
    }
}

impl Default for DecisionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Environment state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentState {
    /// Environment type
    pub environment_type: EnvironmentType,
    
    /// Current conditions
    #[cfg(feature = "std")]
    pub conditions: HashMap<String, f64>,
    
    /// Current conditions (no-std version)
    #[cfg(not(feature = "std"))]
    pub conditions: BTreeMap<String, f64>,
    
    /// Stability measure (0.0 = chaotic, 1.0 = stable)
    pub stability: f64,
    
    /// Resource availability
    pub resource_availability: f64,
}

impl Default for EnvironmentState {
    fn default() -> Self {
        EnvironmentState {
            environment_type: EnvironmentType::Unknown,
            #[cfg(feature = "std")]
            conditions: HashMap::new(),
            #[cfg(not(feature = "std"))]
            conditions: BTreeMap::new(),
            stability: 0.5,
            resource_availability: 1.0,
        }
    }
}

/// Types of operating environments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvironmentType {
    /// Stable, predictable environment
    Stable,
    /// Dynamic, changing environment
    Dynamic,
    /// Hostile, adversarial environment
    Hostile,
    /// Resource-constrained environment
    Constrained,
    /// Unknown environment type
    Unknown,
}

/// Action that an agent can take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Action identifier
    pub id: String,
    
    /// Action type
    pub action_type: ActionType,
    
    /// Expected cost of performing this action
    pub cost: f64,
    
    /// Expected benefit/reward
    pub expected_reward: f64,
    
    /// Risk level (0.0 = safe, 1.0 = very risky)
    pub risk: f64,
    
    /// Prerequisites for this action
    pub prerequisites: Vec<String>,
    
    /// Expected duration
    pub duration: Option<u64>,
}

impl Action {
    /// Calculate action utility (benefit - cost - risk adjustment)
    pub fn utility(&self) -> f64 {
        self.expected_reward - self.cost - (self.risk * 0.5)
    }
    
    /// Check if action is viable given current resources
    pub fn is_viable(&self, available_resources: f64) -> bool {
        self.cost <= available_resources && self.risk <= 0.8
    }
}

/// Types of actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Computational action
    Compute,
    /// Communication action
    Communicate,
    /// Resource allocation action
    Allocate,
    /// Learning/training action
    Learn,
    /// Monitoring action
    Monitor,
    /// Healing/recovery action
    Heal,
    /// Coordination action
    Coordinate,
    /// Planning action
    Plan,
}

/// Goals that guide agent behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Goal identifier
    pub id: String,
    
    /// Goal description
    pub description: String,
    
    /// Goal type
    pub goal_type: GoalType,
    
    /// Priority level (1-10)
    pub priority: u8,
    
    /// Progress towards goal (0.0 to 1.0)
    pub progress: f64,
    
    /// Target completion time
    pub target_completion: Option<u64>,
    
    /// Success criteria
    pub success_criteria: Vec<String>,
}

impl Goal {
    /// Check if goal is completed
    pub fn is_completed(&self) -> bool {
        self.progress >= 1.0
    }
    
    /// Check if goal is overdue
    pub fn is_overdue(&self, current_time: u64) -> bool {
        if let Some(target) = self.target_completion {
            current_time > target
        } else {
            false
        }
    }
    
    /// Get urgency score based on priority and time remaining
    pub fn urgency_score(&self, current_time: u64) -> f64 {
        let priority_factor = f64::from(self.priority) / 10.0;
        let time_factor = if let Some(target) = self.target_completion {
            if current_time >= target {
                1.0 // Overdue
            } else {
                let remaining = target - current_time;
                1.0 - (remaining as f64 / target as f64).min(1.0)
            }
        } else {
            0.5 // No deadline
        };
        
        (priority_factor + time_factor) / 2.0
    }
}

/// Types of goals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalType {
    /// Performance optimization goal
    Performance,
    /// Learning and improvement goal
    Learning,
    /// Resource efficiency goal
    Efficiency,
    /// Task completion goal
    Task,
    /// Collaboration goal
    Collaboration,
    /// Survival/maintenance goal
    Survival,
}

/// Historical events for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalEvent {
    /// Event timestamp
    pub timestamp: u64,
    
    /// Event type
    pub event_type: EventType,
    
    /// Event description
    pub description: String,
    
    /// Outcome or result
    pub outcome: EventOutcome,
    
    /// Lessons learned
    pub lessons: Vec<String>,
}

/// Types of historical events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Decision was made
    Decision,
    /// Action was taken
    Action,
    /// Problem was encountered
    Problem,
    /// Success was achieved
    Success,
    /// Failure occurred
    Failure,
    /// Learning occurred
    Learning,
}

/// Outcomes of events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventOutcome {
    /// Positive outcome
    Positive,
    /// Negative outcome
    Negative,
    /// Neutral outcome
    Neutral,
    /// Mixed outcome
    Mixed,
}

/// Resource constraints for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory available (MB)
    pub max_memory_mb: u32,
    
    /// Maximum CPU usage (0.0 to 1.0)
    pub max_cpu_usage: f64,
    
    /// Maximum network bandwidth (Mbps)
    pub max_network_mbps: u32,
    
    /// Maximum execution time (seconds)
    pub max_execution_time: u64,
    
    /// Energy constraints (arbitrary units)
    pub energy_budget: f64,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        ResourceConstraints {
            max_memory_mb: 1024,
            max_cpu_usage: 0.8,
            max_network_mbps: 100,
            max_execution_time: 300,
            energy_budget: 1000.0,
        }
    }
}

/// Resource allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Memory allocation (MB)
    pub memory_mb: u32,
    
    /// CPU allocation (0.0 to 1.0)
    pub cpu_allocation: f64,
    
    /// Network allocation (Mbps)
    pub network_mbps: u32,
    
    /// Storage allocation (MB)
    pub storage_mb: u32,
    
    /// Priority level for resource access
    pub priority: ResourcePriority,
    
    /// Duration of allocation
    pub duration: Option<u64>,
}

/// Resource priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ResourcePriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Feedback for adaptation processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationFeedback {
    /// Performance score (0.0 to 1.0)
    pub performance_score: f64,
    
    /// Efficiency rating (0.0 to 1.0)
    pub efficiency_rating: f64,
    
    /// Quality metrics
    #[cfg(feature = "std")]
    pub quality_metrics: HashMap<String, f64>,
    
    /// Quality metrics (no-std version)
    #[cfg(not(feature = "std"))]
    pub quality_metrics: BTreeMap<String, f64>,
    
    /// Recommendation for improvement
    pub recommendations: Vec<String>,
    
    /// Learning rate adjustment
    pub learning_rate_adjustment: f64,
}

impl AdaptationFeedback {
    /// Create positive feedback
    pub fn positive(score: f64) -> Self {
        AdaptationFeedback {
            performance_score: score,
            efficiency_rating: score,
            #[cfg(feature = "std")]
            quality_metrics: HashMap::new(),
            #[cfg(not(feature = "std"))]
            quality_metrics: BTreeMap::new(),
            recommendations: Vec::new(),
            learning_rate_adjustment: 1.1, // Increase learning rate
        }
    }
    
    /// Create negative feedback
    pub fn negative(score: f64) -> Self {
        AdaptationFeedback {
            performance_score: score,
            efficiency_rating: score,
            #[cfg(feature = "std")]
            quality_metrics: HashMap::new(),
            #[cfg(not(feature = "std"))]
            quality_metrics: BTreeMap::new(),
            recommendations: Vec::new(),
            learning_rate_adjustment: 0.9, // Decrease learning rate
        }
    }
    
    /// Check if feedback indicates good performance
    pub fn is_positive(&self) -> bool {
        self.performance_score > 0.7 && self.efficiency_rating > 0.7
    }
    
    /// Get overall quality score
    pub fn overall_quality(&self) -> f64 {
        let metrics_avg = if self.quality_metrics.is_empty() {
            0.5
        } else {
            self.quality_metrics.values().sum::<f64>() / self.quality_metrics.len() as f64
        };
        
        (self.performance_score + self.efficiency_rating + metrics_avg) / 3.0
    }
}

/// Emergent state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentState {
    /// State identifier
    pub id: String,
    
    /// State type
    pub state_type: EmergentStateType,
    
    /// Emergence strength (0.0 to 1.0)
    pub strength: f64,
    
    /// Contributing factors
    pub factors: Vec<String>,
    
    /// Expected duration
    pub duration: Option<u64>,
    
    /// Impact on system
    pub impact: EmergentImpact,
}

/// Types of emergent states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergentStateType {
    /// Collective intelligence emergence
    CollectiveIntelligence,
    /// Swarm synchronization
    Synchronization,
    /// Adaptive specialization
    Specialization,
    /// Self-organization
    SelfOrganization,
    /// Pattern formation
    PatternFormation,
    /// Phase transition
    PhaseTransition,
}

/// Impact of emergent behaviors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentImpact {
    /// Performance change
    pub performance_delta: f64,
    
    /// Efficiency change
    pub efficiency_delta: f64,
    
    /// Stability change
    pub stability_delta: f64,
    
    /// Affected agents
    pub affected_agents: Vec<String>,
    
    /// Propagation rate
    pub propagation_rate: f64,
}

/// Cognitive model for advanced reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveModel {
    /// Model type
    pub model_type: CognitiveModelType,
    
    /// Processing capacity
    pub capacity: f64,
    
    /// Current load
    pub current_load: f64,
    
    /// Learning parameters
    #[cfg(feature = "std")]
    pub parameters: HashMap<String, f64>,
    
    /// Learning parameters (no-std version)
    #[cfg(not(feature = "std"))]
    pub parameters: BTreeMap<String, f64>,
    
    /// Model accuracy
    pub accuracy: f64,
}

/// Types of cognitive models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveModelType {
    /// Rule-based reasoning
    RuleBased,
    /// Neural network
    Neural,
    /// Bayesian inference
    Bayesian,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Evolutionary computation
    Evolutionary,
    /// Hybrid model
    Hybrid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autonomous_capability_complexity() {
        assert_eq!(AutonomousCapability::SelfMonitoring.complexity(), 2);
        assert_eq!(AutonomousCapability::EmergentBehavior.complexity(), 10);
    }

    #[test]
    fn test_action_utility() {
        let action = Action {
            id: "test".to_string(),
            action_type: ActionType::Compute,
            cost: 10.0,
            expected_reward: 20.0,
            risk: 0.2,
            prerequisites: Vec::new(),
            duration: None,
        };
        
        assert_eq!(action.utility(), 9.9); // 20 - 10 - (0.2 * 0.5)
    }

    #[test]
    fn test_goal_completion() {
        let goal = Goal {
            id: "test".to_string(),
            description: "Test goal".to_string(),
            goal_type: GoalType::Task,
            priority: 5,
            progress: 1.0,
            target_completion: None,
            success_criteria: Vec::new(),
        };
        
        assert!(goal.is_completed());
    }

    #[test]
    fn test_adaptation_feedback() {
        let feedback = AdaptationFeedback::positive(0.8);
        assert!(feedback.is_positive());
        assert!(feedback.learning_rate_adjustment > 1.0);
        
        let feedback = AdaptationFeedback::negative(0.3);
        assert!(!feedback.is_positive());
        assert!(feedback.learning_rate_adjustment < 1.0);
    }
}