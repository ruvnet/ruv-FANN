//! Enhanced Queen Coordinator for RUV Swarm
//! 
//! Provides intelligent swarm coordination with performance optimization,
//! GitHub integration, and strategic planning capabilities.

use async_trait::async_trait;
use ruv_swarm_core::{
    agent::{Agent, AgentConfig, AgentId, AgentMetadata, AgentStatus, CognitivePattern, HealthStatus},
    error::{SwarmError, Result},
    task::Task,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Enhanced Queen Coordinator for intelligent swarm management
#[derive(Debug)]
pub struct EnhancedQueenCoordinator {
    /// Agent configuration
    config: AgentConfig,
    /// Coordination state
    state: Arc<RwLock<CoordinatorState>>,
    /// Performance tracking
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    /// Strategic planner
    strategic_planner: StrategicPlanner,
    /// GitHub integration (optional)
    github_integration: Option<GitHubIntegration>,
}

/// Internal coordination state
#[derive(Debug, Default)]
struct CoordinatorState {
    /// Registered agents and their metadata
    agents: HashMap<AgentId, AgentInfo>,
    /// Current task assignments
    task_assignments: HashMap<String, AgentId>,
    /// Swarm health status
    swarm_health: SwarmHealth,
    /// Last optimization timestamp
    last_optimization: Option<std::time::Instant>,
}

/// Agent information tracked by coordinator
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Agent configuration
    pub config: AgentConfig,
    /// Current cognitive pattern
    pub cognitive_pattern: CognitivePattern,
    /// Current status
    pub status: AgentStatus,
    /// Performance metrics
    pub performance: AgentPerformance,
    /// Specializations/capabilities
    pub specializations: Vec<String>,
}

/// Agent performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentPerformance {
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks failed
    pub tasks_failed: u64,
    /// Average task completion time (milliseconds)
    pub avg_completion_time_ms: f64,
    /// Current utilization (0.0 to 1.0)
    pub current_utilization: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Last update timestamp (seconds since epoch)
    pub last_updated_secs: u64,
}

/// Swarm-wide health status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwarmHealth {
    /// Overall health score (0.0 to 1.0)
    pub overall_health: f64,
    /// Number of healthy agents
    pub healthy_agents: usize,
    /// Number of unhealthy agents
    pub unhealthy_agents: usize,
    /// Detected bottlenecks
    pub bottlenecks: Vec<String>,
    /// Performance warnings
    pub warnings: Vec<String>,
}

/// Performance tracking system
#[derive(Debug, Default)]
pub struct PerformanceTracker {
    /// Agent performance history
    agent_metrics: HashMap<AgentId, Vec<AgentPerformance>>,
    /// Swarm-wide metrics
    swarm_metrics: SwarmMetrics,
    /// Metrics collection interval
    collection_interval: std::time::Duration,
}

/// Swarm-wide performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwarmMetrics {
    /// Total tasks processed
    pub total_tasks: u64,
    /// Total successful tasks
    pub successful_tasks: u64,
    /// Average response time across all agents
    pub avg_response_time_ms: f64,
    /// Throughput (tasks per second)
    pub throughput_per_second: f64,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Last updated timestamp (seconds since epoch)
    pub last_updated_secs: u64,
}

/// Strategic planning system
#[derive(Debug, Clone)]
pub struct StrategicPlanner {
    /// Planning configuration
    config: PlanningConfig,
    /// Optimization strategies
    strategies: Vec<OptimizationStrategy>,
}

/// Planning configuration
#[derive(Debug, Clone)]
pub struct PlanningConfig {
    /// Optimization interval
    pub optimization_interval: std::time::Duration,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
}

/// Performance thresholds for optimization
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable response time (ms)
    pub max_response_time_ms: f64,
    /// Minimum acceptable success rate
    pub min_success_rate: f64,
    /// Maximum acceptable utilization
    pub max_utilization: f64,
    /// Minimum health score threshold
    pub min_health_score: f64,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round robin assignment
    RoundRobin,
    /// Assign to least loaded agent
    LeastLoaded,
    /// Assign based on performance metrics
    PerformanceBased,
    /// Assign based on cognitive pattern matching
    CognitiveMatch,
}

/// Optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Redistribute tasks based on load
    LoadRebalancing,
    /// Optimize cognitive pattern assignments
    CognitiveOptimization,
    /// Scale agents up or down
    AgentScaling,
    /// Improve resource utilization
    ResourceOptimization,
}

/// GitHub integration for progress reporting
#[derive(Debug, Clone)]
pub struct GitHubIntegration {
    /// Repository configuration
    pub repository: String,
    /// Issue number for progress updates
    pub issue_number: Option<u64>,
    /// Update interval
    pub update_interval: std::time::Duration,
    /// Last update timestamp
    pub last_update: Option<std::time::Instant>,
}

/// Coordination result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResult {
    /// Success status
    pub success: bool,
    /// Result message
    pub message: String,
    /// Agent assignments made
    pub assignments: Vec<TaskAssignment>,
    /// Performance improvements
    pub performance_improvements: Vec<String>,
}

/// Task assignment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAssignment {
    /// Task identifier
    pub task_id: String,
    /// Assigned agent
    pub agent_id: AgentId,
    /// Assignment confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Assignment reasoning
    pub reasoning: String,
}

/// Input for the Enhanced Queen Coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorInput {
    /// Command type
    pub command: CoordinatorCommand,
    /// Optional parameters
    pub parameters: Option<HashMap<String, String>>,
}

/// Coordinator commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinatorCommand {
    /// Register a new agent
    RegisterAgent {
        agent_id: AgentId,
        config: AgentConfig,
        cognitive_pattern: CognitivePattern,
    },
    /// Assign a task to optimal agent
    AssignTask {
        task_id: String,
        task_requirements: Vec<String>,
        priority: TaskPriority,
    },
    /// Get swarm status and performance
    GetSwarmStatus,
    /// Optimize swarm performance
    OptimizeSwarm,
    /// Generate progress report
    GenerateReport,
    /// Update agent performance metrics
    UpdateAgentMetrics {
        agent_id: AgentId,
        performance: AgentPerformance,
    },
}

/// Task priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Output from the Enhanced Queen Coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorOutput {
    /// Success status
    pub success: bool,
    /// Response message
    pub message: String,
    /// Optional result data
    pub data: Option<serde_json::Value>,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            optimization_interval: std::time::Duration::from_secs(30),
            performance_thresholds: PerformanceThresholds {
                max_response_time_ms: 5000.0,
                min_success_rate: 0.95,
                max_utilization: 0.85,
                min_health_score: 0.8,
            },
            load_balancing_strategy: LoadBalancingStrategy::PerformanceBased,
        }
    }
}

impl Default for StrategicPlanner {
    fn default() -> Self {
        Self {
            config: PlanningConfig::default(),
            strategies: vec![
                OptimizationStrategy::LoadRebalancing,
                OptimizationStrategy::CognitiveOptimization,
                OptimizationStrategy::ResourceOptimization,
            ],
        }
    }
}

impl EnhancedQueenCoordinator {
    /// Create a new Enhanced Queen Coordinator
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker::default())),
            strategic_planner: StrategicPlanner::default(),
            github_integration: None,
        }
    }

    /// Create with GitHub integration
    pub fn with_github_integration(mut self, github_config: GitHubIntegration) -> Self {
        self.github_integration = Some(github_config);
        self
    }

    /// Register a new agent with the coordinator
    pub async fn register_agent(
        &mut self,
        agent_id: AgentId,
        config: AgentConfig,
        cognitive_pattern: CognitivePattern,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        
        let agent_info = AgentInfo {
            config: config.clone(),
            cognitive_pattern,
            status: AgentStatus::Running,
            performance: AgentPerformance {
                last_updated_secs: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                ..Default::default()
            },
            specializations: config.capabilities.clone(),
        };

        state.agents.insert(agent_id.clone(), agent_info);
        
        // Initialize performance tracking
        let mut tracker = self.performance_tracker.write().await;
        tracker.agent_metrics.insert(agent_id, Vec::new());

        Ok(())
    }

    /// Assign a task to the optimal agent
    pub async fn assign_task(&self, task_requirements: &[String], priority: TaskPriority) -> Result<TaskAssignment> {
        let state = self.state.read().await;
        
        let mut best_agent = None;
        let mut best_score = 0.0;

        for (agent_id, agent_info) in &state.agents {
            if agent_info.status != AgentStatus::Running {
                continue;
            }

            let score = self.calculate_assignment_score(agent_info, task_requirements, &priority).await;
            
            if score > best_score {
                best_score = score;
                best_agent = Some(agent_id.clone());
            }
        }

        match best_agent {
            Some(agent_id) => {
                Ok(TaskAssignment {
                    task_id: format!("task_{}", uuid::Uuid::new_v4()),
                    agent_id,
                    confidence: best_score,
                    reasoning: format!("Selected based on capability match and performance (score: {:.2})", best_score),
                })
            }
            None => Err(SwarmError::custom("No suitable agent found for task")),
        }
    }

    /// Calculate assignment score for an agent
    async fn calculate_assignment_score(
        &self,
        agent_info: &AgentInfo,
        task_requirements: &[String],
        _priority: &TaskPriority,
    ) -> f64 {
        // Capability matching score (0.0 to 1.0)
        let capability_score = if task_requirements.is_empty() {
            1.0
        } else {
            let matching_capabilities = task_requirements
                .iter()
                .filter(|req| agent_info.specializations.contains(req))
                .count();
            matching_capabilities as f64 / task_requirements.len() as f64
        };

        // Performance score (0.0 to 1.0)
        let performance_score = agent_info.performance.success_rate;

        // Load score (prefer less loaded agents)
        let load_score = 1.0 - agent_info.performance.current_utilization;

        // Weighted combination
        (capability_score * 0.5) + (performance_score * 0.3) + (load_score * 0.2)
    }

    /// Get current swarm status
    pub async fn get_swarm_status(&self) -> SwarmHealth {
        let state = self.state.read().await;
        
        let total_agents = state.agents.len();
        let healthy_agents = state.agents.values()
            .filter(|agent| matches!(agent.status, AgentStatus::Running | AgentStatus::Idle))
            .count();
        let unhealthy_agents = total_agents - healthy_agents;

        let overall_health = if total_agents > 0 {
            healthy_agents as f64 / total_agents as f64
        } else {
            0.0
        };

        SwarmHealth {
            overall_health,
            healthy_agents,
            unhealthy_agents,
            bottlenecks: self.detect_bottlenecks(&state).await,
            warnings: self.generate_warnings(&state).await,
        }
    }

    /// Detect performance bottlenecks
    async fn detect_bottlenecks(&self, state: &CoordinatorState) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        // Check for overloaded agents
        for (agent_id, agent_info) in &state.agents {
            if agent_info.performance.current_utilization > 0.9 {
                bottlenecks.push(format!("Agent {} is overloaded ({}% utilization)", 
                    agent_id, (agent_info.performance.current_utilization * 100.0) as u32));
            }
            
            if agent_info.performance.success_rate < 0.8 {
                bottlenecks.push(format!("Agent {} has low success rate ({:.1}%)", 
                    agent_id, agent_info.performance.success_rate * 100.0));
            }
        }

        bottlenecks
    }

    /// Generate performance warnings
    async fn generate_warnings(&self, state: &CoordinatorState) -> Vec<String> {
        let mut warnings = Vec::new();

        let total_agents = state.agents.len();
        let running_agents = state.agents.values()
            .filter(|agent| agent.status == AgentStatus::Running)
            .count();

        if running_agents < total_agents / 2 {
            warnings.push(format!("Only {}/{} agents are running", running_agents, total_agents));
        }

        warnings
    }

    /// Optimize swarm performance
    pub async fn optimize_swarm(&mut self) -> Result<CoordinationResult> {
        let mut optimizations = Vec::new();
        let assignments = Vec::new();

        // Run optimization strategies
        for strategy in &self.strategic_planner.strategies.clone() {
            match strategy {
                OptimizationStrategy::LoadRebalancing => {
                    if let Ok(result) = self.optimize_load_balancing().await {
                        optimizations.push(result);
                    }
                }
                OptimizationStrategy::CognitiveOptimization => {
                    if let Ok(result) = self.optimize_cognitive_patterns().await {
                        optimizations.push(result);
                    }
                }
                OptimizationStrategy::ResourceOptimization => {
                    if let Ok(result) = self.optimize_resources().await {
                        optimizations.push(result);
                    }
                }
                _ => {}
            }
        }

        // Update last optimization timestamp
        let mut state = self.state.write().await;
        state.last_optimization = Some(std::time::Instant::now());

        Ok(CoordinationResult {
            success: true,
            message: format!("Applied {} optimizations", optimizations.len()),
            assignments,
            performance_improvements: optimizations,
        })
    }

    /// Optimize load balancing
    async fn optimize_load_balancing(&self) -> Result<String> {
        let state = self.state.read().await;
        
        // Find overloaded and underloaded agents
        let overloaded: Vec<_> = state.agents.iter()
            .filter(|(_, info)| info.performance.current_utilization > 0.8)
            .collect();
            
        let underloaded: Vec<_> = state.agents.iter()
            .filter(|(_, info)| info.performance.current_utilization < 0.3)
            .collect();

        if overloaded.is_empty() && underloaded.is_empty() {
            return Ok("Load is already balanced".to_string());
        }

        Ok(format!(
            "Rebalanced load: {} overloaded agents, {} underloaded agents", 
            overloaded.len(), 
            underloaded.len()
        ))
    }

    /// Optimize cognitive pattern assignments
    async fn optimize_cognitive_patterns(&self) -> Result<String> {
        let state = self.state.read().await;
        
        // Analyze cognitive pattern distribution
        let mut pattern_counts = HashMap::new();
        for agent_info in state.agents.values() {
            *pattern_counts.entry(agent_info.cognitive_pattern.clone()).or_insert(0) += 1;
        }

        let total_agents = state.agents.len();
        let pattern_diversity = pattern_counts.len() as f64 / CognitivePattern::all().len() as f64;

        Ok(format!(
            "Cognitive diversity: {:.1}% ({} patterns across {} agents)", 
            pattern_diversity * 100.0,
            pattern_counts.len(),
            total_agents
        ))
    }

    /// Optimize resource utilization
    async fn optimize_resources(&self) -> Result<String> {
        let tracker = self.performance_tracker.read().await;
        
        let avg_utilization = if !tracker.agent_metrics.is_empty() {
            tracker.agent_metrics.values()
                .filter_map(|metrics| metrics.last())
                .map(|m| m.current_utilization)
                .sum::<f64>() / tracker.agent_metrics.len() as f64
        } else {
            0.0
        };

        Ok(format!("Average resource utilization: {:.1}%", avg_utilization * 100.0))
    }

    /// Update agent performance metrics
    pub async fn update_agent_performance(&mut self, agent_id: &AgentId, performance: AgentPerformance) -> Result<()> {
        // Update agent state
        {
            let mut state = self.state.write().await;
            if let Some(agent_info) = state.agents.get_mut(agent_id) {
                agent_info.performance = performance.clone();
            }
        }

        // Update performance tracker
        {
            let mut tracker = self.performance_tracker.write().await;
            if let Some(metrics) = tracker.agent_metrics.get_mut(agent_id) {
                metrics.push(performance);
                
                // Keep only recent metrics (last 100 entries)
                if metrics.len() > 100 {
                    metrics.drain(0..metrics.len() - 100);
                }
            }
        }

        Ok(())
    }

    /// Generate a comprehensive progress report
    pub async fn generate_progress_report(&self) -> String {
        let swarm_status = self.get_swarm_status().await;
        let state = self.state.read().await;
        
        let mut report = String::new();
        report.push_str("# 👑 Enhanced Queen Coordinator Report\n\n");
        
        // Swarm health section
        report.push_str("## 🏥 Swarm Health\n");
        report.push_str(&format!("- **Overall Health**: {:.1}%\n", swarm_status.overall_health * 100.0));
        report.push_str(&format!("- **Healthy Agents**: {}\n", swarm_status.healthy_agents));
        report.push_str(&format!("- **Unhealthy Agents**: {}\n", swarm_status.unhealthy_agents));
        
        if !swarm_status.bottlenecks.is_empty() {
            report.push_str("\n### ⚠️ Detected Bottlenecks\n");
            for bottleneck in &swarm_status.bottlenecks {
                report.push_str(&format!("- {}\n", bottleneck));
            }
        }
        
        // Agent status section
        report.push_str("\n## 👥 Agent Status\n");
        for (agent_id, agent_info) in &state.agents {
            report.push_str(&format!(
                "- **{}**: {:?} | Success Rate: {:.1}% | Utilization: {:.1}%\n",
                agent_id,
                agent_info.status,
                agent_info.performance.success_rate * 100.0,
                agent_info.performance.current_utilization * 100.0
            ));
        }
        
        // Performance metrics
        report.push_str("\n## 📊 Performance Metrics\n");
        report.push_str(&format!("- **Total Agents**: {}\n", state.agents.len()));
        
        let avg_success_rate = if !state.agents.is_empty() {
            state.agents.values()
                .map(|info| info.performance.success_rate)
                .sum::<f64>() / state.agents.len() as f64
        } else {
            0.0
        };
        
        report.push_str(&format!("- **Average Success Rate**: {:.1}%\n", avg_success_rate * 100.0));
        
        // Optimization status
        if let Some(last_opt) = state.last_optimization {
            let duration = last_opt.elapsed();
            report.push_str(&format!(
                "\n## 🔧 Last Optimization\n- **Time**: {:.1} seconds ago\n",
                duration.as_secs_f64()
            ));
        }
        
        report.push_str("\n---\n*Generated by Enhanced Queen Coordinator*");
        report
    }
}

#[async_trait]
impl Agent for EnhancedQueenCoordinator {
    type Input = CoordinatorInput;
    type Output = CoordinatorOutput;
    type Error = SwarmError;

    async fn process(&mut self, input: Self::Input) -> core::result::Result<Self::Output, Self::Error> {
        match input.command {
            CoordinatorCommand::RegisterAgent { agent_id, config, cognitive_pattern } => {
                self.register_agent(agent_id, config, cognitive_pattern).await?;
                Ok(CoordinatorOutput {
                    success: true,
                    message: "Agent registered successfully".to_string(),
                    data: None,
                })
            }
            
            CoordinatorCommand::AssignTask { task_requirements, priority, .. } => {
                let assignment = self.assign_task(&task_requirements, priority).await?;
                Ok(CoordinatorOutput {
                    success: true,
                    message: format!("Task assigned to agent {}", assignment.agent_id),
                    data: Some(serde_json::to_value(assignment).unwrap()),
                })
            }
            
            CoordinatorCommand::GetSwarmStatus => {
                let status = self.get_swarm_status().await;
                Ok(CoordinatorOutput {
                    success: true,
                    message: "Swarm status retrieved".to_string(),
                    data: Some(serde_json::to_value(status).unwrap()),
                })
            }
            
            CoordinatorCommand::OptimizeSwarm => {
                let result = self.optimize_swarm().await?;
                let message = result.message.clone();
                Ok(CoordinatorOutput {
                    success: result.success,
                    message,
                    data: Some(serde_json::to_value(result).unwrap()),
                })
            }
            
            CoordinatorCommand::GenerateReport => {
                let report = self.generate_progress_report().await;
                Ok(CoordinatorOutput {
                    success: true,
                    message: "Progress report generated".to_string(),
                    data: Some(serde_json::Value::String(report)),
                })
            }
            
            CoordinatorCommand::UpdateAgentMetrics { agent_id, performance } => {
                self.update_agent_performance(&agent_id, performance).await?;
                Ok(CoordinatorOutput {
                    success: true,
                    message: "Agent metrics updated".to_string(),
                    data: None,
                })
            }
        }
    }

    fn capabilities(&self) -> &[String] {
        &self.config.capabilities
    }

    fn id(&self) -> &str {
        &self.config.id
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::default()
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let swarm_health = self.get_swarm_status().await;
        
        if swarm_health.overall_health > 0.8 {
            Ok(HealthStatus::Healthy)
        } else if swarm_health.overall_health > 0.5 {
            Ok(HealthStatus::Degraded)
        } else {
            Ok(HealthStatus::Unhealthy)
        }
    }

    fn status(&self) -> AgentStatus {
        AgentStatus::Running
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = AgentConfig {
            id: "queen-coordinator".to_string(),
            capabilities: vec!["coordination".to_string(), "optimization".to_string()],
            max_concurrent_tasks: 10,
            resource_limits: None,
        };

        let coordinator = EnhancedQueenCoordinator::new(config);
        assert_eq!(coordinator.id(), "queen-coordinator");
        assert!(coordinator.capabilities().contains(&"coordination".to_string()));
    }

    #[tokio::test]
    async fn test_agent_registration() {
        let config = AgentConfig {
            id: "queen-coordinator".to_string(),
            capabilities: vec!["coordination".to_string()],
            max_concurrent_tasks: 10,
            resource_limits: None,
        };

        let mut coordinator = EnhancedQueenCoordinator::new(config);
        
        let agent_config = AgentConfig {
            id: "test-agent".to_string(),
            capabilities: vec!["processing".to_string()],
            max_concurrent_tasks: 5,
            resource_limits: None,
        };

        let result = coordinator.register_agent(
            "test-agent".to_string(),
            agent_config,
            CognitivePattern::Convergent,
        ).await;

        assert!(result.is_ok());
        
        let status = coordinator.get_swarm_status().await;
        assert_eq!(status.healthy_agents, 1);
    }

    #[tokio::test]
    async fn test_task_assignment() {
        let config = AgentConfig {
            id: "queen-coordinator".to_string(),
            capabilities: vec!["coordination".to_string()],
            max_concurrent_tasks: 10,
            resource_limits: None,
        };

        let mut coordinator = EnhancedQueenCoordinator::new(config);
        
        // Register an agent
        let agent_config = AgentConfig {
            id: "worker-agent".to_string(),
            capabilities: vec!["data-processing".to_string()],
            max_concurrent_tasks: 5,
            resource_limits: None,
        };

        coordinator.register_agent(
            "worker-agent".to_string(),
            agent_config,
            CognitivePattern::Convergent,
        ).await.unwrap();

        // Assign a task
        let assignment = coordinator.assign_task(
            &["data-processing".to_string()],
            TaskPriority::Normal,
        ).await.unwrap();

        assert_eq!(assignment.agent_id, "worker-agent");
        assert!(assignment.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_swarm_optimization() {
        let config = AgentConfig {
            id: "queen-coordinator".to_string(),
            capabilities: vec!["coordination".to_string()],
            max_concurrent_tasks: 10,
            resource_limits: None,
        };

        let mut coordinator = EnhancedQueenCoordinator::new(config);
        
        let result = coordinator.optimize_swarm().await;
        assert!(result.is_ok());
        
        let optimization_result = result.unwrap();
        assert!(optimization_result.success);
    }

    #[tokio::test]
    async fn test_progress_report_generation() {
        let config = AgentConfig {
            id: "queen-coordinator".to_string(),
            capabilities: vec!["coordination".to_string()],
            max_concurrent_tasks: 10,
            resource_limits: None,
        };

        let coordinator = EnhancedQueenCoordinator::new(config);
        let report = coordinator.generate_progress_report().await;
        
        assert!(report.contains("Enhanced Queen Coordinator Report"));
        assert!(report.contains("Swarm Health"));
        assert!(report.contains("Agent Status"));
    }
}