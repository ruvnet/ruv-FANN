//! Types specific to MCP server implementation

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Agent type for MCP
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    Researcher,
    Coder,
    Analyst,
    Tester,
    Reviewer,
    Documenter,
}

/// Agent capabilities for MCP
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub languages: Vec<String>,
    pub frameworks: Vec<String>,
    pub tools: Vec<String>,
    pub specializations: Vec<String>,
    pub max_concurrent_tasks: usize,
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Swarm strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmStrategy {
    Research,
    Development,
    Analysis,
    Testing,
    Optimization,
    Maintenance,
}

/// Coordination mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationMode {
    Centralized,
    Distributed,
    Hierarchical,
    Mesh,
    Hybrid,
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub strategy: SwarmStrategy,
    pub mode: CoordinationMode,
    pub max_agents: usize,
    pub parallel: bool,
    pub timeout: std::time::Duration,
}

/// Swarm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmState {
    pub agents: Vec<AgentInfo>,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub total_agents: usize,
}

/// Agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: Uuid,
    pub agent_type: AgentType,
    pub name: Option<String>,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub current_tasks: Vec<Uuid>,
}

/// Swarm metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
    pub total_agents: usize,
    pub active_agents: usize,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub average_response_time: f64,
    pub success_rate: f64,
    pub resource_utilization: f64,
}

/// Swarm status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    pub is_running: bool,
    pub uptime_secs: u64,
    pub version: String,
    pub config: serde_json::Value,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub impact: String,
    pub priority: TaskPriority,
    pub estimated_improvement: f64,
}

/// Workflow result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub success: bool,
    pub steps_completed: usize,
    pub total_steps: usize,
    pub outputs: serde_json::Value,
    pub errors: Vec<String>,
    pub duration_ms: u64,
}

/// Task creation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCreationResult {
    pub task_id: Uuid,
    pub assigned_agent: Option<Uuid>,
    pub estimated_completion_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// Orchestration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResult {
    pub task_id: Uuid,
    pub success: bool,
    pub agents_used: Vec<Uuid>,
    pub duration_ms: u64,
    pub outputs: serde_json::Value,
}

// Response types for MCP tools

/// Response for agent spawning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnAgentResponse {
    pub agent_id: Uuid,
    pub status: String,
    pub message: String,
}

/// Response for task orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResponse {
    pub task_id: Uuid,
    pub status: String,
    pub assigned_agents: Vec<Uuid>,
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
    pub message: String,
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub agent_id: Uuid,
    pub data: serde_json::Value,
    pub confidence: f64,
}

/// Query response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub query: String,
    pub results: Vec<QueryResult>,
    pub total_matches: usize,
    pub execution_time_ms: u64,
}

/// Monitoring response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringResponse {
    pub status: String,
    pub metrics: SwarmMetrics,
    pub alerts: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}


/// Optimization response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResponse {
    pub target_metric: String,
    pub current_value: f64,
    pub target_value: f64,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub estimated_improvement: f64,
    pub implementation_status: String,
    pub message: String,
}

/// Memory operation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResponse {
    pub key: String,
    pub operation: String,
    pub success: bool,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Task definition for creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    pub task_type: String,
    pub description: String,
    pub requirements: Option<Vec<String>>,
}

/// Task response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponse {
    pub task_id: Uuid,
    pub task_type: String,
    pub description: String,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub estimated_duration: Option<std::time::Duration>,
    pub assigned_agents: Vec<Uuid>,
    pub progress: f64,
    pub message: String,
}

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub name: String,
    pub steps: Vec<WorkflowStep>,
}

/// Workflow step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub name: String,
    pub task_type: String,
    pub dependencies: Vec<String>,
}

/// Workflow response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResponse {
    pub workflow_id: Uuid,
    pub workflow_name: String,
    pub status: String,
    pub steps_completed: usize,
    pub total_steps: usize,
    pub current_step: Option<String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
    pub message: String,
}

/// Agent list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentListResponse {
    pub agents: Vec<AgentInfo>,
    pub total_count: usize,
    pub active_count: usize,
    pub filter_applied: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Agent metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub agent_id: Uuid,
    pub response_time: f64,
    pub tasks_completed: usize,
    pub success_rate: f64,
    pub error_count: usize,
}

/// Agent metrics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetricsResponse {
    pub metrics: Vec<AgentMetrics>,
    pub time_range: String,
    pub total_agents: usize,
    pub summary: AgentMetricsSummary,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Agent metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetricsSummary {
    pub average_response_time: f64,
    pub total_tasks_completed: usize,
    pub average_success_rate: f64,
    pub total_errors: usize,
}
