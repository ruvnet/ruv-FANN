//! Swarm trait definitions and common interfaces
//!
//! This module provides the core traits that all swarm implementations
//! must implement, enabling polymorphic usage and testing.

use crate::agent::{AgentId, AgentStatus, DynamicAgent};
use crate::error::Result;
use crate::task::{Task, TaskId};
use async_trait::async_trait;

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, vec::Vec};
#[cfg(feature = "std")]
use std::collections::HashMap;

/// Core swarm metrics that all swarm implementations should provide
#[derive(Debug, Clone)]
pub struct SwarmMetricsCore {
    /// Total number of agents in the swarm
    pub total_agents: usize,
    /// Number of agents currently active and available
    pub active_agents: usize,
    /// Number of tasks waiting in the queue
    pub queued_tasks: usize,
    /// Number of tasks currently assigned to agents
    pub assigned_tasks: usize,
    /// Total number of inter-agent connections
    pub total_connections: usize,
}

/// Synchronous swarm operations trait
pub trait SwarmSync {
    /// Register an agent with the swarm
    /// 
    /// # Errors
    /// 
    /// Returns an error if the agent cannot be registered.
    fn register_agent(&mut self, agent: DynamicAgent) -> Result<()>;
    
    /// Remove an agent from the swarm
    /// 
    /// # Errors
    /// 
    /// Returns an error if the agent cannot be removed.
    fn unregister_agent(&mut self, agent_id: &AgentId) -> Result<()>;
    
    /// Submit a task to the swarm
    /// 
    /// # Errors
    /// 
    /// Returns an error if the task cannot be submitted.
    fn submit_task(&mut self, task: Task) -> Result<()>;
    
    /// Get the status of all agents
    fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus>;
    
    /// Get the current task queue size
    fn task_queue_size(&self) -> usize;
    
    /// Get assigned tasks count
    fn assigned_tasks_count(&self) -> usize;
    
    /// Get agent by ID
    fn get_agent(&self, agent_id: &AgentId) -> Option<&DynamicAgent>;
    
    /// Get mutable agent by ID
    fn get_agent_mut(&mut self, agent_id: &AgentId) -> Option<&mut DynamicAgent>;
}

/// Asynchronous swarm operations trait
#[async_trait]
pub trait SwarmAsync {
    /// Register an agent with the swarm
    async fn register_agent(&self, agent: DynamicAgent) -> Result<()>;
    
    /// Remove an agent from the swarm
    async fn unregister_agent(&self, agent_id: &AgentId) -> Result<()>;
    
    /// Submit a task to the swarm
    async fn submit_task(&self, task: Task) -> Result<()>;
    
    /// Distribute tasks to agents based on distribution strategy
    async fn distribute_tasks(&self) -> Result<Vec<(TaskId, AgentId)>>;
    
    /// Get the status of all agents
    async fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus>;
    
    /// Get the current task queue size
    async fn task_queue_size(&self) -> usize;
    
    /// Get assigned tasks count
    async fn assigned_tasks_count(&self) -> usize;
    
    /// Check if agent exists by ID
    async fn has_agent(&self, agent_id: &AgentId) -> bool;
    
    /// Start all agents
    async fn start_all_agents(&self) -> Result<()>;
    
    /// Shutdown all agents
    async fn shutdown_all_agents(&self) -> Result<()>;
}

/// Mixed async/sync swarm operations trait
#[async_trait]
pub trait SwarmMixed: SwarmSync {
    /// Distribute tasks to agents (async version)
    async fn distribute_tasks(&mut self) -> Result<Vec<(TaskId, AgentId)>>;
    
    /// Start all agents (async version)
    async fn start_all_agents(&mut self) -> Result<()>;
    
    /// Shutdown all agents (async version)
    async fn shutdown_all_agents(&mut self) -> Result<()>;
}

/// Swarm orchestration trait that combines common operations
pub trait SwarmOrchestrator {
    /// The type of metrics this swarm provides
    type Metrics;
    
    /// Get swarm metrics
    fn metrics(&self) -> Self::Metrics;
    
    /// Check if the swarm is healthy
    fn is_healthy(&self) -> bool {
        let metrics = self.metrics();
        // Default health check - can be overridden by implementors
        true // Simplified for now
    }
    
    /// Get swarm configuration summary
    fn config_summary(&self) -> SwarmConfigSummary;
}

/// Summary of swarm configuration for debugging and monitoring
#[derive(Debug, Clone)]
pub struct SwarmConfigSummary {
    /// Maximum number of agents allowed
    pub max_agents: usize,
    /// Distribution strategy name
    pub distribution_strategy: &'static str,
    /// Topology type name
    pub topology_type: &'static str,
    /// Whether auto-scaling is enabled
    pub auto_scaling_enabled: bool,
    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,
}

/// Swarm factory trait for creating different types of swarms
pub trait SwarmFactory {
    /// The type of swarm this factory creates
    type Swarm;
    /// The type of configuration this factory uses
    type Config;
    
    /// Create a new swarm with the given configuration
    fn create_swarm(config: Self::Config) -> Self::Swarm;
}

/// Swarm builder trait for fluent configuration
pub trait SwarmBuilder {
    /// The type of swarm being built
    type Swarm;
    
    /// Set the maximum number of agents
    #[must_use]
    fn max_agents(self, max_agents: usize) -> Self;
    
    /// Enable or disable auto-scaling
    #[must_use]
    fn auto_scaling(self, enabled: bool) -> Self;
    
    /// Set the health check interval
    #[must_use]
    fn health_check_interval(self, interval_ms: u64) -> Self;
    
    /// Build the swarm with the configured parameters
    fn build(self) -> Self::Swarm;
}

/// Swarm lifecycle trait for managing swarm state
#[async_trait]
pub trait SwarmLifecycle {
    /// Initialize the swarm
    async fn initialize(&mut self) -> Result<()>;
    
    /// Start the swarm and all its agents
    async fn start(&mut self) -> Result<()>;
    
    /// Pause the swarm operations
    async fn pause(&mut self) -> Result<()>;
    
    /// Resume the swarm operations
    async fn resume(&mut self) -> Result<()>;
    
    /// Stop the swarm and all its agents
    async fn stop(&mut self) -> Result<()>;
    
    /// Get the current lifecycle state
    fn lifecycle_state(&self) -> SwarmLifecycleState;
}

/// Swarm lifecycle states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwarmLifecycleState {
    /// Swarm is uninitialized
    Uninitialized,
    /// Swarm is initialized but not started
    Initialized,
    /// Swarm is running normally
    Running,
    /// Swarm is paused
    Paused,
    /// Swarm is stopped
    Stopped,
    /// Swarm is in error state
    Error,
}

/// Swarm monitoring trait for observability
pub trait SwarmMonitoring {
    /// Get current swarm health status
    fn health_status(&self) -> SwarmHealthStatus;
    
    /// Get performance metrics
    fn performance_metrics(&self) -> SwarmPerformanceMetrics;
    
    /// Get error statistics
    fn error_statistics(&self) -> SwarmErrorStatistics;
}

/// Swarm health status
#[derive(Debug, Clone)]
pub struct SwarmHealthStatus {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,
    /// List of health issues
    pub issues: Vec<String>,
    /// Number of healthy agents
    pub healthy_agents: usize,
    /// Number of unhealthy agents
    pub unhealthy_agents: usize,
}

/// Swarm performance metrics
#[derive(Debug, Clone)]
pub struct SwarmPerformanceMetrics {
    /// Average task completion time in milliseconds
    pub avg_task_completion_ms: f64,
    /// Tasks completed per second
    pub tasks_per_second: f64,
    /// Average agent utilization (0.0 to 1.0)
    pub avg_agent_utilization: f64,
    /// Total throughput (tasks/hour)
    pub throughput_per_hour: f64,
}

/// Swarm error statistics
#[derive(Debug, Clone)]
pub struct SwarmErrorStatistics {
    /// Total number of errors
    pub total_errors: usize,
    /// Error rate (errors per hour)
    pub error_rate: f64,
    /// Most common error types
    pub common_errors: Vec<(String, usize)>,
    /// Error trend (increasing/decreasing)
    pub trend: ErrorTrend,
}

/// Error trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorTrend {
    /// Error rate is increasing
    Increasing,
    /// Error rate is stable
    Stable,
    /// Error rate is decreasing
    Decreasing,
}

/// Swarm testing trait for unit tests and integration tests
#[cfg(test)]
pub trait SwarmTesting {
    /// Create a test swarm with default configuration
    fn create_test_swarm() -> Self;
    
    /// Add a mock agent for testing
    fn add_mock_agent(&mut self, agent_id: &str);
    
    /// Add multiple mock agents for testing
    fn add_mock_agents(&mut self, count: usize);
    
    /// Submit a test task
    fn submit_test_task(&mut self, task_id: &str);
    
    /// Verify swarm state for testing
    /// 
    /// # Errors
    /// 
    /// Returns error if swarm state is invalid or verification fails.
    fn verify_state(&self) -> Result<()>;
}