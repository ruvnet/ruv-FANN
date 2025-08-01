//! Comprehensive tests for swarm traits
//! Tests all the trait abstractions to ensure they work correctly

use crate::swarm_trait::*;
use crate::agent::{AgentId, AgentStatus, DynamicAgent};
use crate::task::{Task, TaskId};
use crate::error::{Result, SwarmError};
use async_trait::async_trait;
use std::collections::HashMap;

// Mock implementation of SwarmSync trait
struct MockSwarmSync {
    agents: HashMap<AgentId, DynamicAgent>,
    tasks: Vec<Task>,
    task_assignments: HashMap<TaskId, AgentId>,
}

impl MockSwarmSync {
    fn new() -> Self {
        Self {
            agents: HashMap::new(),
            tasks: Vec::new(),
            task_assignments: HashMap::new(),
        }
    }
}

impl SwarmSync for MockSwarmSync {
    fn register_agent(&mut self, agent: DynamicAgent) -> Result<()> {
        if self.agents.len() >= 10 {
            return Err(SwarmError::ResourceExhausted {
                resource: "agent slots".into(),
            });
        }
        let id = agent.id().to_string();
        self.agents.insert(id, agent);
        Ok(())
    }
    
    fn unregister_agent(&mut self, agent_id: &AgentId) -> Result<()> {
        self.agents.remove(agent_id)
            .ok_or_else(|| SwarmError::AgentNotFound { id: agent_id.clone() })?;
        Ok(())
    }
    
    fn submit_task(&mut self, task: Task) -> Result<()> {
        self.tasks.push(task);
        Ok(())
    }
    
    fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus> {
        self.agents.iter()
            .map(|(id, agent)| (id.clone(), agent.status()))
            .collect()
    }
    
    fn task_queue_size(&self) -> usize {
        self.tasks.len()
    }
    
    fn assigned_tasks_count(&self) -> usize {
        self.task_assignments.len()
    }
    
    fn get_agent(&self, agent_id: &AgentId) -> Option<&DynamicAgent> {
        self.agents.get(agent_id)
    }
    
    fn get_agent_mut(&mut self, agent_id: &AgentId) -> Option<&mut DynamicAgent> {
        self.agents.get_mut(agent_id)
    }
}

// Mock implementation of SwarmAsync trait
struct MockSwarmAsync {
    agents: std::sync::Arc<tokio::sync::RwLock<HashMap<AgentId, DynamicAgent>>>,
    tasks: std::sync::Arc<tokio::sync::Mutex<Vec<Task>>>,
    task_assignments: std::sync::Arc<tokio::sync::RwLock<HashMap<TaskId, AgentId>>>,
}

impl MockSwarmAsync {
    fn new() -> Self {
        Self {
            agents: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            tasks: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
            task_assignments: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl SwarmAsync for MockSwarmAsync {
    async fn register_agent(&self, agent: DynamicAgent) -> Result<()> {
        let mut agents = self.agents.write().await;
        if agents.len() >= 10 {
            return Err(SwarmError::ResourceExhausted {
                resource: "agent slots".into(),
            });
        }
        let id = agent.id().to_string();
        agents.insert(id, agent);
        Ok(())
    }
    
    async fn unregister_agent(&self, agent_id: &AgentId) -> Result<()> {
        let mut agents = self.agents.write().await;
        agents.remove(agent_id)
            .ok_or_else(|| SwarmError::AgentNotFound { id: agent_id.clone() })?;
        Ok(())
    }
    
    async fn submit_task(&self, task: Task) -> Result<()> {
        let mut tasks = self.tasks.lock().await;
        tasks.push(task);
        Ok(())
    }
    
    async fn distribute_tasks(&self) -> Result<Vec<(TaskId, AgentId)>> {
        let mut tasks = self.tasks.lock().await;
        let agents = self.agents.read().await;
        let mut assignments = Vec::new();
        
        // Simple round-robin distribution
        let agent_ids: Vec<AgentId> = agents.keys().cloned().collect();
        if agent_ids.is_empty() {
            return Ok(assignments);
        }
        
        let mut next_agent = 0;
        while let Some(task) = tasks.pop() {
            let task_id = task.id.clone();
            let agent_id = &agent_ids[next_agent % agent_ids.len()];
            
            // Check if agent can handle task
            if let Some(agent) = agents.get(agent_id) {
                if agent.can_handle(&task) {
                    assignments.push((task_id, agent_id.clone()));
                    next_agent += 1;
                } else {
                    // Put task back if no agent can handle it
                    tasks.push(task);
                    break;
                }
            }
        }
        
        Ok(assignments)
    }
    
    async fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus> {
        let agents = self.agents.read().await;
        agents.iter()
            .map(|(id, agent)| (id.clone(), agent.status()))
            .collect()
    }
    
    async fn task_queue_size(&self) -> usize {
        let tasks = self.tasks.lock().await;
        tasks.len()
    }
    
    async fn assigned_tasks_count(&self) -> usize {
        let assignments = self.task_assignments.read().await;
        assignments.len()
    }
    
    async fn has_agent(&self, agent_id: &AgentId) -> bool {
        let agents = self.agents.read().await;
        agents.contains_key(agent_id)
    }
    
    async fn start_all_agents(&self) -> Result<()> {
        let mut agents = self.agents.write().await;
        for agent in agents.values_mut() {
            agent.start().await?;
        }
        Ok(())
    }
    
    async fn shutdown_all_agents(&self) -> Result<()> {
        let mut agents = self.agents.write().await;
        for agent in agents.values_mut() {
            agent.shutdown().await?;
        }
        Ok(())
    }
}

// Mock implementation of SwarmOrchestrator
struct MockSwarmOrchestrator {
    agents: HashMap<AgentId, DynamicAgent>,
    max_agents: usize,
}

impl SwarmOrchestrator for MockSwarmOrchestrator {
    type Metrics = SwarmMetricsCore;
    
    fn metrics(&self) -> Self::Metrics {
        SwarmMetricsCore {
            total_agents: self.agents.len(),
            active_agents: self.agents.values()
                .filter(|a| a.status() == AgentStatus::Running)
                .count(),
            queued_tasks: 0,
            assigned_tasks: 0,
            total_connections: 0,
        }
    }
    
    fn config_summary(&self) -> SwarmConfigSummary {
        SwarmConfigSummary {
            max_agents: self.max_agents,
            distribution_strategy: "RoundRobin",
            topology_type: "Mesh",
            auto_scaling_enabled: false,
            health_check_interval_ms: 5000,
        }
    }
}

// Mock implementation of SwarmMonitoring
struct MockSwarmMonitoring {
    error_count: usize,
    tasks_completed: usize,
}

impl SwarmMonitoring for MockSwarmMonitoring {
    fn health_status(&self) -> SwarmHealthStatus {
        SwarmHealthStatus {
            health_score: if self.error_count == 0 { 1.0 } else { 0.5 },
            issues: if self.error_count > 0 { 
                vec!["Errors detected".to_string()] 
            } else { 
                vec![] 
            },
            healthy_agents: 5,
            unhealthy_agents: self.error_count,
        }
    }
    
    fn performance_metrics(&self) -> SwarmPerformanceMetrics {
        SwarmPerformanceMetrics {
            avg_task_completion_ms: 25.5,
            tasks_per_second: 10.0,
            avg_agent_utilization: 0.75,
            throughput_per_hour: 36000.0,
        }
    }
    
    fn error_statistics(&self) -> SwarmErrorStatistics {
        SwarmErrorStatistics {
            total_errors: self.error_count,
            #[allow(clippy::cast_precision_loss)]
            error_rate: self.error_count as f64 / 100.0,
            common_errors: vec![
                ("Timeout".to_string(), self.error_count / 2),
                ("AgentNotFound".to_string(), self.error_count / 2),
            ],
            trend: if self.error_count > 10 {
                ErrorTrend::Increasing
            } else {
                ErrorTrend::Stable
            },
        }
    }
}

// Tests for SwarmSync trait
#[test]
fn test_swarm_sync_trait() {
    let mut swarm = MockSwarmSync::new();
    
    // Test agent registration
    let agent = DynamicAgent::new("test-agent", vec!["compute".to_string()]);
    assert!(swarm.register_agent(agent).is_ok());
    
    // Test agent retrieval
    assert!(swarm.get_agent(&"test-agent".to_string()).is_some());
    
    // Test task submission
    let task = Task::new("task-1", "compute");
    assert!(swarm.submit_task(task).is_ok());
    assert_eq!(swarm.task_queue_size(), 1);
    
    // Test agent statuses
    let statuses = swarm.agent_statuses();
    assert_eq!(statuses.len(), 1);
    
    // Test unregister
    assert!(swarm.unregister_agent(&"test-agent".to_string()).is_ok());
    assert!(swarm.get_agent(&"test-agent".to_string()).is_none());
}

#[test]
fn test_swarm_sync_max_agents() {
    let mut swarm = MockSwarmSync::new();
    
    // Fill up to max agents (10)
    for i in 0..10 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        assert!(swarm.register_agent(agent).is_ok());
    }
    
    // Try to add one more
    let agent = DynamicAgent::new("agent-11", vec!["compute".to_string()]);
    let result = swarm.register_agent(agent);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), SwarmError::ResourceExhausted { .. }));
}

#[test]
fn test_swarm_sync_agent_mut() {
    let mut swarm = MockSwarmSync::new();
    
    let agent = DynamicAgent::new("test-agent", vec!["compute".to_string()]);
    swarm.register_agent(agent).unwrap();
    
    // Get mutable reference and modify
    if let Some(agent) = swarm.get_agent_mut(&"test-agent".to_string()) {
        agent.set_status(AgentStatus::Busy);
    }
    
    // Verify modification
    let statuses = swarm.agent_statuses();
    assert_eq!(statuses.get("test-agent"), Some(&AgentStatus::Busy));
}

// Tests for SwarmAsync trait
#[tokio::test]
async fn test_swarm_async_trait() {
    let swarm = MockSwarmAsync::new();
    
    // Test agent registration
    let agent = DynamicAgent::new("async-agent", vec!["compute".to_string()]);
    assert!(swarm.register_agent(agent).await.is_ok());
    
    // Test has_agent
    assert!(swarm.has_agent(&"async-agent".to_string()).await);
    assert!(!swarm.has_agent(&"nonexistent".to_string()).await);
    
    // Test task submission
    let task = Task::new("async-task", "compute")
        .require_capability("compute");
    assert!(swarm.submit_task(task).await.is_ok());
    assert_eq!(swarm.task_queue_size().await, 1);
    
    // Test task distribution
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].1, "async-agent");
}

#[tokio::test]
async fn test_swarm_async_lifecycle() {
    let swarm = MockSwarmAsync::new();
    
    // Register multiple agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).await.unwrap();
    }
    
    // Start all agents
    assert!(swarm.start_all_agents().await.is_ok());
    
    // Verify all are running
    let statuses = swarm.agent_statuses().await;
    for status in statuses.values() {
        assert_eq!(*status, AgentStatus::Running);
    }
    
    // Shutdown all agents
    assert!(swarm.shutdown_all_agents().await.is_ok());
}

#[tokio::test]
async fn test_swarm_async_task_distribution_no_capable_agents() {
    let swarm = MockSwarmAsync::new();
    
    // Register agent with different capability
    let agent = DynamicAgent::new("storage-agent", vec!["storage".to_string()]);
    swarm.register_agent(agent).await.unwrap();
    
    // Submit task requiring compute capability
    let task = Task::new("compute-task", "compute")
        .require_capability("compute");
    swarm.submit_task(task).await.unwrap();
    
    // Distribution should return empty (no capable agents)
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 0);
    
    // Task should still be in queue
    assert_eq!(swarm.task_queue_size().await, 1);
}

// Tests for SwarmOrchestrator trait
#[test]
fn test_swarm_orchestrator_trait() {
    let mut orchestrator = MockSwarmOrchestrator {
        agents: HashMap::new(),
        max_agents: 100,
    };
    
    // Add some agents
    for i in 0..5 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        orchestrator.agents.insert(format!("agent-{i}"), agent);
    }
    
    // Test metrics
    let metrics = orchestrator.metrics();
    assert_eq!(metrics.total_agents, 5);
    assert_eq!(metrics.active_agents, 5);
    
    // Test health check
    assert!(orchestrator.is_healthy());
    
    // Test config summary
    let config = orchestrator.config_summary();
    assert_eq!(config.max_agents, 100);
    assert_eq!(config.distribution_strategy, "RoundRobin");
}

// Tests for SwarmMonitoring trait
#[test]
#[allow(clippy::float_cmp)]
fn test_swarm_monitoring_trait() {
    let monitor = MockSwarmMonitoring {
        error_count: 0,
        tasks_completed: 100,
    };
    
    // Test health status
    let health = monitor.health_status();
    assert_eq!(health.health_score, 1.0);
    assert!(health.issues.is_empty());
    
    // Test performance metrics
    let perf = monitor.performance_metrics();
    assert_eq!(perf.avg_task_completion_ms, 25.5);
    assert_eq!(perf.tasks_per_second, 10.0);
    
    // Test error statistics
    let errors = monitor.error_statistics();
    assert_eq!(errors.total_errors, 0);
    assert_eq!(errors.trend, ErrorTrend::Stable);
}

#[test]
#[allow(clippy::float_cmp)]
fn test_swarm_monitoring_with_errors() {
    let monitor = MockSwarmMonitoring {
        error_count: 20,
        tasks_completed: 80,
    };
    
    // Test health status with errors
    let health = monitor.health_status();
    assert_eq!(health.health_score, 0.5);
    assert!(!health.issues.is_empty());
    assert_eq!(health.unhealthy_agents, 20);
    
    // Test error statistics
    let errors = monitor.error_statistics();
    assert_eq!(errors.total_errors, 20);
    assert_eq!(errors.trend, ErrorTrend::Increasing);
    assert_eq!(errors.common_errors.len(), 2);
}

// Tests for SwarmLifecycleState
#[test]
fn test_swarm_lifecycle_states() {
    let states = [
        SwarmLifecycleState::Uninitialized,
        SwarmLifecycleState::Initialized,
        SwarmLifecycleState::Running,
        SwarmLifecycleState::Paused,
        SwarmLifecycleState::Stopped,
        SwarmLifecycleState::Error,
    ];
    
    // Verify all states are distinct
    for (i, state1) in states.iter().enumerate() {
        for (j, state2) in states.iter().enumerate() {
            if i == j {
                assert_eq!(state1, state2);
            } else {
                assert_ne!(state1, state2);
            }
        }
    }
}

// Tests for ErrorTrend
#[test]
fn test_error_trend_values() {
    assert_ne!(ErrorTrend::Increasing, ErrorTrend::Stable);
    assert_ne!(ErrorTrend::Stable, ErrorTrend::Decreasing);
    assert_ne!(ErrorTrend::Increasing, ErrorTrend::Decreasing);
}

// Test SwarmMetricsCore structure
#[test]
fn test_swarm_metrics_core() {
    let metrics = SwarmMetricsCore {
        total_agents: 10,
        active_agents: 8,
        queued_tasks: 5,
        assigned_tasks: 3,
        total_connections: 20,
    };
    
    assert_eq!(metrics.total_agents, 10);
    assert_eq!(metrics.active_agents, 8);
    assert_eq!(metrics.queued_tasks, 5);
    assert_eq!(metrics.assigned_tasks, 3);
    assert_eq!(metrics.total_connections, 20);
}

// Test SwarmConfigSummary
#[test]
fn test_swarm_config_summary() {
    let config = SwarmConfigSummary {
        max_agents: 50,
        distribution_strategy: "LeastLoaded",
        topology_type: "Hierarchical",
        auto_scaling_enabled: true,
        health_check_interval_ms: 3000,
    };
    
    assert_eq!(config.max_agents, 50);
    assert!(config.auto_scaling_enabled);
    assert_eq!(config.health_check_interval_ms, 3000);
}

// Test SwarmHealthStatus
#[test]
#[allow(clippy::float_cmp)]
fn test_swarm_health_status() {
    let health = SwarmHealthStatus {
        health_score: 0.85,
        issues: vec!["High memory usage".to_string()],
        healthy_agents: 17,
        unhealthy_agents: 3,
    };
    
    assert_eq!(health.health_score, 0.85);
    assert_eq!(health.issues.len(), 1);
    assert_eq!(health.healthy_agents, 17);
    assert_eq!(health.unhealthy_agents, 3);
}

// Test SwarmPerformanceMetrics
#[test]
#[allow(clippy::float_cmp)]
fn test_swarm_performance_metrics() {
    let perf = SwarmPerformanceMetrics {
        avg_task_completion_ms: 150.0,
        tasks_per_second: 20.5,
        avg_agent_utilization: 0.82,
        throughput_per_hour: 73800.0,
    };
    
    assert_eq!(perf.avg_task_completion_ms, 150.0);
    assert_eq!(perf.tasks_per_second, 20.5);
    assert_eq!(perf.avg_agent_utilization, 0.82);
    assert_eq!(perf.throughput_per_hour, 73800.0);
}

// Test SwarmErrorStatistics
#[test]
#[allow(clippy::float_cmp)]
fn test_swarm_error_statistics() {
    let errors = SwarmErrorStatistics {
        total_errors: 42,
        error_rate: 1.2,
        common_errors: vec![
            ("NetworkTimeout".to_string(), 20),
            ("AgentCrash".to_string(), 15),
            ("TaskOverflow".to_string(), 7),
        ],
        trend: ErrorTrend::Decreasing,
    };
    
    assert_eq!(errors.total_errors, 42);
    assert_eq!(errors.error_rate, 1.2);
    assert_eq!(errors.common_errors.len(), 3);
    assert_eq!(errors.common_errors[0].1, 20);
    assert_eq!(errors.trend, ErrorTrend::Decreasing);
}