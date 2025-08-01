//! Comprehensive swarm tests

use crate::swarm::*;
use crate::topology::*;
use crate::agent::{AgentStatus, DynamicAgent};
use crate::task::{DistributionStrategy, Task, TaskPayload};
use crate::error::SwarmError;

#[test]
fn test_swarm_config_creation() {
    let config = SwarmConfig {
        max_agents: 10,
        topology_type: TopologyType::Mesh,
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    };
    
    assert_eq!(config.max_agents, 10);
    assert_eq!(config.topology_type, TopologyType::Mesh);
    assert_eq!(config.distribution_strategy, DistributionStrategy::LeastLoaded);
}

#[test]
fn test_swarm_creation() {
    let config = SwarmConfig {
        max_agents: 5,
        topology_type: TopologyType::Star,
        ..Default::default()
    };
    
    let swarm = Swarm::new(config);
    
    // Verify swarm was created successfully
    assert_eq!(swarm.metrics().total_agents, 0);
    assert_eq!(swarm.metrics().queued_tasks, 0);
    assert_eq!(swarm.metrics().assigned_tasks, 0);
}

#[test]
fn test_swarm_metrics() {
    let swarm = Swarm::new(SwarmConfig::default());
    let metrics = swarm.metrics();
    
    // Basic metrics validation
    assert_eq!(metrics.total_agents, 0);
    assert_eq!(metrics.queued_tasks, 0);
    assert_eq!(metrics.assigned_tasks, 0);
    assert_eq!(metrics.active_agents, 0);
    assert_eq!(metrics.total_connections, 0);
}

#[test]
fn test_swarm_config_defaults() {
    let config = SwarmConfig::default();
    assert_eq!(config.topology_type, TopologyType::default());
    assert!(!config.enable_auto_scaling);
    assert_eq!(config.max_agents, 100);
    assert_eq!(config.health_check_interval_ms, 5000);
    assert_eq!(config.distribution_strategy, DistributionStrategy::LeastLoaded);
}

#[test]
fn test_agent_registration() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register an agent
    let agent = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    assert!(swarm.register_agent(agent).is_ok());
    
    // Verify agent was registered
    assert_eq!(swarm.metrics().total_agents, 1);
    assert!(swarm.get_agent(&"agent-1".to_string()).is_some());
}

#[test]
fn test_agent_registration_max_limit() {
    let config = SwarmConfig {
        max_agents: 2,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register agents up to limit
    let agent1 = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    let agent2 = DynamicAgent::new("agent-2", vec!["compute".to_string()]);
    assert!(swarm.register_agent(agent1).is_ok());
    assert!(swarm.register_agent(agent2).is_ok());
    
    // Try to exceed limit
    let agent3 = DynamicAgent::new("agent-3", vec!["compute".to_string()]);
    let result = swarm.register_agent(agent3);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), SwarmError::ResourceExhausted { .. }));
}

#[test]
fn test_agent_unregistration() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register and then unregister an agent
    let agent = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    swarm.register_agent(agent).unwrap();
    
    assert!(swarm.unregister_agent(&"agent-1".to_string()).is_ok());
    assert_eq!(swarm.metrics().total_agents, 0);
    assert!(swarm.get_agent(&"agent-1".to_string()).is_none());
}

#[test]
fn test_agent_unregistration_nonexistent() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Try to unregister non-existent agent
    let result = swarm.unregister_agent(&"nonexistent".to_string());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), SwarmError::AgentNotFound { .. }));
}

#[test]
fn test_task_submission() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Submit tasks
    let task1 = Task::new("task-1", "compute")
        .with_payload(TaskPayload::Binary(vec![1, 2, 3]));
    let task2 = Task::new("task-2", "analyze")
        .with_payload(TaskPayload::Json(serde_json::json!({ "type": "data" }).to_string()));
    
    assert!(swarm.submit_task(task1).is_ok());
    assert!(swarm.submit_task(task2).is_ok());
    
    assert_eq!(swarm.task_queue_size(), 2);
}

#[tokio::test]
async fn test_task_distribution_no_agents() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Submit task with no agents
    let task = Task::new("task-1", "compute");
    swarm.submit_task(task).unwrap();
    
    // Distribution should return empty
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 0);
    
    // Task should remain in queue
    assert_eq!(swarm.task_queue_size(), 1);
}

#[tokio::test]
async fn test_task_distribution_basic() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register agent
    let agent = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    swarm.register_agent(agent).unwrap();
    
    // Submit matching task
    let task = Task::new("task-1", "compute")
        .require_capability("compute");
    swarm.submit_task(task).unwrap();
    
    // Distribute tasks
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].0.to_string(), "task-1");
    assert_eq!(assignments[0].1, "agent-1");
    
    // Task should be removed from queue
    assert_eq!(swarm.task_queue_size(), 0);
    assert_eq!(swarm.assigned_tasks_count(), 1);
}

#[tokio::test]
async fn test_task_distribution_capability_mismatch() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register agent with different capability
    let agent = DynamicAgent::new("agent-1", vec!["storage".to_string()]);
    swarm.register_agent(agent).unwrap();
    
    // Submit task requiring different capability
    let task = Task::new("task-1", "compute")
        .require_capability("compute");
    swarm.submit_task(task).unwrap();
    
    // Distribution should fail to assign
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 0);
    
    // Task should remain in queue
    assert_eq!(swarm.task_queue_size(), 1);
}

#[tokio::test]
async fn test_distribution_strategy_least_loaded() {
    let config = SwarmConfig {
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register multiple agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).unwrap();
    }
    
    // Submit multiple tasks
    for i in 0..6 {
        let task = Task::new(format!("task-{i}"), "compute")
            .require_capability("compute");
        swarm.submit_task(task).unwrap();
    }
    
    // Distribute tasks
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 6);
    
    // Each agent should get 2 tasks (balanced load)
    let mut agent_task_count = std::collections::HashMap::new();
    for (_, agent_id) in assignments {
        *agent_task_count.entry(agent_id).or_insert(0) += 1;
    }
    
    // Verify relatively balanced distribution
    for count in agent_task_count.values() {
        assert!(*count >= 1 && *count <= 3);
    }
}

#[test]
fn test_mesh_topology_registration() {
    let config = SwarmConfig {
        topology_type: TopologyType::Mesh,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register 3 agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).unwrap();
    }
    
    // In mesh topology, all agents should be connected
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 3);
    // Each pair connected: (0,1), (0,2), (1,2) = 3 connections
    assert_eq!(metrics.total_connections, 3);
}

#[test]
fn test_star_topology_registration() {
    let config = SwarmConfig {
        topology_type: TopologyType::Star,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register agents - first one becomes coordinator
    for i in 0..4 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).unwrap();
    }
    
    // In star topology, only connections to first agent
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 4);
    // Star connections: 3 agents connected to coordinator = 3 connections
    assert_eq!(metrics.total_connections, 3);
}

#[test]
fn test_star_topology_coordinator_removal() {
    let config = SwarmConfig {
        topology_type: TopologyType::Star,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register 4 agents
    for i in 0..4 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).unwrap();
    }
    
    // Verify initial star topology
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 4);
    assert_eq!(metrics.total_connections, 3);
    
    // Remove the coordinator (agent-0)
    swarm.unregister_agent(&"agent-0".to_string()).unwrap();
    
    // Verify topology is maintained with new coordinator
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 3);
    // Should still have 2 connections (agent-2 -> agent-1, agent-3 -> agent-1)
    assert_eq!(metrics.total_connections, 2);
}

#[test]
fn test_agent_status_tracking() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register agents with different statuses
    let mut agent1 = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    agent1.set_status(AgentStatus::Running);
    
    let mut agent2 = DynamicAgent::new("agent-2", vec!["compute".to_string()]);
    agent2.set_status(AgentStatus::Busy);
    
    let mut agent3 = DynamicAgent::new("agent-3", vec!["compute".to_string()]);
    agent3.set_status(AgentStatus::Offline);
    
    swarm.register_agent(agent1).unwrap();
    swarm.register_agent(agent2).unwrap();
    swarm.register_agent(agent3).unwrap();
    
    let statuses = swarm.agent_statuses();
    assert_eq!(statuses.len(), 3);
    assert_eq!(statuses.get("agent-1"), Some(&AgentStatus::Running));
    assert_eq!(statuses.get("agent-2"), Some(&AgentStatus::Busy));
    assert_eq!(statuses.get("agent-3"), Some(&AgentStatus::Offline));
    
    // Only Running agents should be active
    let metrics = swarm.metrics();
    assert_eq!(metrics.active_agents, 1);
}

#[tokio::test]
async fn test_agent_lifecycle_management() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register multiple agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).unwrap();
    }
    
    // Start all agents
    assert!(swarm.start_all_agents().await.is_ok());
    
    // Verify all are running
    let statuses = swarm.agent_statuses();
    for status in statuses.values() {
        assert_eq!(*status, AgentStatus::Running);
    }
    
    // Shutdown all agents
    assert!(swarm.shutdown_all_agents().await.is_ok());
    
    // Verify all are offline
    let statuses = swarm.agent_statuses();
    for status in statuses.values() {
        assert_eq!(*status, AgentStatus::Offline);
    }
}

#[test]
fn test_get_agent_mut() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    let agent = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    swarm.register_agent(agent).unwrap();
    
    // Modify agent through mutable reference
    if let Some(agent) = swarm.get_agent_mut(&"agent-1".to_string()) {
        agent.set_status(AgentStatus::Busy);
    }
    
    // Verify modification
    let statuses = swarm.agent_statuses();
    assert_eq!(statuses.get("agent-1"), Some(&AgentStatus::Busy));
}

#[tokio::test]
async fn test_task_distribution_with_busy_agents() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register agents with different statuses
    let mut busy_agent = DynamicAgent::new("busy-agent", vec!["compute".to_string()]);
    busy_agent.set_status(AgentStatus::Busy);
    
    let running_agent = DynamicAgent::new("running-agent", vec!["compute".to_string()]);
    
    swarm.register_agent(busy_agent).unwrap();
    swarm.register_agent(running_agent).unwrap();
    
    // Submit task
    let task = Task::new("task-1", "compute")
        .require_capability("compute");
    swarm.submit_task(task).unwrap();
    
    // Only running agent should get the task
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].1, "running-agent");
}

#[tokio::test]
async fn test_agent_load_tracking() {
    let config = SwarmConfig {
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register agents
    let agent1 = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    let agent2 = DynamicAgent::new("agent-2", vec!["compute".to_string()]);
    swarm.register_agent(agent1).unwrap();
    swarm.register_agent(agent2).unwrap();
    
    // Submit multiple tasks
    for i in 0..3 {
        let task = Task::new(format!("task-{i}"), "compute")
            .require_capability("compute");
        swarm.submit_task(task).unwrap();
    }
    
    // Distribute tasks - should balance load
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 3);
    
    // Both agents should have tasks
    let mut agent1_tasks = 0;
    let mut agent2_tasks = 0;
    for (_, agent_id) in &assignments {
        if agent_id == "agent-1" {
            agent1_tasks += 1;
        } else if agent_id == "agent-2" {
            agent2_tasks += 1;
        }
    }
    
    assert!(agent1_tasks > 0);
    assert!(agent2_tasks > 0);
}