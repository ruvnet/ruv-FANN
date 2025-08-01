//! Simple async swarm tests that match the actual async API

#[cfg(feature = "std")]
use crate::async_swarm::*;
#[cfg(feature = "std")]
use crate::agent::*;
#[cfg(feature = "std")]
use crate::task::*;
#[cfg(feature = "std")]
use crate::topology::*;

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_config_creation() {
    let config = AsyncSwarmConfig {
        max_agents: 10,
        topology_type: TopologyType::Mesh,
        max_concurrent_tasks_per_agent: 5,
        ..Default::default()
    };
    
    assert_eq!(config.max_agents, 10);
    assert_eq!(config.topology_type, TopologyType::Mesh);
    assert_eq!(config.max_concurrent_tasks_per_agent, 5);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_creation() {
    let config = AsyncSwarmConfig {
        max_agents: 5,
        topology_type: TopologyType::Star,
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Verify swarm was created successfully
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 0);
    assert_eq!(metrics.queued_tasks, 0);
    assert_eq!(metrics.assigned_tasks, 0);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_metrics() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    let metrics = swarm.metrics().await;
    
    // Basic metrics validation
    assert_eq!(metrics.total_agents, 0);
    assert_eq!(metrics.queued_tasks, 0);
    assert_eq!(metrics.assigned_tasks, 0);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(metrics.avg_agent_load, 0.0);
    }
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_config_defaults() {
    let config = AsyncSwarmConfig::default();
    assert_eq!(config.topology_type, TopologyType::default());
    assert!(!config.enable_auto_scaling);
    assert_eq!(config.max_concurrent_tasks_per_agent, 10);
    assert_eq!(config.task_timeout_ms, 30000);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_task_submission() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    // Create a simple task
    let task = Task::new("test-task-1", "compute")
        .with_payload(TaskPayload::Binary(vec![1, 2, 3, 4]));
    
    // Submit task to swarm
    let result = swarm.submit_task(task).await;
    assert!(result.is_ok());
    
    // Verify task was queued
    let queue_size = swarm.task_queue_size().await;
    assert_eq!(queue_size, 1);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_agent_registration() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    // Create a dynamic agent
    let agent = DynamicAgent::new("test-agent".to_string(), vec!["compute".to_string()]);
    let agent_id = agent.id().to_string();
    
    // Register the agent
    let result = swarm.register_agent(agent).await;
    assert!(result.is_ok());
    
    // Verify agent was registered
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 1);
    
    // Verify agent exists
    let has_agent = swarm.has_agent(&agent_id).await;
    assert!(has_agent);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_agent_unregistration() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    // Create and register a dynamic agent
    let agent = DynamicAgent::new("test-agent".to_string(), vec!["compute".to_string()]);
    let agent_id = agent.id().to_string();
    
    swarm.register_agent(agent).await.unwrap();
    
    // Verify agent is registered
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 1);
    
    // Unregister the agent
    let result = swarm.unregister_agent(&agent_id).await;
    assert!(result.is_ok());
    
    // Verify agent was removed
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 0);
    
    // Verify agent no longer exists
    let has_agent = swarm.has_agent(&agent_id).await;
    assert!(!has_agent);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_task_distribution() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    // Create and register a dynamic agent
    let agent = DynamicAgent::new("test-agent".to_string(), vec!["compute".to_string()]);
    swarm.register_agent(agent).await.unwrap();
    
    // Create and submit a task
    let task = Task::new("test-task-1", "compute")
        .with_payload(TaskPayload::Binary(vec![1, 2, 3, 4]));
    
    swarm.submit_task(task).await.unwrap();
    
    // Distribute tasks
    let assignments = swarm.distribute_tasks().await.unwrap();
    
    // Verify task was assigned
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].0.to_string(), "test-task-1");
    assert_eq!(assignments[0].1, "test-agent");
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_agent_status() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    // Create and register a dynamic agent
    let agent = DynamicAgent::new("test-agent".to_string(), vec!["compute".to_string()]);
    swarm.register_agent(agent).await.unwrap();
    
    // Get agent statuses
    let statuses = swarm.agent_statuses().await;
    
    // Verify agent status
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses.get("test-agent"), Some(&AgentStatus::Running));
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_health_monitoring() {
    let config = AsyncSwarmConfig {
        health_check_interval_ms: 50, // Fast interval for testing
        ..Default::default()
    };
    
    let mut swarm = AsyncSwarm::new(config);
    
    // Register a healthy agent
    let agent = DynamicAgent::new("healthy-agent".to_string(), vec!["compute".to_string()]);
    swarm.register_agent(agent).await.unwrap();
    
    // Start health monitoring
    let result = swarm.start_health_monitoring();
    assert!(result.is_ok());
    
    // Wait a bit for health checks
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Stop health monitoring
    swarm.stop_health_monitoring();
    
    // Verify no errors occurred
    let statuses = swarm.agent_statuses().await;
    assert_eq!(statuses.get("healthy-agent"), Some(&AgentStatus::Running));
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_concurrent_task_processing() {
    let config = AsyncSwarmConfig {
        task_timeout_ms: 100,
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Register multiple agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).await.unwrap();
    }
    
    // Submit multiple tasks
    for i in 0u8..5u8 {
        let task = Task::new(format!("task-{i}"), "compute")
            .with_payload(TaskPayload::Binary(vec![i]));
        swarm.submit_task(task).await.unwrap();
    }
    
    // Process tasks concurrently
    let results = swarm.process_tasks_concurrently(2).await.unwrap();
    
    // Verify results
    assert_eq!(results.len(), 5);
    for (task_id, result) in results {
        assert!(result.is_ok(), "Task {task_id} failed");
    }
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_error_scenarios() {
    let config = AsyncSwarmConfig {
        max_agents: 2,
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Test: Register agents up to limit
    let agent1 = DynamicAgent::new("agent-1".to_string(), vec!["compute".to_string()]);
    let agent2 = DynamicAgent::new("agent-2".to_string(), vec!["compute".to_string()]);
    let agent3 = DynamicAgent::new("agent-3".to_string(), vec!["compute".to_string()]);
    
    swarm.register_agent(agent1).await.unwrap();
    swarm.register_agent(agent2).await.unwrap();
    
    // This should fail - exceeds max agents
    let result = swarm.register_agent(agent3).await;
    assert!(result.is_err());
    
    // Test: Unregister non-existent agent
    let result = swarm.unregister_agent(&"non-existent".to_string()).await;
    assert!(result.is_err());
    
    // Test: Task with no capable agents
    let task = Task::new("special-task", "special")
        .require_capability("special-capability")
        .with_payload(TaskPayload::Binary(vec![1, 2, 3]));
    
    swarm.submit_task(task).await.unwrap();
    
    // Distribute should not assign task to any agent
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 0);
    
    // Task should be back in queue
    assert_eq!(swarm.task_queue_size().await, 1);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_distribution_strategies() {
    // Test different distribution strategies
    
    // Test 1: LeastLoaded strategy
    let config = AsyncSwarmConfig {
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Register agents
    let agent1 = DynamicAgent::new("agent-1".to_string(), vec!["compute".to_string()]);
    let agent2 = DynamicAgent::new("agent-2".to_string(), vec!["compute".to_string()]);
    
    swarm.register_agent(agent1).await.unwrap();
    swarm.register_agent(agent2).await.unwrap();
    
    // Submit multiple tasks
    for i in 0u8..4u8 {
        let task = Task::new(format!("task-{i}"), "compute")
            .with_payload(TaskPayload::Binary(vec![i]));
        swarm.submit_task(task).await.unwrap();
    }
    
    // Distribute tasks - should balance load
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 4);
    
    // Test 2: Random strategy
    let config = AsyncSwarmConfig {
        distribution_strategy: DistributionStrategy::Random,
        ..Default::default()
    };
    
    let random_swarm = AsyncSwarm::new(config);
    
    // Register agents
    let agent1 = DynamicAgent::new("agent-1".to_string(), vec!["compute".to_string()]);
    let agent2 = DynamicAgent::new("agent-2".to_string(), vec!["compute".to_string()]);
    
    random_swarm.register_agent(agent1).await.unwrap();
    random_swarm.register_agent(agent2).await.unwrap();
    
    // Submit a task
    let task = Task::new("random-task", "compute")
        .with_payload(TaskPayload::Binary(vec![1, 2, 3]));
    random_swarm.submit_task(task).await.unwrap();
    
    // Distribute - should assign to some agent
    let assignments = random_swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 1);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_topology_connections() {
    // Test Mesh topology
    let config = AsyncSwarmConfig {
        topology_type: TopologyType::Mesh,
        ..Default::default()
    };
    
    let mesh_swarm = AsyncSwarm::new(config);
    
    // Register 3 agents in mesh
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("mesh-agent-{i}"), vec!["compute".to_string()]);
        mesh_swarm.register_agent(agent).await.unwrap();
    }
    
    // In mesh, all agents should be connected
    let metrics = mesh_swarm.metrics().await;
    assert_eq!(metrics.total_agents, 3);
    // Each agent connects to all others: 3 agents * 2 connections each / 2 (bidirectional) = 3
    assert!(metrics.total_connections > 0);
    
    // Test Star topology
    let config = AsyncSwarmConfig {
        topology_type: TopologyType::Star,
        ..Default::default()
    };
    
    let star_swarm = AsyncSwarm::new(config);
    
    // Register 3 agents in star
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("star-agent-{i}"), vec!["compute".to_string()]);
        star_swarm.register_agent(agent).await.unwrap();
    }
    
    // In star, agents connect to first agent (coordinator)
    let star_metrics = star_swarm.metrics().await;
    assert_eq!(star_metrics.total_agents, 3);
    assert!(star_metrics.total_connections > 0);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_lifecycle_operations() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    // Register multiple agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).await.unwrap();
    }
    
    // Start all agents
    let result = swarm.start_all_agents().await;
    assert!(result.is_ok());
    
    // Verify all agents are running
    let statuses = swarm.agent_statuses().await;
    assert_eq!(statuses.len(), 3);
    for (_id, status) in statuses {
        assert_eq!(status, AgentStatus::Running);
    }
    
    // Shutdown all agents
    let result = swarm.shutdown_all_agents().await;
    assert!(result.is_ok());
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_metrics_accuracy() {
    let swarm = AsyncSwarm::new(AsyncSwarmConfig::default());
    
    // Initial metrics
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 0);
    assert_eq!(metrics.active_agents, 0);
    assert_eq!(metrics.queued_tasks, 0);
    assert_eq!(metrics.assigned_tasks, 0);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(metrics.avg_agent_load, 0.0);
    }
    
    // Add agents and tasks
    let agent1 = DynamicAgent::new("agent-1".to_string(), vec!["compute".to_string()]);
    let agent2 = DynamicAgent::new("agent-2".to_string(), vec!["compute".to_string()]);
    
    swarm.register_agent(agent1).await.unwrap();
    swarm.register_agent(agent2).await.unwrap();
    
    // Submit tasks
    for i in 0u8..3u8 {
        let task = Task::new(format!("task-{i}"), "compute")
            .with_payload(TaskPayload::Binary(vec![i]));
        swarm.submit_task(task).await.unwrap();
    }
    
    // Check metrics before distribution
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 2);
    assert_eq!(metrics.active_agents, 2);
    assert_eq!(metrics.queued_tasks, 3);
    assert_eq!(metrics.assigned_tasks, 0);
    
    // Distribute tasks
    swarm.distribute_tasks().await.unwrap();
    
    // Check metrics after distribution
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 2);
    assert_eq!(metrics.active_agents, 2);
    assert_eq!(metrics.queued_tasks, 0);
    assert_eq!(metrics.assigned_tasks, 3);
    assert!(metrics.avg_agent_load > 0.0);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_auto_scaling_config() {
    let config = AsyncSwarmConfig {
        enable_auto_scaling: true,
        max_agents: 10,
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Register initial agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).await.unwrap();
    }
    
    // Verify auto-scaling is configured (actual scaling would require more implementation)
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 3);
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_task_timeout() {
    let config = AsyncSwarmConfig {
        task_timeout_ms: 50, // Very short timeout
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Register an agent
    let agent = DynamicAgent::new("agent-1".to_string(), vec!["compute".to_string()]);
    swarm.register_agent(agent).await.unwrap();
    
    // Submit a task
    let task = Task::new("timeout-task", "compute")
        .with_payload(TaskPayload::Binary(vec![1, 2, 3]));
    swarm.submit_task(task).await.unwrap();
    
    // Process with timeout
    let results = swarm.process_tasks_concurrently(1).await.unwrap();
    
    // Should complete (in our mock, tasks complete quickly)
    assert_eq!(results.len(), 1);
}