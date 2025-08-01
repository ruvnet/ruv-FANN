//! Comprehensive integration tests for swarm behavior

use crate::agent::{AgentStatus, DynamicAgent};
use crate::async_swarm::{AsyncSwarm, AsyncSwarmConfig};
use crate::error::SwarmError;
use crate::swarm::{Swarm, SwarmConfig};
use crate::task::{DistributionStrategy, Task, TaskPayload, TaskPriority};
use crate::topology::TopologyType;

#[test]
fn test_swarm_error_retriable() {
    let timeout_error = SwarmError::Timeout { duration_ms: 1000 };
    assert!(timeout_error.is_retriable());
    
    let comm_error = SwarmError::CommunicationError { 
        reason: "Network issue".to_string() 
    };
    assert!(comm_error.is_retriable());
    
    let resource_error = SwarmError::ResourceExhausted { 
        resource: "memory".to_string() 
    };
    assert!(resource_error.is_retriable());
    
    // Non-retriable errors
    let agent_error = SwarmError::AgentNotFound { 
        id: "missing".to_string() 
    };
    assert!(!agent_error.is_retriable());
}

#[test]
fn test_custom_error() {
    let error = SwarmError::custom("Test error message");
    assert!(format!("{error}").contains("Test error message"));
}

// Integration test: Complete swarm workflow
#[tokio::test]
async fn test_swarm_complete_workflow() {
    let config = SwarmConfig {
        topology_type: TopologyType::Mesh,
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Step 1: Register agents with different capabilities
    let compute_agent = DynamicAgent::new("compute-1", vec!["compute".to_string(), "ml".to_string()]);
    let storage_agent = DynamicAgent::new("storage-1", vec!["storage".to_string(), "cache".to_string()]);
    let analysis_agent = DynamicAgent::new("analysis-1", vec!["analyze".to_string(), "compute".to_string()]);
    
    swarm.register_agent(compute_agent).unwrap();
    swarm.register_agent(storage_agent).unwrap();
    swarm.register_agent(analysis_agent).unwrap();
    
    // Step 2: Submit various tasks
    let tasks = vec![
        Task::new("ml-task", "machine-learning")
            .require_capability("ml")
            .with_priority(TaskPriority::High),
        Task::new("store-data", "storage")
            .require_capability("storage")
            .with_payload(TaskPayload::Binary(vec![1, 2, 3, 4, 5])),
        Task::new("compute-stats", "computation")
            .require_capability("compute"),
        Task::new("analyze-results", "analysis")
            .require_capability("analyze"),
    ];
    
    for task in tasks {
        swarm.submit_task(task).unwrap();
    }
    
    // Step 3: Start all agents
    swarm.start_all_agents().await.unwrap();
    
    // Step 4: Distribute tasks
    let assignments = swarm.distribute_tasks().await.unwrap();
    
    // Verify assignments
    assert_eq!(assignments.len(), 4);
    
    // Verify each task was assigned to a capable agent
    let mut ml_assigned = false;
    let mut storage_assigned = false;
    let mut compute_assigned = false;
    let mut analyze_assigned = false;
    
    for (task_id, agent_id) in &assignments {
        match task_id.to_string().as_str() {
            "ml-task" => {
                assert_eq!(agent_id, "compute-1");
                ml_assigned = true;
            }
            "store-data" => {
                assert_eq!(agent_id, "storage-1");
                storage_assigned = true;
            }
            "compute-stats" => {
                assert!(agent_id == "compute-1" || agent_id == "analysis-1");
                compute_assigned = true;
            }
            "analyze-results" => {
                assert_eq!(agent_id, "analysis-1");
                analyze_assigned = true;
            }
            _ => panic!("Unexpected task ID"),
        }
    }
    
    assert!(ml_assigned && storage_assigned && compute_assigned && analyze_assigned);
    
    // Step 5: Verify metrics
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 3);
    assert_eq!(metrics.active_agents, 3);
    assert_eq!(metrics.queued_tasks, 0);
    assert_eq!(metrics.assigned_tasks, 4);
    assert_eq!(metrics.total_connections, 3); // Mesh topology
}

// Async integration test
#[tokio::test]
async fn test_async_swarm_workflow() {
    let config = AsyncSwarmConfig {
        topology_type: TopologyType::Star,
        distribution_strategy: DistributionStrategy::RoundRobin,
        task_timeout_ms: 100,
        max_concurrent_tasks_per_agent: 2,
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Register coordinator agent (first in star topology)
    let coordinator = DynamicAgent::new("coordinator", vec!["coordinate".to_string(), "compute".to_string()]);
    swarm.register_agent(coordinator).await.unwrap();
    
    // Register worker agents
    for i in 1..=3 {
        let worker = DynamicAgent::new(format!("worker-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(worker).await.unwrap();
    }
    
    // Submit tasks
    for i in 0..6 {
        let task = Task::new(format!("task-{i}"), "computation")
            .require_capability("compute")
            .with_payload(TaskPayload::Json(serde_json::json!({ "index": i }).to_string()));
        swarm.submit_task(task).await.unwrap();
    }
    
    // Process tasks concurrently
    let results = swarm.process_tasks_concurrently(3).await.unwrap();
    assert_eq!(results.len(), 6);
    
    // Verify all tasks completed
    for (task_id, result) in results {
        assert!(result.is_ok(), "Task {task_id} failed");
    }
    
    // Check metrics
    let metrics = swarm.metrics().await;
    assert_eq!(metrics.total_agents, 4);
    assert_eq!(metrics.total_connections, 3); // Star: 3 workers connected to coordinator
}

// Test multi-phase workflow
#[tokio::test]
async fn test_multi_phase_swarm_workflow() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Phase 1: Setup agents
    let agents = vec![
        ("preprocessor", vec!["preprocess", "validate"]),
        ("processor", vec!["process", "compute"]),
        ("postprocessor", vec!["postprocess", "finalize"]),
    ];
    
    for (id, capabilities) in agents {
        let agent = DynamicAgent::new(
            id, 
            capabilities.into_iter().map(String::from).collect()
        );
        swarm.register_agent(agent).unwrap();
    }
    
    swarm.start_all_agents().await.unwrap();
    
    // Phase 2: Preprocessing tasks
    for i in 0..3 {
        let task = Task::new(format!("preprocess-{i}"), "preprocessing")
            .require_capability("preprocess");
        swarm.submit_task(task).unwrap();
    }
    
    let phase1_assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(phase1_assignments.len(), 3);
    for (_, agent_id) in &phase1_assignments {
        assert_eq!(agent_id, "preprocessor");
    }
    
    // Phase 3: Processing tasks
    for i in 0..3 {
        let task = Task::new(format!("process-{i}"), "processing")
            .require_capability("process");
        swarm.submit_task(task).unwrap();
    }
    
    let phase2_assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(phase2_assignments.len(), 3);
    for (_, agent_id) in &phase2_assignments {
        assert_eq!(agent_id, "processor");
    }
    
    // Phase 4: Postprocessing tasks
    for i in 0..3 {
        let task = Task::new(format!("postprocess-{i}"), "postprocessing")
            .require_capability("postprocess");
        swarm.submit_task(task).unwrap();
    }
    
    let phase3_assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(phase3_assignments.len(), 3);
    for (_, agent_id) in &phase3_assignments {
        assert_eq!(agent_id, "postprocessor");
    }
    
    // Verify final state
    let metrics = swarm.metrics();
    assert_eq!(metrics.assigned_tasks, 9);
    assert_eq!(metrics.queued_tasks, 0);
}

// Test error recovery workflow
#[tokio::test]
async fn test_swarm_error_recovery() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register agents
    let healthy_agent = DynamicAgent::new("healthy", vec!["compute".to_string()]);
    let mut error_agent = DynamicAgent::new("error", vec!["compute".to_string()]);
    error_agent.set_status(AgentStatus::Error);
    
    swarm.register_agent(healthy_agent).unwrap();
    swarm.register_agent(error_agent).unwrap();
    
    // Submit tasks
    for i in 0..3 {
        let task = Task::new(format!("task-{i}"), "compute")
            .require_capability("compute");
        swarm.submit_task(task).unwrap();
    }
    
    // Distribute - should only assign to healthy agent
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 3);
    for (_, agent_id) in &assignments {
        assert_eq!(agent_id, "healthy");
    }
    
    // Fix error agent
    if let Some(agent) = swarm.get_agent_mut(&"error".to_string()) {
        agent.set_status(AgentStatus::Running);
    }
    
    // Submit more tasks
    for i in 3..6 {
        let task = Task::new(format!("task-{i}"), "compute")
            .require_capability("compute");
        swarm.submit_task(task).unwrap();
    }
    
    // Now both agents should get tasks
    let assignments2 = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments2.len(), 3);
    
    let mut healthy_count = 0;
    let mut error_count = 0;
    for (_, agent_id) in &assignments2 {
        match agent_id.as_str() {
            "healthy" => healthy_count += 1,
            "error" => error_count += 1,
            _ => panic!("Unexpected agent"),
        }
    }
    
    // With LeastLoaded strategy, the previously-error agent (now fixed) should get all new tasks
    // because it has 0 tasks while healthy agent already has 3 assigned
    assert_eq!(error_count, 3);
    assert_eq!(healthy_count, 0);
}

// Test dynamic topology changes
#[test]
fn test_dynamic_topology_changes() {
    let config = SwarmConfig {
        topology_type: TopologyType::Mesh,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Start with 2 agents
    swarm.register_agent(DynamicAgent::new("agent-1", vec!["compute".to_string()])).unwrap();
    swarm.register_agent(DynamicAgent::new("agent-2", vec!["compute".to_string()])).unwrap();
    
    let metrics1 = swarm.metrics();
    assert_eq!(metrics1.total_connections, 1); // 2 agents, 1 connection
    
    // Add more agents
    swarm.register_agent(DynamicAgent::new("agent-3", vec!["compute".to_string()])).unwrap();
    swarm.register_agent(DynamicAgent::new("agent-4", vec!["compute".to_string()])).unwrap();
    
    let metrics2 = swarm.metrics();
    assert_eq!(metrics2.total_connections, 6); // 4 agents, 6 connections (fully connected)
    
    // Remove an agent
    swarm.unregister_agent(&"agent-2".to_string()).unwrap();
    
    let metrics3 = swarm.metrics();
    assert_eq!(metrics3.total_connections, 3); // 3 agents, 3 connections
}

// Test load balancing fairness
#[tokio::test]
async fn test_load_balancing_fairness() {
    let config = SwarmConfig {
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register 3 identical agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).unwrap();
    }
    
    // Submit many tasks to test distribution
    for i in 0..30 {
        let task = Task::new(format!("task-{i}"), "compute")
            .require_capability("compute");
        swarm.submit_task(task).unwrap();
    }
    
    // Distribute all tasks
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 30);
    
    // Count tasks per agent
    let mut agent_loads = std::collections::HashMap::new();
    for (_, agent_id) in assignments {
        *agent_loads.entry(agent_id).or_insert(0) += 1;
    }
    
    // Each agent should have approximately 10 tasks (±2 for rounding)
    for (_agent, load) in agent_loads {
        assert!((8..=12).contains(&load), "Unbalanced load: {load}");
    }
}