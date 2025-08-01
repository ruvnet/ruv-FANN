//! Basic Swarm Example
//!
//! This example demonstrates how to create a simple swarm, register agents,
//! and submit tasks for processing.

use ruv_swarm_core::{
    agent::DynamicAgent,
    error::Result,
    swarm::{Swarm, SwarmConfig},
    task::{Task, TaskPriority},
    topology::TopologyType,
};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Basic Swarm Example ===\n");

    // Create swarm configuration
    let config = SwarmConfig {
        topology_type: TopologyType::Mesh,
        max_agents: 10,
        enable_auto_scaling: false,
        health_check_interval_ms: 5000,
        ..Default::default()
    };

    // Initialize swarm
    println!("Initializing swarm...");
    let mut swarm = Swarm::new(config);

    // Register worker agents
    println!("Registering agents...");
    let agent1 = DynamicAgent::new(
        "agent-1",
        vec!["compute".to_string(), "data-processing".to_string()],
    );
    swarm.register_agent(agent1)?;
    println!("  Agent 1 registered");

    let agent2 = DynamicAgent::new(
        "agent-2",
        vec!["compute".to_string(), "analysis".to_string()],
    );
    swarm.register_agent(agent2)?;
    println!("  Agent 2 registered");

    let agent3 = DynamicAgent::new(
        "agent-3",
        vec!["compute".to_string(), "aggregation".to_string()],
    );
    swarm.register_agent(agent3)?;
    println!("  Agent 3 registered");

    // Wait for agents to be ready
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create and submit tasks
    println!("\nSubmitting tasks...");
    let task1 = Task::new("task-1", "data-processing")
        .with_priority(TaskPriority::High)
        .require_capability("compute");

    let task2 = Task::new("task-2", "analysis")
        .with_priority(TaskPriority::Normal)
        .require_capability("analysis");

    let task3 = Task::new("task-3", "aggregation")
        .with_priority(TaskPriority::Normal)
        .require_capability("aggregation");

    swarm.submit_task(task1)?;
    swarm.submit_task(task2)?;
    swarm.submit_task(task3)?;

    // Distribute tasks to agents
    let assignments = swarm.distribute_tasks().await?;
    println!("Task assignments: {:?}", assignments);

    // Get swarm metrics
    let metrics = swarm.metrics();
    println!("\nSwarm Metrics:");
    println!("  Total agents: {}", metrics.total_agents);
    println!("  Active agents: {}", metrics.active_agents);
    println!("  Queued tasks: {}", metrics.queued_tasks);
    println!("  Assigned tasks: {}", metrics.assigned_tasks);
    println!("  Total connections: {}", metrics.total_connections);

    // Get agent statuses
    let statuses = swarm.agent_statuses();
    println!("\nAgent Statuses:");
    for (id, status) in statuses {
        println!("  {}: {:?}", id, status);
    }

    // Start all agents
    println!("\nStarting all agents...");
    swarm.start_all_agents().await?;

    // Graceful shutdown
    println!("\nShutting down swarm...");
    swarm.shutdown_all_agents().await?;

    println!("Example completed!");
    Ok(())
}
