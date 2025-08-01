//! Integration test suite for ruv-swarm
//! 
//! This module organizes all integration tests for the swarm framework,
//! including end-to-end scenarios, cognitive diversity patterns,
//! persistence, performance benchmarks, and chaos testing.

#![cfg(test)]

// Test modules
mod swarm_integration;
mod cognitive_diversity;
mod persistence_integration;
mod performance_benchmarks;
mod chaos_testing;
mod failure_scenarios;
mod integration_test;

// Re-export common test utilities
pub mod test_utils {
    use ruv_swarm_core::{
        agent::{Agent, AgentId, AgentType},
        swarm::{Swarm, SwarmConfig, Topology},
        task::{Task, TaskResult},
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    /// Standard test swarm configuration
    pub fn test_config() -> SwarmConfig {
        SwarmConfig {
            topology: Topology::Mesh,
            max_agents: 20,
            heartbeat_interval: std::time::Duration::from_secs(1),
            task_timeout: std::time::Duration::from_secs(30),
            persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
        }
    }
    
    /// Create a test swarm with predefined agents
    pub async fn create_test_swarm_with_agents(
        agent_types: Vec<AgentType>,
    ) -> Result<Swarm, Box<dyn std::error::Error>> {
        let mut swarm = Swarm::new(test_config()).await?;
        
        for agent_type in agent_types {
            swarm.spawn(agent_type).await?;
        }
        
        Ok(swarm)
    }
    
    /// Generate test neural network data
    pub fn generate_nn_test_data() -> Vec<(Vec<f32>, Vec<f32>)> {
        // XOR problem
        vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ]
    }
    
    /// Assert task completed successfully
    pub fn assert_task_success(result: &TaskResult) {
        match result {
            TaskResult::Success { .. } => {},
            TaskResult::Failed { error, .. } => {
                panic!("Task failed with error: {}", error);
            }
            _ => panic!("Unexpected task result: {:?}", result),
        }
    }
    
    /// Wait for swarm to reach stable state
    pub async fn wait_for_stability(swarm: &Swarm, timeout: std::time::Duration) {
        let start = std::time::Instant::now();
        
        while start.elapsed() < timeout {
            if swarm.is_stable().await {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        
        panic!("Swarm did not reach stable state within timeout");
    }
}