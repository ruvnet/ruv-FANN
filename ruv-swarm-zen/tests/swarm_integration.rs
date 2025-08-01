//! End-to-end integration tests for swarm behavior
//! 
//! These tests validate the complete swarm orchestration lifecycle,
//! from agent spawning to distributed task execution and result aggregation.

use ruv_swarm_core::{
    agent::{Agent, AgentId, AgentType, Capability},
    error::SwarmError,
    swarm::{Swarm, SwarmConfig, Topology},
    task::{Task, TaskResult, Strategy},
    message::{Message, MessageType},
};
use ruv_swarm_agents::{NeuralProcessor, DataAnalyzer, Coordinator};
use ruv_swarm_persistence::{MemoryPersistence, SqlitePersistence};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Duration;

/// Test helper to create a swarm with a specific topology
async fn create_test_swarm(topology: Topology) -> Result<Swarm, SwarmError> {
    let config = SwarmConfig {
        topology,
        max_agents: 10,
        heartbeat_interval: Duration::from_secs(1),
        task_timeout: Duration::from_secs(30),
        persistence: Box::new(MemoryPersistence::new()),
    };
    
    Swarm::new(config).await
}

/// Test helper to load test data for neural network training
fn load_test_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    // XOR problem as test data
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swarm_initialization() {
        let swarm = create_test_swarm(Topology::Star).await.unwrap();
        assert_eq!(swarm.agent_count(), 0);
        assert_eq!(swarm.topology(), Topology::Star);
    }

    #[tokio::test]
    async fn test_agent_spawning() {
        let mut swarm = create_test_swarm(Topology::Mesh).await.unwrap();
        
        let agent_id = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        assert!(swarm.has_agent(&agent_id));
        assert_eq!(swarm.agent_count(), 1);
        
        // Spawn multiple agents
        let mut agents = vec![];
        for _ in 0..3 {
            let id = swarm.spawn(AgentType::DataAnalyzer).await.unwrap();
            agents.push(id);
        }
        
        assert_eq!(swarm.agent_count(), 4);
        
        // Verify all agents are properly registered
        for agent_id in &agents {
            assert!(swarm.has_agent(agent_id));
        }
    }

    #[tokio::test]
    async fn test_distributed_neural_training() {
        let mut swarm = create_test_swarm(Topology::Mesh).await.unwrap();
        
        // Spawn neural processing agents
        let agents = vec![
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
        ];
        
        // Create training task
        let task = Task::TrainModel {
            data: load_test_data(),
            strategy: Strategy::DataParallel,
            epochs: 100,
            learning_rate: 0.1,
        };
        
        // Distribute training task
        let result = swarm.orchestrate(task).await.unwrap();
        
        match result {
            TaskResult::ModelTrained { accuracy, loss, .. } => {
                assert!(accuracy > 0.8, "Model accuracy should be > 80%, got {}", accuracy);
                assert!(loss < 0.2, "Model loss should be < 0.2, got {}", loss);
            }
            _ => panic!("Expected ModelTrained result"),
        }
    }

    #[tokio::test]
    async fn test_hierarchical_task_distribution() {
        let mut swarm = create_test_swarm(Topology::Hierarchical).await.unwrap();
        
        // Spawn coordinator at top level
        let coordinator = swarm.spawn(AgentType::Coordinator).await.unwrap();
        
        // Spawn worker agents
        let workers = vec![
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::DataAnalyzer).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
        ];
        
        // Set hierarchy
        swarm.set_hierarchy(coordinator.clone(), workers.clone()).await.unwrap();
        
        // Create complex task requiring coordination
        let task = Task::ComplexAnalysis {
            subtasks: vec![
                Task::DataPreprocessing { data: load_test_data() },
                Task::TrainModel { 
                    data: load_test_data(), 
                    strategy: Strategy::ModelParallel,
                    epochs: 50,
                    learning_rate: 0.05,
                },
                Task::EvaluateModel { test_data: load_test_data() },
            ],
        };
        
        // Execute hierarchical task
        let result = swarm.orchestrate(task).await.unwrap();
        
        match result {
            TaskResult::ComplexAnalysisComplete { subtask_results, .. } => {
                assert_eq!(subtask_results.len(), 3);
                // Verify all subtasks completed successfully
                for (i, subtask_result) in subtask_results.iter().enumerate() {
                    assert!(subtask_result.is_success(), "Subtask {} failed", i);
                }
            }
            _ => panic!("Expected ComplexAnalysisComplete result"),
        }
    }

    #[tokio::test]
    async fn test_agent_communication() {
        let mut swarm = create_test_swarm(Topology::FullyConnected).await.unwrap();
        
        let agent1 = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        let agent2 = swarm.spawn(AgentType::DataAnalyzer).await.unwrap();
        
        // Test direct messaging
        let message = Message {
            from: agent1.clone(),
            to: Some(agent2.clone()),
            msg_type: MessageType::Data(vec![1.0, 2.0, 3.0]),
            timestamp: std::time::SystemTime::now(),
        };
        
        swarm.send_message(message).await.unwrap();
        
        // Verify message was received
        let received = swarm.get_agent_messages(&agent2).await.unwrap();
        assert_eq!(received.len(), 1);
        
        match &received[0].msg_type {
            MessageType::Data(data) => assert_eq!(data, &vec![1.0, 2.0, 3.0]),
            _ => panic!("Expected Data message type"),
        }
    }

    #[tokio::test]
    async fn test_broadcast_communication() {
        let mut swarm = create_test_swarm(Topology::FullyConnected).await.unwrap();
        
        let sender = swarm.spawn(AgentType::Coordinator).await.unwrap();
        let receivers = vec![
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::DataAnalyzer).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
        ];
        
        // Broadcast message
        let message = Message {
            from: sender.clone(),
            to: None, // Broadcast
            msg_type: MessageType::Broadcast("Start processing".to_string()),
            timestamp: std::time::SystemTime::now(),
        };
        
        swarm.send_message(message).await.unwrap();
        
        // Verify all agents received the broadcast
        for receiver in &receivers {
            let messages = swarm.get_agent_messages(receiver).await.unwrap();
            assert_eq!(messages.len(), 1);
            
            match &messages[0].msg_type {
                MessageType::Broadcast(msg) => assert_eq!(msg, "Start processing"),
                _ => panic!("Expected Broadcast message type"),
            }
        }
    }

    #[tokio::test]
    async fn test_swarm_state_persistence() {
        // Create swarm with SQLite persistence
        let persistence = SqlitePersistence::new(":memory:").await.unwrap();
        let config = SwarmConfig {
            topology: Topology::Star,
            max_agents: 10,
            heartbeat_interval: Duration::from_secs(1),
            task_timeout: Duration::from_secs(30),
            persistence: Box::new(persistence),
        };
        
        let mut swarm = Swarm::new(config).await.unwrap();
        
        // Create some state
        let agent1 = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        let agent2 = swarm.spawn(AgentType::DataAnalyzer).await.unwrap();
        
        // Save swarm state
        swarm.save_state().await.unwrap();
        
        // Simulate restart by creating new swarm with same persistence
        let persistence2 = SqlitePersistence::new(":memory:").await.unwrap();
        let config2 = SwarmConfig {
            topology: Topology::Star,
            max_agents: 10,
            heartbeat_interval: Duration::from_secs(1),
            task_timeout: Duration::from_secs(30),
            persistence: Box::new(persistence2),
        };
        
        let swarm2 = Swarm::restore(config2).await.unwrap();
        
        // Verify state was restored
        assert_eq!(swarm2.agent_count(), 2);
        assert!(swarm2.has_agent(&agent1));
        assert!(swarm2.has_agent(&agent2));
    }

    #[tokio::test]
    async fn test_task_result_aggregation() {
        let mut swarm = create_test_swarm(Topology::Star).await.unwrap();
        
        // Spawn multiple neural processors
        let agents: Vec<_> = (0..3)
            .map(|_| async {
                swarm.spawn(AgentType::NeuralProcessor).await.unwrap()
            })
            .collect::<Vec<_>>();
        
        let agents = futures::future::join_all(agents).await;
        
        // Create ensemble training task
        let task = Task::EnsembleTraining {
            data: load_test_data(),
            n_models: 3,
            aggregation_method: "voting".to_string(),
        };
        
        let result = swarm.orchestrate(task).await.unwrap();
        
        match result {
            TaskResult::EnsembleTrained { models, combined_accuracy, .. } => {
                assert_eq!(models.len(), 3);
                assert!(combined_accuracy > 0.85, "Ensemble accuracy should be > 85%");
            }
            _ => panic!("Expected EnsembleTrained result"),
        }
    }

    #[tokio::test]
    async fn test_dynamic_topology_change() {
        let mut swarm = create_test_swarm(Topology::Star).await.unwrap();
        
        // Spawn agents
        let agents: Vec<_> = (0..5)
            .map(|_| async {
                swarm.spawn(AgentType::NeuralProcessor).await.unwrap()
            })
            .collect::<Vec<_>>();
        
        let agents = futures::future::join_all(agents).await;
        
        // Change topology to mesh
        swarm.change_topology(Topology::Mesh).await.unwrap();
        assert_eq!(swarm.topology(), Topology::Mesh);
        
        // Verify all agents can communicate
        let test_message = Message {
            from: agents[0].clone(),
            to: None,
            msg_type: MessageType::Broadcast("Topology changed".to_string()),
            timestamp: std::time::SystemTime::now(),
        };
        
        swarm.send_message(test_message).await.unwrap();
        
        // All agents should receive the broadcast in mesh topology
        for agent in &agents[1..] {
            let messages = swarm.get_agent_messages(agent).await.unwrap();
            assert!(!messages.is_empty());
        }
    }

    #[tokio::test]
    async fn test_concurrent_task_execution() {
        let mut swarm = create_test_swarm(Topology::FullyConnected).await.unwrap();
        
        // Spawn agents
        for _ in 0..5 {
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        }
        
        // Create multiple concurrent tasks
        let tasks = vec![
            Task::TrainModel {
                data: load_test_data(),
                strategy: Strategy::DataParallel,
                epochs: 50,
                learning_rate: 0.1,
            },
            Task::DataPreprocessing {
                data: load_test_data(),
            },
            Task::EvaluateModel {
                test_data: load_test_data(),
            },
        ];
        
        // Execute tasks concurrently
        let results = futures::future::join_all(
            tasks.into_iter().map(|task| swarm.orchestrate(task))
        ).await;
        
        // Verify all tasks completed successfully
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok(), "Task {} failed: {:?}", i, result);
        }
    }

    #[tokio::test]
    async fn test_resource_allocation() {
        let mut swarm = create_test_swarm(Topology::Star).await.unwrap();
        
        // Spawn agents with resource constraints
        let config = SwarmConfig {
            topology: Topology::Star,
            max_agents: 3, // Limited agents
            heartbeat_interval: Duration::from_secs(1),
            task_timeout: Duration::from_secs(30),
            persistence: Box::new(MemoryPersistence::new()),
        };
        
        let mut swarm = Swarm::new(config).await.unwrap();
        
        // Fill up agent capacity
        for _ in 0..3 {
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        }
        
        // Try to spawn one more - should fail
        let result = swarm.spawn(AgentType::DataAnalyzer).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SwarmError::MaxAgentsReached));
    }
}