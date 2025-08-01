//! Tests for failure scenarios and recovery mechanisms
//! 
//! These tests validate the swarm's behavior under various failure conditions
//! including agent crashes, network partitions, message loss, and recovery.

use ruv_swarm_core::{
    agent::{Agent, AgentId, AgentType, AgentStatus},
    swarm::{Swarm, SwarmConfig, Topology},
    task::{Task, TaskResult, TaskStatus},
    error::{SwarmError, AgentError},
    message::{Message, MessageType},
    network::{NetworkPartition, PartitionStrategy},
};
use ruv_swarm_transport::{Transport, TransportError};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Helper to create a swarm with failure injection capabilities
async fn create_failure_swarm() -> Result<Swarm, Box<dyn std::error::Error>> {
    let config = SwarmConfig {
        topology: Topology::Mesh,
        max_agents: 20,
        heartbeat_interval: Duration::from_millis(500), // Faster for testing
        task_timeout: Duration::from_secs(10),
        persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
        failure_detection_enabled: true,
        recovery_enabled: true,
    };
    
    Swarm::new(config).await.map_err(Into::into)
}

/// Failure injection transport wrapper
struct FailureInjectionTransport {
    inner: Box<dyn Transport>,
    failure_rate: Arc<RwLock<f64>>,
    partition: Arc<RwLock<Option<NetworkPartition>>>,
    dropped_messages: Arc<Mutex<Vec<Message>>>,
}

impl FailureInjectionTransport {
    fn new(inner: Box<dyn Transport>) -> Self {
        Self {
            inner,
            failure_rate: Arc::new(RwLock::new(0.0)),
            partition: Arc::new(RwLock::new(None)),
            dropped_messages: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    async fn set_failure_rate(&self, rate: f64) {
        *self.failure_rate.write().await = rate;
    }
    
    async fn set_partition(&self, partition: Option<NetworkPartition>) {
        *self.partition.write().await = partition;
    }
}

#[async_trait::async_trait]
impl Transport for FailureInjectionTransport {
    async fn send(&self, msg: Message) -> Result<(), TransportError> {
        // Check for network partition
        if let Some(partition) = self.partition.read().await.as_ref() {
            if partition.is_partitioned(&msg.from, &msg.to.unwrap_or_default()) {
                return Err(TransportError::NetworkPartition);
            }
        }
        
        // Simulate message loss
        let failure_rate = *self.failure_rate.read().await;
        if rand::random::<f64>() < failure_rate {
            self.dropped_messages.lock().await.push(msg);
            return Err(TransportError::MessageLost);
        }
        
        self.inner.send(msg).await
    }
    
    async fn receive(&mut self) -> Result<Message, TransportError> {
        self.inner.receive().await
    }
    
    fn connect(&mut self, peer: PeerId) -> Result<(), TransportError> {
        self.inner.connect(peer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_crash_detection() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Spawn agents
        let healthy_agent = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        let crash_agent = swarm.spawn(AgentType::DataAnalyzer).await.unwrap();
        
        // Simulate agent crash
        swarm.simulate_agent_crash(&crash_agent).await.unwrap();
        
        // Wait for failure detection
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Verify crash was detected
        let agent_status = swarm.get_agent_status(&crash_agent).await.unwrap();
        assert_eq!(agent_status, AgentStatus::Crashed);
        
        // Verify healthy agent is still running
        let healthy_status = swarm.get_agent_status(&healthy_agent).await.unwrap();
        assert_eq!(healthy_status, AgentStatus::Active);
        
        // Verify swarm health metrics
        let health = swarm.get_health_status().await.unwrap();
        assert_eq!(health.failed_agents, 1);
        assert_eq!(health.active_agents, 1);
    }

    #[tokio::test]
    async fn test_agent_recovery() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Spawn agents with recovery enabled
        let agent1 = swarm.spawn_with_recovery(AgentType::NeuralProcessor).await.unwrap();
        let agent2 = swarm.spawn_with_recovery(AgentType::DataAnalyzer).await.unwrap();
        
        // Create task
        let task = Task::DataProcessing {
            input_data: vec![1.0, 2.0, 3.0],
            operations: vec!["normalize"],
            output_format: "json",
        };
        
        let task_id = swarm.submit_task(task).await.unwrap();
        
        // Simulate agent crash during task execution
        tokio::time::sleep(Duration::from_millis(100)).await;
        swarm.simulate_agent_crash(&agent1).await.unwrap();
        
        // Wait for recovery
        tokio::time::sleep(Duration::from_secs(3)).await;
        
        // Verify agent was recovered
        let new_status = swarm.get_agent_status(&agent1).await.unwrap();
        assert_eq!(new_status, AgentStatus::Active);
        
        // Verify task was reassigned and completed
        let task_status = swarm.get_task_status(&task_id).await.unwrap();
        assert!(matches!(task_status, TaskStatus::Completed(_)));
    }

    #[tokio::test]
    async fn test_network_partition() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Create agents
        let agents: Vec<_> = futures::future::join_all(
            (0..6).map(|_| swarm.spawn(AgentType::NeuralProcessor))
        ).await.into_iter().collect::<Result<_, _>>().unwrap();
        
        // Create network partition (split agents into two groups)
        let partition = NetworkPartition::new(
            PartitionStrategy::Split {
                group1: agents[..3].to_vec(),
                group2: agents[3..].to_vec(),
            }
        );
        
        swarm.inject_network_partition(partition).await.unwrap();
        
        // Try to send message across partition
        let result = swarm.send_direct_message(
            &agents[0],
            &agents[4],
            b"test message".to_vec(),
        ).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SwarmError::NetworkPartition));
        
        // Messages within same partition should work
        let result = swarm.send_direct_message(
            &agents[0],
            &agents[1],
            b"test message".to_vec(),
        ).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_partition_recovery() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Create agents
        let agents: Vec<_> = futures::future::join_all(
            (0..4).map(|_| swarm.spawn(AgentType::NeuralProcessor))
        ).await.into_iter().collect::<Result<_, _>>().unwrap();
        
        // Submit task that requires all agents
        let task = Task::DistributedComputation {
            subtasks: vec![
                ("part1", vec![1.0, 2.0]),
                ("part2", vec![3.0, 4.0]),
                ("part3", vec![5.0, 6.0]),
                ("part4", vec![7.0, 8.0]),
            ],
        };
        
        let task_id = swarm.submit_task(task).await.unwrap();
        
        // Create partition after task starts
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let partition = NetworkPartition::new(
            PartitionStrategy::Split {
                group1: agents[..2].to_vec(),
                group2: agents[2..].to_vec(),
            }
        );
        
        swarm.inject_network_partition(partition).await.unwrap();
        
        // Wait a bit
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        // Heal partition
        swarm.heal_network_partition().await.unwrap();
        
        // Task should eventually complete
        let timeout = Duration::from_secs(10);
        let start = std::time::Instant::now();
        
        loop {
            let status = swarm.get_task_status(&task_id).await.unwrap();
            if matches!(status, TaskStatus::Completed(_)) {
                break;
            }
            
            if start.elapsed() > timeout {
                panic!("Task did not complete after partition healed");
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    #[tokio::test]
    async fn test_message_loss_handling() {
        let transport = FailureInjectionTransport::new(
            Box::new(ruv_swarm_transport::InProcessTransport::new())
        );
        
        let transport_handle = transport.clone();
        
        let config = SwarmConfig {
            topology: Topology::FullyConnected,
            max_agents: 10,
            heartbeat_interval: Duration::from_millis(500),
            task_timeout: Duration::from_secs(10),
            persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
            transport: Box::new(transport),
            message_retry_enabled: true,
            max_retries: 3,
        };
        
        let mut swarm = Swarm::new(config).await.unwrap();
        
        // Spawn agents
        let agent1 = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        let agent2 = swarm.spawn(AgentType::DataAnalyzer).await.unwrap();
        
        // Set 50% message loss rate
        transport_handle.set_failure_rate(0.5).await;
        
        // Send messages with retry
        let mut success_count = 0;
        let total_messages = 20;
        
        for i in 0..total_messages {
            let result = swarm.send_reliable_message(
                &agent1,
                &agent2,
                format!("message_{}", i).into_bytes(),
            ).await;
            
            if result.is_ok() {
                success_count += 1;
            }
        }
        
        // Despite 50% loss rate, most messages should succeed due to retries
        assert!(success_count as f64 / total_messages as f64 > 0.8);
        
        // Check dropped messages were tracked
        let dropped = transport_handle.dropped_messages.lock().await;
        assert!(!dropped.is_empty());
    }

    #[tokio::test]
    async fn test_cascading_failures() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Create dependent agents
        let coordinator = swarm.spawn(AgentType::Coordinator).await.unwrap();
        let workers = vec![
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
        ];
        
        // Set up dependencies
        for worker in &workers {
            swarm.add_dependency(&coordinator, worker).await.unwrap();
        }
        
        // Start dependent task
        let task = Task::HierarchicalComputation {
            coordinator: coordinator.clone(),
            subtasks: vec![
                ("train_model", vec![1.0, 2.0]),
                ("validate_model", vec![3.0, 4.0]),
                ("test_model", vec![5.0, 6.0]),
            ],
        };
        
        let task_id = swarm.submit_task(task).await.unwrap();
        
        // Crash coordinator
        tokio::time::sleep(Duration::from_millis(200)).await;
        swarm.simulate_agent_crash(&coordinator).await.unwrap();
        
        // Wait for cascading failure detection
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // All dependent tasks should be marked as failed
        let task_status = swarm.get_task_status(&task_id).await.unwrap();
        assert!(matches!(task_status, TaskStatus::Failed(_)));
        
        // Workers should detect coordinator failure
        for worker in &workers {
            let status = swarm.get_agent_status(worker).await.unwrap();
            assert!(matches!(status, AgentStatus::Orphaned | AgentStatus::Idle));
        }
    }

    #[tokio::test]
    async fn test_resource_exhaustion() {
        let config = SwarmConfig {
            topology: Topology::Star,
            max_agents: 5, // Limited resources
            heartbeat_interval: Duration::from_millis(500),
            task_timeout: Duration::from_secs(10),
            persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
            memory_limit: Some(100 * 1024 * 1024), // 100MB limit
        };
        
        let mut swarm = Swarm::new(config).await.unwrap();
        
        // Fill up agent capacity
        for _ in 0..5 {
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        }
        
        // Try to spawn more - should fail
        let result = swarm.spawn(AgentType::DataAnalyzer).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SwarmError::ResourceExhausted));
        
        // Submit memory-intensive task
        let large_task = Task::MemoryIntensive {
            data_size_mb: 200, // Exceeds limit
        };
        
        let result = swarm.submit_task(large_task).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SwarmError::MemoryLimitExceeded));
    }

    #[tokio::test]
    async fn test_byzantine_agent_detection() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Enable byzantine fault detection
        swarm.enable_byzantine_detection().await.unwrap();
        
        // Spawn agents
        let honest_agents = vec![
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
            swarm.spawn(AgentType::NeuralProcessor).await.unwrap(),
        ];
        
        let byzantine_agent = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
        
        // Configure byzantine behavior
        swarm.inject_byzantine_behavior(&byzantine_agent, |msg| {
            // Corrupt messages
            let mut corrupted = msg.clone();
            if let MessageType::Data(ref mut data) = corrupted.msg_type {
                for value in data {
                    *value = -*value; // Invert values
                }
            }
            corrupted
        }).await.unwrap();
        
        // Run consensus task
        let task = Task::ConsensusComputation {
            input: vec![1.0, 2.0, 3.0],
            min_agreement: 0.75,
        };
        
        let result = swarm.orchestrate(task).await.unwrap();
        
        // Byzantine agent should be detected and excluded
        let byzantine_status = swarm.get_agent_status(&byzantine_agent).await.unwrap();
        assert_eq!(byzantine_status, AgentStatus::Quarantined);
        
        // Result should be correct despite byzantine agent
        match result {
            TaskResult::ConsensusReached { value, agreement, .. } => {
                assert!(agreement > 0.75);
                assert_eq!(value, vec![1.0, 2.0, 3.0]); // Correct result
            }
            _ => panic!("Expected consensus result"),
        }
    }

    #[tokio::test]
    async fn test_deadlock_detection() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Enable deadlock detection
        swarm.enable_deadlock_detection().await.unwrap();
        
        // Create agents
        let agent1 = swarm.spawn(AgentType::DataProcessor).await.unwrap();
        let agent2 = swarm.spawn(AgentType::DataProcessor).await.unwrap();
        let agent3 = swarm.spawn(AgentType::DataProcessor).await.unwrap();
        
        // Create circular dependency task
        let task = Task::CircularDependency {
            steps: vec![
                (agent1.clone(), agent2.clone(), "resource_a"),
                (agent2.clone(), agent3.clone(), "resource_b"),
                (agent3.clone(), agent1.clone(), "resource_c"),
            ],
        };
        
        let task_id = swarm.submit_task(task).await.unwrap();
        
        // Wait for deadlock detection
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Deadlock should be detected
        let deadlocks = swarm.get_detected_deadlocks().await.unwrap();
        assert!(!deadlocks.is_empty());
        
        // Task should be marked as deadlocked
        let task_status = swarm.get_task_status(&task_id).await.unwrap();
        assert!(matches!(task_status, TaskStatus::Deadlocked));
        
        // Swarm should attempt resolution
        let resolution = swarm.resolve_deadlock(&deadlocks[0]).await.unwrap();
        assert!(resolution.resolved);
    }

    #[tokio::test]
    async fn test_recovery_strategies() {
        let mut swarm = create_failure_swarm().await.unwrap();
        
        // Test different recovery strategies
        let strategies = vec![
            RecoveryStrategy::Restart,
            RecoveryStrategy::Migrate,
            RecoveryStrategy::Replicate,
            RecoveryStrategy::Checkpoint,
        ];
        
        for strategy in strategies {
            swarm.set_recovery_strategy(strategy.clone()).await.unwrap();
            
            let agent = swarm.spawn(AgentType::NeuralProcessor).await.unwrap();
            
            // Submit task
            let task = Task::SimpleComputation {
                input: vec![1.0, 2.0, 3.0],
            };
            
            let task_id = swarm.submit_task_to_agent(task, &agent).await.unwrap();
            
            // Simulate failure
            tokio::time::sleep(Duration::from_millis(100)).await;
            swarm.simulate_agent_crash(&agent).await.unwrap();
            
            // Wait for recovery
            tokio::time::sleep(Duration::from_secs(3)).await;
            
            // Verify recovery based on strategy
            match strategy {
                RecoveryStrategy::Restart => {
                    let status = swarm.get_agent_status(&agent).await.unwrap();
                    assert_eq!(status, AgentStatus::Active);
                }
                RecoveryStrategy::Migrate => {
                    let task_agent = swarm.get_task_agent(&task_id).await.unwrap();
                    assert_ne!(task_agent, agent); // Task migrated to different agent
                }
                RecoveryStrategy::Replicate => {
                    let replicas = swarm.get_agent_replicas(&agent).await.unwrap();
                    assert!(!replicas.is_empty());
                }
                RecoveryStrategy::Checkpoint => {
                    let task_status = swarm.get_task_status(&task_id).await.unwrap();
                    assert!(matches!(task_status, TaskStatus::Resumed));
                }
            }
        }
    }
}