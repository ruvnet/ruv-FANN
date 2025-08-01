//! Async swarm orchestrator implementation
//!
//! This module provides an async-first swarm orchestrator that manages
//! distributed AI agents with full async/await support throughout the pipeline.

use crate::agent::{AgentId, AgentStatus, DynamicAgent};
use crate::error::{Result, SwarmError};
use crate::task::{DistributionStrategy, Task, TaskId};
use crate::topology::{Topology, TopologyType};

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    collections::BTreeMap as HashMap,
    format,
    string::{String, ToString},
    vec::Vec,
};
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(feature = "std")]
use tokio::sync::{Mutex, RwLock};

/// Async swarm configuration
#[derive(Debug, Clone)]
pub struct AsyncSwarmConfig {
    /// The network topology type for agent connections
    pub topology_type: TopologyType,
    /// Strategy for distributing tasks among agents
    pub distribution_strategy: DistributionStrategy,
    /// Maximum number of agents allowed in the swarm
    pub max_agents: usize,
    /// Whether to enable automatic scaling of agent count
    pub enable_auto_scaling: bool,
    /// Interval in milliseconds between health checks
    pub health_check_interval_ms: u64,
    /// Maximum concurrent tasks per agent
    pub max_concurrent_tasks_per_agent: usize,
    /// Task timeout in milliseconds
    pub task_timeout_ms: u64,
}

impl Default for AsyncSwarmConfig {
    fn default() -> Self {
        AsyncSwarmConfig {
            topology_type: TopologyType::Mesh,
            distribution_strategy: DistributionStrategy::LeastLoaded,
            max_agents: 100,
            enable_auto_scaling: false,
            health_check_interval_ms: 5000,
            max_concurrent_tasks_per_agent: 10,
            task_timeout_ms: 30000,
        }
    }
}

/// Async swarm orchestrator with full async/await support
#[cfg(feature = "std")]
pub struct AsyncSwarm {
    config: AsyncSwarmConfig,
    agents: Arc<RwLock<HashMap<AgentId, DynamicAgent>>>,
    topology: Arc<Mutex<Topology>>,
    task_queue: Arc<Mutex<Vec<Task>>>,
    task_assignments: Arc<RwLock<HashMap<TaskId, AgentId>>>,
    agent_loads: Arc<RwLock<HashMap<AgentId, usize>>>,
    health_check_handle: Option<tokio::task::JoinHandle<()>>,
    /// Coordinator agent ID for star topology
    star_coordinator: Arc<RwLock<Option<AgentId>>>,
}

#[cfg(feature = "std")]
impl AsyncSwarm {
    /// Create a new async swarm orchestrator
    pub fn new(config: AsyncSwarmConfig) -> Self {
        AsyncSwarm {
            topology: Arc::new(Mutex::new(Topology::new(config.topology_type))),
            config,
            agents: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(Vec::new())),
            task_assignments: Arc::new(RwLock::new(HashMap::new())),
            agent_loads: Arc::new(RwLock::new(HashMap::new())),
            health_check_handle: None,
            star_coordinator: Arc::new(RwLock::new(None)),
        }
    }

    /// Register an agent with the swarm
    /// 
    /// # Errors
    /// 
    /// Returns an error if the maximum number of agents has been reached.
    pub async fn register_agent(&self, agent: DynamicAgent) -> Result<()> {
        let mut agents = self.agents.write().await;
        
        if agents.len() >= self.config.max_agents {
            return Err(SwarmError::ResourceExhausted {
                resource: "agent slots".into(),
            });
        }

        let agent_id = agent.id().to_string();
        agents.insert(agent_id.clone(), agent);
        
        // Initialize agent load tracking
        let mut agent_loads = self.agent_loads.write().await;
        agent_loads.insert(agent_id.clone(), 0);
        drop(agent_loads);

        // Update topology based on type
        let mut topology = self.topology.lock().await;
        match self.config.topology_type {
            TopologyType::Mesh => {
                // Connect to all existing agents
                for existing_id in agents.keys() {
                    if existing_id != &agent_id {
                        topology.add_connection(agent_id.clone(), existing_id.clone());
                    }
                }
            }
            TopologyType::Star => {
                // First agent becomes the coordinator
                let mut star_coordinator = self.star_coordinator.write().await;
                if star_coordinator.is_none() {
                    *star_coordinator = Some(agent_id.clone());
                } else {
                    // Connect new agent to the coordinator
                    if let Some(coordinator) = star_coordinator.as_ref() {
                        topology.add_connection(agent_id.clone(), coordinator.clone());
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Remove an agent from the swarm
    /// 
    /// # Errors
    /// 
    /// Returns an error if the agent is not found in the swarm.
    pub async fn unregister_agent(&self, agent_id: &AgentId) -> Result<()> {
        let mut agents = self.agents.write().await;
        
        agents
            .remove(agent_id)
            .ok_or_else(|| SwarmError::AgentNotFound {
                id: agent_id.to_string(),
            })?;

        // Remove from load tracking
        let mut agent_loads = self.agent_loads.write().await;
        agent_loads.remove(agent_id);
        drop(agent_loads);

        // Remove from topology
        let mut topology = self.topology.lock().await;
        let neighbors = topology.get_neighbors(agent_id).cloned();
        if let Some(neighbors) = neighbors {
            for neighbor in neighbors {
                topology.remove_connection(agent_id, &neighbor);
            }
        }
        
        // Handle star topology coordinator removal
        if self.config.topology_type == TopologyType::Star {
            let mut star_coordinator = self.star_coordinator.write().await;
            if let Some(ref coordinator) = *star_coordinator {
                if coordinator == agent_id {
                    // If coordinator is removed, elect a new one and reconnect all agents
                    *star_coordinator = agents.keys().next().cloned();
                    
                    if let Some(new_coordinator) = star_coordinator.as_ref() {
                        // Reconnect all agents to the new coordinator
                        for agent in agents.keys() {
                            if agent != new_coordinator {
                                topology.add_connection(agent.clone(), new_coordinator.clone());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Submit a task to the swarm
    /// 
    /// # Errors
    /// 
    /// Currently does not return errors, but may in future implementations.
    pub async fn submit_task(&self, task: Task) -> Result<()> {
        let mut task_queue = self.task_queue.lock().await;
        task_queue.push(task);
        Ok(())
    }

    /// Distribute tasks to agents based on distribution strategy
    /// 
    /// # Errors
    /// 
    /// Returns an error if task assignment fails.
    pub async fn distribute_tasks(&self) -> Result<Vec<(TaskId, AgentId)>> {
        let mut assignments = Vec::new();
        let mut task_queue = self.task_queue.lock().await;
        let mut tasks_to_assign = Vec::new();

        // Collect tasks that can be assigned
        while let Some(task) = task_queue.pop() {
            tasks_to_assign.push(task);
        }
        drop(task_queue);

        for task in tasks_to_assign {
            if let Ok(Some(agent_id)) = self.select_agent_for_task(&task).await {
                let task_id = task.id.clone();

                // Assign task to agent
                let mut task_assignments = self.task_assignments.write().await;
                task_assignments.insert(task_id.clone(), agent_id.clone());
                drop(task_assignments);

                // Update agent load
                let mut agent_loads = self.agent_loads.write().await;
                if let Some(load) = agent_loads.get_mut(&agent_id) {
                    *load += 1;
                }
                drop(agent_loads);

                assignments.push((task_id, agent_id));
            } else {
                // No suitable agent found, put task back in queue
                let mut task_queue = self.task_queue.lock().await;
                task_queue.push(task);
            }
        }

        Ok(assignments)
    }

    /// Select an agent for a task based on distribution strategy
    async fn select_agent_for_task(&self, task: &Task) -> Result<Option<AgentId>> {
        let agents = self.agents.read().await;
        let agent_loads = self.agent_loads.read().await;

        let available_agents: Vec<&AgentId> = agents
            .iter()
            .filter(|(_, agent)| agent.status() == AgentStatus::Running && agent.can_handle(task))
            .map(|(id, _)| id)
            .collect();

        if available_agents.is_empty() {
            return Ok(None);
        }

        let selected = match self.config.distribution_strategy {
            DistributionStrategy::RoundRobin => {
                // Simple round-robin (would need state to track last assigned)
                available_agents.first().copied()
            }
            DistributionStrategy::LeastLoaded => {
                // Select agent with lowest load
                available_agents
                    .iter()
                    .min_by_key(|id| agent_loads.get(id.as_str()).unwrap_or(&0))
                    .copied()
            }
            DistributionStrategy::Random => {
                // Random selection using thread-local RNG
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                available_agents.choose(&mut rng).copied()
            }
            DistributionStrategy::Priority => {
                // Priority-based (would consider task priority)
                available_agents.first().copied()
            }
            DistributionStrategy::CapabilityBased => {
                // Already filtered by capability
                available_agents.first().copied()
            }
        };

        Ok(selected.cloned())
    }

    /// Get the status of all agents
    pub async fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus> {
        let agents = self.agents.read().await;
        agents
            .iter()
            .map(|(id, agent)| (id.clone(), agent.status()))
            .collect()
    }

    /// Get the current task queue size
    pub async fn task_queue_size(&self) -> usize {
        let task_queue = self.task_queue.lock().await;
        task_queue.len()
    }

    /// Get assigned tasks count
    pub async fn assigned_tasks_count(&self) -> usize {
        let task_assignments = self.task_assignments.read().await;
        task_assignments.len()
    }

    /// Check if agent exists by ID
    pub async fn has_agent(&self, agent_id: &AgentId) -> bool {
        let agents = self.agents.read().await;
        agents.contains_key(agent_id)
    }

    /// Start all agents
    /// 
    /// # Errors
    /// 
    /// Returns an error if any agent fails to start.
    pub async fn start_all_agents(&self) -> Result<()> {
        let mut agents = self.agents.write().await;
        
        for (id, agent) in agents.iter_mut() {
            agent.start().await.map_err(|e| {
                SwarmError::Custom(format!("Failed to start agent {id}: {e:?}"))
            })?;
        }
        
        Ok(())
    }

    /// Shutdown all agents
    /// 
    /// # Errors
    /// 
    /// Returns an error if any agent fails to shutdown.
    pub async fn shutdown_all_agents(&self) -> Result<()> {
        let mut agents = self.agents.write().await;
        
        for (id, agent) in agents.iter_mut() {
            agent.shutdown().await.map_err(|e| {
                SwarmError::Custom(format!("Failed to shutdown agent {id}: {e:?}"))
            })?;
        }
        
        Ok(())
    }

    /// Start health check monitoring
    /// 
    /// # Errors
    /// 
    /// Returns an error if the monitoring system fails to start.
    pub fn start_health_monitoring(&mut self) -> Result<()> {
        if self.health_check_handle.is_some() {
            return Ok(()); // Already running
        }

        let agents = self.agents.clone();
        let interval = self.config.health_check_interval_ms;
        
        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(
                tokio::time::Duration::from_millis(interval)
            );
            
            loop {
                interval_timer.tick().await;
                
                let agents = agents.read().await;
                for (id, agent) in agents.iter() {
                    if agent.status() == AgentStatus::Error {
                        // Log error or trigger recovery
                        eprintln!("Agent {id} in error state");
                    }
                }
            }
        });
        
        self.health_check_handle = Some(handle);
        Ok(())
    }

    /// Stop health check monitoring
    pub fn stop_health_monitoring(&mut self) {
        if let Some(handle) = self.health_check_handle.take() {
            handle.abort();
        }
    }

    /// Process tasks concurrently with timeout
    /// 
    /// # Errors
    /// 
    /// Returns an error if task processing fails.
    pub async fn process_tasks_concurrently(&self, max_concurrent: usize) -> Result<Vec<(TaskId, Result<()>)>> {
        use futures::stream::{FuturesUnordered, StreamExt};
        use tokio::time::{timeout, Duration};

        let assignments = self.distribute_tasks().await?;
        let mut results = Vec::new();
        
        let mut futures = FuturesUnordered::new();
        let mut pending = 0;
        
        for (task_id, agent_id) in assignments {
            if pending >= max_concurrent {
                // Wait for some tasks to complete
                if let Some((completed_task_id, result)) = futures.next().await {
                    results.push((completed_task_id, result));
                    pending -= 1;
                }
            }
            
            let agents = self.agents.clone();
            let task_timeout = Duration::from_millis(self.config.task_timeout_ms);
            
            futures.push(async move {
                let result = timeout(task_timeout, async {
                    let mut agents = agents.write().await;
                    if let Some(agent) = agents.get_mut(&agent_id) {
                        // In a real implementation, we would get the task and process it
                        // For now, just simulate processing
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        Ok(())
                    } else {
                        Err(SwarmError::AgentNotFound { id: agent_id.to_string() })
                    }
                }).await;
                
                match result {
                    Ok(task_result) => (task_id, task_result),
                    Err(_) => (task_id, Err(SwarmError::Custom("Task timeout".to_string()))),
                }
            });
            
            pending += 1;
        }
        
        // Wait for remaining tasks to complete
        while let Some((task_id, result)) = futures.next().await {
            results.push((task_id, result));
        }
        
        Ok(results)
    }

    /// Get swarm metrics
    pub async fn metrics(&self) -> AsyncSwarmMetrics {
        let agents = self.agents.read().await;
        let task_queue = self.task_queue.lock().await;
        let task_assignments = self.task_assignments.read().await;
        let topology = self.topology.lock().await;

        AsyncSwarmMetrics {
            total_agents: agents.len(),
            active_agents: agents
                .iter()
                .filter(|(_, agent)| agent.status() == AgentStatus::Running)
                .count(),
            queued_tasks: task_queue.len(),
            assigned_tasks: task_assignments.len(),
            total_connections: topology.connection_count(),
            avg_agent_load: {
                let agent_loads = self.agent_loads.read().await;
                if agent_loads.is_empty() {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        let sum = agent_loads.values().sum::<usize>() as f64;
                        let count = agent_loads.len() as f64;
                        sum / count
                    }
                }
            },
        }
    }
}

#[cfg(feature = "std")]
impl Drop for AsyncSwarm {
    fn drop(&mut self) {
        if let Some(handle) = self.health_check_handle.take() {
            handle.abort();
        }
    }
}

/// Async swarm metrics
#[derive(Debug, Clone)]
pub struct AsyncSwarmMetrics {
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
    /// Average load per agent
    pub avg_agent_load: f64,
}

/// Async swarm trait for testing and extensibility
#[cfg(feature = "std")]
pub trait AsyncSwarmTrait {
    /// Register an agent with the swarm
    fn register_agent(&self, agent: DynamicAgent) -> impl std::future::Future<Output = Result<()>> + Send;
    
    /// Submit a task to the swarm
    fn submit_task(&self, task: Task) -> impl std::future::Future<Output = Result<()>> + Send;
    
    /// Distribute tasks to agents
    fn distribute_tasks(&self) -> impl std::future::Future<Output = Result<Vec<(TaskId, AgentId)>>> + Send;
    
    /// Get swarm metrics
    fn metrics(&self) -> impl std::future::Future<Output = AsyncSwarmMetrics> + Send;
}

#[cfg(feature = "std")]
impl AsyncSwarmTrait for AsyncSwarm {
    async fn register_agent(&self, agent: DynamicAgent) -> Result<()> {
        self.register_agent(agent).await
    }

    async fn submit_task(&self, task: Task) -> Result<()> {
        self.submit_task(task).await
    }

    async fn distribute_tasks(&self) -> Result<Vec<(TaskId, AgentId)>> {
        self.distribute_tasks().await
    }

    async fn metrics(&self) -> AsyncSwarmMetrics {
        self.metrics().await
    }
}