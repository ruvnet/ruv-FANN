//! Swarm orchestrator implementation

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

/// Swarm configuration
#[derive(Debug, Clone)]
pub struct SwarmConfig {
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
}

impl Default for SwarmConfig {
    fn default() -> Self {
        SwarmConfig {
            topology_type: TopologyType::Mesh,
            distribution_strategy: DistributionStrategy::LeastLoaded,
            max_agents: 100,
            enable_auto_scaling: false,
            health_check_interval_ms: 5000,
        }
    }
}

/// Swarm orchestrator
pub struct Swarm {
    config: SwarmConfig,
    agents: HashMap<AgentId, DynamicAgent>,
    topology: Topology,
    task_queue: Vec<Task>,
    task_assignments: HashMap<TaskId, AgentId>,
    agent_loads: HashMap<AgentId, usize>,
    /// Coordinator agent ID for star topology
    star_coordinator: Option<AgentId>,
}

impl Swarm {
    /// Create a new swarm orchestrator
    pub fn new(config: SwarmConfig) -> Self {
        Swarm {
            topology: Topology::new(config.topology_type),
            config,
            agents: HashMap::new(),
            task_queue: Vec::new(),
            task_assignments: HashMap::new(),
            agent_loads: HashMap::new(),
            star_coordinator: None,
        }
    }

    /// Register an agent with the swarm
    /// 
    /// # Errors
    /// 
    /// Returns an error if the maximum number of agents has been reached.
    pub fn register_agent(&mut self, agent: DynamicAgent) -> Result<()> {
        if self.agents.len() >= self.config.max_agents {
            return Err(SwarmError::ResourceExhausted {
                resource: "agent slots".into(),
            });
        }

        let agent_id = agent.id().to_string();
        self.agents.insert(agent_id.clone(), agent);
        self.agent_loads.insert(agent_id.clone(), 0);

        // Update topology based on type
        match self.config.topology_type {
            TopologyType::Mesh => {
                // Connect to all existing agents
                for existing_id in self.agents.keys() {
                    if existing_id != &agent_id {
                        self.topology
                            .add_connection(agent_id.clone(), existing_id.clone());
                    }
                }
            }
            TopologyType::Star => {
                // First agent becomes the coordinator
                if self.star_coordinator.is_none() {
                    self.star_coordinator = Some(agent_id.clone());
                } else {
                    // Connect new agent to the coordinator
                    if let Some(coordinator) = &self.star_coordinator {
                        self.topology
                            .add_connection(agent_id.clone(), coordinator.clone());
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
    pub fn unregister_agent(&mut self, agent_id: &AgentId) -> Result<()> {
        self.agents
            .remove(agent_id)
            .ok_or_else(|| SwarmError::AgentNotFound {
                id: agent_id.to_string(),
            })?;

        self.agent_loads.remove(agent_id);

        // Remove from topology
        let neighbors = self.topology.get_neighbors(agent_id).cloned();
        if let Some(neighbors) = neighbors {
            for neighbor in neighbors {
                self.topology.remove_connection(agent_id, &neighbor);
            }
        }

        // Handle star topology coordinator removal
        if self.config.topology_type == TopologyType::Star {
            if let Some(ref coordinator) = self.star_coordinator {
                if coordinator == agent_id {
                    // If coordinator is removed, elect a new one and reconnect all agents
                    self.star_coordinator = self.agents.keys().next().cloned();
                    
                    if let Some(new_coordinator) = &self.star_coordinator {
                        // Reconnect all agents to the new coordinator
                        for agent in self.agents.keys() {
                            if agent != new_coordinator {
                                self.topology.add_connection(agent.clone(), new_coordinator.clone());
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
    pub fn submit_task(&mut self, task: Task) -> Result<()> {
        self.task_queue.push(task);
        Ok(())
    }

    /// Assign tasks to agents based on distribution strategy
    /// 
    /// # Errors
    /// 
    /// Returns an error if task assignment fails.
    #[allow(clippy::unused_async)]
    pub async fn distribute_tasks(&mut self) -> Result<Vec<(TaskId, AgentId)>> {
        let mut assignments = Vec::new();
        let mut tasks_to_assign = Vec::new();

        // Collect tasks that can be assigned
        while let Some(task) = self.task_queue.pop() {
            tasks_to_assign.push(task);
        }

        for task in tasks_to_assign {
            if let Some(agent_id) = self.select_agent_for_task(&task) {
                let task_id = task.id.clone();

                // Assign task to agent
                self.task_assignments
                    .insert(task_id.clone(), agent_id.clone());

                // Update agent load
                if let Some(load) = self.agent_loads.get_mut(&agent_id) {
                    *load += 1;
                }

                // In a real implementation, we would process the task here
                // For now, just add to assignments
                assignments.push((task_id, agent_id));
            } else {
                // No suitable agent found, put task back in queue
                self.task_queue.push(task);
            }
        }

        Ok(assignments)
    }

    /// Select an agent for a task based on distribution strategy
    fn select_agent_for_task(&self, task: &Task) -> Option<AgentId> {
        let available_agents: Vec<&AgentId> = self
            .agents
            .iter()
            .filter(|(_, agent)| agent.status() == AgentStatus::Running && agent.can_handle(task))
            .map(|(id, _)| id)
            .collect();

        if available_agents.is_empty() {
            return None;
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
                    .min_by_key(|id| self.agent_loads.get(id.as_str()).unwrap_or(&0))
                    .copied()
            }
            DistributionStrategy::Random => {
                // Random selection (simplified - would use proper RNG)
                available_agents.first().copied()
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

        selected.cloned()
    }

    /// Get the status of all agents
    pub fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus> {
        self.agents
            .iter()
            .map(|(id, agent)| (id.clone(), agent.status()))
            .collect()
    }

    /// Get the current task queue size
    pub fn task_queue_size(&self) -> usize {
        self.task_queue.len()
    }

    /// Get assigned tasks count
    pub fn assigned_tasks_count(&self) -> usize {
        self.task_assignments.len()
    }

    /// Get agent by ID
    pub fn get_agent(&self, agent_id: &AgentId) -> Option<&DynamicAgent> {
        self.agents.get(agent_id)
    }

    /// Get mutable agent by ID
    pub fn get_agent_mut(&mut self, agent_id: &AgentId) -> Option<&mut DynamicAgent> {
        self.agents.get_mut(agent_id)
    }

    /// Start all agents
    /// 
    /// # Errors
    /// 
    /// Returns an error if any agent fails to start.
    pub async fn start_all_agents(&mut self) -> Result<()> {
        for (id, agent) in &mut self.agents {
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
    pub async fn shutdown_all_agents(&mut self) -> Result<()> {
        for (id, agent) in &mut self.agents {
            agent.shutdown().await.map_err(|e| {
                SwarmError::Custom(format!("Failed to shutdown agent {id}: {e:?}"))
            })?;
        }
        Ok(())
    }

    /// Get swarm metrics
    pub fn metrics(&self) -> SwarmMetrics {
        SwarmMetrics {
            total_agents: self.agents.len(),
            active_agents: self
                .agents
                .iter()
                .filter(|(_, agent)| agent.status() == AgentStatus::Running)
                .count(),
            queued_tasks: self.task_queue.len(),
            assigned_tasks: self.task_assignments.len(),
            total_connections: self.topology.connection_count(),
        }
    }
}

/// Swarm metrics
#[derive(Debug, Clone)]
pub struct SwarmMetrics {
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
