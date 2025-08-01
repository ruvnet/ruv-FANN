//! Swarm topology definitions

use crate::agent::AgentId;

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box, collections::BTreeMap as HashMap, collections::BTreeSet as HashSet, string::String,
    vec, vec::Vec,
};
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};

/// Type of swarm topology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum TopologyType {
    /// All agents can communicate with all other agents
    #[default]
    Mesh,
    /// Agents organized in a tree structure
    Hierarchical,
    /// Agents organized in groups with group leaders
    Clustered,
    /// Agents in a chain, each communicating with neighbors
    Pipeline,
    /// Star topology with a central coordinator
    Star,
    /// Custom topology defined by user
    Custom,
}


/// Swarm topology configuration
#[derive(Debug, Clone)]
pub struct Topology {
    /// The type of topology being used (mesh, star, ring, etc.)
    pub topology_type: TopologyType,
    /// Map of agent IDs to their connected neighbors
    pub connections: HashMap<AgentId, HashSet<AgentId>>,
    /// Named groups of agents for organizational purposes
    pub groups: HashMap<String, Vec<AgentId>>,
    /// Optional hierarchical structure for tree-based topologies
    pub hierarchy: Option<HierarchyNode>,
}

impl Topology {
    /// Create a new topology
    pub fn new(topology_type: TopologyType) -> Self {
        Topology {
            topology_type,
            connections: HashMap::new(),
            groups: HashMap::new(),
            hierarchy: None,
        }
    }

    /// Create a mesh topology where all agents are connected
    pub fn mesh(agents: &[AgentId]) -> Self {
        let mut topology = Topology::new(TopologyType::Mesh);

        for agent in agents {
            let mut connections = HashSet::new();
            for other in agents {
                if agent != other {
                    connections.insert(other.clone());
                }
            }
            topology.connections.insert(agent.clone(), connections);
        }

        topology
    }

    /// Create a star topology with a central coordinator
    pub fn star(center: &str, agents: &[AgentId]) -> Self {
        let mut topology = Topology::new(TopologyType::Star);

        // Center connects to all agents
        let center_connections: HashSet<AgentId> = agents.iter().cloned().collect();
        topology
            .connections
            .insert(center.to_string(), center_connections);

        // All agents connect only to center
        for agent in agents {
            let mut connections = HashSet::new();
            connections.insert(center.to_string());
            topology.connections.insert(agent.clone(), connections);
        }

        topology
    }

    /// Create a pipeline topology
    pub fn pipeline(agents: &[AgentId]) -> Self {
        let mut topology = Topology::new(TopologyType::Pipeline);

        for (i, agent) in agents.iter().enumerate() {
            let mut connections = HashSet::new();

            // Connect to previous agent
            if i > 0 {
                connections.insert(agents[i - 1].clone());
            }

            // Connect to next agent
            if i < agents.len() - 1 {
                connections.insert(agents[i + 1].clone());
            }

            topology.connections.insert(agent.clone(), connections);
        }

        topology
    }

    /// Add a connection between two agents
    pub fn add_connection(&mut self, from: AgentId, to: AgentId) {
        self.connections
            .entry(from.clone())
            .or_default()
            .insert(to.clone());

        // For undirected graphs, add reverse connection
        if self.topology_type != TopologyType::Pipeline {
            self.connections
                .entry(to)
                .or_default()
                .insert(from);
        }
    }

    /// Remove a connection between two agents
    pub fn remove_connection(&mut self, from: &AgentId, to: &AgentId) {
        if let Some(connections) = self.connections.get_mut(from) {
            connections.remove(to);
        }

        // For undirected graphs, remove reverse connection
        if self.topology_type != TopologyType::Pipeline {
            if let Some(connections) = self.connections.get_mut(to) {
                connections.remove(from);
            }
        }
    }

    /// Get all agents connected to a specific agent
    pub fn get_neighbors(&self, agent: &AgentId) -> Option<&HashSet<AgentId>> {
        self.connections.get(agent)
    }

    /// Check if two agents are connected
    pub fn are_connected(&self, from: &AgentId, to: &AgentId) -> bool {
        self.connections
            .get(from)
            .is_some_and(|connections| connections.contains(to))
    }

    /// Add an agent to a group
    pub fn add_to_group(&mut self, group_name: impl Into<String>, agent: AgentId) {
        self.groups
            .entry(group_name.into())
            .or_default()
            .push(agent);
    }

    /// Get all agents in a group
    pub fn get_group(&self, group_name: &str) -> Option<&Vec<AgentId>> {
        self.groups.get(group_name)
    }

    /// Get the total number of connections in the topology
    pub fn connection_count(&self) -> usize {
        self.connections
            .values()
            .map(std::collections::HashSet::len)
            .sum::<usize>()
            / 2 // Divide by 2 for undirected graphs
    }

    /// Check if the topology is fully connected
    pub fn is_fully_connected(&self) -> bool {
        let agent_count = self.connections.len();
        if agent_count == 0 {
            return true;
        }

        // Use BFS to check connectivity
        let Some(start) = self.connections.keys().next() else {
            return true; // Empty graph is considered fully connected
        };
        let mut visited = HashSet::new();
        let mut queue = vec![start.clone()];

        while let Some(current) = queue.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if let Some(neighbors) = self.connections.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        queue.push(neighbor.clone());
                    }
                }
            }
        }

        visited.len() == agent_count
    }
}

/// Hierarchy node for hierarchical topologies
#[derive(Debug, Clone)]
pub struct HierarchyNode {
    /// The unique identifier of the agent at this hierarchy node
    pub agent_id: AgentId,
    /// Child nodes in the hierarchy tree
    pub children: Vec<HierarchyNode>,
    /// The depth level of this node in the hierarchy (0 = root)
    pub level: usize,
}

impl HierarchyNode {
    /// Create a new hierarchy node
    pub fn new(agent_id: AgentId, level: usize) -> Self {
        HierarchyNode {
            agent_id,
            children: Vec::new(),
            level,
        }
    }

    /// Add a child node
    pub fn add_child(&mut self, child: HierarchyNode) {
        self.children.push(child);
    }

    /// Find a node by agent ID
    pub fn find(&self, agent_id: &AgentId) -> Option<&HierarchyNode> {
        if &self.agent_id == agent_id {
            return Some(self);
        }

        for child in &self.children {
            if let Some(found) = child.find(agent_id) {
                return Some(found);
            }
        }

        None
    }

    /// Get all agents at a specific level
    pub fn agents_at_level(&self, target_level: usize) -> Vec<&AgentId> {
        let mut agents = Vec::new();

        if self.level == target_level {
            agents.push(&self.agent_id);
        }

        for child in &self.children {
            agents.extend(child.agents_at_level(target_level));
        }

        agents
    }
}
