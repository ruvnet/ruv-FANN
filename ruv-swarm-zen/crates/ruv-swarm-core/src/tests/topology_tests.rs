//! Simple topology tests that match the actual API

use crate::topology::*;

#[test]
fn test_topology_creation() {
    let topology = Topology::new(TopologyType::Mesh);
    assert_eq!(topology.topology_type, TopologyType::Mesh);
    assert!(topology.connections.is_empty());
    assert!(topology.groups.is_empty());
    assert!(topology.hierarchy.is_none());
}

#[test]
fn test_mesh_topology() {
    let agents: Vec<String> = vec!["agent-1".to_string(), "agent-2".to_string(), "agent-3".to_string()];
    let topology = Topology::mesh(&agents);
    
    // Each agent should be connected to all others in mesh topology
    // Total connections: 3 agents * 2 connections each / 2 (undirected) = 3
    assert_eq!(topology.connection_count(), 3);
    assert!(topology.is_fully_connected());
}

#[test]
fn test_star_topology() {
    let center = "center".to_string();
    let agents: Vec<String> = vec!["agent-1".to_string(), "agent-2".to_string(), "agent-3".to_string(), "agent-4".to_string()];
    
    let topology = Topology::star(&center, &agents);
    
    // Center should be connected to all agents
    let center_neighbors = topology.get_neighbors(&center).unwrap();
    assert_eq!(center_neighbors.len(), 4);
    
    // Each agent should only be connected to center
    for agent in &agents {
        let neighbors = topology.get_neighbors(agent).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&center));
    }
}

#[test]
fn test_pipeline_topology() {
    let agents: Vec<String> = vec!["stage-1".to_string(), "stage-2".to_string(), "stage-3".to_string(), "stage-4".to_string()];
    let topology = Topology::pipeline(&agents);
    
    // First agent connects only forward
    let first_neighbors = topology.get_neighbors(&agents[0]).unwrap();
    assert_eq!(first_neighbors.len(), 1);
    assert!(first_neighbors.contains(&agents[1]));
    
    // Last agent connects only backward
    let last_neighbors = topology.get_neighbors(&agents[3]).unwrap();
    assert_eq!(last_neighbors.len(), 1);
    assert!(last_neighbors.contains(&agents[2]));
}

#[test]
fn test_add_connection() {
    let mut topology = Topology::new(TopologyType::Custom);
    
    let agent1 = "agent-1".to_string();
    let agent2 = "agent-2".to_string();
    
    topology.add_connection(agent1.clone(), agent2.clone());
    
    assert!(topology.are_connected(&agent1, &agent2));
    assert!(topology.are_connected(&agent2, &agent1)); // Undirected
}

#[test]
fn test_remove_connection() {
    let mut topology = Topology::new(TopologyType::Custom);
    
    let agent1 = "agent-1".to_string();
    let agent2 = "agent-2".to_string();
    
    topology.add_connection(agent1.clone(), agent2.clone());
    assert!(topology.are_connected(&agent1, &agent2));
    
    topology.remove_connection(&agent1, &agent2);
    assert!(!topology.are_connected(&agent1, &agent2));
    assert!(!topology.are_connected(&agent2, &agent1));
}

#[test]
fn test_topology_type_default() {
    assert_eq!(TopologyType::default(), TopologyType::Mesh);
}

#[test] 
fn test_connectivity_check() {
    // Empty topology is considered fully connected
    let empty = Topology::new(TopologyType::Custom);
    assert!(empty.is_fully_connected());
}

#[test]
fn test_hierarchy_node_creation() {
    let node = HierarchyNode::new("root".to_string(), 0);
    assert_eq!(node.agent_id, "root");
    assert_eq!(node.level, 0);
    assert!(node.children.is_empty());
}

#[test]
fn test_hierarchy_node_add_child() {
    let mut root = HierarchyNode::new("root".to_string(), 0);
    let child1 = HierarchyNode::new("child1".to_string(), 1);
    let child2 = HierarchyNode::new("child2".to_string(), 1);
    
    root.add_child(child1);
    root.add_child(child2);
    
    assert_eq!(root.children.len(), 2);
    assert_eq!(root.children[0].agent_id, "child1");
    assert_eq!(root.children[1].agent_id, "child2");
}

#[test]
fn test_hierarchy_node_find() {
    let mut root = HierarchyNode::new("root".to_string(), 0);
    let mut child1 = HierarchyNode::new("child1".to_string(), 1);
    let grandchild = HierarchyNode::new("grandchild".to_string(), 2);
    
    child1.add_child(grandchild);
    root.add_child(child1);
    
    // Find at different levels
    assert!(root.find(&"root".to_string()).is_some());
    assert!(root.find(&"child1".to_string()).is_some());
    assert!(root.find(&"grandchild".to_string()).is_some());
    assert!(root.find(&"nonexistent".to_string()).is_none());
}

#[test]
fn test_hierarchy_node_agents_at_level() {
    let mut root = HierarchyNode::new("root".to_string(), 0);
    let mut child1 = HierarchyNode::new("child1".to_string(), 1);
    let mut child2 = HierarchyNode::new("child2".to_string(), 1);
    let grandchild1 = HierarchyNode::new("grandchild1".to_string(), 2);
    let grandchild2 = HierarchyNode::new("grandchild2".to_string(), 2);
    
    child1.add_child(grandchild1);
    child2.add_child(grandchild2);
    root.add_child(child1);
    root.add_child(child2);
    
    let level0 = root.agents_at_level(0);
    assert_eq!(level0.len(), 1);
    assert_eq!(level0[0], &"root".to_string());
    
    let level1 = root.agents_at_level(1);
    assert_eq!(level1.len(), 2);
    
    let level2 = root.agents_at_level(2);
    assert_eq!(level2.len(), 2);
}

#[test]
fn test_topology_groups() {
    let mut topology = Topology::new(TopologyType::Clustered);
    
    topology.add_to_group("compute", "agent-1".to_string());
    topology.add_to_group("compute", "agent-2".to_string());
    topology.add_to_group("storage", "agent-3".to_string());
    
    let compute_group = topology.get_group("compute").unwrap();
    assert_eq!(compute_group.len(), 2);
    assert!(compute_group.contains(&"agent-1".to_string()));
    assert!(compute_group.contains(&"agent-2".to_string()));
    
    let storage_group = topology.get_group("storage").unwrap();
    assert_eq!(storage_group.len(), 1);
    assert!(storage_group.contains(&"agent-3".to_string()));
    
    assert!(topology.get_group("nonexistent").is_none());
}

#[test]
fn test_topology_with_hierarchy() {
    let mut topology = Topology::new(TopologyType::Hierarchical);
    let root = HierarchyNode::new("coordinator".to_string(), 0);
    topology.hierarchy = Some(root);
    
    assert!(topology.hierarchy.is_some());
    assert_eq!(topology.hierarchy.as_ref().unwrap().agent_id, "coordinator");
}

#[test]
fn test_complex_connectivity() {
    let mut topology = Topology::new(TopologyType::Custom);
    
    // Create a more complex network
    let agents: Vec<String> = (0..10).map(|i| format!("agent-{i}")).collect();
    
    // Create some clusters
    for i in 0..3 {
        topology.add_connection(agents[i].clone(), agents[(i + 1) % 3].clone());
    }
    for i in 3..6 {
        topology.add_connection(agents[i].clone(), agents[(i + 1) % 3 + 3].clone());
    }
    
    // Connect clusters
    topology.add_connection(agents[0].clone(), agents[3].clone());
    
    // Add isolated agents
    topology.add_connection(agents[6].clone(), agents[7].clone());
    
    // Agents 8 and 9 are completely isolated
    
    assert!(!topology.is_fully_connected());
    
    // Check specific connections
    assert!(topology.are_connected(&agents[0], &agents[1]));
    assert!(topology.are_connected(&agents[0], &agents[3]));
    assert!(!topology.are_connected(&agents[0], &agents[6]));
}

#[test]
fn test_pipeline_topology_directionality() {
    let agents: Vec<String> = vec!["stage-1".to_string(), "stage-2".to_string(), "stage-3".to_string()];
    let mut topology = Topology::pipeline(&agents);
    
    // Pipeline should maintain directionality
    topology.topology_type = TopologyType::Pipeline;
    
    // In a pipeline, connections should be unidirectional
    topology.remove_connection(&"stage-2".to_string(), &"stage-1".to_string());
    
    // stage-1 should still connect to stage-2
    assert!(topology.are_connected(&"stage-1".to_string(), &"stage-2".to_string()));
    // But stage-2 should not connect back to stage-1 after removal
    assert!(!topology.are_connected(&"stage-2".to_string(), &"stage-1".to_string()));
}

#[test]
fn test_connection_count_accuracy() {
    let mut topology = Topology::new(TopologyType::Custom);
    
    // Add connections
    topology.add_connection("a".to_string(), "b".to_string());
    topology.add_connection("b".to_string(), "c".to_string());
    topology.add_connection("c".to_string(), "a".to_string());
    
    // Should have 3 bidirectional connections
    assert_eq!(topology.connection_count(), 3);
    
    // Add duplicate connection - should not increase count
    topology.add_connection("a".to_string(), "b".to_string());
    assert_eq!(topology.connection_count(), 3);
    
    // Remove a connection
    topology.remove_connection(&"a".to_string(), &"b".to_string());
    assert_eq!(topology.connection_count(), 2);
}

#[test]
fn test_clustered_topology() {
    let mut topology = Topology::new(TopologyType::Clustered);
    
    // Create three clusters
    let cluster1: Vec<String> = vec!["c1-1".to_string(), "c1-2".to_string(), "c1-3".to_string()];
    let cluster2: Vec<String> = vec!["c2-1".to_string(), "c2-2".to_string(), "c2-3".to_string()];
    let cluster3: Vec<String> = vec!["c3-1".to_string(), "c3-2".to_string()];
    
    // Add agents to groups
    for agent in &cluster1 {
        topology.add_to_group("cluster1", agent.clone());
    }
    for agent in &cluster2 {
        topology.add_to_group("cluster2", agent.clone());
    }
    for agent in &cluster3 {
        topology.add_to_group("cluster3", agent.clone());
    }
    
    // Connect agents within clusters
    for i in 0..cluster1.len() {
        for j in i+1..cluster1.len() {
            topology.add_connection(cluster1[i].clone(), cluster1[j].clone());
        }
    }
    
    // Connect cluster leaders
    topology.add_connection(cluster1[0].clone(), cluster2[0].clone());
    topology.add_connection(cluster2[0].clone(), cluster3[0].clone());
    
    // Verify group sizes
    assert_eq!(topology.get_group("cluster1").unwrap().len(), 3);
    assert_eq!(topology.get_group("cluster2").unwrap().len(), 3);
    assert_eq!(topology.get_group("cluster3").unwrap().len(), 2);
    
    // Verify inter-cluster connectivity
    assert!(topology.are_connected(&cluster1[0], &cluster2[0]));
    assert!(!topology.are_connected(&cluster1[1], &cluster2[1]));
}

#[test]
fn test_empty_topology_edge_cases() {
    let topology = Topology::new(TopologyType::Custom);
    
    // Empty topology should handle these gracefully
    assert_eq!(topology.connection_count(), 0);
    assert!(topology.is_fully_connected()); // Empty is considered fully connected
    assert!(topology.get_neighbors(&"nonexistent".to_string()).is_none());
    assert!(!topology.are_connected(&"a".to_string(), &"b".to_string()));
}

#[test]
fn test_single_agent_topology() {
    let mut topology = Topology::new(TopologyType::Mesh);
    let agent = "solo".to_string();
    
    // Add a single agent with no connections
    topology.connections.insert(agent.clone(), std::collections::HashSet::new());
    
    assert_eq!(topology.connection_count(), 0);
    assert!(topology.is_fully_connected()); // Single agent is fully connected to itself
    assert_eq!(topology.get_neighbors(&agent).unwrap().len(), 0);
}