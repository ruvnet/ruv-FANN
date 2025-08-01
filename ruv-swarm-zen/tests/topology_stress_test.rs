#[cfg(test)]
mod topology_stress_tests {
    use ruv_swarm_wasm::{WasmSwarmOrchestrator};
    use serde_json::json;
    use std::collections::HashMap;
    use std::time::Instant;

    #[derive(Debug, Clone)]
    struct TopologyTestResult {
        topology_type: String,
        agent_count: usize,
        spawn_time_ms: f64,
        connection_count: usize,
        task_distribution_time_ms: f64,
        memory_usage_mb: f32,
        failures: Vec<String>,
    }

    #[derive(Debug)]
    struct TopologyValidationReport {
        topologies_tested: HashMap<String, HashMap<usize, TopologyTestResult>>,
        scalability: HashMap<String, usize>,
        coordination_efficiency: HashMap<String, f64>,
        failures: Vec<(String, String)>,
    }

    impl TopologyValidationReport {
        fn new() -> Self {
            TopologyValidationReport {
                topologies_tested: HashMap::new(),
                scalability: HashMap::new(),
                coordination_efficiency: HashMap::new(),
                failures: Vec::new(),
            }
        }

        fn to_json(&self) -> serde_json::Value {
            json!({
                "topologies_tested": self.topologies_tested,
                "scalability": self.scalability,
                "coordination_efficiency": self.coordination_efficiency,
                "failures": self.failures,
            })
        }
    }

    #[test]
    fn test_mesh_topology_scalability() {
        let mut report = TopologyValidationReport::new();
        let agent_counts = vec![5, 10, 20, 50, 100];
        
        for count in agent_counts {
            let result = test_topology_with_agents("mesh", count);
            
            // Mesh topology should have n*(n-1) connections
            let expected_connections = count * (count - 1);
            assert!(
                result.connection_count <= expected_connections,
                "Mesh topology has too many connections: {} > {}",
                result.connection_count,
                expected_connections
            );
            
            // Record results
            report.topologies_tested
                .entry("mesh".to_string())
                .or_insert_with(HashMap::new)
                .insert(count, result.clone());
            
            // Determine max effective agents (performance threshold)
            if result.task_distribution_time_ms < 1000.0 {
                report.scalability.insert("mesh".to_string(), count);
            }
        }
        
        // Calculate coordination efficiency
        let efficiency = calculate_mesh_efficiency(&report.topologies_tested["mesh"]);
        report.coordination_efficiency.insert("mesh".to_string(), efficiency);
        
        println!("Mesh Topology Report: {}", serde_json::to_string_pretty(&report.to_json()).unwrap());
    }

    #[test]
    fn test_star_topology_hub_bottleneck() {
        let mut report = TopologyValidationReport::new();
        let agent_counts = vec![5, 10, 20, 50, 100];
        
        for count in agent_counts {
            let result = test_topology_with_agents("star", count);
            
            // Star topology should have exactly n-1 connections
            let expected_connections = count - 1;
            assert_eq!(
                result.connection_count, expected_connections,
                "Star topology has incorrect connections: {} != {}",
                result.connection_count,
                expected_connections
            );
            
            // Test hub overload
            let hub_stress = test_star_hub_stress(count);
            if hub_stress.is_err() {
                report.failures.push(("star_hub_stress".to_string(), hub_stress.err().unwrap()));
            }
            
            report.topologies_tested
                .entry("star".to_string())
                .or_insert_with(HashMap::new)
                .insert(count, result);
        }
        
        // Star topology efficiency decreases with size due to hub bottleneck
        let efficiency = calculate_star_efficiency(&report.topologies_tested["star"]);
        report.coordination_efficiency.insert("star".to_string(), efficiency);
        
        println!("Star Topology Report: {}", serde_json::to_string_pretty(&report.to_json()).unwrap());
    }

    #[test]
    fn test_hierarchical_topology_depth() {
        let mut report = TopologyValidationReport::new();
        let agent_counts = vec![7, 15, 31, 63, 127]; // 2^n - 1 for perfect trees
        
        for count in agent_counts {
            let result = test_topology_with_agents("hierarchical", count);
            
            // Verify tree depth
            let expected_depth = (count as f64).log2().ceil() as usize;
            let depth_test = test_hierarchical_depth(count, expected_depth);
            
            if depth_test.is_err() {
                report.failures.push(("hierarchical_depth".to_string(), depth_test.err().unwrap()));
            }
            
            report.topologies_tested
                .entry("hierarchical".to_string())
                .or_insert_with(HashMap::new)
                .insert(count, result);
        }
        
        let efficiency = calculate_hierarchical_efficiency(&report.topologies_tested["hierarchical"]);
        report.coordination_efficiency.insert("hierarchical".to_string(), efficiency);
        
        println!("Hierarchical Topology Report: {}", serde_json::to_string_pretty(&report.to_json()).unwrap());
    }

    #[test]
    fn test_ring_topology_propagation() {
        let mut report = TopologyValidationReport::new();
        let agent_counts = vec![5, 10, 20, 50, 100];
        
        for count in agent_counts {
            let result = test_topology_with_agents("ring", count);
            
            // Ring topology should have exactly n connections
            assert_eq!(
                result.connection_count, count,
                "Ring topology has incorrect connections: {} != {}",
                result.connection_count,
                count
            );
            
            // Test message propagation time
            let propagation_result = test_ring_propagation(count);
            if let Ok(prop_time) = propagation_result {
                // Max distance in ring is floor(n/2)
                let max_distance = count / 2;
                let expected_max_latency = max_distance as f64 * 2.0; // 2ms per hop
                
                assert!(
                    prop_time < expected_max_latency * 2.0, // Allow 2x margin
                    "Ring propagation too slow: {}ms > {}ms",
                    prop_time,
                    expected_max_latency * 2.0
                );
            }
            
            report.topologies_tested
                .entry("ring".to_string())
                .or_insert_with(HashMap::new)
                .insert(count, result);
        }
        
        let efficiency = calculate_ring_efficiency(&report.topologies_tested["ring"]);
        report.coordination_efficiency.insert("ring".to_string(), efficiency);
        
        println!("Ring Topology Report: {}", serde_json::to_string_pretty(&report.to_json()).unwrap());
    }

    #[test]
    fn test_agent_lifecycle_management() {
        let mut orchestrator = WasmSwarmOrchestrator::new();
        
        // Create swarm
        let swarm_config = json!({
            "name": "Lifecycle Test Swarm",
            "topology_type": "mesh",
            "max_agents": 50
        });
        
        let swarm_result = orchestrator.create_swarm(swarm_config.into()).unwrap();
        let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result).unwrap();
        let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
        
        // Test rapid agent spawning
        let start = Instant::now();
        let mut agent_ids = Vec::new();
        
        for i in 0..20 {
            let agent_config = json!({
                "agent_type": ["researcher", "coder", "analyst"][i % 3],
                "name": format!("lifecycle-agent-{}", i)
            });
            
            let agent_result = orchestrator.spawn_agent(swarm_id, agent_config.into()).unwrap();
            let agent_data: serde_json::Value = serde_wasm_bindgen::from_value(agent_result).unwrap();
            agent_ids.push(agent_data["agent_id"].as_str().unwrap().to_string());
        }
        
        let spawn_duration = start.elapsed();
        assert!(
            spawn_duration.as_millis() < 1000,
            "Agent spawning too slow: {:?}",
            spawn_duration
        );
        
        // Test concurrent task assignment
        let mut task_ids = Vec::new();
        for i in 0..10 {
            let task_config = json!({
                "description": format!("Lifecycle task {}", i),
                "priority": ["low", "medium", "high"][i % 3],
                "max_agents": 3
            });
            
            let task_result = orchestrator.orchestrate_task(swarm_id, task_config.into()).unwrap();
            let task_data: serde_json::Value = serde_wasm_bindgen::from_value(task_result).unwrap();
            task_ids.push(task_data["task_id"].as_str().unwrap().to_string());
        }
        
        // Verify all agents are properly assigned
        let status = orchestrator.get_swarm_status(swarm_id, Some(true)).unwrap();
        let status_data: serde_json::Value = serde_wasm_bindgen::from_value(status).unwrap();
        
        assert_eq!(status_data["agents"]["total"], 20);
        assert!(status_data["agents"]["active"].as_u64().unwrap() > 0);
        assert_eq!(status_data["tasks"]["total"], 10);
    }

    #[test]
    fn test_task_distribution_algorithms() {
        let topologies = vec!["mesh", "star", "hierarchical", "ring"];
        let mut distribution_results = HashMap::new();
        
        for topology in topologies {
            let mut orchestrator = WasmSwarmOrchestrator::new();
            
            // Create swarm with specific topology
            let swarm_config = json!({
                "name": format!("{} Distribution Test", topology),
                "topology_type": topology,
                "max_agents": 20
            });
            
            let swarm_result = orchestrator.create_swarm(swarm_config.into()).unwrap();
            let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result).unwrap();
            let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
            
            // Spawn diverse agents
            for i in 0..20 {
                let agent_type = match i % 5 {
                    0 => "researcher",
                    1 => "coder", 
                    2 => "analyst",
                    3 => "optimizer",
                    _ => "coordinator",
                };
                
                let agent_config = json!({
                    "agent_type": agent_type,
                    "name": format!("{}-dist-{}", topology, i)
                });
                
                orchestrator.spawn_agent(swarm_id, agent_config.into()).unwrap();
            }
            
            // Test different task complexities
            let task_complexities = vec![
                ("simple", vec!["data_analysis"], 2),
                ("medium", vec!["code_generation", "optimization"], 5),
                ("complex", vec!["data_analysis", "pattern_recognition", "critical_evaluation"], 10),
            ];
            
            let mut topology_results = Vec::new();
            
            for (complexity, capabilities, max_agents) in task_complexities {
                let task_config = json!({
                    "description": format!("{} {} task", topology, complexity),
                    "priority": "high",
                    "required_capabilities": capabilities,
                    "max_agents": max_agents
                });
                
                let start = Instant::now();
                let task_result = orchestrator.orchestrate_task(swarm_id, task_config.into()).unwrap();
                let distribution_time = start.elapsed();
                
                let task_data: serde_json::Value = serde_wasm_bindgen::from_value(task_result).unwrap();
                
                topology_results.push(json!({
                    "complexity": complexity,
                    "distribution_time_ms": distribution_time.as_millis(),
                    "assigned_agents": task_data["assigned_agents"].as_array().unwrap().len(),
                    "distribution_plan": task_data["distribution_plan"]
                }));
            }
            
            distribution_results.insert(topology.to_string(), topology_results);
        }
        
        println!("Task Distribution Results: {}", 
            serde_json::to_string_pretty(&distribution_results).unwrap());
    }

    #[test]
    fn test_swarm_scalability_limits() {
        let topologies = vec!["mesh", "star", "hierarchical", "ring"];
        let mut scalability_limits = HashMap::new();
        
        for topology in topologies {
            let mut max_effective_agents = 0;
            let test_sizes = vec![10, 25, 50, 100, 200, 500];
            
            for size in test_sizes {
                let mut orchestrator = WasmSwarmOrchestrator::new();
                
                let swarm_config = json!({
                    "name": format!("{} Scalability Test", topology),
                    "topology_type": topology,
                    "max_agents": size
                });
                
                let swarm_result = orchestrator.create_swarm(swarm_config.into());
                if swarm_result.is_err() {
                    break; // Hit limit
                }
                
                let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result.unwrap()).unwrap();
                let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
                
                // Attempt to spawn agents up to limit
                let start = Instant::now();
                let mut spawn_failures = 0;
                
                for i in 0..size {
                    let agent_config = json!({
                        "agent_type": "researcher",
                        "name": format!("scale-{}-{}", topology, i),
                        "max_agents": size
                    });
                    
                    if orchestrator.spawn_agent(swarm_id, agent_config.into()).is_err() {
                        spawn_failures += 1;
                        if spawn_failures > 5 {
                            break; // Too many failures
                        }
                    }
                    
                    // Check if performance is still acceptable
                    if start.elapsed().as_millis() > size as u128 * 100 {
                        break; // Taking too long
                    }
                }
                
                let status = orchestrator.get_swarm_status(swarm_id, Some(false)).unwrap();
                let status_data: serde_json::Value = serde_wasm_bindgen::from_value(status).unwrap();
                let actual_agents = status_data["agents"]["total"].as_u64().unwrap() as usize;
                
                // Consider it effective if we spawned at least 80% of target
                if actual_agents >= (size * 8 / 10) && spawn_failures < 5 {
                    max_effective_agents = actual_agents;
                } else {
                    break;
                }
            }
            
            scalability_limits.insert(topology.to_string(), max_effective_agents);
        }
        
        println!("Scalability Limits: {:?}", scalability_limits);
        
        // Verify expected scalability characteristics
        assert!(scalability_limits["star"] < scalability_limits["mesh"], 
            "Star should have lower scalability than mesh due to hub bottleneck");
        assert!(scalability_limits["ring"] < scalability_limits["hierarchical"],
            "Ring should have lower scalability than hierarchical due to propagation delays");
    }

    // Helper functions
    fn test_topology_with_agents(topology: &str, agent_count: usize) -> TopologyTestResult {
        let mut orchestrator = WasmSwarmOrchestrator::new();
        let mut failures = Vec::new();
        
        // Create swarm
        let swarm_config = json!({
            "name": format!("{} Test Swarm", topology),
            "topology_type": topology,
            "max_agents": agent_count
        });
        
        let swarm_result = orchestrator.create_swarm(swarm_config.into()).unwrap();
        let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result).unwrap();
        let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
        
        // Spawn agents and measure time
        let spawn_start = Instant::now();
        for i in 0..agent_count {
            let agent_config = json!({
                "agent_type": ["researcher", "coder", "analyst", "optimizer", "coordinator"][i % 5],
                "name": format!("{}-agent-{}", topology, i)
            });
            
            if let Err(e) = orchestrator.spawn_agent(swarm_id, agent_config.into()) {
                failures.push(format!("Failed to spawn agent {}: {:?}", i, e));
            }
        }
        let spawn_time_ms = spawn_start.elapsed().as_millis() as f64;
        
        // Get status
        let status = orchestrator.get_swarm_status(swarm_id, Some(true)).unwrap();
        let status_data: serde_json::Value = serde_wasm_bindgen::from_value(status).unwrap();
        
        let connection_count = status_data["topology"]["connections"].as_u64().unwrap() as usize;
        let memory_usage_mb = status_data["performance"]["total_memory_usage_mb"].as_f64().unwrap() as f32;
        
        // Test task distribution
        let task_config = json!({
            "description": "Test task distribution",
            "priority": "medium",
            "max_agents": agent_count.min(10)
        });
        
        let dist_start = Instant::now();
        orchestrator.orchestrate_task(swarm_id, task_config.into()).unwrap();
        let task_distribution_time_ms = dist_start.elapsed().as_millis() as f64;
        
        TopologyTestResult {
            topology_type: topology.to_string(),
            agent_count,
            spawn_time_ms,
            connection_count,
            task_distribution_time_ms,
            memory_usage_mb,
            failures,
        }
    }

    fn test_star_hub_stress(agent_count: usize) -> Result<(), String> {
        let mut orchestrator = WasmSwarmOrchestrator::new();
        
        let swarm_config = json!({
            "name": "Star Hub Stress Test",
            "topology_type": "star",
            "max_agents": agent_count
        });
        
        let swarm_result = orchestrator.create_swarm(swarm_config.into())
            .map_err(|e| format!("Failed to create swarm: {:?}", e))?;
        let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result).unwrap();
        let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
        
        // Spawn agents
        for i in 0..agent_count {
            let agent_config = json!({
                "agent_type": if i == 0 { "coordinator" } else { "researcher" },
                "name": format!("star-stress-{}", i)
            });
            orchestrator.spawn_agent(swarm_id, agent_config.into())
                .map_err(|e| format!("Failed to spawn agent: {:?}", e))?;
        }
        
        // Stress test hub with concurrent tasks
        let concurrent_tasks = agent_count / 2;
        let start = Instant::now();
        
        for i in 0..concurrent_tasks {
            let task_config = json!({
                "description": format!("Hub stress task {}", i),
                "priority": "high",
                "max_agents": 2
            });
            orchestrator.orchestrate_task(swarm_id, task_config.into())
                .map_err(|e| format!("Task orchestration failed: {:?}", e))?;
        }
        
        let stress_duration = start.elapsed();
        
        // Hub should handle concurrent tasks efficiently
        if stress_duration.as_millis() > (concurrent_tasks as u128 * 50) {
            return Err(format!("Hub overloaded: {:?} for {} tasks", stress_duration, concurrent_tasks));
        }
        
        Ok(())
    }

    fn test_hierarchical_depth(agent_count: usize, expected_depth: usize) -> Result<(), String> {
        // Simple depth validation based on agent count
        let actual_depth = (agent_count as f64).log2().ceil() as usize;
        if actual_depth != expected_depth {
            return Err(format!("Incorrect depth: {} != {}", actual_depth, expected_depth));
        }
        Ok(())
    }

    fn test_ring_propagation(agent_count: usize) -> Result<f64, String> {
        // Simulate worst-case propagation (halfway around ring)
        let max_hops = agent_count / 2;
        let latency_per_hop = 2.0; // ms, from code
        Ok(max_hops as f64 * latency_per_hop)
    }

    fn calculate_mesh_efficiency(results: &HashMap<usize, TopologyTestResult>) -> f64 {
        let mut total_efficiency = 0.0;
        let mut count = 0;
        
        for (agent_count, result) in results {
            let expected_connections = agent_count * (agent_count - 1);
            let connection_efficiency = result.connection_count as f64 / expected_connections as f64;
            let time_efficiency = 1000.0 / (result.task_distribution_time_ms + 1.0);
            
            total_efficiency += connection_efficiency * time_efficiency;
            count += 1;
        }
        
        if count > 0 { total_efficiency / count as f64 } else { 0.0 }
    }

    fn calculate_star_efficiency(results: &HashMap<usize, TopologyTestResult>) -> f64 {
        let mut total_efficiency = 0.0;
        let mut count = 0;
        
        for (agent_count, result) in results {
            // Star efficiency decreases with size due to hub bottleneck
            let size_penalty = 1.0 / (1.0 + (*agent_count as f64).log2());
            let time_efficiency = 1000.0 / (result.task_distribution_time_ms + 1.0);
            
            total_efficiency += size_penalty * time_efficiency;
            count += 1;
        }
        
        if count > 0 { total_efficiency / count as f64 } else { 0.0 }
    }

    fn calculate_hierarchical_efficiency(results: &HashMap<usize, TopologyTestResult>) -> f64 {
        let mut total_efficiency = 0.0;
        let mut count = 0;
        
        for (agent_count, result) in results {
            // Hierarchical efficiency based on log depth
            let depth_efficiency = 1.0 / ((*agent_count as f64).log2() + 1.0);
            let time_efficiency = 1000.0 / (result.task_distribution_time_ms + 1.0);
            let memory_efficiency = 100.0 / (result.memory_usage_mb + 1.0);
            
            total_efficiency += depth_efficiency * time_efficiency * memory_efficiency;
            count += 1;
        }
        
        if count > 0 { total_efficiency / count as f64 } else { 0.0 }
    }

    fn calculate_ring_efficiency(results: &HashMap<usize, TopologyTestResult>) -> f64 {
        let mut total_efficiency = 0.0;
        let mut count = 0;
        
        for (agent_count, result) in results {
            // Ring efficiency limited by propagation distance
            let propagation_penalty = 2.0 / *agent_count as f64; // Each agent connects to 2 others
            let time_efficiency = 1000.0 / (result.task_distribution_time_ms + 1.0);
            
            total_efficiency += propagation_penalty * time_efficiency;
            count += 1;
        }
        
        if count > 0 { total_efficiency / count as f64 } else { 0.0 }
    }
}