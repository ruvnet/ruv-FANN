#[cfg(test)]
mod swarm_orchestration_tests {
    use ruv_swarm_wasm::{
        WasmSwarmOrchestrator, CognitiveDiversityEngine, 
        NeuralSwarmCoordinator, CognitiveNeuralArchitectures
    };
    use serde_json::json;

    #[test]
    fn test_swarm_creation_with_topologies() {
        let mut orchestrator = WasmSwarmOrchestrator::new();
        
        // Test all topology types
        let topologies = vec!["mesh", "star", "hierarchical", "ring"];
        
        for topology in topologies {
            let config = json!({
                "name": format!("Test {} Swarm", topology),
                "topology_type": topology,
                "max_agents": 5,
                "enable_cognitive_diversity": true
            });
            
            let result = orchestrator.create_swarm(config.into()).unwrap();
            let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();
            
            assert_eq!(swarm_data["topology"], topology);
            assert!(swarm_data["swarm_id"].as_str().unwrap().starts_with("swarm_"));
        }
    }

    #[test]
    fn test_cognitive_diversity_engine() {
        let engine = CognitiveDiversityEngine::new();
        
        // Test pattern retrieval
        let patterns = engine.get_cognitive_patterns();
        let patterns_data: serde_json::Value = serde_wasm_bindgen::from_value(patterns).unwrap();
        
        assert!(patterns_data.as_object().unwrap().contains_key("convergent"));
        assert!(patterns_data.as_object().unwrap().contains_key("divergent"));
        assert!(patterns_data.as_object().unwrap().contains_key("systems"));
        assert!(patterns_data.as_object().unwrap().contains_key("critical"));
        assert!(patterns_data.as_object().unwrap().contains_key("lateral"));
    }

    #[test]
    fn test_agent_spawning_with_cognitive_patterns() {
        let mut orchestrator = WasmSwarmOrchestrator::new();
        
        // Create swarm
        let swarm_config = json!({
            "name": "Test Swarm",
            "topology_type": "mesh",
            "max_agents": 10
        });
        
        let swarm_result = orchestrator.create_swarm(swarm_config.into()).unwrap();
        let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result).unwrap();
        let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
        
        // Test spawning different agent types
        let agent_types = vec![
            ("researcher", "divergent"),
            ("coder", "convergent"),
            ("analyst", "critical"),
            ("optimizer", "systems"),
            ("coordinator", "systems")
        ];
        
        for (agent_type, expected_pattern) in agent_types {
            let agent_config = json!({
                "agent_type": agent_type,
                "name": format!("test-{}", agent_type)
            });
            
            let agent_result = orchestrator.spawn_agent(swarm_id, agent_config.into()).unwrap();
            let agent_data: serde_json::Value = serde_wasm_bindgen::from_value(agent_result).unwrap();
            
            assert_eq!(agent_data["type"], agent_type);
            assert_eq!(agent_data["cognitive_pattern"], expected_pattern);
            assert!(agent_data["agent_id"].as_str().unwrap().starts_with("agent_"));
        }
    }

    #[test]
    fn test_neural_swarm_coordination() {
        let coordinator = NeuralSwarmCoordinator::new("mesh");
        
        // Test distributed training configuration
        let training_config = json!({
            "training_mode": "DataParallel",
            "agent_ids": ["agent_1", "agent_2", "agent_3"],
            "dataset_config": {
                "dataset_size": 1000,
                "feature_dim": 32,
                "num_classes": 5
            },
            "optimization_config": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "synchronization_interval": 50
        });
        
        let training_result = coordinator.coordinate_neural_training(training_config.into()).unwrap();
        let training_data: serde_json::Value = serde_wasm_bindgen::from_value(training_result).unwrap();
        
        assert!(training_data["final_loss"].as_f64().unwrap() < 1.0);
        assert!(training_data["convergence_achieved"].as_bool().unwrap());
    }

    #[test]
    fn test_knowledge_synchronization() {
        let mut coordinator = NeuralSwarmCoordinator::new("star");
        
        let sync_request = json!({
            "sync_type": "Knowledge",
            "participating_agents": ["agent_1", "agent_2"],
            "knowledge_domains": ["patterns", "models"],
            "sync_depth": {
                "layers": [1, 2],
                "percentage": 0.5
            }
        });
        
        let sync_result = coordinator.synchronize_agent_knowledge(sync_request.into()).unwrap();
        let sync_data: serde_json::Value = serde_wasm_bindgen::from_value(sync_result).unwrap();
        
        assert_eq!(sync_data["sync_type"], "star");
        assert!(sync_data["success"].as_bool().unwrap());
        assert!(sync_data["sync_time_ms"].as_f64().unwrap() < 100.0); // Meeting <100ms requirement
    }

    #[test]
    fn test_cognitive_neural_architectures() {
        let architectures = CognitiveNeuralArchitectures::new();
        
        // Test retrieving different architectures
        let convergent = architectures.get_convergent_architecture();
        let convergent_data: serde_json::Value = serde_wasm_bindgen::from_value(convergent).unwrap();
        
        assert!(convergent_data["encoder"]["layers"].as_array().unwrap().len() == 3);
        assert!(convergent_data["processor"]["residual_connections"].as_bool().unwrap());
        
        let divergent = architectures.get_divergent_architecture();
        let divergent_data: serde_json::Value = serde_wasm_bindgen::from_value(divergent).unwrap();
        
        assert!(divergent_data["parallel_paths"].as_array().unwrap().len() == 3);
        assert_eq!(divergent_data["fusion_mechanism"]["fusion_type"], "attention_weighted");
    }

    #[test]
    fn test_swarm_monitoring_and_metrics() {
        let mut orchestrator = WasmSwarmOrchestrator::new();
        
        // Create and populate swarm
        let swarm_config = json!({
            "name": "Metrics Test Swarm",
            "topology_type": "hierarchical",
            "max_agents": 5
        });
        
        let swarm_result = orchestrator.create_swarm(swarm_config.into()).unwrap();
        let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result).unwrap();
        let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
        
        // Spawn some agents
        for i in 0..3 {
            let agent_config = json!({
                "agent_type": "researcher",
                "name": format!("agent-{}", i)
            });
            orchestrator.spawn_agent(swarm_id, agent_config.into()).unwrap();
        }
        
        // Monitor swarm
        let monitoring_result = orchestrator.monitor_swarm(swarm_id, 1000.0).unwrap();
        let monitoring_data: serde_json::Value = serde_wasm_bindgen::from_value(monitoring_result).unwrap();
        
        assert!(monitoring_data["monitoring_session"]["duration_ms"].as_f64().unwrap() == 1000.0);
        assert!(monitoring_data["agent_activity"].as_array().unwrap().len() == 3);
        
        // Get detailed status
        let status_result = orchestrator.get_swarm_status(swarm_id, Some(true)).unwrap();
        let status_data: serde_json::Value = serde_wasm_bindgen::from_value(status_result).unwrap();
        
        assert_eq!(status_data["agents"]["total"], 3);
        assert!(status_data.get("detailed_agents").is_some());
        assert!(status_data.get("topology_details").is_some());
    }

    #[test]
    fn test_task_orchestration_with_distribution() {
        let mut orchestrator = WasmSwarmOrchestrator::new();
        
        // Setup swarm with agents
        let swarm_config = json!({
            "name": "Task Test Swarm",
            "topology_type": "ring",
            "max_agents": 10
        });
        
        let swarm_result = orchestrator.create_swarm(swarm_config.into()).unwrap();
        let swarm_data: serde_json::Value = serde_wasm_bindgen::from_value(swarm_result).unwrap();
        let swarm_id = swarm_data["swarm_id"].as_str().unwrap();
        
        // Spawn agents with different capabilities
        let agents = vec![
            ("coder", vec!["code_generation", "optimization"]),
            ("analyst", vec!["data_analysis", "statistical_analysis"]),
            ("researcher", vec!["data_analysis", "pattern_recognition"])
        ];
        
        for (agent_type, _caps) in agents {
            let agent_config = json!({
                "agent_type": agent_type
            });
            orchestrator.spawn_agent(swarm_id, agent_config.into()).unwrap();
        }
        
        // Orchestrate a task requiring specific capabilities
        let task_config = json!({
            "description": "Analyze dataset and generate optimized code",
            "priority": "high",
            "required_capabilities": ["data_analysis", "code_generation"],
            "max_agents": 2,
            "estimated_duration_ms": 5000.0
        });
        
        let task_result = orchestrator.orchestrate_task(swarm_id, task_config.into()).unwrap();
        let task_data: serde_json::Value = serde_wasm_bindgen::from_value(task_result).unwrap();
        
        assert_eq!(task_data["status"], "orchestrated");
        assert!(task_data["assigned_agents"].as_array().unwrap().len() <= 2);
        assert!(task_data["distribution_plan"].is_object());
    }

    #[test]
    fn test_collective_intelligence_emergence() {
        let mut coordinator = NeuralSwarmCoordinator::new("mesh");
        
        let emergence_config = json!({
            "enable_self_organization": true,
            "enable_collective_learning": true,
            "pattern_formation_rules": ["consensus", "specialization"],
            "adaptation_threshold": 0.7
        });
        
        let emergence_result = coordinator.enable_neural_emergence(emergence_config.into()).unwrap();
        let emergence_data: serde_json::Value = serde_wasm_bindgen::from_value(emergence_result).unwrap();
        
        assert!(emergence_data["detected_patterns"].as_array().unwrap().len() > 0);
        assert!(emergence_data["collective_performance_gain"].as_f64().unwrap() > 0.0);
    }
}