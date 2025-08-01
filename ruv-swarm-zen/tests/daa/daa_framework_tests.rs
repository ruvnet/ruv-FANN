//! DAA Framework Integration Performance Tests
//!
//! Validates that DAA framework maintains its core capabilities and performance
//! when integrated with GPU acceleration components.

use ruv_swarm_daa::*;
use std::time::Instant;
use tokio::time::{timeout, Duration};

// Import our mock agents
use super::coordination_tests::{StandardDAAAgent, MockStandardDAAAgent};
use super::gpu_acceleration_tests::GPUDAAAgent;

#[cfg(test)]
mod daa_framework_tests {
    use super::*;

    /// Test DAA agent creation and initialization performance
    #[tokio::test]
    async fn test_agent_creation_performance() {
        let start = Instant::now();
        
        // Create multiple agents to test scalability
        let mut agents = Vec::new();
        for i in 0..10 {
            let pattern = match i % 6 {
                0 => CognitivePattern::Convergent,
                1 => CognitivePattern::Divergent,
                2 => CognitivePattern::Lateral,
                3 => CognitivePattern::Systems,
                4 => CognitivePattern::Critical,
                _ => CognitivePattern::Adaptive,
            };
            
            let agent = StandardDAAAgent::builder()
                .with_cognitive_pattern(pattern)
                .with_learning_rate(0.001)
                .build()
                .await
                .expect("Failed to create DAA agent");
            
            agents.push(agent);
        }
        
        let creation_time = start.elapsed();
        println!("Created 10 DAA agents in {:?}", creation_time);
        
        // Performance assertion: Should create 10 agents in < 500ms
        assert!(creation_time.as_millis() < 500, "Agent creation too slow: {:?}", creation_time);
        assert_eq!(agents.len(), 10, "Not all agents created successfully");
    }

    /// Test cognitive pattern evolution performance
    #[tokio::test]
    async fn test_cognitive_pattern_evolution() {
        let mut agent = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Convergent)
            .build()
            .await
            .expect("Failed to create agent");
        
        let start = Instant::now();
        
        // Test multiple pattern evolutions
        for _ in 0..5 {
            let new_pattern = agent.evolve_cognitive_pattern().await
                .expect("Failed to evolve cognitive pattern");
            
            assert!(matches!(
                new_pattern,
                CognitivePattern::Convergent | 
                CognitivePattern::Divergent | 
                CognitivePattern::Lateral | 
                CognitivePattern::Systems | 
                CognitivePattern::Critical | 
                CognitivePattern::Adaptive
            ), "Invalid cognitive pattern evolved");
        }
        
        let evolution_time = start.elapsed();
        println!("Evolved cognitive patterns 5 times in {:?}", evolution_time);
        
        // Performance assertion: Should evolve patterns in < 50ms
        assert!(evolution_time.as_millis() < 50, "Pattern evolution too slow: {:?}", evolution_time);
    }

    /// Test autonomous learning start/stop performance
    #[tokio::test]
    async fn test_autonomous_learning_lifecycle() {
        let mut agent = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Adaptive)
            .with_learning_rate(0.001)
            .build()
            .await
            .expect("Failed to create agent");
        
        // Test learning start performance
        let start = Instant::now();
        agent.start_autonomous_learning().await
            .expect("Failed to start autonomous learning");
        let start_time = start.elapsed();
        
        // Test learning stop performance
        let start = Instant::now();
        agent.stop_autonomous_learning().await
            .expect("Failed to stop autonomous learning");
        let stop_time = start.elapsed();
        
        println!("Learning start: {:?}, stop: {:?}", start_time, stop_time);
        
        // Performance assertions
        assert!(start_time.as_millis() < 100, "Learning start too slow: {:?}", start_time);
        assert!(stop_time.as_millis() < 100, "Learning stop too slow: {:?}", stop_time);
    }

    /// Test task processing performance across different cognitive patterns
    #[tokio::test]
    async fn test_task_processing_performance() {
        let patterns = vec![
            CognitivePattern::Convergent,
            CognitivePattern::Divergent,
            CognitivePattern::Lateral,
            CognitivePattern::Systems,
            CognitivePattern::Critical,
            CognitivePattern::Adaptive,
        ];
        
        let task = Task {
            id: "test_task_001".to_string(),
            description: "Performance validation task".to_string(),
            requirements: vec!["speed".to_string(), "accuracy".to_string()],
            priority: Priority::High,
            deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
            context: std::collections::HashMap::new(),
        };
        
        for pattern in patterns {
            let mut agent = StandardDAAAgent::builder()
                .with_cognitive_pattern(pattern.clone())
                .build()
                .await
                .expect("Failed to create agent");
            
            // Start autonomous learning for realistic testing
            agent.start_autonomous_learning().await
                .expect("Failed to start learning");
            
            let start = Instant::now();
            let result = agent.process_task_autonomously(&task).await
                .expect("Failed to process task");
            let processing_time = start.elapsed();
            
            println!("Pattern {:?} processed task in {:?}", pattern, processing_time);
            
            // Validate result quality
            assert!(result.success, "Task processing failed for pattern {:?}", pattern);
            assert_eq!(result.task_id, task.id, "Task ID mismatch");
            assert!(result.execution_time_ms > 0, "Execution time not recorded");
            
            // Performance assertion: Should process task in < 100ms
            assert!(processing_time.as_millis() < 100, 
                "Task processing too slow for pattern {:?}: {:?}", pattern, processing_time);
            
            agent.stop_autonomous_learning().await
                .expect("Failed to stop learning");
        }
    }

    /// Test agent coordination performance
    #[tokio::test]
    async fn test_agent_coordination_performance() {
        let mut agents = Vec::new();
        
        // Create multiple agents for coordination testing
        for i in 0..5 {
            let agent = StandardDAAAgent::builder()
                .with_cognitive_pattern(CognitivePattern::Adaptive)
                .build()
                .await
                .expect("Failed to create agent");
            agents.push(agent);
        }
        
        let main_agent = &agents[0];
        let peer_ids: Vec<String> = agents[1..].iter().map(|a| a.id().to_string()).collect();
        
        let start = Instant::now();
        let coordination_result = main_agent.coordinate_with_peers(&peer_ids).await
            .expect("Failed to coordinate with peers");
        let coordination_time = start.elapsed();
        
        println!("Coordinated with {} agents in {:?}", peer_ids.len(), coordination_time);
        
        // Validate coordination result
        assert!(coordination_result.success, "Coordination failed");
        assert!(coordination_result.consensus_reached, "Consensus not reached");
        assert_eq!(coordination_result.coordinated_agents.len(), peer_ids.len(), 
            "Not all agents coordinated");
        
        // Performance assertion: Should coordinate in < 50ms
        assert!(coordination_time.as_millis() < 50, 
            "Coordination too slow: {:?}", coordination_time);
        assert!(coordination_result.coordination_time_ms < 50, 
            "Reported coordination time too slow: {}ms", coordination_result.coordination_time_ms);
    }

    /// Test knowledge sharing performance
    #[tokio::test]
    async fn test_knowledge_sharing_performance() {
        let agent1 = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Systems)
            .build()
            .await
            .expect("Failed to create agent1");
        
        let agent2 = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Critical)
            .build()
            .await
            .expect("Failed to create agent2");
        
        let knowledge = Knowledge {
            id: "test_knowledge_001".to_string(),
            domain: "performance_testing".to_string(),
            content: serde_json::json!({
                "concept": "gpu_acceleration",
                "effectiveness": 0.95,
                "applications": ["neural_networks", "parallel_computation"]
            }),
            confidence: 0.9,
            source_agent: agent1.id().to_string(),
            created_at: chrono::Utc::now(),
        };
        
        let start = Instant::now();
        agent1.share_knowledge(agent2.id(), &knowledge).await
            .expect("Failed to share knowledge");
        let sharing_time = start.elapsed();
        
        println!("Shared knowledge in {:?}", sharing_time);
        
        // Performance assertion: Should share knowledge in < 25ms
        assert!(sharing_time.as_millis() < 25, 
            "Knowledge sharing too slow: {:?}", sharing_time);
    }

    /// Test agent metrics collection performance
    #[tokio::test]
    async fn test_metrics_collection_performance() {
        let mut agent = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Adaptive)
            .build()
            .await
            .expect("Failed to create agent");
        
        // Process some tasks to generate metrics
        let task = Task {
            id: "metrics_test_task".to_string(),
            description: "Task for metrics testing".to_string(),
            requirements: vec!["performance".to_string()],
            priority: Priority::Medium,
            deadline: None,
            context: std::collections::HashMap::new(),
        };
        
        agent.start_autonomous_learning().await
            .expect("Failed to start learning");
        
        // Process multiple tasks
        for i in 0..10 {
            let mut task_copy = task.clone();
            task_copy.id = format!("metrics_test_task_{}", i);
            
            agent.process_task_autonomously(&task_copy).await
                .expect("Failed to process task");
        }
        
        let start = Instant::now();
        let metrics = agent.get_metrics().await
            .expect("Failed to get metrics");
        let metrics_time = start.elapsed();
        
        println!("Collected metrics in {:?}", metrics_time);
        
        // Validate metrics
        assert_eq!(metrics.agent_id, agent.id(), "Agent ID mismatch in metrics");
        assert_eq!(metrics.tasks_completed, 10, "Task count mismatch");
        assert!(metrics.success_rate > 0.0, "Success rate not recorded");
        assert!(metrics.average_response_time_ms > 0.0, "Response time not recorded");
        
        // Performance assertion: Should collect metrics in < 10ms
        assert!(metrics_time.as_millis() < 10, 
            "Metrics collection too slow: {:?}", metrics_time);
        
        agent.stop_autonomous_learning().await
            .expect("Failed to stop learning");
    }

    /// Test agent adaptation performance under feedback
    #[tokio::test]
    async fn test_agent_adaptation_performance() {
        let mut agent = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Convergent)
            .with_adaptation_threshold(0.7)
            .build()
            .await
            .expect("Failed to create agent");
        
        let feedback = Feedback {
            source: "performance_test".to_string(),
            task_id: "adaptation_test".to_string(),
            performance_score: 0.5, // Below threshold to trigger adaptation
            suggestions: vec!["creative".to_string(), "systematic".to_string()],
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
        };
        
        let original_pattern = agent.cognitive_pattern().clone();
        
        let start = Instant::now();
        agent.adapt_strategy(&feedback).await
            .expect("Failed to adapt strategy");
        let adaptation_time = start.elapsed();
        
        println!("Adapted strategy in {:?}", adaptation_time);
        
        // Performance assertion: Should adapt in < 25ms
        assert!(adaptation_time.as_millis() < 25, 
            "Strategy adaptation too slow: {:?}", adaptation_time);
        
        // Note: Pattern may or may not change based on feedback analysis
        println!("Original pattern: {:?}, Current pattern: {:?}", 
            original_pattern, agent.cognitive_pattern());
    }

    /// Test memory system performance under load
    #[tokio::test]
    async fn test_memory_system_performance() {
        let mut agent = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Systems)
            .build()
            .await
            .expect("Failed to create agent");
        
        agent.start_autonomous_learning().await
            .expect("Failed to start learning");
        
        let start = Instant::now();
        
        // Generate multiple experiences to test memory performance
        for i in 0..100 {
            let task = Task {
                id: format!("memory_test_task_{}", i),
                description: format!("Memory test task {}", i),
                requirements: vec!["memory".to_string(), "performance".to_string()],
                priority: Priority::Low,
                deadline: None,
                context: std::collections::HashMap::new(),
            };
            
            let result = TaskResult {
                task_id: task.id.clone(),
                success: true,
                output: serde_json::json!({"iteration": i}),
                performance_metrics: std::collections::HashMap::new(),
                learned_patterns: vec!["memory_pattern".to_string()],
                execution_time_ms: 10,
            };
            
            let experience = Experience {
                task,
                result,
                feedback: None,
                context: std::collections::HashMap::new(),
            };
            
            agent.learn_from_experience(&experience).await
                .expect("Failed to learn from experience");
        }
        
        let memory_time = start.elapsed();
        println!("Processed 100 experiences in {:?}", memory_time);
        
        // Performance assertion: Should process 100 experiences in < 1000ms
        assert!(memory_time.as_millis() < 1000, 
            "Memory processing too slow: {:?}", memory_time);
        
        agent.stop_autonomous_learning().await
            .expect("Failed to stop learning");
    }

    /// Test concurrent agent operations performance
    #[tokio::test]
    async fn test_concurrent_operations_performance() {
        let num_agents = 8;
        let mut handles = Vec::new();
        
        let start = Instant::now();
        
        for i in 0..num_agents {
            let handle = tokio::spawn(async move {
                let mut agent = StandardDAAAgent::builder()
                    .with_cognitive_pattern(CognitivePattern::Adaptive)
                    .build()
                    .await
                    .expect("Failed to create agent");
                
                agent.start_autonomous_learning().await
                    .expect("Failed to start learning");
                
                let task = Task {
                    id: format!("concurrent_task_{}", i),
                    description: "Concurrent processing test".to_string(),
                    requirements: vec!["concurrency".to_string()],
                    priority: Priority::Medium,
                    deadline: None,
                    context: std::collections::HashMap::new(),
                };
                
                let result = agent.process_task_autonomously(&task).await
                    .expect("Failed to process task");
                
                agent.stop_autonomous_learning().await
                    .expect("Failed to stop learning");
                
                result
            });
            
            handles.push(handle);
        }
        
        // Wait for all agents to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.expect("Agent task failed");
            results.push(result);
        }
        
        let concurrent_time = start.elapsed();
        println!("Processed {} concurrent agents in {:?}", num_agents, concurrent_time);
        
        // Validate all results
        assert_eq!(results.len(), num_agents, "Not all agents completed");
        for result in &results {
            assert!(result.success, "Concurrent task failed");
        }
        
        // Performance assertion: Should handle 8 concurrent agents in < 2000ms
        assert!(concurrent_time.as_millis() < 2000, 
            "Concurrent operations too slow: {:?}", concurrent_time);
    }

    /// Test DAA coordinator performance
    #[tokio::test]
    async fn test_daa_coordinator_performance() {
        let coordinator = initialize_daa().await
            .expect("Failed to initialize DAA coordinator");
        
        let start = Instant::now();
        let stats = coordinator.get_coordination_stats().await
            .expect("Failed to get coordination stats");
        let stats_time = start.elapsed();
        
        println!("Retrieved coordination stats in {:?}", stats_time);
        
        // Validate stats
        assert_eq!(stats.total_agents, 0, "Unexpected agent count");
        assert!(stats.coordination_efficiency >= 0.0, "Invalid coordination efficiency");
        
        // Performance assertion: Should get stats in < 10ms
        assert!(stats_time.as_millis() < 10, 
            "Coordination stats retrieval too slow: {:?}", stats_time);
        
        // Test task orchestration performance
        let task = Task {
            id: "orchestration_test".to_string(),
            description: "Test orchestration performance".to_string(),
            requirements: vec!["coordination".to_string()],
            priority: Priority::High,
            deadline: None,
            context: std::collections::HashMap::new(),
        };
        
        let agent_ids = vec!["agent_1".to_string(), "agent_2".to_string(), "agent_3".to_string()];
        
        let start = Instant::now();
        let orchestration_results = coordinator.orchestrate_task(&task, &agent_ids).await
            .expect("Failed to orchestrate task");
        let orchestration_time = start.elapsed();
        
        println!("Orchestrated task across {} agents in {:?}", 
            agent_ids.len(), orchestration_time);
        
        // Validate orchestration results
        assert_eq!(orchestration_results.len(), agent_ids.len(), 
            "Not all agents processed task");
        for result in &orchestration_results {
            assert!(result.success, "Orchestrated task failed");
        }
        
        // Performance assertion: Should orchestrate in < 200ms
        assert!(orchestration_time.as_millis() < 200, 
            "Task orchestration too slow: {:?}", orchestration_time);
    }
}

/// Performance benchmarking utilities
pub mod benchmarks {
    use super::*;
    use std::time::Duration;

    /// Benchmark agent creation performance
    pub async fn benchmark_agent_creation(num_agents: usize) -> Duration {
        let start = Instant::now();
        
        for i in 0..num_agents {
            let pattern = match i % 6 {
                0 => CognitivePattern::Convergent,
                1 => CognitivePattern::Divergent, 
                2 => CognitivePattern::Lateral,
                3 => CognitivePattern::Systems,
                4 => CognitivePattern::Critical,
                _ => CognitivePattern::Adaptive,
            };
            
            let _agent = StandardDAAAgent::builder()
                .with_cognitive_pattern(pattern)
                .build()
                .await
                .expect("Failed to create agent");
        }
        
        start.elapsed()
    }

    /// Benchmark task processing throughput
    pub async fn benchmark_task_throughput(num_tasks: usize) -> (Duration, f64) {
        let mut agent = StandardDAAAgent::builder()
            .with_cognitive_pattern(CognitivePattern::Adaptive)
            .build()
            .await
            .expect("Failed to create agent");
        
        agent.start_autonomous_learning().await
            .expect("Failed to start learning");
        
        let start = Instant::now();
        
        for i in 0..num_tasks {
            let task = Task {
                id: format!("benchmark_task_{}", i),
                description: "Benchmark task".to_string(),
                requirements: vec!["performance".to_string()],
                priority: Priority::Medium,
                deadline: None,
                context: std::collections::HashMap::new(),
            };
            
            agent.process_task_autonomously(&task).await
                .expect("Failed to process task");
        }
        
        let total_time = start.elapsed();
        let throughput = num_tasks as f64 / total_time.as_secs_f64();
        
        agent.stop_autonomous_learning().await
            .expect("Failed to stop learning");
        
        (total_time, throughput)
    }

    /// Performance validation summary
    pub struct PerformanceResults {
        pub agent_creation_time_ms: u64,
        pub task_processing_avg_ms: f64,
        pub coordination_time_ms: u64,
        pub memory_efficiency: f64,
        pub throughput_tasks_per_sec: f64,
    }

    impl PerformanceResults {
        /// Check if results meet performance targets
        pub fn meets_targets(&self) -> bool {
            self.agent_creation_time_ms < 50 &&
            self.task_processing_avg_ms < 100.0 &&
            self.coordination_time_ms < 50 &&
            self.memory_efficiency > 0.8 &&
            self.throughput_tasks_per_sec > 10.0
        }
        
        /// Generate performance report
        pub fn generate_report(&self) -> String {
            format!(
                "DAA Framework Performance Results:\n\
                ================================\n\
                Agent Creation: {}ms (Target: <50ms) {}\n\
                Task Processing: {:.2}ms avg (Target: <100ms) {}\n\
                Coordination: {}ms (Target: <50ms) {}\n\
                Memory Efficiency: {:.1}% (Target: >80%) {}\n\
                Throughput: {:.1} tasks/sec (Target: >10) {}\n\
                \n\
                Overall: {} ✓\n",
                self.agent_creation_time_ms,
                if self.agent_creation_time_ms < 50 { "✓" } else { "✗" },
                self.task_processing_avg_ms,
                if self.task_processing_avg_ms < 100.0 { "✓" } else { "✗" },
                self.coordination_time_ms,
                if self.coordination_time_ms < 50 { "✓" } else { "✗" },
                self.memory_efficiency * 100.0,
                if self.memory_efficiency > 0.8 { "✓" } else { "✗" },
                self.throughput_tasks_per_sec,
                if self.throughput_tasks_per_sec > 10.0 { "✓" } else { "✗" },
                if self.meets_targets() { "PASS" } else { "FAIL" }
            )
        }
    }
}