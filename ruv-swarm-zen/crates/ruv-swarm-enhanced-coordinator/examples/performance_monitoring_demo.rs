//! Performance Monitoring and Optimization Demo
//! 
//! This example demonstrates the comprehensive performance monitoring,
//! load balancing, and resource optimization capabilities of ruv-swarm.

use std::time::{Duration, Instant};
use tokio::time::sleep;

use ruv_swarm_core::{
    // Core swarm types
    Swarm, SwarmConfig, Task, TaskId, Priority, Agent,
    // Performance monitoring
    SwarmMetrics as PerformanceSwarmMetrics, LoadBalancer, LoadBalancerConfig,
    ResourceOptimizer, PerformanceDashboard, DashboardConfig,
    // Agent types
    AgentId, AgentMetadata, AgentStatus,
};

/// Demo agent that simulates various workload patterns
#[derive(Debug)]
struct DemoAgent {
    id: String,
    capabilities: Vec<String>,
    processing_speed: f64, // Tasks per second
    error_rate: f64,       // Probability of errors
    current_load: usize,
}

impl DemoAgent {
    fn new(id: String, capabilities: Vec<String>, processing_speed: f64, error_rate: f64) -> Self {
        Self {
            id,
            capabilities,
            processing_speed,
            error_rate,
            current_load: 0,
        }
    }

    fn simulate_processing_time(&self, task_complexity: f64) -> Duration {
        let base_time = 1.0 / self.processing_speed;
        let complexity_factor = 1.0 + task_complexity;
        let load_factor = 1.0 + (self.current_load as f64 * 0.1);
        
        Duration::from_secs_f64(base_time * complexity_factor * load_factor)
    }
}

#[async_trait::async_trait]
impl Agent for DemoAgent {
    type Input = Task;
    type Output = String;
    type Error = String;

    async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        // Simulate processing time based on agent characteristics
        let complexity = input.required_capabilities.len() as f64;
        let processing_time = self.simulate_processing_time(complexity);
        
        self.current_load += 1;
        
        // Simulate work
        sleep(processing_time).await;
        
        // Simulate potential errors
        if rand::random::<f64>() < self.error_rate {
            self.current_load = self.current_load.saturating_sub(1);
            return Err(format!("Processing failed for task {}", input.id));
        }
        
        self.current_load = self.current_load.saturating_sub(1);
        Ok(format!("Task {} completed by agent {}", input.id, self.id))
    }

    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn status(&self) -> AgentStatus {
        if self.current_load > 5 {
            AgentStatus::Busy
        } else {
            AgentStatus::Running
        }
    }

    fn metadata(&self) -> AgentMetadata {
        AgentMetadata {
            version: "1.0.0".to_string(),
            capabilities: self.capabilities.clone(),
            description: Some(format!("Demo agent with {:.1} tasks/sec processing speed", self.processing_speed)),
            ..Default::default()
        }
    }
}

/// Demonstration of the complete performance monitoring system
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RUV Swarm Performance Monitoring Demo");
    println!("==========================================\n");

    // Initialize performance monitoring system
    let mut performance_metrics = PerformanceSwarmMetrics::new();
    let mut load_balancer = LoadBalancer::new(LoadBalancerConfig::default());
    let mut resource_optimizer = ResourceOptimizer::default();
    let mut dashboard = PerformanceDashboard::new(DashboardConfig::default());

    println!("📊 Initializing performance monitoring components...");
    
    // Create swarm with various agent types to simulate different performance characteristics
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // High-performance agent
    let fast_agent = DemoAgent::new(
        "speed-demon".to_string(),
        vec!["data-processing".to_string(), "analytics".to_string()],
        2.0, // 2 tasks per second
        0.02, // 2% error rate
    );
    
    // Medium-performance agent
    let medium_agent = DemoAgent::new(
        "steady-worker".to_string(),
        vec!["data-processing".to_string(), "validation".to_string()],
        1.0, // 1 task per second
        0.05, // 5% error rate
    );
    
    // Slow but reliable agent
    let slow_agent = DemoAgent::new(
        "careful-processor".to_string(),
        vec!["validation".to_string(), "quality-check".to_string()],
        0.5, // 0.5 tasks per second
        0.01, // 1% error rate
    );

    // Add agents to swarm
    swarm.register_agent(Box::new(fast_agent))?;
    swarm.register_agent(Box::new(medium_agent))?;
    swarm.register_agent(Box::new(slow_agent))?;

    println!("✅ Swarm initialized with 3 agents of varying performance characteristics\n");

    // Start performance monitoring
    println!("📈 Starting performance monitoring simulation...\n");
    
    let simulation_start = Instant::now();
    let mut task_counter = 0;

    // Simulate workload over time with varying intensities
    for round in 1..=5 {
        println!("--- Round {} ---", round);
        
        // Generate tasks with different complexities
        let task_batch_size = match round {
            1 => 3,  // Light load
            2 => 6,  // Medium load
            3 => 10, // Heavy load
            4 => 8,  // Decreasing load
            5 => 4,  // Light load again
            _ => 5,
        };

        println!("🎯 Generating {} tasks for processing", task_batch_size);

        // Submit tasks to swarm
        for i in 0..task_batch_size {
            task_counter += 1;
            let task = Task {
                id: format!("task-{:03}", task_counter),
                priority: match i % 3 {
                    0 => Priority::High,
                    1 => Priority::Medium,
                    _ => Priority::Low,
                },
                required_capabilities: match i % 4 {
                    0 => vec!["data-processing".to_string()],
                    1 => vec!["validation".to_string()],
                    2 => vec!["analytics".to_string()],
                    _ => vec!["quality-check".to_string()],
                },
                payload: format!("Task {} payload", task_counter).into_bytes(),
                metadata: std::collections::HashMap::new(),
            };
            
            swarm.submit_task(task)?;
        }

        // Process tasks and collect metrics
        let task_start = Instant::now();
        let assignments = swarm.distribute_tasks().await?;
        let processing_time = task_start.elapsed();

        println!("⚡ Assigned {} tasks in {:?}", assignments.len(), processing_time);

        // Record performance metrics
        for (task_id, agent_id) in &assignments {
            // Simulate task completion timing
            let completion_time = Duration::from_millis(500 + (rand::random::<u64>() % 2000));
            performance_metrics.record_task_completion(agent_id.clone(), completion_time);
        }

        // Update resource usage
        resource_optimizer.monitor_system_resources()?;

        // Update dashboard with current metrics
        dashboard.update_metrics(performance_metrics.clone());

        // Analyze workload and detect bottlenecks
        let workload_analysis = load_balancer.analyze_workload_distribution(&swarm);
        let bottlenecks = performance_metrics.detect_bottlenecks();

        println!("📊 Workload Analysis:");
        println!("   • Total agents: {}", workload_analysis.total_agents);
        println!("   • Average load per agent: {:.2}", workload_analysis.avg_load_per_agent);
        println!("   • Load imbalance score: {:.3}", workload_analysis.load_imbalance_score);
        println!("   • Distribution efficiency: {:.3}", workload_analysis.distribution_efficiency);

        if !bottlenecks.is_empty() {
            println!("⚠️  Detected {} performance bottleneck(s):", bottlenecks.len());
            for bottleneck in &bottlenecks {
                match bottleneck {
                    ruv_swarm_core::PerformanceBottleneck::SlowAgent { agent_id, avg_completion_time, severity } => {
                        println!("   • Slow agent {}: {:?} (severity: {:?})", agent_id, avg_completion_time, severity);
                    },
                    ruv_swarm_core::PerformanceBottleneck::HighCpuUsage { current_usage, severity } => {
                        println!("   • High CPU usage: {:.1}% (severity: {:?})", current_usage, severity);
                    },
                    ruv_swarm_core::PerformanceBottleneck::HighMemoryUsage { current_usage, limit, severity } => {
                        println!("   • High memory usage: {:.1}/{:.1} MB (severity: {:?})", current_usage, limit, severity);
                    },
                    _ => {
                        println!("   • Other bottleneck detected");
                    }
                }
            }

            // Attempt load balancing if needed
            if workload_analysis.load_imbalance_score > 0.3 {
                println!("⚖️  Attempting load rebalancing...");
                let reassignments = load_balancer.rebalance_tasks(&mut swarm)?;
                if !reassignments.is_empty() {
                    println!("   ✅ Rebalanced {} tasks", reassignments.len());
                }
            }
        }

        // Calculate agent efficiency scores
        let agent_ids = vec!["speed-demon", "steady-worker", "careful-processor"];
        println!("👥 Agent Performance:");
        for agent_id in &agent_ids {
            let efficiency = performance_metrics.get_agent_efficiency(agent_id);
            println!("   • {}: {:.3} efficiency score", agent_id, efficiency);
        }

        // Resource optimization
        let memory_optimization = resource_optimizer.optimize_memory_usage()?;
        if memory_optimization.memory_freed > 0.0 {
            println!("🧹 Memory optimization: Freed {:.1} MB ({:.1}% improvement)", 
                   memory_optimization.memory_freed, 
                   memory_optimization.improvement_percentage);
        }

        println!(); // Empty line for readability
        
        // Wait before next round
        sleep(Duration::from_secs(2)).await;
    }

    let total_simulation_time = simulation_start.elapsed();
    
    println!("🎉 Simulation completed! Total time: {:?}\n", total_simulation_time);

    // Generate comprehensive performance report
    println!("📋 Generating Performance Report...");
    println!("=====================================");
    
    let performance_report = dashboard.generate_performance_report();
    
    println!("📊 Overall Health Score: {:.3}", performance_report.overall_health_score);
    println!("📈 Basic Statistics:");
    println!("   • Total tasks processed: {}", performance_report.basic_statistics.total_tasks_processed);
    println!("   • Average completion time: {:?}", performance_report.basic_statistics.avg_completion_time);
    println!("   • P95 completion time: {:?}", performance_report.basic_statistics.p95_completion_time);
    println!("   • Average throughput: {:.2} tasks/min", performance_report.basic_statistics.avg_throughput);
    println!("   • Peak throughput: {:.2} tasks/min", performance_report.basic_statistics.peak_throughput);
    println!("   • Average error rate: {:.2}%", performance_report.basic_statistics.avg_error_rate);

    println!("\n🔍 Performance Insights:");
    for insight in &performance_report.performance_insights {
        println!("   • {} ({})", insight.title, format!("{:?}", insight.severity));
        println!("     {}", insight.description);
    }

    println!("\n💡 Recommendations:");
    for recommendation in &performance_report.recommendations {
        println!("   • {} ({})", recommendation.title, format!("{:?}", recommendation.priority));
        println!("     {}", recommendation.description);
        println!("     Estimated improvement: {:.1}%", recommendation.estimated_impact.performance_improvement * 100.0);
    }

    // Real-time metrics display
    println!("\n⚡ Real-time Metrics:");
    let rt_metrics = dashboard.get_real_time_metrics();
    println!("   • Current throughput: {:.2} tasks/min", rt_metrics.current_throughput);
    println!("   • Active agents: {}", rt_metrics.active_agents);
    println!("   • Average response time: {:?}", rt_metrics.avg_response_time);
    println!("   • Error rate: {:.2}%", rt_metrics.error_rate);
    println!("   • CPU usage: {:.1}%", rt_metrics.cpu_usage);
    println!("   • Memory usage: {:.1}%", rt_metrics.memory_usage);
    println!("   • Network latency: {:?}", rt_metrics.network_latency);

    // Export metrics for analysis
    println!("\n📤 Exporting Metrics...");
    let json_export = dashboard.export_metrics_for_analysis(ruv_swarm_core::ExportFormat::Json);
    let csv_export = dashboard.export_metrics_for_analysis(ruv_swarm_core::ExportFormat::Csv);
    
    println!("   ✅ JSON export: {} bytes", json_export.size_bytes);
    println!("   ✅ CSV export: {} bytes", csv_export.size_bytes);

    // System health dashboard
    println!("\n🏥 System Health Dashboard:");
    let health_dashboard = dashboard.get_system_health_dashboard();
    println!("   • Overall status: {:?}", health_dashboard.overall_status);
    println!("   • Performance summary:");
    println!("     - Overall score: {:.3}", health_dashboard.performance_summary.overall_score);
    println!("     - Throughput score: {:.3}", health_dashboard.performance_summary.throughput_score);
    println!("     - Efficiency score: {:.3}", health_dashboard.performance_summary.efficiency_score);
    println!("     - Reliability score: {:.3}", health_dashboard.performance_summary.reliability_score);

    // Agent-specific reports
    println!("\n👤 Agent-Specific Performance Reports:");
    for agent_id in &agent_ids {
        let agent_report = dashboard.generate_agent_report(agent_id);
        println!("   • Agent {}: {:.3} efficiency, {:.3} success rate", 
               agent_report.agent_id, 
               agent_report.efficiency_score,
               agent_report.task_completion_stats.success_rate);
    }

    println!("\n🎯 Performance Monitoring Demo Complete!");
    println!("Key achievements:");
    println!("✅ Real-time performance tracking");
    println!("✅ Bottleneck detection and resolution");
    println!("✅ Dynamic load balancing");
    println!("✅ Resource optimization");
    println!("✅ Comprehensive reporting");
    println!("✅ Multi-format metrics export");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_agent_creation() {
        let agent = DemoAgent::new(
            "test".to_string(),
            vec!["test-capability".to_string()],
            1.0,
            0.1,
        );
        
        assert_eq!(agent.id(), "test");
        assert_eq!(agent.capabilities(), &["test-capability"]);
        assert_eq!(agent.processing_speed, 1.0);
        assert_eq!(agent.error_rate, 0.1);
    }

    #[test]
    fn test_processing_time_calculation() {
        let agent = DemoAgent::new(
            "test".to_string(),
            vec![],
            2.0, // 2 tasks per second = 0.5 seconds base time
            0.0,
        );
        
        let processing_time = agent.simulate_processing_time(1.0); // complexity factor of 2
        
        // Base time (0.5s) * complexity factor (2.0) * load factor (1.0) = 1.0s
        assert!(processing_time >= Duration::from_millis(900));
        assert!(processing_time <= Duration::from_millis(1100));
    }

    #[tokio::test]
    async fn test_agent_processing() {
        let mut agent = DemoAgent::new(
            "test".to_string(),
            vec!["data-processing".to_string()],
            10.0, // Very fast for testing
            0.0,  // No errors
        );
        
        let task = Task {
            id: "test-task".to_string(),
            priority: Priority::Medium,
            required_capabilities: vec!["data-processing".to_string()],
            payload: b"test".to_vec(),
            metadata: std::collections::HashMap::new(),
        };
        
        let result = agent.process(task).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("test-task"));
    }
}