//! Basic coordination example demonstrating core Enhanced Queen Coordinator functionality
//!
//! This example shows how to:
//! - Create and configure the enhanced coordinator
//! - Register agents with different cognitive patterns
//! - Submit and track tasks
//! - Monitor performance metrics
//! - Generate strategic analysis

use std::time::Duration;
use tokio;
use ruv_swarm_enhanced_coordinator::{
    enhanced_queen_coordinator::*,
    CoordinatorConfig, EnhancedQueenCoordinator,
    TaskPriority, TaskResult, Task, TaskId, CognitivePattern,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("🚀 Enhanced Queen Coordinator - Basic Coordination Example");
    println!("=========================================================");

    // Create coordinator with custom configuration
    let config = CoordinatorConfig {
        max_agents: 20,
        health_check_interval: Duration::from_secs(30),
        task_timeout: Duration::from_secs(120),
        performance_tracking_window: Duration::from_secs(300),
        github_integration_enabled: false,
        strategic_planning_interval: Duration::from_secs(30),
        optimization_threshold: 0.75,
    };

    let coordinator = EnhancedQueenCoordinator::new(config);
    println!("✅ Coordinator initialized with custom configuration");

    // Phase 1: Register diverse agents
    println!("\n📝 Phase 1: Registering Agents");
    println!("------------------------------");

    let agent_configurations = vec![
        ("alice_researcher", vec!["research", "literature", "analysis"], CognitivePattern::Divergent),
        ("bob_analyst", vec!["analysis", "statistics", "data"], CognitivePattern::Convergent),
        ("charlie_creative", vec!["design", "ideation", "innovation"], CognitivePattern::Lateral),
        ("diana_architect", vec!["architecture", "systems", "integration"], CognitivePattern::Systems),
        ("eve_critic", vec!["review", "validation", "testing"], CognitivePattern::Critical),
        ("frank_theorist", vec!["theory", "modeling", "abstraction"], CognitivePattern::Abstract),
    ];

    for (name, capabilities, pattern) in agent_configurations {
        let caps: Vec<String> = capabilities.into_iter().map(|s| s.to_string()).collect();
        
        match coordinator.register_agent(name.to_string(), caps.clone(), pattern).await {
            Ok(_) => println!("  ✅ Registered {} with {:?} pattern and capabilities: {:?}", 
                            name, pattern, caps),
            Err(e) => println!("  ❌ Failed to register {}: {}", name, e),
        }
    }

    // Check initial swarm status
    let initial_status = coordinator.get_swarm_status().await;
    println!("\n📊 Initial Swarm Status:");
    println!("  Total Agents: {}", initial_status.total_agents);
    println!("  Active Agents: {}", initial_status.active_agents);
    println!("  Topology: {}", initial_status.topology);

    // Phase 2: Strategic Analysis
    println!("\n🧠 Phase 2: Strategic Analysis");
    println!("------------------------------");

    let strategic_analysis = coordinator.strategic_analysis().await;
    
    println!("Agent Distribution by Cognitive Pattern:");
    for (pattern, count) in &strategic_analysis.agent_distribution {
        println!("  {:?}: {} agents", pattern, count);
    }
    
    println!("Performance Metrics:");
    println!("  Workload Balance: {:.2}", strategic_analysis.workload_balance);
    println!("  Communication Efficiency: {:.2}", strategic_analysis.communication_efficiency);
    println!("  Coordination Effectiveness: {:.2}", strategic_analysis.coordination_effectiveness);
    
    println!("Resource Utilization:");
    println!("  CPU Usage: {:.1}%", strategic_analysis.resource_utilization.cpu_usage * 100.0);
    println!("  Memory Usage: {:.1}%", strategic_analysis.resource_utilization.memory_usage * 100.0);
    println!("  Network Bandwidth: {:.1}%", strategic_analysis.resource_utilization.network_bandwidth * 100.0);
    
    if !strategic_analysis.bottlenecks.is_empty() {
        println!("Identified Bottlenecks:");
        for (i, bottleneck) in strategic_analysis.bottlenecks.iter().enumerate() {
            println!("  {}. {:?} (Severity: {:.2})", i + 1, bottleneck.bottleneck_type, bottleneck.severity);
        }
    } else {
        println!("  ✅ No bottlenecks identified");
    }

    // Phase 3: Task Submission and Execution
    println!("\n⚡ Phase 3: Task Execution");
    println!("-------------------------");

    let research_tasks = vec![
        ("market_research", "research", vec!["research", "analysis"], TaskPriority::High),
        ("data_analysis", "analysis", vec!["analysis", "statistics"], TaskPriority::Normal),
        ("creative_brainstorm", "ideation", vec!["design", "ideation"], TaskPriority::Normal),
        ("system_architecture", "architecture", vec!["architecture", "systems"], TaskPriority::High),
        ("quality_review", "review", vec!["review", "validation"], TaskPriority::Normal),
        ("theoretical_model", "theory", vec!["theory", "modeling"], TaskPriority::Low),
    ];

    let mut active_assignments = Vec::new();

    for (task_name, task_type, required_caps, priority) in research_tasks {
        let mut task = Task::new(task_name, task_type)
            .with_priority(priority);
        
        for cap in required_caps {
            task = task.require_capability(cap);
        }

        match coordinator.submit_task(task).await {
            Ok(assignment) => {
                println!("  ✅ Task '{}' assigned to '{}' with confidence {:.2}", 
                        task_name, assignment.assigned_agent, assignment.confidence);
                println!("     Rationale: {}", assignment.assignment_rationale);
                active_assignments.push((assignment, task_name));
            }
            Err(e) => {
                println!("  ❌ Failed to assign task '{}': {}", task_name, e);
            }
        }
    }

    // Check swarm status after task submission
    let mid_status = coordinator.get_swarm_status().await;
    println!("\n📊 Swarm Status After Task Submission:");
    println!("  Active Tasks: {}", mid_status.active_tasks);
    println!("  Queued Tasks: {}", mid_status.queued_tasks);

    // Phase 4: Simulate Task Completion
    println!("\n🔄 Phase 4: Task Completion Simulation");
    println!("--------------------------------------");

    for (assignment, task_name) in active_assignments {
        // Simulate realistic execution times and success rates
        let (execution_time, success, output) = match task_name {
            "market_research" => (150, true, "Comprehensive market analysis completed with key insights"),
            "data_analysis" => (200, true, "Statistical analysis revealed significant patterns"),
            "creative_brainstorm" => (120, true, "Generated 15 innovative concepts for consideration"),
            "system_architecture" => (300, true, "Scalable architecture design with performance optimizations"),
            "quality_review" => (90, true, "Quality assessment passed with minor recommendations"),
            "theoretical_model" => (400, true, "Mathematical model validated with 95% confidence"),
            _ => (100, true, "Task completed successfully"),
        };

        // Small delay to simulate actual execution time
        tokio::time::sleep(Duration::from_millis(50)).await;

        let task_result = if success {
            TaskResult::success(output)
        } else {
            TaskResult::failure("Task encountered unexpected challenges")
        }
        .with_task_id(assignment.task_id.clone())
        .with_execution_time(execution_time);

        match coordinator.complete_task(assignment.task_id.clone(), task_result).await {
            Ok(_) => println!("  ✅ Completed '{}' in {}ms", task_name, execution_time),
            Err(e) => println!("  ❌ Error completing '{}': {}", task_name, e),
        }
    }

    // Phase 5: Performance Analysis
    println!("\n📈 Phase 5: Performance Analysis");
    println!("--------------------------------");

    let performance_report = coordinator.generate_performance_report().await;
    
    println!("Swarm Performance Summary:");
    println!("  Total Tasks Processed: {}", performance_report.swarm_summary.total_tasks_processed);
    println!("  Overall Success Rate: {:.1}%", performance_report.swarm_summary.overall_success_rate * 100.0);
    println!("  Average Response Time: {:.1}ms", performance_report.swarm_summary.average_response_time_ms);
    println!("  Resource Efficiency: {:.1}%", performance_report.swarm_summary.resource_efficiency * 100.0);

    println!("\nAgent Performance Details:");
    for (agent_id, metrics) in &performance_report.agent_performances {
        println!("  🤖 {}:", agent_id);
        println!("    Tasks Completed: {}", metrics.tasks_completed);
        println!("    Tasks Failed: {}", metrics.tasks_failed);
        println!("    Success Rate: {:.1}%", metrics.success_rate * 100.0);
        println!("    Avg Response Time: {:.1}ms", metrics.average_response_time_ms);
        println!("    Throughput: {:.1} tasks/hour", metrics.throughput_per_hour);
        println!("    Resource Efficiency: {:.1}%", metrics.resource_efficiency * 100.0);
        println!("    Cognitive Effectiveness: {:.1}%", metrics.cognitive_pattern_effectiveness * 100.0);
    }

    if !performance_report.recommendations.is_empty() {
        println!("\n💡 Optimization Recommendations:");
        for (i, recommendation) in performance_report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, recommendation);
        }
    }

    // Phase 6: Final Strategic Analysis
    println!("\n🎯 Phase 6: Post-Execution Analysis");
    println!("-----------------------------------");

    let final_analysis = coordinator.strategic_analysis().await;
    
    println!("Performance Improvements:");
    println!("  Coordination Effectiveness: {:.2} → {:.2}", 
            strategic_analysis.coordination_effectiveness,
            final_analysis.coordination_effectiveness);
    println!("  Workload Balance: {:.2} → {:.2}", 
            strategic_analysis.workload_balance,
            final_analysis.workload_balance);

    println!("\nScalability Insights:");
    println!("  Current Capacity: {:.0} agents", final_analysis.scalability_metrics.current_capacity);
    println!("  Projected Capacity: {:.0} agents", final_analysis.scalability_metrics.projected_capacity);
    println!("  Optimal Agent Count: {} agents", final_analysis.scalability_metrics.optimal_agent_count);
    println!("  Scaling Efficiency: {:.1}%", final_analysis.scalability_metrics.scaling_efficiency * 100.0);

    // Phase 7: MCP Integration Demonstration
    println!("\n🔗 Phase 7: MCP Integration");
    println!("---------------------------");

    match coordinator.coordinate_with_mcp_tools().await {
        Ok(_) => println!("  ✅ MCP coordination completed successfully"),
        Err(e) => println!("  ⚠️  MCP coordination encountered issues: {}", e),
    }

    // Final status
    let final_status = coordinator.get_swarm_status().await;
    println!("\n🏁 Final Swarm Status:");
    println!("  Total Agents: {}", final_status.total_agents);
    println!("  Active Agents: {}", final_status.active_agents);
    println!("  Total Tasks Processed: {}", final_status.metrics.total_tasks_processed);
    println!("  Successful Tasks: {}", final_status.metrics.successful_tasks);
    println!("  Failed Tasks: {}", final_status.metrics.failed_tasks);
    println!("  Average Response Time: {:.1}ms", final_status.metrics.average_response_time_ms);
    println!("  Agent Utilization: {:.1}%", final_status.metrics.agent_utilization * 100.0);

    println!("\n🎉 Basic coordination example completed successfully!");
    println!("The Enhanced Queen Coordinator has demonstrated:");
    println!("  ✅ Multi-agent registration with cognitive diversity");
    println!("  ✅ Intelligent task assignment based on capabilities and patterns");
    println!("  ✅ Real-time performance tracking and metrics collection");
    println!("  ✅ Strategic analysis and optimization recommendations");
    println!("  ✅ Scalability assessment and capacity planning");
    println!("  ✅ MCP integration for enhanced coordination");

    Ok(())
}