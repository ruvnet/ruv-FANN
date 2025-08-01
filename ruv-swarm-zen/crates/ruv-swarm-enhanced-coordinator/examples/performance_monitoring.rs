//! Performance monitoring example demonstrating real-time metrics and optimization
//!
//! This example shows how to:
//! - Track detailed performance metrics across multiple agents
//! - Identify bottlenecks and performance trends
//! - Generate optimization recommendations
//! - Monitor resource utilization and scaling efficiency

use std::time::Duration;
use tokio;
use rand::Rng;
use ruv_swarm_enhanced_coordinator::{
    enhanced_queen_coordinator::*,
    prelude::*,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with detailed output
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();

    println!("📊 Enhanced Queen Coordinator - Performance Monitoring Example");
    println!("==============================================================");

    // Create coordinator optimized for performance monitoring
    let config = CoordinatorConfig {
        max_agents: 50,
        health_check_interval: Duration::from_secs(15),
        task_timeout: Duration::from_secs(180),
        performance_tracking_window: Duration::from_secs(300),
        github_integration_enabled: false,
        strategic_planning_interval: Duration::from_secs(20),
        optimization_threshold: 0.75,
    };

    let coordinator = EnhancedQueenCoordinator::new(config);
    println!("🚀 Performance monitoring coordinator initialized");

    // Phase 1: Create Diverse Agent Pool for Performance Testing
    println!("\n🏭 Phase 1: Agent Pool Creation");
    println!("------------------------------");

    let agent_types = vec![
        ("compute", vec!["compute", "math", "algorithm"], CognitivePattern::Convergent, 8),
        ("analysis", vec!["analysis", "data", "statistics"], CognitivePattern::Divergent, 6),
        ("creative", vec!["design", "ideation", "innovation"], CognitivePattern::Lateral, 4),
        ("system", vec!["architecture", "integration", "coordination"], CognitivePattern::Systems, 5),
        ("review", vec!["testing", "validation", "quality"], CognitivePattern::Critical, 7),
        ("research", vec!["research", "literature", "theory"], CognitivePattern::Abstract, 3),
    ];

    let mut total_agents = 0;
    for (agent_type, capabilities, pattern, count) in agent_types {
        for i in 1..=count {
            let agent_id = format!("{}_{:02}", agent_type, i);
            let caps: Vec<String> = capabilities.iter().map(|s| s.to_string()).collect();
            
            match coordinator.register_agent(agent_id.clone(), caps, pattern).await {
                Ok(_) => {
                    total_agents += 1;
                    if i == 1 {
                        println!("  ✅ {} agents: {} pattern", count, agent_type);
                    }
                }
                Err(e) => println!("  ❌ Failed to register {}: {}", agent_id, e),
            }
        }
    }

    println!("📈 Agent pool created: {} agents across 6 cognitive patterns", total_agents);

    // Phase 2: Baseline Performance Analysis
    println!("\n📊 Phase 2: Baseline Analysis");
    println!("-----------------------------");

    let baseline_analysis = coordinator.strategic_analysis().await;
    
    println!("Initial Swarm Configuration:");
    for (pattern, count) in &baseline_analysis.agent_distribution {
        println!("  {:?}: {} agents ({:.1}%)", 
                pattern, count, (*count as f64 / total_agents as f64) * 100.0);
    }
    
    println!("\nBaseline Metrics:");
    println!("  Coordination Effectiveness: {:.1}%", baseline_analysis.coordination_effectiveness * 100.0);
    println!("  Workload Balance: {:.2}", baseline_analysis.workload_balance);
    println!("  Communication Efficiency: {:.1}%", baseline_analysis.communication_efficiency * 100.0);
    println!("  Resource Utilization: CPU {:.1}%, Memory {:.1}%, Network {:.1}%",
            baseline_analysis.resource_utilization.cpu_usage * 100.0,
            baseline_analysis.resource_utilization.memory_usage * 100.0,
            baseline_analysis.resource_utilization.network_bandwidth * 100.0);

    // Phase 3: Performance Load Testing
    println!("\n⚡ Phase 3: Performance Load Testing");
    println!("-----------------------------------");

    // Create diverse workload scenarios
    let workload_scenarios = vec![
        ("light_compute", "compute", vec!["compute"], 50, TaskPriority::Normal),
        ("data_analysis", "analysis", vec!["analysis", "data"], 30, TaskPriority::High),
        ("creative_work", "ideation", vec!["design", "ideation"], 20, TaskPriority::Normal),
        ("system_integration", "architecture", vec!["architecture", "integration"], 15, TaskPriority::High),
        ("quality_assurance", "testing", vec!["testing", "validation"], 25, TaskPriority::Normal),
        ("research_tasks", "research", vec!["research", "theory"], 10, TaskPriority::Low),
    ];

    let mut all_assignments = Vec::new();
    let mut task_metadata = Vec::new();
    let load_test_start = std::time::Instant::now();

    println!("🔄 Submitting tasks across {} scenarios...", workload_scenarios.len());

    for (scenario_name, task_type, required_caps, task_count, priority) in workload_scenarios {
        println!("  📝 {} scenario: {} tasks", scenario_name, task_count);
        
        for i in 1..=task_count {
            let task_id = format!("{}_{:03}", scenario_name, i);
            let mut task = Task::new(task_id, task_type)
                .with_priority(priority);
            
            for cap in &required_caps {
                task = task.require_capability(cap);
            }

            match coordinator.submit_task(task).await {
                Ok(assignment) => {
                    all_assignments.push(assignment);
                    task_metadata.push((scenario_name, task_type, i));
                }
                Err(e) => {
                    println!("    ❌ Task submission failed: {}", e);
                }
            }
        }
    }

    let submission_time = load_test_start.elapsed();
    println!("⏱️  Task submission completed in {:?}", submission_time);
    println!("📊 {} tasks successfully queued", all_assignments.len());

    // Monitor performance during execution
    println!("\n📈 Phase 4: Real-Time Performance Monitoring");
    println!("--------------------------------------------");

    let monitoring_start = std::time::Instant::now();
    let mut completed_tasks = 0;
    let mut performance_snapshots = Vec::new();

    // Process tasks in batches to simulate realistic execution
    let batch_size = 10;
    for (batch_idx, assignment_batch) in all_assignments.chunks(batch_size).enumerate() {
        println!("  🔄 Processing batch {} ({} tasks)...", batch_idx + 1, assignment_batch.len());
        
        // Simulate parallel task execution with varying performance
        for assignment in assignment_batch {
            let (scenario_name, task_type, task_number) = &task_metadata[completed_tasks];
            
            // Simulate realistic execution times and success rates based on task type
            let (base_time, success_probability, complexity_factor) = match *task_type {
                "compute" => (80, 0.95, 1.0),
                "analysis" => (150, 0.90, 1.3),
                "ideation" => (120, 0.85, 1.5),
                "architecture" => (200, 0.88, 1.8),
                "testing" => (100, 0.92, 1.1),
                "research" => (300, 0.80, 2.0),
                _ => (100, 0.90, 1.0),
            };
            
            // Add random variation to simulate real-world conditions
            let mut rng = rand::thread_rng();
            let variation = rng.gen_range(0.7..1.3);
            let actual_time = (base_time as f64 * complexity_factor * variation) as u64;
            let success = rng.gen::<f64>() < success_probability;
            
            // Add some delay to simulate actual work
            tokio::time::sleep(Duration::from_millis(5)).await;
            
            let task_result = if success {
                TaskResult::success(format!("{} task completed efficiently", scenario_name))
            } else {
                TaskResult::failure(format!("{} task encountered processing issues", scenario_name))
            }
            .with_task_id(assignment.task_id.clone())
            .with_execution_time(actual_time);

            coordinator.complete_task(assignment.task_id.clone(), task_result).await?;
            completed_tasks += 1;
        }
        
        // Take performance snapshot after each batch
        let snapshot_time = monitoring_start.elapsed();
        let current_status = coordinator.get_swarm_status().await;
        let current_analysis = coordinator.strategic_analysis().await;
        
        performance_snapshots.push((
            snapshot_time,
            current_status.metrics.clone(),
            current_analysis.coordination_effectiveness,
            current_analysis.resource_utilization.clone(),
        ));
        
        println!("    ✅ Batch {} completed - {} total tasks finished", 
                batch_idx + 1, completed_tasks);
        println!("       Coordination effectiveness: {:.1}%", 
                current_analysis.coordination_effectiveness * 100.0);
    }

    let total_execution_time = monitoring_start.elapsed();
    println!("🏁 All tasks completed in {:?}", total_execution_time);

    // Phase 5: Comprehensive Performance Analysis
    println!("\n📈 Phase 5: Performance Analysis");
    println!("--------------------------------");

    let final_report = coordinator.generate_performance_report().await;
    let final_analysis = coordinator.strategic_analysis().await;

    println!("Overall Performance Summary:");
    println!("  📋 Total Tasks: {}", final_report.swarm_summary.total_tasks_processed);
    println!("  ✅ Success Rate: {:.1}%", final_report.swarm_summary.overall_success_rate * 100.0);
    println!("  ⏱️  Average Response Time: {:.1}ms", final_report.swarm_summary.average_response_time_ms);
    println!("  🎯 Resource Efficiency: {:.1}%", final_report.swarm_summary.resource_efficiency * 100.0);
    println!("  ⚡ Throughput: {:.1} tasks/second", 
            final_report.swarm_summary.total_tasks_processed as f64 / total_execution_time.as_secs_f64());

    // Analyze performance by agent type
    println!("\nPerformance by Agent Type:");
    let mut agent_type_performance = std::collections::HashMap::new();
    
    for (agent_id, metrics) in &final_report.agent_performances {
        let agent_type = agent_id.split('_').next().unwrap_or("unknown");
        let entry = agent_type_performance.entry(agent_type).or_insert((0, 0.0, 0.0, 0));
        entry.0 += 1; // count
        entry.1 += metrics.success_rate; // success rate sum
        entry.2 += metrics.average_response_time_ms; // response time sum
        entry.3 += metrics.tasks_completed; // task count sum
    }
    
    for (agent_type, (count, success_sum, time_sum, task_sum)) in agent_type_performance {
        println!("  🤖 {}: {} agents, {:.1}% avg success, {:.1}ms avg time, {} total tasks",
                agent_type,
                count,
                (success_sum / count as f64) * 100.0,
                time_sum / count as f64,
                task_sum);
    }

    // Performance trend analysis
    println!("\nPerformance Trends Over Time:");
    for (i, (timestamp, metrics, coordination, resources)) in performance_snapshots.iter().enumerate() {
        println!("  📊 Snapshot {} ({}s): {:.1}% coordination, {:.1}% CPU, {:.0} tasks/s",
                i + 1,
                timestamp.as_secs(),
                coordination * 100.0,
                resources.cpu_usage * 100.0,
                if timestamp.as_secs() > 0 {
                    metrics.total_tasks_processed as f64 / timestamp.as_secs_f64()
                } else { 0.0 });
    }

    // Bottleneck identification
    println!("\nBottleneck Analysis:");
    if final_analysis.bottlenecks.is_empty() {
        println!("  ✅ No significant bottlenecks detected");
    } else {
        for (i, bottleneck) in final_analysis.bottlenecks.iter().enumerate() {
            println!("  ⚠️  {}: {:?} (severity: {:.2})", 
                    i + 1, bottleneck.bottleneck_type, bottleneck.severity);
            println!("     Impact: {}", bottleneck.impact_assessment);
            for action in &bottleneck.recommended_actions {
                println!("     • {}", action);
            }
        }
    }

    // Resource utilization analysis
    println!("\nResource Utilization Analysis:");
    println!("  💻 CPU Usage: {:.1}% (target: <80%)", final_analysis.resource_utilization.cpu_usage * 100.0);
    println!("  🧠 Memory Usage: {:.1}% (target: <70%)", final_analysis.resource_utilization.memory_usage * 100.0);
    println!("  🌐 Network Usage: {:.1}% (target: <60%)", final_analysis.resource_utilization.network_bandwidth * 100.0);
    println!("  💾 Storage I/O: {:.1}% (target: <50%)", final_analysis.resource_utilization.storage_io * 100.0);
    
    if let Some(gpu_usage) = final_analysis.resource_utilization.gpu_usage {
        println!("  🎮 GPU Usage: {:.1}% (target: <90%)", gpu_usage * 100.0);
    }

    // Scalability assessment
    println!("\nScalability Assessment:");
    println!("  📊 Current Capacity: {:.0} agents", final_analysis.scalability_metrics.current_capacity);
    println!("  🚀 Projected Capacity: {:.0} agents", final_analysis.scalability_metrics.projected_capacity);
    println!("  🎯 Optimal Team Size: {} agents", final_analysis.scalability_metrics.optimal_agent_count);
    println!("  ⚡ Scaling Efficiency: {:.1}%", final_analysis.scalability_metrics.scaling_efficiency * 100.0);
    println!("  📈 Coordination Overhead: {:.1}%", final_analysis.scalability_metrics.coordination_overhead_scaling * 100.0);

    // Phase 6: Optimization Recommendations
    println!("\n💡 Phase 6: Optimization Recommendations");
    println!("----------------------------------------");

    let mut recommendations = Vec::new();
    
    // Performance-based recommendations
    if final_report.swarm_summary.overall_success_rate < 0.90 {
        recommendations.push("Consider increasing task timeout or agent training");
    }
    
    if final_report.swarm_summary.average_response_time_ms > 200.0 {
        recommendations.push("Optimize task distribution algorithm for better load balancing");
    }
    
    if final_analysis.resource_utilization.cpu_usage > 0.85 {
        recommendations.push("Scale up compute resources or add more agents");
    }
    
    if final_analysis.coordination_effectiveness < 0.80 {
        recommendations.push("Review swarm topology and communication patterns");
    }
    
    // Cognitive diversity recommendations
    let pattern_counts: Vec<_> = final_analysis.agent_distribution.values().collect();
    let max_count = pattern_counts.iter().max().unwrap_or(&&0);
    let min_count = pattern_counts.iter().min().unwrap_or(&&0);
    
    if max_count > &0 && (*max_count as f64 / *min_count as f64) > 3.0 {
        recommendations.push("Improve cognitive diversity by balancing agent patterns");
    }
    
    // Workload balance recommendations
    if final_analysis.workload_balance < 0.70 {
        recommendations.push("Implement dynamic workload redistribution");
    }

    if recommendations.is_empty() {
        println!("  ✅ System is performing optimally - no immediate optimizations needed");
    } else {
        println!("  📝 Generated {} optimization recommendations:", recommendations.len());
        for (i, rec) in recommendations.iter().enumerate() {
            println!("    {}. {}", i + 1, rec);
        }
    }

    // Phase 7: Performance Benchmarking
    println!("\n🏆 Phase 7: Performance Benchmarks");
    println!("----------------------------------");

    let benchmark_scores = calculate_performance_scores(&final_report, &final_analysis);
    
    println!("Performance Benchmark Scores (0-100):");
    println!("  🚀 Throughput Score: {:.1}", benchmark_scores.throughput);
    println!("  🎯 Accuracy Score: {:.1}", benchmark_scores.accuracy);
    println!("  ⚡ Efficiency Score: {:.1}", benchmark_scores.efficiency);
    println!("  🤝 Coordination Score: {:.1}", benchmark_scores.coordination);
    println!("  📈 Scalability Score: {:.1}", benchmark_scores.scalability);
    println!("  🏅 Overall Score: {:.1}", benchmark_scores.overall);

    let performance_grade = match benchmark_scores.overall {
        90.0..=100.0 => "A+ (Excellent)",
        80.0..=89.9 => "A (Very Good)",
        70.0..=79.9 => "B (Good)",
        60.0..=69.9 => "C (Acceptable)",
        50.0..=59.9 => "D (Needs Improvement)",
        _ => "F (Poor)"
    };
    
    println!("  🎖️  Performance Grade: {}", performance_grade);

    // Final summary with actionable insights
    println!("\n🎉 Performance Monitoring Complete!");
    println!("===================================");
    println!("Key Insights:");
    println!("  📊 Processed {} tasks across {} agents in {:?}", 
            final_report.swarm_summary.total_tasks_processed,
            final_report.swarm_summary.total_agents,
            total_execution_time);
    println!("  ⚡ Average throughput: {:.1} tasks/second", 
            final_report.swarm_summary.total_tasks_processed as f64 / total_execution_time.as_secs_f64());
    println!("  🎯 System efficiency: {:.1}%", final_report.swarm_summary.resource_efficiency * 100.0);
    println!("  🤝 Coordination quality: {:.1}%", final_analysis.coordination_effectiveness * 100.0);
    
    println!("\nNext Steps for Production:");
    println!("  1. Implement continuous monitoring with these metrics");
    println!("  2. Set up alerting for performance degradation");
    println!("  3. Configure auto-scaling based on utilization thresholds");
    println!("  4. Establish performance baselines for regression testing");
    println!("  5. Schedule regular optimization reviews");

    Ok(())
}

/// Performance benchmark scores
#[derive(Debug)]
struct BenchmarkScores {
    throughput: f64,
    accuracy: f64,
    efficiency: f64,
    coordination: f64,
    scalability: f64,
    overall: f64,
}

/// Calculate comprehensive performance benchmark scores
fn calculate_performance_scores(report: &PerformanceReport, analysis: &SwarmAnalysis) -> BenchmarkScores {
    // Throughput score (0-100) based on tasks per second
    let tasks_per_second = 1.0; // Simplified - would calculate from actual data
    let throughput = (tasks_per_second * 20.0).min(100.0); // Max 5 tasks/sec = 100 points
    
    // Accuracy score based on success rate
    let accuracy = report.swarm_summary.overall_success_rate * 100.0;
    
    // Efficiency score based on resource utilization and response time
    let resource_efficiency = (1.0 - analysis.resource_utilization.cpu_usage) * 100.0;
    let time_efficiency = (1000.0 / report.swarm_summary.average_response_time_ms.max(10.0)).min(1.0) * 100.0;
    let efficiency = (resource_efficiency + time_efficiency) / 2.0;
    
    // Coordination score
    let coordination = analysis.coordination_effectiveness * 100.0;
    
    // Scalability score
    let scalability = analysis.scalability_metrics.scaling_efficiency * 100.0;
    
    // Overall score (weighted average)
    let overall = (throughput * 0.25) + (accuracy * 0.25) + (efficiency * 0.20) + 
                  (coordination * 0.15) + (scalability * 0.15);
    
    BenchmarkScores {
        throughput,
        accuracy,
        efficiency,
        coordination,
        scalability,
        overall,
    }
}