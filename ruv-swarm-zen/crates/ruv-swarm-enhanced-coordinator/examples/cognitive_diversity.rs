//! Cognitive Diversity Example
//! 
//! This example demonstrates how different cognitive styles in agents
//! can lead to better problem-solving through diverse approaches.

use ruv_swarm::{
    agent::{AgentType, CognitiveStyle},
    swarm::{Swarm, SwarmConfig},
    topology::Topology,
    task::{Task, TaskType},
    Result,
};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("=== Cognitive Diversity Example ===\n");

    // Create swarm with cognitive diversity enabled
    let config = SwarmConfig {
        max_agents: 12,
        topology: Topology::SmallWorld,
        cognitive_diversity: true,
        collaboration_threshold: 0.7,
        ..Default::default()
    };

    let mut swarm = Swarm::new(config)?;

    // Spawn agents with different cognitive styles
    println!("Creating cognitively diverse agent team...\n");

    // Analytical agents - good for data analysis and optimization
    let analysts = vec![
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Analytical).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Analytical).await?,
    ];
    println!("Analytical Agents ({}): Focus on data analysis and optimization", analysts.len());

    // Creative agents - good for novel solutions and exploration
    let creatives = vec![
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Creative).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Creative).await?,
    ];
    println!("Creative Agents ({}): Focus on innovative solutions", creatives.len());

    // Strategic agents - good for planning and coordination
    let strategists = vec![
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Strategic).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Strategic).await?,
    ];
    println!("Strategic Agents ({}): Focus on planning and coordination", strategists.len());

    // Practical agents - good for implementation and execution
    let practitioners = vec![
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Practical).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::Practical).await?,
    ];
    println!("Practical Agents ({}): Focus on implementation", practitioners.len());

    // Detail-oriented agents - good for quality assurance and refinement
    let detail_oriented = vec![
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::DetailOriented).await?,
        swarm.spawn_agent_with_style(AgentType::Worker, CognitiveStyle::DetailOriented).await?,
    ];
    println!("Detail-Oriented Agents ({}): Focus on quality and precision", detail_oriented.len());

    // Create a complex problem that benefits from diverse approaches
    println!("\n=== Complex Problem Solving ===");
    println!("Problem: Design and optimize a recommendation system\n");

    // Phase 1: Research and Analysis
    println!("Phase 1: Research and Analysis");
    let research_task = Task::new(
        "Research recommendation algorithms",
        TaskType::Research,
        vec![
            "Survey collaborative filtering methods",
            "Analyze content-based approaches",
            "Investigate hybrid algorithms",
            "Review deep learning techniques",
        ],
    )?;

    let start_time = Instant::now();
    let research_result = swarm.orchestrate_with_diversity(research_task).await?;
    println!("  Completed in {:?}", start_time.elapsed());
    println!("  Lead agents: {:?}", research_result.lead_agents);
    println!("  Key findings: {} items\n", research_result.outputs.len());

    // Phase 2: Creative Design
    println!("Phase 2: Creative Design");
    let design_task = Task::new(
        "Design innovative recommendation features",
        TaskType::Creative,
        vec![
            "Brainstorm novel user interaction patterns",
            "Design adaptive learning mechanisms",
            "Create personalization strategies",
            "Develop explainable AI features",
        ],
    )?;

    let start_time = Instant::now();
    let design_result = swarm.orchestrate_with_diversity(design_task).await?;
    println!("  Completed in {:?}", start_time.elapsed());
    println!("  Lead agents: {:?}", design_result.lead_agents);
    println!("  Creative solutions: {} concepts\n", design_result.outputs.len());

    // Phase 3: Strategic Planning
    println!("Phase 3: Strategic Planning");
    let planning_task = Task::new(
        "Plan system architecture and deployment",
        TaskType::Strategic,
        vec![
            "Design scalable architecture",
            "Plan phased rollout strategy",
            "Define performance metrics",
            "Create risk mitigation plan",
        ],
    )?;

    let start_time = Instant::now();
    let planning_result = swarm.orchestrate_with_diversity(planning_task).await?;
    println!("  Completed in {:?}", start_time.elapsed());
    println!("  Lead agents: {:?}", planning_result.lead_agents);
    println!("  Strategic decisions: {} items\n", planning_result.outputs.len());

    // Phase 4: Implementation
    println!("Phase 4: Implementation");
    let implementation_task = Task::new(
        "Implement core recommendation engine",
        TaskType::Implementation,
        vec![
            "Build data processing pipeline",
            "Implement ML algorithms",
            "Create API endpoints",
            "Develop caching layer",
        ],
    )?;

    let start_time = Instant::now();
    let impl_result = swarm.orchestrate_with_diversity(implementation_task).await?;
    println!("  Completed in {:?}", start_time.elapsed());
    println!("  Lead agents: {:?}", impl_result.lead_agents);
    println!("  Components built: {} modules\n", impl_result.outputs.len());

    // Phase 5: Quality Assurance
    println!("Phase 5: Quality Assurance");
    let qa_task = Task::new(
        "Comprehensive testing and optimization",
        TaskType::Analysis,
        vec![
            "Unit test all components",
            "Performance benchmarking",
            "Security audit",
            "User experience testing",
        ],
    )?;

    let start_time = Instant::now();
    let qa_result = swarm.orchestrate_with_diversity(qa_task).await?;
    println!("  Completed in {:?}", start_time.elapsed());
    println!("  Lead agents: {:?}", qa_result.lead_agents);
    println!("  Issues found and fixed: {} items\n", qa_result.outputs.len());

    // Analyze collaboration patterns
    println!("=== Collaboration Analysis ===");
    let collab_stats = swarm.get_collaboration_stats();
    
    println!("Cross-Style Collaboration Matrix:");
    for (style1, style2, count) in &collab_stats.collaboration_matrix {
        if count > &0 {
            println!("  {:?} <-> {:?}: {} interactions", style1, style2, count);
        }
    }

    println!("\nTask Performance by Cognitive Style:");
    for (task_type, style, performance) in &collab_stats.task_performance {
        println!("  {:?} + {:?}: {:.2}% success rate", 
                 task_type, style, performance * 100.0);
    }

    println!("\nDiversity Impact:");
    println!("  Solution quality improvement: {:.1}%", 
             collab_stats.diversity_impact.quality_improvement * 100.0);
    println!("  Time to solution reduction: {:.1}%", 
             collab_stats.diversity_impact.time_reduction * 100.0);
    println!("  Novel solutions generated: {}", 
             collab_stats.diversity_impact.novel_solutions);

    // Shutdown
    swarm.shutdown().await?;
    println!("\nExample completed!");
    
    Ok(())
}