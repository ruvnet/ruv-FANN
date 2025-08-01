//! Tests for cognitive diversity patterns in swarm behavior
//! 
//! These tests validate different cognitive patterns and strategies,
//! including convergent/divergent thinking, mixed cognitive teams,
//! and emergent behaviors from diverse agent configurations.

use ruv_swarm_core::{
    agent::{Agent, AgentType, CognitivePattern},
    swarm::{Swarm, SwarmConfig, Topology},
    task::{Task, TaskResult, Strategy},
    metrics::{PerformanceMetrics, DiversityMetrics},
};
use ruv_swarm_agents::{
    ConvergentThinker, DivergentThinker, AnalyticalProcessor,
    CreativeGenerator, PatternRecognizer, SystemIntegrator,
};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};

/// Helper to create a cognitively diverse swarm
async fn create_diverse_swarm(
    patterns: Vec<CognitivePattern>,
) -> Result<Swarm, Box<dyn std::error::Error>> {
    let config = SwarmConfig {
        topology: Topology::Mesh,
        max_agents: 20,
        heartbeat_interval: Duration::from_secs(1),
        task_timeout: Duration::from_secs(60),
        persistence: Box::new(ruv_swarm_persistence::MemoryPersistence::new()),
    };
    
    let mut swarm = Swarm::new(config).await?;
    
    for pattern in patterns {
        let agent_type = match pattern {
            CognitivePattern::Convergent => AgentType::ConvergentThinker,
            CognitivePattern::Divergent => AgentType::DivergentThinker,
            CognitivePattern::Analytical => AgentType::AnalyticalProcessor,
            CognitivePattern::Creative => AgentType::CreativeGenerator,
            CognitivePattern::PatternMatching => AgentType::PatternRecognizer,
            CognitivePattern::Integrative => AgentType::SystemIntegrator,
        };
        
        swarm.spawn_with_pattern(agent_type, pattern).await?;
    }
    
    Ok(swarm)
}

/// Generate a complex problem that benefits from diverse thinking
fn generate_complex_problem() -> Task {
    Task::ComplexProblemSolving {
        problem_domain: "optimization".to_string(),
        constraints: vec![
            "minimize_cost".to_string(),
            "maximize_efficiency".to_string(),
            "ensure_reliability".to_string(),
        ],
        data: vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ],
        required_confidence: 0.85,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_convergent_vs_divergent_strategies() {
        // Test pure convergent team
        let convergent_swarm = create_diverse_swarm(vec![
            CognitivePattern::Convergent,
            CognitivePattern::Convergent,
            CognitivePattern::Convergent,
        ]).await.unwrap();
        
        // Test pure divergent team  
        let divergent_swarm = create_diverse_swarm(vec![
            CognitivePattern::Divergent,
            CognitivePattern::Divergent,
            CognitivePattern::Divergent,
        ]).await.unwrap();
        
        let problem = generate_complex_problem();
        
        // Measure convergent team performance
        let start = std::time::Instant::now();
        let convergent_result = convergent_swarm.orchestrate(problem.clone()).await.unwrap();
        let convergent_time = start.elapsed();
        
        // Measure divergent team performance
        let start = std::time::Instant::now();
        let divergent_result = divergent_swarm.orchestrate(problem).await.unwrap();
        let divergent_time = start.elapsed();
        
        // Analyze results
        match (convergent_result, divergent_result) {
            (
                TaskResult::ProblemSolved { solutions: conv_sols, .. },
                TaskResult::ProblemSolved { solutions: div_sols, .. }
            ) => {
                // Convergent teams should produce fewer, more refined solutions
                assert!(conv_sols.len() < div_sols.len());
                
                // Divergent teams should produce more diverse solutions
                let conv_diversity = calculate_solution_diversity(&conv_sols);
                let div_diversity = calculate_solution_diversity(&div_sols);
                assert!(div_diversity > conv_diversity);
                
                // Convergent teams should be faster for well-defined problems
                assert!(convergent_time < divergent_time);
            }
            _ => panic!("Expected ProblemSolved results"),
        }
    }

    #[tokio::test]
    async fn test_mixed_cognitive_teams() {
        // Create balanced team
        let balanced_swarm = create_diverse_swarm(vec![
            CognitivePattern::Convergent,
            CognitivePattern::Divergent,
            CognitivePattern::Analytical,
            CognitivePattern::Creative,
            CognitivePattern::PatternMatching,
            CognitivePattern::Integrative,
        ]).await.unwrap();
        
        // Create specialized teams
        let analytical_team = create_diverse_swarm(vec![
            CognitivePattern::Analytical,
            CognitivePattern::Analytical,
            CognitivePattern::PatternMatching,
            CognitivePattern::PatternMatching,
        ]).await.unwrap();
        
        let creative_team = create_diverse_swarm(vec![
            CognitivePattern::Creative,
            CognitivePattern::Creative,
            CognitivePattern::Divergent,
            CognitivePattern::Divergent,
        ]).await.unwrap();
        
        // Test on different problem types
        let analytical_problem = Task::DataAnalysis {
            dataset: generate_test_dataset(1000),
            analysis_type: "statistical".to_string(),
        };
        
        let creative_problem = Task::CreativeGeneration {
            prompt: "novel optimization algorithm".to_string(),
            constraints: vec!["must be efficient".to_string()],
        };
        
        // Balanced team should perform well on both
        let balanced_analytical = balanced_swarm.orchestrate(analytical_problem.clone()).await.unwrap();
        let balanced_creative = balanced_swarm.orchestrate(creative_problem.clone()).await.unwrap();
        
        // Specialized teams should excel in their domains
        let specialized_analytical = analytical_team.orchestrate(analytical_problem).await.unwrap();
        let specialized_creative = creative_team.orchestrate(creative_problem).await.unwrap();
        
        // Verify specialized teams outperform in their domains
        assert!(
            get_task_quality(&specialized_analytical) > get_task_quality(&balanced_analytical) * 0.9,
            "Analytical team should excel at analytical tasks"
        );
        
        assert!(
            get_task_quality(&specialized_creative) > get_task_quality(&balanced_creative) * 0.9,
            "Creative team should excel at creative tasks"
        );
    }

    #[tokio::test]
    async fn test_emergent_behaviors() {
        let mut swarm = create_diverse_swarm(vec![
            CognitivePattern::Convergent,
            CognitivePattern::Divergent,
            CognitivePattern::Analytical,
            CognitivePattern::Creative,
            CognitivePattern::PatternMatching,
        ]).await.unwrap();
        
        // Enable behavior tracking
        swarm.enable_behavior_tracking().await.unwrap();
        
        // Run complex collaborative task
        let collaborative_task = Task::CollaborativeProblemSolving {
            phases: vec![
                "brainstorming".to_string(),
                "analysis".to_string(),
                "synthesis".to_string(),
                "refinement".to_string(),
            ],
            data: generate_complex_problem().into(),
        };
        
        let result = swarm.orchestrate(collaborative_task).await.unwrap();
        
        // Analyze emergent behaviors
        let behaviors = swarm.get_emergent_behaviors().await.unwrap();
        
        // Check for expected emergent patterns
        assert!(behaviors.contains_key("spontaneous_collaboration"));
        assert!(behaviors.contains_key("role_specialization"));
        assert!(behaviors.contains_key("information_cascades"));
        
        // Verify diversity metrics improved performance
        let diversity_metrics = swarm.get_diversity_metrics().await.unwrap();
        assert!(diversity_metrics.cognitive_diversity > 0.7);
        assert!(diversity_metrics.strategy_diversity > 0.6);
        assert!(diversity_metrics.solution_diversity > 0.8);
    }

    #[tokio::test]
    async fn test_cognitive_pattern_adaptation() {
        let mut swarm = create_diverse_swarm(vec![
            CognitivePattern::Convergent,
            CognitivePattern::Divergent,
            CognitivePattern::Analytical,
        ]).await.unwrap();
        
        // Enable adaptive behavior
        swarm.enable_adaptation().await.unwrap();
        
        // Run series of tasks that favor different patterns
        let tasks = vec![
            // Analytical task
            Task::DataAnalysis {
                dataset: generate_test_dataset(100),
                analysis_type: "regression".to_string(),
            },
            // Creative task
            Task::CreativeGeneration {
                prompt: "innovative solution".to_string(),
                constraints: vec![],
            },
            // Convergent task
            Task::Optimization {
                objective: "minimize_error".to_string(),
                parameters: vec![1.0, 2.0, 3.0],
            },
        ];
        
        let initial_patterns = swarm.get_agent_patterns().await.unwrap();
        
        // Execute tasks
        for task in tasks {
            swarm.orchestrate(task).await.unwrap();
        }
        
        let adapted_patterns = swarm.get_agent_patterns().await.unwrap();
        
        // Verify agents adapted their patterns
        let adaptation_score = calculate_adaptation_score(&initial_patterns, &adapted_patterns);
        assert!(adaptation_score > 0.3, "Agents should show pattern adaptation");
    }

    #[tokio::test]
    async fn test_diversity_impact_on_performance() {
        let diversity_levels = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut performance_results = HashMap::new();
        
        for diversity in diversity_levels {
            let patterns = generate_patterns_with_diversity(6, diversity);
            let swarm = create_diverse_swarm(patterns).await.unwrap();
            
            let problem = generate_complex_problem();
            let start = std::time::Instant::now();
            let result = swarm.orchestrate(problem).await.unwrap();
            let duration = start.elapsed();
            
            let quality = get_task_quality(&result);
            performance_results.insert(diversity.to_string(), (quality, duration));
        }
        
        // Analyze relationship between diversity and performance
        let optimal_diversity = find_optimal_diversity(&performance_results);
        assert!(optimal_diversity > 0.4 && optimal_diversity < 0.8,
            "Moderate diversity should be optimal");
    }

    #[tokio::test]
    async fn test_cognitive_load_distribution() {
        let swarm = create_diverse_swarm(vec![
            CognitivePattern::Analytical,
            CognitivePattern::Creative,
            CognitivePattern::PatternMatching,
            CognitivePattern::Integrative,
        ]).await.unwrap();
        
        // Create high cognitive load task
        let complex_task = Task::MultiObjectiveOptimization {
            objectives: vec![
                "minimize_cost".to_string(),
                "maximize_quality".to_string(),
                "minimize_time".to_string(),
                "maximize_reliability".to_string(),
            ],
            constraints: generate_complex_constraints(),
            data: generate_large_dataset(),
        };
        
        // Monitor cognitive load during execution
        let load_monitor = swarm.start_load_monitoring().await.unwrap();
        let result = swarm.orchestrate(complex_task).await.unwrap();
        let load_data = load_monitor.stop().await.unwrap();
        
        // Verify load was distributed effectively
        let max_load = load_data.values().max().unwrap();
        let min_load = load_data.values().min().unwrap();
        let avg_load = load_data.values().sum::<f64>() / load_data.len() as f64;
        
        assert!(max_load / avg_load < 1.5, "Load should be well distributed");
        assert!(min_load / avg_load > 0.5, "No agent should be underutilized");
    }

    #[tokio::test]
    async fn test_cultural_evolution() {
        let mut swarm = create_diverse_swarm(vec![
            CognitivePattern::Convergent,
            CognitivePattern::Divergent,
            CognitivePattern::Analytical,
            CognitivePattern::Creative,
        ]).await.unwrap();
        
        // Enable cultural evolution
        swarm.enable_cultural_evolution().await.unwrap();
        
        // Run multiple generations of tasks
        for generation in 0..10 {
            let task = Task::GenerationalLearning {
                generation,
                problem: Box::new(generate_complex_problem()),
                inheritance_rate: 0.7,
            };
            
            swarm.orchestrate(task).await.unwrap();
        }
        
        // Analyze cultural traits that emerged
        let cultural_traits = swarm.get_cultural_traits().await.unwrap();
        
        assert!(!cultural_traits.is_empty(), "Cultural traits should emerge");
        assert!(cultural_traits.contains_key("problem_solving_heuristics"));
        assert!(cultural_traits.contains_key("communication_patterns"));
        
        // Verify performance improved over generations
        let generational_performance = swarm.get_generational_metrics().await.unwrap();
        let early_avg = generational_performance[..3].iter().sum::<f64>() / 3.0;
        let late_avg = generational_performance[7..].iter().sum::<f64>() / 3.0;
        
        assert!(late_avg > early_avg * 1.2, "Performance should improve over generations");
    }
}

// Helper functions

fn calculate_solution_diversity(solutions: &[Vec<f64>]) -> f64 {
    if solutions.len() < 2 {
        return 0.0;
    }
    
    let mut total_distance = 0.0;
    let mut count = 0;
    
    for i in 0..solutions.len() {
        for j in i+1..solutions.len() {
            let distance = euclidean_distance(&solutions[i], &solutions[j]);
            total_distance += distance;
            count += 1;
        }
    }
    
    total_distance / count as f64
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn generate_test_dataset(size: usize) -> Vec<Vec<f64>> {
    (0..size)
        .map(|i| vec![i as f64, (i as f64).sin(), (i as f64).cos()])
        .collect()
}

fn get_task_quality(result: &TaskResult) -> f64 {
    match result {
        TaskResult::ProblemSolved { quality_score, .. } => *quality_score,
        TaskResult::AnalysisComplete { accuracy, .. } => *accuracy,
        TaskResult::GenerationComplete { novelty_score, .. } => *novelty_score,
        _ => 0.5,
    }
}

fn generate_patterns_with_diversity(count: usize, diversity: f64) -> Vec<CognitivePattern> {
    let patterns = vec![
        CognitivePattern::Convergent,
        CognitivePattern::Divergent,
        CognitivePattern::Analytical,
        CognitivePattern::Creative,
        CognitivePattern::PatternMatching,
        CognitivePattern::Integrative,
    ];
    
    let n_unique = ((patterns.len() as f64) * diversity).ceil() as usize;
    let n_unique = n_unique.max(1).min(patterns.len());
    
    let mut result = Vec::new();
    for i in 0..count {
        result.push(patterns[i % n_unique].clone());
    }
    
    result
}

fn find_optimal_diversity(results: &HashMap<String, (f64, Duration)>) -> f64 {
    let mut best_diversity = 0.0;
    let mut best_score = 0.0;
    
    for (diversity_str, (quality, duration)) in results {
        let diversity: f64 = diversity_str.parse().unwrap();
        // Combined score: quality - time_penalty
        let score = quality - (duration.as_secs_f64() / 100.0);
        
        if score > best_score {
            best_score = score;
            best_diversity = diversity;
        }
    }
    
    best_diversity
}

fn generate_complex_constraints() -> Vec<String> {
    vec![
        "x + y <= 100".to_string(),
        "x * y >= 50".to_string(),
        "x^2 + y^2 <= 1000".to_string(),
        "x > 0 && y > 0".to_string(),
    ]
}

fn generate_large_dataset() -> Vec<Vec<f64>> {
    generate_test_dataset(10000)
}

fn calculate_adaptation_score(
    initial: &HashMap<AgentId, CognitivePattern>,
    adapted: &HashMap<AgentId, CognitivePattern>,
) -> f64 {
    let mut changes = 0;
    let total = initial.len();
    
    for (agent_id, initial_pattern) in initial {
        if let Some(adapted_pattern) = adapted.get(agent_id) {
            if initial_pattern != adapted_pattern {
                changes += 1;
            }
        }
    }
    
    changes as f64 / total as f64
}