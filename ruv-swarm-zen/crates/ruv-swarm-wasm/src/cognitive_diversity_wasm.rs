// cognitive_diversity_wasm.rs - Cognitive diversity and pattern management

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[wasm_bindgen]
pub struct CognitiveDiversityEngine {
    patterns: HashMap<String, CognitivePattern>,
    diversity_metrics: DiversityMetrics,
    pattern_interactions: HashMap<String, Vec<PatternInteraction>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CognitivePattern {
    pub name: String,
    pub description: String,
    pub processing_style: ProcessingStyle,
    pub neural_config: NeuralConfiguration,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub optimal_tasks: Vec<String>,
    pub interaction_weights: HashMap<String, f32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessingStyle {
    pub focus_type: String, // "narrow", "broad", "adaptive"
    pub decision_speed: String, // "fast", "deliberate", "balanced"
    pub risk_tolerance: String, // "conservative", "moderate", "aggressive"
    pub information_processing: String, // "sequential", "parallel", "hybrid"
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeuralConfiguration {
    pub architecture_type: String,
    pub layer_sizes: Vec<usize>,
    pub activation_functions: Vec<String>,
    pub learning_parameters: LearningParameters,
    pub specialized_modules: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LearningParameters {
    pub learning_rate: f32,
    pub momentum: f32,
    pub adaptation_rate: f32,
    pub memory_retention: f32,
    pub exploration_factor: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DiversityMetrics {
    pub overall_diversity_score: f32,
    pub pattern_distribution: HashMap<String, f32>,
    pub interaction_balance: f32,
    pub redundancy_factor: f32,
    pub coverage_score: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PatternInteraction {
    pub pattern_a: String,
    pub pattern_b: String,
    pub interaction_type: String, // "synergistic", "complementary", "conflicting"
    pub interaction_strength: f32,
    pub optimal_ratio: f32,
}

#[wasm_bindgen]
impl CognitiveDiversityEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CognitiveDiversityEngine {
        let mut engine = CognitiveDiversityEngine {
            patterns: HashMap::new(),
            diversity_metrics: DiversityMetrics {
                overall_diversity_score: 0.0,
                pattern_distribution: HashMap::new(),
                interaction_balance: 0.0,
                redundancy_factor: 0.0,
                coverage_score: 0.0,
            },
            pattern_interactions: HashMap::new(),
        };
        
        engine.initialize_cognitive_patterns();
        engine.initialize_pattern_interactions();
        
        engine
    }
    
    #[wasm_bindgen]
    pub fn get_cognitive_patterns(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.patterns).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn analyze_swarm_diversity(&mut self, swarm_composition: JsValue) -> Result<JsValue, JsValue> {
        let composition: Vec<AgentComposition> = serde_wasm_bindgen::from_value(swarm_composition)
            .map_err(|e| JsValue::from_str(&format!("Invalid swarm composition: {}", e)))?;
        
        // Calculate pattern distribution
        let mut pattern_counts = HashMap::new();
        let total_agents = composition.len() as f32;
        
        for agent in &composition {
            *pattern_counts.entry(agent.cognitive_pattern.clone()).or_insert(0.0) += 1.0;
        }
        
        // Convert counts to percentages
        let pattern_distribution: HashMap<String, f32> = pattern_counts
            .into_iter()
            .map(|(pattern, count)| (pattern, count / total_agents))
            .collect();
        
        // Calculate diversity score using Shannon diversity index
        let diversity_score = self.calculate_shannon_diversity(&pattern_distribution);
        
        // Calculate interaction balance
        let interaction_balance = self.calculate_interaction_balance(&composition);
        
        // Calculate redundancy factor
        let redundancy_factor = self.calculate_redundancy_factor(&composition);
        
        // Calculate coverage score
        let coverage_score = self.calculate_coverage_score(&pattern_distribution);
        
        self.diversity_metrics = DiversityMetrics {
            overall_diversity_score: diversity_score,
            pattern_distribution,
            interaction_balance,
            redundancy_factor,
            coverage_score,
        };
        
        let analysis = serde_json::json!({
            "diversity_metrics": self.diversity_metrics,
            "recommendations": self.generate_diversity_recommendations(),
            "optimal_additions": self.suggest_optimal_additions(&composition),
            "risk_assessment": self.assess_diversity_risks()
        });
        
        Ok(serde_wasm_bindgen::to_value(&analysis).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn recommend_cognitive_pattern(&self, task_requirements: JsValue, current_swarm: JsValue) -> Result<JsValue, JsValue> {
        let requirements: TaskRequirements = serde_wasm_bindgen::from_value(task_requirements)
            .map_err(|e| JsValue::from_str(&format!("Invalid task requirements: {}", e)))?;
        
        let swarm: Vec<AgentComposition> = serde_wasm_bindgen::from_value(current_swarm)
            .map_err(|e| JsValue::from_str(&format!("Invalid swarm composition: {}", e)))?;
        
        // Analyze task requirements and match with cognitive patterns
        let mut pattern_scores = HashMap::new();
        
        for (pattern_name, pattern) in &self.patterns {
            let mut score = 0.0;
            
            // Score based on task type alignment
            for optimal_task in &pattern.optimal_tasks {
                if requirements.task_type.contains(optimal_task) {
                    score += 2.0;
                }
            }
            
            // Score based on required capabilities
            for capability in &requirements.required_capabilities {
                if pattern.strengths.contains(capability) {
                    score += 1.5;
                }
            }
            
            // Adjust score based on current swarm composition (diversity bonus)
            let current_pattern_count = swarm.iter()
                .filter(|agent| agent.cognitive_pattern == *pattern_name)
                .count() as f32;
            
            let diversity_bonus = if current_pattern_count == 0.0 {
                1.5 // Bonus for adding new pattern
            } else if current_pattern_count < swarm.len() as f32 * 0.3 {
                1.0 // Neutral for balanced representation
            } else {
                0.5 // Penalty for overrepresentation
            };
            
            score *= diversity_bonus;
            pattern_scores.insert(pattern_name.clone(), score);
        }
        
        // Find best pattern
        let best_pattern = pattern_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(name, score)| (name.clone(), *score));
        
        let recommendation = serde_json::json!({
            "recommended_pattern": best_pattern.as_ref().map(|(name, _)| name),
            "confidence_score": best_pattern.as_ref().map(|(_, score)| score).unwrap_or(0.0),
            "pattern_scores": pattern_scores,
            "reasoning": self.explain_pattern_recommendation(&requirements, best_pattern.as_ref().map(|(name, _)| name.as_str()).unwrap_or("convergent")),
            "alternative_patterns": self.get_alternative_patterns(&pattern_scores, 3)
        });
        
        Ok(serde_wasm_bindgen::to_value(&recommendation).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn optimize_swarm_composition(&self, current_swarm: JsValue, optimization_goals: JsValue) -> Result<JsValue, JsValue> {
        let swarm: Vec<AgentComposition> = serde_wasm_bindgen::from_value(current_swarm)
            .map_err(|e| JsValue::from_str(&format!("Invalid swarm composition: {}", e)))?;
        
        let goals: OptimizationGoals = serde_wasm_bindgen::from_value(optimization_goals)
            .map_err(|e| JsValue::from_str(&format!("Invalid optimization goals: {}", e)))?;
        
        let current_metrics = self.calculate_composition_metrics(&swarm);
        let optimization_plan = self.generate_optimization_plan(&swarm, &goals, &current_metrics);
        
        let result = serde_json::json!({
            "current_metrics": current_metrics,
            "optimization_plan": optimization_plan,
            "expected_improvements": self.calculate_expected_improvements(&optimization_plan),
            "implementation_steps": self.generate_implementation_steps(&optimization_plan)
        });
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
    
    fn initialize_cognitive_patterns(&mut self) {
        // Convergent Thinking Pattern
        self.patterns.insert("convergent".to_string(), CognitivePattern {
            name: "Convergent Thinking".to_string(),
            description: "Focused, analytical problem-solving with emphasis on finding optimal solutions".to_string(),
            processing_style: ProcessingStyle {
                focus_type: "narrow".to_string(),
                decision_speed: "deliberate".to_string(),
                risk_tolerance: "conservative".to_string(),
                information_processing: "sequential".to_string(),
            },
            neural_config: NeuralConfiguration {
                architecture_type: "feedforward".to_string(),
                layer_sizes: vec![10, 64, 32, 5],
                activation_functions: vec!["relu".to_string(), "relu".to_string(), "sigmoid".to_string()],
                learning_parameters: LearningParameters {
                    learning_rate: 0.001,
                    momentum: 0.9,
                    adaptation_rate: 0.1,
                    memory_retention: 0.95,
                    exploration_factor: 0.1,
                },
                specialized_modules: vec!["optimization".to_string(), "pattern_matching".to_string()],
            },
            strengths: vec!["optimization".to_string(), "analytical_reasoning".to_string(), "systematic_approach".to_string()],
            weaknesses: vec!["creativity".to_string(), "adaptability".to_string(), "novel_solutions".to_string()],
            optimal_tasks: vec!["optimization".to_string(), "debugging".to_string(), "quality_assurance".to_string()],
            interaction_weights: HashMap::new(),
        });
        
        // Divergent Thinking Pattern
        self.patterns.insert("divergent".to_string(), CognitivePattern {
            name: "Divergent Thinking".to_string(),
            description: "Creative, exploratory thinking with emphasis on generating multiple solutions".to_string(),
            processing_style: ProcessingStyle {
                focus_type: "broad".to_string(),
                decision_speed: "fast".to_string(),
                risk_tolerance: "aggressive".to_string(),
                information_processing: "parallel".to_string(),
            },
            neural_config: NeuralConfiguration {
                architecture_type: "recurrent".to_string(),
                layer_sizes: vec![10, 128, 64, 32, 5],
                activation_functions: vec!["sigmoid".to_string(), "tanh".to_string(), "sigmoid".to_string(), "sigmoid".to_string()],
                learning_parameters: LearningParameters {
                    learning_rate: 0.01,
                    momentum: 0.7,
                    adaptation_rate: 0.3,
                    memory_retention: 0.8,
                    exploration_factor: 0.4,
                },
                specialized_modules: vec!["creativity".to_string(), "pattern_generation".to_string(), "ideation".to_string()],
            },
            strengths: vec!["creativity".to_string(), "brainstorming".to_string(), "alternative_solutions".to_string()],
            weaknesses: vec!["focus".to_string(), "optimization".to_string(), "consistency".to_string()],
            optimal_tasks: vec!["research".to_string(), "ideation".to_string(), "exploration".to_string()],
            interaction_weights: HashMap::new(),
        });
        
        // Systems Thinking Pattern
        self.patterns.insert("systems".to_string(), CognitivePattern {
            name: "Systems Thinking".to_string(),
            description: "Holistic approach focusing on relationships and emergent properties".to_string(),
            processing_style: ProcessingStyle {
                focus_type: "adaptive".to_string(),
                decision_speed: "balanced".to_string(),
                risk_tolerance: "moderate".to_string(),
                information_processing: "hybrid".to_string(),
            },
            neural_config: NeuralConfiguration {
                architecture_type: "modular".to_string(),
                layer_sizes: vec![10, 96, 48, 5],
                activation_functions: vec!["tanh".to_string(), "relu".to_string(), "sigmoid".to_string()],
                learning_parameters: LearningParameters {
                    learning_rate: 0.005,
                    momentum: 0.85,
                    adaptation_rate: 0.2,
                    memory_retention: 0.9,
                    exploration_factor: 0.25,
                },
                specialized_modules: vec!["integration".to_string(), "relationship_mapping".to_string()],
            },
            strengths: vec!["holistic_analysis".to_string(), "integration".to_string(), "complexity_management".to_string()],
            weaknesses: vec!["detail_focus".to_string(), "quick_decisions".to_string()],
            optimal_tasks: vec!["architecture".to_string(), "coordination".to_string(), "planning".to_string()],
            interaction_weights: HashMap::new(),
        });
        
        // Critical Thinking Pattern
        self.patterns.insert("critical".to_string(), CognitivePattern {
            name: "Critical Thinking".to_string(),
            description: "Evaluative reasoning with emphasis on analysis and validation".to_string(),
            processing_style: ProcessingStyle {
                focus_type: "narrow".to_string(),
                decision_speed: "deliberate".to_string(),
                risk_tolerance: "conservative".to_string(),
                information_processing: "sequential".to_string(),
            },
            neural_config: NeuralConfiguration {
                architecture_type: "feedforward".to_string(),
                layer_sizes: vec![10, 80, 40, 5],
                activation_functions: vec!["relu".to_string(), "relu".to_string(), "sigmoid".to_string()],
                learning_parameters: LearningParameters {
                    learning_rate: 0.003,
                    momentum: 0.88,
                    adaptation_rate: 0.15,
                    memory_retention: 0.92,
                    exploration_factor: 0.15,
                },
                specialized_modules: vec!["validation".to_string(), "error_detection".to_string()],
            },
            strengths: vec!["critical_evaluation".to_string(), "error_detection".to_string(), "validation".to_string()],
            weaknesses: vec!["innovation".to_string(), "speed".to_string()],
            optimal_tasks: vec!["testing".to_string(), "review".to_string(), "analysis".to_string()],
            interaction_weights: HashMap::new(),
        });
        
        // Lateral Thinking Pattern
        self.patterns.insert("lateral".to_string(), CognitivePattern {
            name: "Lateral Thinking".to_string(),
            description: "Innovative problem-solving through indirect approaches".to_string(),
            processing_style: ProcessingStyle {
                focus_type: "broad".to_string(),
                decision_speed: "fast".to_string(),
                risk_tolerance: "aggressive".to_string(),
                information_processing: "parallel".to_string(),
            },
            neural_config: NeuralConfiguration {
                architecture_type: "hybrid".to_string(),
                layer_sizes: vec![10, 112, 56, 28, 5],
                activation_functions: vec!["sigmoid".to_string(), "tanh".to_string(), "relu".to_string(), "sigmoid".to_string()],
                learning_parameters: LearningParameters {
                    learning_rate: 0.015,
                    momentum: 0.75,
                    adaptation_rate: 0.35,
                    memory_retention: 0.75,
                    exploration_factor: 0.5,
                },
                specialized_modules: vec!["innovation".to_string(), "lateral_connections".to_string()],
            },
            strengths: vec!["innovative_solutions".to_string(), "unconventional_approaches".to_string()],
            weaknesses: vec!["consistency".to_string(), "direct_solutions".to_string()],
            optimal_tasks: vec!["innovation".to_string(), "problem_solving".to_string(), "breakthrough".to_string()],
            interaction_weights: HashMap::new(),
        });
    }
    
    fn initialize_pattern_interactions(&mut self) {
        // Define how different cognitive patterns interact
        let interactions = vec![
            PatternInteraction {
                pattern_a: "convergent".to_string(),
                pattern_b: "divergent".to_string(),
                interaction_type: "complementary".to_string(),
                interaction_strength: 0.8,
                optimal_ratio: 0.6, // 60% convergent, 40% divergent
            },
            PatternInteraction {
                pattern_a: "systems".to_string(),
                pattern_b: "critical".to_string(),
                interaction_type: "synergistic".to_string(),
                interaction_strength: 0.9,
                optimal_ratio: 0.5, // Equal representation
            },
            PatternInteraction {
                pattern_a: "lateral".to_string(),
                pattern_b: "convergent".to_string(),
                interaction_type: "complementary".to_string(),
                interaction_strength: 0.7,
                optimal_ratio: 0.3, // 30% lateral, 70% convergent
            },
            PatternInteraction {
                pattern_a: "divergent".to_string(),
                pattern_b: "critical".to_string(),
                interaction_type: "complementary".to_string(),
                interaction_strength: 0.75,
                optimal_ratio: 0.4, // 40% divergent, 60% critical
            },
            PatternInteraction {
                pattern_a: "systems".to_string(),
                pattern_b: "lateral".to_string(),
                interaction_type: "synergistic".to_string(),
                interaction_strength: 0.85,
                optimal_ratio: 0.6, // 60% systems, 40% lateral
            },
        ];
        
        for interaction in interactions {
            self.pattern_interactions
                .entry(interaction.pattern_a.clone())
                .or_insert_with(Vec::new)
                .push(interaction.clone());
            
            // Add reverse interaction
            let reverse = PatternInteraction {
                pattern_a: interaction.pattern_b,
                pattern_b: interaction.pattern_a,
                interaction_type: interaction.interaction_type,
                interaction_strength: interaction.interaction_strength,
                optimal_ratio: 1.0 - interaction.optimal_ratio,
            };
            
            self.pattern_interactions
                .entry(reverse.pattern_a.clone())
                .or_insert_with(Vec::new)
                .push(reverse);
        }
    }
    
    fn calculate_shannon_diversity(&self, distribution: &HashMap<String, f32>) -> f32 {
        let mut diversity = 0.0;
        for proportion in distribution.values() {
            if *proportion > 0.0 {
                diversity -= proportion * proportion.ln();
            }
        }
        diversity
    }
    
    fn calculate_interaction_balance(&self, composition: &[AgentComposition]) -> f32 {
        let mut interaction_score = 0.0;
        let mut total_interactions = 0;
        
        for i in 0..composition.len() {
            for j in i+1..composition.len() {
                let pattern_a = &composition[i].cognitive_pattern;
                let pattern_b = &composition[j].cognitive_pattern;
                
                if let Some(interactions) = self.pattern_interactions.get(pattern_a) {
                    for interaction in interactions {
                        if &interaction.pattern_b == pattern_b {
                            interaction_score += interaction.interaction_strength;
                            total_interactions += 1;
                        }
                    }
                }
            }
        }
        
        if total_interactions > 0 {
            interaction_score / total_interactions as f32
        } else {
            0.0
        }
    }
    
    fn calculate_redundancy_factor(&self, composition: &[AgentComposition]) -> f32 {
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        for agent in composition {
            *pattern_counts.entry(agent.cognitive_pattern.clone()).or_insert(0) += 1;
        }
        
        let max_count = pattern_counts.values().max().copied().unwrap_or(0) as f32;
        let total = composition.len() as f32;
        
        if total > 0.0 {
            1.0 - (max_count / total)
        } else {
            0.0
        }
    }
    
    fn calculate_coverage_score(&self, distribution: &HashMap<String, f32>) -> f32 {
        let covered_patterns = distribution.len() as f32;
        let total_patterns = self.patterns.len() as f32;
        
        if total_patterns > 0.0 {
            covered_patterns / total_patterns
        } else {
            0.0
        }
    }
    
    fn generate_diversity_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if self.diversity_metrics.overall_diversity_score < 1.0 {
            recommendations.push("Consider adding agents with underrepresented cognitive patterns".to_string());
        }
        
        if self.diversity_metrics.redundancy_factor < 0.5 {
            recommendations.push("High redundancy detected - diversify agent cognitive patterns".to_string());
        }
        
        if self.diversity_metrics.coverage_score < 0.6 {
            recommendations.push("Low pattern coverage - add agents with missing cognitive patterns".to_string());
        }
        
        recommendations
    }
    
    fn suggest_optimal_additions(&self, composition: &[AgentComposition]) -> Vec<String> {
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        for agent in composition {
            *pattern_counts.entry(agent.cognitive_pattern.clone()).or_insert(0) += 1;
        }
        
        let mut suggestions = Vec::new();
        for pattern in self.patterns.keys() {
            if !pattern_counts.contains_key(pattern) {
                suggestions.push(pattern.clone());
            }
        }
        
        suggestions
    }
    
    fn assess_diversity_risks(&self) -> serde_json::Value {
        let mut risks = Vec::new();
        
        if self.diversity_metrics.overall_diversity_score < 0.5 {
            risks.push(serde_json::json!({
                "risk": "Low diversity",
                "severity": "high",
                "impact": "Reduced problem-solving capability"
            }));
        }
        
        if self.diversity_metrics.interaction_balance < 0.3 {
            risks.push(serde_json::json!({
                "risk": "Poor interaction balance",
                "severity": "medium",
                "impact": "Suboptimal agent collaboration"
            }));
        }
        
        serde_json::json!(risks)
    }
    
    fn explain_pattern_recommendation(&self, requirements: &TaskRequirements, pattern_name: &str) -> String {
        format!(
            "Pattern '{}' recommended for task type '{}' based on required capabilities and current swarm diversity",
            pattern_name, requirements.task_type
        )
    }
    
    fn get_alternative_patterns(&self, scores: &HashMap<String, f32>, count: usize) -> Vec<(String, f32)> {
        let mut sorted_patterns: Vec<(String, f32)> = scores
            .iter()
            .map(|(name, score)| (name.clone(), *score))
            .collect();
        
        sorted_patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted_patterns.into_iter().skip(1).take(count).collect()
    }
    
    fn calculate_composition_metrics(&self, swarm: &[AgentComposition]) -> serde_json::Value {
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        for agent in swarm {
            *pattern_counts.entry(agent.cognitive_pattern.clone()).or_insert(0) += 1;
        }
        
        serde_json::json!({
            "total_agents": swarm.len(),
            "pattern_distribution": pattern_counts,
            "diversity_score": self.diversity_metrics.overall_diversity_score,
            "coverage_score": self.diversity_metrics.coverage_score
        })
    }
    
    fn generate_optimization_plan(&self, _swarm: &[AgentComposition], goals: &OptimizationGoals, _metrics: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "target_diversity": goals.target_diversity_score,
            "preferred_patterns": goals.preferred_patterns,
            "suggested_changes": "Add agents with underrepresented patterns",
            "priority": "balance diversity while maintaining performance"
        })
    }
    
    fn calculate_expected_improvements(&self, _plan: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "diversity_improvement": "+0.3",
            "coverage_improvement": "+0.2",
            "interaction_balance_improvement": "+0.15"
        })
    }
    
    fn generate_implementation_steps(&self, _plan: &serde_json::Value) -> Vec<String> {
        vec![
            "Identify underrepresented cognitive patterns".to_string(),
            "Spawn new agents with selected patterns".to_string(),
            "Monitor diversity metrics after each addition".to_string(),
            "Adjust based on performance feedback".to_string(),
        ]
    }
}

#[derive(Serialize, Deserialize)]
pub struct AgentComposition {
    pub agent_id: String,
    pub agent_type: String,
    pub cognitive_pattern: String,
    pub capabilities: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct TaskRequirements {
    pub task_type: String,
    pub required_capabilities: Vec<String>,
    pub complexity_level: String,
    pub time_constraints: Option<f64>,
    pub quality_requirements: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct OptimizationGoals {
    pub target_diversity_score: f32,
    pub preferred_patterns: Vec<String>,
    pub performance_priorities: Vec<String>,
    pub constraints: Vec<String>,
}