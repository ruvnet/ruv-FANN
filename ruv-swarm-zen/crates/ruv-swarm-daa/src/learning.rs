//! Learning engine for autonomous agent learning and adaptation

use crate::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Learning engine for coordinating agent learning across domains
pub struct LearningEngine {
    learning_models: HashMap<String, DomainLearningModel>,
    global_knowledge: Arc<RwLock<GlobalKnowledgeBase>>,
    adaptation_strategies: Vec<AdaptationStrategy>,
    meta_learning_state: MetaLearningState,
}

/// Domain-specific learning model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainLearningModel {
    pub domain: String,
    pub proficiency_scores: HashMap<String, f64>,
    pub learning_patterns: Vec<LearningPattern>,
    pub knowledge_items: Vec<KnowledgeItem>,
    pub adaptation_history: Vec<AdaptationRecord>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Global knowledge base shared across agents
pub struct GlobalKnowledgeBase {
    pub knowledge_graph: HashMap<String, Vec<String>>, // concept -> related concepts
    pub concept_weights: HashMap<String, f64>,
    pub cross_domain_mappings: HashMap<String, Vec<String>>,
    pub shared_experiences: Vec<SharedExperience>,
}

/// Learning pattern structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPattern {
    pub id: String,
    pub pattern_type: LearningPatternType,
    pub effectiveness: f64,
    pub applicability_domains: Vec<String>,
    pub usage_count: u32,
    pub discovered_at: chrono::DateTime<chrono::Utc>,
}

/// Types of learning patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningPatternType {
    /// Incremental learning from examples
    Incremental,
    /// Transfer learning between domains
    Transfer,
    /// Meta-learning for rapid adaptation
    MetaLearning,
    /// Reinforcement learning from feedback
    Reinforcement,
    /// Collaborative learning from peers
    Collaborative,
    /// Self-supervised learning from data
    SelfSupervised,
}

/// Adaptation strategy for learning optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStrategy {
    pub name: String,
    pub strategy_type: AdaptationStrategyType,
    pub effectiveness_scores: HashMap<String, f64>, // domain -> effectiveness
    pub parameters: HashMap<String, f64>,
    pub usage_history: Vec<StrategyUsage>,
}

/// Types of adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategyType {
    /// Adjust learning rate based on performance
    AdaptiveLearningRate,
    /// Change cognitive pattern based on task type
    CognitivePatternSwitch,
    /// Transfer knowledge from similar domains
    KnowledgeTransfer,
    /// Ensemble multiple learning approaches
    EnsembleLearning,
    /// Active learning with uncertainty sampling
    ActiveLearning,
}

/// Record of adaptation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trigger_event: String,
    pub strategy_applied: String,
    pub before_metrics: HashMap<String, f64>,
    pub after_metrics: HashMap<String, f64>,
    pub success_score: f64,
}

/// Strategy usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyUsage {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub domain: String,
    pub context: HashMap<String, serde_json::Value>,
    pub outcome_score: f64,
}

/// Meta-learning state for rapid adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningState {
    pub meta_parameters: HashMap<String, f64>,
    pub rapid_adaptation_models: HashMap<String, RapidAdaptationModel>,
    pub transfer_learning_mappings: HashMap<String, Vec<String>>,
    pub learning_to_learn_progress: f64,
}

/// Rapid adaptation model for quick domain transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RapidAdaptationModel {
    pub base_domain: String,
    pub target_domains: Vec<String>,
    pub adaptation_parameters: HashMap<String, f64>,
    pub few_shot_examples: Vec<Experience>,
    pub performance_metrics: HashMap<String, f64>,
}

/// Shared experience between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedExperience {
    pub experience: Experience,
    pub sharing_agent: String,
    pub receiving_agents: Vec<String>,
    pub effectiveness_scores: HashMap<String, f64>,
    pub shared_at: chrono::DateTime<chrono::Utc>,
}

impl Default for LearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningEngine {
    /// Create a new learning engine
    pub fn new() -> Self {
        Self {
            learning_models: HashMap::new(),
            global_knowledge: Arc::new(RwLock::new(GlobalKnowledgeBase::new())),
            adaptation_strategies: Self::initialize_adaptation_strategies(),
            meta_learning_state: MetaLearningState::new(),
        }
    }

    /// Initialize meta-learning capabilities (coordinator compatibility)
    pub async fn initialize_meta_learning(&mut self) -> DAAResult<()> {
        self.meta_learning_state.learning_to_learn_progress = 0.1;
        tracing::info!("Initialized meta-learning capabilities");
        Ok(())
    }

    /// Adapt from coordination events (coordinator compatibility)
    pub async fn adapt_from_events(
        &mut self,
        events: &[crate::CoordinationEvent],
    ) -> DAAResult<()> {
        for event in events {
            // Extract learning signals from coordination events
            if let Some(participants) = events.first().map(|e| &e.participants) {
                for participant in participants {
                    // Update learning models based on coordination outcomes
                    if !self.learning_models.contains_key(participant) {
                        self.learning_models.insert(
                            participant.clone(),
                            DomainLearningModel::new("coordination".to_string()),
                        );
                    }
                }
            }
        }

        tracing::info!("Adapted learning from {} coordination events", events.len());
        Ok(())
    }

    /// Get patterns count (coordinator compatibility)
    pub async fn get_patterns_count(&self) -> DAAResult<usize> {
        let total_patterns: usize = self
            .learning_models
            .values()
            .map(|model| model.learning_patterns.len())
            .sum();
        Ok(total_patterns)
    }

    /// Add learning experience for an agent
    pub async fn add_learning_experience(
        &mut self,
        agent_id: &str,
        domain: &str,
        experience: &Experience,
    ) -> DAAResult<()> {
        // Get or create domain learning model
        let model_key = format!("{}:{}", agent_id, domain);
        if !self.learning_models.contains_key(&model_key) {
            self.learning_models.insert(
                model_key.clone(),
                DomainLearningModel::new(domain.to_string()),
            );
        }

        // Extract learning patterns from experience before getting mutable reference
        let patterns = self.extract_learning_patterns(experience).await?;

        // Calculate proficiency updates based on experience
        let task_type = experience.task.description.clone();
        let performance = if experience.result.success { 1.0 } else { 0.0 };
        let learning_rate = 0.1;

        // Now get mutable reference to model and update it
        let model = self.learning_models.get_mut(&model_key).unwrap();
        model.learning_patterns.extend(patterns);

        // Update proficiency scores inline to avoid borrowing issues
        let current_score = model
            .proficiency_scores
            .get(&task_type)
            .copied()
            .unwrap_or(0.5);
        let new_score = current_score + learning_rate * (performance - current_score);
        model.proficiency_scores.insert(task_type, new_score);

        // Add to global knowledge base
        let mut global_kb = self.global_knowledge.write().await;
        global_kb
            .add_shared_experience(SharedExperience {
                experience: experience.clone(),
                sharing_agent: agent_id.to_string(),
                receiving_agents: Vec::new(),
                effectiveness_scores: HashMap::new(),
                shared_at: chrono::Utc::now(),
            })
            .await?;

        model.last_updated = chrono::Utc::now();

        tracing::info!(
            "Added learning experience for agent {} in domain {}",
            agent_id,
            domain
        );
        Ok(())
    }

    /// Get learning recommendations for an agent
    pub async fn get_learning_recommendations(
        &self,
        agent_id: &str,
        current_domain: &str,
        target_task: &Task,
    ) -> DAAResult<Vec<LearningRecommendation>> {
        let mut recommendations = Vec::new();

        // Analyze current proficiency
        let model_key = format!("{}:{}", agent_id, current_domain);
        let current_proficiency = self
            .learning_models
            .get(&model_key)
            .map(|model| self.calculate_task_proficiency(model, target_task))
            .unwrap_or(0.0);

        // If proficiency is low, recommend specific learning strategies
        if current_proficiency < 0.7 {
            recommendations.push(LearningRecommendation {
                recommendation_type: RecommendationType::IncreaseTraining,
                priority: Priority::High,
                description: "Increase training in current domain".to_string(),
                estimated_improvement: 0.2,
                estimated_time_hours: 2.0,
            });
        }

        // Check for transfer learning opportunities
        let transfer_opportunities = self
            .find_transfer_opportunities(agent_id, current_domain, target_task)
            .await?;

        for opportunity in transfer_opportunities {
            recommendations.push(LearningRecommendation {
                recommendation_type: RecommendationType::TransferLearning,
                priority: Priority::Medium,
                description: format!("Transfer knowledge from {}", opportunity.source_domain),
                estimated_improvement: opportunity.expected_improvement,
                estimated_time_hours: 0.5,
            });
        }

        // Meta-learning recommendations
        if self.meta_learning_state.learning_to_learn_progress < 0.8 {
            recommendations.push(LearningRecommendation {
                recommendation_type: RecommendationType::MetaLearning,
                priority: Priority::Medium,
                description: "Improve meta-learning capabilities".to_string(),
                estimated_improvement: 0.15,
                estimated_time_hours: 1.0,
            });
        }

        Ok(recommendations)
    }

    /// Apply adaptation strategy based on performance
    pub async fn apply_adaptation_strategy(
        &mut self,
        agent_id: &str,
        domain: &str,
        performance_metrics: &HashMap<String, f64>,
    ) -> DAAResult<AdaptationResult> {
        // Analyze performance and select best strategy
        let best_strategy = self
            .select_adaptation_strategy(domain, performance_metrics)
            .await?;

        // Apply the strategy
        let adaptation_result = match best_strategy.strategy_type {
            AdaptationStrategyType::AdaptiveLearningRate => {
                self.apply_learning_rate_adaptation(agent_id, domain, performance_metrics)
                    .await?
            }
            AdaptationStrategyType::CognitivePatternSwitch => {
                self.apply_cognitive_pattern_switch(agent_id, domain, performance_metrics)
                    .await?
            }
            AdaptationStrategyType::KnowledgeTransfer => {
                self.apply_knowledge_transfer(agent_id, domain, performance_metrics)
                    .await?
            }
            AdaptationStrategyType::EnsembleLearning => {
                self.apply_ensemble_learning(agent_id, domain, performance_metrics)
                    .await?
            }
            AdaptationStrategyType::ActiveLearning => {
                self.apply_active_learning(agent_id, domain, performance_metrics)
                    .await?
            }
        };

        // Clone strategy name before mutable borrow
        let strategy_name = best_strategy.name.clone();

        // Record adaptation event
        let model_key = format!("{}:{}", agent_id, domain);
        if let Some(model) = self.learning_models.get_mut(&model_key) {
            model.adaptation_history.push(AdaptationRecord {
                timestamp: chrono::Utc::now(),
                trigger_event: "Performance-based adaptation".to_string(),
                strategy_applied: strategy_name,
                before_metrics: performance_metrics.clone(),
                after_metrics: adaptation_result.updated_metrics.clone(),
                success_score: adaptation_result.improvement_score,
            });
        }

        Ok(adaptation_result)
    }

    /// Get cross-agent knowledge sharing opportunities
    pub async fn get_knowledge_sharing_opportunities(
        &self,
        requesting_agent: &str,
        domain: &str,
    ) -> DAAResult<Vec<KnowledgeSharingOpportunity>> {
        let mut opportunities = Vec::new();
        let _global_kb = self.global_knowledge.read().await;

        // Find agents with high proficiency in the domain
        for (model_key, model) in &self.learning_models {
            if model.domain == domain && !model_key.starts_with(requesting_agent) {
                let agent_id = model_key.split(':').next().unwrap_or("");
                let avg_proficiency: f64 = model.proficiency_scores.values().sum::<f64>()
                    / model.proficiency_scores.len() as f64;

                if avg_proficiency > 0.8 {
                    opportunities.push(KnowledgeSharingOpportunity {
                        source_agent: agent_id.to_string(),
                        domain: domain.to_string(),
                        knowledge_quality: avg_proficiency,
                        compatibility_score: self
                            .calculate_agent_compatibility(requesting_agent, agent_id)
                            .await
                            .unwrap_or(0.5),
                        estimated_benefit: avg_proficiency * 0.3,
                    });
                }
            }
        }

        // Sort by potential benefit (handle NaN values safely)
        opportunities.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(opportunities)
    }

    /// Initialize default adaptation strategies
    fn initialize_adaptation_strategies() -> Vec<AdaptationStrategy> {
        vec![
            AdaptationStrategy {
                name: "Adaptive Learning Rate".to_string(),
                strategy_type: AdaptationStrategyType::AdaptiveLearningRate,
                effectiveness_scores: HashMap::new(),
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("initial_rate".to_string(), 0.001);
                    params.insert("decay_factor".to_string(), 0.95);
                    params.insert("min_rate".to_string(), 0.0001);
                    params
                },
                usage_history: Vec::new(),
            },
            AdaptationStrategy {
                name: "Cognitive Pattern Switch".to_string(),
                strategy_type: AdaptationStrategyType::CognitivePatternSwitch,
                effectiveness_scores: HashMap::new(),
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("switch_threshold".to_string(), 0.6);
                    params.insert("exploration_factor".to_string(), 0.2);
                    params
                },
                usage_history: Vec::new(),
            },
            AdaptationStrategy {
                name: "Knowledge Transfer".to_string(),
                strategy_type: AdaptationStrategyType::KnowledgeTransfer,
                effectiveness_scores: HashMap::new(),
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("similarity_threshold".to_string(), 0.7);
                    params.insert("transfer_weight".to_string(), 0.5);
                    params
                },
                usage_history: Vec::new(),
            },
        ]
    }

    /// Extract learning patterns from experience
    async fn extract_learning_patterns(
        &self,
        experience: &Experience,
    ) -> DAAResult<Vec<LearningPattern>> {
        let mut patterns = Vec::new();

        // Analyze task success patterns
        if experience.result.success {
            patterns.push(LearningPattern {
                id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearningPatternType::Incremental,
                effectiveness: experience
                    .result
                    .performance_metrics
                    .get("accuracy")
                    .copied()
                    .unwrap_or(0.8),
                applicability_domains: vec![experience.task.description.clone()],
                usage_count: 1,
                discovered_at: chrono::Utc::now(),
            });
        }

        // Analyze execution time patterns
        if experience.result.execution_time_ms < 1000 {
            patterns.push(LearningPattern {
                id: uuid::Uuid::new_v4().to_string(),
                pattern_type: LearningPatternType::SelfSupervised,
                effectiveness: 0.9,
                applicability_domains: vec!["fast_execution".to_string()],
                usage_count: 1,
                discovered_at: chrono::Utc::now(),
            });
        }

        Ok(patterns)
    }

    /// Update proficiency scores based on experience
    async fn update_proficiency_scores(
        &self,
        model: &mut DomainLearningModel,
        experience: &Experience,
    ) -> DAAResult<()> {
        let task_type = experience.task.description.clone();
        let performance = if experience.result.success { 1.0 } else { 0.0 };

        // Update or create proficiency score
        let current_score = model
            .proficiency_scores
            .get(&task_type)
            .copied()
            .unwrap_or(0.5);
        let learning_rate = 0.1;
        let new_score = current_score + learning_rate * (performance - current_score);

        model.proficiency_scores.insert(task_type, new_score);

        Ok(())
    }

    /// Calculate task-specific proficiency
    fn calculate_task_proficiency(&self, model: &DomainLearningModel, task: &Task) -> f64 {
        model
            .proficiency_scores
            .get(&task.description)
            .copied()
            .unwrap_or(0.0)
    }

    /// Find transfer learning opportunities
    async fn find_transfer_opportunities(
        &self,
        agent_id: &str,
        current_domain: &str,
        _task: &Task,
    ) -> DAAResult<Vec<TransferOpportunity>> {
        let mut opportunities = Vec::new();

        // Find domains with high proficiency
        for (model_key, model) in &self.learning_models {
            if model_key.starts_with(agent_id) && model.domain != current_domain {
                let avg_proficiency: f64 = model.proficiency_scores.values().sum::<f64>()
                    / model.proficiency_scores.len() as f64;

                if avg_proficiency > 0.7 {
                    opportunities.push(TransferOpportunity {
                        source_domain: model.domain.clone(),
                        target_domain: current_domain.to_string(),
                        similarity_score: 0.8, // Simplified
                        expected_improvement: avg_proficiency * 0.3,
                    });
                }
            }
        }

        Ok(opportunities)
    }

    /// Select best adaptation strategy
    async fn select_adaptation_strategy(
        &self,
        domain: &str,
        _performance_metrics: &HashMap<String, f64>,
    ) -> DAAResult<&AdaptationStrategy> {
        // Simple selection based on domain effectiveness
        let best_strategy = self
            .adaptation_strategies
            .iter()
            .max_by(|a, b| {
                let a_score = a.effectiveness_scores.get(domain).copied().unwrap_or(0.5);
                let b_score = b.effectiveness_scores.get(domain).copied().unwrap_or(0.5);
                // Handle potential NaN values in scores
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| DAAError::LearningError {
                message: "No adaptation strategies available".to_string(),
            })?;

        Ok(best_strategy)
    }

    /// Apply learning rate adaptation
    async fn apply_learning_rate_adaptation(
        &self,
        _agent_id: &str,
        _domain: &str,
        performance_metrics: &HashMap<String, f64>,
    ) -> DAAResult<AdaptationResult> {
        let mut updated_metrics = performance_metrics.clone();
        let improvement = 0.1;

        // Simulate learning rate adjustment effect
        if let Some(accuracy) = updated_metrics.get_mut("accuracy") {
            *accuracy += improvement;
        }

        Ok(AdaptationResult {
            strategy_applied: "Adaptive Learning Rate".to_string(),
            updated_metrics,
            improvement_score: improvement,
            success: true,
        })
    }

    /// Apply cognitive pattern switch
    async fn apply_cognitive_pattern_switch(
        &self,
        _agent_id: &str,
        _domain: &str,
        performance_metrics: &HashMap<String, f64>,
    ) -> DAAResult<AdaptationResult> {
        let mut updated_metrics = performance_metrics.clone();
        let improvement = 0.15;

        // Simulate cognitive pattern switch effect
        if let Some(efficiency) = updated_metrics.get_mut("efficiency") {
            *efficiency += improvement;
        }

        Ok(AdaptationResult {
            strategy_applied: "Cognitive Pattern Switch".to_string(),
            updated_metrics,
            improvement_score: improvement,
            success: true,
        })
    }

    /// Apply knowledge transfer
    async fn apply_knowledge_transfer(
        &self,
        _agent_id: &str,
        _domain: &str,
        performance_metrics: &HashMap<String, f64>,
    ) -> DAAResult<AdaptationResult> {
        let mut updated_metrics = performance_metrics.clone();
        let improvement = 0.2;

        // Simulate knowledge transfer effect
        if let Some(learning_speed) = updated_metrics.get_mut("learning_speed") {
            *learning_speed += improvement;
        }

        Ok(AdaptationResult {
            strategy_applied: "Knowledge Transfer".to_string(),
            updated_metrics,
            improvement_score: improvement,
            success: true,
        })
    }

    /// Apply ensemble learning
    async fn apply_ensemble_learning(
        &self,
        _agent_id: &str,
        _domain: &str,
        performance_metrics: &HashMap<String, f64>,
    ) -> DAAResult<AdaptationResult> {
        let mut updated_metrics = performance_metrics.clone();
        let improvement = 0.12;

        // Simulate ensemble learning effect
        if let Some(robustness) = updated_metrics.get_mut("robustness") {
            *robustness += improvement;
        }

        Ok(AdaptationResult {
            strategy_applied: "Ensemble Learning".to_string(),
            updated_metrics,
            improvement_score: improvement,
            success: true,
        })
    }

    /// Apply active learning
    async fn apply_active_learning(
        &self,
        _agent_id: &str,
        _domain: &str,
        performance_metrics: &HashMap<String, f64>,
    ) -> DAAResult<AdaptationResult> {
        let mut updated_metrics = performance_metrics.clone();
        let improvement = 0.18;

        // Simulate active learning effect
        if let Some(sample_efficiency) = updated_metrics.get_mut("sample_efficiency") {
            *sample_efficiency += improvement;
        }

        Ok(AdaptationResult {
            strategy_applied: "Active Learning".to_string(),
            updated_metrics,
            improvement_score: improvement,
            success: true,
        })
    }

    /// Calculate compatibility between agents
    async fn calculate_agent_compatibility(&self, _agent1: &str, _agent2: &str) -> DAAResult<f64> {
        // Simplified compatibility calculation
        // In a real implementation, this would consider cognitive patterns,
        // learning styles, domain expertise, etc.
        Ok(0.8)
    }
}

// Helper structures and implementations

/// Learning recommendation for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: Priority,
    pub description: String,
    pub estimated_improvement: f64,
    pub estimated_time_hours: f64,
}

/// Types of learning recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    IncreaseTraining,
    TransferLearning,
    MetaLearning,
    CollaborativeLearning,
    StrategyAdjustment,
}

/// Transfer learning opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferOpportunity {
    pub source_domain: String,
    pub target_domain: String,
    pub similarity_score: f64,
    pub expected_improvement: f64,
}

/// Knowledge sharing opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSharingOpportunity {
    pub source_agent: String,
    pub domain: String,
    pub knowledge_quality: f64,
    pub compatibility_score: f64,
    pub estimated_benefit: f64,
}

/// Result of adaptation application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    pub strategy_applied: String,
    pub updated_metrics: HashMap<String, f64>,
    pub improvement_score: f64,
    pub success: bool,
}

impl DomainLearningModel {
    pub fn new(domain: String) -> Self {
        Self {
            domain,
            proficiency_scores: HashMap::new(),
            learning_patterns: Vec::new(),
            knowledge_items: Vec::new(),
            adaptation_history: Vec::new(),
            last_updated: chrono::Utc::now(),
        }
    }
}

impl Default for GlobalKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalKnowledgeBase {
    pub fn new() -> Self {
        Self {
            knowledge_graph: HashMap::new(),
            concept_weights: HashMap::new(),
            cross_domain_mappings: HashMap::new(),
            shared_experiences: Vec::new(),
        }
    }

    pub async fn add_shared_experience(&mut self, experience: SharedExperience) -> DAAResult<()> {
        self.shared_experiences.push(experience);

        // Limit size to prevent memory issues
        if self.shared_experiences.len() > 10000 {
            self.shared_experiences.remove(0);
        }

        Ok(())
    }
}

impl Default for MetaLearningState {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaLearningState {
    pub fn new() -> Self {
        Self {
            meta_parameters: HashMap::new(),
            rapid_adaptation_models: HashMap::new(),
            transfer_learning_mappings: HashMap::new(),
            learning_to_learn_progress: 0.0,
        }
    }
}
