//! Self-adaptation and learning capabilities for autonomous agents

#[cfg(feature = "async")]
use async_trait::async_trait;

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

use crate::{
    types::{AdaptationFeedback, DecisionContext},
    DAAResult,
};

/// Self-adaptation capabilities for autonomous agents
#[cfg_attr(feature = "async", async_trait)]
pub trait SelfAdaptation: Send + Sync {
    /// Adaptation state type
    type AdaptationState: Send + Sync;

    /// Learning model type
    type LearningModel: Send + Sync;

    /// Adapt agent behavior based on feedback
    #[cfg(feature = "async")]
    async fn adapt(&mut self, feedback: &AdaptationFeedback) -> DAAResult<AdaptationResult>;

    /// Adapt (sync version)
    #[cfg(not(feature = "async"))]
    fn adapt(&mut self, feedback: &AdaptationFeedback) -> DAAResult<AdaptationResult>;

    /// Learn from experience
    #[cfg(feature = "async")]
    async fn learn(&mut self, experience: &Experience) -> DAAResult<LearningResult>;

    /// Learn (sync version)
    #[cfg(not(feature = "async"))]
    fn learn(&mut self, experience: &Experience) -> DAAResult<LearningResult>;

    /// Evaluate current adaptation state
    fn evaluate_adaptation(&self) -> AdaptationEvaluation;

    /// Get adaptation parameters
    fn adaptation_parameters(&self) -> &AdaptationParameters;

    /// Update adaptation parameters
    fn update_parameters(&mut self, parameters: AdaptationParameters);

    /// Check if adaptation is needed
    fn needs_adaptation(&self, feedback: &AdaptationFeedback) -> bool {
        feedback.performance_score < 0.7 || feedback.efficiency_score < 0.7
    }

    /// Get current learning model
    fn learning_model(&self) -> &Self::LearningModel;

    /// Update learning model
    #[cfg(feature = "async")]
    async fn update_learning_model(&mut self, model: Self::LearningModel) -> DAAResult<()>;

    /// Update learning model (sync version)
    #[cfg(not(feature = "async"))]
    fn update_learning_model(&mut self, model: Self::LearningModel) -> DAAResult<()>;
}

/// Learning strategy implementation
#[cfg_attr(feature = "async", async_trait)]
pub trait LearningStrategy: Send + Sync {
    /// Strategy type
    type Strategy: Send + Sync;

    /// Training data type
    type TrainingData: Send + Sync;

    /// Model type
    type Model: Send + Sync;

    /// Train the learning model
    #[cfg(feature = "async")]
    async fn train(&mut self, data: &Self::TrainingData) -> DAAResult<TrainingResult>;

    /// Train (sync version)
    #[cfg(not(feature = "async"))]
    fn train(&mut self, data: &Self::TrainingData) -> DAAResult<TrainingResult>;

    /// Predict using the trained model
    #[cfg(feature = "async")]
    async fn predict(&self, input: &DecisionContext) -> DAAResult<Prediction>;

    /// Predict (sync version)
    #[cfg(not(feature = "async"))]
    fn predict(&self, input: &DecisionContext) -> DAAResult<Prediction>;

    /// Update model with new data
    #[cfg(feature = "async")]
    async fn update_model(&mut self, data: &Self::TrainingData) -> DAAResult<()>;

    /// Update model (sync version)
    #[cfg(not(feature = "async"))]
    fn update_model(&mut self, data: &Self::TrainingData) -> DAAResult<()>;

    /// Evaluate model performance
    fn evaluate_model(&self, test_data: &Self::TrainingData) -> ModelEvaluation;

    /// Get strategy type
    fn strategy_type(&self) -> StrategyType;

    /// Get model metrics
    fn model_metrics(&self) -> ModelMetrics;
}

/// Evolutionary optimization for agent improvement
#[cfg_attr(feature = "async", async_trait)]
pub trait EvolutionaryOptimization: Send + Sync {
    /// Individual type for evolution
    type Individual: Send + Sync + Clone;

    /// Fitness type
    type Fitness: Send + Sync + PartialOrd;

    /// Initialize population
    fn initialize_population(&mut self, size: usize) -> DAAResult<Population<Self::Individual>>;

    /// Evaluate fitness of individuals
    #[cfg(feature = "async")]
    async fn evaluate_fitness(
        &self,
        individuals: &[Self::Individual],
    ) -> DAAResult<Vec<Self::Fitness>>;

    /// Evaluate fitness (sync version)
    #[cfg(not(feature = "async"))]
    fn evaluate_fitness(&self, individuals: &[Self::Individual]) -> DAAResult<Vec<Self::Fitness>>;

    /// Select parents for reproduction
    fn select_parents(
        &self,
        population: &Population<Self::Individual>,
        fitness: &[Self::Fitness],
    ) -> DAAResult<Vec<Self::Individual>>;

    /// Crossover operation
    fn crossover(
        &self,
        parent1: &Self::Individual,
        parent2: &Self::Individual,
    ) -> DAAResult<Vec<Self::Individual>>;

    /// Mutation operation
    fn mutate(&self, individual: &mut Self::Individual) -> DAAResult<()>;

    /// Run evolutionary algorithm
    #[cfg(feature = "async")]
    async fn evolve(&mut self, generations: usize) -> DAAResult<EvolutionResult<Self::Individual>>;

    /// Evolve (sync version)
    #[cfg(not(feature = "async"))]
    fn evolve(&mut self, generations: usize) -> DAAResult<EvolutionResult<Self::Individual>>;

    /// Get evolution parameters
    fn evolution_parameters(&self) -> &EvolutionParameters;
}

/// Reinforcement learning implementation
#[cfg_attr(feature = "async", async_trait)]
pub trait ReinforcementLearning: Send + Sync {
    /// State type
    type State: Send + Sync;

    /// Action type
    type Action: Send + Sync;

    /// Reward type
    type Reward: Send + Sync;

    /// Choose action based on current state
    #[cfg(feature = "async")]
    async fn choose_action(&mut self, state: &Self::State) -> DAAResult<Self::Action>;

    /// Choose action (sync version)
    #[cfg(not(feature = "async"))]
    fn choose_action(&mut self, state: &Self::State) -> DAAResult<Self::Action>;

    /// Update Q-values or policy based on experience
    #[cfg(feature = "async")]
    async fn update(
        &mut self,
        state: &Self::State,
        action: &Self::Action,
        reward: &Self::Reward,
        next_state: &Self::State,
    ) -> DAAResult<()>;

    /// Update (sync version)
    #[cfg(not(feature = "async"))]
    fn update(
        &mut self,
        state: &Self::State,
        action: &Self::Action,
        reward: &Self::Reward,
        next_state: &Self::State,
    ) -> DAAResult<()>;

    /// Get action value estimate
    fn get_action_value(&self, state: &Self::State, action: &Self::Action) -> f64;

    /// Get current policy
    fn get_policy(&self, state: &Self::State) -> DAAResult<ActionProbabilities<Self::Action>>;

    /// Set exploration rate
    fn set_exploration_rate(&mut self, rate: f64);

    /// Get exploration rate
    fn exploration_rate(&self) -> f64;

    /// Set learning rate
    fn set_learning_rate(&mut self, rate: f64);

    /// Get learning rate
    fn learning_rate(&self) -> f64;
}

/// Adaptation result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    /// Adaptation success
    pub success: bool,

    /// Adaptation type performed
    pub adaptation_type: AdaptationType,

    /// Parameters changed
    pub parameters_changed: Vec<ParameterChange>,

    /// Performance improvement
    pub performance_improvement: f64,

    /// Adaptation time
    pub adaptation_time: u64,

    /// Side effects
    pub side_effects: Vec<String>,
}

/// Types of adaptations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Learning rate adjustment
    LearningRateAdjustment,
    /// Parameter tuning
    ParameterTuning,
    /// Strategy change
    StrategyChange,
    /// Model update
    ModelUpdate,
    /// Behavior modification
    BehaviorModification,
    /// Resource reallocation
    ResourceReallocation,
}

/// Parameter change record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterChange {
    /// Parameter name
    pub parameter_name: String,

    /// Old value
    pub old_value: f64,

    /// New value
    pub new_value: f64,

    /// Change reason
    pub reason: String,
}

/// Experience for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Experience identifier
    pub id: String,

    /// State when experience occurred
    pub state: String, // Serialized state

    /// Action taken
    pub action: String, // Serialized action

    /// Outcome observed
    pub outcome: String, // Serialized outcome

    /// Reward received
    pub reward: f64,

    /// Success indicator
    pub success: bool,

    /// Timestamp
    pub timestamp: u64,

    /// Context information
    pub context: Vec<String>,
}

impl Experience {
    /// Create positive experience
    pub fn positive(state: String, action: String, outcome: String, reward: f64) -> Self {
        Experience {
            id: format!("exp_{}", uuid::Uuid::new_v4()),
            state,
            action,
            outcome,
            reward,
            success: true,
            timestamp: 0, // Would use actual timestamp in real implementation
            context: Vec::new(),
        }
    }

    /// Create negative experience
    pub fn negative(state: String, action: String, outcome: String, penalty: f64) -> Self {
        Experience {
            id: format!("exp_{}", uuid::Uuid::new_v4()),
            state,
            action,
            outcome,
            reward: -penalty,
            success: false,
            timestamp: 0, // Would use actual timestamp in real implementation
            context: Vec::new(),
        }
    }
}

/// Learning result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResult {
    /// Learning success
    pub success: bool,

    /// Model accuracy improvement
    pub accuracy_improvement: f64,

    /// Learning iterations performed
    pub iterations: u32,

    /// Convergence status
    pub converged: bool,

    /// Learning time
    pub learning_time: u64,

    /// Model complexity
    pub model_complexity: f64,
}

/// Adaptation evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvaluation {
    /// Current adaptation level (0.0 to 1.0)
    pub adaptation_level: f64,

    /// Adaptation effectiveness (0.0 to 1.0)
    pub effectiveness: f64,

    /// Adaptation stability (0.0 to 1.0)
    pub stability: f64,

    /// Number of recent adaptations
    pub recent_adaptations: u32,

    /// Adaptation trend
    pub trend: AdaptationTrend,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Adaptation trend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationTrend {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Declining trend
    Declining,
    /// Oscillating trend
    Oscillating,
    /// Unknown trend
    Unknown,
}

/// Adaptation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParameters {
    /// Learning rate
    pub learning_rate: f64,

    /// Adaptation threshold
    pub adaptation_threshold: f64,

    /// Exploration rate
    pub exploration_rate: f64,

    /// Memory retention factor
    pub memory_retention: f64,

    /// Adaptation frequency
    pub adaptation_frequency: AdaptationFrequency,

    /// Maximum adaptations per period
    pub max_adaptations_per_period: u32,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        AdaptationParameters {
            learning_rate: 0.01,
            adaptation_threshold: 0.1,
            exploration_rate: 0.1,
            memory_retention: 0.9,
            adaptation_frequency: AdaptationFrequency::Adaptive,
            max_adaptations_per_period: 10,
        }
    }
}

/// Adaptation frequency settings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationFrequency {
    /// Continuous adaptation
    Continuous,
    /// Periodic adaptation
    Periodic(u64), // Period in milliseconds
    /// Event-driven adaptation
    EventDriven,
    /// Adaptive frequency
    Adaptive,
}

/// Training result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Training success
    pub success: bool,

    /// Final accuracy
    pub final_accuracy: f64,

    /// Training loss
    pub final_loss: f64,

    /// Epochs completed
    pub epochs: u32,

    /// Training time
    pub training_time: u64,

    /// Convergence achieved
    pub converged: bool,
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Predicted value
    pub value: String, // Serialized prediction

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Prediction alternatives
    pub alternatives: Vec<PredictionAlternative>,

    /// Prediction explanation
    pub explanation: Vec<String>,

    /// Uncertainty measure
    pub uncertainty: f64,
}

/// Alternative predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionAlternative {
    /// Alternative value
    pub value: String,

    /// Probability
    pub probability: f64,

    /// Explanation
    pub explanation: String,
}

/// Model evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluation {
    /// Accuracy score
    pub accuracy: f64,

    /// Precision score
    pub precision: f64,

    /// Recall score
    pub recall: f64,

    /// F1 score
    pub f1_score: f64,

    /// Loss value
    pub loss: f64,

    /// Evaluation time
    pub evaluation_time: u64,
}

/// Strategy types for learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyType {
    /// Supervised learning
    Supervised,
    /// Unsupervised learning
    Unsupervised,
    /// Reinforcement learning
    Reinforcement,
    /// Transfer learning
    Transfer,
    /// Online learning
    Online,
    /// Ensemble learning
    Ensemble,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Model accuracy
    pub accuracy: f64,

    /// Training time
    pub training_time: u64,

    /// Inference time
    pub inference_time: u64,

    /// Model size
    pub model_size: u64,

    /// Memory usage
    pub memory_usage: u64,

    /// Update frequency
    pub update_frequency: f64,
}

/// Population for evolutionary algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population<T> {
    /// Individuals in population
    pub individuals: Vec<T>,

    /// Population size
    pub size: usize,

    /// Generation number
    pub generation: u32,

    /// Best fitness achieved
    pub best_fitness: Option<f64>,

    /// Average fitness
    pub average_fitness: f64,
}

impl<T> Population<T> {
    /// Create new population
    pub fn new(individuals: Vec<T>) -> Self {
        let size = individuals.len();
        Population {
            individuals,
            size,
            generation: 0,
            best_fitness: None,
            average_fitness: 0.0,
        }
    }

    /// Get population size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Advance to next generation
    pub fn next_generation(&mut self) {
        self.generation += 1;
    }
}

/// Evolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult<T> {
    /// Best individual found
    pub best_individual: T,

    /// Best fitness achieved
    pub best_fitness: f64,

    /// Final population
    pub final_population: Population<T>,

    /// Generations completed
    pub generations: u32,

    /// Evolution time
    pub evolution_time: u64,

    /// Convergence achieved
    pub converged: bool,
}

/// Evolution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionParameters {
    /// Population size
    pub population_size: usize,

    /// Crossover rate
    pub crossover_rate: f64,

    /// Mutation rate
    pub mutation_rate: f64,

    /// Selection pressure
    pub selection_pressure: f64,

    /// Elitism ratio
    pub elitism_ratio: f64,

    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for EvolutionParameters {
    fn default() -> Self {
        EvolutionParameters {
            population_size: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection_pressure: 2.0,
            elitism_ratio: 0.1,
            convergence_threshold: 0.001,
        }
    }
}

/// Action probabilities for policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionProbabilities<A> {
    /// Action-probability pairs
    pub probabilities: Vec<(A, f64)>,

    /// Total probability mass
    pub total_mass: f64,
}

impl<A> ActionProbabilities<A> {
    /// Create new action probabilities
    pub fn new(probabilities: Vec<(A, f64)>) -> Self {
        let total_mass = probabilities.iter().map(|(_, p)| p).sum();
        ActionProbabilities {
            probabilities,
            total_mass,
        }
    }

    /// Normalize probabilities
    pub fn normalize(&mut self) {
        if self.total_mass > 0.0 {
            for (_, p) in &mut self.probabilities {
                *p /= self.total_mass;
            }
            self.total_mass = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_creation() {
        let exp = Experience::positive(
            "state1".to_string(),
            "action1".to_string(),
            "outcome1".to_string(),
            1.0,
        );

        assert!(exp.success);
        assert_eq!(exp.reward, 1.0);
    }

    #[test]
    fn test_adaptation_parameters_default() {
        let params = AdaptationParameters::default();
        assert_eq!(params.learning_rate, 0.01);
        assert_eq!(params.exploration_rate, 0.1);
    }

    #[test]
    fn test_population_creation() {
        let individuals = vec![1, 2, 3, 4, 5];
        let population = Population::new(individuals);

        assert_eq!(population.size(), 5);
        assert_eq!(population.generation, 0);
    }

    #[test]
    fn test_action_probabilities_normalization() {
        let mut probs =
            ActionProbabilities::new(vec![("action1", 2.0), ("action2", 3.0), ("action3", 1.0)]);

        assert_eq!(probs.total_mass, 6.0);
        probs.normalize();
        assert_eq!(probs.total_mass, 1.0);
    }
}
