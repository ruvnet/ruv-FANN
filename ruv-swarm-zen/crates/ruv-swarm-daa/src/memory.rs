//! Memory management for DAA agents

use crate::*;
use std::collections::HashMap;

/// Memory manager for agent memory systems
pub struct MemoryManager {
    pub agent_memories: HashMap<String, AgentMemorySystem>,
    pub shared_memory: SharedMemorySpace,
    pub memory_policies: Vec<MemoryPolicy>,
}

/// Agent memory system
pub struct AgentMemorySystem {
    pub working_memory: WorkingMemory,
    pub long_term_memory: LongTermMemory,
    pub episodic_memory: EpisodicMemoryStore,
    pub semantic_memory: SemanticMemoryStore,
}

/// Working memory for immediate processing
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    pub capacity: usize,
    pub current_items: Vec<MemoryItem>,
    pub attention_weights: HashMap<String, f64>,
}

/// Long-term memory storage
pub struct LongTermMemory {
    pub consolidated_memories: HashMap<String, ConsolidatedMemory>,
    pub memory_strength: HashMap<String, f64>,
    pub last_accessed: HashMap<String, chrono::DateTime<chrono::Utc>>,
}

/// Episodic memory store
pub struct EpisodicMemoryStore {
    pub episodes: Vec<EpisodicMemory>,
    pub temporal_index: HashMap<chrono::DateTime<chrono::Utc>, Vec<String>>,
    pub contextual_index: HashMap<String, Vec<String>>,
}

/// Semantic memory store
pub struct SemanticMemoryStore {
    pub concepts: HashMap<String, ConceptNode>,
    pub relationships: Vec<ConceptRelationship>,
    pub concept_hierarchies: HashMap<String, Vec<String>>,
}

/// Shared memory space across agents
pub struct SharedMemorySpace {
    pub global_knowledge: HashMap<String, GlobalKnowledgeItem>,
    pub coordination_state: HashMap<String, serde_json::Value>,
    pub collective_experiences: Vec<CollectiveExperience>,
}

/// Consolidated memory item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedMemory {
    pub id: String,
    pub content: serde_json::Value,
    pub importance: f64,
    pub consolidation_time: chrono::DateTime<chrono::Utc>,
    pub access_count: u32,
    pub related_memories: Vec<String>,
}

/// Concept node in semantic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub concept: String,
    pub attributes: HashMap<String, serde_json::Value>,
    pub activation_level: f64,
    pub learned_associations: Vec<String>,
}

/// Relationship between concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelationship {
    pub from_concept: String,
    pub to_concept: String,
    pub relationship_type: String,
    pub strength: f64,
}

/// Global knowledge item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalKnowledgeItem {
    pub id: String,
    pub knowledge: Knowledge,
    pub contributors: Vec<String>,
    pub validation_score: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Collective experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveExperience {
    pub id: String,
    pub participating_agents: Vec<String>,
    pub shared_context: HashMap<String, serde_json::Value>,
    pub outcomes: Vec<TaskResult>,
    pub lessons_learned: Vec<String>,
}

/// Memory policy for memory management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPolicy {
    pub policy_type: MemoryPolicyType,
    pub parameters: HashMap<String, f64>,
    pub applicable_agents: Vec<String>,
}

/// Types of memory policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPolicyType {
    ForgetOldMemories,
    ConsolidateImportantMemories,
    ShareValuableKnowledge,
    PruneUnusedConnections,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            agent_memories: HashMap::new(),
            shared_memory: SharedMemorySpace::new(),
            memory_policies: Self::default_policies(),
        }
    }

    fn default_policies() -> Vec<MemoryPolicy> {
        vec![MemoryPolicy {
            policy_type: MemoryPolicyType::ForgetOldMemories,
            parameters: {
                let mut params = HashMap::new();
                params.insert("max_age_days".to_string(), 30.0);
                params.insert("min_importance".to_string(), 0.1);
                params
            },
            applicable_agents: vec!["*".to_string()], // All agents
        }]
    }
}

impl Default for SharedMemorySpace {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedMemorySpace {
    pub fn new() -> Self {
        Self {
            global_knowledge: HashMap::new(),
            coordination_state: HashMap::new(),
            collective_experiences: Vec::new(),
        }
    }
}
