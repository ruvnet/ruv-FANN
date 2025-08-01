//! Orchestrator implementation using real ruv-swarm-core with persistence
//!
//! This module provides a higher-level orchestration interface that wraps
//! the ruv-swarm-core Swarm and provides real metrics and persistence.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use serde_json::Value;
use uuid::Uuid;
use chrono::Utc;

use ruv_swarm_core::{
    Swarm, SwarmConfig, SwarmMetrics, Task, TaskId, Priority as TaskPriority, 
    Agent, AgentId, AgentStatus, DynamicAgent, SwarmError
};

use ruv_swarm_persistence::{
    Storage, SqliteStorage, StorageError,
    AgentModel, TaskModel, EventModel, MetricModel, MessageModel,
    models::{AgentStatus as PersistenceAgentStatus, TaskStatus, TaskPriority as PersistenceTaskPriority},
};

use crate::types::{
    AgentType, AgentCapabilities, AgentInfo, SwarmState, 
    OptimizationRecommendation, AgentMetrics, WorkflowDefinition
};

/// SwarmOrchestrator provides a higher-level interface for the MCP server with real persistence
pub struct SwarmOrchestrator {
    swarm: Arc<RwLock<Swarm>>,
    storage: Arc<SqliteStorage>,
    session_data: Arc<RwLock<HashMap<String, Value>>>,
    metrics: Arc<RwLock<OrchestratorMetrics>>,
}

/// Real-time metrics tracking
struct OrchestratorMetrics {
    total_tasks_created: u64,
    total_tasks_completed: u64,
    total_tasks_failed: u64,
    average_task_duration_ms: f64,
    last_task_metrics: HashMap<String, TaskMetrics>,
}

struct TaskMetrics {
    start_time: Instant,
    end_time: Option<Instant>,
    assigned_agents: Vec<String>,
}

impl SwarmOrchestrator {
    /// Create a new SwarmOrchestrator with persistence
    pub async fn new() -> Self {
        let config = SwarmConfig::default();
        let swarm = Swarm::new(config);
        
        // Initialize SQLite storage with persistent file
        let db_path = std::env::var("RUV_SWARM_DB_PATH")
            .unwrap_or_else(|_| "ruv-swarm-mcp.db".to_string());
        let storage = SqliteStorage::new(&db_path).await
            .expect("Failed to create storage");
        
        tracing::info!("Using SQLite database at: {}", db_path);
        
        Self {
            swarm: Arc::new(RwLock::new(swarm)),
            storage: Arc::new(storage),
            session_data: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(OrchestratorMetrics {
                total_tasks_created: 0,
                total_tasks_completed: 0,
                total_tasks_failed: 0,
                average_task_duration_ms: 0.0,
                last_task_metrics: HashMap::new(),
            })),
        }
    }

    /// Spawn a new agent with persistence
    pub async fn spawn_agent(
        &self,
        agent_type: AgentType,
        name: String,
        capabilities: AgentCapabilities,
    ) -> Result<Uuid, SwarmError> {
        let start_time = Instant::now();
        let agent_id = Uuid::new_v4();
        let agent_id_str = format!("{}-{}", name, agent_id);
        
        // Create DynamicAgent
        let dynamic_agent = DynamicAgent::new(agent_id_str.clone(), capabilities.tools.clone());
        
        // Register with swarm
        let mut swarm = self.swarm.write().await;
        swarm.register_agent(dynamic_agent)?;
        
        // Persist agent to database
        let agent_model = AgentModel::new(
            name.clone(),
            agent_type.to_string(),
            capabilities.tools.clone()
        );
        
        self.storage.store_agent(&agent_model).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        // Log agent spawn event
        let event = EventModel::new(
            "agent_spawn".to_string(),
            serde_json::json!({
                "agent_id": agent_model.id.clone(),
                "agent_type": agent_type.to_string(),
                "capabilities": capabilities,
                "spawn_time_ms": start_time.elapsed().as_millis()
            })
        );
        self.storage.store_event(&event).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        // Record spawn metrics
        let metric = MetricModel::new(
            "agent_spawn_time".to_string(),
            start_time.elapsed().as_millis() as f64,
            "milliseconds".to_string()
        );
        self.storage.store_metric(&metric).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        Ok(agent_id)
    }

    /// Create a new task with persistence
    pub async fn create_task(
        &self,
        task_type: String,
        description: String,
        requirements: Vec<String>,
        strategy: String,
    ) -> Result<Uuid, SwarmError> {
        let start_time = Instant::now();
        let task_id = Uuid::new_v4();
        let task_id_str = task_id.to_string();
        
        // Create core task
        let mut task = Task::new(task_id_str.clone(), task_type.clone())
            .with_priority(TaskPriority::Normal);
        
        for req in requirements.iter() {
            task = task.require_capability(req.clone());
        }
        
        // Submit to swarm
        let mut swarm = self.swarm.write().await;
        swarm.submit_task(task)?;
        
        // Create persistence model
        let task_model = TaskModel::new(
            task_type.clone(),
            serde_json::json!({
                "description": description,
                "requirements": requirements,
                "strategy": strategy,
            }),
            PersistenceTaskPriority::Medium
        );
        
        self.storage.store_task(&task_model).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_tasks_created += 1;
        metrics.last_task_metrics.insert(task_id_str.clone(), TaskMetrics {
            start_time,
            end_time: None,
            assigned_agents: vec![],
        });
        
        // Record task creation metric
        let metric = MetricModel::new(
            "task_creation_time".to_string(),
            start_time.elapsed().as_millis() as f64,
            "milliseconds".to_string()
        );
        self.storage.store_metric(&metric).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        Ok(task_id)
    }

    /// Get current swarm state with real metrics
    pub async fn get_swarm_state(&self) -> Result<SwarmState, SwarmError> {
        let swarm = self.swarm.read().await;
        let metrics = swarm.metrics();
        
        // Get real agent info from database
        let db_agents = self.storage.list_agents().await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        let agents = db_agents.into_iter().map(|agent| AgentInfo {
            id: Uuid::parse_str(&agent.id).unwrap_or_else(|_| Uuid::new_v4()),
            agent_type: agent.agent_type.parse().unwrap_or_default(),
            name: Some(agent.name),
            status: agent.status.to_string(),
            created_at: agent.created_at,
            current_tasks: vec![], // Would need to query tasks by agent
        }).collect();
        
        // Get task counts from database
        let pending_tasks = self.storage.get_pending_tasks().await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        let completed_count = self.metrics.read().await.total_tasks_completed;
        
        Ok(SwarmState {
            agents,
            active_tasks: metrics.assigned_tasks,
            completed_tasks: completed_count as usize,
            total_agents: metrics.total_agents,
        })
    }

    /// Get performance metrics with real data
    pub async fn get_performance_metrics(&self) -> Result<crate::types::SwarmMetrics, SwarmError> {
        let swarm = self.swarm.read().await;
        let core_metrics = swarm.metrics();
        let orchestrator_metrics = self.metrics.read().await;
        
        // Query real metrics from database
        let recent_metrics = self.storage.get_aggregated_metrics(
            "task_completion_time",
            Utc::now().timestamp() - 3600, // Last hour
            Utc::now().timestamp()
        ).await.map_err(|e| SwarmError::custom(e.to_string()))?;
        
        let avg_response_time = if !recent_metrics.is_empty() {
            recent_metrics.iter().map(|m| m.value).sum::<f64>() / recent_metrics.len() as f64
        } else {
            orchestrator_metrics.average_task_duration_ms
        };
        
        let success_rate = if orchestrator_metrics.total_tasks_created > 0 {
            orchestrator_metrics.total_tasks_completed as f64 / 
            orchestrator_metrics.total_tasks_created as f64
        } else {
            1.0
        };
        
        // Calculate resource utilization based on active agents
        let active_agents = self.storage.list_agents_by_status("busy").await
            .map_err(|e| SwarmError::custom(e.to_string()))?
            .len();
        
        let resource_utilization = if core_metrics.total_agents > 0 {
            active_agents as f64 / core_metrics.total_agents as f64
        } else {
            0.0
        };
        
        Ok(crate::types::SwarmMetrics {
            total_agents: core_metrics.total_agents,
            active_agents: active_agents,
            total_tasks: orchestrator_metrics.total_tasks_created as usize,
            completed_tasks: orchestrator_metrics.total_tasks_completed as usize,
            average_response_time: avg_response_time,
            success_rate,
            resource_utilization,
        })
    }

    /// Optimize performance with real analysis
    pub async fn optimize_performance(
        &self,
        target_metric: String,
        threshold: f64,
    ) -> Result<Vec<OptimizationRecommendation>, SwarmError> {
        let metrics = self.get_performance_metrics().await?;
        let mut recommendations = Vec::new();
        
        match target_metric.as_str() {
            "throughput" => {
                if metrics.resource_utilization < 0.5 {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: "scaling".to_string(),
                        description: "Low resource utilization detected. Consider reducing agent count.".to_string(),
                        impact: "low".to_string(),
                        priority: crate::types::TaskPriority::Low,
                        estimated_improvement: 10.0,
                    });
                } else if metrics.resource_utilization > 0.9 {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: "scaling".to_string(),
                        description: "High resource utilization. Consider adding more agents.".to_string(),
                        impact: "high".to_string(),
                        priority: crate::types::TaskPriority::High,
                        estimated_improvement: 25.0,
                    });
                }
            },
            "latency" => {
                if metrics.average_response_time > threshold {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: "performance".to_string(),
                        description: format!("Average response time ({:.2}ms) exceeds threshold ({:.2}ms)", 
                            metrics.average_response_time, threshold),
                        impact: "medium".to_string(),
                        priority: crate::types::TaskPriority::Medium,
                        estimated_improvement: 15.0,
                    });
                }
            },
            "success_rate" => {
                if metrics.success_rate < threshold {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: "reliability".to_string(),
                        description: format!("Success rate ({:.2}%) below threshold ({:.2}%)", 
                            metrics.success_rate * 100.0, threshold * 100.0),
                        impact: "critical".to_string(),
                        priority: crate::types::TaskPriority::Critical,
                        estimated_improvement: 30.0,
                    });
                }
            },
            _ => {}
        }
        
        // Log optimization analysis
        let event = EventModel::new(
            "optimization_analysis".to_string(),
            serde_json::json!({
                "target_metric": target_metric,
                "threshold": threshold,
                "recommendations": recommendations.len(),
                "current_metrics": metrics,
            })
        );
        self.storage.store_event(&event).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        Ok(recommendations)
    }

    /// Store session data
    pub async fn store_session_data(
        &self,
        key: String,
        value: Value,
        metadata: Option<Value>,
    ) -> Result<(), SwarmError> {
        let mut data = self.session_data.write().await;
        data.insert(key.clone(), value.clone());
        
        // Also persist important session data as an event
        if let Some(meta) = metadata {
            let event = EventModel::new(
                "session_data_stored".to_string(),
                serde_json::json!({
                    "key": key,
                    "metadata": meta,
                })
            );
            self.storage.store_event(&event).await
                .map_err(|e| SwarmError::custom(e.to_string()))?;
        }
        
        Ok(())
    }

    /// Get session data
    pub async fn get_session_data(&self, key: &str) -> Result<Option<Value>, SwarmError> {
        let data = self.session_data.read().await;
        Ok(data.get(key).cloned())
    }

    /// List agents with real data
    pub async fn list_agents(&self) -> Result<Vec<AgentInfo>, SwarmError> {
        let db_agents = self.storage.list_agents().await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        let swarm = self.swarm.read().await;
        let statuses = swarm.agent_statuses();
        
        let agents = db_agents.into_iter().map(|agent| {
            // Get real-time status from swarm if available
            let runtime_status = statuses.get(&agent.id)
                .map(|s| format!("{:?}", s))
                .unwrap_or_else(|| agent.status.to_string());
            
            AgentInfo {
                id: Uuid::parse_str(&agent.id).unwrap_or_else(|_| Uuid::new_v4()),
                agent_type: agent.agent_type.parse().unwrap_or_default(),
                name: Some(agent.name),
                status: runtime_status,
                created_at: agent.created_at,
                current_tasks: vec![], // Would need to query active tasks
            }
        }).collect();
        
        Ok(agents)
    }

    /// Get agent metrics with real data
    pub async fn get_agent_metrics(&self, agent_id: Uuid) -> Result<AgentMetrics, SwarmError> {
        let agent_id_str = agent_id.to_string();
        
        // Get real metrics from database
        let response_metrics = self.storage.get_metrics_by_agent(
            &agent_id_str,
            "task_completion_time"
        ).await.map_err(|e| SwarmError::custom(e.to_string()))?;
        
        let task_metrics = self.storage.get_metrics_by_agent(
            &agent_id_str,
            "tasks_completed"
        ).await.map_err(|e| SwarmError::custom(e.to_string()))?;
        
        let error_metrics = self.storage.get_metrics_by_agent(
            &agent_id_str,
            "task_errors"
        ).await.map_err(|e| SwarmError::custom(e.to_string()))?;
        
        let response_time = if !response_metrics.is_empty() {
            response_metrics.iter().map(|m| m.value).sum::<f64>() / response_metrics.len() as f64
        } else {
            0.0
        };
        
        let tasks_completed = task_metrics.len() as u32;
        let error_count = error_metrics.len() as u32;
        let success_rate = if tasks_completed > 0 {
            (tasks_completed - error_count) as f64 / tasks_completed as f64
        } else {
            1.0
        };
        
        Ok(AgentMetrics {
            agent_id,
            response_time,
            tasks_completed: tasks_completed as usize,
            success_rate,
            error_count: error_count as usize,
        })
    }

    /// Get all agent metrics
    pub async fn get_all_agent_metrics(&self) -> Result<Vec<AgentMetrics>, SwarmError> {
        let agents = self.list_agents().await?;
        let mut metrics = Vec::new();
        
        for agent in agents {
            match self.get_agent_metrics(agent.id).await {
                Ok(m) => metrics.push(m),
                Err(_) => {
                    // If no metrics available, provide defaults
                    metrics.push(AgentMetrics {
                        agent_id: agent.id,
                        response_time: 0.0,
                        tasks_completed: 0,
                        success_rate: 1.0,
                        error_count: 0,
                    });
                }
            }
        }
        
        Ok(metrics)
    }

    /// Execute workflow with real task tracking
    pub async fn execute_workflow(&self, workflow: WorkflowDefinition) -> Result<Uuid, SwarmError> {
        let workflow_id = Uuid::new_v4();
        let start_time = Instant::now();
        
        // Log workflow start event
        let event = EventModel::new(
            "workflow_start".to_string(),
            serde_json::json!({
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "total_steps": workflow.steps.len(),
            })
        );
        self.storage.store_event(&event).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        // Create tasks for each step
        for (idx, step) in workflow.steps.iter().enumerate() {
            let task_id = Uuid::new_v4();
            let task = Task::new(task_id.to_string(), step.task_type.clone())
                .with_priority(TaskPriority::Normal);
            
            let mut swarm = self.swarm.write().await;
            swarm.submit_task(task)?;
            
            // Store task with workflow reference
            let mut task_model = TaskModel::new(
                step.task_type.clone(),
                serde_json::json!({
                    "workflow_id": workflow_id,
                    "step_index": idx,
                    "step_data": step,
                }),
                PersistenceTaskPriority::Medium
            );
            task_model.dependencies = step.dependencies.clone();
            
            self.storage.store_task(&task_model).await
                .map_err(|e| SwarmError::custom(e.to_string()))?;
        }
        
        // Record workflow creation metric
        let metric = MetricModel::new(
            "workflow_creation_time".to_string(),
            start_time.elapsed().as_millis() as f64,
            "milliseconds".to_string()
        );
        self.storage.store_metric(&metric).await
            .map_err(|e| SwarmError::custom(e.to_string()))?;
        
        Ok(workflow_id)
    }

    /// Mark a task as completed (for testing/demo purposes)
    pub async fn complete_task(&self, task_id: &str, result: Value) -> Result<(), SwarmError> {
        // Update task in database
        if let Ok(Some(mut task)) = self.storage.get_task(task_id).await {
            task.complete(result);
            self.storage.update_task(&task).await
                .map_err(|e| SwarmError::custom(e.to_string()))?;
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.total_tasks_completed += 1;
            
            if let Some(task_metrics) = metrics.last_task_metrics.get_mut(task_id) {
                task_metrics.end_time = Some(Instant::now());
                let duration = task_metrics.end_time.unwrap().duration_since(task_metrics.start_time);
                
                // Update average
                let total = metrics.total_tasks_completed as f64;
                metrics.average_task_duration_ms = 
                    (metrics.average_task_duration_ms * (total - 1.0) + duration.as_millis() as f64) / total;
                
                // Record completion metric
                let metric = MetricModel::new(
                    "task_completion_time".to_string(),
                    duration.as_millis() as f64,
                    "milliseconds".to_string()
                );
                self.storage.store_metric(&metric).await
                    .map_err(|e| SwarmError::custom(e.to_string()))?;
            }
        }
        
        Ok(())
    }
}


impl Default for AgentType {
    fn default() -> Self {
        AgentType::Researcher
    }
}

impl std::str::FromStr for AgentType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "researcher" => Ok(AgentType::Researcher),
            "coder" => Ok(AgentType::Coder),
            "analyst" => Ok(AgentType::Analyst),
            "tester" => Ok(AgentType::Tester),
            "reviewer" => Ok(AgentType::Reviewer),
            "documenter" => Ok(AgentType::Documenter),
            _ => Ok(AgentType::Researcher), // Default fallback
        }
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentType::Researcher => write!(f, "Researcher"),
            AgentType::Coder => write!(f, "Coder"),
            AgentType::Analyst => write!(f, "Analyst"),
            AgentType::Tester => write!(f, "Tester"),
            AgentType::Reviewer => write!(f, "Reviewer"),
            AgentType::Documenter => write!(f, "Documenter"),
        }
    }
}