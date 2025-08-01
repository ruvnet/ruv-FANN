//! Real MCP Service implementation for stdio transport
//!
//! This connects the actual swarm orchestration to MCP using the rmcp SDK pattern

use rmcp::{
    Error as McpError, 
    model::*, 
    tool, 
    tool_handler, 
    tool_router,
    handler::server::router::tool::ToolRouter,
    handler::server::tool::Parameters,
    ServerHandler
};
use std::{sync::Arc, future::Future};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;
use schemars::JsonSchema;

use crate::orchestrator::SwarmOrchestrator;

/// Real RUV Swarm MCP Service
#[derive(Clone)]
pub struct RealSwarmService {
    orchestrator: Arc<SwarmOrchestrator>,
    tool_router: ToolRouter<Self>,
}

// Parameter structs for tools
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SpawnParams {
    agent_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct OrchestrateParams {
    objective: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    strategy: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct OptimizeParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    target_metric: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct MemoryStoreParams {
    key: String,
    value: Value,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct MemoryGetParams {
    key: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct TaskCreateParams {
    task_type: String,
    description: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct AgentMetricsParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
}

#[tool_router]
impl RealSwarmService {
    pub fn new(orchestrator: Arc<SwarmOrchestrator>) -> Self {
        Self {
            orchestrator,
            tool_router: Self::tool_router(),
        }
    }

    /// Spawn a new agent (researcher, coder, analyst, tester, reviewer, documenter)
    #[tool]
    async fn spawn(&self, params: Parameters<SpawnParams>) -> Result<CallToolResult, McpError> {
        let capabilities = crate::types::AgentCapabilities {
            languages: vec!["rust".to_string(), "python".to_string()],
            frameworks: vec!["tokio".to_string()],
            tools: vec![params.0.agent_type.clone()],
            specializations: vec![params.0.agent_type.clone()],
            max_concurrent_tasks: 10,
        };
        
        let agent_id = self.orchestrator.spawn_agent(
            params.0.agent_type.parse().unwrap_or_default(),
            params.0.name.unwrap_or_else(|| format!("agent-{}", Uuid::new_v4())),
            capabilities
        ).await.map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            format!("Agent spawned with ID: {}", agent_id)
        )]))
    }

    /// Orchestrate a task with specified strategy (research, development, analysis, testing, optimization, maintenance)
    #[tool]
    async fn orchestrate(&self, params: Parameters<OrchestrateParams>) -> Result<CallToolResult, McpError> {
        let task_id = self.orchestrator.create_task(
            params.0.strategy.clone().unwrap_or_else(|| "development".to_string()),
            params.0.objective.clone(),
            vec![],
            params.0.strategy.unwrap_or_else(|| "adaptive".to_string()),
        ).await.map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            format!("Task '{}' created with ID: {}", params.0.objective, task_id)
        )]))
    }

    /// Query the current swarm state and active agents
    #[tool]
    async fn query(&self) -> Result<CallToolResult, McpError> {
        let state = self.orchestrator.get_swarm_state()
            .await.map_err(|e| McpError::internal_error(e.to_string(), None))?;
        
        Ok(CallToolResult::success(vec![Content::text(
            format!("Swarm state: {} agents, {} active tasks, {} completed tasks", 
                state.total_agents, state.active_tasks, state.completed_tasks)
        )]))
    }

    /// Monitor swarm activity and performance
    #[tool]
    async fn monitor(&self) -> Result<CallToolResult, McpError> {
        let metrics = self.orchestrator.get_performance_metrics()
            .await.map_err(|e| McpError::internal_error(e.to_string(), None))?;
        
        Ok(CallToolResult::success(vec![Content::text(
            format!("Swarm metrics: {} agents ({} active), {:.2}% success rate, {:.2}ms avg response time", 
                metrics.total_agents, metrics.active_agents, 
                metrics.success_rate * 100.0, metrics.average_response_time)
        )]))
    }

    /// Optimize swarm performance for target metric (throughput, latency, resource_usage, cost, quality)
    #[tool]
    async fn optimize(&self, params: Parameters<OptimizeParams>) -> Result<CallToolResult, McpError> {
        let metric = params.0.target_metric.unwrap_or_else(|| "throughput".to_string());
        let recommendations = self.orchestrator.optimize_performance(metric.clone(), 0.8)
            .await.map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            format!("Optimization for '{}': {} recommendations generated", metric, recommendations.len())
        )]))
    }

    /// Store data in swarm memory
    #[tool]
    async fn memory_store(&self, params: Parameters<MemoryStoreParams>) -> Result<CallToolResult, McpError> {
        self.orchestrator.store_session_data(params.0.key.clone(), params.0.value.clone(), None)
            .await.map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            format!("Data stored for key: {}", params.0.key)
        )]))
    }

    /// Retrieve data from swarm memory
    #[tool]
    async fn memory_get(&self, params: Parameters<MemoryGetParams>) -> Result<CallToolResult, McpError> {
        let value = self.orchestrator.get_session_data(&params.0.key)
            .await.map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let message = if let Some(data) = value {
            format!("Retrieved: {}", serde_json::to_string_pretty(&data).unwrap())
        } else {
            format!("No data found for key: {}", params.0.key)
        };

        Ok(CallToolResult::success(vec![Content::text(message)]))
    }

    /// Create a new task
    #[tool]
    async fn task_create(&self, params: Parameters<TaskCreateParams>) -> Result<CallToolResult, McpError> {
        let task_id = self.orchestrator.create_task(
            params.0.task_type.clone(),
            params.0.description.clone(),
            vec![],
            "default".to_string(),
        ).await.map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            format!("Task created with ID: {}", task_id)
        )]))
    }

    /// List all agents in the swarm
    #[tool]
    async fn agent_list(&self) -> Result<CallToolResult, McpError> {
        let agents = self.orchestrator.list_agents()
            .await.map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let summary = if agents.is_empty() {
            "No agents currently in the swarm".to_string()
        } else {
            format!("Active agents ({}):\n{}", 
                agents.len(),
                agents.iter()
                    .map(|a| format!("  - {} ({}): {}", a.id, a.agent_type, a.status))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }

    /// Get agent performance metrics
    #[tool]
    async fn agent_metrics(&self, params: Parameters<AgentMetricsParams>) -> Result<CallToolResult, McpError> {
        let metrics = if let Some(id_str) = params.0.agent_id.as_ref() {
            let id = Uuid::parse_str(&id_str)
                .map_err(|e| McpError::invalid_params(format!("Invalid UUID: {}", e), None))?;
            vec![self.orchestrator.get_agent_metrics(id)
                .await.map_err(|e| McpError::internal_error(e.to_string(), None))?]
        } else {
            self.orchestrator.get_all_agent_metrics()
                .await.map_err(|e| McpError::internal_error(e.to_string(), None))?
        };

        let summary = format!("Agent metrics:\n{}", 
            metrics.iter()
                .map(|m| format!("  - Agent {}: {:.2}ms response, {:.2}% success rate", 
                    m.agent_id, m.response_time, m.success_rate * 100.0))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }
}

// Implement the ServerHandler trait
#[tool_handler]
impl ServerHandler for RealSwarmService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("RUV Swarm orchestration for massive GPU agent coordination. Use 'spawn' to create agents, 'orchestrate' to run tasks, 'query' to check status.".into()),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}