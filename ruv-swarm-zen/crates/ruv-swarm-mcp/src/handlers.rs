//! MCP Request handlers for RUV-Swarm

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::{
    limits::ResourceLimiter, orchestrator::SwarmOrchestrator, tools::ToolRegistry, types::*,
    validation::InputValidator, McpRequest, McpResponse, Session,
};

/// Request handler
pub struct RequestHandler {
    orchestrator: Arc<SwarmOrchestrator>,
    tools: Arc<ToolRegistry>,
    session: Arc<Session>,
    limiter: Arc<ResourceLimiter>,
    tx: mpsc::Sender<axum::extract::ws::Message>,
    validator: InputValidator,
}

impl RequestHandler {
    pub fn new(
        orchestrator: Arc<SwarmOrchestrator>,
        tools: Arc<ToolRegistry>,
        session: Arc<Session>,
        limiter: Arc<ResourceLimiter>,
        tx: mpsc::Sender<axum::extract::ws::Message>,
    ) -> Self {
        Self {
            orchestrator,
            tools,
            session,
            limiter,
            tx,
            validator: InputValidator::default(),
        }
    }

    /// Check if a tool name is valid
    fn is_valid_tool(&self, tool_name: &str) -> bool {
        matches!(
            tool_name,
            "ruv-swarm.spawn"
                | "ruv-swarm.orchestrate"
                | "ruv-swarm.query"
                | "ruv-swarm.monitor"
                | "ruv-swarm.optimize"
                | "ruv-swarm.memory.store"
                | "ruv-swarm.memory.get"
                | "ruv-swarm.task.create"
                | "ruv-swarm.workflow.execute"
                | "ruv-swarm.agent.list"
                | "ruv-swarm.agent.metrics"
        )
    }

    /// Get list of available tools
    fn get_available_tools(&self) -> Vec<&'static str> {
        vec![
            "ruv-swarm.spawn",
            "ruv-swarm.orchestrate",
            "ruv-swarm.query",
            "ruv-swarm.monitor",
            "ruv-swarm.optimize",
            "ruv-swarm.memory.store",
            "ruv-swarm.memory.get",
            "ruv-swarm.task.create",
            "ruv-swarm.workflow.execute",
            "ruv-swarm.agent.list",
            "ruv-swarm.agent.metrics",
        ]
    }

    /// Create a detailed error response with suggestions
    fn create_detailed_error(
        &self,
        id: Option<Value>,
        code: i32,
        error_type: &str,
        message: &str,
        suggestions: Vec<&str>,
    ) -> McpResponse {
        // In production, only return safe error messages
        // Debug mode can include more details for development
        let safe_message = if cfg!(debug_assertions) {
            let error_details = json!({
                "error_type": error_type,
                "message": message,
                "suggestions": suggestions,
                "timestamp": chrono::Utc::now(),
                "session_id": self.session.id
            });
            serde_json::to_string(&error_details).unwrap_or_else(|_| message.to_string())
        } else {
            // Production: return only the message without internal details
            message.to_string()
        };

        McpResponse::error(id, code, safe_message)
    }

    /// Handle MCP request
    pub async fn handle_request(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        match request.method.as_str() {
            // Standard MCP methods
            "initialize" => self.handle_initialize(request).await,
            "tools/list" => self.handle_tools_list(request).await,
            "tools/call" => self.handle_tool_call(request).await,
            "notifications/subscribe" => self.handle_subscribe(request).await,
            "notifications/unsubscribe" => self.handle_unsubscribe(request).await,

            // Custom methods
            "ruv-swarm/status" => self.handle_swarm_status(request).await,
            "ruv-swarm/metrics" => self.handle_swarm_metrics(request).await,

            _ => Ok(McpResponse::error(
                request.id,
                -32601,
                "Method not found".to_string(),
            )),
        }
    }

    /// Handle initialize request
    async fn handle_initialize(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        info!("MCP client initializing session: {}", self.session.id);

        let result = json!({
            "protocolVersion": "1.0",
            "serverInfo": {
                "name": "ruv-swarm-mcp",
                "version": env!("CARGO_PKG_VERSION"),
                "capabilities": {
                    "tools": true,
                    "notifications": true,
                    "streaming": true,
                    "batch": true,
                }
            },
            "sessionId": self.session.id,
        });

        Ok(McpResponse::success(request.id, result))
    }

    /// Handle tools/list request
    async fn handle_tools_list(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        let tools = self.tools.list_tools();
        let result = json!({
            "tools": tools,
        });

        Ok(McpResponse::success(request.id, result))
    }

    /// Handle tools/call request with enhanced error handling
    async fn handle_tool_call(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        let params = request.params.ok_or_else(|| {
            anyhow::anyhow!("VALIDATION_ERROR: Missing params for tool call. Please provide parameters object with 'name' and 'arguments' fields.")
        })?;

        let tool_name = params.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("VALIDATION_ERROR: Missing required 'name' field in tool call. Expected string value with tool name."))?;

        let default_params = json!({});
        let tool_params = params.get("arguments").unwrap_or(&default_params);

        debug!("Calling tool: {} with params: {:?}", tool_name, tool_params);

        // Validate tool exists
        if !self.is_valid_tool(tool_name) {
            return Ok(self.create_detailed_error(
                request.id,
                -32602,
                "TOOL_NOT_FOUND",
                "Unknown tool specified",
                vec![
                    "Check the tool name spelling",
                    "Use tools/list to see available tools",
                ],
            ));
        }

        // Handle tool-specific validation and execution
        // Clone request.id to avoid move issues across match arms
        let request_id = request.id.clone();
        let result = match tool_name {
            "ruv-swarm.spawn" => self.handle_spawn(request_id.clone(), tool_params).await,
            "ruv-swarm.orchestrate" => {
                self.handle_orchestrate(request_id.clone(), tool_params)
                    .await
            }
            "ruv-swarm.query" => self.handle_query(request_id.clone(), tool_params).await,
            "ruv-swarm.monitor" => self.handle_monitor(request_id.clone(), tool_params).await,
            "ruv-swarm.optimize" => self.handle_optimize(request_id.clone(), tool_params).await,
            "ruv-swarm.memory.store" => {
                self.handle_memory_store(request_id.clone(), tool_params)
                    .await
            }
            "ruv-swarm.memory.get" => {
                self.handle_memory_get(request_id.clone(), tool_params)
                    .await
            }
            "ruv-swarm.task.create" => {
                self.handle_task_create(request_id.clone(), tool_params)
                    .await
            }
            "ruv-swarm.workflow.execute" => {
                self.handle_workflow_execute(request_id.clone(), tool_params)
                    .await
            }
            "ruv-swarm.agent.list" => {
                self.handle_agent_list(request_id.clone(), tool_params)
                    .await
            }
            "ruv-swarm.agent.metrics" => {
                self.handle_agent_metrics(request_id.clone(), tool_params)
                    .await
            }
            _ => unreachable!("Tool validation should have caught this"),
        };

        // Enhanced error handling for common issues
        match result {
            Err(e) => {
                error!("Tool execution failed for {}: {}", tool_name, e);

                let error_msg = e.to_string();
                let (error_type, suggestions) = if error_msg.contains("Missing required parameter")
                {
                    (
                        "VALIDATION_ERROR",
                        vec![
                            "Check all required parameters are provided",
                            "Verify parameter names and types",
                            "Consult tool documentation for parameter requirements",
                        ],
                    )
                } else if error_msg.contains("agent") && error_msg.contains("not found") {
                    (
                        "AGENT_ERROR",
                        vec![
                            "Verify the agent ID is correct",
                            "Check if the agent was properly spawned",
                            "Use agent.list to see available agents",
                        ],
                    )
                } else if error_msg.contains("swarm") && error_msg.contains("not") {
                    (
                        "SWARM_ERROR",
                        vec![
                            "Initialize a swarm first using swarm initialization",
                            "Check swarm ID is correct",
                            "Verify swarm is active and healthy",
                        ],
                    )
                } else if error_msg.contains("timeout") {
                    (
                        "TIMEOUT_ERROR",
                        vec![
                            "Increase timeout duration",
                            "Reduce task complexity",
                            "Check system resources",
                        ],
                    )
                } else {
                    (
                        "EXECUTION_ERROR",
                        vec![
                            "Check system logs for details",
                            "Verify system resources are available",
                            "Retry the operation",
                        ],
                    )
                };

                Ok(self.create_detailed_error(
                    request_id,
                    -32603,
                    error_type,
                    &error_msg,
                    suggestions,
                ))
            }
            Ok(response) => Ok(response),
        }
    }

    /// Handle spawn tool with enhanced validation
    async fn handle_spawn(&self, id: Option<Value>, params: &Value) -> anyhow::Result<McpResponse> {
        // Validate parameters first
        if let Err(e) = self.validator.validate_spawn_params(params) {
            return Ok(self.create_detailed_error(
                id,
                -32602,
                "VALIDATION_ERROR",
                &e.to_string(),
                vec![
                    "Check parameter types and values",
                    "Ensure required fields are present"
                ]
            ));
        }

        // Safe to unwrap after validation
        let agent_type_str = params.get("agent_type")
            .and_then(|v| v.as_str())
            .unwrap();

        let agent_type = match agent_type_str {
            "researcher" => AgentType::Researcher,
            "coder" => AgentType::Coder,
            "analyst" => AgentType::Analyst,
            "tester" => AgentType::Tester,
            "reviewer" => AgentType::Reviewer,
            "documenter" => AgentType::Documenter,
            _ => unreachable!(), // Validation ensures this won't happen
        };

        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let capabilities = if let Some(cap_value) = params.get("capabilities") {
            serde_json::from_value(cap_value.clone())?
        } else {
            AgentCapabilities::default()
        };

        // Check resource limits before spawning
        if let Err(e) = self.limiter.check_agent_limit(&self.session.id).await {
            return Ok(self.create_detailed_error(
                id,
                -32005,
                "RESOURCE_LIMIT",
                &e.to_string(),
                vec![
                    "Agent limit reached for this session",
                    "Consider terminating unused agents"
                ]
            ));
        }

        // Spawn agent
        let agent_id = self
            .orchestrator
            .spawn_agent(agent_type, name, capabilities)
            .await?;

        // Update resource tracking
        if let Err(e) = self.limiter.increment_agents(&self.session.id).await {
            error!("Failed to update agent count: {}", e);
        }

        let result = json!({
            "agent_id": agent_id,
            "agent_type": agent_type_str,
            "status": "active",
            "created_at": chrono::Utc::now(),
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle orchestrate tool
    async fn handle_orchestrate(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        // Validate parameters first
        if let Err(e) = self.validator.validate_orchestrate_params(params) {
            return Ok(self.create_detailed_error(
                id,
                -32602,
                "VALIDATION_ERROR",
                &e.to_string(),
                vec![
                    "Check parameter types and values",
                    "Ensure objective is provided"
                ]
            ));
        }

        // Safe to unwrap after validation
        let objective = params
            .get("objective")
            .and_then(|v| v.as_str())
            .unwrap();

        let strategy_str = params
            .get("strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("development");

        let strategy = match strategy_str {
            "research" => SwarmStrategy::Research,
            "development" => SwarmStrategy::Development,
            "analysis" => SwarmStrategy::Analysis,
            "testing" => SwarmStrategy::Testing,
            _ => SwarmStrategy::Development, // Validation ensures only valid values
        };

        let mode_str = params
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("hierarchical");

        let mode = match mode_str {
            "centralized" => CoordinationMode::Centralized,
            "distributed" => CoordinationMode::Distributed,
            "hierarchical" => CoordinationMode::Hierarchical,
            "mesh" => CoordinationMode::Mesh,
            "hybrid" => CoordinationMode::Hybrid,
            _ => CoordinationMode::Hierarchical,
        };

        let max_agents = params
            .get("max_agents")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let parallel = params
            .get("parallel")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Create orchestrator config
        let config = OrchestratorConfig {
            strategy,
            mode,
            max_agents,
            parallel,
            timeout: Duration::from_secs(300),
        };

        // Start orchestration
        let task_id = Uuid::new_v4();
        let orchestrator = self.orchestrator.clone();
        let objective_str = objective.to_string();

        // Spawn async task
        tokio::spawn(async move {
            match orchestrator
                .orchestrate_task(&task_id, &objective_str, config)
                .await
            {
                Ok(result) => {
                    info!("Orchestration completed: {:?}", result);
                }
                Err(e) => {
                    error!("Orchestration failed: {}", e);
                }
            }
        });

        let result = json!({
            "task_id": task_id,
            "objective": objective,
            "strategy": strategy_str,
            "mode": mode_str,
            "status": "started",
            "started_at": chrono::Utc::now(),
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle query tool
    async fn handle_query(&self, id: Option<Value>, params: &Value) -> anyhow::Result<McpResponse> {
        let _filter = params.get("filter").cloned();
        let include_metrics = params
            .get("include_metrics")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let state = self.orchestrator.get_swarm_state().await?;

        let mut result = json!({
            "agents": state.agents,
            "active_tasks": state.active_tasks,
            "completed_tasks": state.completed_tasks,
            "total_agents": state.total_agents,
        });

        if include_metrics {
            let metrics = self.orchestrator.get_metrics().await?;
            result["metrics"] = json!(metrics);
        }

        Ok(McpResponse::success(id, result))
    }

    /// Handle monitor tool
    async fn handle_monitor(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        let event_types = params
            .get("event_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec!["all".to_string()]);

        let duration_secs = params
            .get("duration_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(60);

        // Subscribe to events
        let mut event_rx = self.orchestrator.subscribe_events().await?;
        let tx = self.tx.clone();

        // Spawn monitoring task
        tokio::spawn(async move {
            let start = tokio::time::Instant::now();
            let duration = Duration::from_secs(duration_secs);

            while start.elapsed() < duration {
                tokio::select! {
                    Some(event) = event_rx.recv() => {
                        let notification = json!({
                            "method": "ruv-swarm/event",
                            "params": {
                                "event": event,
                                "timestamp": chrono::Utc::now(),
                            }
                        });

                        if let Ok(json) = serde_json::to_string(&notification) {
                            let _ = tx.send(axum::extract::ws::Message::Text(json)).await;
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                }
            }
        });

        let result = json!({
            "status": "monitoring",
            "duration_secs": duration_secs,
            "event_types": event_types,
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle optimize tool
    async fn handle_optimize(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        let target_metric = params
            .get("target_metric")
            .and_then(|v| v.as_str())
            .unwrap_or("throughput");

        let auto_apply = params
            .get("auto_apply")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Get optimization recommendations
        let recommendations = self.orchestrator.analyze_performance().await?;

        if auto_apply {
            // Apply optimizations
            for rec in &recommendations {
                self.orchestrator.apply_optimization(rec).await?;
            }
        }

        let result = json!({
            "target_metric": target_metric,
            "recommendations": recommendations,
            "applied": auto_apply,
            "timestamp": chrono::Utc::now(),
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle memory store
    async fn handle_memory_store(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        // Validate parameters first
        if let Err(e) = self.validator.validate_memory_store_params(params) {
            return Ok(self.create_detailed_error(
                id,
                -32602,
                "VALIDATION_ERROR",
                &e.to_string(),
                vec![
                    "Check parameter types and values",
                    "Ensure key and value are provided"
                ]
            ));
        }

        // Safe to unwrap after validation
        let key = params
            .get("key")
            .and_then(|v| v.as_str())
            .unwrap();

        let value = params
            .get("value")
            .unwrap();

        let ttl_secs = params.get("ttl_secs").and_then(|v| v.as_u64());

        // Check memory limits before storing
        let value_size = serde_json::to_string(&value).unwrap_or_default().len();
        if let Err(e) = self.limiter.check_memory_limit(&self.session.id, value_size).await {
            return Ok(self.create_detailed_error(
                id,
                -32006,
                "RESOURCE_LIMIT",
                &e.to_string(),
                vec![
                    "Memory limit reached for this session",
                    "Consider removing unused data"
                ]
            ));
        }

        // Store in session metadata with TTL support
        self.session.metadata.insert(key.to_string(), value.clone());

        // Update resource tracking
        if let Err(e) = self.limiter.update_memory_usage(&self.session.id, value_size, true).await {
            error!("Failed to update memory usage: {}", e);
        }

        // If TTL is specified, schedule cleanup
        if let Some(ttl) = ttl_secs {
            let _session_id = self.session.id;
            let key_copy = key.to_string();
            let session_meta = self.session.metadata.clone();

            tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_secs(ttl)).await;
                session_meta.remove(&key_copy);
            });
        }

        let result = json!({
            "key": key,
            "stored": true,
            "timestamp": chrono::Utc::now(),
            "ttl_secs": ttl_secs,
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle memory get
    async fn handle_memory_get(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        let key = params
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: key"))?;

        let value = self.session.metadata.get(key).map(|v| v.clone());

        let result = json!({
            "key": key,
            "value": value,
            "found": value.is_some(),
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle task create
    async fn handle_task_create(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        let task_type = params
            .get("task_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: task_type"))?;

        let description = params
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: description"))?;

        let priority_str = params
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("medium");

        let priority = match priority_str {
            "low" => TaskPriority::Low,
            "medium" => TaskPriority::Medium,
            "high" => TaskPriority::High,
            "critical" => TaskPriority::Critical,
            _ => TaskPriority::Medium,
        };

        let assigned_agent = params
            .get("assigned_agent")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        // Create task
        let task_id = self
            .orchestrator
            .create_task(
                task_type.to_string(),
                description.to_string(),
                priority,
                assigned_agent,
            )
            .await?;

        let result = json!({
            "task_id": task_id,
            "task_type": task_type,
            "description": description,
            "priority": priority_str,
            "status": "pending",
            "created_at": chrono::Utc::now(),
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle workflow execute
    async fn handle_workflow_execute(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        // Validate parameters first, including path traversal protection
        if let Err(e) = self.validator.validate_workflow_params(params) {
            return Ok(self.create_detailed_error(
                id,
                -32602,
                "VALIDATION_ERROR",
                &e.to_string(),
                vec![
                    "Check workflow path format",
                    "Ensure path is relative and safe"
                ]
            ));
        }

        // Safe to unwrap after validation
        let workflow_path = params
            .get("workflow_path")
            .and_then(|v| v.as_str())
            .unwrap();

        let parameters = params.get("parameters").cloned().unwrap_or(json!({}));

        let async_execution = params
            .get("async_execution")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let workflow_id = Uuid::new_v4();

        if async_execution {
            // Execute asynchronously
            let orchestrator = self.orchestrator.clone();
            let workflow_path = workflow_path.to_string();

            tokio::spawn(async move {
                match orchestrator
                    .execute_workflow(&workflow_id, &workflow_path, parameters)
                    .await
                {
                    Ok(result) => {
                        info!("Workflow completed: {:?}", result);
                    }
                    Err(e) => {
                        error!("Workflow failed: {}", e);
                    }
                }
            });

            let result = json!({
                "workflow_id": workflow_id,
                "status": "started",
                "async": true,
            });

            Ok(McpResponse::success(id, result))
        } else {
            // Execute synchronously
            let result = self
                .orchestrator
                .execute_workflow(&workflow_id, workflow_path, parameters)
                .await?;

            Ok(McpResponse::success(
                id,
                json!({
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "result": result,
                }),
            ))
        }
    }

    /// Handle agent list
    async fn handle_agent_list(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        let include_inactive = params
            .get("include_inactive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let sort_by = params
            .get("sort_by")
            .and_then(|v| v.as_str())
            .unwrap_or("created_at");

        let agents = self.orchestrator.list_agents(include_inactive).await?;

        let result = json!({
            "agents": agents,
            "count": agents.len(),
            "include_inactive": include_inactive,
            "sorted_by": sort_by,
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle agent metrics
    async fn handle_agent_metrics(
        &self,
        id: Option<Value>,
        params: &Value,
    ) -> anyhow::Result<McpResponse> {
        let agent_id = params
            .get("agent_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());

        let metric_type = params
            .get("metric")
            .and_then(|v| v.as_str())
            .unwrap_or("all");

        let metrics = if let Some(agent_id) = agent_id {
            // Get metrics for specific agent
            self.orchestrator.get_agent_metrics(&agent_id).await?
        } else {
            // Get metrics for all agents
            self.orchestrator.get_all_agent_metrics().await?
        };

        let filtered_metrics = match metric_type {
            "cpu" => json!({
                "cpu_usage": metrics.get("cpu_usage").unwrap_or(&json!({})),
                "cpu_utilization": metrics.get("cpu_utilization").unwrap_or(&json!({})),
            }),
            "memory" => json!({
                "memory_usage": metrics.get("memory_usage").unwrap_or(&json!({})),
                "memory_peak": metrics.get("memory_peak").unwrap_or(&json!({})),
            }),
            "tasks" => json!({
                "tasks_completed": metrics.get("tasks_completed").unwrap_or(&json!(0)),
                "tasks_failed": metrics.get("tasks_failed").unwrap_or(&json!(0)),
                "tasks_in_progress": metrics.get("tasks_in_progress").unwrap_or(&json!(0)),
                "average_task_duration": metrics.get("average_task_duration").unwrap_or(&json!(0)),
            }),
            "performance" => json!({
                "throughput": metrics.get("throughput").unwrap_or(&json!({})),
                "response_time": metrics.get("response_time").unwrap_or(&json!({})),
                "error_rate": metrics.get("error_rate").unwrap_or(&json!({})),
            }),
            "all" => metrics,
            _ => metrics,
        };

        let result = json!({
            "agent_id": agent_id.map(|id| id.to_string()),
            "metric_type": metric_type,
            "metrics": filtered_metrics,
            "timestamp": chrono::Utc::now(),
        });

        Ok(McpResponse::success(id, result))
    }

    /// Handle subscribe
    async fn handle_subscribe(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        // Implementation would handle event subscriptions
        Ok(McpResponse::success(
            request.id,
            json!({
                "subscribed": true,
            }),
        ))
    }

    /// Handle unsubscribe
    async fn handle_unsubscribe(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        // Implementation would handle event unsubscriptions
        Ok(McpResponse::success(
            request.id,
            json!({
                "unsubscribed": true,
            }),
        ))
    }

    /// Handle swarm status
    async fn handle_swarm_status(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        let status = self.orchestrator.get_status().await?;
        Ok(McpResponse::success(request.id, json!(status)))
    }

    /// Handle swarm metrics
    async fn handle_swarm_metrics(&self, request: McpRequest) -> anyhow::Result<McpResponse> {
        let metrics = self.orchestrator.get_metrics().await?;
        Ok(McpResponse::success(request.id, json!(metrics)))
    }
}
