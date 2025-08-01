//! MCP Tool definitions for RUV-Swarm

use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Tool parameter schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub param_type: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// Tool definition
#[derive(Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    #[serde(skip)]
    pub handler: Option<Arc<dyn ToolHandler>>,
}

/// Tool handler trait
pub trait ToolHandler: Send + Sync {
    fn handle(&self, params: Value) -> anyhow::Result<Value>;
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("handler", &self.handler.as_ref().map(|_| "<handler>"))
            .finish()
    }
}

/// Tool registry
pub struct ToolRegistry {
    tools: DashMap<String, Tool>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: DashMap::new(),
        }
    }

    pub fn register(&self, tool: Tool) {
        let name = tool.name.clone();
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<Tool> {
        self.tools.get(name).map(|t| t.clone())
    }

    pub fn list_tools(&self) -> Vec<Tool> {
        self.tools
            .iter()
            .map(|entry| {
                let mut tool = entry.value().clone();
                // Don't serialize the handler
                tool.handler = None;
                tool
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.tools.len()
    }
}

/// Register all tools
pub fn register_tools(registry: &ToolRegistry) {
    // Spawn agent tool
    registry.register(Tool {
        name: "ruv-swarm.spawn".to_string(),
        description: "Spawn a new agent in the swarm".to_string(),
        parameters: vec![
            ToolParameter {
                name: "agent_type".to_string(),
                description: "Type of agent to spawn".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: Some(vec![
                    "researcher".to_string(),
                    "coder".to_string(),
                    "analyst".to_string(),
                    "tester".to_string(),
                    "reviewer".to_string(),
                    "documenter".to_string(),
                ]),
            },
            ToolParameter {
                name: "name".to_string(),
                description: "Optional name for the agent".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "capabilities".to_string(),
                description: "Agent capabilities configuration".to_string(),
                param_type: "object".to_string(),
                required: false,
                default: Some(json!({})),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Orchestrate task tool
    registry.register(Tool {
        name: "ruv-swarm.orchestrate".to_string(),
        description: "Orchestrate a swarm task with specified strategy".to_string(),
        parameters: vec![
            ToolParameter {
                name: "objective".to_string(),
                description: "Task objective or goal".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "strategy".to_string(),
                description: "Orchestration strategy".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("development")),
                enum_values: Some(vec![
                    "research".to_string(),
                    "development".to_string(),
                    "analysis".to_string(),
                    "testing".to_string(),
                    "optimization".to_string(),
                    "maintenance".to_string(),
                ]),
            },
            ToolParameter {
                name: "mode".to_string(),
                description: "Coordination mode".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("hierarchical")),
                enum_values: Some(vec![
                    "centralized".to_string(),
                    "distributed".to_string(),
                    "hierarchical".to_string(),
                    "mesh".to_string(),
                    "hybrid".to_string(),
                ]),
            },
            ToolParameter {
                name: "max_agents".to_string(),
                description: "Maximum number of agents".to_string(),
                param_type: "integer".to_string(),
                required: false,
                default: Some(json!(5)),
                enum_values: None,
            },
            ToolParameter {
                name: "parallel".to_string(),
                description: "Enable parallel execution".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                default: Some(json!(true)),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Query swarm state tool
    registry.register(Tool {
        name: "ruv-swarm.query".to_string(),
        description: "Query the current swarm state and active agents".to_string(),
        parameters: vec![
            ToolParameter {
                name: "filter".to_string(),
                description: "Filter criteria for agents".to_string(),
                param_type: "object".to_string(),
                required: false,
                default: Some(json!({})),
                enum_values: None,
            },
            ToolParameter {
                name: "include_metrics".to_string(),
                description: "Include performance metrics".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                default: Some(json!(false)),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Monitor events tool
    registry.register(Tool {
        name: "ruv-swarm.monitor".to_string(),
        description: "Subscribe to swarm events and monitor activity".to_string(),
        parameters: vec![
            ToolParameter {
                name: "event_types".to_string(),
                description: "Types of events to monitor".to_string(),
                param_type: "array".to_string(),
                required: false,
                default: Some(json!(["all"])),
                enum_values: None,
            },
            ToolParameter {
                name: "agent_filter".to_string(),
                description: "Filter events by agent ID or type".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "duration_secs".to_string(),
                description: "Monitoring duration in seconds".to_string(),
                param_type: "integer".to_string(),
                required: false,
                default: Some(json!(60)),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Optimize performance tool
    registry.register(Tool {
        name: "ruv-swarm.optimize".to_string(),
        description: "Optimize swarm performance and resource allocation".to_string(),
        parameters: vec![
            ToolParameter {
                name: "target_metric".to_string(),
                description: "Metric to optimize for".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("throughput")),
                enum_values: Some(vec![
                    "throughput".to_string(),
                    "latency".to_string(),
                    "resource_usage".to_string(),
                    "cost".to_string(),
                    "quality".to_string(),
                ]),
            },
            ToolParameter {
                name: "constraints".to_string(),
                description: "Optimization constraints".to_string(),
                param_type: "object".to_string(),
                required: false,
                default: Some(json!({})),
                enum_values: None,
            },
            ToolParameter {
                name: "auto_apply".to_string(),
                description: "Automatically apply optimizations".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                default: Some(json!(false)),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Memory store tool
    registry.register(Tool {
        name: "ruv-swarm.memory.store".to_string(),
        description: "Store data in swarm persistent memory".to_string(),
        parameters: vec![
            ToolParameter {
                name: "key".to_string(),
                description: "Memory key identifier".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "value".to_string(),
                description: "Data to store".to_string(),
                param_type: "any".to_string(),
                required: true,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "ttl_secs".to_string(),
                description: "Time to live in seconds".to_string(),
                param_type: "integer".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Memory retrieve tool
    registry.register(Tool {
        name: "ruv-swarm.memory.get".to_string(),
        description: "Retrieve data from swarm memory".to_string(),
        parameters: vec![ToolParameter {
            name: "key".to_string(),
            description: "Memory key to retrieve".to_string(),
            param_type: "string".to_string(),
            required: true,
            default: None,
            enum_values: None,
        }],
        handler: None,
    });

    // Task create tool
    registry.register(Tool {
        name: "ruv-swarm.task.create".to_string(),
        description: "Create a new task for agents".to_string(),
        parameters: vec![
            ToolParameter {
                name: "task_type".to_string(),
                description: "Type of task".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "description".to_string(),
                description: "Task description".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "priority".to_string(),
                description: "Task priority".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("medium")),
                enum_values: Some(vec![
                    "low".to_string(),
                    "medium".to_string(),
                    "high".to_string(),
                    "critical".to_string(),
                ]),
            },
            ToolParameter {
                name: "assigned_agent".to_string(),
                description: "Agent to assign the task to".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Workflow execute tool
    registry.register(Tool {
        name: "ruv-swarm.workflow.execute".to_string(),
        description: "Execute a workflow automation file".to_string(),
        parameters: vec![
            ToolParameter {
                name: "workflow_path".to_string(),
                description: "Path to workflow file".to_string(),
                param_type: "string".to_string(),
                required: true,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "parameters".to_string(),
                description: "Workflow parameters".to_string(),
                param_type: "object".to_string(),
                required: false,
                default: Some(json!({})),
                enum_values: None,
            },
            ToolParameter {
                name: "async_execution".to_string(),
                description: "Execute asynchronously".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                default: Some(json!(false)),
                enum_values: None,
            },
        ],
        handler: None,
    });

    // Agent list tool
    registry.register(Tool {
        name: "ruv-swarm.agent.list".to_string(),
        description: "List all active agents in the swarm".to_string(),
        parameters: vec![
            ToolParameter {
                name: "include_inactive".to_string(),
                description: "Include inactive agents".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                default: Some(json!(false)),
                enum_values: None,
            },
            ToolParameter {
                name: "sort_by".to_string(),
                description: "Field to sort by".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("created_at")),
                enum_values: Some(vec![
                    "created_at".to_string(),
                    "name".to_string(),
                    "type".to_string(),
                    "status".to_string(),
                ]),
            },
        ],
        handler: None,
    });

    // Agent metrics tool
    registry.register(Tool {
        name: "ruv-swarm.agent.metrics".to_string(),
        description: "Get performance metrics for agents".to_string(),
        parameters: vec![
            ToolParameter {
                name: "agent_id".to_string(),
                description: "Specific agent ID (optional)".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: None,
                enum_values: None,
            },
            ToolParameter {
                name: "metric".to_string(),
                description: "Metric type to retrieve".to_string(),
                param_type: "string".to_string(),
                required: false,
                default: Some(json!("all")),
                enum_values: Some(vec![
                    "all".to_string(),
                    "cpu".to_string(),
                    "memory".to_string(),
                    "tasks".to_string(),
                    "performance".to_string(),
                ]),
            },
        ],
        handler: None,
    });
}
