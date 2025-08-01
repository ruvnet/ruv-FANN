use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::AgentType;

#[derive(Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub max_concurrent_tasks: Option<u32>,
    pub memory_limit: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct TaskRequest {
    pub id: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub timeout: Option<u32>,
}

#[derive(Serialize, Deserialize)]
pub struct TaskResponse {
    pub task_id: String,
    pub status: String,
    pub result: serde_json::Value,
    pub execution_time: f64,
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct AgentMetrics {
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub average_execution_time: f64,
    pub memory_usage: u64,
    pub cpu_usage: f64,
}

// TODO: This will be replaced with actual Agent implementation
struct Agent {
    id: String,
    agent_type: AgentType,
    status: String,
    tasks_completed: u32,
}

#[wasm_bindgen]
pub struct JsAgent {
    inner: Agent,
}

impl JsAgent {
    pub fn new(id: String, agent_type: String) -> Self {
        let agent_type = match agent_type.as_str() {
            "researcher" => AgentType::Researcher,
            "coder" => AgentType::Coder,
            "analyst" => AgentType::Analyst,
            "optimizer" => AgentType::Optimizer,
            "coordinator" => AgentType::Coordinator,
            _ => AgentType::Researcher,
        };

        let inner = Agent {
            id,
            agent_type,
            status: "idle".to_string(),
            tasks_completed: 0,
        };

        JsAgent { inner }
    }
}

#[wasm_bindgen]
impl JsAgent {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn agent_type(&self) -> String {
        format!("{:?}", self.inner.agent_type)
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        self.inner.status.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn tasks_completed(&self) -> u32 {
        self.inner.tasks_completed
    }

    #[wasm_bindgen]
    pub async fn execute(&mut self, task: JsValue) -> Result<JsValue, JsValue> {
        let task_request: TaskRequest = serde_wasm_bindgen::from_value(task)
            .map_err(|e| JsValue::from_str(&format!("Invalid task request: {}", e)))?;

        // Update status
        self.inner.status = "working".to_string();

        // TODO: Implement actual task execution
        // For now, simulate work with a delay
        let start_time = js_sys::Date::now();
        
        // Simulate async work
        wasm_bindgen_futures::JsFuture::from(js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 100)
                .unwrap();
        }))
        .await
        .map_err(|e| JsValue::from_str(&format!("Execution failed: {:?}", e)))?;

        let execution_time = (js_sys::Date::now() - start_time) / 1000.0;

        // Update metrics
        self.inner.tasks_completed += 1;
        self.inner.status = "idle".to_string();

        let response = TaskResponse {
            task_id: task_request.id,
            status: "completed".to_string(),
            result: serde_json::json!({
                "message": "Task executed successfully",
                "agent_type": format!("{:?}", self.inner.agent_type),
                "description": task_request.description,
            }),
            execution_time,
            error: None,
        };

        serde_wasm_bindgen::to_value(&response)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize response: {}", e)))
    }

    #[wasm_bindgen]
    pub fn get_metrics(&self) -> Result<JsValue, JsValue> {
        let metrics = AgentMetrics {
            tasks_completed: self.inner.tasks_completed,
            tasks_failed: 0,
            average_execution_time: 0.1,
            memory_usage: 0,
            cpu_usage: 0.0,
        };

        serde_wasm_bindgen::to_value(&metrics)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize metrics: {}", e)))
    }

    #[wasm_bindgen]
    pub fn get_capabilities(&self) -> Result<JsValue, JsValue> {
        let capabilities = match self.inner.agent_type {
            AgentType::Researcher => vec![
                "information_gathering",
                "data_analysis",
                "report_generation",
                "web_search",
            ],
            AgentType::Coder => vec![
                "code_generation",
                "refactoring",
                "bug_fixing",
                "test_writing",
            ],
            AgentType::Analyst => vec![
                "data_analysis",
                "pattern_recognition",
                "statistical_modeling",
                "visualization",
            ],
            AgentType::Optimizer => vec![
                "performance_tuning",
                "resource_optimization",
                "algorithm_optimization",
                "cost_reduction",
            ],
            AgentType::Coordinator => vec![
                "task_distribution",
                "workflow_management",
                "resource_allocation",
                "progress_tracking",
            ],
        };

        serde_wasm_bindgen::to_value(&capabilities)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize capabilities: {}", e)))
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.status = "idle".to_string();
        self.inner.tasks_completed = 0;
    }
}