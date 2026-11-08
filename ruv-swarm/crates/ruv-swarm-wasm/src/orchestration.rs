//! Local, bounded agent state and capability assignment. Handlers execute outside WASM.
use serde::Deserialize;
use wasm_bindgen::prelude::*;

const MAX_AGENTS: usize = 1000;
const MAX_CONFIG_BYTES: usize = 64 * 1024;
const MAX_CAPABILITIES: usize = 64;

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_bytes && !value.chars().any(char::is_control)
}

fn validate_capabilities(values: &[String]) -> Result<(), String> {
    if values.len() > MAX_CAPABILITIES || values.iter().any(|value| !valid_text(value, 128)) {
        return Err("Require at most 64 nonempty capability names, each at most 128 bytes".into());
    }
    Ok(())
}

fn parse_config<T: serde::de::DeserializeOwned>(config: &str) -> Result<T, String> {
    if config.len() > MAX_CONFIG_BYTES {
        return Err("Configuration exceeds 64 KiB".into());
    }
    serde_json::from_str(config).map_err(|error| format!("Invalid configuration: {error}"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct SpawnConfig {
    id: Option<String>,
    name: Option<String>,
    #[serde(rename = "type", default = "generic_type")]
    agent_type: String,
    #[serde(default)]
    capabilities: Vec<String>,
}

fn generic_type() -> String {
    "generic".into()
}
fn normal_priority() -> String {
    "normal".into()
}
fn default_max_agents() -> usize {
    3
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct TaskConfig {
    description: String,
    #[serde(default = "normal_priority")]
    priority: String,
    #[serde(default = "default_max_agents")]
    max_agents: usize,
    #[serde(default)]
    required_capabilities: Vec<String>,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmAgent {
    id: String,
    name: String,
    agent_type: String,
    status: String,
    capabilities: Vec<String>,
}

#[wasm_bindgen]
impl WasmAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str, agent_type: &str) -> Result<Self, JsValue> {
        Self::checked_new(id, agent_type).map_err(|error| JsValue::from_str(&error))
    }
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn agent_type(&self) -> String {
        self.agent_type.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        self.status.clone()
    }
    pub fn set_status(&mut self, status: &str) -> Result<(), JsValue> {
        self.set_status_checked(status)
            .map_err(|error| JsValue::from_str(&error))
    }
    pub fn add_capability(&mut self, capability: &str) -> Result<(), JsValue> {
        if !valid_text(capability, 128) || self.capabilities.len() >= MAX_CAPABILITIES {
            return Err(JsValue::from_str(
                "Capability is invalid or the limit has been reached",
            ));
        }
        if !self.has_capability(capability) {
            self.capabilities.push(capability.into());
        }
        Ok(())
    }
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|value| value == capability)
    }
}

impl WasmAgent {
    fn checked_new(id: &str, agent_type: &str) -> Result<Self, String> {
        if !valid_text(id, 256) || !valid_text(agent_type, 128) {
            return Err(
                "Agent ID/type must be nonempty, bounded text without control characters".into(),
            );
        }
        Ok(Self {
            id: id.into(),
            name: id.into(),
            agent_type: agent_type.into(),
            status: "idle".into(),
            capabilities: vec![],
        })
    }
    fn set_status_checked(&mut self, status: &str) -> Result<(), String> {
        if !matches!(status, "idle" | "busy" | "offline" | "error") {
            return Err("Status must be idle, busy, offline or error".into());
        }
        self.status = status.into();
        Ok(())
    }
}

#[wasm_bindgen]
pub struct WasmTaskResult {
    task_id: String,
    description: String,
    status: String,
    assigned_agents: Vec<String>,
    priority: String,
}

#[wasm_bindgen]
impl WasmTaskResult {
    #[wasm_bindgen(getter)]
    pub fn task_id(&self) -> String {
        self.task_id.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.description.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        self.status.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn assigned_agents(&self) -> Vec<String> {
        self.assigned_agents.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn priority(&self) -> String {
        self.priority.clone()
    }
}

#[wasm_bindgen]
pub struct WasmSwarmOrchestrator {
    agents: Vec<WasmAgent>,
    topology: String,
    task_counter: u64,
    agent_counter: u64,
}

#[wasm_bindgen]
impl WasmSwarmOrchestrator {
    #[wasm_bindgen(constructor)]
    pub fn new(topology: &str) -> Result<Self, JsValue> {
        Self::checked_new(topology).map_err(|error| JsValue::from_str(&error))
    }
    pub fn spawn(&mut self, config: &str) -> Result<String, JsValue> {
        self.spawn_checked(config)
            .map_err(|error| JsValue::from_str(&error))
    }
    pub fn orchestrate(&mut self, config: &str) -> Result<WasmTaskResult, JsValue> {
        self.orchestrate_checked(config)
            .map_err(|error| JsValue::from_str(&error))
    }
    pub fn add_agent(&mut self, agent_id: &str) -> Result<(), JsValue> {
        let config = serde_json::json!({"id": agent_id, "type": "generic"}).to_string();
        self.spawn(&config).map(|_| ())
    }
    /// Release or mark a worker after the application executes an assigned task.
    pub fn update_agent_status(&mut self, id: &str, status: &str) -> Result<(), JsValue> {
        self.update_status_checked(id, status)
            .map_err(|error| JsValue::from_str(&error))
    }
    pub fn get_agent_count(&self) -> usize {
        self.agents.len()
    }
    pub fn get_topology(&self) -> String {
        self.topology.clone()
    }
    pub fn get_status(&self, detailed: bool) -> String {
        let mut status = serde_json::json!({
            "agents": {"total": self.agents.len(),
                "idle": self.agents.iter().filter(|agent| agent.status == "idle").count(),
                "busy": self.agents.iter().filter(|agent| agent.status == "busy").count()},
            "topology": self.topology,
            "execution": "application_owned",
        });
        if detailed {
            status["agent_details"] = self
                .agents
                .iter()
                .map(|agent| {
                    serde_json::json!({
                        "id": agent.id, "name": agent.name, "type": agent.agent_type,
                        "status": agent.status, "capabilities": agent.capabilities,
                    })
                })
                .collect();
        }
        status.to_string()
    }
}

impl WasmSwarmOrchestrator {
    fn checked_new(topology: &str) -> Result<Self, String> {
        if !matches!(topology, "mesh" | "ring" | "star" | "hierarchical") {
            return Err("Topology must be mesh, ring, star or hierarchical".into());
        }
        Ok(Self {
            agents: vec![],
            topology: topology.into(),
            task_counter: 0,
            agent_counter: 0,
        })
    }
    fn spawn_checked(&mut self, config: &str) -> Result<String, String> {
        let config: SpawnConfig = parse_config(config)?;
        if self.agents.len() >= MAX_AGENTS {
            return Err("Agent limit reached (1000)".into());
        }
        validate_capabilities(&config.capabilities)?;
        let mut counter = self
            .agent_counter
            .checked_add(1)
            .ok_or("Agent ID counter exhausted")?;
        let id = if let Some(id) = config.id {
            id
        } else {
            loop {
                let id = format!("agent-{counter}");
                if !self.agents.iter().any(|agent| agent.id == id) {
                    break id;
                }
                counter = counter.checked_add(1).ok_or("Agent ID counter exhausted")?;
            }
        };
        if self.agents.iter().any(|agent| agent.id == id) {
            return Err("Duplicate agent ID".into());
        }
        let mut agent = WasmAgent::checked_new(&id, &config.agent_type)?;
        if let Some(name) = config.name {
            if !valid_text(&name, 256) {
                return Err("Agent name is invalid".into());
            }
            agent.name = name;
        }
        for capability in config.capabilities {
            if !agent.has_capability(&capability) {
                agent.capabilities.push(capability);
            }
        }
        let result = serde_json::json!({"agent_id": agent.id, "name": agent.name,
            "type": agent.agent_type, "status": agent.status, "capabilities": agent.capabilities});
        self.agents.push(agent);
        self.agent_counter = counter;
        Ok(result.to_string())
    }
    fn orchestrate_checked(&mut self, config: &str) -> Result<WasmTaskResult, String> {
        let config: TaskConfig = parse_config(config)?;
        validate_capabilities(&config.required_capabilities)?;
        if config.description.trim().is_empty()
            || config.description.len() > 16 * 1024
            || config
                .description
                .chars()
                .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
            || !matches!(
                config.priority.as_str(),
                "low" | "normal" | "medium" | "high" | "critical"
            )
            || !(1..=MAX_AGENTS).contains(&config.max_agents)
        {
            return Err("Invalid task description, priority or maxAgents (1..=1000)".into());
        }
        let counter = self
            .task_counter
            .checked_add(1)
            .ok_or("Task ID counter exhausted")?;
        let mut assigned_agents = vec![];
        for agent in &mut self.agents {
            if assigned_agents.len() >= config.max_agents {
                break;
            }
            if agent.status == "idle"
                && config
                    .required_capabilities
                    .iter()
                    .all(|capability| agent.has_capability(capability))
            {
                agent.status = "busy".into();
                assigned_agents.push(agent.id.clone());
            }
        }
        self.task_counter = counter;
        Ok(WasmTaskResult {
            task_id: format!("task-{counter}"),
            description: config.description,
            status: if assigned_agents.is_empty() {
                "unassigned"
            } else {
                "assigned"
            }
            .into(),
            assigned_agents,
            priority: config.priority,
        })
    }
    fn update_status_checked(&mut self, id: &str, status: &str) -> Result<(), String> {
        self.agents
            .iter_mut()
            .find(|agent| agent.id == id)
            .ok_or("Agent not found")?
            .set_status_checked(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn honors_spawn_configuration_and_rejects_unsupported_fields() {
        let mut swarm = WasmSwarmOrchestrator::checked_new("mesh").unwrap();
        let result = swarm
            .spawn_checked(
                r#"{"id":"coder","name":"Code worker","type":"coder","capabilities":["rust"]}"#,
            )
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(result["type"], "coder");
        assert_eq!(result["name"], "Code worker");
        assert_eq!(result["capabilities"], serde_json::json!(["rust"]));
        assert!(result.get("neural_network_id").is_none());
        assert!(swarm.spawn_checked(r#"{"id":"coder"}"#).is_err());
        assert!(swarm.spawn_checked(r#"{"model":"LSTM"}"#).is_err());
        assert_eq!(swarm.get_agent_count(), 1);
        swarm.spawn_checked(r#"{"id":"agent-3"}"#).unwrap();
        assert!(swarm.spawn_checked("{}").unwrap().contains("agent-4"));
    }
    #[test]
    fn assigns_actual_task_fields_and_only_compatible_idle_agents() {
        let mut swarm = WasmSwarmOrchestrator::checked_new("ring").unwrap();
        swarm
            .spawn_checked(r#"{"id":"rust-worker","capabilities":["rust"]}"#)
            .unwrap();
        swarm
            .spawn_checked(r#"{"id":"other","capabilities":["python"]}"#)
            .unwrap();
        let config = r#"{"description":"Review Rust","priority":"high","maxAgents":1,"requiredCapabilities":["rust"]}"#;
        let task = swarm.orchestrate_checked(config).unwrap();
        assert_eq!(task.description(), "Review Rust");
        assert_eq!(task.priority(), "high");
        assert_eq!(task.status(), "assigned");
        assert_eq!(task.assigned_agents(), vec!["rust-worker"]);
        assert_eq!(
            swarm.orchestrate_checked(config).unwrap().status(),
            "unassigned"
        );
        swarm.update_status_checked("rust-worker", "idle").unwrap();
        assert_eq!(
            swarm.orchestrate_checked(config).unwrap().assigned_agents(),
            vec!["rust-worker"]
        );
    }
    #[test]
    fn rejects_invalid_tasks_before_mutating_worker_state() {
        let mut swarm = WasmSwarmOrchestrator::checked_new("mesh").unwrap();
        swarm.spawn_checked("{}").unwrap();
        assert!(swarm
            .orchestrate_checked(r#"{"description":"task","maxAgents":0}"#)
            .is_err());
        assert_eq!(swarm.agents[0].status, "idle");
        assert_eq!(swarm.task_counter, 0);
        assert!(swarm
            .spawn_checked(&"x".repeat(MAX_CONFIG_BYTES + 1))
            .is_err());
        assert!(WasmSwarmOrchestrator::checked_new("quantum").is_err());
    }
}
