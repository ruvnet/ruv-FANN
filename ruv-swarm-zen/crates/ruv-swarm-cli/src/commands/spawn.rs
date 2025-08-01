use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::config::Config;
use crate::output::{OutputHandler, StatusLevel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
    pub memory: Option<String>,
    pub parent_id: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Initializing,
    Ready,
    Busy,
    Idle,
    Error(String),
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub avg_task_duration_ms: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            tasks_completed: 0,
            tasks_failed: 0,
            avg_task_duration_ms: 0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
        }
    }
}

/// Execute the spawn command
pub async fn execute(
    config: &Config,
    output: &OutputHandler,
    agent_type: String,
    capabilities: Vec<String>,
    name: Option<String>,
    memory: Option<String>,
    parent: Option<String>,
) -> Result<()> {
    output.section("Spawning New Agent");

    // Validate agent type
    let valid_types = vec![
        "researcher",
        "coder",
        "analyst",
        "reviewer",
        "orchestrator",
        "tester",
        "debugger",
        "documenter",
        "architect",
        "optimizer",
    ];

    if !valid_types.contains(&agent_type.as_str()) {
        output.error(&format!(
            "Invalid agent type '{}'. Valid options: {}",
            agent_type,
            valid_types.join(", ")
        ));
        return Err(anyhow::anyhow!("Invalid agent type"));
    }

    // Load current swarm configuration
    let swarm_config = load_current_swarm(output).await?;

    // Check if we've reached the agent limit
    let current_agent_count = get_agent_count(&swarm_config).await?;
    if current_agent_count >= config.swarm.max_agents {
        output.error(&format!(
            "Cannot spawn new agent: swarm limit of {} agents reached",
            config.swarm.max_agents
        ));
        return Err(anyhow::anyhow!("Agent limit reached"));
    }

    // Generate agent ID and name
    let agent_id = Uuid::new_v4().to_string();
    let agent_name = name.unwrap_or_else(|| format!("{}-{}", agent_type, &agent_id[..8]));

    // Merge default capabilities with provided ones
    let mut all_capabilities = get_default_capabilities(&agent_type);
    all_capabilities.extend(capabilities);
    all_capabilities.sort();
    all_capabilities.dedup();

    // Validate parent agent if hierarchical topology
    if swarm_config.topology == "hierarchical" && parent.is_none() && agent_type != "orchestrator" {
        output.error("Hierarchical topology requires a parent agent ID (except for orchestrators)");
        return Err(anyhow::anyhow!(
            "Parent agent required for hierarchical topology"
        ));
    }

    if let Some(parent_id) = &parent {
        // Verify parent exists
        if !agent_exists(parent_id, &swarm_config).await? {
            output.error(&format!("Parent agent '{}' not found", parent_id));
            return Err(anyhow::anyhow!("Parent agent not found"));
        }
    }

    // Create the agent
    let agent = Agent {
        id: agent_id.clone(),
        name: agent_name.clone(),
        agent_type: agent_type.clone(),
        capabilities: all_capabilities.clone(),
        status: AgentStatus::Initializing,
        memory: memory.clone(),
        parent_id: parent.clone(),
        created_at: Utc::now(),
        last_heartbeat: Utc::now(),
        metrics: AgentMetrics::default(),
    };

    // Display agent details
    output.section("Agent Configuration");
    output.key_value(&[
        ("ID".to_string(), agent.id.clone()),
        ("Name".to_string(), agent.name.clone()),
        ("Type".to_string(), agent.agent_type.clone()),
        ("Capabilities".to_string(), agent.capabilities.join(", ")),
        (
            "Parent".to_string(),
            agent
                .parent_id
                .clone()
                .unwrap_or_else(|| "None".to_string()),
        ),
        (
            "Memory".to_string(),
            agent.memory.clone().unwrap_or_else(|| "None".to_string()),
        ),
    ]);

    // Spawn the agent
    let spinner = output.spinner("Initializing agent...");

    // Initialize agent runtime
    initialize_agent_runtime(&agent, config).await?;

    // Register with swarm
    register_agent_with_swarm(&agent, &swarm_config).await?;

    // Set up agent connections
    setup_agent_connections(&agent, &swarm_config).await?;

    // Load initial memory if provided
    if let Some(memory_content) = &agent.memory {
        load_agent_memory(&agent, memory_content).await?;
    }

    // Start agent heartbeat
    start_agent_heartbeat(&agent, config).await?;

    if let Some(pb) = spinner {
        pb.finish_with_message("Agent spawned successfully");
    }

    // Update agent status to ready
    update_agent_status(&agent.id, AgentStatus::Ready).await?;

    output.success(&format!(
        "Agent '{}' ({}) spawned successfully!",
        agent_name, agent_type
    ));

    // Show agent capabilities
    output.section("Agent Capabilities");
    output.list(&agent.capabilities, false);

    // Show next steps
    output.section("Next Steps");
    output.list(
        &[
            format!(
                "View agent status: ruv-swarm status --agent-type {}",
                agent_type
            ),
            format!("Assign task to agent: ruv-swarm orchestrate <strategy> <task>"),
            format!(
                "Monitor agent activity: ruv-swarm monitor --filter agent:{}",
                agent.id
            ),
        ],
        true,
    );

    Ok(())
}

fn get_default_capabilities(agent_type: &str) -> Vec<String> {
    match agent_type {
        "researcher" => vec![
            "web_search".to_string(),
            "document_analysis".to_string(),
            "summarization".to_string(),
            "fact_checking".to_string(),
        ],
        "coder" => vec![
            "code_generation".to_string(),
            "refactoring".to_string(),
            "debugging".to_string(),
            "testing".to_string(),
            "documentation".to_string(),
        ],
        "analyst" => vec![
            "data_analysis".to_string(),
            "visualization".to_string(),
            "reporting".to_string(),
            "pattern_recognition".to_string(),
        ],
        "reviewer" => vec![
            "code_review".to_string(),
            "quality_assurance".to_string(),
            "best_practices".to_string(),
            "security_audit".to_string(),
        ],
        "orchestrator" => vec![
            "coordination".to_string(),
            "task_distribution".to_string(),
            "resource_management".to_string(),
            "conflict_resolution".to_string(),
        ],
        "tester" => vec![
            "unit_testing".to_string(),
            "integration_testing".to_string(),
            "performance_testing".to_string(),
            "test_generation".to_string(),
        ],
        "debugger" => vec![
            "error_analysis".to_string(),
            "root_cause_analysis".to_string(),
            "trace_analysis".to_string(),
            "fix_suggestion".to_string(),
        ],
        "documenter" => vec![
            "api_documentation".to_string(),
            "user_guides".to_string(),
            "technical_writing".to_string(),
            "diagram_generation".to_string(),
        ],
        "architect" => vec![
            "system_design".to_string(),
            "architecture_review".to_string(),
            "technology_selection".to_string(),
            "scalability_planning".to_string(),
        ],
        "optimizer" => vec![
            "performance_optimization".to_string(),
            "resource_optimization".to_string(),
            "algorithm_optimization".to_string(),
            "cost_optimization".to_string(),
        ],
        _ => vec!["general".to_string()],
    }
}

async fn load_current_swarm(output: &OutputHandler) -> Result<crate::commands::init::SwarmInit> {
    let config_dir = directories::ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| std::path::Path::new(".").to_path_buf());

    let current_file = config_dir.join("current-swarm.json");

    if !current_file.exists() {
        output.error("No active swarm found. Run 'ruv-swarm init' first.");
        return Err(anyhow::anyhow!("No active swarm"));
    }

    let content = std::fs::read_to_string(current_file)?;
    serde_json::from_str(&content).context("Failed to parse swarm configuration")
}

async fn get_agent_count(swarm_config: &crate::commands::init::SwarmInit) -> Result<usize> {
    // In a real implementation, this would query the persistence backend
    // For now, we'll simulate by reading from a file
    let agents_file = get_agents_file(swarm_config)?;

    if agents_file.exists() {
        let content = std::fs::read_to_string(&agents_file)?;
        let agents: Vec<Agent> = serde_json::from_str(&content).unwrap_or_default();
        Ok(agents.len())
    } else {
        Ok(swarm_config.initial_agents.len())
    }
}

async fn agent_exists(
    agent_id: &str,
    swarm_config: &crate::commands::init::SwarmInit,
) -> Result<bool> {
    let agents_file = get_agents_file(swarm_config)?;

    if agents_file.exists() {
        let content = std::fs::read_to_string(&agents_file)?;
        let agents: Vec<Agent> = serde_json::from_str(&content).unwrap_or_default();
        Ok(agents.iter().any(|a| a.id == agent_id))
    } else {
        Ok(false)
    }
}

fn get_agents_file(swarm_config: &crate::commands::init::SwarmInit) -> Result<std::path::PathBuf> {
    let config_dir = directories::ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| std::path::Path::new(".").to_path_buf());

    Ok(config_dir.join(format!("agents-{}.json", swarm_config.swarm_id)))
}

async fn initialize_agent_runtime(agent: &Agent, config: &Config) -> Result<()> {
    // Simulate agent runtime initialization
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    Ok(())
}

async fn register_agent_with_swarm(
    agent: &Agent,
    swarm_config: &crate::commands::init::SwarmInit,
) -> Result<()> {
    // Add agent to persistence
    let agents_file = get_agents_file(swarm_config)?;

    let mut agents: Vec<Agent> = if agents_file.exists() {
        let content = std::fs::read_to_string(&agents_file)?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };

    agents.push(agent.clone());

    let content = serde_json::to_string_pretty(&agents)?;
    std::fs::write(&agents_file, content)?;

    Ok(())
}

async fn setup_agent_connections(
    agent: &Agent,
    swarm_config: &crate::commands::init::SwarmInit,
) -> Result<()> {
    // Simulate setting up network connections based on topology
    match swarm_config.topology.as_str() {
        "mesh" => {
            // Connect to all other agents
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        "hierarchical" => {
            // Connect to parent and children
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        "ring" => {
            // Connect to neighbors in ring
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        "star" => {
            // Connect to central hub
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        _ => {}
    }

    Ok(())
}

async fn load_agent_memory(agent: &Agent, memory_content: &str) -> Result<()> {
    // Simulate loading memory/context
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    Ok(())
}

async fn start_agent_heartbeat(agent: &Agent, config: &Config) -> Result<()> {
    // In a real implementation, this would start a background task
    // For now, we just simulate it
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    Ok(())
}

async fn update_agent_status(agent_id: &str, status: AgentStatus) -> Result<()> {
    // Update agent status in persistence
    // For now, this is a no-op in the simulation
    Ok(())
}
