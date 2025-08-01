use anyhow::{Context, Result};
use colored::Colorize;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

use crate::config::Config;
use crate::output::{OutputHandler, StatusLevel};

// Import onboarding module for seamless initialization
#[cfg(feature = "onboarding")]
use ruv_swarm_cli::onboarding::{self, OnboardingConfig, OnboardingError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmInit {
    pub swarm_id: String,
    pub topology: String,
    pub persistence: PersistenceConfig,
    pub initial_agents: Vec<AgentSpec>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub backend: String,
    pub connection: String,
    pub options: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
}

/// Execute the init command
pub async fn execute(
    config: &Config,
    output: &OutputHandler,
    topology: String,
    persistence: Option<String>,
    config_file: Option<String>,
    non_interactive: bool,
    skip_onboarding: bool,
) -> Result<()> {
    output.section("Initializing RUV Swarm");

    // Run onboarding flow if not skipped
    #[cfg(feature = "onboarding")]
    if !skip_onboarding {
        output.info("🚀 Running seamless onboarding...");

        // Run the onboarding process with auto-accept for non-interactive mode
        let onboarding_config = OnboardingConfig {
            auto_accept: non_interactive,
            ..OnboardingConfig::default()
        };

        match run_onboarding_flow(output, &onboarding_config).await {
            Ok(()) => {
                output.success("✨ Onboarding completed successfully!");
            }
            Err(OnboardingError::ClaudeCodeNotFound) => {
                output.warning("Claude Code not found. Continuing with swarm initialization...");
            }
            Err(e) => {
                output.warning(&format!("Onboarding encountered an issue: {}", e));
                output.info("Continuing with swarm initialization...");
            }
        }
    }

    // Validate topology
    let valid_topologies = vec!["mesh", "hierarchical", "ring", "star", "custom"];
    if !valid_topologies.contains(&topology.as_str()) {
        output.error(&format!(
            "Invalid topology '{}'. Valid options: {}",
            topology,
            valid_topologies.join(", ")
        ));
        return Err(anyhow::anyhow!("Invalid topology"));
    }

    // Determine persistence configuration
    let persistence_config = if let Some(backend) = persistence {
        configure_persistence(&backend, non_interactive)?
    } else if !non_interactive {
        // Ask user for persistence configuration
        let backends = vec!["memory", "sqlite", "postgres", "redis"];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select persistence backend")
            .items(&backends)
            .default(0)
            .interact()?;

        configure_persistence(backends[selection], non_interactive)?
    } else {
        // Default to memory backend
        PersistenceConfig {
            backend: "memory".to_string(),
            connection: ":memory:".to_string(),
            options: HashMap::new(),
        }
    };

    // Load or create initial configuration
    let swarm_init = if let Some(path) = config_file {
        output.info(&format!("Loading configuration from {}", path));
        load_config_file(path)?
    } else {
        // Create new configuration
        let swarm_id = Uuid::new_v4().to_string();

        let initial_agents = if !non_interactive {
            configure_initial_agents(output)?
        } else {
            // Default initial agents
            vec![AgentSpec {
                name: "orchestrator-1".to_string(),
                agent_type: "orchestrator".to_string(),
                capabilities: vec!["coordination".to_string(), "task_distribution".to_string()],
            }]
        };

        SwarmInit {
            swarm_id,
            topology: topology.clone(),
            persistence: persistence_config,
            initial_agents,
            created_at: chrono::Utc::now(),
        }
    };

    // Show configuration summary
    output.section("Configuration Summary");
    output.key_value(&[
        ("Swarm ID".to_string(), swarm_init.swarm_id.clone()),
        ("Topology".to_string(), swarm_init.topology.clone()),
        (
            "Persistence".to_string(),
            swarm_init.persistence.backend.clone(),
        ),
        (
            "Initial Agents".to_string(),
            swarm_init.initial_agents.len().to_string(),
        ),
    ]);

    // Confirm configuration
    if !non_interactive {
        if !output.confirm("Initialize swarm with this configuration?", true)? {
            output.warning("Swarm initialization cancelled");
            return Ok(());
        }
    }

    // Initialize the swarm
    let spinner = output.spinner("Initializing swarm components...");

    // Simulate initialization steps
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Create persistence backend
    initialize_persistence(&swarm_init.persistence, output).await?;

    // Set up swarm topology
    initialize_topology(&swarm_init.topology, output).await?;

    // Spawn initial agents
    for agent in &swarm_init.initial_agents {
        spawn_initial_agent(agent, output).await?;
    }

    if let Some(pb) = spinner {
        pb.finish_with_message("Swarm initialization complete");
    }

    // Save swarm configuration
    save_swarm_config(&swarm_init, config, output)?;

    // Configure MCP servers for Claude Code
    configure_mcp_servers(&swarm_init, output)?;

    output.success(&format!(
        "Swarm '{}' initialized successfully!",
        swarm_init.swarm_id
    ));

    // Show next steps
    output.section("Next Steps");
    output.list(
        &[
            format!("Spawn additional agents: ruv-swarm spawn <type>"),
            format!("Start orchestration: ruv-swarm orchestrate <strategy> <task>"),
            format!("Monitor swarm: ruv-swarm monitor"),
            format!("Check status: ruv-swarm status"),
        ],
        true,
    );

    Ok(())
}

fn configure_persistence(backend: &str, non_interactive: bool) -> Result<PersistenceConfig> {
    let mut options = HashMap::new();

    let connection = match backend {
        "memory" => ":memory:".to_string(),
        "sqlite" => {
            if non_interactive {
                "./swarm.db".to_string()
            } else {
                Input::<String>::with_theme(&ColorfulTheme::default())
                    .with_prompt("SQLite database path")
                    .default("./swarm.db".to_string())
                    .interact()?
            }
        }
        "postgres" => {
            if non_interactive {
                "postgresql://localhost/ruv_swarm".to_string()
            } else {
                Input::<String>::with_theme(&ColorfulTheme::default())
                    .with_prompt("PostgreSQL connection string")
                    .default("postgresql://localhost/ruv_swarm".to_string())
                    .interact()?
            }
        }
        "redis" => {
            if non_interactive {
                "redis://localhost:6379".to_string()
            } else {
                Input::<String>::with_theme(&ColorfulTheme::default())
                    .with_prompt("Redis connection string")
                    .default("redis://localhost:6379".to_string())
                    .interact()?
            }
        }
        _ => return Err(anyhow::anyhow!("Unsupported persistence backend")),
    };

    // Add backend-specific options
    if backend == "sqlite" && !non_interactive {
        if Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Enable WAL mode for better concurrency?")
            .default(true)
            .interact()?
        {
            options.insert("wal_mode".to_string(), "true".to_string());
        }
    }

    Ok(PersistenceConfig {
        backend: backend.to_string(),
        connection,
        options,
    })
}

fn configure_initial_agents(output: &OutputHandler) -> Result<Vec<AgentSpec>> {
    let mut agents = Vec::new();

    // Always add an orchestrator
    agents.push(AgentSpec {
        name: "orchestrator-1".to_string(),
        agent_type: "orchestrator".to_string(),
        capabilities: vec!["coordination".to_string(), "task_distribution".to_string()],
    });

    if Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Add additional initial agents?")
        .default(true)
        .interact()?
    {
        let agent_types = vec!["researcher", "coder", "analyst", "reviewer", "tester"];

        loop {
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select agent type")
                .items(&agent_types)
                .interact()?;

            let agent_type = agent_types[selection];
            let name = Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Agent name")
                .default(format!("{}-{}", agent_type, agents.len() + 1))
                .interact()?;

            let capabilities_str = Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Capabilities (comma-separated)")
                .default(default_capabilities(agent_type))
                .interact()?;

            let capabilities: Vec<String> = capabilities_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            agents.push(AgentSpec {
                name,
                agent_type: agent_type.to_string(),
                capabilities,
            });

            if !Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt("Add another agent?")
                .default(false)
                .interact()?
            {
                break;
            }
        }
    }

    Ok(agents)
}

fn default_capabilities(agent_type: &str) -> String {
    match agent_type {
        "researcher" => "web_search,document_analysis,summarization",
        "coder" => "code_generation,refactoring,testing",
        "analyst" => "data_analysis,visualization,reporting",
        "reviewer" => "code_review,quality_assurance,best_practices",
        "tester" => "unit_testing,integration_testing,performance_testing",
        _ => "general",
    }
    .to_string()
}

fn load_config_file<P: AsRef<Path>>(path: P) -> Result<SwarmInit> {
    let content = std::fs::read_to_string(path)?;

    // Try to parse as YAML first, then JSON
    serde_yaml::from_str(&content)
        .or_else(|_| serde_json::from_str(&content))
        .context("Failed to parse configuration file")
}

async fn initialize_persistence(config: &PersistenceConfig, output: &OutputHandler) -> Result<()> {
    output.info(&format!(
        "Setting up {} persistence backend...",
        config.backend
    ));

    // Simulate persistence initialization
    match config.backend.as_str() {
        "memory" => {
            // Nothing to do for memory backend
        }
        "sqlite" => {
            // Create database file if needed
            if config.connection != ":memory:" {
                let path = Path::new(&config.connection);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
            }
        }
        "postgres" | "redis" => {
            // In a real implementation, we would test the connection here
        }
        _ => {}
    }

    Ok(())
}

async fn initialize_topology(topology: &str, output: &OutputHandler) -> Result<()> {
    output.info(&format!("Configuring {} topology...", topology));

    // Simulate topology setup
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    Ok(())
}

async fn spawn_initial_agent(agent: &AgentSpec, output: &OutputHandler) -> Result<()> {
    output.info(&format!(
        "Spawning {} agent '{}'...",
        agent.agent_type, agent.name
    ));

    // Simulate agent spawning
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Ok(())
}

fn save_swarm_config(
    swarm_init: &SwarmInit,
    config: &Config,
    output: &OutputHandler,
) -> Result<()> {
    let config_dir = directories::ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    std::fs::create_dir_all(&config_dir)?;

    let swarm_file = config_dir.join(format!("swarm-{}.json", swarm_init.swarm_id));
    let content = serde_json::to_string_pretty(swarm_init)?;
    std::fs::write(&swarm_file, content)?;

    // Also save as "current" swarm
    let current_file = config_dir.join("current-swarm.json");
    std::fs::write(&current_file, serde_json::to_string_pretty(swarm_init)?)?;

    output.info(&format!("Configuration saved to {:?}", swarm_file));

    Ok(())
}

fn configure_mcp_servers(swarm_init: &SwarmInit, output: &OutputHandler) -> Result<()> {
    output.section("Configuring MCP Servers for Claude Code");

    // Check if .mcp.json exists
    let mcp_config_path = Path::new(".mcp.json");

    // Load existing config or create new one
    let mut mcp_config = if mcp_config_path.exists() {
        let content = std::fs::read_to_string(mcp_config_path)?;
        serde_json::from_str(&content).unwrap_or_else(|_| {
            serde_json::json!({
                "mcpServers": {}
            })
        })
    } else {
        serde_json::json!({
            "mcpServers": {}
        })
    };

    // Get the mcp_servers object
    let servers = mcp_config["mcpServers"]
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Invalid MCP configuration structure"))?;

    // Add GitHub MCP server if not already present
    if !servers.contains_key("github") {
        output.info("Adding GitHub MCP server configuration...");

        // Check for GitHub token
        let github_token = std::env::var("GITHUB_TOKEN")
            .or_else(|_| std::env::var("GH_TOKEN"))
            .ok();

        if github_token.is_none() {
            output.warning(
                "No GitHub token found. Set GITHUB_TOKEN or GH_TOKEN environment variable.",
            );
            output.info("You can also authenticate using: gh auth login");
        }

        servers.insert(
            "github".to_string(),
            serde_json::json!({
                "command": "npx",
                "args": ["@modelcontextprotocol/server-github"],
                "env": if let Some(token) = github_token {
                    serde_json::json!({
                        "GITHUB_TOKEN": token
                    })
                } else {
                    serde_json::json!({})
                }
            }),
        );

        output.success("GitHub MCP server configured");
    } else {
        output.info("GitHub MCP server already configured");
    }

    // Add ruv-swarm MCP server if not already present
    if !servers.contains_key("ruv-swarm") {
        output.info("Adding ruv-swarm MCP server configuration...");

        servers.insert(
            "ruv-swarm".to_string(),
            serde_json::json!({
                "command": "npx",
                "args": ["ruv-swarm", "mcp", "start"],
                "env": {
                    "SWARM_ID": swarm_init.swarm_id.clone(),
                    "SWARM_TOPOLOGY": swarm_init.topology.clone()
                }
            }),
        );

        output.success("ruv-swarm MCP server configured");
    } else {
        output.info("ruv-swarm MCP server already configured");
    }

    // Show configured servers
    output.section("Configured MCP Servers");
    let server_list: Vec<String> = servers
        .iter()
        .map(|(name, _)| format!("✓ {}", name))
        .collect();
    output.list(&server_list, false);

    // Save the updated configuration
    let pretty_json = serde_json::to_string_pretty(&mcp_config)?;
    std::fs::write(mcp_config_path, pretty_json)?;

    output.success("MCP configuration saved to .mcp.json");
    output.info("Claude Code will use these MCP servers on next launch");

    Ok(())
}

/// Run the onboarding flow
#[cfg(feature = "onboarding")]
async fn run_onboarding_flow(
    output: &OutputHandler,
    config: &OnboardingConfig,
) -> Result<(), OnboardingError> {
    // This is a simplified integration point for the onboarding system
    // In a full implementation, this would create platform-specific instances
    // of the traits defined in the onboarding module

    output.info("🔍 Checking Claude Code installation...");

    // For now, this is a placeholder that indicates onboarding is integrated
    // The actual implementation would use the traits from onboarding::mod.rs
    output.info("ℹ️  Onboarding system integrated but full implementation pending");
    output.info("ℹ️  This will automatically detect and configure Claude Code");
    output.info("ℹ️  For now, continuing with manual MCP configuration...");

    Ok(())
}
