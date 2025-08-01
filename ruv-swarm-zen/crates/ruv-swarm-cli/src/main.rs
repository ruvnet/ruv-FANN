use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Generator, Shell};
use colored::Colorize;
use std::io;
use tracing::{debug, error, info};
use tracing_subscriber::EnvFilter;

mod commands;
mod config;
mod output;

use commands::{init, monitor, orchestrate, spawn, status};
use config::{Config, Profile};
use output::{OutputFormat, OutputHandler};

/// Distributed swarm orchestration with cognitive diversity
#[derive(Parser, Debug)]
#[command(name = "ruv-swarm")]
#[command(version)]
#[command(about = "Distributed swarm orchestration with cognitive diversity", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, global = true, env = "RUV_SWARM_CONFIG")]
    config: Option<String>,

    /// Profile to use (dev, prod, test)
    #[arg(
        short,
        long,
        global = true,
        env = "RUV_SWARM_PROFILE",
        default_value = "dev"
    )]
    profile: Profile,

    /// Output format
    #[arg(short, long, global = true, value_enum, default_value = "auto")]
    output: OutputFormat,

    /// Enable verbose logging
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Disable color output
    #[arg(long, global = true)]
    no_color: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Initialize a new swarm with specified topology
    Init {
        /// Swarm topology (mesh, hierarchical, ring, star, custom)
        #[arg(value_enum)]
        topology: String,

        /// Persistence backend (memory, sqlite, postgres, redis)
        #[arg(short = 'b', long)]
        persistence: Option<String>,

        /// Initial swarm configuration file
        #[arg(short = 'f', long)]
        config_file: Option<String>,

        /// Skip interactive setup
        #[arg(long)]
        non_interactive: bool,

        /// Skip onboarding flow
        #[arg(long)]
        skip_onboarding: bool,
    },

    /// Spawn a new agent in the swarm
    Spawn {
        /// Agent type (researcher, coder, analyst, reviewer, orchestrator)
        agent_type: String,

        /// Agent capabilities (comma-separated)
        #[arg(short = 'a', long, value_delimiter = ',')]
        capabilities: Vec<String>,

        /// Agent name (auto-generated if not provided)
        #[arg(short, long)]
        name: Option<String>,

        /// Initial memory/context for the agent
        #[arg(short, long)]
        memory: Option<String>,

        /// Parent agent ID for hierarchical topologies
        #[arg(short = 'P', long)]
        parent: Option<String>,
    },

    /// Orchestrate a distributed task across the swarm
    Orchestrate {
        /// Orchestration strategy (parallel, sequential, adaptive, consensus)
        #[arg(value_enum)]
        strategy: String,

        /// Task description or task file path
        task: String,

        /// Maximum number of agents to use
        #[arg(short, long)]
        max_agents: Option<usize>,

        /// Task timeout in seconds
        #[arg(short, long)]
        timeout: Option<u64>,

        /// Priority level (1-10)
        #[arg(short = 'r', long, default_value = "5")]
        priority: u8,

        /// Watch task progress in real-time
        #[arg(short, long)]
        watch: bool,
    },

    /// Show current swarm status
    Status {
        /// Show detailed agent information
        #[arg(short, long)]
        detailed: bool,

        /// Filter by agent type
        #[arg(short = 't', long)]
        agent_type: Option<String>,

        /// Show only active agents
        #[arg(short, long)]
        active_only: bool,

        /// Include performance metrics
        #[arg(short, long)]
        metrics: bool,
    },

    /// Monitor swarm activity in real-time
    Monitor {
        /// Refresh interval in seconds
        #[arg(short, long, default_value = "1")]
        interval: u64,

        /// Filter events by type
        #[arg(short, long)]
        filter: Option<String>,

        /// Maximum number of events to display
        #[arg(short, long, default_value = "100")]
        max_events: usize,

        /// Export monitoring data to file
        #[arg(short, long)]
        export: Option<String>,
    },

    /// Generate shell completions
    Completion {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize color settings
    if cli.no_color {
        colored::control::set_override(false);
    }

    // Initialize logging
    init_logging(cli.verbose)?;

    // Load configuration
    let config = Config::load(cli.config.as_deref(), &cli.profile)
        .context("Failed to load configuration")?;

    // Create output handler
    let output = OutputHandler::new(cli.output);

    // Execute command
    let result = match cli.command {
        Commands::Init {
            topology,
            persistence,
            config_file,
            non_interactive,
            skip_onboarding,
        } => {
            init::execute(
                &config,
                &output,
                topology,
                persistence,
                config_file,
                non_interactive,
                skip_onboarding,
            )
            .await
        }

        Commands::Spawn {
            agent_type,
            capabilities,
            name,
            memory,
            parent,
        } => {
            spawn::execute(
                &config,
                &output,
                agent_type,
                capabilities,
                name,
                memory,
                parent,
            )
            .await
        }

        Commands::Orchestrate {
            strategy,
            task,
            max_agents,
            timeout,
            priority,
            watch,
        } => {
            orchestrate::execute(
                &config, &output, strategy, task, max_agents, timeout, priority, watch,
            )
            .await
        }

        Commands::Status {
            detailed,
            agent_type,
            active_only,
            metrics,
        } => status::execute(&config, &output, detailed, agent_type, active_only, metrics).await,

        Commands::Monitor {
            interval,
            filter,
            max_events: _,
            export: _,
        } => {
            monitor::execute(
                &config,
                &monitor::MonitorArgs {
                    filter,
                    watch: false,
                    interval,
                },
                &output,
            )
            .await
        }

        Commands::Completion { shell } => {
            generate_completion(shell);
            Ok(())
        }
    };

    if let Err(e) = result {
        error!("{}: {}", "Error".red().bold(), e);
        std::process::exit(1);
    }

    Ok(())
}

fn init_logging(verbosity: u8) -> Result<()> {
    let filter = match verbosity {
        0 => EnvFilter::new("warn"),
        1 => EnvFilter::new("info"),
        2 => EnvFilter::new("debug"),
        _ => EnvFilter::new("trace"),
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(verbosity >= 2)
        .with_thread_ids(verbosity >= 3)
        .with_file(verbosity >= 3)
        .with_line_number(verbosity >= 3)
        .init();

    Ok(())
}

fn generate_completion<G: Generator>(generator: G) {
    let mut cmd = Cli::command();
    generate(generator, &mut cmd, "ruv-swarm", &mut io::stdout());
}
