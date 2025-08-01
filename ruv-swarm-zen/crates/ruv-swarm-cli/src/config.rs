use anyhow::{Context, Result};
use clap::ValueEnum;
use config::{Config as ConfigBuilder, ConfigError, Environment, File};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Configuration profile
#[derive(Debug, Clone, Copy, PartialEq, ValueEnum, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    #[default]
    Dev,
    Prod,
    Test,
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Profile::Dev => write!(f, "dev"),
            Profile::Prod => write!(f, "prod"),
            Profile::Test => write!(f, "test"),
        }
    }
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Active profile
    #[serde(skip)]
    pub profile: Profile,

    /// Swarm configuration
    pub swarm: SwarmConfig,

    /// Agent configuration
    pub agent: AgentConfig,

    /// Persistence configuration
    pub persistence: PersistenceConfig,

    /// API configuration
    pub api: ApiConfig,

    /// Monitoring configuration
    pub monitoring: MonitoringConfig,

    /// Custom environment-specific overrides
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Default topology
    pub default_topology: String,

    /// Maximum swarm size
    pub max_agents: usize,

    /// Agent heartbeat interval (seconds)
    pub heartbeat_interval: u64,

    /// Task queue size
    pub task_queue_size: usize,

    /// Enable auto-scaling
    pub auto_scaling: bool,

    /// Minimum agents to maintain
    pub min_agents: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Default agent capabilities
    pub default_capabilities: Vec<String>,

    /// Agent timeout (seconds)
    pub timeout: u64,

    /// Maximum retries for failed tasks
    pub max_retries: u32,

    /// Memory limit per agent (MB)
    pub memory_limit: usize,

    /// Enable agent checkpointing
    pub enable_checkpointing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Backend type (memory, sqlite, postgres, redis)
    pub backend: String,

    /// Connection string or path
    pub connection: String,

    /// Enable persistence encryption
    pub encryption: bool,

    /// Backup configuration
    pub backup: BackupConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,

    /// Backup interval (hours)
    pub interval: u64,

    /// Backup retention (days)
    pub retention_days: u32,

    /// Backup location
    pub location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API endpoint URL
    pub endpoint: String,

    /// API key (can be overridden by env var)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Request timeout (seconds)
    pub timeout: u64,

    /// Maximum retries
    pub max_retries: u32,

    /// Rate limiting (requests per minute)
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,

    /// Metrics export interval (seconds)
    pub export_interval: u64,

    /// Metrics retention (hours)
    pub retention_hours: u32,

    /// Alert thresholds
    pub alerts: AlertConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// CPU usage threshold (percentage)
    pub cpu_threshold: f32,

    /// Memory usage threshold (percentage)
    pub memory_threshold: f32,

    /// Task failure rate threshold (percentage)
    pub failure_rate_threshold: f32,

    /// Agent offline threshold (seconds)
    pub offline_threshold: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            profile: Profile::Dev,
            swarm: SwarmConfig {
                default_topology: "mesh".to_string(),
                max_agents: 100,
                heartbeat_interval: 30,
                task_queue_size: 1000,
                auto_scaling: true,
                min_agents: 1,
            },
            agent: AgentConfig {
                default_capabilities: vec!["general".to_string()],
                timeout: 300,
                max_retries: 3,
                memory_limit: 512,
                enable_checkpointing: true,
            },
            persistence: PersistenceConfig {
                backend: "memory".to_string(),
                connection: ":memory:".to_string(),
                encryption: false,
                backup: BackupConfig {
                    enabled: false,
                    interval: 24,
                    retention_days: 7,
                    location: "./backups".to_string(),
                },
            },
            api: ApiConfig {
                endpoint: "http://localhost:8080".to_string(),
                api_key: None,
                timeout: 30,
                max_retries: 3,
                rate_limit: Some(60),
            },
            monitoring: MonitoringConfig {
                enabled: true,
                export_interval: 60,
                retention_hours: 24,
                alerts: AlertConfig {
                    cpu_threshold: 80.0,
                    memory_threshold: 90.0,
                    failure_rate_threshold: 10.0,
                    offline_threshold: 120,
                },
            },
            custom: HashMap::new(),
        }
    }
}

impl Config {
    /// Load configuration from files and environment
    pub fn load(config_path: Option<&str>, profile: &Profile) -> Result<Self> {
        let mut builder = ConfigBuilder::builder();

        // Load default configuration
        builder = builder.add_source(ConfigBuilder::try_from(&Config::default())?);

        // Load system configuration files
        if let Some(config_dir) = Self::config_dir() {
            // Global config
            let global_config = config_dir.join("config.toml");
            if global_config.exists() {
                builder = builder.add_source(File::from(global_config));
            }

            // Profile-specific config
            let profile_config = config_dir.join(format!("config.{}.toml", profile));
            if profile_config.exists() {
                builder = builder.add_source(File::from(profile_config));
            }
        }

        // Load user-specified config file
        if let Some(path) = config_path {
            builder = builder.add_source(File::from(Path::new(path)));
        }

        // Override with environment variables
        builder = builder.add_source(
            Environment::with_prefix("RUV_SWARM")
                .separator("__")
                .try_parsing(true),
        );

        // Build and deserialize
        let mut config: Config = builder
            .build()
            .context("Failed to build configuration")?
            .try_deserialize()
            .context("Failed to deserialize configuration")?;

        config.profile = *profile;

        // Handle API key from environment
        if config.api.api_key.is_none() {
            if let Ok(api_key) = std::env::var("RUV_SWARM_API_KEY") {
                config.api.api_key = Some(api_key);
            }
        }

        Ok(config)
    }

    /// Get configuration directory
    fn config_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "ruv-fann", "ruv-swarm")
            .map(|dirs| dirs.config_dir().to_path_buf())
    }

    /// Save configuration to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let toml =
            toml::to_string_pretty(self).context("Failed to serialize configuration to TOML")?;

        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        std::fs::write(path, toml).context("Failed to write configuration file")?;

        Ok(())
    }

    /// Create default configuration files
    pub fn init_config_files() -> Result<PathBuf> {
        let config_dir =
            Self::config_dir().context("Failed to determine configuration directory")?;

        std::fs::create_dir_all(&config_dir)?;

        // Create default configs for each profile
        for profile in [Profile::Dev, Profile::Prod, Profile::Test] {
            let mut config = Config::default();
            config.profile = profile;

            // Adjust settings per profile
            match profile {
                Profile::Dev => {
                    config.swarm.max_agents = 10;
                    config.monitoring.enabled = true;
                }
                Profile::Prod => {
                    config.persistence.backend = "postgres".to_string();
                    config.persistence.encryption = true;
                    config.persistence.backup.enabled = true;
                    config.monitoring.alerts.cpu_threshold = 70.0;
                }
                Profile::Test => {
                    config.swarm.max_agents = 5;
                    config.agent.timeout = 60;
                }
            }

            let config_file = config_dir.join(format!("config.{}.toml", profile));
            config.save(&config_file)?;
        }

        Ok(config_dir)
    }
}
