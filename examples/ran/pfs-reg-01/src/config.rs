//! Configuration for the Model Registry service

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Main configuration for the Model Registry service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Server configuration
    pub server: ServerConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Performance configuration
    pub performance: PerformanceConfig,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            database: DatabaseConfig::default(),
            storage: StorageConfig::default(),
            monitoring: MonitoringConfig::default(),
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host
    pub host: String,
    
    /// Server port
    pub port: u16,
    
    /// Enable TLS
    pub tls_enabled: bool,
    
    /// TLS certificate path
    pub tls_cert_path: Option<PathBuf>,
    
    /// TLS private key path
    pub tls_key_path: Option<PathBuf>,
    
    /// Server timeout in seconds
    pub timeout_seconds: u64,
    
    /// Maximum message size in bytes
    pub max_message_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 50052,
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
            timeout_seconds: 30,
            max_message_size: 4 * 1024 * 1024, // 4MB
        }
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database URL
    pub url: String,
    
    /// Maximum number of connections
    pub max_connections: u32,
    
    /// Connection timeout in seconds
    pub connect_timeout_seconds: u64,
    
    /// Query timeout in seconds
    pub query_timeout_seconds: u64,
    
    /// Enable migration on startup
    pub auto_migrate: bool,
    
    /// Enable SQL logging
    pub enable_logging: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "sqlite:./pfs_registry.db".to_string(),
            max_connections: 10,
            connect_timeout_seconds: 10,
            query_timeout_seconds: 30,
            auto_migrate: true,
            enable_logging: false,
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage backend type
    pub backend: StorageBackend,
    
    /// Base storage path for filesystem backend
    pub base_path: PathBuf,
    
    /// Enable compression
    pub enable_compression: bool,
    
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    
    /// Compression level (0-9)
    pub compression_level: u32,
    
    /// Enable integrity checking
    pub enable_integrity_check: bool,
    
    /// Cleanup interval in hours
    pub cleanup_interval_hours: u64,
    
    /// Maximum artifact size in bytes
    pub max_artifact_size_bytes: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Filesystem,
            base_path: PathBuf::from("./storage"),
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Gzip,
            compression_level: 6,
            enable_integrity_check: true,
            cleanup_interval_hours: 24,
            max_artifact_size_bytes: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Storage backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    Filesystem,
    S3,
    GCS,
    Azure,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Zstd,
    Lz4,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    pub enable_metrics: bool,
    
    /// Metrics server port
    pub metrics_port: u16,
    
    /// Enable health checks
    pub enable_health_checks: bool,
    
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
    
    /// Enable tracing
    pub enable_tracing: bool,
    
    /// Tracing endpoint
    pub tracing_endpoint: Option<String>,
    
    /// Log level
    pub log_level: String,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_port: 9090,
            enable_health_checks: true,
            health_check_interval_seconds: 30,
            enable_tracing: true,
            tracing_endpoint: None,
            log_level: "info".to_string(),
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable authentication
    pub enable_auth: bool,
    
    /// JWT secret for token validation
    pub jwt_secret: Option<String>,
    
    /// Token expiration time in hours
    pub token_expiration_hours: u64,
    
    /// Enable API key authentication
    pub enable_api_keys: bool,
    
    /// Valid API keys
    pub api_keys: Vec<String>,
    
    /// Enable audit logging
    pub enable_audit_log: bool,
    
    /// Audit log path
    pub audit_log_path: Option<PathBuf>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_auth: false,
            jwt_secret: None,
            token_expiration_hours: 24,
            enable_api_keys: false,
            api_keys: vec![],
            enable_audit_log: false,
            audit_log_path: None,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads
    pub worker_threads: Option<usize>,
    
    /// Enable connection pooling
    pub enable_connection_pooling: bool,
    
    /// Connection pool size
    pub connection_pool_size: u32,
    
    /// Enable caching
    pub enable_caching: bool,
    
    /// Cache size (number of items)
    pub cache_size: usize,
    
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    
    /// Enable request rate limiting
    pub enable_rate_limiting: bool,
    
    /// Rate limit per IP (requests per minute)
    pub rate_limit_per_minute: u32,
    
    /// Batch size for bulk operations
    pub batch_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: None, // Use system default
            enable_connection_pooling: true,
            connection_pool_size: 10,
            enable_caching: true,
            cache_size: 1000,
            cache_ttl_seconds: 3600, // 1 hour
            enable_rate_limiting: false,
            rate_limit_per_minute: 100,
            batch_size: 100,
        }
    }
}

impl RegistryConfig {
    /// Load configuration from file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = RegistryConfig::default();
        
        if let Ok(host) = std::env::var("REGISTRY_HOST") {
            config.server.host = host;
        }
        
        if let Ok(port) = std::env::var("REGISTRY_PORT") {
            config.server.port = port.parse()?;
        }
        
        if let Ok(db_url) = std::env::var("DATABASE_URL") {
            config.database.url = db_url;
        }
        
        if let Ok(storage_path) = std::env::var("STORAGE_PATH") {
            config.storage.base_path = PathBuf::from(storage_path);
        }
        
        if let Ok(log_level) = std::env::var("LOG_LEVEL") {
            config.monitoring.log_level = log_level;
        }
        
        Ok(config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate server config
        if self.server.port == 0 {
            return Err(anyhow::anyhow!("Server port must be > 0"));
        }
        
        // Validate database config
        if self.database.url.is_empty() {
            return Err(anyhow::anyhow!("Database URL cannot be empty"));
        }
        
        // Validate storage config
        if !self.storage.base_path.exists() {
            std::fs::create_dir_all(&self.storage.base_path)?;
        }
        
        // Validate TLS config
        if self.server.tls_enabled {
            if self.server.tls_cert_path.is_none() || self.server.tls_key_path.is_none() {
                return Err(anyhow::anyhow!("TLS cert and key paths required when TLS is enabled"));
            }
        }
        
        Ok(())
    }
}