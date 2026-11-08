//! Plugin System for Custom Model Support
//!
//! The plugin system provides dynamic loading of custom neural network models
//! through shared libraries with comprehensive safety measures, version management,
//! and automatic registration capabilities.

use crate::{
    BaseModel, ModelConfig, ModelInfo, RegistryError, RegistryResult,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::io::Read;
#[cfg(feature = "plugin-system")]
use std::sync::Arc;
use parking_lot::RwLock;
use serde_json::Value;
use sha2::{Sha256, Digest};

#[cfg(feature = "plugin-system")]
use libloading::{Library, Symbol};

/// Plugin descriptor containing metadata and model information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PluginDescriptor {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin license
    pub license: String,
    /// Plugin homepage URL
    pub homepage: Option<String>,
    /// Plugin repository URL
    pub repository: Option<String>,
    /// Plugin keywords
    pub keywords: Vec<String>,
    /// Minimum runtime version required
    pub min_runtime_version: String,
    /// Maximum runtime version supported
    pub max_runtime_version: Option<String>,
    /// Models provided by this plugin
    pub models: Vec<ModelInfo>,
    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,
    /// Plugin capabilities
    pub capabilities: PluginCapabilities,
    /// Plugin metadata
    pub metadata: HashMap<String, Value>,
    /// Plugin checksum for integrity verification
    pub checksum: Option<String>,
    /// Plugin signature for security verification
    pub signature: Option<String>,
}

/// Plugin dependency specification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PluginDependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub version: String,
    /// Whether dependency is optional
    pub optional: bool,
    /// Features required from dependency
    pub features: Vec<String>,
}

/// Plugin capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PluginCapabilities {
    /// Supports model creation
    pub model_creation: bool,
    /// Supports model training
    pub model_training: bool,
    /// Supports model inference
    pub model_inference: bool,
    /// Supports model serialization
    pub model_serialization: bool,
    /// Supports GPU acceleration
    pub gpu_acceleration: bool,
    /// Supports distributed computing
    pub distributed_computing: bool,
    /// Thread-safe operations
    pub thread_safe: bool,
    /// Memory-safe operations
    pub memory_safe: bool,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            model_creation: true,
            model_training: true,
            model_inference: true,
            model_serialization: false,
            gpu_acceleration: false,
            distributed_computing: false,
            thread_safe: true,
            memory_safe: true,
        }
    }
}

/// Plugin state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is not loaded
    Unloaded,
    /// Plugin is loading
    Loading,
    /// Plugin is loaded and active
    Active,
    /// Plugin failed to load
    Failed(String),
    /// Plugin is being unloaded
    Unloading,
}

/// Plugin security level
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityLevel {
    /// No security checks (unsafe)
    None,
    /// Basic checksum verification
    Checksum,
    /// Digital signature verification
    Signature,
    /// Full security with sandboxing
    Sandboxed,
}

/// Plugin loading configuration
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Security level for plugin loading
    pub security_level: SecurityLevel,
    /// Allowed plugin directories
    pub allowed_directories: Vec<PathBuf>,
    /// Blocked plugin directories
    pub blocked_directories: Vec<PathBuf>,
    /// Maximum number of loaded plugins
    pub max_plugins: usize,
    /// Plugin loading timeout in milliseconds
    pub loading_timeout_ms: u64,
    /// Enable plugin caching
    pub enable_caching: bool,
    /// Plugin cache directory
    pub cache_directory: PathBuf,
    /// Trusted plugin authors
    pub trusted_authors: Vec<String>,
    /// Allowed plugin capabilities
    pub allowed_capabilities: Vec<String>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Checksum,
            allowed_directories: vec![
                PathBuf::from("plugins"),
                dirs::home_dir().unwrap_or_default().join(".neuro-divergent/plugins"),
            ],
            blocked_directories: Vec::new(),
            max_plugins: 50,
            loading_timeout_ms: 10000,
            enable_caching: true,
            cache_directory: dirs::cache_dir()
                .unwrap_or_default()
                .join("neuro-divergent")
                .join("plugins"),
            trusted_authors: Vec::new(),
            allowed_capabilities: vec![
                "model_creation".to_string(),
                "model_training".to_string(),
                "model_inference".to_string(),
                "model_serialization".to_string(),
            ],
        }
    }
}

/// Plugin interface that all plugins must implement
pub trait PluginInterface {
    /// Get plugin descriptor
    fn descriptor(&self) -> &PluginDescriptor;
    
    /// Initialize the plugin
    fn initialize(&mut self) -> RegistryResult<()>;
    
    /// Shutdown the plugin
    fn shutdown(&mut self) -> RegistryResult<()>;
    
    /// Create a model with f32 precision
    fn create_model_f32(&self, name: &str, config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<f32>>>;
    
    /// Create a model with f64 precision
    fn create_model_f64(&self, name: &str, config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<f64>>>;
    
    /// List available models
    fn list_models(&self) -> Vec<ModelInfo>;
    
    /// Get model information
    fn get_model_info(&self, name: &str) -> Option<ModelInfo>;
    
    /// Check plugin health
    fn health_check(&self) -> RegistryResult<PluginHealth>;
    
    /// Get plugin version
    fn version(&self) -> &str;
    
    /// Check compatibility with runtime version
    fn is_compatible(&self, runtime_version: &str) -> bool;
}

/// Plugin health status
#[derive(Debug, Clone)]
pub struct PluginHealth {
    /// Plugin is healthy
    pub healthy: bool,
    /// Health check timestamp
    pub timestamp: std::time::SystemTime,
    /// Error messages if unhealthy
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

/// Main plugin structure
pub struct Plugin {
    /// Plugin descriptor
    pub descriptor: PluginDescriptor,
    /// Plugin state
    pub state: RwLock<PluginState>,
    /// Plugin library handle
    #[cfg(feature = "plugin-system")]
    pub library: Option<Arc<Library>>,
    /// Plugin interface
    pub interface: Option<Box<dyn PluginInterface + Send + Sync>>,
    /// Plugin configuration
    pub config: PluginConfig,
    /// Plugin load time
    pub loaded_at: Option<std::time::SystemTime>,
    /// Plugin statistics
    pub stats: RwLock<PluginStats>,
}

/// Plugin statistics
#[derive(Debug, Default, Clone)]
pub struct PluginStats {
    /// Models created through this plugin
    pub models_created: u64,
    /// Total inference calls
    pub inference_calls: u64,
    /// Total training calls
    pub training_calls: u64,
    /// Average model creation time
    pub avg_creation_time_ms: f64,
    /// Error count
    pub error_count: u64,
    /// Last error timestamp
    pub last_error: Option<std::time::SystemTime>,
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
}

impl Plugin {
    /// Create a new plugin from descriptor
    pub fn new(descriptor: PluginDescriptor, config: PluginConfig) -> Self {
        Self {
            descriptor,
            state: RwLock::new(PluginState::Unloaded),
            #[cfg(feature = "plugin-system")]
            library: None,
            interface: None,
            config,
            loaded_at: None,
            stats: RwLock::new(PluginStats::default()),
        }
    }
    
    /// Load plugin from path
    #[cfg(feature = "plugin-system")]
    pub fn load_from_path<P: AsRef<Path>>(path: P, config: PluginConfig) -> RegistryResult<Self> {
        let path = path.as_ref();
        
        let canonical_path = Self::checked_path(path, &config)?;
        let path = canonical_path.as_path();

        // Authorize the descriptor independently: an allowed binary can have
        // a neighboring TOML symlink pointing outside the allowed directory.
        let descriptor_path = Self::checked_path(&Self::descriptor_path(path), &config)?;
        let descriptor = Self::load_descriptor(descriptor_path)?;
        
        // Verify security
        Self::verify_security(&descriptor, path, &config)?;
        
        // TODO: Implement safe plugin loading without unsafe code
        Err(RegistryError::UnsupportedOperation(
            "Plugin system not available without unsafe code".to_string()
        ))
    }
    
    /// Read plugin metadata without loading or authenticating executable code.
    ///
    /// The descriptor must resolve inside an allowed directory and outside every
    /// blocked directory. The read is limited to 1 MiB, including files that grow.
    pub fn load_from_descriptor<P: AsRef<Path>>(descriptor_path: P, config: PluginConfig) -> RegistryResult<Self> {
        let descriptor_path = Self::descriptor_path(descriptor_path.as_ref());
        let descriptor_path = Self::checked_path(&descriptor_path, &config)?;
        let descriptor = Self::load_descriptor(descriptor_path)?;
        Ok(Self::new(descriptor, config))
    }
    
    /// Load plugin descriptor from file
    fn load_descriptor<P: AsRef<Path>>(path: P) -> RegistryResult<PluginDescriptor> {
        const MAX_DESCRIPTOR_BYTES: u64 = 1024 * 1024;
        let descriptor_path = Self::descriptor_path(path.as_ref());
        let metadata = descriptor_path.metadata()?;
        if !metadata.is_file() || metadata.len() > MAX_DESCRIPTOR_BYTES {
            return Err(RegistryError::PluginError(
                "Plugin descriptor must be a regular file no larger than 1 MiB".to_string()
            ));
        }
        let mut content = String::new();
        std::fs::File::open(descriptor_path)?
            .take(MAX_DESCRIPTOR_BYTES + 1)
            .read_to_string(&mut content)?;
        if content.len() as u64 > MAX_DESCRIPTOR_BYTES {
            return Err(RegistryError::PluginError("Plugin descriptor exceeds 1 MiB".to_string()));
        }

        let descriptor: PluginDescriptor = toml::from_str(&content)
            .map_err(|e| RegistryError::PluginError(format!("Failed to parse descriptor: {}", e)))?;
        
        Ok(descriptor)
    }
    
    fn descriptor_path(path: &Path) -> PathBuf {
        if path.is_dir() {
            path.join("plugin.toml")
        } else if path.extension().map_or(false, |ext| ext == "toml") {
            path.to_path_buf()
        } else {
            path.with_extension("toml")
        }
    }

    fn checked_path(path: &Path, config: &PluginConfig) -> RegistryResult<PathBuf> {
        let canonical = path.canonicalize()?;
        for directory in &config.blocked_directories {
            match directory.canonicalize() {
                Ok(directory) if canonical.starts_with(&directory) => {
                    return Err(RegistryError::PluginError("Plugin path is blocked".to_string()));
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        if !config.allowed_directories.iter().any(|directory| {
            directory.canonicalize().map_or(false, |directory| canonical.starts_with(directory))
        }) {
            return Err(RegistryError::PluginError("Plugin path is outside allowed directories".to_string()));
        }
        Ok(canonical)
    }

    /// Verify plugin security
    #[allow(dead_code)]
    fn verify_security(descriptor: &PluginDescriptor, path: &Path, config: &PluginConfig) -> RegistryResult<()> {
        match config.security_level {
            SecurityLevel::None => Ok(()),
            
            SecurityLevel::Checksum => {
                let expected_checksum = descriptor.checksum.as_ref().ok_or_else(|| {
                    RegistryError::PluginError("Plugin checksum is required".to_string())
                })?;
                let actual_checksum = Self::calculate_checksum(path)?;
                if !actual_checksum.eq_ignore_ascii_case(expected_checksum) {
                    return Err(RegistryError::PluginError(
                        "Plugin checksum verification failed".to_string()
                    ));
                }
                Ok(())
            },
            
            SecurityLevel::Signature => {
                Err(RegistryError::UnsupportedOperation(
                    "Digital signature verification is not implemented".to_string()
                ))
            },
            
            SecurityLevel::Sandboxed => {
                Err(RegistryError::UnsupportedOperation(
                    "Sandboxed execution is not implemented".to_string()
                ))
            },
        }
    }
    
    /// Calculate file checksum
    #[allow(dead_code)]
    fn calculate_checksum<P: AsRef<Path>>(path: P) -> RegistryResult<String> {
        let mut file = std::fs::File::open(path)?;
        let mut buffer = [0_u8; 64 * 1024];
        let mut hasher = Sha256::new();
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }
    
    /// Get plugin interface from library
    #[cfg(feature = "plugin-system")]
    fn get_plugin_interface(_library: &Library) -> RegistryResult<Box<dyn PluginInterface + Send + Sync>> {
        // TODO: Implement safe plugin interface loading
        Err(RegistryError::UnsupportedOperation(
            "Plugin interface loading not available without unsafe code".to_string()
        ))
    }
    
    /// Unload the plugin
    pub fn unload(&mut self) -> RegistryResult<()> {
        *self.state.write() = PluginState::Unloading;
        
        // Shutdown plugin interface
        if let Some(ref mut interface) = self.interface {
            interface.shutdown()?;
        }
        
        // Clear interface
        self.interface = None;
        
        // Clear library
        #[cfg(feature = "plugin-system")]
        {
            self.library = None;
        }
        
        *self.state.write() = PluginState::Unloaded;
        log::info!("Unloaded plugin '{}'", self.descriptor.name);
        
        Ok(())
    }
    
    /// Check if plugin is active
    pub fn is_active(&self) -> bool {
        matches!(*self.state.read(), PluginState::Active)
    }
    
    /// Get plugin state
    pub fn get_state(&self) -> PluginState {
        self.state.read().clone()
    }
    
    /// Create a model with f32 precision through this plugin
    pub fn create_model_f32(&self, name: &str, config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<f32>>> {
        if !self.is_active() {
            return Err(RegistryError::PluginError(
                format!("Plugin '{}' is not active", self.descriptor.name)
            ));
        }
        
        if let Some(ref interface) = self.interface {
            let start_time = std::time::Instant::now();
            let result = interface.create_model_f32(name, config);
            let creation_time = start_time.elapsed().as_millis() as f64;
            
            // Update statistics
            {
                let mut stats = self.stats.write();
                stats.models_created += 1;
                stats.avg_creation_time_ms = 
                    (stats.avg_creation_time_ms + creation_time) / 2.0;
                
                if result.is_err() {
                    stats.error_count += 1;
                    stats.last_error = Some(std::time::SystemTime::now());
                }
            }
            
            result
        } else {
            Err(RegistryError::PluginError("Plugin interface not available".to_string()))
        }
    }
    
    /// Create a model with f64 precision through this plugin
    pub fn create_model_f64(&self, name: &str, config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<f64>>> {
        if !self.is_active() {
            return Err(RegistryError::PluginError(
                format!("Plugin '{}' is not active", self.descriptor.name)
            ));
        }
        
        if let Some(ref interface) = self.interface {
            let start_time = std::time::Instant::now();
            let result = interface.create_model_f64(name, config);
            let creation_time = start_time.elapsed().as_millis() as f64;
            
            // Update statistics
            {
                let mut stats = self.stats.write();
                stats.models_created += 1;
                stats.avg_creation_time_ms = 
                    (stats.avg_creation_time_ms + creation_time) / 2.0;
                
                if result.is_err() {
                    stats.error_count += 1;
                    stats.last_error = Some(std::time::SystemTime::now());
                }
            }
            
            result
        } else {
            Err(RegistryError::PluginError("Plugin interface not available".to_string()))
        }
    }
    
    /// List models provided by this plugin
    pub fn list_models(&self) -> Vec<ModelInfo> {
        if let Some(ref interface) = self.interface {
            interface.list_models()
        } else {
            self.descriptor.models.clone()
        }
    }
    
    /// Get model information
    pub fn get_model_info(&self, name: &str) -> Option<ModelInfo> {
        if let Some(ref interface) = self.interface {
            interface.get_model_info(name)
        } else {
            self.descriptor.models.iter()
                .find(|m| m.name == name)
                .cloned()
        }
    }
    
    /// Perform health check
    pub fn health_check(&self) -> RegistryResult<PluginHealth> {
        if let Some(ref interface) = self.interface {
            interface.health_check()
        } else {
            Ok(PluginHealth {
                healthy: self.is_active(),
                timestamp: std::time::SystemTime::now(),
                errors: if self.is_active() { Vec::new() } else { vec!["Plugin not active".to_string()] },
                warnings: Vec::new(),
                metrics: HashMap::new(),
            })
        }
    }
    
    /// Get plugin statistics
    pub fn get_stats(&self) -> PluginStats {
        self.stats.read().clone()
    }
    
    /// Check version compatibility
    pub fn is_compatible(&self, runtime_version: &str) -> bool {
        if let Some(ref interface) = self.interface {
            interface.is_compatible(runtime_version)
        } else {
            // Basic version check using descriptor
            self.check_version_compatibility(runtime_version)
        }
    }
    
    /// Check version compatibility using descriptor
    fn check_version_compatibility(&self, runtime_version: &str) -> bool {
        let Ok(runtime) = semver::Version::parse(runtime_version) else { return false; };
        let Ok(minimum) = semver::Version::parse(&self.descriptor.min_runtime_version) else { return false; };
        if runtime < minimum {
            return false;
        }
        match &self.descriptor.max_runtime_version {
            Some(maximum) => semver::Version::parse(maximum)
                .map_or(false, |maximum| runtime <= maximum),
            None => true,
        }
    }
}

impl Clone for Plugin {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
            state: RwLock::new(self.state.read().clone()),
            #[cfg(feature = "plugin-system")]
            library: self.library.clone(),
            interface: None, // Cannot clone trait objects
            config: self.config.clone(),
            loaded_at: self.loaded_at,
            stats: RwLock::new(self.stats.read().clone()),
        }
    }
}

/// Plugin manager for handling multiple plugins
pub struct PluginManager {
    /// Loaded plugins
    plugins: RwLock<HashMap<String, Plugin>>,
    /// Plugin configuration
    config: RwLock<PluginConfig>,
    /// Plugin statistics
    stats: RwLock<PluginManagerStats>,
}

/// Plugin manager statistics
#[derive(Debug, Default, Clone)]
pub struct PluginManagerStats {
    /// Total plugins loaded
    pub total_loaded: usize,
    /// Active plugins
    pub active_plugins: usize,
    /// Failed plugins
    pub failed_plugins: usize,
    /// Total models from plugins
    pub total_plugin_models: usize,
    /// Plugin loading errors
    pub loading_errors: u64,
    /// Last health check
    pub last_health_check: Option<std::time::SystemTime>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(config: PluginConfig) -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            config: RwLock::new(config),
            stats: RwLock::new(PluginManagerStats::default()),
        }
    }
    
    /// Load plugin from path
    #[cfg(feature = "plugin-system")]
    pub fn load_plugin<P: AsRef<Path>>(&self, path: P) -> RegistryResult<String> {
        let config = self.config.read().clone();
        let plugin = Plugin::load_from_path(path, config)?;
        let name = plugin.descriptor.name.clone();
        
        self.plugins.write().insert(name.clone(), plugin);
        self.update_stats();
        
        Ok(name)
    }
    
    /// Load plugin from descriptor
    pub fn load_plugin_from_descriptor<P: AsRef<Path>>(&self, descriptor_path: P) -> RegistryResult<String> {
        let config = self.config.read().clone();
        let plugin = Plugin::load_from_descriptor(descriptor_path, config)?;
        let name = plugin.descriptor.name.clone();
        
        self.plugins.write().insert(name.clone(), plugin);
        self.update_stats();
        
        Ok(name)
    }
    
    /// Unload plugin
    pub fn unload_plugin(&self, name: &str) -> RegistryResult<()> {
        let mut plugins = self.plugins.write();
        if let Some(mut plugin) = plugins.remove(name) {
            plugin.unload()?;
            self.update_stats();
            Ok(())
        } else {
            Err(RegistryError::PluginError(format!("Plugin '{}' not found", name)))
        }
    }
    
    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Plugin> {
        self.plugins.read().get(name).cloned()
    }
    
    /// List all plugins
    pub fn list_plugins(&self) -> Vec<Plugin> {
        self.plugins.read().values().cloned().collect()
    }
    
    /// Get active plugins
    pub fn get_active_plugins(&self) -> Vec<Plugin> {
        self.plugins.read()
            .values()
            .filter(|p| p.is_active())
            .cloned()
            .collect()
    }
    
    /// Perform health check on all plugins
    pub fn health_check_all(&self) -> HashMap<String, PluginHealth> {
        let plugins = self.plugins.read();
        let mut results = HashMap::new();
        
        for (name, plugin) in plugins.iter() {
            match plugin.health_check() {
                Ok(health) => {
                    results.insert(name.clone(), health);
                }
                Err(e) => {
                    results.insert(name.clone(), PluginHealth {
                        healthy: false,
                        timestamp: std::time::SystemTime::now(),
                        errors: vec![e.to_string()],
                        warnings: Vec::new(),
                        metrics: HashMap::new(),
                    });
                }
            }
        }
        
        let mut stats = self.stats.write();
        stats.last_health_check = Some(std::time::SystemTime::now());
        
        results
    }
    
    /// Get manager statistics
    pub fn get_stats(&self) -> PluginManagerStats {
        self.stats.read().clone()
    }
    
    /// Update statistics
    fn update_stats(&self) {
        let plugins = self.plugins.read();
        let mut stats = self.stats.write();
        
        stats.total_loaded = plugins.len();
        stats.active_plugins = plugins.values().filter(|p| p.is_active()).count();
        stats.failed_plugins = plugins.values()
            .filter(|p| matches!(p.get_state(), PluginState::Failed(_)))
            .count();
        stats.total_plugin_models = plugins.values()
            .map(|p| p.descriptor.models.len())
            .sum();
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new(PluginConfig::default())
    }
}

/// Load plugins from directory
pub fn load_plugins_from_directory<P: AsRef<Path>>(dir: P) -> RegistryResult<Vec<Plugin>> {
    let dir = dir.as_ref();
    if !dir.exists() || !dir.is_dir() {
        return Err(RegistryError::PluginError(
            format!("Plugin directory {:?} does not exist or is not a directory", dir)
        ));
    }
    
    let mut plugins = Vec::new();
    // The caller explicitly selected this directory for metadata discovery.
    let config = PluginConfig {
        allowed_directories: vec![dir.canonicalize()?],
        ..PluginConfig::default()
    };

    for entry in walkdir::WalkDir::new(dir)
        .min_depth(1)
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        
        // Look for plugin descriptor files
        if path.file_name().map_or(false, |name| name == "plugin.toml") {
            match Plugin::load_from_descriptor(path, config.clone()) {
                Ok(plugin) => {
                    plugins.push(plugin);
                }
                Err(e) => {
                    log::warn!("Failed to load plugin from {:?}: {}", path, e);
                }
            }
        }
        
        // Look for shared library files
        #[cfg(feature = "plugin-system")]
        {
            if path.extension().map_or(false, |ext| {
                ext == "so" || ext == "dylib" || ext == "dll"
            }) {
                match Plugin::load_from_path(path, config.clone()) {
                    Ok(plugin) => {
                        plugins.push(plugin);
                    }
                    Err(e) => {
                        log::warn!("Failed to load plugin from {:?}: {}", path, e);
                    }
                }
            }
        }
    }
    
    log::info!("Loaded {} plugins from {:?}", plugins.len(), dir);
    Ok(plugins)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_descriptor() -> PluginDescriptor {
        PluginDescriptor {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test plugin".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            homepage: None,
            repository: None,
            keywords: vec!["test".to_string()],
            min_runtime_version: "1.0.0".to_string(),
            max_runtime_version: None,
            models: Vec::new(),
            dependencies: Vec::new(),
            capabilities: PluginCapabilities::default(),
            metadata: HashMap::new(),
            checksum: None,
            signature: None,
        }
    }
    
    #[test]
    fn test_plugin_descriptor() {
        let descriptor = create_test_descriptor();
        assert_eq!(descriptor.name, "test_plugin");
        assert_eq!(descriptor.version, "1.0.0");
        assert_eq!(descriptor.author, "Test Author");
    }
    
    #[test]
    fn test_plugin_capabilities_default() {
        let caps = PluginCapabilities::default();
        assert!(caps.model_creation);
        assert!(caps.model_training);
        assert!(caps.model_inference);
        assert!(!caps.model_serialization);
        assert!(!caps.gpu_acceleration);
        assert!(caps.thread_safe);
        assert!(caps.memory_safe);
    }
    
    #[test]
    fn test_plugin_config_default() {
        let config = PluginConfig::default();
        assert_eq!(config.security_level, SecurityLevel::Checksum);
        assert_eq!(config.max_plugins, 50);
        assert_eq!(config.loading_timeout_ms, 10000);
        assert!(config.enable_caching);
    }
    
    #[test]
    fn test_plugin_new() {
        let descriptor = create_test_descriptor();
        let config = PluginConfig::default();
        let plugin = Plugin::new(descriptor.clone(), config);
        
        assert_eq!(plugin.descriptor.name, "test_plugin");
        assert_eq!(plugin.get_state(), PluginState::Unloaded);
        assert!(!plugin.is_active());
    }
    
    #[test]
    fn test_plugin_manager_new() {
        let config = PluginConfig::default();
        let manager = PluginManager::new(config);
        
        let stats = manager.get_stats();
        assert_eq!(stats.total_loaded, 0);
        assert_eq!(stats.active_plugins, 0);
        assert_eq!(stats.failed_plugins, 0);
    }
    
    #[test]
    fn test_plugin_health() {
        let health = PluginHealth {
            healthy: true,
            timestamp: std::time::SystemTime::now(),
            errors: Vec::new(),
            warnings: vec!["Test warning".to_string()],
            metrics: HashMap::new(),
        };
        
        assert!(health.healthy);
        assert_eq!(health.warnings.len(), 1);
        assert_eq!(health.errors.len(), 0);
    }
    
    #[test]
    fn test_plugin_dependency() {
        let dep = PluginDependency {
            name: "test_dep".to_string(),
            version: "^1.0".to_string(),
            optional: false,
            features: vec!["feature1".to_string()],
        };
        
        assert_eq!(dep.name, "test_dep");
        assert!(!dep.optional);
        assert_eq!(dep.features.len(), 1);
    }
}

#[cfg(test)]
#[path = "plugin_security_tests.rs"]
mod security_tests;
