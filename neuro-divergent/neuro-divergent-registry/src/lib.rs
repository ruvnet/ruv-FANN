#![doc = include_str!("../README.md")]

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs, rust_2018_idioms)]
#![deny(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

// Module declarations
pub mod factory;
pub mod registry;
pub mod plugin;
pub mod discovery;

// Re-exports for public API
pub use factory::*;
pub use registry::*;
pub use plugin::*;
pub use discovery::*;

/// Numeric types supported by the registry system
pub trait Float: 
    num_traits::Float + 
    Send + 
    Sync + 
    std::fmt::Debug + 
    std::fmt::Display + 
    serde::Serialize + 
    serde::de::DeserializeOwned + 
    'static 
{
}

impl Float for f32 {}
impl Float for f64 {}

/// Base trait that all neural network models must implement
pub trait BaseModel<T: Float>: Send + Sync {
    /// Get the model name
    fn name(&self) -> &str;
    
    /// Get the model category
    fn category(&self) -> ModelCategory;
    
    /// Get model configuration
    fn config(&self) -> ModelConfig;
    
    /// Forward pass through the model
    fn forward(&self, input: &[T]) -> Result<Vec<T>, ModelError>;
    
    /// Backward pass for training
    fn backward(&mut self, gradient: &[T]) -> Result<Vec<T>, ModelError>;
    
    /// Update model parameters
    fn update_parameters(&mut self, learning_rate: T) -> Result<(), ModelError>;
    
    /// Get model parameters
    fn parameters(&self) -> Vec<&[T]>;
    
    /// Get mutable model parameters
    fn parameters_mut(&mut self) -> Vec<&mut [T]>;
    
    /// Get model parameter count
    fn parameter_count(&self) -> usize;
    
    /// Reset model to initial state
    fn reset(&mut self) -> Result<(), ModelError>;
    
    /// Check if model is trained
    fn is_trained(&self) -> bool;
    
    /// Get model version
    fn version(&self) -> &str { "1.0.0" }
    
    /// Get model description
    fn description(&self) -> &str { "Neural network model" }
    
    /// Get required input size
    fn input_size(&self) -> Option<usize> { None }
    
    /// Get output size
    fn output_size(&self) -> Option<usize> { None }
    
    /// Check if model supports online learning
    fn supports_online_learning(&self) -> bool { false }
    
    /// Check if model supports batch processing
    fn supports_batch_processing(&self) -> bool { true }
    
    /// Get model metadata
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }
}

/// Model categories for organization and discovery
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelCategory {
    /// Basic models: MLP, DLinear, NLinear, MLPMultivariate
    Basic,
    /// Recurrent models: RNN, LSTM, GRU
    Recurrent,
    /// Advanced models: NBEATS, NBEATSx, NHITS, TiDE
    Advanced,
    /// Transformer models: TFT, Informer, AutoFormer, FedFormer, PatchTST, iTransformer
    Transformer,
    /// Specialized models: DeepAR, DeepNPTS, TCN, BiTCN, TimesNet, StemGNN, TSMixer, TSMixerx, TimeLLM
    Specialized,
    /// Custom user-defined models
    Custom,
}

impl ModelCategory {
    /// Get all available categories
    pub fn all() -> Vec<Self> {
        vec![
            Self::Basic,
            Self::Recurrent,
            Self::Advanced,
            Self::Transformer,
            Self::Specialized,
            Self::Custom,
        ]
    }
    
    /// Get category description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Basic => "Basic neural network models for general use",
            Self::Recurrent => "Recurrent neural networks for sequential data",
            Self::Advanced => "Advanced architectures for complex time series",
            Self::Transformer => "Transformer-based models for attention mechanisms",
            Self::Specialized => "Specialized models for specific domains",
            Self::Custom => "User-defined custom models",
        }
    }
}

/// Model configuration for creation and serialization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    /// Model category
    pub category: ModelCategory,
    /// Model parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Required input size
    pub input_size: Option<usize>,
    /// Output size
    pub output_size: Option<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(name: impl Into<String>, category: ModelCategory) -> Self {
        Self {
            name: name.into(),
            category,
            parameters: HashMap::new(),
            version: "1.0.0".to_string(),
            description: "Neural network model".to_string(),
            input_size: None,
            output_size: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Set a parameter value
    pub fn set_parameter(&mut self, key: impl Into<String>, value: serde_json::Value) -> &mut Self {
        self.parameters.insert(key.into(), value);
        self
    }
    
    /// Get a parameter value
    pub fn get_parameter(&self, key: &str) -> Option<&serde_json::Value> {
        self.parameters.get(key)
    }
    
    /// Set model dimensions
    pub fn set_dimensions(&mut self, input_size: Option<usize>, output_size: Option<usize>) -> &mut Self {
        self.input_size = input_size;
        self.output_size = output_size;
        self
    }
    
    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) -> &mut Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Model information for registry and discovery
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model category
    pub category: ModelCategory,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Available parameter types
    pub parameter_types: Vec<String>,
    /// Required input size
    pub input_size: Option<usize>,
    /// Output size
    pub output_size: Option<usize>,
    /// Model capabilities
    pub capabilities: ModelCapabilities,
    /// Model performance metrics
    pub performance: Option<ModelPerformance>,
    /// Model metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Model capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilities {
    /// Supports online learning
    pub online_learning: bool,
    /// Supports batch processing
    pub batch_processing: bool,
    /// Supports streaming
    pub streaming: bool,
    /// Supports multi-threading
    pub multi_threading: bool,
    /// Supports GPU acceleration
    pub gpu_acceleration: bool,
    /// Supports quantization
    pub quantization: bool,
    /// Supports pruning
    pub pruning: bool,
    /// Supports fine-tuning
    pub fine_tuning: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            online_learning: false,
            batch_processing: true,
            streaming: false,
            multi_threading: true,
            gpu_acceleration: false,
            quantization: false,
            pruning: false,
            fine_tuning: true,
        }
    }
}

/// Model performance metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelPerformance {
    /// Creation time in milliseconds
    pub creation_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Forward pass time in microseconds
    pub forward_pass_time_us: f64,
    /// Backward pass time in microseconds
    pub backward_pass_time_us: Option<f64>,
    /// Parameter count
    pub parameter_count: usize,
    /// Model size in bytes
    pub model_size_bytes: usize,
    /// Throughput (samples per second)
    pub throughput_samples_per_sec: f64,
    /// Benchmark timestamp
    pub benchmark_timestamp: std::time::SystemTime,
}

/// Registry errors
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Model not found
    #[error("Model '{0}' not found in registry")]
    ModelNotFound(String),
    
    /// Model already exists
    #[error("Model '{0}' already exists in registry")]
    ModelAlreadyExists(String),
    
    /// Invalid configuration
    #[error("Invalid model configuration: {0}")]
    InvalidConfiguration(String),
    
    /// Plugin error
    #[error("Plugin error: {0}")]
    PluginError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    /// Model creation error
    #[error("Model creation error: {0}")]
    ModelCreationError(String),
    
    /// Model operation error
    #[error("Model operation error: {0}")]
    ModelOperationError(String),
    
    /// Threading error
    #[error("Threading error: {0}")]
    ThreadingError(String),
    
    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    /// Resource error
    #[error("Resource error: {0}")]
    ResourceError(String),
}

/// Model errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Invalid dimensions
    #[error("Invalid dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { 
        /// Expected dimension size
        expected: usize, 
        /// Actual dimension size
        actual: usize 
    },
    
    /// Model not trained
    #[error("Model not trained")]
    ModelNotTrained,
    
    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),
    
    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    /// Parameter error
    #[error("Parameter error: {0}")]
    ParameterError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Result type for registry operations
pub type RegistryResult<T> = Result<T, RegistryError>;

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Global registry instance
static GLOBAL_REGISTRY: once_cell::sync::Lazy<Arc<parking_lot::RwLock<ModelRegistry>>> =
    once_cell::sync::Lazy::new(|| {
        Arc::new(parking_lot::RwLock::new(ModelRegistry::new()))
    });

/// Get the global registry instance (read-only)
pub fn global_registry() -> Arc<parking_lot::RwLock<ModelRegistry>> {
    GLOBAL_REGISTRY.clone()
}

/// Initialize the global registry with built-in models
pub fn initialize_registry() -> RegistryResult<()> {
    log::info!("Initializing global model registry");
    
    let registry = global_registry();
    let reg = registry.write();
    
    // Register built-in models through discovery
    let discovered = discovery::discover_builtin_models()?;
    for model_info in discovered {
        reg.register_info(model_info)?;
    }
    
    log::info!("Global registry initialized with {} models", reg.len());
    Ok(())
}

/// Registry initialization configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegistryConfig {
    /// Enable plugin system
    pub enable_plugins: bool,
    /// Plugin directories to scan
    pub plugin_directories: Vec<std::path::PathBuf>,
    /// Enable benchmarking
    pub enable_benchmarking: bool,
    /// Maximum number of models to cache
    pub max_cache_size: usize,
    /// Enable async operations
    pub enable_async: bool,
    /// Model discovery configuration
    pub discovery_config: discovery::DiscoveryConfig,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            enable_plugins: true,
            plugin_directories: vec![
                std::path::PathBuf::from("plugins"),
                dirs::home_dir().unwrap_or_default().join(".neuro-divergent/plugins"),
            ],
            enable_benchmarking: true,
            max_cache_size: 100,
            enable_async: false,
            discovery_config: discovery::DiscoveryConfig::default(),
        }
    }
}

/// Initialize registry with custom configuration
pub fn initialize_registry_with_config(config: RegistryConfig) -> RegistryResult<()> {
    log::info!("Initializing global model registry with custom configuration");
    
    let registry = global_registry();
    let reg = registry.write();
    
    // Set configuration
    reg.set_config(config.clone());
    
    // Register built-in models
    let discovered = discovery::discover_builtin_models_with_config(&config.discovery_config)?;
    for model_info in discovered {
        reg.register_info(model_info)?;
    }
    
    // Load plugins if enabled
    if config.enable_plugins {
        for plugin_dir in &config.plugin_directories {
            if plugin_dir.exists() {
                match plugin::load_plugins_from_directory(plugin_dir) {
                    Ok(plugins) => {
                        for plugin in plugins {
                            reg.register_plugin(plugin)?;
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to load plugins from {:?}: {}", plugin_dir, e);
                    }
                }
            }
        }
    }
    
    log::info!("Global registry initialized with {} models", reg.len());
    Ok(())
}

/// Shutdown the registry and cleanup resources
pub fn shutdown_registry() -> RegistryResult<()> {
    log::info!("Shutting down global model registry");
    
    let registry = global_registry();
    let reg = registry.write();
    
    // Cleanup plugins
    reg.cleanup_plugins()?;
    
    // Clear cache
    reg.clear_cache();
    
    log::info!("Global registry shutdown completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_category_all() {
        let categories = ModelCategory::all();
        assert_eq!(categories.len(), 6);
        assert!(categories.contains(&ModelCategory::Basic));
        assert!(categories.contains(&ModelCategory::Recurrent));
        assert!(categories.contains(&ModelCategory::Advanced));
        assert!(categories.contains(&ModelCategory::Transformer));
        assert!(categories.contains(&ModelCategory::Specialized));
        assert!(categories.contains(&ModelCategory::Custom));
    }
    
    #[test]
    fn test_model_config_creation() {
        let mut config = ModelConfig::new("test_model", ModelCategory::Basic);
        config.set_parameter("learning_rate", serde_json::json!(0.001));
        config.set_dimensions(Some(10), Some(1));
        
        assert_eq!(config.name, "test_model");
        assert_eq!(config.category, ModelCategory::Basic);
        assert_eq!(config.input_size, Some(10));
        assert_eq!(config.output_size, Some(1));
        assert!(config.get_parameter("learning_rate").is_some());
    }
    
    #[test]
    fn test_model_capabilities_default() {
        let caps = ModelCapabilities::default();
        assert!(!caps.online_learning);
        assert!(caps.batch_processing);
        assert!(!caps.streaming);
        assert!(caps.multi_threading);
        assert!(!caps.gpu_acceleration);
        assert!(!caps.quantization);
        assert!(!caps.pruning);
        assert!(caps.fine_tuning);
    }
    
    #[test]
    fn test_registry_config_default() {
        let config = RegistryConfig::default();
        assert!(config.enable_plugins);
        assert!(config.enable_benchmarking);
        assert_eq!(config.max_cache_size, 100);
        assert!(!config.enable_async);
        assert_eq!(config.plugin_directories.len(), 2);
    }
}