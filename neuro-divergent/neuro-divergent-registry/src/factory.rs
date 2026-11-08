//! Model Factory for Dynamic Model Creation
//!
//! The ModelFactory provides a unified interface for creating neural network models
//! from string names, configurations, or templates. It supports both synchronous
//! and asynchronous model creation with built-in caching and performance optimization.

use crate::{
    BaseModel, Float, ModelConfig, ModelInfo, ModelCategory,
    ModelPerformance, RegistryError, RegistryResult, global_registry
};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde_json::Value;
use std::time::Instant;

/// Model creation function signature
pub type ModelCreatorFn<T> = Box<dyn Fn(&ModelConfig) -> RegistryResult<Box<dyn BaseModel<T>>> + Send + Sync>;

/// Model template for creating models with common configurations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelTemplate {
    /// Template name
    pub name: String,
    /// Base model name
    pub base_model: String,
    /// Default configuration
    pub default_config: ModelConfig,
    /// Parameter constraints
    pub parameter_constraints: HashMap<String, ParameterConstraint>,
    /// Template description
    pub description: String,
    /// Template category
    pub category: ModelCategory,
}

/// Parameter constraints for model creation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParameterConstraint {
    /// Parameter type
    pub param_type: ParameterType,
    /// Minimum value (for numeric types)
    pub min_value: Option<f64>,
    /// Maximum value (for numeric types)
    pub max_value: Option<f64>,
    /// Allowed values (for enum types)
    pub allowed_values: Option<Vec<Value>>,
    /// Required parameter
    pub required: bool,
    /// Default value
    pub default_value: Option<Value>,
    /// Parameter description
    pub description: String,
}

/// Parameter types for validation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ParameterType {
    /// Integer parameter
    Integer,
    /// Float parameter
    Float,
    /// Boolean parameter
    Boolean,
    /// String parameter
    String,
    /// Array parameter
    Array,
    /// Object parameter
    Object,
    /// Enum parameter
    Enum,
}

/// Model creation options
#[derive(Debug, Clone)]
pub struct CreationOptions {
    /// Enable performance benchmarking
    pub benchmark: bool,
    /// Enable model validation
    pub validate: bool,
    /// Cache created model
    pub cache: bool,
    /// Timeout for model creation (in milliseconds)
    pub timeout_ms: Option<u64>,
    /// Custom parameters to override
    pub overrides: HashMap<String, Value>,
}

impl Default for CreationOptions {
    fn default() -> Self {
        Self {
            benchmark: false,
            validate: true,
            cache: true,
            timeout_ms: Some(30000), // 30 seconds
            overrides: HashMap::new(),
        }
    }
}

/// Model creation context
#[derive(Debug)]
pub struct CreationContext {
    /// Creation start time
    pub start_time: Instant,
    /// Model name being created
    pub model_name: String,
    /// Configuration used
    pub config: ModelConfig,
    /// Creation options
    pub options: CreationOptions,
    /// Performance metrics
    pub performance: Option<ModelPerformance>,
}

/// Model Factory for dynamic model creation
pub struct ModelFactory {
    /// Model creators by name (placeholder - not yet implemented)
    #[allow(dead_code)]
    creators: RwLock<HashMap<String, Box<dyn Fn(&ModelConfig) -> RegistryResult<Box<dyn BaseModel<f32>>> + Send + Sync>>>,
    /// Model creators by name for f64 (placeholder - not yet implemented)
    #[allow(dead_code)]
    creators_f64: RwLock<HashMap<String, Box<dyn Fn(&ModelConfig) -> RegistryResult<Box<dyn BaseModel<f64>>> + Send + Sync>>>,
    /// Model templates
    templates: RwLock<HashMap<String, ModelTemplate>>,
    /// Model cache
    cache: RwLock<HashMap<String, Arc<dyn std::any::Any + Send + Sync>>>,
    /// Creation statistics
    stats: RwLock<CreationStats>,
}

/// Factory creation statistics
#[derive(Debug, Default)]
pub struct CreationStats {
    /// Total models created
    pub total_created: u64,
    /// Total creation time
    pub total_creation_time_ms: f64,
    /// Models created by category
    pub by_category: HashMap<ModelCategory, u64>,
    /// Average creation time by model
    pub average_times: HashMap<String, f64>,
    /// Cache hit ratio
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
}

impl ModelFactory {
    /// Create a new model factory
    pub fn new() -> Self {
        Self {
            creators: RwLock::new(HashMap::new()),
            creators_f64: RwLock::new(HashMap::new()),
            templates: RwLock::new(HashMap::new()),
            cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(CreationStats::default()),
        }
    }
    
    /// Register a model creator function
    pub fn register_creator<T: Float>(
        &self,
        _name: &str,
        _creator: Box<dyn Fn(&ModelConfig) -> RegistryResult<Box<dyn BaseModel<T>>> + Send + Sync>,
    ) -> RegistryResult<()> {
        // TODO: Implement safe type-erased creator registration
        // For now, we'll just return an error for unsupported operations
        Err(RegistryError::UnsupportedOperation(
            "Generic creator registration not yet implemented without unsafe code".to_string()
        ))
    }
    
    /// Create a model from name
    pub fn create<T: Float>(&self, name: &str) -> RegistryResult<Box<dyn BaseModel<T>>> {
        self.create_with_options(name, CreationOptions::default())
    }
    
    /// Create a model from name with options
    pub fn create_with_options<T: Float>(
        &self,
        name: &str,
        options: CreationOptions,
    ) -> RegistryResult<Box<dyn BaseModel<T>>> {
        let start_time = Instant::now();
        
        // Check cache first if enabled
        if options.cache {
            let cache_key = format!("{}_{}", name, std::any::type_name::<T>());
            if let Some(cached) = self.get_from_cache::<T>(&cache_key) {
                self.stats.write().cache_hits += 1;
                log::debug!("Cache hit for model '{}'", name);
                return Ok(cached);
            }
            self.stats.write().cache_misses += 1;
        }
        
        // Get model info from registry
        let registry = global_registry();
        let reg = registry.read();
        let model_info = reg.get_model_info(name)
            .ok_or_else(|| RegistryError::ModelNotFound(name.to_string()))?;
        
        // Create default config
        let mut config = ModelConfig::new(name, model_info.category);
        
        // Apply overrides
        for (key, value) in &options.overrides {
            config.set_parameter(key, value.clone());
        }
        
        // Validate configuration if enabled
        if options.validate {
            self.validate_config(&config, &model_info)?;
        }
        
        // Create the model
        let model = self.create_from_config_internal::<T>(&config, &options)?;
        
        // Calculate performance metrics
        let creation_time = start_time.elapsed().as_millis() as f64;
        
        if options.benchmark {
            // TODO: Add more detailed benchmarking
            log::debug!("Model '{}' created in {:.2}ms", name, creation_time);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_created += 1;
            stats.total_creation_time_ms += creation_time;
            *stats.by_category.entry(model_info.category).or_insert(0) += 1;
            
            let avg_entry = stats.average_times.entry(name.to_string()).or_insert(0.0);
            *avg_entry = (*avg_entry + creation_time) / 2.0;
        }
        
        // Cache the model if enabled
        if options.cache {
            let cache_key = format!("{}_{}", name, std::any::type_name::<T>());
            self.cache_model(&cache_key, &model);
        }
        
        Ok(model)
    }
    
    /// Create a model from configuration
    pub fn create_from_config<T: Float>(&self, config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<T>>> {
        self.create_from_config_with_options(config, CreationOptions::default())
    }
    
    /// Create a model from configuration with options
    pub fn create_from_config_with_options<T: Float>(
        &self,
        config: &ModelConfig,
        options: CreationOptions,
    ) -> RegistryResult<Box<dyn BaseModel<T>>> {
        let mut final_config = config.clone();
        
        // Apply overrides
        for (key, value) in &options.overrides {
            final_config.set_parameter(key, value.clone());
        }
        
        // Validate if enabled
        if options.validate {
            let registry = global_registry();
            let reg = registry.read();
            if let Some(model_info) = reg.get_model_info(&config.name) {
                self.validate_config(&final_config, &model_info)?;
            }
        }
        
        self.create_from_config_internal(&final_config, &options)
    }
    
    /// Internal method to create model from config
    fn create_from_config_internal<T: Float>(
        &self,
        _config: &ModelConfig,
        _options: &CreationOptions,
    ) -> RegistryResult<Box<dyn BaseModel<T>>> {
        // TODO: Implement safe type-erased model creation
        // For now, return an error since this requires unsafe operations
        Err(RegistryError::UnsupportedOperation(
            "Generic model creation not yet implemented without unsafe code".to_string()
        ))
    }
    
    /// Create multiple models in parallel
    pub fn create_models<T: Float>(&self, names: Vec<&str>) -> RegistryResult<Vec<Box<dyn BaseModel<T>>>> {
        self.create_models_with_options(names, CreationOptions::default())
    }
    
    /// Create multiple models with options
    pub fn create_models_with_options<T: Float>(
        &self,
        names: Vec<&str>,
        options: CreationOptions,
    ) -> RegistryResult<Vec<Box<dyn BaseModel<T>>>> {
        // Sequential implementation to avoid lifetime issues
        // In a production system, you'd want a different approach for parallel execution
        let mut results = Vec::new();
        let mut errors = Vec::new();
        
        for name in names {
            match self.create_with_options::<T>(name, options.clone()) {
                Ok(model) => results.push(model),
                Err(e) => errors.push(format!("{}: {}", name, e)),
            }
        }
        
        if !errors.is_empty() {
            return Err(RegistryError::ModelCreationError(errors.join(", ")));
        }
        
        Ok(results)
    }
    
    /// Create model from template
    pub fn create_from_template<T: Float>(&self, template_name: &str) -> RegistryResult<Box<dyn BaseModel<T>>> {
        let templates = self.templates.read();
        let template = templates.get(template_name)
            .ok_or_else(|| RegistryError::ModelNotFound(template_name.to_string()))?;
        
        self.create_from_config(&template.default_config)
    }
    
    /// Register a model template
    pub fn register_template(&self, template: ModelTemplate) -> RegistryResult<()> {
        let name = template.name.clone();
        self.templates.write().insert(name.clone(), template);
        log::debug!("Registered model template '{}'", name);
        Ok(())
    }
    
    /// List all available models
    pub fn list_models() -> Vec<ModelInfo> {
        let registry = global_registry();
        let reg = registry.read();
        reg.list_all()
    }
    
    /// List models by category
    pub fn list_by_category(category: ModelCategory) -> Vec<ModelInfo> {
        let registry = global_registry();
        let reg = registry.read();
        reg.list_by_category(category)
    }
    
    /// Get model information
    pub fn get_model_info(name: &str) -> Option<ModelInfo> {
        let registry = global_registry();
        let reg = registry.read();
        reg.get_model_info(name)
    }
    
    /// List available templates
    pub fn list_templates(&self) -> Vec<ModelTemplate> {
        let templates = self.templates.read();
        templates.values().cloned().collect()
    }
    
    /// Get creation statistics
    pub fn get_stats(&self) -> CreationStats {
        self.stats.read().clone()
    }
    
    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
        log::debug!("Model cache cleared");
    }
    
    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }
    
    /// Validate model configuration
    fn validate_config(&self, config: &ModelConfig, model_info: &ModelInfo) -> RegistryResult<()> {
        // Validate required parameters based on model info
        // This is a basic validation - more sophisticated validation would be model-specific
        
        if let Some(input_size) = model_info.input_size {
            if let Some(config_input_size) = config.input_size {
                if config_input_size != input_size {
                    return Err(RegistryError::InvalidConfiguration(
                        format!("Input size mismatch: expected {}, got {}", input_size, config_input_size)
                    ));
                }
            }
        }
        
        if let Some(output_size) = model_info.output_size {
            if let Some(config_output_size) = config.output_size {
                if config_output_size != output_size {
                    return Err(RegistryError::InvalidConfiguration(
                        format!("Output size mismatch: expected {}, got {}", output_size, config_output_size)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get model from cache
    fn get_from_cache<T: Float>(&self, key: &str) -> Option<Box<dyn BaseModel<T>>> {
        let cache = self.cache.read();
        if let Some(cached) = cache.get(key) {
            // This is unsafe but necessary for type erasure
            // In a real implementation, we'd need a more sophisticated approach
            if let Some(_model) = cached.downcast_ref::<Box<dyn BaseModel<T>>>() {
                // We can't clone trait objects directly, so this is a placeholder
                // In practice, we'd need models to implement Clone or use Arc<Mutex<Model>>
                log::warn!("Cache retrieval not fully implemented - model cloning needed");
            }
        }
        None
    }
    
    /// Cache a model
    fn cache_model<T: Float>(&self, _key: &str, _model: &Box<dyn BaseModel<T>>) {
        // Placeholder for caching - would need Arc<Mutex<Model>> or similar
        // for thread-safe sharing of models
        log::debug!("Model caching not fully implemented - needs Arc<Mutex<Model>>");
    }
}

impl Default for ModelFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CreationStats {
    fn clone(&self) -> Self {
        Self {
            total_created: self.total_created,
            total_creation_time_ms: self.total_creation_time_ms,
            by_category: self.by_category.clone(),
            average_times: self.average_times.clone(),
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
        }
    }
}

/// Global model factory instance
static GLOBAL_FACTORY: once_cell::sync::Lazy<ModelFactory> = 
    once_cell::sync::Lazy::new(|| ModelFactory::new());

/// Get the global model factory
pub fn global_factory() -> &'static ModelFactory {
    &GLOBAL_FACTORY
}

/// Convenience functions using global factory
impl ModelFactory {
    /// Create model using global factory
    pub fn create_global<T: Float>(name: &str) -> RegistryResult<Box<dyn BaseModel<T>>> {
        global_factory().create(name)
    }
    
    /// Create model from config using global factory
    pub fn create_from_config_global<T: Float>(config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<T>>> {
        global_factory().create_from_config(config)
    }
    
    /// Create multiple models using global factory
    pub fn create_models_global<T: Float>(names: Vec<&str>) -> RegistryResult<Vec<Box<dyn BaseModel<T>>>> {
        global_factory().create_models(names)
    }
}

/// Builder pattern for model creation
pub struct ModelBuilder {
    config: ModelConfig,
    options: CreationOptions,
}

impl ModelBuilder {
    /// Create a new model builder
    pub fn new(name: &str, category: ModelCategory) -> Self {
        Self {
            config: ModelConfig::new(name, category),
            options: CreationOptions::default(),
        }
    }
    
    /// Set a parameter
    pub fn parameter(mut self, key: &str, value: Value) -> Self {
        self.config.set_parameter(key, value);
        self
    }
    
    /// Set input size
    pub fn input_size(mut self, size: usize) -> Self {
        self.config.input_size = Some(size);
        self
    }
    
    /// Set output size
    pub fn output_size(mut self, size: usize) -> Self {
        self.config.output_size = Some(size);
        self
    }
    
    /// Enable benchmarking
    pub fn benchmark(mut self, enable: bool) -> Self {
        self.options.benchmark = enable;
        self
    }
    
    /// Enable validation
    pub fn validate(mut self, enable: bool) -> Self {
        self.options.validate = enable;
        self
    }
    
    /// Enable caching
    pub fn cache(mut self, enable: bool) -> Self {
        self.options.cache = enable;
        self
    }
    
    /// Set timeout
    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.options.timeout_ms = Some(timeout_ms);
        self
    }
    
    /// Add override parameter
    pub fn override_parameter(mut self, key: &str, value: Value) -> Self {
        self.options.overrides.insert(key.to_string(), value);
        self
    }
    
    /// Build the model
    pub fn build<T: Float>(self) -> RegistryResult<Box<dyn BaseModel<T>>> {
        global_factory().create_from_config_with_options(&self.config, self.options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_creation_options_default() {
        let options = CreationOptions::default();
        assert!(!options.benchmark);
        assert!(options.validate);
        assert!(options.cache);
        assert_eq!(options.timeout_ms, Some(30000));
        assert!(options.overrides.is_empty());
    }
    
    #[test]
    fn test_model_factory_new() {
        let factory = ModelFactory::new();
        assert_eq!(factory.cache_size(), 0);
        
        let stats = factory.get_stats();
        assert_eq!(stats.total_created, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }
    
    #[test]
    fn test_model_builder() {
        let builder = ModelBuilder::new("test_model", ModelCategory::Basic)
            .input_size(10)
            .output_size(1)
            .parameter("learning_rate", serde_json::json!(0.001))
            .benchmark(true)
            .validate(false);
        
        assert_eq!(builder.config.input_size, Some(10));
        assert_eq!(builder.config.output_size, Some(1));
        assert!(builder.options.benchmark);
        assert!(!builder.options.validate);
    }
    
    #[test]
    fn test_parameter_constraint() {
        let constraint = ParameterConstraint {
            param_type: ParameterType::Float,
            min_value: Some(0.0),
            max_value: Some(1.0),
            allowed_values: None,
            required: true,
            default_value: Some(serde_json::json!(0.01)),
            description: "Learning rate parameter".to_string(),
        };
        
        assert!(constraint.required);
        assert_eq!(constraint.min_value, Some(0.0));
        assert_eq!(constraint.max_value, Some(1.0));
    }
    
    #[test]
    fn test_creation_stats_clone() {
        let mut stats = CreationStats::default();
        stats.total_created = 5;
        stats.by_category.insert(ModelCategory::Basic, 3);
        
        let cloned = stats.clone();
        assert_eq!(cloned.total_created, 5);
        assert_eq!(cloned.by_category.get(&ModelCategory::Basic), Some(&3));
    }
    
    #[test]
    fn test_global_factory() {
        let factory = global_factory();
        assert_eq!(factory.cache_size(), 0);
    }
}