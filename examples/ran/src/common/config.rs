//! Configuration management for the RAN Intelligence Platform

use crate::common::RanConfig;
use crate::{Result, RanError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Configuration loader and manager
pub struct ConfigManager {
    config: RanConfig,
}

impl ConfigManager {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| RanError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        let config: RanConfig = serde_json::from_str(&content)
            .map_err(|e| RanError::ConfigError(format!("Failed to parse config: {}", e)))?;
        
        Ok(Self { config })
    }
    
    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self {
            config: RanConfig::default(),
        }
    }
    
    /// Get the configuration
    pub fn config(&self) -> &RanConfig {
        &self.config
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.config)
            .map_err(|e| RanError::ConfigError(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(path, content)
            .map_err(|e| RanError::ConfigError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
}