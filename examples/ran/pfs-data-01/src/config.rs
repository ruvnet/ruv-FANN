//! Configuration management for the data ingestion service

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{IngestionError, IngestionResult};
use crate::schema::StandardSchema;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// File processing configuration
    pub file_patterns: Vec<String>,
    pub recursive: bool,
    pub exclude_patterns: Vec<String>,
    
    /// Data normalization configuration
    pub schema: StandardSchema,
    
    /// Performance configuration
    pub batch_size: usize,
    pub max_concurrent_files: usize,
    pub processing_timeout_seconds: u64,
    
    /// Error handling configuration
    pub max_error_rate: f64,
    pub skip_malformed_rows: bool,
    pub max_retries: usize,
    pub retry_delay_seconds: u64,
    
    /// Output configuration
    pub compression_codec: String,
    pub row_group_size: usize,
    pub enable_statistics: bool,
    pub enable_dictionary_encoding: bool,
    
    /// Monitoring configuration
    pub enable_metrics: bool,
    pub metrics_interval_seconds: u64,
    pub log_level: String,
    
    /// Resource limits
    pub max_memory_mb: usize,
    pub max_file_size_mb: usize,
    pub max_concurrent_jobs: usize,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            file_patterns: vec!["*.csv".to_string(), "*.json".to_string()],
            recursive: true,
            exclude_patterns: vec![
                "*.tmp".to_string(),
                "*.processing".to_string(),
                ".*".to_string(), // Hidden files
            ],
            schema: StandardSchema::default(),
            batch_size: crate::DEFAULT_BATCH_SIZE,
            max_concurrent_files: crate::DEFAULT_MAX_CONCURRENT_FILES,
            processing_timeout_seconds: 300, // 5 minutes
            max_error_rate: crate::DEFAULT_MAX_ERROR_RATE,
            skip_malformed_rows: true,
            max_retries: 3,
            retry_delay_seconds: 30,
            compression_codec: crate::DEFAULT_COMPRESSION.to_string(),
            row_group_size: crate::DEFAULT_ROW_GROUP_SIZE,
            enable_statistics: true,
            enable_dictionary_encoding: true,
            enable_metrics: true,
            metrics_interval_seconds: 60,
            log_level: "info".to_string(),
            max_memory_mb: 8192, // 8GB
            max_file_size_mb: 10240, // 10GB
            max_concurrent_jobs: 10,
        }
    }
}

impl IngestionConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> IngestionResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| IngestionError::config(format!("Failed to parse config file: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
    
    pub fn validate(&self) -> IngestionResult<()> {
        if self.file_patterns.is_empty() {
            return Err(IngestionError::config("file_patterns cannot be empty"));
        }
        
        if self.batch_size == 0 {
            return Err(IngestionError::config("batch_size must be greater than 0"));
        }
        
        if self.max_concurrent_files == 0 {
            return Err(IngestionError::config("max_concurrent_files must be greater than 0"));
        }
        
        if self.max_error_rate < 0.0 || self.max_error_rate > 1.0 {
            return Err(IngestionError::config("max_error_rate must be between 0.0 and 1.0"));
        }
        
        if self.row_group_size == 0 {
            return Err(IngestionError::config("row_group_size must be greater than 0"));
        }
        
        if self.max_memory_mb == 0 {
            return Err(IngestionError::config("max_memory_mb must be greater than 0"));
        }
        
        if self.max_file_size_mb == 0 {
            return Err(IngestionError::config("max_file_size_mb must be greater than 0"));
        }
        
        // Validate compression codec
        match self.compression_codec.as_str() {
            "snappy" | "gzip" | "lz4" | "brotli" | "zstd" | "uncompressed" => {},
            _ => return Err(IngestionError::config(
                format!("unsupported compression codec: {}", self.compression_codec)
            )),
        }
        
        // Validate log level
        match self.log_level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {},
            _ => return Err(IngestionError::config(
                format!("invalid log level: {}", self.log_level)
            )),
        }
        
        self.schema.validate()?;
        
        Ok(())
    }
    
    pub fn with_file_patterns(mut self, patterns: Vec<String>) -> Self {
        self.file_patterns = patterns;
        self
    }
    
    pub fn with_schema(mut self, schema: StandardSchema) -> Self {
        self.schema = schema;
        self
    }
    
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    
    pub fn with_max_error_rate(mut self, max_error_rate: f64) -> Self {
        self.max_error_rate = max_error_rate;
        self
    }
    
    pub fn with_compression(mut self, codec: String) -> Self {
        self.compression_codec = codec;
        self
    }
    
    pub fn optimized_for_performance() -> Self {
        Self {
            batch_size: 50000,
            max_concurrent_files: 8,
            compression_codec: "lz4".to_string(),
            row_group_size: 2000000,
            enable_dictionary_encoding: false,
            ..Default::default()
        }
    }
    
    pub fn optimized_for_memory() -> Self {
        Self {
            batch_size: 5000,
            max_concurrent_files: 2,
            compression_codec: "gzip".to_string(),
            row_group_size: 100000,
            max_memory_mb: 2048,
            ..Default::default()
        }
    }
    
    pub fn optimized_for_quality() -> Self {
        Self {
            max_error_rate: 0.001, // 0.1%
            skip_malformed_rows: false,
            max_retries: 5,
            enable_statistics: true,
            enable_dictionary_encoding: true,
            ..Default::default()
        }
    }
    
    /// Get memory limit in bytes
    pub fn max_memory_bytes(&self) -> usize {
        self.max_memory_mb * 1024 * 1024
    }
    
    /// Get file size limit in bytes
    pub fn max_file_size_bytes(&self) -> usize {
        self.max_file_size_mb * 1024 * 1024
    }
    
    /// Check if file matches any pattern
    pub fn matches_pattern(&self, path: &std::path::Path) -> bool {
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        // Check exclude patterns first
        for pattern in &self.exclude_patterns {
            if glob_match(pattern, filename) {
                return false;
            }
        }
        
        // Check include patterns
        for pattern in &self.file_patterns {
            if glob_match(pattern, filename) {
                return true;
            }
        }
        
        false
    }
}

/// Simple glob pattern matching
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    
    if pattern.starts_with("*.") {
        let ext = &pattern[2..];
        return text.ends_with(ext);
    }
    
    if pattern.starts_with("*") {
        let suffix = &pattern[1..];
        return text.ends_with(suffix);
    }
    
    if pattern.ends_with("*") {
        let prefix = &pattern[..pattern.len()-1];
        return text.starts_with(prefix);
    }
    
    pattern == text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = IngestionConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation_errors() {
        let mut config = IngestionConfig::default();
        
        config.batch_size = 0;
        assert!(config.validate().is_err());
        
        config.batch_size = 1000;
        config.max_error_rate = 1.5;
        assert!(config.validate().is_err());
        
        config.max_error_rate = 0.01;
        config.compression_codec = "invalid".to_string();
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_pattern_matching() {
        let config = IngestionConfig::default();
        
        assert!(config.matches_pattern(std::path::Path::new("data.csv")));
        assert!(config.matches_pattern(std::path::Path::new("data.json")));
        assert!(!config.matches_pattern(std::path::Path::new("data.txt")));
        assert!(!config.matches_pattern(std::path::Path::new("data.tmp")));
        assert!(!config.matches_pattern(std::path::Path::new(".hidden")));
    }
    
    #[test]
    fn test_glob_matching() {
        assert!(glob_match("*.csv", "data.csv"));
        assert!(glob_match("*.json", "test.json"));
        assert!(!glob_match("*.csv", "data.txt"));
        assert!(glob_match("data*", "data.csv"));
        assert!(glob_match("*test", "mytest"));
        assert!(glob_match("*", "anything"));
    }
    
    #[test]
    fn test_optimized_configs() {
        let perf_config = IngestionConfig::optimized_for_performance();
        assert_eq!(perf_config.batch_size, 50000);
        assert_eq!(perf_config.compression_codec, "lz4");
        
        let mem_config = IngestionConfig::optimized_for_memory();
        assert_eq!(mem_config.batch_size, 5000);
        assert_eq!(mem_config.max_memory_mb, 2048);
        
        let quality_config = IngestionConfig::optimized_for_quality();
        assert_eq!(quality_config.max_error_rate, 0.001);
        assert!(!quality_config.skip_malformed_rows);
    }
    
    #[test]
    fn test_memory_calculations() {
        let config = IngestionConfig::default();
        assert_eq!(config.max_memory_bytes(), 8192 * 1024 * 1024);
        assert_eq!(config.max_file_size_bytes(), 10240 * 1024 * 1024);
    }
}