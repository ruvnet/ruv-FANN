//! Input validation for MCP requests
//!
//! This module provides comprehensive input validation to prevent
//! injection attacks, resource exhaustion, and other security issues.

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::path::{Component, Path};

/// Validation error types
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Invalid field type: {field} (expected {expected})")]
    InvalidType { field: String, expected: String },
    
    #[error("Value out of range: {field} (min: {min}, max: {max})")]
    OutOfRange {
        field: String,
        min: String,
        max: String,
    },
    
    #[error("Invalid string format: {0}")]
    InvalidFormat(String),
    
    #[error("Path traversal attempt detected")]
    PathTraversal,
    
    #[error("String too long: {field} (max: {max} characters)")]
    StringTooLong { field: String, max: usize },
    
    #[error("Invalid enum value: {field} (allowed: {allowed})")]
    InvalidEnum { field: String, allowed: String },
    
    #[error("Array too large: {field} (max: {max} items)")]
    ArrayTooLarge { field: String, max: usize },
}

/// Input validator for MCP requests
pub struct InputValidator {
    /// Maximum string length for most fields
    max_string_length: usize,
    /// Maximum array size
    max_array_size: usize,
    /// Maximum number size
    max_number_value: u64,
}

impl Default for InputValidator {
    fn default() -> Self {
        Self {
            max_string_length: 1024,  // 1KB for strings
            max_array_size: 100,      // Max 100 items in arrays
            max_number_value: 1_000_000, // Reasonable upper bound
        }
    }
}

impl InputValidator {
    /// Validate a required string field
    pub fn validate_string(
        &self,
        params: &Value,
        field: &str,
        required: bool,
    ) -> Result<Option<String>> {
        match params.get(field) {
            Some(value) => {
                let str_value = value
                    .as_str()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: field.to_string(),
                        expected: "string".to_string(),
                    })?;
                
                // Check length
                if str_value.len() > self.max_string_length {
                    return Err(ValidationError::StringTooLong {
                        field: field.to_string(),
                        max: self.max_string_length,
                    }.into());
                }
                
                // Basic injection prevention - no null bytes
                if str_value.contains('\0') {
                    return Err(ValidationError::InvalidFormat(
                        "String contains null bytes".to_string()
                    ).into());
                }
                
                Ok(Some(str_value.to_string()))
            }
            None => {
                if required {
                    Err(ValidationError::MissingField(field.to_string()).into())
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Validate a number field with bounds
    pub fn validate_number(
        &self,
        params: &Value,
        field: &str,
        min: Option<u64>,
        max: Option<u64>,
        required: bool,
    ) -> Result<Option<u64>> {
        match params.get(field) {
            Some(value) => {
                let num_value = value
                    .as_u64()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: field.to_string(),
                        expected: "number".to_string(),
                    })?;
                
                // Check bounds
                if let Some(min_val) = min {
                    if num_value < min_val {
                        return Err(ValidationError::OutOfRange {
                            field: field.to_string(),
                            min: min_val.to_string(),
                            max: max.unwrap_or(self.max_number_value).to_string(),
                        }.into());
                    }
                }
                
                if let Some(max_val) = max {
                    if num_value > max_val {
                        return Err(ValidationError::OutOfRange {
                            field: field.to_string(),
                            min: min.unwrap_or(0).to_string(),
                            max: max_val.to_string(),
                        }.into());
                    }
                }
                
                // Global max check
                if num_value > self.max_number_value {
                    return Err(ValidationError::OutOfRange {
                        field: field.to_string(),
                        min: "0".to_string(),
                        max: self.max_number_value.to_string(),
                    }.into());
                }
                
                Ok(Some(num_value))
            }
            None => {
                if required {
                    Err(ValidationError::MissingField(field.to_string()).into())
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Validate an enum field
    pub fn validate_enum<T: AsRef<str>>(
        &self,
        params: &Value,
        field: &str,
        allowed_values: &[T],
        required: bool,
    ) -> Result<Option<String>> {
        match self.validate_string(params, field, required)? {
            Some(value) => {
                let value_ref = value.as_str();
                if allowed_values.iter().any(|v| v.as_ref() == value_ref) {
                    Ok(Some(value))
                } else {
                    let allowed_str = allowed_values
                        .iter()
                        .map(|v| v.as_ref())
                        .collect::<Vec<_>>()
                        .join(", ");
                    Err(ValidationError::InvalidEnum {
                        field: field.to_string(),
                        allowed: allowed_str,
                    }.into())
                }
            }
            None => Ok(None),
        }
    }

    /// Validate an array field
    pub fn validate_array(
        &self,
        params: &Value,
        field: &str,
        required: bool,
    ) -> Result<Option<Vec<Value>>> {
        match params.get(field) {
            Some(value) => {
                let arr_value = value
                    .as_array()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: field.to_string(),
                        expected: "array".to_string(),
                    })?;
                
                // Check size
                if arr_value.len() > self.max_array_size {
                    return Err(ValidationError::ArrayTooLarge {
                        field: field.to_string(),
                        max: self.max_array_size,
                    }.into());
                }
                
                Ok(Some(arr_value.clone()))
            }
            None => {
                if required {
                    Err(ValidationError::MissingField(field.to_string()).into())
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Validate a file path to prevent directory traversal
    pub fn validate_path(&self, path_str: &str) -> Result<String> {
        let path = Path::new(path_str);
        
        // Check for absolute paths - not allowed
        if path.is_absolute() {
            return Err(ValidationError::PathTraversal.into());
        }
        
        // Check each component
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    // Reject any parent directory references
                    return Err(ValidationError::PathTraversal.into());
                }
                Component::Normal(os_str) => {
                    // Ensure valid UTF-8
                    os_str.to_str()
                        .ok_or_else(|| anyhow!("Invalid UTF-8 in path"))?;
                }
                _ => {}
            }
        }
        
        // Additional checks for common attack patterns
        let path_lower = path_str.to_lowercase();
        let dangerous_patterns = [
            "..",
            "./",
            ".\\",
            "%2e%2e",
            "%252e%252e",
            "..%2f",
            "..%5c",
            "%c0%ae",
            "%c1%9c",
        ];
        
        for pattern in &dangerous_patterns {
            if path_lower.contains(pattern) {
                return Err(ValidationError::PathTraversal.into());
            }
        }
        
        Ok(path_str.to_string())
    }

    /// Validate boolean field
    pub fn validate_bool(
        &self,
        params: &Value,
        field: &str,
        required: bool,
    ) -> Result<Option<bool>> {
        match params.get(field) {
            Some(value) => {
                let bool_value = value
                    .as_bool()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: field.to_string(),
                        expected: "boolean".to_string(),
                    })?;
                Ok(Some(bool_value))
            }
            None => {
                if required {
                    Err(ValidationError::MissingField(field.to_string()).into())
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// Specific validators for MCP tool parameters
impl InputValidator {
    /// Validate spawn agent parameters
    pub fn validate_spawn_params(&self, params: &Value) -> Result<()> {
        // Validate agent_type (required)
        self.validate_enum(
            params,
            "agent_type",
            &["researcher", "coder", "analyst", "tester", "reviewer", "documenter"],
            true,
        )?;
        
        // Validate name (optional)
        if let Some(name) = self.validate_string(params, "name", false)? {
            // Additional name validation
            if name.len() > 100 {
                return Err(anyhow!("Agent name too long (max 100 characters)"));
            }
        }
        
        // Validate capabilities (optional)
        if params.get("capabilities").is_some() {
            // Ensure it's an object
            params.get("capabilities")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow!("capabilities must be an object"))?;
        }
        
        Ok(())
    }

    /// Validate orchestrate parameters
    pub fn validate_orchestrate_params(&self, params: &Value) -> Result<()> {
        // Validate objective (required)
        self.validate_string(params, "objective", true)?;
        
        // Validate strategy (optional)
        self.validate_enum(
            params,
            "strategy",
            &["development", "research", "analysis", "testing"],
            false,
        )?;
        
        // Validate max_agents (optional)
        self.validate_number(params, "max_agents", Some(1), Some(100), false)?;
        
        // Validate max_iterations (optional)
        self.validate_number(params, "max_iterations", Some(1), Some(1000), false)?;
        
        // Validate timeout_minutes (optional)
        self.validate_number(params, "timeout_minutes", Some(1), Some(600), false)?;
        
        Ok(())
    }

    /// Validate memory store parameters
    pub fn validate_memory_store_params(&self, params: &Value) -> Result<()> {
        // Validate key (required)
        let key = self.validate_string(params, "key", true)?
            .ok_or_else(|| anyhow!("key is required"))?;
        
        // Key format validation
        if key.len() > 256 {
            return Err(anyhow!("Memory key too long (max 256 characters)"));
        }
        
        // Validate value (required)
        if params.get("value").is_none() {
            return Err(ValidationError::MissingField("value".to_string()).into());
        }
        
        // Validate ttl_secs (optional)
        self.validate_number(params, "ttl_secs", Some(1), Some(86400 * 30), false)?; // Max 30 days
        
        Ok(())
    }

    /// Validate workflow execution parameters
    pub fn validate_workflow_params(&self, params: &Value) -> Result<()> {
        // Validate workflow_path (required)
        let workflow_path = self.validate_string(params, "workflow_path", true)?
            .ok_or_else(|| anyhow!("workflow_path is required"))?;
        
        // Validate path security
        self.validate_path(&workflow_path)?;
        
        // Validate parameters (optional)
        if params.get("parameters").is_some() {
            // Ensure it's an object
            params.get("parameters")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow!("parameters must be an object"))?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_string_validation() {
        let validator = InputValidator::default();
        let params = json!({
            "name": "test_agent",
            "long_string": "x".repeat(2000),
        });
        
        // Valid string
        assert!(validator.validate_string(&params, "name", true).is_ok());
        
        // Too long string
        assert!(validator.validate_string(&params, "long_string", true).is_err());
        
        // Missing required field
        assert!(validator.validate_string(&params, "missing", true).is_err());
        
        // Missing optional field
        assert!(validator.validate_string(&params, "missing", false).is_ok());
    }

    #[test]
    fn test_path_traversal_prevention() {
        let validator = InputValidator::default();
        
        // Valid paths
        assert!(validator.validate_path("workflows/test.yaml").is_ok());
        assert!(validator.validate_path("subdir/file.txt").is_ok());
        
        // Invalid paths
        assert!(validator.validate_path("../etc/passwd").is_err());
        assert!(validator.validate_path("..\\windows\\system32").is_err());
        assert!(validator.validate_path("/etc/passwd").is_err());
        assert!(validator.validate_path("workflows/../../../etc/passwd").is_err());
        assert!(validator.validate_path("workflows/%2e%2e/secret").is_err());
    }

    #[test]
    fn test_number_validation() {
        let validator = InputValidator::default();
        let params = json!({
            "count": 50,
            "huge": 10000000,
        });
        
        // Valid number in range
        assert!(validator.validate_number(&params, "count", Some(1), Some(100), true).is_ok());
        
        // Number too large
        assert!(validator.validate_number(&params, "huge", Some(1), Some(1000), true).is_err());
    }

    #[test]
    fn test_enum_validation() {
        let validator = InputValidator::default();
        let params = json!({
            "agent_type": "researcher",
            "invalid_type": "hacker",
        });
        
        let allowed = ["researcher", "coder", "analyst"];
        
        // Valid enum
        assert!(validator.validate_enum(&params, "agent_type", &allowed, true).is_ok());
        
        // Invalid enum
        assert!(validator.validate_enum(&params, "invalid_type", &allowed, true).is_err());
    }
}