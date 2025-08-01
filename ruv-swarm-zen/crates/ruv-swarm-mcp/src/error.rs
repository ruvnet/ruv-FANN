//! Secure error handling and sanitization
//!
//! This module provides utilities for sanitizing error messages to prevent
//! information leakage in production environments.

use std::fmt;

/// Security error types that map internal errors to safe external messages
#[derive(Debug, Clone)]
pub enum SecurityError {
    /// Invalid input parameters
    InvalidInput,
    /// Authentication required or failed
    AuthenticationRequired,
    /// Insufficient permissions
    Unauthorized,
    /// Resource not found
    NotFound,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Server error (generic)
    InternalError,
    /// Invalid request format
    BadRequest,
    /// Method not allowed
    MethodNotAllowed,
    /// Resource conflict
    Conflict,
    /// Service temporarily unavailable
    ServiceUnavailable,
}

impl SecurityError {
    /// Convert from various error types to a security error
    pub fn from_error(err: &anyhow::Error) -> Self {
        // TODO: Add more sophisticated error mapping based on error types
        // For now, default to internal error to avoid leaking information
        if err.to_string().contains("validation") || err.to_string().contains("invalid") {
            Self::InvalidInput
        } else if err.to_string().contains("auth") || err.to_string().contains("token") {
            Self::AuthenticationRequired
        } else if err.to_string().contains("permission") || err.to_string().contains("forbidden") {
            Self::Unauthorized
        } else if err.to_string().contains("not found") || err.to_string().contains("missing") {
            Self::NotFound
        } else if err.to_string().contains("rate") || err.to_string().contains("limit") {
            Self::RateLimitExceeded
        } else {
            Self::InternalError
        }
    }

    /// Get the user-safe error message
    pub fn message(&self) -> &'static str {
        match self {
            Self::InvalidInput => "Invalid input parameters provided",
            Self::AuthenticationRequired => "Authentication required",
            Self::Unauthorized => "Insufficient permissions for this operation",
            Self::NotFound => "Requested resource not found",
            Self::RateLimitExceeded => "Rate limit exceeded, please try again later",
            Self::InternalError => "Internal server error",
            Self::BadRequest => "Invalid request format",
            Self::MethodNotAllowed => "Method not allowed",
            Self::Conflict => "Resource conflict",
            Self::ServiceUnavailable => "Service temporarily unavailable",
        }
    }

    /// Get the error code for JSON-RPC
    pub fn code(&self) -> i32 {
        match self {
            Self::InvalidInput => -32602,         // Invalid params
            Self::AuthenticationRequired => -32001, // Custom auth error
            Self::Unauthorized => -32002,         // Custom permission error
            Self::NotFound => -32003,            // Custom not found error
            Self::RateLimitExceeded => -32004,   // Custom rate limit error
            Self::InternalError => -32603,       // Internal error
            Self::BadRequest => -32600,          // Invalid request
            Self::MethodNotAllowed => -32601,    // Method not found
            Self::Conflict => -32005,            // Custom conflict error
            Self::ServiceUnavailable => -32006,  // Custom unavailable error
        }
    }

    /// Get HTTP status code equivalent
    pub fn http_status(&self) -> u16 {
        match self {
            Self::InvalidInput => 400,
            Self::AuthenticationRequired => 401,
            Self::Unauthorized => 403,
            Self::NotFound => 404,
            Self::RateLimitExceeded => 429,
            Self::InternalError => 500,
            Self::BadRequest => 400,
            Self::MethodNotAllowed => 405,
            Self::Conflict => 409,
            Self::ServiceUnavailable => 503,
        }
    }
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for SecurityError {}

/// Sanitize error messages for production use
pub fn sanitize_error(err: &anyhow::Error, debug_mode: bool) -> String {
    if debug_mode {
        // In debug mode, return the full error chain for development
        format!("{:#}", err)
    } else {
        // In production, return safe error messages only
        let security_error = SecurityError::from_error(err);
        security_error.message().to_string()
    }
}

/// Sanitize error messages with custom mapping
pub fn sanitize_error_with_context(
    err: &anyhow::Error,
    context: &str,
    debug_mode: bool,
) -> String {
    if debug_mode {
        format!("{}: {:#}", context, err)
    } else {
        // Use context to provide better error mapping
        let security_error = match context {
            "authentication" => SecurityError::AuthenticationRequired,
            "authorization" => SecurityError::Unauthorized,
            "validation" | "input" => SecurityError::InvalidInput,
            "rate_limit" => SecurityError::RateLimitExceeded,
            _ => SecurityError::from_error(err),
        };
        security_error.message().to_string()
    }
}

/// Log security-relevant errors while returning safe messages
pub fn log_and_sanitize_error(
    err: &anyhow::Error,
    session_id: &uuid::Uuid,
    debug_mode: bool,
) -> String {
    // Always log the full error internally for debugging
    tracing::error!(
        session_id = %session_id,
        error = %err,
        error_chain = ?err,
        "Security error occurred"
    );

    // Return sanitized message to user
    sanitize_error(err, debug_mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn test_error_sanitization() {
        let err = anyhow!("Database connection failed: timeout connecting to postgres://user:pass@localhost/db");
        
        // Production mode should hide sensitive details
        let sanitized = sanitize_error(&err, false);
        assert_eq!(sanitized, "Internal server error");
        assert!(!sanitized.contains("postgres"));
        assert!(!sanitized.contains("user:pass"));
        
        // Debug mode should show full error
        let debug_output = sanitize_error(&err, true);
        assert!(debug_output.contains("Database connection failed"));
    }

    #[test]
    fn test_validation_error_mapping() {
        let err = anyhow!("Validation failed: invalid email format");
        let sanitized = sanitize_error(&err, false);
        assert_eq!(sanitized, "Invalid input parameters provided");
    }

    #[test]
    fn test_auth_error_mapping() {
        let err = anyhow!("JWT token expired");
        let sanitized = sanitize_error(&err, false);
        assert_eq!(sanitized, "Authentication required");
    }
}