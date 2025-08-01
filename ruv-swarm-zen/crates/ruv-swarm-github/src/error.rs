//! Error types for GitHub integration

use thiserror::Error;

/// Result type alias for GitHub operations
pub type Result<T> = std::result::Result<T, GitHubError>;

/// Errors that can occur during GitHub API operations
#[derive(Error, Debug)]
pub enum GitHubError {
    /// Authentication failed
    #[error("GitHub authentication failed: {message}")]
    AuthenticationError { message: String },

    /// API rate limit exceeded
    #[error("GitHub API rate limit exceeded. Reset at: {reset_time}")]
    RateLimitExceeded { reset_time: String },

    /// Network connection error
    #[error("Network error: {source}")]
    NetworkError {
        #[from]
        source: reqwest::Error,
    },

    /// GitHub API returned an error
    #[error("GitHub API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    /// Repository not found or access denied
    #[error("Repository not found or access denied: {repo}")]
    RepositoryNotFound { repo: String },

    /// Issue not found
    #[error("Issue #{number} not found in repository {repo}")]
    IssueNotFound { number: u64, repo: String },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    ConfigurationError { message: String },

    /// Serialization/deserialization error
    #[error("Serialization error: {source}")]
    SerializationError {
        #[from]
        source: serde_json::Error,
    },

    /// URL parsing error
    #[error("Invalid URL: {source}")]
    UrlError {
        #[from]
        source: url::ParseError,
    },

    /// Generic error with context
    #[error("GitHub operation failed: {message}")]
    OperationFailed { message: String },

    /// Retry attempts exhausted
    #[error("Maximum retry attempts ({max_retries}) exhausted for operation: {operation}")]
    MaxRetriesExceeded {
        max_retries: u32,
        operation: String,
    },

    /// Timeout error
    #[error("Operation timed out after {timeout_ms}ms: {operation}")]
    TimeoutError {
        timeout_ms: u64,
        operation: String,
    },
}

impl GitHubError {
    /// Create a new authentication error
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::AuthenticationError {
            message: message.into(),
        }
    }

    /// Create a new API error
    pub fn api_error(status: u16, message: impl Into<String>) -> Self {
        Self::ApiError {
            status,
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create a new operation failed error
    pub fn operation_failed(message: impl Into<String>) -> Self {
        Self::OperationFailed {
            message: message.into(),
        }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::NetworkError { .. }
                | Self::ApiError { status, .. } if *status >= 500
                | Self::TimeoutError { .. }
        )
    }

    /// Check if error is due to rate limiting
    pub fn is_rate_limited(&self) -> bool {
        matches!(self, Self::RateLimitExceeded { .. })
    }
}

/// Convert octocrab errors to GitHubError
impl From<octocrab::Error> for GitHubError {
    fn from(error: octocrab::Error) -> Self {
        match error {
            octocrab::Error::GitHub { source, .. } => {
                if let Some(status) = source.status_code {
                    match status.as_u16() {
                        401 => GitHubError::authentication("Invalid or expired token"),
                        403 => {
                            if source.message.contains("rate limit") {
                                GitHubError::RateLimitExceeded {
                                    reset_time: "unknown".to_string(),
                                }
                            } else {
                                GitHubError::api_error(403, source.message)
                            }
                        }
                        404 => GitHubError::RepositoryNotFound {
                            repo: "unknown".to_string(),
                        },
                        _ => GitHubError::api_error(status.as_u16(), source.message),
                    }
                } else {
                    GitHubError::operation_failed(source.message)
                }
            }
            octocrab::Error::Http { source, .. } => GitHubError::NetworkError { source },
            octocrab::Error::Serde { source, .. } => GitHubError::SerializationError { source },
            _ => GitHubError::operation_failed(error.to_string()),
        }
    }
}