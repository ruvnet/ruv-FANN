//! Configuration management for GitHub integration

use crate::error::{GitHubError, Result};
use figment::{Figment, providers::{Env, Serialized}};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// GitHub API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubConfig {
    /// GitHub personal access token
    pub token: String,
    
    /// Repository owner (username or organization)
    pub repo_owner: String,
    
    /// Repository name
    pub repo_name: String,
    
    /// Issue number for posting updates (optional)
    pub issue_number: Option<u64>,
    
    /// Pull request number for posting updates (optional)
    pub pr_number: Option<u64>,
    
    /// Maximum number of retry attempts
    pub max_retries: u32,
    
    /// Initial retry delay in milliseconds
    pub retry_delay_ms: u64,
    
    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: u64,
    
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    
    /// Rate limit requests per hour
    pub rate_limit_per_hour: u32,
    
    /// User agent string for API requests
    pub user_agent: String,
    
    /// Base URL for GitHub API (for GitHub Enterprise)
    pub api_base_url: String,
    
    /// Whether to verify SSL certificates
    pub verify_ssl: bool,
    
    /// Update frequency in minutes
    pub update_frequency_minutes: u64,
}

impl Default for GitHubConfig {
    fn default() -> Self {
        Self {
            token: String::new(),
            repo_owner: String::new(),
            repo_name: String::new(),
            issue_number: None,
            pr_number: None,
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            timeout_seconds: 30,
            rate_limit_per_hour: 5000, // GitHub's default rate limit
            user_agent: format!("ruv-swarm-github/{}", crate::VERSION),
            api_base_url: "https://api.github.com".to_string(),
            verify_ssl: true,
            update_frequency_minutes: 5,
        }
    }
}

impl GitHubConfig {
    /// Create a new configuration from environment variables and defaults
    pub fn from_env() -> Result<Self> {
        let config: GitHubConfig = Figment::new()
            .merge(Serialized::defaults(GitHubConfig::default()))
            .merge(Env::prefixed("GITHUB_"))
            .extract()
            .map_err(|e| GitHubError::configuration(format!("Failed to load configuration: {e}")))?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Create a new configuration with required fields
    pub fn new(
        token: impl Into<String>,
        repo_owner: impl Into<String>,
        repo_name: impl Into<String>,
    ) -> Self {
        Self {
            token: token.into(),
            repo_owner: repo_owner.into(),
            repo_name: repo_name.into(),
            ..Default::default()
        }
    }
    
    /// Set the issue number for updates
    pub fn with_issue(mut self, issue_number: u64) -> Self {
        self.issue_number = Some(issue_number);
        self.pr_number = None; // Clear PR number if issue is set
        self
    }
    
    /// Set the pull request number for updates
    pub fn with_pr(mut self, pr_number: u64) -> Self {
        self.pr_number = Some(pr_number);
        self.issue_number = None; // Clear issue number if PR is set
        self
    }
    
    /// Set retry configuration
    pub fn with_retry_config(
        mut self,
        max_retries: u32,
        initial_delay_ms: u64,
        max_delay_ms: u64,
    ) -> Self {
        self.max_retries = max_retries;
        self.retry_delay_ms = initial_delay_ms;
        self.max_retry_delay_ms = max_delay_ms;
        self
    }
    
    /// Set rate limiting configuration
    pub fn with_rate_limit(mut self, requests_per_hour: u32) -> Self {
        self.rate_limit_per_hour = requests_per_hour;
        self
    }
    
    /// Set timeout configuration
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }
    
    /// Set custom user agent
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }
    
    /// Set GitHub Enterprise API URL
    pub fn with_enterprise_url(mut self, api_base_url: impl Into<String>) -> Self {
        self.api_base_url = api_base_url.into();
        self
    }
    
    /// Get the repository identifier as "owner/name"
    pub fn repo_identifier(&self) -> String {
        format!("{}/{}", self.repo_owner, self.repo_name)
    }
    
    /// Get retry delay as Duration
    pub fn retry_delay(&self) -> Duration {
        Duration::from_millis(self.retry_delay_ms)
    }
    
    /// Get max retry delay as Duration
    pub fn max_retry_delay(&self) -> Duration {
        Duration::from_millis(self.max_retry_delay_ms)
    }
    
    /// Get timeout as Duration
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds)
    }
    
    /// Get update frequency as Duration
    pub fn update_frequency(&self) -> Duration {
        Duration::from_secs(self.update_frequency_minutes * 60)
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.token.is_empty() {
            return Err(GitHubError::configuration(
                "GitHub token is required. Set GITHUB_TOKEN environment variable"
            ));
        }
        
        if self.repo_owner.is_empty() {
            return Err(GitHubError::configuration(
                "Repository owner is required. Set GITHUB_REPO_OWNER environment variable"
            ));
        }
        
        if self.repo_name.is_empty() {
            return Err(GitHubError::configuration(
                "Repository name is required. Set GITHUB_REPO_NAME environment variable"
            ));
        }
        
        if self.issue_number.is_some() && self.pr_number.is_some() {
            return Err(GitHubError::configuration(
                "Cannot specify both issue_number and pr_number. Choose one target"
            ));
        }
        
        if self.max_retries == 0 {
            return Err(GitHubError::configuration(
                "max_retries must be greater than 0"
            ));
        }
        
        if self.retry_delay_ms == 0 {
            return Err(GitHubError::configuration(
                "retry_delay_ms must be greater than 0"
            ));
        }
        
        if self.max_retry_delay_ms < self.retry_delay_ms {
            return Err(GitHubError::configuration(
                "max_retry_delay_ms must be greater than or equal to retry_delay_ms"
            ));
        }
        
        if self.timeout_seconds == 0 {
            return Err(GitHubError::configuration(
                "timeout_seconds must be greater than 0"
            ));
        }
        
        if self.rate_limit_per_hour == 0 {
            return Err(GitHubError::configuration(
                "rate_limit_per_hour must be greater than 0"
            ));
        }
        
        // Validate URL format
        url::Url::parse(&self.api_base_url)
            .map_err(|e| GitHubError::configuration(format!("Invalid API base URL: {e}")))?;
        
        Ok(())
    }
}