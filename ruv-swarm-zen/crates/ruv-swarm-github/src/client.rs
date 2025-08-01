//! GitHub API client implementation

use crate::{
    config::GitHubConfig,
    error::{GitHubError, Result},
    rate_limiter::RateLimiter,
};
use backoff::{future::retry, ExponentialBackoff};
use chrono::{DateTime, Utc};
use octocrab::{models::IssueState, Octocrab, OctocrabBuilder};
use serde_json::Value;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// GitHub API client with rate limiting and retry logic
pub struct GitHubClient {
    octocrab: Octocrab,
    config: GitHubConfig,
    rate_limiter: RateLimiter,
}

impl GitHubClient {
    /// Create a new GitHub client
    /// 
    /// # Arguments
    /// 
    /// * `config` - GitHub configuration including authentication and repository details
    /// 
    /// # Errors
    /// 
    /// Returns an error if client initialization fails or authentication is invalid
    pub async fn new(config: GitHubConfig) -> Result<Self> {
        config.validate()?;
        
        let mut builder = OctocrabBuilder::new();
        
        // Set authentication token
        builder = builder.personal_token(config.token.clone());
        
        // Set custom user agent
        builder = builder.user_agent(&config.user_agent)?;
        
        // Set base URL for GitHub Enterprise
        if config.api_base_url != "https://api.github.com" {
            builder = builder.base_uri(&config.api_base_url)?;
        }
        
        let octocrab = builder.build()?;
        
        // Create rate limiter
        let rate_limiter = RateLimiter::new(config.rate_limit_per_hour)?;
        
        let client = Self {
            octocrab,
            config,
            rate_limiter,
        };
        
        // Test authentication
        client.test_authentication().await?;
        
        info!(
            "GitHub client initialized for repository: {}",
            client.config.repo_identifier()
        );
        
        Ok(client)
    }
    
    /// Test GitHub authentication
    async fn test_authentication(&self) -> Result<()> {
        debug!("Testing GitHub authentication");
        
        let operation = || async {
            self.rate_limiter.wait_for_permission().await?;
            
            let timeout_duration = self.config.timeout();
            let auth_check = timeout(
                timeout_duration,
                self.octocrab.current().user()
            ).await
            .map_err(|_| GitHubError::TimeoutError {
                timeout_ms: timeout_duration.as_millis() as u64,
                operation: "authentication test".to_string(),
            })?;
            
            auth_check.map_err(GitHubError::from)
        };
        
        let backoff = self.create_backoff();
        retry(backoff, operation).await.map_err(|e| match e {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { err, .. } => {
                GitHubError::MaxRetriesExceeded {
                    max_retries: self.config.max_retries,
                    operation: "authentication test".to_string(),
                }
            }
        })?;
        
        debug!("GitHub authentication successful");
        Ok(())
    }
    
    /// Post a comment to an issue
    /// 
    /// # Arguments
    /// 
    /// * `issue_number` - The issue number to comment on
    /// * `comment` - The comment text to post
    /// 
    /// # Errors
    /// 
    /// Returns an error if the comment cannot be posted
    pub async fn post_issue_comment(&self, issue_number: u64, comment: &str) -> Result<()> {
        info!("Posting comment to issue #{}", issue_number);
        debug!("Comment content preview: {}", &comment[..comment.len().min(100)]);
        
        let operation = || async {
            self.rate_limiter.wait_for_permission().await?;
            
            let timeout_duration = self.config.timeout();
            let comment_result = timeout(
                timeout_duration,
                self.octocrab
                    .issues(&self.config.repo_owner, &self.config.repo_name)
                    .create_comment(issue_number, comment)
            ).await
            .map_err(|_| GitHubError::TimeoutError {
                timeout_ms: timeout_duration.as_millis() as u64,
                operation: format!("post comment to issue #{}", issue_number),
            })?;
            
            comment_result.map_err(GitHubError::from)
        };
        
        let backoff = self.create_backoff();
        retry(backoff, operation).await.map_err(|e| match e {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { err, .. } => {
                GitHubError::MaxRetriesExceeded {
                    max_retries: self.config.max_retries,
                    operation: format!("post comment to issue #{}", issue_number),
                }
            }
        })?;
        
        info!("Successfully posted comment to issue #{}", issue_number);
        Ok(())
    }
    
    /// Update an issue's description/body
    /// 
    /// # Arguments
    /// 
    /// * `issue_number` - The issue number to update
    /// * `body` - The new issue body content
    /// 
    /// # Errors
    /// 
    /// Returns an error if the issue cannot be updated
    pub async fn update_issue_description(&self, issue_number: u64, body: &str) -> Result<()> {
        info!("Updating description for issue #{}", issue_number);
        debug!("New body preview: {}", &body[..body.len().min(100)]);
        
        let operation = || async {
            self.rate_limiter.wait_for_permission().await?;
            
            let timeout_duration = self.config.timeout();
            let update_result = timeout(
                timeout_duration,
                self.octocrab
                    .issues(&self.config.repo_owner, &self.config.repo_name)
                    .update(issue_number)
                    .body(body)
                    .send()
            ).await
            .map_err(|_| GitHubError::TimeoutError {
                timeout_ms: timeout_duration.as_millis() as u64,
                operation: format!("update issue #{}", issue_number),
            })?;
            
            update_result.map_err(GitHubError::from)
        };
        
        let backoff = self.create_backoff();
        retry(backoff, operation).await.map_err(|e| match e {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { err, .. } => {
                GitHubError::MaxRetriesExceeded {
                    max_retries: self.config.max_retries,
                    operation: format!("update issue #{}", issue_number),
                }
            }
        })?;
        
        info!("Successfully updated description for issue #{}", issue_number);
        Ok(())
    }
    
    /// Post a comment to a pull request
    /// 
    /// # Arguments
    /// 
    /// * `pr_number` - The pull request number to comment on
    /// * `comment` - The comment text to post
    /// 
    /// # Errors
    /// 
    /// Returns an error if the comment cannot be posted
    pub async fn post_pr_comment(&self, pr_number: u64, comment: &str) -> Result<()> {
        info!("Posting comment to PR #{}", pr_number);
        debug!("Comment content preview: {}", &comment[..comment.len().min(100)]);
        
        let operation = || async {
            self.rate_limiter.wait_for_permission().await?;
            
            let timeout_duration = self.config.timeout();
            let comment_result = timeout(
                timeout_duration,
                self.octocrab
                    .pulls(&self.config.repo_owner, &self.config.repo_name)
                    .create_review_comment(pr_number, comment, "", 1) // Simple comment
            ).await
            .map_err(|_| GitHubError::TimeoutError {
                timeout_ms: timeout_duration.as_millis() as u64,
                operation: format!("post comment to PR #{}", pr_number),
            })?;
            
            comment_result.map_err(GitHubError::from)
        };
        
        let backoff = self.create_backoff();
        retry(backoff, operation).await.map_err(|e| match e {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { err, .. } => {
                GitHubError::MaxRetriesExceeded {
                    max_retries: self.config.max_retries,
                    operation: format!("post comment to PR #{}", pr_number),
                }
            }
        })?;
        
        info!("Successfully posted comment to PR #{}", pr_number);
        Ok(())
    }
    
    /// Create a new issue
    /// 
    /// # Arguments
    /// 
    /// * `title` - The issue title
    /// * `body` - The issue body content
    /// * `labels` - Optional labels to apply
    /// 
    /// # Errors
    /// 
    /// Returns an error if the issue cannot be created
    pub async fn create_issue(
        &self,
        title: &str,
        body: &str,
        labels: Option<Vec<String>>,
    ) -> Result<u64> {
        info!("Creating new issue: {}", title);
        
        let operation = || async {
            self.rate_limiter.wait_for_permission().await?;
            
            let timeout_duration = self.config.timeout();
            
            let mut issue_builder = self.octocrab
                .issues(&self.config.repo_owner, &self.config.repo_name)
                .create(title)
                .body(body);
            
            if let Some(ref labels) = labels {
                issue_builder = issue_builder.labels(labels.iter().map(String::as_str));
            }
            
            let create_result = timeout(
                timeout_duration,
                issue_builder.send()
            ).await
            .map_err(|_| GitHubError::TimeoutError {
                timeout_ms: timeout_duration.as_millis() as u64,
                operation: "create issue".to_string(),
            })?;
            
            create_result.map_err(GitHubError::from)
        };
        
        let backoff = self.create_backoff();
        let issue = retry(backoff, operation).await.map_err(|e| match e {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { err, .. } => {
                GitHubError::MaxRetriesExceeded {
                    max_retries: self.config.max_retries,
                    operation: "create issue".to_string(),
                }
            }
        })?;
        
        let issue_number = issue.number;
        info!("Successfully created issue #{}", issue_number);
        Ok(issue_number)
    }
    
    /// Get repository information
    /// 
    /// # Errors
    /// 
    /// Returns an error if repository information cannot be retrieved
    pub async fn get_repository_info(&self) -> Result<Value> {
        debug!("Fetching repository information");
        
        let operation = || async {
            self.rate_limiter.wait_for_permission().await?;
            
            let timeout_duration = self.config.timeout();
            let repo_result = timeout(
                timeout_duration,
                self.octocrab
                    .repos(&self.config.repo_owner, &self.config.repo_name)
                    .get()
            ).await
            .map_err(|_| GitHubError::TimeoutError {
                timeout_ms: timeout_duration.as_millis() as u64,
                operation: "get repository info".to_string(),
            })?;
            
            repo_result.map_err(GitHubError::from)
        };
        
        let backoff = self.create_backoff();
        let repo = retry(backoff, operation).await.map_err(|e| match e {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { err, .. } => {
                GitHubError::MaxRetriesExceeded {
                    max_retries: self.config.max_retries,
                    operation: "get repository info".to_string(),
                }
            }
        })?;
        
        let repo_json = serde_json::to_value(repo)?;
        debug!("Successfully retrieved repository information");
        Ok(repo_json)
    }
    
    /// Get current rate limit status
    /// 
    /// # Errors
    /// 
    /// Returns an error if rate limit information cannot be retrieved
    pub async fn get_rate_limit_status(&self) -> Result<Value> {
        debug!("Fetching rate limit status");
        
        let operation = || async {
            self.rate_limiter.wait_for_permission().await?;
            
            let timeout_duration = self.config.timeout();
            let rate_limit_result = timeout(
                timeout_duration,
                self.octocrab.ratelimit().get()
            ).await
            .map_err(|_| GitHubError::TimeoutError {
                timeout_ms: timeout_duration.as_millis() as u64,
                operation: "get rate limit status".to_string(),
            })?;
            
            rate_limit_result.map_err(GitHubError::from)
        };
        
        let backoff = self.create_backoff();
        let rate_limit = retry(backoff, operation).await.map_err(|e| match e {
            backoff::Error::Permanent(err) => err,
            backoff::Error::Transient { err, .. } => {
                GitHubError::MaxRetriesExceeded {
                    max_retries: self.config.max_retries,
                    operation: "get rate limit status".to_string(),
                }
            }
        })?;
        
        let rate_limit_json = serde_json::to_value(rate_limit)?;
        debug!("Successfully retrieved rate limit status");
        Ok(rate_limit_json)
    }
    
    /// Get configuration
    pub fn config(&self) -> &GitHubConfig {
        &self.config
    }
    
    /// Check if we can make a request immediately
    pub fn can_make_request_immediately(&self) -> bool {
        self.rate_limiter.can_proceed_immediately()
    }
    
    /// Get estimated wait time for next request
    pub fn estimated_wait_time(&self) -> Option<Duration> {
        self.rate_limiter.estimated_wait_time()
    }
    
    /// Create exponential backoff configuration
    fn create_backoff(&self) -> ExponentialBackoff {
        ExponentialBackoff {
            current_interval: self.config.retry_delay(),
            initial_interval: self.config.retry_delay(),
            multiplier: 2.0,
            max_interval: self.config.max_retry_delay(),
            max_elapsed_time: Some(self.config.max_retry_delay() * self.config.max_retries),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    fn get_test_config() -> Option<GitHubConfig> {
        let token = env::var("GITHUB_TOKEN").ok()?;
        let owner = env::var("GITHUB_REPO_OWNER").ok()?;
        let name = env::var("GITHUB_REPO_NAME").ok()?;
        
        Some(GitHubConfig::new(token, owner, name))
    }
    
    #[tokio::test]
    async fn test_client_creation_without_token() {
        let config = GitHubConfig::new("", "owner", "repo");
        let result = GitHubClient::new(config).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_config_validation() {
        let config = GitHubConfig::new("token", "", "repo");
        let result = config.validate();
        assert!(result.is_err());
    }
    
    // Integration tests (require real GitHub token)
    #[tokio::test]
    #[ignore = "requires real GitHub token"]
    async fn test_authentication_with_real_token() {
        let Some(config) = get_test_config() else {
            return;
        };
        
        let result = GitHubClient::new(config).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    #[ignore = "requires real GitHub token"]
    async fn test_get_repository_info() {
        let Some(config) = get_test_config() else {
            return;
        };
        
        let client = GitHubClient::new(config).await.unwrap();
        let result = client.get_repository_info().await;
        assert!(result.is_ok());
    }
}