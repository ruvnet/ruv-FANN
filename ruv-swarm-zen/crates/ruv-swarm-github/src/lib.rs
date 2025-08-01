//! GitHub API integration for RUV Swarm coordinator progress reporting
//!
//! This crate provides production-ready GitHub API integration for posting
//! swarm orchestration progress updates to GitHub issues and pull requests.
//!
//! # Features
//!
//! - **Real GitHub API Integration**: Uses octocrab for authenticated API calls
//! - **Rate Limiting**: Respects GitHub API rate limits with exponential backoff
//! - **Error Handling**: Comprehensive error handling for network and API failures
//! - **Progress Reporting**: Formats and posts swarm status updates
//! - **Authentication**: Secure token-based authentication
//!
//! # Example
//!
//! ```rust,no_run
//! use ruv_swarm_github::{GitHubClient, ProgressReporter, GitHubConfig};
//! use ruv_swarm_core::{Swarm, SwarmConfig};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = GitHubConfig {
//!         token: std::env::var("GITHUB_TOKEN")?,
//!         repo_owner: "ruvnet".to_string(),
//!         repo_name: "ruv-FANN".to_string(),
//!         issue_number: Some(123),
//!         ..Default::default()
//!     };
//!     
//!     let client = GitHubClient::new(config).await?;
//!     let reporter = ProgressReporter::new(client);
//!     
//!     // Create a swarm and post progress
//!     let swarm = Swarm::new(SwarmConfig::default());
//!     reporter.post_swarm_status(&swarm, "Initial swarm setup complete").await?;
//!     
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod client;
pub mod error;
pub mod reporter;
pub mod config;
pub mod rate_limiter;

// Re-export commonly used types
pub use client::GitHubClient;
pub use error::{GitHubError, Result};
pub use reporter::ProgressReporter;
pub use config::GitHubConfig;
pub use rate_limiter::RateLimiter;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::client::GitHubClient;
    pub use crate::error::{GitHubError, Result};
    pub use crate::reporter::ProgressReporter;
    pub use crate::config::GitHubConfig;
}