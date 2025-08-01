# ruv-swarm-github

[![Crates.io](https://img.shields.io/crates/v/ruv-swarm-github.svg)](https://crates.io/crates/ruv-swarm-github)
[![Documentation](https://docs.rs/ruv-swarm-github/badge.svg)](https://docs.rs/ruv-swarm-github)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Production-ready GitHub API integration for RUV Swarm coordinator progress reporting.

## Features

- **Real GitHub API Integration**: Uses octocrab for authenticated API calls
- **Rate Limiting**: Respects GitHub API rate limits with intelligent backoff
- **Error Handling**: Comprehensive error handling for network and API failures
- **Progress Reporting**: Formats and posts detailed swarm status updates
- **Authentication**: Secure token-based authentication with validation
- **Retry Logic**: Exponential backoff with configurable retry attempts
- **Multiple Targets**: Support for issues, pull requests, and automatic issue creation

## Quick Start

### Environment Setup

```bash
export GITHUB_TOKEN="your_github_token_here"
export GITHUB_REPO_OWNER="ruvnet"
export GITHUB_REPO_NAME="ruv-FANN"
export GITHUB_ISSUE_NUMBER="123"  # Optional: specific issue to update
```

### Basic Usage

```rust
use ruv_swarm_github::{GitHubClient, ProgressReporter, GitHubConfig};
use ruv_swarm_core::{Swarm, SwarmConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration from environment
    let config = GitHubConfig::from_env()?;
    
    // Create GitHub client
    let client = GitHubClient::new(config).await?;
    let mut reporter = ProgressReporter::new(client);
    
    // Create and configure swarm
    let swarm = Swarm::new(SwarmConfig::default());
    
    // Post initial status
    reporter.post_swarm_status(&swarm, "Swarm initialization complete").await?;
    
    Ok(())
}
```

### Manual Configuration

```rust
use ruv_swarm_github::{GitHubClient, GitHubConfig};

// Create configuration manually
let config = GitHubConfig::new("your_token", "owner", "repo")
    .with_issue(123)
    .with_retry_config(5, 1000, 30000)
    .with_rate_limit(4000)
    .with_timeout(45);

let client = GitHubClient::new(config).await?;
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GITHUB_TOKEN` | Personal access token | Yes | - |
| `GITHUB_REPO_OWNER` | Repository owner | Yes | - |
| `GITHUB_REPO_NAME` | Repository name | Yes | - |
| `GITHUB_ISSUE_NUMBER` | Issue number for updates | No | - |
| `GITHUB_PR_NUMBER` | PR number for updates | No | - |
| `GITHUB_MAX_RETRIES` | Maximum retry attempts | No | 3 |
| `GITHUB_RETRY_DELAY_MS` | Initial retry delay | No | 1000 |
| `GITHUB_TIMEOUT_SECONDS` | Request timeout | No | 30 |
| `GITHUB_RATE_LIMIT_PER_HOUR` | Rate limit | No | 5000 |

### Token Requirements

Your GitHub token needs the following permissions:
- `repo` - Full repository access (for private repos)
- `public_repo` - Public repository access (for public repos)
- `issues` - Create and update issues
- `pull_requests` - Comment on pull requests

## Features

### Swarm Status Reporting

```rust
// Generate comprehensive status report
reporter.post_swarm_status(&swarm, "Hourly status update").await?;
```

Example output:
```markdown
# 🐝 Swarm Orchestration Status Report

**Generated:** 2024-01-20 15:30:00 UTC
**Repository:** ruvnet/ruv-FANN
**Context:** Hourly status update

---

## 📊 Overview

| Metric | Value |
|--------|-------|
| Total Agents | 8 |
| Active Agents | 7 |
| Queued Tasks | 12 |
| Assigned Tasks | 15 |
| Total Connections | 28 |

## 🤖 Agent Status Breakdown

- 🟢 **Running**: 7 agents
- 🟡 **Idle**: 1 agent
- 🔴 **Error**: 0 agents

## 🏥 Health Indicators

**Overall Health:** 🟢 92.5%
```

### Performance Metrics

```rust
use ruv_swarm_core::swarm_trait::SwarmPerformanceMetrics;

let metrics = SwarmPerformanceMetrics {
    tasks_completed: 1250,
    tasks_failed: 23,
    avg_response_time_ms: 145.7,
    success_rate: 0.982,
    // ...
};

reporter.post_performance_metrics(&metrics, "Daily performance summary").await?;
```

### Error Analysis

```rust
use ruv_swarm_core::swarm_trait::SwarmErrorStatistics;

let errors = SwarmErrorStatistics {
    total_errors: 45,
    critical_errors: 2,
    warnings: 43,
    error_rate: 0.036,
    // ...
};

reporter.post_error_statistics(&errors, "Error analysis report").await?;
```

## Advanced Features

### Rate Limiting

Automatic rate limiting with exponential backoff:

```rust
let config = GitHubConfig::new("token", "owner", "repo")
    .with_rate_limit(3000) // 3000 requests per hour
    .with_retry_config(5, 2000, 60000); // 5 retries, 2s initial, 60s max

let client = GitHubClient::new(config).await?;

// Check if we can make immediate request
if client.can_make_request_immediately() {
    // Make request
} else {
    let wait_time = client.estimated_wait_time();
    println!("Need to wait {:?} before next request", wait_time);
}
```

### Error Handling

Comprehensive error types with retryability detection:

```rust
use ruv_swarm_github::GitHubError;

match client.post_issue_comment(123, "Update").await {
    Ok(_) => println!("Success!"),
    Err(GitHubError::RateLimitExceeded { reset_time }) => {
        println!("Rate limited, reset at: {}", reset_time);
    }
    Err(GitHubError::AuthenticationError { message }) => {
        println!("Auth failed: {}", message);
    }
    Err(e) if e.is_retryable() => {
        println!("Retryable error: {}", e);
    }
    Err(e) => {
        println!("Permanent error: {}", e);
    }
}
```

### Multiple Targets

Support for different update targets:

```rust
// Update specific issue
let config = GitHubConfig::new("token", "owner", "repo").with_issue(123);

// Update specific PR
let config = GitHubConfig::new("token", "owner", "repo").with_pr(456);

// Auto-create issue if no target specified
let config = GitHubConfig::new("token", "owner", "repo");
```

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

Requires real GitHub token:

```bash
export GITHUB_TOKEN="your_token"
export GITHUB_REPO_OWNER="your_username"
export GITHUB_REPO_NAME="test_repo"

cargo test --ignored
```

## Error Recovery

The client includes sophisticated error recovery:

- **Network Errors**: Automatic retry with exponential backoff
- **Rate Limiting**: Intelligent waiting with proper reset time handling
- **API Errors**: Differentiation between retryable and permanent errors
- **Timeout Handling**: Configurable timeouts with proper cleanup

## Security

- **Token Security**: Tokens are never logged or exposed in error messages
- **SSL Verification**: Enabled by default, can be disabled for testing
- **Request Validation**: All inputs are validated before API calls
- **Rate Limiting**: Prevents accidental API abuse

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.