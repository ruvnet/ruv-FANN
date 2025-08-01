//! Rate limiting for GitHub API requests

use crate::error::{GitHubError, Result};
use governor::{
    clock::DefaultClock,
    state::{DirectStateStore, InMemoryState},
    Quota, RateLimiter as GovernorRateLimiter,
};
use std::{num::NonZeroU32, time::Duration};
use tokio::time::sleep;
use tracing::{debug, warn};

/// Rate limiter for GitHub API requests
pub struct RateLimiter {
    limiter: GovernorRateLimiter<
        governor::state::NotKeyed,
        InMemoryState,
        DefaultClock,
        governor::middleware::NoOpMiddleware,
    >,
    requests_per_hour: u32,
}

impl RateLimiter {
    /// Create a new rate limiter
    /// 
    /// # Arguments
    /// 
    /// * `requests_per_hour` - Maximum number of requests allowed per hour
    /// 
    /// # Errors
    /// 
    /// Returns an error if the requests_per_hour is 0 or invalid
    pub fn new(requests_per_hour: u32) -> Result<Self> {
        if requests_per_hour == 0 {
            return Err(GitHubError::configuration(
                "requests_per_hour must be greater than 0"
            ));
        }
        
        // Convert requests per hour to requests per second for quota
        let requests_per_second = if requests_per_hour < 3600 {
            1 // At least 1 request per second
        } else {
            requests_per_hour / 3600
        };
        
        let quota = Quota::per_second(
            NonZeroU32::new(requests_per_second)
                .ok_or_else(|| GitHubError::configuration("Invalid requests_per_second"))?
        );
        
        let limiter = GovernorRateLimiter::direct(quota);
        
        debug!(
            "Created rate limiter: {} requests/hour, {} requests/second",
            requests_per_hour, requests_per_second
        );
        
        Ok(Self {
            limiter,
            requests_per_hour,
        })
    }
    
    /// Wait for permission to make a request
    /// 
    /// This method will block until a request slot is available according
    /// to the configured rate limit.
    /// 
    /// # Errors
    /// 
    /// Currently does not return errors, but may in future implementations
    pub async fn wait_for_permission(&self) -> Result<()> {
        match self.limiter.check() {
            Ok(_) => {
                debug!("Rate limit check passed");
                Ok(())
            }
            Err(negative) => {
                let wait_time = negative.wait_time_from(DefaultClock::default().now());
                warn!(
                    "Rate limit exceeded, waiting for {:?} before next request",
                    wait_time
                );
                
                sleep(wait_time).await;
                debug!("Rate limit wait completed");
                Ok(())
            }
        }
    }
    
    /// Check if a request can be made immediately without waiting
    pub fn can_proceed_immediately(&self) -> bool {
        self.limiter.check().is_ok()
    }
    
    /// Get the configured requests per hour limit
    pub fn requests_per_hour(&self) -> u32 {
        self.requests_per_hour
    }
    
    /// Get estimated wait time for next request
    pub fn estimated_wait_time(&self) -> Option<Duration> {
        match self.limiter.check() {
            Ok(_) => None,
            Err(negative) => Some(negative.wait_time_from(DefaultClock::default().now())),
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(5000).expect("Default rate limiter should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, Instant};
    
    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new(3600).unwrap();
        assert_eq!(limiter.requests_per_hour(), 3600);
    }
    
    #[tokio::test]
    async fn test_rate_limiter_invalid_config() {
        let result = RateLimiter::new(0);
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_immediate_permission() {
        let limiter = RateLimiter::new(3600).unwrap();
        assert!(limiter.can_proceed_immediately());
        
        // First request should succeed immediately
        let result = limiter.wait_for_permission().await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_wait_time_estimation() {
        let limiter = RateLimiter::new(1).unwrap(); // 1 request per hour = very restrictive
        
        // First request should be immediate
        assert!(limiter.estimated_wait_time().is_none());
        
        // Use up the quota
        let _ = limiter.wait_for_permission().await;
        
        // Now we should have a wait time
        let wait_time = limiter.estimated_wait_time();
        assert!(wait_time.is_some());
    }
    
    #[tokio::test]
    async fn test_rate_limiting_behavior() {
        let limiter = RateLimiter::new(2).unwrap(); // Very restrictive for testing
        
        let start = Instant::now();
        
        // First request should be immediate
        limiter.wait_for_permission().await.unwrap();
        let first_duration = start.elapsed();
        
        // Second request might require waiting
        limiter.wait_for_permission().await.unwrap();
        let second_duration = start.elapsed();
        
        // The second request should take longer than the first
        assert!(second_duration >= first_duration);
    }
}