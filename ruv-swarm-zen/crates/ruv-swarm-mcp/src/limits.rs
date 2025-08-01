//! Resource limits and quotas for security
//!
//! This module implements resource limits to prevent DoS attacks
//! and resource exhaustion vulnerabilities.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Resource limit configuration
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum number of agents per session
    pub max_agents_per_session: usize,
    /// Maximum memory size per session (in bytes)
    pub max_memory_per_session: usize,
    /// Maximum concurrent tasks per session
    pub max_concurrent_tasks: usize,
    /// Maximum session duration
    pub max_session_duration: Duration,
    /// Maximum WebSocket message size (in bytes)
    pub max_message_size: usize,
    /// Maximum workflow execution time
    pub max_workflow_duration: Duration,
    /// Maximum memory entries per session
    pub max_memory_entries: usize,
    /// Maximum size per memory value (in bytes)
    pub max_memory_value_size: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_agents_per_session: 10_000_000,                     // 10M agents for massive GPU swarms
            max_memory_per_session: 1024 * 1024 * 1024 * 1024,     // 1TB memory
            max_concurrent_tasks: 10_000_000,                       // 10M concurrent tasks
            max_session_duration: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            max_message_size: 1024 * 1024 * 1024,                  // 1GB messages
            max_workflow_duration: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            max_memory_entries: 100_000_000,                        // 100M entries
            max_memory_value_size: 1024 * 1024 * 1024,             // 1GB per value
        }
    }
}

/// Session resource usage tracking
#[derive(Debug, Clone)]
pub struct SessionUsage {
    pub session_id: Uuid,
    pub agent_count: usize,
    pub memory_usage: usize,
    pub active_tasks: usize,
    pub memory_entries: usize,
    pub created_at: Instant,
    pub last_activity: Instant,
}

impl SessionUsage {
    pub fn new(session_id: Uuid) -> Self {
        let now = Instant::now();
        Self {
            session_id,
            agent_count: 0,
            memory_usage: 0,
            active_tasks: 0,
            memory_entries: 0,
            created_at: now,
            last_activity: now,
        }
    }

    /// Update last activity timestamp
    pub fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Check if session has expired
    pub fn is_expired(&self, max_duration: Duration) -> bool {
        self.created_at.elapsed() > max_duration
    }

    /// Get session age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last activity
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed()
    }
}

/// Resource limiter for enforcing quotas
pub struct ResourceLimiter {
    limits: ResourceLimits,
    sessions: Arc<RwLock<HashMap<Uuid, SessionUsage>>>,
}

impl ResourceLimiter {
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize session tracking
    pub async fn init_session(&self, session_id: Uuid) {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, SessionUsage::new(session_id));
    }

    /// Remove session tracking
    pub async fn remove_session(&self, session_id: &Uuid) {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
    }

    /// Check if agent spawn is allowed
    pub async fn check_agent_limit(&self, session_id: &Uuid) -> Result<(), LimitError> {
        let mut sessions = self.sessions.write().await;
        let usage = sessions
            .get_mut(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        usage.touch();

        if usage.agent_count >= self.limits.max_agents_per_session {
            return Err(LimitError::AgentLimitExceeded {
                current: usage.agent_count,
                limit: self.limits.max_agents_per_session,
            });
        }

        Ok(())
    }

    /// Increment agent count
    pub async fn increment_agents(&self, session_id: &Uuid) -> Result<(), LimitError> {
        let mut sessions = self.sessions.write().await;
        let usage = sessions
            .get_mut(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        usage.agent_count += 1;
        usage.touch();

        Ok(())
    }

    /// Check if task creation is allowed
    pub async fn check_task_limit(&self, session_id: &Uuid) -> Result<(), LimitError> {
        let mut sessions = self.sessions.write().await;
        let usage = sessions
            .get_mut(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        usage.touch();

        if usage.active_tasks >= self.limits.max_concurrent_tasks {
            return Err(LimitError::TaskLimitExceeded {
                current: usage.active_tasks,
                limit: self.limits.max_concurrent_tasks,
            });
        }

        Ok(())
    }

    /// Increment task count
    pub async fn increment_tasks(&self, session_id: &Uuid) -> Result<(), LimitError> {
        let mut sessions = self.sessions.write().await;
        let usage = sessions
            .get_mut(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        usage.active_tasks += 1;
        usage.touch();

        Ok(())
    }

    /// Decrement task count
    pub async fn decrement_tasks(&self, session_id: &Uuid) -> Result<(), LimitError> {
        let mut sessions = self.sessions.write().await;
        let usage = sessions
            .get_mut(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        if usage.active_tasks > 0 {
            usage.active_tasks -= 1;
        }
        usage.touch();

        Ok(())
    }

    /// Check memory storage limits
    pub async fn check_memory_limit(
        &self,
        session_id: &Uuid,
        value_size: usize,
    ) -> Result<(), LimitError> {
        let mut sessions = self.sessions.write().await;
        let usage = sessions
            .get_mut(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        usage.touch();

        // Check value size limit
        if value_size > self.limits.max_memory_value_size {
            return Err(LimitError::MemoryValueTooLarge {
                size: value_size,
                limit: self.limits.max_memory_value_size,
            });
        }

        // Check total memory usage
        if usage.memory_usage + value_size > self.limits.max_memory_per_session {
            return Err(LimitError::MemoryLimitExceeded {
                current: usage.memory_usage,
                requested: value_size,
                limit: self.limits.max_memory_per_session,
            });
        }

        // Check entry count
        if usage.memory_entries >= self.limits.max_memory_entries {
            return Err(LimitError::MemoryEntriesExceeded {
                current: usage.memory_entries,
                limit: self.limits.max_memory_entries,
            });
        }

        Ok(())
    }

    /// Update memory usage
    pub async fn update_memory_usage(
        &self,
        session_id: &Uuid,
        value_size: usize,
        is_add: bool,
    ) -> Result<(), LimitError> {
        let mut sessions = self.sessions.write().await;
        let usage = sessions
            .get_mut(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        if is_add {
            usage.memory_usage += value_size;
            usage.memory_entries += 1;
        } else {
            usage.memory_usage = usage.memory_usage.saturating_sub(value_size);
            if usage.memory_entries > 0 {
                usage.memory_entries -= 1;
            }
        }
        usage.touch();

        Ok(())
    }

    /// Check session expiration
    pub async fn check_session_expiration(&self, session_id: &Uuid) -> Result<(), LimitError> {
        let sessions = self.sessions.read().await;
        let usage = sessions
            .get(session_id)
            .ok_or(LimitError::SessionNotFound)?;

        if usage.is_expired(self.limits.max_session_duration) {
            return Err(LimitError::SessionExpired {
                age: usage.age(),
                limit: self.limits.max_session_duration,
            });
        }

        Ok(())
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> usize {
        let mut sessions = self.sessions.write().await;
        let expired: Vec<Uuid> = sessions
            .iter()
            .filter(|(_, usage)| usage.is_expired(self.limits.max_session_duration))
            .map(|(id, _)| *id)
            .collect();

        let count = expired.len();
        for id in expired {
            sessions.remove(&id);
        }

        count
    }

    /// Get session statistics
    pub async fn get_session_stats(&self, session_id: &Uuid) -> Option<SessionStats> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).map(|usage| SessionStats {
            agent_count: usage.agent_count,
            memory_usage: usage.memory_usage,
            active_tasks: usage.active_tasks,
            memory_entries: usage.memory_entries,
            age: usage.age(),
            idle_time: usage.idle_time(),
        })
    }

    /// Get all sessions statistics
    pub async fn get_all_stats(&self) -> Vec<(Uuid, SessionStats)> {
        let sessions = self.sessions.read().await;
        sessions
            .iter()
            .map(|(id, usage)| {
                (
                    *id,
                    SessionStats {
                        agent_count: usage.agent_count,
                        memory_usage: usage.memory_usage,
                        active_tasks: usage.active_tasks,
                        memory_entries: usage.memory_entries,
                        age: usage.age(),
                        idle_time: usage.idle_time(),
                    },
                )
            })
            .collect()
    }
}

/// Session statistics
#[derive(Debug, Clone)]
pub struct SessionStats {
    pub agent_count: usize,
    pub memory_usage: usize,
    pub active_tasks: usize,
    pub memory_entries: usize,
    pub age: Duration,
    pub idle_time: Duration,
}

/// Resource limit errors
#[derive(Debug, thiserror::Error)]
pub enum LimitError {
    #[error("Session not found")]
    SessionNotFound,

    #[error("Agent limit exceeded: {current}/{limit}")]
    AgentLimitExceeded { current: usize, limit: usize },

    #[error("Task limit exceeded: {current}/{limit}")]
    TaskLimitExceeded { current: usize, limit: usize },

    #[error("Memory limit exceeded: current {current} + requested {requested} > limit {limit}")]
    MemoryLimitExceeded {
        current: usize,
        requested: usize,
        limit: usize,
    },

    #[error("Memory value too large: {size} > limit {limit}")]
    MemoryValueTooLarge { size: usize, limit: usize },

    #[error("Memory entries exceeded: {current}/{limit}")]
    MemoryEntriesExceeded { current: usize, limit: usize },

    #[error("Session expired: age {age:?} > limit {limit:?}")]
    SessionExpired {
        age: Duration,
        limit: Duration,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_limits() {
        let mut limits = ResourceLimits::default();
        limits.max_agents_per_session = 2;
        
        let limiter = ResourceLimiter::new(limits);
        let session_id = Uuid::new_v4();
        
        limiter.init_session(session_id).await;
        
        // Should allow first two agents
        assert!(limiter.check_agent_limit(&session_id).await.is_ok());
        assert!(limiter.increment_agents(&session_id).await.is_ok());
        
        assert!(limiter.check_agent_limit(&session_id).await.is_ok());
        assert!(limiter.increment_agents(&session_id).await.is_ok());
        
        // Should reject third agent
        assert!(limiter.check_agent_limit(&session_id).await.is_err());
    }

    #[tokio::test]
    async fn test_memory_limits() {
        let mut limits = ResourceLimits::default();
        limits.max_memory_per_session = 1000;
        limits.max_memory_value_size = 500;
        
        let limiter = ResourceLimiter::new(limits);
        let session_id = Uuid::new_v4();
        
        limiter.init_session(session_id).await;
        
        // Should allow within limits
        assert!(limiter.check_memory_limit(&session_id, 400).await.is_ok());
        assert!(limiter.update_memory_usage(&session_id, 400, true).await.is_ok());
        
        // Should reject oversized value
        assert!(limiter.check_memory_limit(&session_id, 600).await.is_err());
        
        // Should reject when total exceeds limit
        assert!(limiter.check_memory_limit(&session_id, 700).await.is_err());
    }

    #[tokio::test]
    async fn test_session_expiration() {
        let mut limits = ResourceLimits::default();
        limits.max_session_duration = Duration::from_millis(100);
        
        let limiter = ResourceLimiter::new(limits);
        let session_id = Uuid::new_v4();
        
        limiter.init_session(session_id).await;
        
        // Should be valid initially
        assert!(limiter.check_session_expiration(&session_id).await.is_ok());
        
        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should be expired
        assert!(limiter.check_session_expiration(&session_id).await.is_err());
        
        // Cleanup should remove it
        let cleaned = limiter.cleanup_expired_sessions().await;
        assert_eq!(cleaned, 1);
    }
}