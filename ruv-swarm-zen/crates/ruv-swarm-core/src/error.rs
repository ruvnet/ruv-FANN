//! Error types for the swarm orchestration system

use core::fmt;

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(feature = "std")]
use thiserror::Error;

/// Result type alias for swarm operations
pub type Result<T> = core::result::Result<T, SwarmError>;

/// Core error types for swarm orchestration
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Error))]
pub enum SwarmError {
    /// Agent not found in registry
    #[cfg_attr(feature = "std", error("Agent not found: {id}"))]
    AgentNotFound {
        /// The unique identifier of the agent that was not found
        id: String,
    },

    /// Task execution failed
    #[cfg_attr(feature = "std", error("Task execution failed: {reason}"))]
    TaskExecutionFailed {
        /// Description of why the task execution failed
        reason: String,
    },

    /// Invalid swarm topology
    #[cfg_attr(feature = "std", error("Invalid topology: {reason}"))]
    InvalidTopology {
        /// Description of why the topology is invalid
        reason: String,
    },

    /// Communication error between agents
    #[cfg_attr(feature = "std", error("Communication error: {reason}"))]
    CommunicationError {
        /// Description of the communication error
        reason: String,
    },

    /// Resource exhaustion
    #[cfg_attr(feature = "std", error("Resource exhausted: {resource}"))]
    ResourceExhausted {
        /// The type of resource that was exhausted (e.g., "memory", "cpu", "connections")
        resource: String,
    },

    /// Timeout occurred
    #[cfg_attr(feature = "std", error("Operation timed out after {duration_ms}ms"))]
    Timeout {
        /// The duration in milliseconds after which the operation timed out
        duration_ms: u64,
    },

    /// Agent capability mismatch
    #[cfg_attr(
        feature = "std",
        error("Agent {agent_id} lacks capability: {capability}")
    )]
    CapabilityMismatch {
        /// The unique identifier of the agent lacking the capability
        agent_id: String,
        /// The required capability that the agent lacks
        capability: String,
    },

    /// Orchestration strategy error
    #[cfg_attr(feature = "std", error("Strategy error: {reason}"))]
    StrategyError {
        /// Description of the strategy error
        reason: String,
    },

    /// Serialization/deserialization error
    #[cfg_attr(feature = "std", error("Serialization error: {reason}"))]
    SerializationError {
        /// Description of the serialization error
        reason: String,
    },

    /// System-level error
    #[cfg_attr(feature = "std", error("System error: {message}"))]
    SystemError {
        /// Description of the system error
        message: String,
    },

    /// Generic error with custom message
    #[cfg_attr(feature = "std", error("{0}"))]
    Custom(String),
}

#[cfg(not(feature = "std"))]
impl fmt::Display for SwarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwarmError::AgentNotFound { id } => write!(f, "Agent not found: {}", id),
            SwarmError::TaskExecutionFailed { reason } => {
                write!(f, "Task execution failed: {}", reason)
            }
            SwarmError::InvalidTopology { reason } => write!(f, "Invalid topology: {}", reason),
            SwarmError::CommunicationError { reason } => {
                write!(f, "Communication error: {}", reason)
            }
            SwarmError::ResourceExhausted { resource } => {
                write!(f, "Resource exhausted: {}", resource)
            }
            SwarmError::Timeout { duration_ms } => {
                write!(f, "Operation timed out after {}ms", duration_ms)
            }
            SwarmError::CapabilityMismatch {
                agent_id,
                capability,
            } => {
                write!(f, "Agent {} lacks capability: {}", agent_id, capability)
            }
            SwarmError::StrategyError { reason } => write!(f, "Strategy error: {}", reason),
            SwarmError::SerializationError { reason } => {
                write!(f, "Serialization error: {}", reason)
            }
            SwarmError::SystemError { message } => {
                write!(f, "System error: {}", message)
            }
            SwarmError::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

#[cfg(not(feature = "std"))]
impl core::error::Error for SwarmError {}

impl SwarmError {
    /// Create a custom error with a message
    pub fn custom(msg: impl Into<String>) -> Self {
        SwarmError::Custom(msg.into())
    }

    /// Check if the error is retriable
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            SwarmError::CommunicationError { .. }
                | SwarmError::Timeout { .. }
                | SwarmError::ResourceExhausted { .. }
        )
    }
}

/// Agent-specific error type
#[derive(Debug, Clone)]
pub struct AgentError {
    /// The unique identifier of the agent that encountered the error
    pub agent_id: String,
    /// The specific swarm error that occurred
    pub error: SwarmError,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent {} error: {}", self.agent_id, self.error)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AgentError {}

#[cfg(not(feature = "std"))]
impl core::error::Error for AgentError {}
