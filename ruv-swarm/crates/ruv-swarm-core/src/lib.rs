//! Core orchestration and agent traits for RUV Swarm
//!
//! This crate provides the foundational building blocks for creating
//! distributed AI agent swarms with cognitive diversity patterns.
//!
//! # Features
//!
//! - **Agent Trait**: Core abstraction for all swarm agents
//! - **Task Management**: Distributed task queue with priority scheduling
//! - **Swarm Orchestration**: Multiple topology and distribution strategies
//! - **Cognitive Patterns**: Support for diverse thinking patterns
//! - **No-std Support**: Can run in embedded environments
//!
//! # Example
//!
//! ```rust,no_run
//! use ruv_swarm_core::{Agent, Swarm, SwarmConfig, Task, Priority};
//! use async_trait::async_trait;
//!
//! // Define a simple agent
//! struct ComputeAgent {
//!     id: String,
//!     capabilities: Vec<String>,
//! }
//!
//! #[async_trait]
//! impl Agent for ComputeAgent {
//!     type Input = f64;
//!     type Output = f64;
//!     type Error = std::io::Error;
//!
//!     async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
//!         Ok(input * 2.0)
//!     }
//!
//!     fn capabilities(&self) -> &[String] {
//!         &self.capabilities
//!     }
//!
//!     fn id(&self) -> &str {
//!         &self.id
//!     }
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod agent;
pub mod error;
pub mod swarm;
pub mod task;
pub mod topology;
pub mod gpu_task_types;

// Re-export commonly used types
pub use agent::{
    Agent, AgentMessage, AgentMetadata, AgentMetrics, BoxedAgent,
    CognitivePattern, ErasedAgent, HealthStatus, MessageType, ResourceRequirements,
};

pub use error::{Result, SwarmError};

#[cfg(feature = "std")]
pub use swarm::{
    Swarm, SwarmConfig, SwarmMetrics,
};

pub use topology::{Topology, TopologyType};

pub use task::{
    DistributionStrategy, TaskPriority as Priority, Task, TaskId,
    TaskResult, TaskStatus,
};

pub use gpu_task_types::{
    GPUTaskType, GPUTaskOrchestrator, GPUTask, GPUTaskResult,
    CoordinationPattern, ComputeBackend, TrainingAlgorithm, OptimizationTarget,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::agent::{Agent, CognitivePattern};
    pub use crate::error::Result;
    #[cfg(feature = "std")]
    pub use crate::swarm::{Swarm, SwarmConfig};
    pub use crate::task::{TaskPriority as Priority, Task, TaskId};
    pub use crate::gpu_task_types::{
        GPUTaskType, GPUTaskOrchestrator, CoordinationPattern, ComputeBackend,
    };
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library metadata
pub struct Metadata;

impl Metadata {
    /// Get the library name
    pub fn name() -> &'static str {
        env!("CARGO_PKG_NAME")
    }
    
    /// Get the library version
    pub fn version() -> &'static str {
        VERSION
    }
    
    /// Get the library description
    pub fn description() -> &'static str {
        env!("CARGO_PKG_DESCRIPTION")
    }
}