//! Unit tests for swarm core
//! 
//! This module contains meaningful tests for the core functionality of the swarm system.
//! Tests focus on critical paths and core behaviors rather than edge cases.

use crate::{Metadata, VERSION};

#[test]
fn test_version_info() {
    assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
}

#[test]
fn test_metadata() {
    assert_eq!(Metadata::name(), "ruv-swarm-core");
    assert_eq!(Metadata::version(), VERSION);
    assert!(!Metadata::description().is_empty());
}

// Core unit tests for each module
mod agent_tests;
mod task_tests;
mod topology_tests;
mod error_handling_tests;

// Additional test coverage
mod agent_message_tests;
mod custom_payload_tests;
mod agent_trait_tests;

// Swarm tests
mod swarm_tests;
mod async_swarm_tests;
mod swarm_trait_tests;

// Integration tests
mod swarm_integration_tests;