//! DAA (Decentralized Autonomous Agents) Test Module
//!
//! This module contains integration tests for DAA functionality with GPU acceleration,
//! coordination performance, and framework validation.

pub mod coordination_tests;
pub mod daa_framework_tests;
pub mod gpu_acceleration_tests;
pub mod system_performance_tests;
pub mod regression_tests;

pub use coordination_tests::*;
pub use daa_framework_tests::*;
pub use gpu_acceleration_tests::*;
pub use system_performance_tests::*;
pub use regression_tests::*;