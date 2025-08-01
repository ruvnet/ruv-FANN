//! Machine learning integration for RUV Swarm
//!
//! This crate provides neural forecasting capabilities for agent-specific time series prediction,
//! ensemble methods, and swarm-level forecasting coordination.

#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

#[cfg(target_arch = "wasm32")]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(target_arch = "wasm32")]
use alloc::collections::BTreeMap as HashMap;

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use std::format;

#[cfg(not(target_arch = "wasm32"))]
use std::vec;

pub mod agent_forecasting;
pub mod ensemble;
pub mod models;
pub mod time_series;
pub mod integration_example;

#[cfg(target_arch = "wasm32")]
pub mod wasm_bindings;

// Re-export main types
pub use agent_forecasting::{
    AgentForecastContext, AgentForecastingManager, ForecastDomain, ForecastRequirements,
    ModelSpecialization,
};

pub use ensemble::{EnsembleConfig, EnsembleForecaster, EnsembleStrategy, OptimizationMetric};

pub use models::{ForecastModel, ModelFactory, ModelType};

pub use time_series::{TimeSeriesData, TimeSeriesProcessor, TransformationType};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindings::*;
