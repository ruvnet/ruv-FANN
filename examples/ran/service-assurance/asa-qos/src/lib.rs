//! ASA-QOS-01 - Predictive VoLTE Jitter Forecaster
//! 
//! This module implements advanced VoLTE jitter prediction using machine learning
//! and time-series forecasting techniques. It provides:
//! - Real-time jitter forecasting with 10ms accuracy
//! - Quality degradation alerting
//! - Voice quality optimization recommendations
//! - Service assurance dashboard data

pub mod config;
pub mod error;
pub mod forecasting;
pub mod metrics;
pub mod models;
pub mod proto;
pub mod service;
pub mod storage;
pub mod types;
pub mod utils;

pub use config::Config;
pub use error::{Error, Result};
pub use forecasting::JitterForecaster;
pub use service::QosService;
pub use types::*;