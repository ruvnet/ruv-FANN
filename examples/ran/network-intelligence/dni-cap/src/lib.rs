//! DNI-CAP-01: Capacity Cliff Forecaster
//!
//! This module provides advanced capacity planning and forecasting capabilities for RAN networks.
//! It predicts when PRB (Physical Resource Block) utilization will reach critical thresholds
//! (typically 80%) with ±2 months accuracy.
//!
//! ## Key Features
//!
//! - **Long-term Forecasting**: Predicts capacity breaches 6-24 months in advance
//! - **Growth Trend Analysis**: Identifies usage patterns and growth trajectories
//! - **Capacity Cliff Detection**: Warns of rapid utilization increases
//! - **Strategic Planning**: Provides network expansion recommendations
//! - **Investment Optimization**: Prioritizes capacity investments based on predicted needs
//!
//! ## Architecture
//!
//! The system uses multiple forecasting models in ensemble:
//! - LSTM networks for sequential pattern recognition
//! - ARIMA models for trend decomposition
//! - Polynomial regression for long-term trend extrapolation
//! - Exponential smoothing for seasonal adjustments
//!
//! ## Usage
//!
//! ```rust
//! use dni_cap_01::forecasting::{CapacityForecaster, ForecastConfig};
//! use dni_cap_01::models::CapacityCliffPredictor;
//!
//! // Initialize forecaster
//! let config = ForecastConfig::default();
//! let mut forecaster = CapacityForecaster::new(config);
//!
//! // Train on historical data
//! forecaster.train(historical_data)?;
//!
//! // Predict capacity breach
//! let prediction = forecaster.predict_capacity_breach(0.8)?;
//! println!("80% utilization breach predicted in {} months", prediction.months_ahead);
//! ```

pub mod config;
pub mod error;
pub mod forecasting;
pub mod models;
pub mod monitoring;
pub mod planning;
pub mod service;
pub mod types;
pub mod utils;

// Re-export main public API
pub use config::*;
pub use error::*;
pub use forecasting::*;
pub use models::*;
pub use planning::*;
pub use types::*;

/// Current version of the DNI-CAP-01 system
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Target forecast accuracy (±2 months)
pub const TARGET_ACCURACY_MONTHS: f64 = 2.0;

/// Default capacity threshold for breach detection
pub const DEFAULT_CAPACITY_THRESHOLD: f64 = 0.8;

/// Minimum historical data points required for reliable forecasting
pub const MIN_HISTORICAL_POINTS: usize = 12; // 12 months

/// Maximum forecast horizon in months
pub const MAX_FORECAST_HORIZON: usize = 24;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(TARGET_ACCURACY_MONTHS, 2.0);
        assert_eq!(DEFAULT_CAPACITY_THRESHOLD, 0.8);
        assert_eq!(MIN_HISTORICAL_POINTS, 12);
        assert_eq!(MAX_FORECAST_HORIZON, 24);
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}