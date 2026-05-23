//! Configuration types and management system

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};
use chrono::Duration;
use num_traits::{Float, NumCast};

use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

/// Time series frequency enumeration matching NeuralForecast Python API
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frequency {
    // High frequency
    #[serde(rename = "ns")]
    Nanosecond,
    #[serde(rename = "us")]
    Microsecond,
    #[serde(rename = "ms")]
    Millisecond,
    #[serde(rename = "s")]
    Second,
    #[serde(rename = "min")]
    Minute,
    
    // Common frequencies (NeuralForecast compatible)
    #[serde(rename = "H")]
    Hourly,
    #[serde(rename = "D")]
    Daily,
    #[serde(rename = "W")]
    Weekly,
    #[serde(rename = "M")]
    Monthly,
    #[serde(rename = "Q")]
    Quarterly,
    #[serde(rename = "Y")]
    Yearly,
    
    // Business frequencies
    #[serde(rename = "B")]
    BusinessDaily,
    #[serde(rename = "BM")]
    BusinessMonthly,
    #[serde(rename = "BQ")]
    BusinessQuarterly,
    
    // Custom frequency with period
    #[serde(rename = "custom")]
    Custom(String),
}

impl Frequency {
    /// Parse frequency from string (pandas/NeuralForecast compatible)
    pub fn from_str(s: &str) -> NeuroDivergentResult<Self> {
        match s.to_uppercase().as_str() {
            "NS" => Ok(Self::Nanosecond),
            "US" => Ok(Self::Microsecond),
            "MS" => Ok(Self::Millisecond),
            "S" => Ok(Self::Second),
            "MIN" | "T" => Ok(Self::Minute),
            "H" => Ok(Self::Hourly),
            "D" => Ok(Self::Daily),
            "W" => Ok(Self::Weekly),
            "M" => Ok(Self::Monthly),
            "Q" => Ok(Self::Quarterly),
            "Y" | "A" => Ok(Self::Yearly),
            "B" => Ok(Self::BusinessDaily),
            "BM" => Ok(Self::BusinessMonthly),
            "BQ" => Ok(Self::BusinessQuarterly),
            _ => Ok(Self::Custom(s.to_string())),
        }
    }
    
    /// Convert to pandas-compatible frequency string
    pub fn to_pandas_str(&self) -> &str {
        match self {
            Self::Nanosecond => "ns",
            Self::Microsecond => "us",
            Self::Millisecond => "ms",
            Self::Second => "s",
            Self::Minute => "min",
            Self::Hourly => "H",
            Self::Daily => "D",
            Self::Weekly => "W",
            Self::Monthly => "M",
            Self::Quarterly => "Q",
            Self::Yearly => "Y",
            Self::BusinessDaily => "B",
            Self::BusinessMonthly => "BM",
            Self::BusinessQuarterly => "BQ",
            Self::Custom(s) => s,
        }
    }
    
    /// Get the approximate duration between periods
    pub fn duration(&self) -> Duration {
        match self {
            Self::Nanosecond => Duration::nanoseconds(1),
            Self::Microsecond => Duration::microseconds(1),
            Self::Millisecond => Duration::milliseconds(1),
            Self::Second => Duration::seconds(1),
            Self::Minute => Duration::minutes(1),
            Self::Hourly => Duration::hours(1),
            Self::Daily | Self::BusinessDaily => Duration::days(1),
            Self::Weekly => Duration::weeks(1),
            Self::Monthly | Self::BusinessMonthly => Duration::days(30), // Approximate
            Self::Quarterly | Self::BusinessQuarterly => Duration::days(90), // Approximate
            Self::Yearly => Duration::days(365), // Approximate
            Self::Custom(_) => Duration::days(1), // Default fallback
        }
    }
    
    /// Check if frequency is business-related
    pub fn is_business(&self) -> bool {
        matches!(
            self, 
            Self::BusinessDaily | Self::BusinessMonthly | Self::BusinessQuarterly
        )
    }
    
    /// Get seasonal period length for the frequency
    pub fn seasonal_period(&self) -> Option<usize> {
        match self {
            Self::Hourly => Some(24),           // Daily seasonality
            Self::Daily => Some(7),             // Weekly seasonality
            Self::Weekly => Some(52),           // Yearly seasonality
            Self::Monthly => Some(12),          // Yearly seasonality
            Self::Quarterly => Some(4),         // Yearly seasonality
            _ => None,
        }
    }
}

impl fmt::Display for Frequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_pandas_str())
    }
}

/// Scaler types for data preprocessing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalerType {
    /// Standard scaler (zero mean, unit variance)
    #[serde(rename = "standard")]
    StandardScaler,
    /// Min-max scaler (scale to [0, 1])
    #[serde(rename = "minmax")]
    MinMaxScaler,
    /// Robust scaler (median and IQR)
    #[serde(rename = "robust")]
    RobustScaler,
    /// Identity scaler (no scaling)
    #[serde(rename = "identity")]
    IdentityScaler,
}

impl ScalerType {
    /// Get all available scaler types
    pub fn all() -> &'static [ScalerType] {
        &[
            Self::StandardScaler,
            Self::MinMaxScaler, 
            Self::RobustScaler,
            Self::IdentityScaler,
        ]
    }
}

impl fmt::Display for ScalerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::StandardScaler => "standard",
            Self::MinMaxScaler => "minmax",
            Self::RobustScaler => "robust",
            Self::IdentityScaler => "identity",
        };
        write!(f, "{}", name)
    }
}

/// Prediction intervals configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionIntervals {
    /// Confidence levels (e.g., [0.8, 0.9, 0.95])
    pub confidence_levels: Vec<f64>,
    /// Method for computing intervals
    pub method: IntervalMethod,
}

impl PredictionIntervals {
    /// Create new prediction intervals configuration
    pub fn new(confidence_levels: Vec<f64>, method: IntervalMethod) -> NeuroDivergentResult<Self> {
        // Validate confidence levels
        for &level in &confidence_levels {
            if level <= 0.0 || level >= 1.0 {
                return Err(NeuroDivergentError::config(
                    format!("Confidence level must be in (0, 1), got {}", level)
                ));
            }
        }
        
        Ok(Self {
            confidence_levels,
            method,
        })
    }
    
    /// Default prediction intervals (80%, 90%, 95%)
    pub fn default() -> Self {
        Self {
            confidence_levels: vec![0.8, 0.9, 0.95],
            method: IntervalMethod::Quantile,
        }
    }
    
    /// Get quantile levels from confidence levels
    pub fn quantile_levels(&self) -> Vec<f64> {
        let mut quantiles = Vec::new();
        for &level in &self.confidence_levels {
            let alpha = 1.0 - level;
            quantiles.push(alpha / 2.0);      // Lower quantile
            quantiles.push(1.0 - alpha / 2.0); // Upper quantile
        }
        quantiles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        quantiles
    }
}

/// Methods for computing prediction intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntervalMethod {
    /// Quantile regression
    #[serde(rename = "quantile")]
    Quantile,
    /// Conformal prediction
    #[serde(rename = "conformal")]
    ConformalPrediction,
    /// Bootstrap sampling
    #[serde(rename = "bootstrap")]
    Bootstrap,
}

/// Loss functions available for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error
    #[serde(rename = "mse")]
    MSE,
    /// Mean Absolute Error  
    #[serde(rename = "mae")]
    MAE,
    /// Mean Absolute Percentage Error
    #[serde(rename = "mape")]
    MAPE,
    /// Symmetric Mean Absolute Percentage Error
    #[serde(rename = "smape")]
    SMAPE,
    /// Huber Loss (robust to outliers)
    #[serde(rename = "huber")]
    Huber,
    /// Quantile Loss
    #[serde(rename = "quantile")]
    Quantile,
}

impl LossFunction {
    /// Get all available loss functions
    pub fn all() -> &'static [LossFunction] {
        &[
            Self::MSE,
            Self::MAE,
            Self::MAPE,
            Self::SMAPE,
            Self::Huber,
            Self::Quantile,
        ]
    }
    
    /// Check if loss function supports probabilistic outputs
    pub fn supports_probabilistic(&self) -> bool {
        matches!(self, Self::Quantile)
    }
}

impl fmt::Display for LossFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::MSE => "mse",
            Self::MAE => "mae", 
            Self::MAPE => "mape",
            Self::SMAPE => "smape",
            Self::Huber => "huber",
            Self::Quantile => "quantile",
        };
        write!(f, "{}", name)
    }
}

/// Optimizer types for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    #[serde(rename = "sgd")]
    SGD,
    /// Adam optimizer
    #[serde(rename = "adam")]
    Adam,
    /// AdamW optimizer (Adam with weight decay)
    #[serde(rename = "adamw")]
    AdamW,
    /// RMSprop optimizer
    #[serde(rename = "rmsprop")]
    RMSprop,
    /// Adagrad optimizer
    #[serde(rename = "adagrad")]
    Adagrad,
}

impl OptimizerType {
    /// Get all available optimizer types
    pub fn all() -> &'static [OptimizerType] {
        &[
            Self::SGD,
            Self::Adam,
            Self::AdamW,
            Self::RMSprop,
            Self::Adagrad,
        ]
    }
}

impl fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SGD => "sgd",
            Self::Adam => "adam",
            Self::AdamW => "adamw", 
            Self::RMSprop => "rmsprop",
            Self::Adagrad => "adagrad",
        };
        write!(f, "{}", name)
    }
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig<T: Float> {
    /// Type of scheduler
    pub scheduler_type: SchedulerType,
    /// Step size for StepLR
    pub step_size: Option<usize>,
    /// Multiplicative factor
    pub gamma: Option<T>,
    /// Milestones for MultiStepLR
    pub milestones: Option<Vec<usize>>,
    /// Patience for ReduceLROnPlateau
    pub patience: Option<usize>,
    /// Factor for reduction
    pub factor: Option<T>,
}

/// Types of learning rate schedulers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Step learning rate decay
    #[serde(rename = "step")]
    StepLR,
    /// Multi-step learning rate decay
    #[serde(rename = "multistep")]
    MultiStepLR,
    /// Exponential learning rate decay
    #[serde(rename = "exponential")]
    ExponentialLR,
    /// Reduce on plateau
    #[serde(rename = "plateau")]
    ReduceLROnPlateau,
    /// Cosine annealing
    #[serde(rename = "cosine")]
    CosineAnnealingLR,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig<T: Float> {
    /// Metric to monitor
    pub monitor: String,
    /// Number of epochs with no improvement to wait
    pub patience: usize,
    /// Minimum change to qualify as improvement
    pub min_delta: T,
    /// Mode for comparison
    pub mode: EarlyStoppingMode,
}

impl<T: Float> EarlyStoppingConfig<T> {
    /// Create new early stopping configuration
    pub fn new(monitor: String, patience: usize, min_delta: T, mode: EarlyStoppingMode) -> Self {
        Self {
            monitor,
            patience,
            min_delta,
            mode,
        }
    }
    
    /// Default early stopping configuration
    pub fn default() -> Self 
    where
        T: From<f32>,
    {
        Self {
            monitor: "val_loss".to_string(),
            patience: 10,
            min_delta: NumCast::from(0.001f64).unwrap(),
            mode: EarlyStoppingMode::Min,
        }
    }
}

/// Early stopping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EarlyStoppingMode {
    /// Monitor for minimum (e.g., loss)
    #[serde(rename = "min")]
    Min,
    /// Monitor for maximum (e.g., accuracy)
    #[serde(rename = "max")]
    Max,
}

/// Device specification for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// CPU computation
    #[serde(rename = "cpu")]
    CPU,
    /// GPU computation with device index
    #[serde(rename = "gpu")]
    GPU(usize),
}

impl Default for Device {
    fn default() -> Self {
        Self::CPU
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CPU => write!(f, "cpu"),
            Self::GPU(id) => write!(f, "gpu:{}", id),
        }
    }
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of cross-validation windows
    pub n_windows: usize,
    /// Forecast horizon for each window
    pub h: usize,
    /// Step size between windows
    pub step_size: Option<usize>,
    /// Size of test set
    pub test_size: Option<usize>,
    /// Seasonal length for time series split
    pub season_length: Option<usize>,
    /// Whether to refit models for each window
    pub refit: bool,
}

impl CrossValidationConfig {
    /// Create new cross-validation configuration
    pub fn new(n_windows: usize, h: usize) -> Self {
        Self {
            n_windows,
            h,
            step_size: None,
            test_size: None,
            season_length: None,
            refit: true,
        }
    }
    
    /// Set step size between windows
    pub fn with_step_size(mut self, step_size: usize) -> Self {
        self.step_size = Some(step_size);
        self
    }
    
    /// Set test set size
    pub fn with_test_size(mut self, test_size: usize) -> Self {
        self.test_size = Some(test_size);
        self
    }
    
    /// Set seasonal length
    pub fn with_season_length(mut self, season_length: usize) -> Self {
        self.season_length = Some(season_length);
        self
    }
    
    /// Set whether to refit models
    pub fn with_refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }
    
    /// Validate configuration
    pub fn validate(&self) -> NeuroDivergentResult<()> {
        if self.n_windows == 0 {
            return Err(NeuroDivergentError::config("n_windows must be greater than 0"));
        }
        if self.h == 0 {
            return Err(NeuroDivergentError::config("h must be greater than 0"));
        }
        if let Some(step_size) = self.step_size {
            if step_size == 0 {
                return Err(NeuroDivergentError::config("step_size must be greater than 0"));
            }
        }
        Ok(())
    }
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_windows: 3,
            h: 1,
            step_size: None,
            test_size: None,
            season_length: None,
            refit: true,
        }
    }
}

/// Generic configuration value type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue<T: Float> {
    /// Floating point value
    Float(T),
    /// Integer value
    Int(i64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// Vector of floats
    FloatVec(Vec<T>),
    /// Vector of integers
    IntVec(Vec<i64>),
    /// Vector of strings
    StringVec(Vec<String>),
}

impl<T: Float> ConfigValue<T> {
    /// Try to get as float
    pub fn as_float(&self) -> Option<T> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => T::from(*i),
            _ => None,
        }
    }
    
    /// Try to get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Float(f) => f.to_i64(),
            _ => None,
        }
    }
    
    /// Try to get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
    
    /// Try to get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Generic model configuration container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericModelConfig<T: Float> {
    /// Model type identifier
    pub model_type: String,
    /// Forecast horizon
    pub horizon: usize,
    /// Input window size
    pub input_size: usize,
    /// Configuration parameters
    pub parameters: HashMap<String, ConfigValue<T>>,
}

impl<T: Float> GenericModelConfig<T> {
    /// Create new generic configuration
    pub fn new(model_type: String, horizon: usize, input_size: usize) -> Self {
        Self {
            model_type,
            horizon,
            input_size,
            parameters: HashMap::new(),
        }
    }
    
    /// Set a parameter value
    pub fn set_parameter<K: Into<String>>(
        &mut self, 
        key: K, 
        value: ConfigValue<T>
    ) -> &mut Self {
        self.parameters.insert(key.into(), value);
        self
    }
    
    /// Get a parameter value
    pub fn get_parameter(&self, key: &str) -> Option<&ConfigValue<T>> {
        self.parameters.get(key)
    }
    
    /// Get all parameter keys
    pub fn parameter_keys(&self) -> Vec<&String> {
        self.parameters.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_frequency_parsing() {
        assert_eq!(Frequency::from_str("D").unwrap(), Frequency::Daily);
        assert_eq!(Frequency::from_str("H").unwrap(), Frequency::Hourly);
        assert_eq!(Frequency::from_str("M").unwrap(), Frequency::Monthly);
        assert_eq!(Frequency::from_str("custom").unwrap(), Frequency::Custom("custom".to_string()));
    }
    
    #[test] 
    fn test_frequency_display() {
        assert_eq!(Frequency::Daily.to_string(), "D");
        assert_eq!(Frequency::Hourly.to_string(), "H");
        assert_eq!(Frequency::Monthly.to_string(), "M");
    }
    
    #[test]
    fn test_prediction_intervals() {
        let intervals = PredictionIntervals::new(
            vec![0.8, 0.9], 
            IntervalMethod::Quantile
        ).unwrap();
        
        let quantiles = intervals.quantile_levels();
        assert_eq!(quantiles, vec![0.05, 0.1, 0.9, 0.95]);
    }
    
    #[test]
    fn test_cross_validation_config() {
        let config = CrossValidationConfig::new(3, 12)
            .with_step_size(1)
            .with_refit(false);
            
        assert_eq!(config.n_windows, 3);
        assert_eq!(config.h, 12);
        assert_eq!(config.step_size, Some(1));
        assert!(!config.refit);
        
        config.validate().unwrap();
    }
    
    #[test]
    fn test_config_value() {
        let val = ConfigValue::<f32>::Float(3.14);
        assert_eq!(val.as_float(), Some(3.14));
        
        let val = ConfigValue::<f32>::Int(42);
        assert_eq!(val.as_int(), Some(42));
        assert_eq!(val.as_float(), Some(42.0));
    }
}