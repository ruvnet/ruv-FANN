use candle_core::DType;
use std::{error::Error, fmt};

/// Import errors name the affected stage without including tensor contents.
#[derive(Debug)]
pub enum CandleAdapterError {
    Invalid {
        stage: usize,
        reason: &'static str,
    },
    DType {
        stage: usize,
        tensor: &'static str,
        expected: DType,
        actual: DType,
    },
    NonCpu {
        stage: usize,
        tensor: &'static str,
    },
    NonFinite {
        stage: usize,
        tensor: &'static str,
        index: usize,
    },
    Budget {
        resource: &'static str,
        required: usize,
        limit: usize,
    },
    SizeOverflow,
    AllocationFailed,
    Candle(candle_core::Error),
    Json(serde_json::Error),
    Prepared(ruv_fann::PreparedError),
}

impl fmt::Display for CandleAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { stage, reason } => write!(f, "Invalid Candle stage {stage}: {reason}"),
            Self::DType {
                stage,
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "Candle stage {stage} {tensor} dtype is {actual:?}, expected {expected:?}"
            ),
            Self::NonCpu { stage, tensor } => {
                write!(f, "Candle stage {stage} {tensor} must already be on CPU")
            }
            Self::NonFinite {
                stage,
                tensor,
                index,
            } => write!(
                f,
                "Candle stage {stage} {tensor} has non-finite logical value {index}"
            ),
            Self::Budget {
                resource,
                required,
                limit,
            } => write!(f, "{resource} requires {required}, exceeding limit {limit}"),
            Self::SizeOverflow => f.write_str("Candle import size overflow"),
            Self::AllocationFailed => f.write_str("Candle import allocation failed"),
            Self::Candle(error) => write!(f, "Candle tensor access failed: {error}"),
            Self::Json(error) => write!(f, "Dense JSON encoding failed: {error}"),
            Self::Prepared(error) => write!(f, "FANN preparation failed: {error}"),
        }
    }
}

impl Error for CandleAdapterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Candle(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Prepared(error) => Some(error),
            _ => None,
        }
    }
}

impl From<candle_core::Error> for CandleAdapterError {
    fn from(value: candle_core::Error) -> Self {
        Self::Candle(value)
    }
}
impl From<ruv_fann::PreparedError> for CandleAdapterError {
    fn from(value: ruv_fann::PreparedError) -> Self {
        Self::Prepared(value)
    }
}
