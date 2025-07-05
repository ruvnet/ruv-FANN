use thiserror::Error;

#[derive(Error, Debug)]
pub enum SliceError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Prediction error: {0}")]
    Prediction(String),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("SLA error: {0}")]
    Sla(String),
    
    #[error("Monitoring error: {0}")]
    Monitoring(String),
    
    #[error("Alerting error: {0}")]
    Alerting(String),
    
    #[error("Optimization error: {0}")]
    Optimization(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    
    #[error("Anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),
    
    #[error("Tokio task join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),
    
    #[error("Chrono parse error: {0}")]
    ChronoParse(#[from] chrono::ParseError),
    
    #[error("UUID parse error: {0}")]
    UuidParse(#[from] uuid::Error),
    
    #[error("Neural network error: {0}")]
    NeuralNetwork(String),
    
    #[error("Feature engineering error: {0}")]
    FeatureEngineering(String),
    
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    
    #[error("Slice not found: {0}")]
    SliceNotFound(String),
    
    #[error("SLA not found: {0}")]
    SlaNotFound(String),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Authentication error: {0}")]
    Authentication(String),
    
    #[error("Authorization error: {0}")]
    Authorization(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl SliceError {
    pub fn model<T: Into<String>>(msg: T) -> Self {
        SliceError::Model(msg.into())
    }
    
    pub fn prediction<T: Into<String>>(msg: T) -> Self {
        SliceError::Prediction(msg.into())
    }
    
    pub fn storage<T: Into<String>>(msg: T) -> Self {
        SliceError::Storage(msg.into())
    }
    
    pub fn validation<T: Into<String>>(msg: T) -> Self {
        SliceError::Validation(msg.into())
    }
    
    pub fn sla<T: Into<String>>(msg: T) -> Self {
        SliceError::Sla(msg.into())
    }
    
    pub fn monitoring<T: Into<String>>(msg: T) -> Self {
        SliceError::Monitoring(msg.into())
    }
    
    pub fn alerting<T: Into<String>>(msg: T) -> Self {
        SliceError::Alerting(msg.into())
    }
    
    pub fn optimization<T: Into<String>>(msg: T) -> Self {
        SliceError::Optimization(msg.into())
    }
    
    pub fn neural_network<T: Into<String>>(msg: T) -> Self {
        SliceError::NeuralNetwork(msg.into())
    }
    
    pub fn feature_engineering<T: Into<String>>(msg: T) -> Self {
        SliceError::FeatureEngineering(msg.into())
    }
    
    pub fn data_processing<T: Into<String>>(msg: T) -> Self {
        SliceError::DataProcessing(msg.into())
    }
    
    pub fn slice_not_found<T: Into<String>>(slice_id: T) -> Self {
        SliceError::SliceNotFound(slice_id.into())
    }
    
    pub fn sla_not_found<T: Into<String>>(sla_id: T) -> Self {
        SliceError::SlaNotFound(sla_id.into())
    }
    
    pub fn model_not_found<T: Into<String>>(model_id: T) -> Self {
        SliceError::ModelNotFound(model_id.into())
    }
    
    pub fn insufficient_data<T: Into<String>>(msg: T) -> Self {
        SliceError::InsufficientData(msg.into())
    }
    
    pub fn service_unavailable<T: Into<String>>(msg: T) -> Self {
        SliceError::ServiceUnavailable(msg.into())
    }
    
    pub fn timeout<T: Into<String>>(msg: T) -> Self {
        SliceError::Timeout(msg.into())
    }
    
    pub fn rate_limit_exceeded<T: Into<String>>(msg: T) -> Self {
        SliceError::RateLimitExceeded(msg.into())
    }
    
    pub fn authentication<T: Into<String>>(msg: T) -> Self {
        SliceError::Authentication(msg.into())
    }
    
    pub fn authorization<T: Into<String>>(msg: T) -> Self {
        SliceError::Authorization(msg.into())
    }
    
    pub fn internal<T: Into<String>>(msg: T) -> Self {
        SliceError::Internal(msg.into())
    }
}

impl From<SliceError> for tonic::Status {
    fn from(err: SliceError) -> Self {
        match err {
            SliceError::Configuration(msg) => tonic::Status::invalid_argument(msg),
            SliceError::Model(msg) => tonic::Status::internal(msg),
            SliceError::Prediction(msg) => tonic::Status::internal(msg),
            SliceError::Storage(msg) => tonic::Status::internal(msg),
            SliceError::Network(msg) => tonic::Status::unavailable(msg),
            SliceError::Validation(msg) => tonic::Status::invalid_argument(msg),
            SliceError::Sla(msg) => tonic::Status::invalid_argument(msg),
            SliceError::Monitoring(msg) => tonic::Status::internal(msg),
            SliceError::Alerting(msg) => tonic::Status::internal(msg),
            SliceError::Optimization(msg) => tonic::Status::internal(msg),
            SliceError::Serialization(msg) => tonic::Status::internal(msg),
            SliceError::Grpc(status) => status,
            SliceError::Io(e) => tonic::Status::internal(e.to_string()),
            SliceError::Json(e) => tonic::Status::internal(e.to_string()),
            SliceError::Bincode(e) => tonic::Status::internal(e.to_string()),
            SliceError::Anyhow(e) => tonic::Status::internal(e.to_string()),
            SliceError::TokioJoin(e) => tonic::Status::internal(e.to_string()),
            SliceError::ChronoParse(e) => tonic::Status::invalid_argument(e.to_string()),
            SliceError::UuidParse(e) => tonic::Status::invalid_argument(e.to_string()),
            SliceError::NeuralNetwork(msg) => tonic::Status::internal(msg),
            SliceError::FeatureEngineering(msg) => tonic::Status::internal(msg),
            SliceError::DataProcessing(msg) => tonic::Status::internal(msg),
            SliceError::SliceNotFound(msg) => tonic::Status::not_found(msg),
            SliceError::SlaNotFound(msg) => tonic::Status::not_found(msg),
            SliceError::ModelNotFound(msg) => tonic::Status::not_found(msg),
            SliceError::InsufficientData(msg) => tonic::Status::failed_precondition(msg),
            SliceError::ServiceUnavailable(msg) => tonic::Status::unavailable(msg),
            SliceError::Timeout(msg) => tonic::Status::deadline_exceeded(msg),
            SliceError::RateLimitExceeded(msg) => tonic::Status::resource_exhausted(msg),
            SliceError::Authentication(msg) => tonic::Status::unauthenticated(msg),
            SliceError::Authorization(msg) => tonic::Status::permission_denied(msg),
            SliceError::Internal(msg) => tonic::Status::internal(msg),
        }
    }
}

pub type SliceResult<T> = Result<T, SliceError>;

/// Error context for enhanced error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub slice_id: Option<String>,
    pub sla_id: Option<String>,
    pub model_id: Option<String>,
    pub operation: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self {
            slice_id: None,
            sla_id: None,
            model_id: None,
            operation: None,
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_slice_id<T: Into<String>>(mut self, slice_id: T) -> Self {
        self.slice_id = Some(slice_id.into());
        self
    }
    
    pub fn with_sla_id<T: Into<String>>(mut self, sla_id: T) -> Self {
        self.sla_id = Some(sla_id.into());
        self
    }
    
    pub fn with_model_id<T: Into<String>>(mut self, model_id: T) -> Self {
        self.model_id = Some(model_id.into());
        self
    }
    
    pub fn with_operation<T: Into<String>>(mut self, operation: T) -> Self {
        self.operation = Some(operation.into());
        self
    }
    
    pub fn with_info<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced error type with context
#[derive(Error, Debug)]
pub struct SliceErrorWithContext {
    #[source]
    pub error: SliceError,
    pub context: ErrorContext,
}

impl std::fmt::Display for SliceErrorWithContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)?;
        
        if let Some(slice_id) = &self.context.slice_id {
            write!(f, " (slice: {})", slice_id)?;
        }
        
        if let Some(sla_id) = &self.context.sla_id {
            write!(f, " (sla: {})", sla_id)?;
        }
        
        if let Some(model_id) = &self.context.model_id {
            write!(f, " (model: {})", model_id)?;
        }
        
        if let Some(operation) = &self.context.operation {
            write!(f, " (operation: {})", operation)?;
        }
        
        Ok(())
    }
}

impl SliceErrorWithContext {
    pub fn new(error: SliceError, context: ErrorContext) -> Self {
        Self { error, context }
    }
}

impl From<SliceErrorWithContext> for tonic::Status {
    fn from(err: SliceErrorWithContext) -> Self {
        let status: tonic::Status = err.error.into();
        // Add context to status metadata if needed
        status
    }
}

pub type SliceResultWithContext<T> = Result<T, SliceErrorWithContext>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = SliceError::prediction("Test prediction error");
        assert!(matches!(error, SliceError::Prediction(_)));
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new()
            .with_slice_id("test-slice")
            .with_operation("prediction")
            .with_info("key", "value");
        
        assert_eq!(context.slice_id, Some("test-slice".to_string()));
        assert_eq!(context.operation, Some("prediction".to_string()));
        assert_eq!(context.additional_info.get("key"), Some(&"value".to_string()));
    }
    
    #[test]
    fn test_error_with_context() {
        let error = SliceError::prediction("Test error");
        let context = ErrorContext::new().with_slice_id("test-slice");
        let error_with_context = SliceErrorWithContext::new(error, context);
        
        let error_string = error_with_context.to_string();
        assert!(error_string.contains("Test error"));
        assert!(error_string.contains("test-slice"));
    }
    
    #[test]
    fn test_error_to_grpc_status() {
        let error = SliceError::slice_not_found("test-slice");
        let status: tonic::Status = error.into();
        assert_eq!(status.code(), tonic::Code::NotFound);
    }
}