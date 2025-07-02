//! Robust fallback system ensuring graceful degradation

use super::backend::{BackendType, ComputeBackend, CpuBackend, SimdBackend};
use super::error::ComputeError;
use num_traits::Float;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Fallback manager with circuit breaker pattern
pub struct FallbackManager<T: Float + std::fmt::Debug + Send + Sync> {
    primary_backend: Option<Box<dyn ComputeBackend<T>>>,
    fallback_backends: Vec<Box<dyn ComputeBackend<T>>>,
    failure_counts: HashMap<BackendType, usize>,
    last_failure_time: HashMap<BackendType, Instant>,
    circuit_breaker_threshold: usize,
    circuit_breaker_timeout: Duration,
}

impl<T: Float + std::fmt::Debug + Send + Sync> Default for FallbackManager<T>
where
    T: Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync> FallbackManager<T>
where
    T: Send + Sync + 'static,
{
    pub fn new() -> Self {
        let mut fallback_backends = Vec::new();

        // Try to initialize all available backends
        if let Ok(simd) = SimdBackend::initialize() {
            fallback_backends.push(Box::new(simd) as Box<dyn ComputeBackend<T>>);
        }

        if let Ok(cpu) = CpuBackend::initialize() {
            fallback_backends.push(Box::new(cpu) as Box<dyn ComputeBackend<T>>);
        }

        Self {
            primary_backend: None,
            fallback_backends,
            failure_counts: HashMap::new(),
            last_failure_time: HashMap::new(),
            circuit_breaker_threshold: 3,
            circuit_breaker_timeout: Duration::from_secs(60),
        }
    }

    #[cfg(feature = "gpu")]
    pub async fn initialize_primary_backend(&mut self) -> Result<(), ComputeError> {
        // WebGPU backend initialization would go here
        // For now, we'll return an error to indicate it's not implemented
        Err(ComputeError::InitializationError(
            "WebGPU backend initialization not yet implemented".to_string(),
        ))
    }

    pub fn execute_with_fallback<F, R>(&mut self, operation: F) -> Result<R, ComputeError>
    where
        F: Fn(&dyn ComputeBackend<T>) -> Result<R, ComputeError>,
    {
        // Try primary backend first
        if let Some(ref primary) = self.primary_backend {
            let backend_type = primary.backend_type();
            if !self.is_circuit_breaker_open(backend_type) {
                match operation(primary.as_ref()) {
                    Ok(result) => {
                        self.reset_failure_count(backend_type);
                        return Ok(result);
                    }
                    Err(e) => {
                        self.record_failure(backend_type);
                        #[cfg(feature = "logging")]
                        log::warn!("Primary backend failed: {}, falling back", e);
                    }
                }
            }
        }

        // Try fallback backends
        for i in 0..self.fallback_backends.len() {
            let backend_type = self.fallback_backends[i].backend_type();
            if !self.is_circuit_breaker_open(backend_type) {
                match operation(self.fallback_backends[i].as_ref()) {
                    Ok(result) => {
                        self.reset_failure_count(backend_type);
                        return Ok(result);
                    }
                    Err(e) => {
                        self.record_failure(backend_type);
                        #[cfg(feature = "logging")]
                        log::warn!("Fallback backend {:?} failed: {}", backend_type, e);
                    }
                }
            }
        }

        Err(ComputeError::ComputeError(
            "All backends failed".to_string(),
        ))
    }

    pub fn get_available_backends(&self) -> Vec<BackendType> {
        let mut backends = Vec::new();

        if let Some(ref primary) = self.primary_backend {
            backends.push(primary.backend_type());
        }

        for backend in &self.fallback_backends {
            backends.push(backend.backend_type());
        }

        backends
    }

    pub fn set_primary_backend(&mut self, backend: Box<dyn ComputeBackend<T>>) {
        self.primary_backend = Some(backend);
    }

    fn record_failure(&mut self, backend_type: BackendType) {
        let count = self.failure_counts.entry(backend_type).or_insert(0);
        *count += 1;
        self.last_failure_time.insert(backend_type, Instant::now());
    }

    fn reset_failure_count(&mut self, backend_type: BackendType) {
        self.failure_counts.insert(backend_type, 0);
    }

    fn is_circuit_breaker_open(&self, backend_type: BackendType) -> bool {
        if let Some(&failure_count) = self.failure_counts.get(&backend_type) {
            if failure_count >= self.circuit_breaker_threshold {
                if let Some(&last_failure) = self.last_failure_time.get(&backend_type) {
                    return last_failure.elapsed() < self.circuit_breaker_timeout;
                }
            }
        }
        false
    }

    pub fn health_status(&self) -> FallbackHealthStatus {
        let mut backend_health = HashMap::new();

        if let Some(ref primary) = self.primary_backend {
            let health = self.get_backend_health(primary.backend_type());
            backend_health.insert(primary.backend_type(), health);
        }

        for backend in &self.fallback_backends {
            let health = self.get_backend_health(backend.backend_type());
            backend_health.insert(backend.backend_type(), health);
        }

        FallbackHealthStatus {
            backend_health,
            primary_available: self.primary_backend.is_some()
                && !self
                    .is_circuit_breaker_open(self.primary_backend.as_ref().unwrap().backend_type()),
            fallback_count: self.fallback_backends.len(),
        }
    }

    fn get_backend_health(&self, backend_type: BackendType) -> BackendHealth {
        let failure_count = self.failure_counts.get(&backend_type).copied().unwrap_or(0);
        let circuit_breaker_open = self.is_circuit_breaker_open(backend_type);

        BackendHealth {
            failure_count,
            circuit_breaker_open,
            last_failure: self.last_failure_time.get(&backend_type).copied(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FallbackHealthStatus {
    pub backend_health: HashMap<BackendType, BackendHealth>,
    pub primary_available: bool,
    pub fallback_count: usize,
}

#[derive(Debug, Clone)]
pub struct BackendHealth {
    pub failure_count: usize,
    pub circuit_breaker_open: bool,
    pub last_failure: Option<Instant>,
}

impl FallbackHealthStatus {
    pub fn is_healthy(&self) -> bool {
        self.primary_available || self.fallback_count > 0
    }

    pub fn get_active_backends(&self) -> Vec<BackendType> {
        self.backend_health
            .iter()
            .filter(|(_, health)| !health.circuit_breaker_open)
            .map(|(backend_type, _)| *backend_type)
            .collect()
    }
}
