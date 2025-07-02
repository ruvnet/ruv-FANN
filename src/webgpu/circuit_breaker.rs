//! Circuit breaker pattern implementation for backend reliability
//! 
//! Agent: Backend Trait Implementer (implementer + rust-performance)

use crate::webgpu::error::ComputeError;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Circuit breaker for backend failure handling
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
    failure_threshold: usize,
    recovery_timeout: Duration,
    success_threshold: usize,
}

#[derive(Debug, PartialEq)]
enum CircuitState {
    Closed,    // Normal operation
    Open,      // Blocking requests due to failures
    HalfOpen,  // Testing if backend recovered
}

impl CircuitBreaker {
    /// Create new circuit breaker with specified thresholds
    pub fn new(
        failure_threshold: usize,
        recovery_timeout: Duration,
        success_threshold: usize,
    ) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            failure_threshold,
            recovery_timeout,
            success_threshold,
        }
    }
    
    /// Create circuit breaker with default settings for GPU backends
    pub fn new_for_gpu() -> Self {
        Self::new(
            5,                          // 5 failures trigger open
            Duration::from_secs(30),    // 30 second recovery timeout
            3,                          // 3 successes to close
        )
    }
    
    /// Create circuit breaker with default settings for CPU backends
    pub fn new_for_cpu() -> Self {
        Self::new(
            10,                         // 10 failures trigger open (more tolerant)
            Duration::from_secs(5),     // 5 second recovery timeout
            2,                          // 2 successes to close
        )
    }
    
    /// Execute operation through circuit breaker
    pub fn call<F, R>(&mut self, operation: F) -> Result<R, ComputeError>
    where
        F: FnOnce() -> Result<R, ComputeError>,
    {
        match self.state {
            CircuitState::Open => {
                if self.should_attempt_reset() {
                    self.state = CircuitState::HalfOpen;
                    self.success_count = 0;
                } else {
                    return Err(ComputeError::backend_error(
                        "Circuit breaker is open"
                    ));
                }
            }
            CircuitState::HalfOpen | CircuitState::Closed => {}
        }
        
        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure();
                Err(error)
            }
        }
    }
    
    /// Check if circuit breaker is closed (allowing requests)
    pub fn is_closed(&self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => self.should_attempt_reset(),
            CircuitState::HalfOpen => true,
        }
    }
    
    /// Get current failure count
    pub fn failure_count(&self) -> usize {
        self.failure_count
    }
    
    /// Get current success count (for half-open state)
    pub fn success_count(&self) -> usize {
        self.success_count
    }
    
    /// Force circuit breaker to open state
    pub fn force_open(&mut self) {
        self.state = CircuitState::Open;
        self.last_failure_time = Some(Instant::now());
    }
    
    /// Force circuit breaker to closed state (use with caution)
    pub fn force_close(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }
    
    /// Reset circuit breaker to initial state
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }
    
    /// Check if requests can pass through (for use without actual operation)
    pub fn check(&self) -> Result<(), ComputeError> {
        match self.state {
            CircuitState::Open => {
                if !self.should_attempt_reset() {
                    return Err(ComputeError::backend_error(
                        "Circuit breaker is open - backend temporarily unavailable"
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle successful operation
    fn on_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                    self.last_failure_time = None;
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success in closed state
                self.failure_count = 0;
            }
            CircuitState::Open => {
                // Should not happen, but handle gracefully
                self.state = CircuitState::HalfOpen;
                self.success_count = 1;
            }
        }
    }
    
    /// Handle failed operation
    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());
        
        match self.state {
            CircuitState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // Failure in half-open immediately goes back to open
                self.state = CircuitState::Open;
                self.success_count = 0;
            }
            CircuitState::Open => {
                // Already open, just update failure time
            }
        }
    }
    
    /// Check if enough time has passed to attempt reset
    fn should_attempt_reset(&self) -> bool {
        if let Some(last_failure) = self.last_failure_time {
            last_failure.elapsed() >= self.recovery_timeout
        } else {
            true
        }
    }
}

/// Circuit breaker manager for multiple backends
#[derive(Debug)]
pub struct CircuitBreakerManager {
    breakers: std::collections::HashMap<crate::webgpu::backend::BackendType, CircuitBreaker>,
}

impl CircuitBreakerManager {
    /// Create new circuit breaker manager
    pub fn new() -> Self {
        use crate::webgpu::backend::BackendType;
        let mut breakers = std::collections::HashMap::new();
        
        // Create circuit breakers for different backend types
        breakers.insert(BackendType::WebGPU, CircuitBreaker::new_for_gpu());
        breakers.insert(BackendType::CUDA, CircuitBreaker::new_for_gpu());
        breakers.insert(BackendType::OpenCL, CircuitBreaker::new_for_gpu());
        breakers.insert(BackendType::Metal, CircuitBreaker::new_for_gpu());
        breakers.insert(BackendType::SIMD, CircuitBreaker::new_for_cpu());
        breakers.insert(BackendType::CPU, CircuitBreaker::new_for_cpu());
        
        Self { breakers }
    }
    
    /// Get circuit breaker for specific backend type
    pub fn get_breaker(&mut self, backend_type: crate::webgpu::backend::BackendType) -> Option<&mut CircuitBreaker> {
        self.breakers.get_mut(&backend_type)
    }
    
    /// Check if backend is available (circuit breaker closed)
    pub fn is_available(&self, backend_type: crate::webgpu::backend::BackendType) -> bool {
        self.breakers.get(&backend_type)
            .map(|breaker| breaker.is_closed())
            .unwrap_or(false)
    }
    
    /// Execute operation with circuit breaker protection
    pub fn execute_with_protection<F, R>(
        &mut self,
        backend_type: crate::webgpu::backend::BackendType,
        operation: F,
    ) -> Result<R, ComputeError>
    where
        F: FnOnce() -> Result<R, ComputeError>,
    {
        match self.breakers.get_mut(&backend_type) {
            Some(breaker) => breaker.call(operation),
            None => Err(ComputeError::backend_error(
                format!("No circuit breaker for backend {:?}", backend_type)
            )),
        }
    }
    
    /// Get health summary for all backends
    pub fn health_summary(&self) -> std::collections::HashMap<crate::webgpu::backend::BackendType, BackendHealth> {
        self.breakers.iter()
            .map(|(backend_type, breaker)| {
                let health = match breaker.state {
                    CircuitState::Closed => BackendHealth::Healthy,
                    CircuitState::HalfOpen => BackendHealth::Recovering,
                    CircuitState::Open => BackendHealth::Failed,
                };
                (*backend_type, health)
            })
            .collect()
    }
    
    /// Reset all circuit breakers
    pub fn reset_all(&mut self) {
        for breaker in self.breakers.values_mut() {
            breaker.reset();
        }
    }
    
    /// Force all circuit breakers to specific state
    pub fn force_all_state(&mut self, force_open: bool) {
        for breaker in self.breakers.values_mut() {
            if force_open {
                breaker.force_open();
            } else {
                breaker.force_close();
            }
        }
    }
}

impl Default for CircuitBreakerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendHealth {
    Healthy,     // Circuit breaker closed, backend working
    Recovering,  // Circuit breaker half-open, testing recovery
    Failed,      // Circuit breaker open, backend failing
}

/// Circuit breaker statistics for monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub backend_type: crate::webgpu::backend::BackendType,
    pub state: String,
    pub failure_count: usize,
    pub success_count: usize,
    pub last_failure: Option<Instant>,
    pub uptime_ratio: f32,
}

impl CircuitBreakerManager {
    /// Get detailed statistics for all circuit breakers
    pub fn get_stats(&self) -> Vec<CircuitBreakerStats> {
        self.breakers.iter()
            .map(|(backend_type, breaker)| {
                let state_str = match breaker.state {
                    CircuitState::Closed => "Closed",
                    CircuitState::Open => "Open",
                    CircuitState::HalfOpen => "HalfOpen",
                };
                
                // Calculate uptime ratio (simplified)
                let uptime_ratio = if breaker.failure_count == 0 {
                    1.0
                } else {
                    let total_operations = breaker.failure_count + breaker.success_count;
                    if total_operations > 0 {
                        breaker.success_count as f32 / total_operations as f32
                    } else {
                        1.0
                    }
                };
                
                CircuitBreakerStats {
                    backend_type: *backend_type,
                    state: state_str.to_string(),
                    failure_count: breaker.failure_count,
                    success_count: breaker.success_count,
                    last_failure: breaker.last_failure_time,
                    uptime_ratio,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circuit_breaker_basic_operation() {
        let mut breaker = CircuitBreaker::new(3, Duration::from_millis(100), 2);
        
        // Initially closed
        assert!(breaker.is_closed());
        
        // Successful operations should keep it closed
        let result = breaker.call(|| Ok::<i32, ComputeError>(42));
        assert!(result.is_ok());
        assert!(breaker.is_closed());
        
        // Multiple failures should open it
        for _ in 0..3 {
            let _ = breaker.call(|| Err::<i32, ComputeError>(ComputeError::backend_error("test")));
        }
        assert!(!breaker.is_closed());
    }
    
    #[test]
    fn test_circuit_breaker_recovery() {
        let mut breaker = CircuitBreaker::new(2, Duration::from_millis(1), 1);
        
        // Cause failures to open circuit
        for _ in 0..2 {
            let _ = breaker.call(|| Err::<i32, ComputeError>(ComputeError::backend_error("test")));
        }
        assert!(!breaker.is_closed());
        
        // Wait for recovery timeout
        std::thread::sleep(Duration::from_millis(2));
        
        // Should be ready to test recovery
        assert!(breaker.is_closed());
        
        // Successful operation should close circuit
        let result = breaker.call(|| Ok::<i32, ComputeError>(42));
        assert!(result.is_ok());
        assert!(breaker.is_closed());
    }
}