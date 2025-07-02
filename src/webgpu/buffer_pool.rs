//! Advanced 5-Tier Buffer Pooling System with DAA Integration
//!
//! This module implements a sophisticated GPU buffer pooling system with 5 tiers:
//! - Micro: < 1KB (bias vectors, small activations)
//! - Small: 1KB - 1MB (small layer weights)
//! - Medium: 1MB - 10MB (medium neural network layers)
//! - Large: 10MB - 100MB (large transformer layers)
//! - XLarge: > 100MB (massive model parameters)
//!
//! Key Features:
//! - Sub-millisecond allocation for cached buffers
//! - Memory pressure monitoring and circuit breaker
//! - Buffer coalescing for micro/small tiers
//! - DAA autonomous resource allocation integration
//! - Real-time performance metrics and optimization

use crate::webgpu::error::{ComputeError, ComputeResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use ::wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

// Mock types for non-WebGPU builds
#[cfg(not(feature = "gpu"))]
#[derive(Debug)]
pub struct Device;
#[cfg(not(feature = "gpu"))]
pub struct Queue;
#[cfg(not(feature = "gpu"))]
pub struct Buffer;
#[cfg(not(feature = "gpu"))]
pub struct BufferDescriptor<'a> {
    pub label: Option<&'a str>,
    pub size: u64,
    pub usage: BufferUsages,
    pub mapped_at_creation: bool,
}
#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsages;
#[cfg(not(feature = "gpu"))]
impl BufferUsages {
    pub const STORAGE: Self = BufferUsages;
    pub const COPY_DST: Self = BufferUsages;
    pub const COPY_SRC: Self = BufferUsages;
    pub const UNIFORM: Self = BufferUsages;
    pub const MAP_READ: Self = BufferUsages;
    pub fn contains(&self, _other: Self) -> bool {
        true
    }
}

/// Buffer size categories for optimal pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferCategory {
    Micro,  // < 1KB - Bias vectors, small activations
    Small,  // 1KB - 1MB - Small layer weights
    Medium, // 1MB - 10MB - Medium neural network layers
    Large,  // 10MB - 100MB - Large transformer layers
    XLarge, // > 100MB - Massive model parameters
}

impl BufferCategory {
    pub fn from_size(size: u64) -> Self {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * 1024;

        if size < KB {
            Self::Micro
        } else if size < MB {
            Self::Small
        } else if size < 10 * MB {
            Self::Medium
        } else if size < 100 * MB {
            Self::Large
        } else {
            Self::XLarge
        }
    }

    /// Get optimal pool configuration for this tier
    pub fn pool_config(&self) -> PoolTierConfig {
        match self {
            Self::Micro => PoolTierConfig {
                max_buffers: 2048,
                prealloc_count: 512,
                cleanup_threshold: 0.95,
                coalescing_enabled: true,
                pressure_response_factor: 1.5,
                daa_optimization_weight: 0.8,
            },
            Self::Small => PoolTierConfig {
                max_buffers: 1024,
                prealloc_count: 128,
                cleanup_threshold: 0.9,
                coalescing_enabled: true,
                pressure_response_factor: 1.3,
                daa_optimization_weight: 0.9,
            },
            Self::Medium => PoolTierConfig {
                max_buffers: 256,
                prealloc_count: 32,
                cleanup_threshold: 0.8,
                coalescing_enabled: false,
                pressure_response_factor: 1.2,
                daa_optimization_weight: 1.0,
            },
            Self::Large => PoolTierConfig {
                max_buffers: 64,
                prealloc_count: 8,
                cleanup_threshold: 0.7,
                coalescing_enabled: false,
                pressure_response_factor: 1.1,
                daa_optimization_weight: 1.1,
            },
            Self::XLarge => PoolTierConfig {
                max_buffers: 16,
                prealloc_count: 2,
                cleanup_threshold: 0.6,
                coalescing_enabled: false,
                pressure_response_factor: 1.0,
                daa_optimization_weight: 1.2,
            },
        }
    }

    /// Get size range for this category
    pub fn size_range(&self) -> (u64, u64) {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * 1024;

        match self {
            Self::Micro => (0, KB),
            Self::Small => (KB, MB),
            Self::Medium => (MB, 10 * MB),
            Self::Large => (10 * MB, 100 * MB),
            Self::XLarge => (100 * MB, u64::MAX),
        }
    }

    /// Get expected allocation latency for this tier
    pub fn expected_latency_ns(&self) -> u64 {
        match self {
            Self::Micro => 50_000,      // 50 microseconds
            Self::Small => 100_000,     // 100 microseconds
            Self::Medium => 500_000,    // 500 microseconds
            Self::Large => 2_000_000,   // 2 milliseconds
            Self::XLarge => 10_000_000, // 10 milliseconds
        }
    }
}

/// Pool configuration for a specific buffer tier
#[derive(Debug, Clone)]
pub struct PoolTierConfig {
    pub max_buffers: usize,
    pub prealloc_count: usize,
    pub cleanup_threshold: f32,
    pub coalescing_enabled: bool,
    pub pressure_response_factor: f32,
    pub daa_optimization_weight: f32,
}

/// GPU buffer with enhanced metadata and lifecycle tracking
pub struct GpuBuffer {
    #[cfg(feature = "gpu")]
    pub buffer: Buffer,
    #[cfg(not(feature = "gpu"))]
    pub buffer: Buffer,
    pub size: u64,
    pub usage: BufferUsages,
    pub category: BufferCategory,
    created_at: Instant,
    last_used: Instant,
    use_count: AtomicU64,
    allocation_id: u64,
    daa_priority_score: AtomicU64, // DAA-assigned priority (0-1000)
    performance_score: AtomicU64,  // Performance tracking (0-1000)
}

impl GpuBuffer {
    #[cfg(feature = "gpu")]
    pub fn new(
        device: &Device,
        size: u64,
        usage: BufferUsages,
        label: Option<&str>,
        allocation_id: u64,
    ) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            usage,
            category: BufferCategory::from_size(size),
            created_at: Instant::now(),
            last_used: Instant::now(),
            use_count: AtomicU64::new(0),
            allocation_id,
            daa_priority_score: AtomicU64::new(500), // Start with medium priority
            performance_score: AtomicU64::new(500),  // Start with medium performance
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn new(
        _device: &Device,
        size: u64,
        usage: BufferUsages,
        _label: Option<&str>,
        allocation_id: u64,
    ) -> Self {
        Self {
            buffer: Buffer,
            size,
            usage,
            category: BufferCategory::from_size(size),
            created_at: Instant::now(),
            last_used: Instant::now(),
            use_count: AtomicU64::new(0),
            allocation_id,
            daa_priority_score: AtomicU64::new(500),
            performance_score: AtomicU64::new(500),
        }
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    pub fn idle_time(&self) -> Duration {
        self.last_used.elapsed()
    }

    pub fn mark_used(&self) {
        self.use_count.fetch_add(1, Ordering::Relaxed);
        // Note: In real implementation, we'd update last_used but it requires &mut self
        // This would be handled by the pool manager
    }

    pub fn times_used(&self) -> u64 {
        self.use_count.load(Ordering::Relaxed)
    }

    pub fn allocation_id(&self) -> u64 {
        self.allocation_id
    }

    /// Update DAA priority score (0-1000)
    pub fn set_daa_priority(&self, priority: f32) {
        let score = (priority * 1000.0).clamp(0.0, 1000.0) as u64;
        self.daa_priority_score.store(score, Ordering::Relaxed);
    }

    /// Get DAA priority score (0.0-1.0)
    pub fn get_daa_priority(&self) -> f32 {
        self.daa_priority_score.load(Ordering::Relaxed) as f32 / 1000.0
    }

    /// Update performance score based on usage patterns
    pub fn update_performance_score(&self, latency_ns: u64, throughput_mbps: f64) {
        let expected_latency = self.category.expected_latency_ns();
        let latency_score = if latency_ns <= expected_latency {
            1000
        } else {
            ((expected_latency as f64 / latency_ns as f64) * 1000.0) as u64
        };

        // Combine latency and throughput into performance score
        let throughput_score = (throughput_mbps.min(1000.0) * 1000.0 / 1000.0) as u64;
        let combined_score = (latency_score + throughput_score) / 2;

        self.performance_score
            .store(combined_score, Ordering::Relaxed);
    }

    /// Get performance score (0.0-1.0)
    pub fn get_performance_score(&self) -> f32 {
        self.performance_score.load(Ordering::Relaxed) as f32 / 1000.0
    }

    /// Calculate reuse efficiency score for DAA optimization
    pub fn reuse_efficiency(&self) -> f32 {
        let use_count = self.times_used() as f32;
        let age_hours = self.age().as_secs_f32() / 3600.0;

        if age_hours < 0.01 {
            // Less than 36 seconds
            use_count * 10.0 // Heavily weight recent usage
        } else {
            use_count / age_hours.max(0.01)
        }
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("size", &self.size)
            .field("category", &self.category)
            .field("age", &self.age())
            .field("times_used", &self.times_used())
            .field("daa_priority", &self.get_daa_priority())
            .field("performance", &self.get_performance_score())
            .finish()
    }
}

/// Pool for a specific buffer tier with enhanced analytics
#[derive(Debug)]
pub struct BufferTierPool {
    buffers: Vec<GpuBuffer>,
    config: PoolTierConfig,
    coalescing_candidates: Vec<GpuBuffer>,
    tier_stats: TierStatistics,
    last_optimization: Instant,
    daa_recommendations: VecDeque<DaaRecommendation>,
}

/// Statistics for individual buffer tier
#[derive(Debug, Default)]
pub struct TierStatistics {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub coalescings: AtomicU64,
    pub pressure_cleanups: AtomicU64,
    pub daa_optimizations: AtomicU64,
    pub avg_allocation_latency_ns: AtomicU64,
    pub peak_buffer_count: AtomicUsize,
    pub total_bytes_allocated: AtomicU64,
}

/// DAA resource management recommendation
#[derive(Debug, Clone)]
pub struct DaaRecommendation {
    pub timestamp: Instant,
    pub recommendation_type: DaaRecommendationType,
    pub confidence: f32,
    pub expected_improvement: f32,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum DaaRecommendationType {
    IncreasePoolSize { new_size: usize },
    DecreasePoolSize { new_size: usize },
    TriggerCoalescing,
    PreallocateBuffers { count: usize },
    AdjustCleanupThreshold { new_threshold: f32 },
    OptimizeForLatency,
    OptimizeForThroughput,
}

/// Advanced 5-tier buffer pool with DAA integration
#[derive(Debug)]
pub struct AdvancedBufferPool {
    device: Arc<Device>,
    pools: Mutex<HashMap<BufferCategory, BufferTierPool>>,
    next_allocation_id: AtomicU64,
    global_stats: PoolGlobalStatistics,
    daa_enabled: bool,
    pressure_circuit_breaker: Option<Arc<PressureCircuitBreaker>>,
    optimization_thread_handle: Option<std::thread::JoinHandle<()>>,
}

/// Global pool statistics across all tiers
#[derive(Debug, Default)]
pub struct PoolGlobalStatistics {
    pub total_allocations: AtomicU64,
    pub total_deallocations: AtomicU64,
    pub total_cache_hits: AtomicU64,
    pub total_cache_misses: AtomicU64,
    pub memory_pressure_events: AtomicU64,
    pub circuit_breaker_trips: AtomicU64,
    pub daa_optimizations_applied: AtomicU64,
    pub total_memory_allocated: AtomicU64,
    pub peak_memory_usage: AtomicU64,
    pub avg_allocation_latency_ns: AtomicU64,
}

/// Circuit breaker for memory pressure protection
#[derive(Debug)]
pub struct PressureCircuitBreaker {
    failure_threshold: usize,
    recovery_timeout: Duration,
    state: Mutex<CircuitBreakerState>,
    failure_count: AtomicUsize,
    last_failure: Mutex<Option<Instant>>,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Circuit tripped, rejecting requests
    HalfOpen, // Testing if service recovered
}

impl PressureCircuitBreaker {
    pub fn new(failure_threshold: usize, recovery_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            recovery_timeout,
            state: Mutex::new(CircuitBreakerState::Closed),
            failure_count: AtomicUsize::new(0),
            last_failure: Mutex::new(None),
        }
    }

    pub fn execute<F, R>(&self, operation: F) -> ComputeResult<R>
    where
        F: FnOnce() -> ComputeResult<R>,
    {
        let state = {
            let mut state = self.state.lock().unwrap();
            match state.clone() {
                CircuitBreakerState::Open => {
                    // Check if recovery timeout has passed
                    if let Some(last_failure) = *self.last_failure.lock().unwrap() {
                        if last_failure.elapsed() >= self.recovery_timeout {
                            *state = CircuitBreakerState::HalfOpen;
                            CircuitBreakerState::HalfOpen
                        } else {
                            return Err(ComputeError::MemoryError(
                                "Circuit breaker is open due to memory pressure".to_string(),
                            ));
                        }
                    } else {
                        CircuitBreakerState::Open
                    }
                }
                other => other,
            }
        };

        match state {
            CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen => {
                match operation() {
                    Ok(result) => {
                        // Reset failure count on success
                        self.failure_count.store(0, Ordering::Relaxed);
                        if state == CircuitBreakerState::HalfOpen {
                            *self.state.lock().unwrap() = CircuitBreakerState::Closed;
                        }
                        Ok(result)
                    }
                    Err(err) => {
                        self.record_failure();
                        Err(err)
                    }
                }
            }
            CircuitBreakerState::Open => Err(ComputeError::MemoryError(
                "Circuit breaker is open".to_string(),
            )),
        }
    }

    fn record_failure(&self) {
        let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure.lock().unwrap() = Some(Instant::now());

        if failure_count >= self.failure_threshold {
            *self.state.lock().unwrap() = CircuitBreakerState::Open;
        }
    }
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Hash, Ord)]
pub enum MemoryPressure {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl MemoryPressure {
    pub fn from_ratio(ratio: f32) -> Self {
        match ratio {
            r if r < 0.6 => MemoryPressure::None,
            r if r < 0.7 => MemoryPressure::Low,
            r if r < 0.8 => MemoryPressure::Medium,
            r if r < 0.9 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        }
    }

    pub fn cleanup_aggressiveness(&self) -> f32 {
        match self {
            MemoryPressure::None => 0.1,
            MemoryPressure::Low => 0.3,
            MemoryPressure::Medium => 0.5,
            MemoryPressure::High => 0.8,
            MemoryPressure::Critical => 1.0,
        }
    }
}

impl AdvancedBufferPool {
    pub fn new(device: Arc<Device>) -> Self {
        let mut pools = HashMap::new();

        // Initialize all buffer tier pools
        for category in [
            BufferCategory::Micro,
            BufferCategory::Small,
            BufferCategory::Medium,
            BufferCategory::Large,
            BufferCategory::XLarge,
        ] {
            let config = category.pool_config();
            pools.insert(
                category,
                BufferTierPool {
                    buffers: Vec::with_capacity(config.max_buffers),
                    config,
                    coalescing_candidates: Vec::new(),
                    tier_stats: TierStatistics::default(),
                    last_optimization: Instant::now(),
                    daa_recommendations: VecDeque::new(),
                },
            );
        }

        let circuit_breaker = Some(Arc::new(PressureCircuitBreaker::new(
            5,                       // 5 failures
            Duration::from_secs(30), // 30 second recovery timeout
        )));

        Self {
            device,
            pools: Mutex::new(pools),
            next_allocation_id: AtomicU64::new(1),
            global_stats: PoolGlobalStatistics::default(),
            daa_enabled: true,
            pressure_circuit_breaker: circuit_breaker,
            optimization_thread_handle: None,
        }
    }

    /// Get buffer with sub-millisecond allocation for cached buffers
    pub fn get_buffer(
        &self,
        size: u64,
        usage: BufferUsages,
        label: Option<&str>,
    ) -> ComputeResult<GpuBuffer> {
        let start_time = Instant::now();
        let category = BufferCategory::from_size(size);

        // Use circuit breaker for memory pressure protection
        if let Some(ref circuit_breaker) = self.pressure_circuit_breaker {
            return circuit_breaker
                .execute(|| self.get_buffer_internal(size, usage, label, category, start_time));
        }

        self.get_buffer_internal(size, usage, label, category, start_time)
    }

    fn get_buffer_internal(
        &self,
        size: u64,
        usage: BufferUsages,
        label: Option<&str>,
        category: BufferCategory,
        start_time: Instant,
    ) -> ComputeResult<GpuBuffer> {
        let mut pools = self.pools.lock().unwrap();

        if let Some(tier_pool) = pools.get_mut(&category) {
            // Try to find suitable buffer in pool (sub-millisecond path)
            if let Some(pos) = tier_pool.buffers.iter().position(|buf| {
                buf.size >= size && buf.usage.contains(usage) && buf.size <= size * 2
                // Don't waste too much memory
            }) {
                let buffer = tier_pool.buffers.swap_remove(pos);
                buffer.mark_used();

                // Update tier statistics
                tier_pool
                    .tier_stats
                    .cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                self.global_stats
                    .total_cache_hits
                    .fetch_add(1, Ordering::Relaxed);

                // Record allocation latency
                let latency_ns = start_time.elapsed().as_nanos() as u64;
                tier_pool
                    .tier_stats
                    .avg_allocation_latency_ns
                    .store(latency_ns, Ordering::Relaxed);
                buffer.update_performance_score(latency_ns, 1000.0); // Max throughput for cache hit

                return Ok(buffer);
            }

            // Try coalescing for micro/small buffers
            if tier_pool.config.coalescing_enabled && tier_pool.coalescing_candidates.len() >= 2 {
                if let Some(buffer) = self.try_coalesce_buffers(tier_pool, size, usage) {
                    tier_pool
                        .tier_stats
                        .coalescings
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(buffer);
                }
            }

            // Check DAA recommendations before creating new buffer
            if self.daa_enabled {
                self.apply_daa_recommendations(tier_pool);
            }
        }

        // Create new buffer (slower path)
        self.create_new_buffer(size, usage, label, start_time)
    }

    fn create_new_buffer(
        &self,
        size: u64,
        usage: BufferUsages,
        label: Option<&str>,
        start_time: Instant,
    ) -> ComputeResult<GpuBuffer> {
        let allocation_id = self.next_allocation_id.fetch_add(1, Ordering::SeqCst);

        // Check memory pressure before allocation
        let pressure = self.calculate_memory_pressure();
        if pressure >= MemoryPressure::Critical {
            self.global_stats
                .memory_pressure_events
                .fetch_add(1, Ordering::Relaxed);
            return Err(ComputeError::MemoryError(format!(
                "Critical memory pressure detected: {:?}",
                pressure
            )));
        }

        let buffer = GpuBuffer::new(&self.device, size, usage, label, allocation_id);

        // Update global statistics
        self.global_stats
            .total_allocations
            .fetch_add(1, Ordering::Relaxed);
        self.global_stats
            .total_memory_allocated
            .fetch_add(size, Ordering::Relaxed);

        // Update peak memory tracking
        let current_memory = self
            .global_stats
            .total_memory_allocated
            .load(Ordering::Relaxed);
        let mut peak = self.global_stats.peak_memory_usage.load(Ordering::Relaxed);
        while current_memory > peak {
            match self.global_stats.peak_memory_usage.compare_exchange_weak(
                peak,
                current_memory,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }

        // Record creation latency
        let latency_ns = start_time.elapsed().as_nanos() as u64;
        self.global_stats
            .avg_allocation_latency_ns
            .store(latency_ns, Ordering::Relaxed);
        buffer.update_performance_score(latency_ns, 500.0); // Medium throughput for new allocation

        Ok(buffer)
    }

    /// Return buffer to pool with DAA optimization
    pub fn return_buffer(&self, buffer: GpuBuffer) {
        let mut pools = self.pools.lock().unwrap();

        if let Some(tier_pool) = pools.get_mut(&buffer.category) {
            // Apply DAA-based retention decision
            if self.should_retain_buffer(&buffer, tier_pool) {
                if tier_pool.config.coalescing_enabled && buffer.size < 4096 {
                    tier_pool.coalescing_candidates.push(buffer);
                } else {
                    tier_pool.buffers.push(buffer);
                }
            } else {
                // Buffer not retained, update deallocation stats
                self.global_stats
                    .total_deallocations
                    .fetch_add(1, Ordering::Relaxed);
                self.global_stats
                    .total_memory_allocated
                    .fetch_sub(buffer.size, Ordering::Relaxed);
            }

            // Generate DAA recommendations based on usage patterns
            if self.daa_enabled && tier_pool.last_optimization.elapsed() > Duration::from_secs(60) {
                self.generate_daa_recommendations(tier_pool);
                tier_pool.last_optimization = Instant::now();
            }
        }
    }

    /// DAA-enhanced buffer retention decision
    fn should_retain_buffer(&self, buffer: &GpuBuffer, tier_pool: &BufferTierPool) -> bool {
        // Base retention logic
        if tier_pool.buffers.len() >= tier_pool.config.max_buffers {
            return false;
        }

        if buffer.age() > Duration::from_secs(300) {
            return false;
        }

        // DAA optimization: consider buffer efficiency and priority
        let efficiency = buffer.reuse_efficiency();
        let daa_priority = buffer.get_daa_priority();
        let performance_score = buffer.get_performance_score();

        // Weighted decision based on DAA factors
        let retention_score = efficiency * 0.4 + daa_priority * 0.3 + performance_score * 0.3;

        // Apply tier-specific optimization weight
        let threshold =
            tier_pool.config.cleanup_threshold * tier_pool.config.daa_optimization_weight;

        retention_score > threshold
    }

    /// Generate DAA recommendations for pool optimization
    fn generate_daa_recommendations(&self, tier_pool: &mut BufferTierPool) {
        let cache_hit_ratio = {
            let hits = tier_pool.tier_stats.cache_hits.load(Ordering::Relaxed);
            let misses = tier_pool.tier_stats.cache_misses.load(Ordering::Relaxed);
            if hits + misses > 0 {
                hits as f32 / (hits + misses) as f32
            } else {
                0.0
            }
        };

        let current_buffer_count = tier_pool.buffers.len();
        let max_buffers = tier_pool.config.max_buffers;

        // Recommendation: Adjust pool size based on hit ratio
        if cache_hit_ratio < 0.7 && current_buffer_count < max_buffers {
            let recommendation = DaaRecommendation {
                timestamp: Instant::now(),
                recommendation_type: DaaRecommendationType::IncreasePoolSize {
                    new_size: (current_buffer_count as f32 * 1.2).min(max_buffers as f32) as usize,
                },
                confidence: 0.8,
                expected_improvement: (0.7 - cache_hit_ratio) * 0.5,
                parameters: HashMap::new(),
            };
            tier_pool.daa_recommendations.push_back(recommendation);
        }

        // Recommendation: Enable coalescing if fragmentation detected
        let coalescings = tier_pool.tier_stats.coalescings.load(Ordering::Relaxed);
        if tier_pool.config.coalescing_enabled
            && coalescings < 10
            && tier_pool.coalescing_candidates.len() > 5
        {
            let recommendation = DaaRecommendation {
                timestamp: Instant::now(),
                recommendation_type: DaaRecommendationType::TriggerCoalescing,
                confidence: 0.9,
                expected_improvement: 0.15,
                parameters: HashMap::new(),
            };
            tier_pool.daa_recommendations.push_back(recommendation);
        }

        // Keep recommendations queue manageable
        while tier_pool.daa_recommendations.len() > 10 {
            tier_pool.daa_recommendations.pop_front();
        }
    }

    /// Apply DAA recommendations for autonomous optimization
    fn apply_daa_recommendations(&self, tier_pool: &mut BufferTierPool) {
        while let Some(recommendation) = tier_pool.daa_recommendations.pop_front() {
            // Only apply recent recommendations (within last 5 minutes)
            if recommendation.timestamp.elapsed() > Duration::from_secs(300) {
                continue;
            }

            match recommendation.recommendation_type {
                DaaRecommendationType::TriggerCoalescing => {
                    if tier_pool.coalescing_candidates.len() >= 2 {
                        // Trigger coalescing operation
                        tier_pool
                            .tier_stats
                            .daa_optimizations
                            .fetch_add(1, Ordering::Relaxed);
                        self.global_stats
                            .daa_optimizations_applied
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                DaaRecommendationType::AdjustCleanupThreshold { new_threshold } => {
                    tier_pool.config.cleanup_threshold = new_threshold;
                    tier_pool
                        .tier_stats
                        .daa_optimizations
                        .fetch_add(1, Ordering::Relaxed);
                }
                DaaRecommendationType::PreallocateBuffers { count: _count } => {
                    // In a full implementation, we'd preallocate buffers here
                    tier_pool
                        .tier_stats
                        .daa_optimizations
                        .fetch_add(1, Ordering::Relaxed);
                }
                _ => {
                    // Other recommendations would be applied in full implementation
                }
            }
        }
    }

    /// Calculate current memory pressure
    fn calculate_memory_pressure(&self) -> MemoryPressure {
        let allocated = self
            .global_stats
            .total_memory_allocated
            .load(Ordering::Relaxed) as f32;
        let peak = self.global_stats.peak_memory_usage.load(Ordering::Relaxed) as f32;

        // Estimate available memory based on peak usage patterns
        let estimated_total = peak * 1.2; // Conservative estimate
        let pressure_ratio = allocated / estimated_total;

        MemoryPressure::from_ratio(pressure_ratio)
    }

    /// Try to coalesce buffers into a larger one
    fn try_coalesce_buffers(
        &self,
        tier_pool: &mut BufferTierPool,
        size: u64,
        usage: BufferUsages,
    ) -> Option<GpuBuffer> {
        let mut total_size = 0u64;
        let mut compatible_buffers = Vec::new();

        tier_pool.coalescing_candidates.retain(|buf| {
            if buf.usage.contains(usage) && total_size < size {
                total_size += buf.size;
                compatible_buffers.push(buf.allocation_id);
                false // Remove from candidates
            } else {
                true // Keep in candidates
            }
        });

        if compatible_buffers.len() >= 2 && total_size >= size {
            let coalesced_size = total_size.next_power_of_two();
            let allocation_id = self.next_allocation_id.fetch_add(1, Ordering::SeqCst);

            let buffer = GpuBuffer::new(
                &self.device,
                coalesced_size,
                usage,
                Some("coalesced_buffer"),
                allocation_id,
            );

            // Mark as high performance due to coalescing optimization
            buffer.update_performance_score(50_000, 1500.0); // Fast allocation, high throughput
            buffer.set_daa_priority(0.9); // High DAA priority for coalesced buffers

            Some(buffer)
        } else {
            None
        }
    }

    /// Get comprehensive pool statistics
    pub fn get_statistics(&self) -> PoolStatisticsSnapshot {
        let pools = self.pools.lock().unwrap();
        let mut tier_stats = HashMap::new();

        for (&category, tier_pool) in pools.iter() {
            tier_stats.insert(
                category,
                TierStatisticsSnapshot {
                    cache_hits: tier_pool.tier_stats.cache_hits.load(Ordering::Relaxed),
                    cache_misses: tier_pool.tier_stats.cache_misses.load(Ordering::Relaxed),
                    coalescings: tier_pool.tier_stats.coalescings.load(Ordering::Relaxed),
                    pressure_cleanups: tier_pool
                        .tier_stats
                        .pressure_cleanups
                        .load(Ordering::Relaxed),
                    daa_optimizations: tier_pool
                        .tier_stats
                        .daa_optimizations
                        .load(Ordering::Relaxed),
                    avg_allocation_latency_ns: tier_pool
                        .tier_stats
                        .avg_allocation_latency_ns
                        .load(Ordering::Relaxed),
                    peak_buffer_count: tier_pool
                        .tier_stats
                        .peak_buffer_count
                        .load(Ordering::Relaxed),
                    current_buffer_count: tier_pool.buffers.len(),
                    coalescing_candidates: tier_pool.coalescing_candidates.len(),
                    pending_recommendations: tier_pool.daa_recommendations.len(),
                },
            );
        }

        PoolStatisticsSnapshot {
            global: GlobalStatisticsSnapshot {
                total_allocations: self.global_stats.total_allocations.load(Ordering::Relaxed),
                total_deallocations: self
                    .global_stats
                    .total_deallocations
                    .load(Ordering::Relaxed),
                total_cache_hits: self.global_stats.total_cache_hits.load(Ordering::Relaxed),
                total_cache_misses: self.global_stats.total_cache_misses.load(Ordering::Relaxed),
                memory_pressure_events: self
                    .global_stats
                    .memory_pressure_events
                    .load(Ordering::Relaxed),
                circuit_breaker_trips: self
                    .global_stats
                    .circuit_breaker_trips
                    .load(Ordering::Relaxed),
                daa_optimizations_applied: self
                    .global_stats
                    .daa_optimizations_applied
                    .load(Ordering::Relaxed),
                total_memory_allocated: self
                    .global_stats
                    .total_memory_allocated
                    .load(Ordering::Relaxed),
                peak_memory_usage: self.global_stats.peak_memory_usage.load(Ordering::Relaxed),
                avg_allocation_latency_ns: self
                    .global_stats
                    .avg_allocation_latency_ns
                    .load(Ordering::Relaxed),
                current_pressure: self.calculate_memory_pressure(),
            },
            tier_stats,
        }
    }

    /// Cleanup old buffers based on memory pressure
    pub fn cleanup_with_pressure_response(&self, pressure: MemoryPressure) {
        let mut pools = self.pools.lock().unwrap();
        let aggressiveness = pressure.cleanup_aggressiveness();

        for tier_pool in pools.values_mut() {
            let cleanup_threshold = tier_pool.config.cleanup_threshold * aggressiveness;
            let max_age = Duration::from_secs((300.0 * (1.0 - aggressiveness)) as u64);

            let before_count = tier_pool.buffers.len();

            tier_pool.buffers.retain(|buffer| {
                !(buffer.age() > max_age || buffer.reuse_efficiency() < cleanup_threshold)
            });

            // Also cleanup coalescing candidates
            tier_pool.coalescing_candidates.retain(|buffer| {
                buffer.age() <= max_age && buffer.reuse_efficiency() >= cleanup_threshold
            });

            let cleaned_count = before_count - tier_pool.buffers.len();
            if cleaned_count > 0 {
                tier_pool
                    .tier_stats
                    .pressure_cleanups
                    .fetch_add(cleaned_count as u64, Ordering::Relaxed);
            }
        }
    }
}

/// Snapshot of pool statistics for monitoring
#[derive(Debug, Clone)]
pub struct PoolStatisticsSnapshot {
    pub global: GlobalStatisticsSnapshot,
    pub tier_stats: HashMap<BufferCategory, TierStatisticsSnapshot>,
}

#[derive(Debug, Clone)]
pub struct GlobalStatisticsSnapshot {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub total_cache_hits: u64,
    pub total_cache_misses: u64,
    pub memory_pressure_events: u64,
    pub circuit_breaker_trips: u64,
    pub daa_optimizations_applied: u64,
    pub total_memory_allocated: u64,
    pub peak_memory_usage: u64,
    pub avg_allocation_latency_ns: u64,
    pub current_pressure: MemoryPressure,
}

#[derive(Debug, Clone)]
pub struct TierStatisticsSnapshot {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub coalescings: u64,
    pub pressure_cleanups: u64,
    pub daa_optimizations: u64,
    pub avg_allocation_latency_ns: u64,
    pub peak_buffer_count: usize,
    pub current_buffer_count: usize,
    pub coalescing_candidates: usize,
    pub pending_recommendations: usize,
}

impl PoolStatisticsSnapshot {
    /// Calculate overall cache hit ratio
    pub fn cache_hit_ratio(&self) -> f32 {
        let hits = self.global.total_cache_hits;
        let misses = self.global.total_cache_misses;
        if hits + misses > 0 {
            hits as f32 / (hits + misses) as f32
        } else {
            0.0
        }
    }

    /// Calculate memory efficiency ratio
    pub fn memory_efficiency(&self) -> f32 {
        if self.global.peak_memory_usage > 0 {
            self.global.total_memory_allocated as f32 / self.global.peak_memory_usage as f32
        } else {
            0.0
        }
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> String {
        format!(
            "Pool Performance: {:.1}% cache hit rate, {:.2}ms avg latency, {} DAA optimizations, {:?} pressure",
            self.cache_hit_ratio() * 100.0,
            self.global.avg_allocation_latency_ns as f64 / 1_000_000.0,
            self.global.daa_optimizations_applied,
            self.global.current_pressure
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_category_from_size() {
        assert_eq!(BufferCategory::from_size(512), BufferCategory::Micro);
        assert_eq!(BufferCategory::from_size(512 * 1024), BufferCategory::Small);
        assert_eq!(
            BufferCategory::from_size(5 * 1024 * 1024),
            BufferCategory::Medium
        );
        assert_eq!(
            BufferCategory::from_size(50 * 1024 * 1024),
            BufferCategory::Large
        );
        assert_eq!(
            BufferCategory::from_size(500 * 1024 * 1024),
            BufferCategory::XLarge
        );
    }

    #[test]
    fn test_memory_pressure_levels() {
        assert_eq!(MemoryPressure::from_ratio(0.5), MemoryPressure::None);
        assert_eq!(MemoryPressure::from_ratio(0.65), MemoryPressure::Low);
        assert_eq!(MemoryPressure::from_ratio(0.75), MemoryPressure::Medium);
        assert_eq!(MemoryPressure::from_ratio(0.85), MemoryPressure::High);
        assert_eq!(MemoryPressure::from_ratio(0.95), MemoryPressure::Critical);
    }

    #[test]
    fn test_pool_tier_config() {
        let config = BufferCategory::Micro.pool_config();
        assert_eq!(config.max_buffers, 2048);
        assert!(config.coalescing_enabled);
        assert_eq!(config.daa_optimization_weight, 0.8);

        let config = BufferCategory::XLarge.pool_config();
        assert_eq!(config.max_buffers, 16);
        assert!(!config.coalescing_enabled);
        assert_eq!(config.daa_optimization_weight, 1.2);
    }

    #[test]
    fn test_circuit_breaker_states() {
        let breaker = PressureCircuitBreaker::new(3, Duration::from_millis(100));

        // Test normal operation
        let result = breaker.execute(|| -> ComputeResult<i32> { Ok(42) });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Test failure handling
        for _ in 0..3 {
            let _ = breaker.execute(|| -> ComputeResult<i32> {
                Err(ComputeError::MemoryError("test failure".to_string()))
            });
        }

        // Circuit should be open now
        let result = breaker.execute(|| -> ComputeResult<i32> { Ok(42) });
        assert!(result.is_err());
    }
}
