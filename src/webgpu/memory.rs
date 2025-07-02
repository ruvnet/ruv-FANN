//! GPU memory management and buffer pooling

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, atomic::{AtomicU64, AtomicUsize, Ordering}};
use crate::webgpu::error::{ComputeError, ComputeResult};
use crate::webgpu::device::GpuDevice;

/// Buffer category based on size
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferCategory {
    Micro,   // < 1KB - Bias vectors, small activations
    Small,   // 1KB - 1MB - Small layer weights
    Medium,  // 1MB - 10MB - Medium neural network layers
    Large,   // 10MB - 100MB - Large transformer layers
    XLarge,  // > 100MB - Massive model parameters
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
    
    /// Get optimal pool configuration for this tier
    pub fn pool_config(&self) -> PoolTierConfig {
        match self {
            Self::Micro => PoolTierConfig {
                max_buffers: 1024,
                prealloc_count: 256,
                cleanup_threshold: 0.9,
                coalescing_enabled: true,
            },
            Self::Small => PoolTierConfig {
                max_buffers: 512,
                prealloc_count: 64,
                cleanup_threshold: 0.8,
                coalescing_enabled: true,
            },
            Self::Medium => PoolTierConfig {
                max_buffers: 128,
                prealloc_count: 16,
                cleanup_threshold: 0.7,
                coalescing_enabled: false,
            },
            Self::Large => PoolTierConfig {
                max_buffers: 32,
                prealloc_count: 4,
                cleanup_threshold: 0.6,
                coalescing_enabled: false,
            },
            Self::XLarge => PoolTierConfig {
                max_buffers: 8,
                prealloc_count: 1,
                cleanup_threshold: 0.5,
                coalescing_enabled: false,
            },
        }
    }
}

/// Pool configuration for a buffer tier
#[derive(Debug, Clone)]
pub struct PoolTierConfig {
    pub max_buffers: usize,
    pub prealloc_count: usize,
    pub cleanup_threshold: f32,
    pub coalescing_enabled: bool,
}

/// A GPU buffer with metadata
pub struct GpuBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub category: BufferCategory,
    created_at: std::time::Instant,
    last_used: std::time::Instant,
    use_count: AtomicU64,
}

impl GpuBuffer {
    pub fn new(
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
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
            created_at: std::time::Instant::now(),
            last_used: std::time::Instant::now(),
            use_count: AtomicU64::new(0),
        }
    }

    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    pub fn write_data(&self, queue: &wgpu::Queue, data: &[u8]) -> ComputeResult<()> {
        if data.len() as u64 > self.size {
            return Err(ComputeError::buffer_error(format!(
                "Data size {} exceeds buffer size {}", 
                data.len(), 
                self.size
            )));
        }

        queue.write_buffer(&self.buffer, 0, data);
        self.use_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    pub fn times_used(&self) -> u64 {
        self.use_count.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("size", &self.size)
            .field("usage", &self.usage)
            .field("category", &self.category)
            .field("age", &self.age())
            .finish()
    }
}

/// Buffer pool for efficient GPU memory management
#[derive(Debug)]
pub struct BufferPool {
    device: Arc<GpuDevice>,
    pools: Mutex<HashMap<BufferCategory, BufferTierPool>>,
    total_allocated: AtomicU64,
    stats: PoolStats,
}

/// Pool for a specific buffer tier
#[derive(Debug)]
pub struct BufferTierPool {
    buffers: Vec<GpuBuffer>,
    config: PoolTierConfig,
    coalescing_candidates: Vec<GpuBuffer>,
}

#[derive(Debug, Default)]
pub struct PoolStats {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub reuse_count: AtomicU64,
    pub fragmentation_events: AtomicU64,
    pub total_memory_allocated: AtomicUsize,
    pub peak_memory_usage: AtomicUsize,
    pub coalescing_operations: AtomicU64,
    pub defragmentation_runs: AtomicU64,
}

/// Snapshot of pool statistics
#[derive(Debug, Clone)]
pub struct PoolStatsSnapshot {
    pub allocations: u64,
    pub deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub reuse_count: u64,
    pub fragmentation_events: u64,
    pub total_memory_allocated: usize,
    pub peak_memory_usage: usize,
    pub coalescing_operations: u64,
    pub defragmentation_runs: u64,
}

impl BufferPool {
    pub fn new(device: Arc<GpuDevice>) -> Self {
        let mut pools = HashMap::new();
        
        // Initialize buffer pools for each tier
        for category in [BufferCategory::Micro, BufferCategory::Small, 
                        BufferCategory::Medium, BufferCategory::Large, 
                        BufferCategory::XLarge] {
            let config = category.pool_config();
            pools.insert(category, BufferTierPool {
                buffers: Vec::with_capacity(config.max_buffers),
                config,
                coalescing_candidates: Vec::new(),
            });
        }
        
        Self {
            device,
            pools: Mutex::new(pools),
            total_allocated: AtomicU64::new(0),
            stats: PoolStats::default(),
        }
    }

    /// Get a buffer from the pool or create a new one
    pub fn get_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> ComputeResult<GpuBuffer> {
        let category = BufferCategory::from_size(size);
        let mut pools = self.pools.lock().unwrap();
        // Try to find a suitable buffer in the pool
        if let Some(tier_pool) = pools.get_mut(&category) {
            // Look for a buffer that's large enough and has compatible usage
            if let Some(pos) = tier_pool.buffers.iter().position(|buf| 
                buf.size >= size && buf.usage.contains(usage) && 
                buf.size <= size * 2 // Don't waste too much memory
            ) {
                let mut buffer = tier_pool.buffers.swap_remove(pos);
                buffer.last_used = std::time::Instant::now();
                buffer.use_count.fetch_add(1, Ordering::Relaxed);
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.stats.reuse_count.fetch_add(1, Ordering::Relaxed);
                return Ok(buffer);
            }
            
            // Try coalescing for micro/small buffers if enabled
            if tier_pool.config.coalescing_enabled && 
               tier_pool.coalescing_candidates.len() >= 2 {
                if let Some(buffer) = self.try_coalesce_buffers(tier_pool, size, usage) {
                    self.stats.coalescing_operations.fetch_add(1, Ordering::Relaxed);
                    return Ok(buffer);
                }
            }
        }

        // No suitable buffer found, create a new one
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);
        
        let buffer = GpuBuffer::new(&self.device.device, size, usage, label);
        
        // Update memory tracking
        let old_total = self.total_allocated.fetch_add(size, Ordering::SeqCst);
        let new_total = old_total + size;
        self.stats.total_memory_allocated.fetch_add(size as usize, Ordering::Relaxed);
        
        // Update peak usage atomically
        let mut peak = self.stats.peak_memory_usage.load(Ordering::Relaxed);
        while new_total as usize > peak {
            match self.stats.peak_memory_usage.compare_exchange_weak(
                peak, new_total as usize, Ordering::SeqCst, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }

        Ok(buffer)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, buffer: GpuBuffer) {
        let mut pools = self.pools.lock().unwrap();
        
        if let Some(tier_pool) = pools.get_mut(&buffer.category) {
            // Check if we should keep this buffer
            if self.should_return_to_pool(&buffer, tier_pool) {
                // For coalescing-enabled tiers, consider as candidate
                if tier_pool.config.coalescing_enabled && 
                   buffer.size < 4096 { // Only coalesce small buffers
                    tier_pool.coalescing_candidates.push(buffer);
                } else {
                    tier_pool.buffers.push(buffer);
                }
            } else {
                // Pool is full or buffer is too old, drop it
                self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
                self.total_allocated.fetch_sub(buffer.size, Ordering::SeqCst);
            }
        }
    }

    /// Get current pool statistics
    pub fn get_stats(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            allocations: self.stats.allocations.load(Ordering::Relaxed),
            deallocations: self.stats.deallocations.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            reuse_count: self.stats.reuse_count.load(Ordering::Relaxed),
            fragmentation_events: self.stats.fragmentation_events.load(Ordering::Relaxed),
            total_memory_allocated: self.stats.total_memory_allocated.load(Ordering::Relaxed),
            peak_memory_usage: self.stats.peak_memory_usage.load(Ordering::Relaxed),
            coalescing_operations: self.stats.coalescing_operations.load(Ordering::Relaxed),
            defragmentation_runs: self.stats.defragmentation_runs.load(Ordering::Relaxed),
        }
    }
    
    /// Check if buffer should be returned to pool
    fn should_return_to_pool(&self, buffer: &GpuBuffer, tier_pool: &BufferTierPool) -> bool {
        // Don't keep if pool is full
        if tier_pool.buffers.len() >= tier_pool.config.max_buffers {
            return false;
        }
        
        // Don't keep very old buffers
        if buffer.age() > std::time::Duration::from_secs(300) {
            return false;
        }
        
        // Don't keep rarely used buffers
        if buffer.times_used() < 2 && buffer.age() > std::time::Duration::from_secs(60) {
            return false;
        }
        
        true
    }
    
    /// Try to coalesce small buffers into a larger one
    fn try_coalesce_buffers(
        &self,
        tier_pool: &mut BufferTierPool,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<GpuBuffer> {
        // Find compatible buffers to coalesce
        let mut total_size = 0u64;
        let mut compatible_buffers = Vec::new();
        
        tier_pool.coalescing_candidates.retain(|buf| {
            if buf.usage.contains(usage) && total_size < size {
                total_size += buf.size;
                compatible_buffers.push(buf.size);
                false // Remove from candidates
            } else {
                true // Keep in candidates
            }
        });
        
        // Need at least 2 buffers and enough total size
        if compatible_buffers.len() >= 2 && total_size >= size {
            // Create new coalesced buffer
            let coalesced_size = total_size.next_power_of_two();
            let buffer = GpuBuffer::new(
                &self.device.device,
                coalesced_size,
                usage,
                Some("coalesced_buffer"),
            );
            
            self.stats.fragmentation_events.fetch_add(1, Ordering::Relaxed);
            return Some(buffer);
        }
        
        None
    }

    /// Clean up old buffers to free memory
    pub fn cleanup(&self, max_age: std::time::Duration) {
        let mut pools = self.pools.lock().unwrap();
        let mut total_freed = 0u64;

        for tier_pool in pools.values_mut() {
            let before_count = tier_pool.buffers.len();
            
            tier_pool.buffers.retain(|buffer| {
                if buffer.age() > max_age || 
                   (buffer.times_used() == 0 && buffer.age() > std::time::Duration::from_secs(30)) {
                    self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
                    total_freed += buffer.size;
                    false
                } else {
                    true
                }
            });
            
            // Also clean up coalescing candidates
            tier_pool.coalescing_candidates.retain(|buffer| {
                if buffer.age() > std::time::Duration::from_secs(60) {
                    total_freed += buffer.size;
                    false
                } else {
                    true
                }
            });
            
            let after_count = tier_pool.buffers.len();
            if before_count != after_count {
                log::debug!(
                    "Cleaned up {} buffers from {:?} tier",
                    before_count - after_count,
                    tier_pool.buffers.first().map(|b| b.category)
                );
            }
        }
        
        if total_freed > 0 {
            self.total_allocated.fetch_sub(total_freed, Ordering::SeqCst);
        }
    }
    
    /// Defragment memory pools by consolidating buffers
    pub fn defragment(&self) {
        let mut pools = self.pools.lock().unwrap();
        self.stats.defragmentation_runs.fetch_add(1, Ordering::Relaxed);
        
        for tier_pool in pools.values_mut() {
            if !tier_pool.config.coalescing_enabled {
                continue;
            }
            
            // Sort buffers by size for better packing
            tier_pool.buffers.sort_by_key(|b| b.size);
            
            // Move small, rarely used buffers to coalescing candidates
            let threshold = tier_pool.config.max_buffers / 2;
            if tier_pool.buffers.len() > threshold {
                let drain_count = tier_pool.buffers.len() - threshold;
                let candidates: Vec<_> = tier_pool.buffers
                    .drain(0..drain_count)
                    .filter(|b| b.times_used() < 3)
                    .collect();
                    
                tier_pool.coalescing_candidates.extend(candidates);
            }
        }
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Check if we're approaching memory limits
    pub fn memory_pressure(&self) -> f32 {
        let current = self.memory_usage();
        let max_buffer_size = self.device.max_buffer_size();
        
        // Calculate pressure as percentage of maximum possible allocation
        // This is a rough estimate - actual limits may be lower
        let estimated_max = max_buffer_size / 4; // Conservative estimate
        (current as f32) / (estimated_max as f32)
    }
}

/// GPU memory manager with automatic buffer pooling
#[derive(Debug)]
pub struct GpuMemoryManager {
    buffer_pool: BufferPool,
    device: Arc<GpuDevice>,
    memory_pressure_threshold: f32,
    circuit_breaker: Option<Arc<crate::webgpu::circuit_breaker::CircuitBreaker>>,
    pressure_history: Mutex<VecDeque<PressureReading>>,
    last_defrag_time: Mutex<std::time::Instant>,
}

#[derive(Debug, Clone)]
pub struct PressureReading {
    pub pressure: f32,
    pub timestamp: std::time::Instant,
    pub memory_usage: u64,
    pub triggered_cleanup: bool,
}

impl GpuMemoryManager {
    pub fn new(device: Arc<GpuDevice>) -> Self {
        let buffer_pool = BufferPool::new(device.clone());
        
        Self {
            buffer_pool,
            device,
            memory_pressure_threshold: 0.8, // Trigger cleanup at 80% memory usage
            circuit_breaker: None,
            pressure_history: Mutex::new(VecDeque::with_capacity(100)),
            last_defrag_time: Mutex::new(std::time::Instant::now()),
        }
    }

    /// Create a buffer for storing data
    pub fn create_storage_buffer(&self, size: u64, label: Option<&str>) -> ComputeResult<GpuBuffer> {
        // Check memory pressure before allocation
        self.check_memory_pressure()?;
        
        // Consider defragmentation if needed
        self.maybe_defragment();
        
        self.buffer_pool.get_buffer(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            label,
        )
    }

    /// Create a buffer for uniform data
    pub fn create_uniform_buffer(&self, size: u64, label: Option<&str>) -> ComputeResult<GpuBuffer> {
        self.buffer_pool.get_buffer(
            size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    /// Create a buffer for reading results back to CPU
    pub fn create_readback_buffer(&self, size: u64, label: Option<&str>) -> ComputeResult<GpuBuffer> {
        self.buffer_pool.get_buffer(
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    /// Copy data from one buffer to another
    pub fn copy_buffer_to_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &GpuBuffer,
        destination: &GpuBuffer,
        size: u64,
    ) -> ComputeResult<()> {
        if size > source.size || size > destination.size {
            return Err(ComputeError::buffer_error(
                "Copy size exceeds buffer capacity".to_string()
            ));
        }

        encoder.copy_buffer_to_buffer(&source.buffer, 0, &destination.buffer, 0, size);
        Ok(())
    }

    /// Write data to a buffer
    pub fn write_buffer_data(&self, buffer: &GpuBuffer, data: &[u8]) -> ComputeResult<()> {
        buffer.write_data(&self.device.queue, data)
    }

    /// Return a buffer to the pool when done
    pub fn return_buffer(&self, buffer: GpuBuffer) {
        self.buffer_pool.return_buffer(buffer);
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> PoolStatsSnapshot {
        self.buffer_pool.get_stats()
    }
    
    /// Check memory pressure and trigger cleanup if needed
    fn check_memory_pressure(&self) -> ComputeResult<()> {
        let pressure = self.memory_pressure();
        let memory_usage = self.buffer_pool.memory_usage();
        let mut triggered_cleanup = false;
        
        // Check if circuit breaker should trip
        if let Some(ref circuit_breaker) = self.circuit_breaker {
            circuit_breaker.check()?;
        }
        
        // Trigger cleanup if pressure is above threshold
        if pressure > self.memory_pressure_threshold {
            triggered_cleanup = true;
            self.cleanup_old_buffers();
            
            // If still above threshold after cleanup, try defragmentation
            if self.memory_pressure() > 0.9 {
                self.buffer_pool.defragment();
            }
            
            // If still above critical threshold, return error
            if self.memory_pressure() > 0.95 {
                return Err(ComputeError::memory_error(
                    format!("GPU memory pressure too high: {:.0}%", pressure * 100.0)
                ));
            }
        }
        
        // Record pressure reading
        {
            let mut history = self.pressure_history.lock().unwrap();
            history.push_back(PressureReading {
                pressure,
                timestamp: std::time::Instant::now(),
                memory_usage,
                triggered_cleanup,
            });
            
            // Keep history size manageable
            if history.len() > 100 {
                history.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Set circuit breaker for memory management
    pub fn set_circuit_breaker(&mut self, circuit_breaker: Arc<crate::webgpu::circuit_breaker::CircuitBreaker>) {
        self.circuit_breaker = Some(circuit_breaker);
    }

    /// Check current memory pressure (0.0 = no pressure, 1.0 = high pressure)
    pub fn memory_pressure(&self) -> f32 {
        self.buffer_pool.memory_pressure()
    }

    /// Perform cleanup of old buffers
    pub fn cleanup_old_buffers(&self) {
        // Adaptive cleanup based on memory pressure
        let pressure = self.memory_pressure();
        let max_age = if pressure > 0.9 {
            std::time::Duration::from_secs(30)  // Aggressive cleanup
        } else if pressure > 0.7 {
            std::time::Duration::from_secs(120) // Moderate cleanup
        } else {
            std::time::Duration::from_secs(300) // Conservative cleanup
        };
        
        self.buffer_pool.cleanup(max_age);
    }
    
    /// Check if defragmentation should run
    fn maybe_defragment(&self) {
        let mut last_defrag = self.last_defrag_time.lock().unwrap();
        
        // Run defragmentation at most once per minute
        if last_defrag.elapsed() > std::time::Duration::from_secs(60) {
            let stats = self.buffer_pool.get_stats();
            
            // Defragment if we have high fragmentation
            if stats.fragmentation_events > 100 || 
               (stats.cache_misses > stats.cache_hits * 2) {
                self.buffer_pool.defragment();
                *last_defrag = std::time::Instant::now();
            }
        }
    }
    
    /// Get memory pressure trend over time
    pub fn get_pressure_trend(&self) -> PressureTrend {
        let history = self.pressure_history.lock().unwrap();
        
        if history.len() < 5 {
            return PressureTrend::Stable;
        }
        
        // Calculate average pressure over recent readings
        let recent: Vec<_> = history.iter().rev().take(10).collect();
        let recent_avg: f32 = recent.iter().map(|r| r.pressure).sum::<f32>() / recent.len() as f32;
        
        // Calculate average pressure over older readings
        let older: Vec<_> = history.iter().rev().skip(10).take(10).collect();
        if older.is_empty() {
            return PressureTrend::Stable;
        }
        
        let older_avg: f32 = older.iter().map(|r| r.pressure).sum::<f32>() / older.len() as f32;
        
        // Determine trend
        let diff = recent_avg - older_avg;
        if diff > 0.1 {
            PressureTrend::Increasing
        } else if diff < -0.1 {
            PressureTrend::Decreasing
        } else {
            PressureTrend::Stable
        }
    }
    
    /// Get detailed memory analysis
    pub fn analyze_memory_usage(&self) -> MemoryAnalysis {
        let stats = self.buffer_pool.get_stats();
        let pressure = self.memory_pressure();
        let trend = self.get_pressure_trend();
        
        // Calculate efficiency metrics
        let total_operations = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total_operations > 0 {
            stats.cache_hits as f32 / total_operations as f32
        } else {
            0.0
        };
        
        let fragmentation_rate = if stats.allocations > 0 {
            stats.fragmentation_events as f32 / stats.allocations as f32
        } else {
            0.0
        };
        
        MemoryAnalysis {
            current_usage: self.buffer_pool.memory_usage(),
            peak_usage: stats.peak_memory_usage as u64,
            pressure,
            trend,
            hit_rate,
            fragmentation_rate,
            total_allocations: stats.allocations,
            total_deallocations: stats.deallocations,
            reuse_count: stats.reuse_count,
            coalescing_operations: stats.coalescing_operations,
            defragmentation_runs: stats.defragmentation_runs,
        }
    }
}

/// Memory pressure trend analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressureTrend {
    Increasing,
    Stable,
    Decreasing,
}

/// Detailed memory usage analysis
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    pub current_usage: u64,
    pub peak_usage: u64,
    pub pressure: f32,
    pub trend: PressureTrend,
    pub hit_rate: f32,
    pub fragmentation_rate: f32,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub reuse_count: u64,
    pub coalescing_operations: u64,
    pub defragmentation_runs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_category_from_size() {
        assert_eq!(BufferCategory::from_size(512), BufferCategory::Micro);
        assert_eq!(BufferCategory::from_size(512 * 1024), BufferCategory::Small);
        assert_eq!(BufferCategory::from_size(5 * 1024 * 1024), BufferCategory::Medium);
        assert_eq!(BufferCategory::from_size(50 * 1024 * 1024), BufferCategory::Large);
        assert_eq!(BufferCategory::from_size(500 * 1024 * 1024), BufferCategory::XLarge);
    }

    #[test]
    fn test_buffer_category_size_ranges() {
        let (min, max) = BufferCategory::Micro.size_range();
        assert_eq!(min, 0);
        assert_eq!(max, 1024);
        
        let (min, max) = BufferCategory::Small.size_range();
        assert_eq!(min, 1024);
        assert_eq!(max, 1024 * 1024);

        let (min, max) = BufferCategory::XLarge.size_range();
        assert_eq!(min, 100 * 1024 * 1024);
        assert_eq!(max, u64::MAX);
    }

    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats::default();
        assert_eq!(stats.allocations.load(Ordering::Relaxed), 0);
        assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_memory_allocated.load(Ordering::Relaxed), 0);
        assert_eq!(stats.coalescing_operations.load(Ordering::Relaxed), 0);
        assert_eq!(stats.defragmentation_runs.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_pool_tier_config() {
        // Test Micro tier config
        let config = BufferCategory::Micro.pool_config();
        assert_eq!(config.max_buffers, 1024);
        assert_eq!(config.prealloc_count, 256);
        assert!(config.coalescing_enabled);
        
        // Test XLarge tier config
        let config = BufferCategory::XLarge.pool_config();
        assert_eq!(config.max_buffers, 8);
        assert_eq!(config.prealloc_count, 1);
        assert!(!config.coalescing_enabled);
    }
}