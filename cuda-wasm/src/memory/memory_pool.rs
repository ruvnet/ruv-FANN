//! High-performance memory pool for WASM optimization
//!
//! This module provides efficient memory allocation patterns optimized for
//! WASM environments with minimal allocation overhead.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::error::Result;

/// Memory pool configuration for optimal performance
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum pool size per allocation class (in bytes)
    pub max_pool_size: usize,
    /// Minimum allocation size to pool
    pub min_pooled_size: usize,
    /// Maximum allocation size to pool
    pub max_pooled_size: usize,
    /// Number of pre-allocated buffers per size class
    pub prealloc_count: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 16 * 1024 * 1024, // 16MB max per pool
            min_pooled_size: 1024,            // 1KB min
            max_pooled_size: 4 * 1024 * 1024, // 4MB max
            prealloc_count: 8,                // Pre-allocate 8 buffers
        }
    }
}

/// High-performance memory pool optimized for WASM
#[derive(Debug)]
pub struct MemoryPool {
    /// Pools organized by power-of-2 sizes
    pools: Arc<Mutex<HashMap<usize, Vec<Vec<u8>>>>>,
    /// Configuration
    config: PoolConfig,
    /// Statistics for performance monitoring
    stats: Arc<Mutex<PoolStats>>,
}

/// Performance statistics for the memory pool
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total allocations requested
    pub total_allocations: u64,
    /// Cache hits (allocations served from pool)
    pub cache_hits: u64,
    /// Cache misses (new allocations)
    pub cache_misses: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Total bytes served from pool
    pub pooled_bytes_served: u64,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Current memory usage
    pub current_memory_usage: usize,
}

impl MemoryPool {
    /// Create a new memory pool with default configuration
    pub fn new() -> Self {
        Self::with_config(PoolConfig::default())
    }

    /// Create a new memory pool with custom configuration
    pub fn with_config(config: PoolConfig) -> Self {
        let pool = Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            config,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        };
        
        // Pre-allocate common sizes
        pool.preallocate_common_sizes();
        pool
    }

    /// Pre-allocate buffers for common sizes to reduce allocation overhead
    fn preallocate_common_sizes(&self) {
        let common_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
        
        for &size in &common_sizes {
            if size >= self.config.min_pooled_size && size <= self.config.max_pooled_size {
                let pool_size = self.round_to_power_of_2(size);
                let mut pools = self.pools.lock().unwrap();
                let pool = pools.entry(pool_size).or_default();
                
                let count = self.config.prealloc_count.min(self.config.max_pool_size / pool_size);
                for _ in 0..count {
                    pool.push(vec![0; pool_size]);
                }
            }
        }
    }

    /// Allocate a buffer of the specified size
    pub fn allocate(&self, size: usize) -> Vec<u8> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_allocations += 1;
        stats.total_bytes_allocated += size as u64;

        // Don't pool very small or very large allocations
        if size < self.config.min_pooled_size || size > self.config.max_pooled_size {
            stats.cache_misses += 1;
            stats.current_memory_usage += size;
            if stats.current_memory_usage > stats.peak_memory_usage {
                stats.peak_memory_usage = stats.current_memory_usage;
            }
            drop(stats);
            return vec![0; size];
        }

        let pool_size = self.round_to_power_of_2(size);
        let mut pools = self.pools.lock().unwrap();
        
        if let Some(pool) = pools.get_mut(&pool_size) {
            if let Some(mut buffer) = pool.pop() {
                // Cache hit
                stats.cache_hits += 1;
                stats.pooled_bytes_served += pool_size as u64;
                drop(stats);
                drop(pools);
                
                // Resize buffer to exact size needed
                buffer.resize(size, 0);
                return buffer;
            }
        }

        // Cache miss - create new buffer
        stats.cache_misses += 1;
        stats.current_memory_usage += pool_size;
        if stats.current_memory_usage > stats.peak_memory_usage {
            stats.peak_memory_usage = stats.current_memory_usage;
        }
        drop(stats);
        drop(pools);
        
        vec![0; size]
    }

    /// Return a buffer to the pool for reuse
    pub fn deallocate(&self, mut buffer: Vec<u8>) {
        let original_size = buffer.len();
        
        // Don't pool very small or very large allocations
        if original_size < self.config.min_pooled_size || original_size > self.config.max_pooled_size {
            let mut stats = self.stats.lock().unwrap();
            stats.current_memory_usage = stats.current_memory_usage.saturating_sub(original_size);
            return;
        }

        let pool_size = self.round_to_power_of_2(original_size);
        // Clear stale data before reuse without a redundant first resize.
        buffer.clear();
        buffer.resize(pool_size, 0);

        let mut pools = self.pools.lock().unwrap();
        let pool = pools.entry(pool_size).or_default();
        
        // Limit pool size to prevent memory bloat
        if pool.len() < self.config.max_pool_size / pool_size {
            pool.push(buffer);
        } else {
            // Drop the pools lock before taking stats: allocate takes stats
            // first, so retaining pools here could deadlock two threads.
            drop(pools);
            let mut stats = self.stats.lock().unwrap();
            stats.current_memory_usage = stats.current_memory_usage.saturating_sub(pool_size);
        }
    }

    /// Round size up to the next power of 2 for efficient pooling
    fn round_to_power_of_2(&self, size: usize) -> usize {
        if size <= 1 {
            return 1;
        }
        
        // A configured class near usize::MAX must not wrap to zero and loop.
        size.checked_next_power_of_two().unwrap_or(size)
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cache hit ratio as a percentage
    pub fn hit_ratio(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        if stats.total_allocations == 0 {
            return 0.0;
        }
        (stats.cache_hits as f64 / stats.total_allocations as f64) * 100.0
    }

    /// Clear all pools and reset statistics
    pub fn clear(&self) {
        self.pools.lock().unwrap().clear();
        let mut stats = self.stats.lock().unwrap();
        *stats = PoolStats::default();
    }

    /// Get total memory usage across all pools
    pub fn total_pooled_memory(&self) -> usize {
        let pools = self.pools.lock().unwrap();
        pools.iter()
            .map(|(&size, pool)| size * pool.len())
            .sum()
    }

    /// Shrink pools to release unused memory
    pub fn shrink_to_fit(&self) {
        let mut pools = self.pools.lock().unwrap();
        for pool in pools.values_mut() {
            pool.shrink_to_fit();
        }
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Global memory pool instance for efficient allocation
static GLOBAL_POOL: std::sync::OnceLock<MemoryPool> = std::sync::OnceLock::new();

/// Get or initialize the global memory pool
pub fn global_pool() -> &'static MemoryPool {
    GLOBAL_POOL.get_or_init(MemoryPool::new)
}

/// Allocate from the global pool
pub fn allocate(size: usize) -> Vec<u8> {
    global_pool().allocate(size)
}

/// Deallocate to the global pool
pub fn deallocate(buffer: Vec<u8>) {
    global_pool().deallocate(buffer);
}

/// Get global pool statistics
pub fn global_stats() -> PoolStats {
    global_pool().stats()
}

/// High-level memory management for kernel operations
pub struct KernelMemoryManager {
    pool: Arc<MemoryPool>,
    allocations: Mutex<HashMap<usize, (Vec<u8>, usize)>>,
}

impl KernelMemoryManager {
    /// Create a new kernel memory manager
    pub fn new() -> Self {
        Self {
            pool: Arc::new(MemoryPool::new()),
            allocations: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate aligned memory for kernel operations
    pub fn allocate_kernel_memory(&self, size: usize, alignment: usize) -> Result<*mut u8> {
        if size == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(crate::memory_error!("Kernel size must be nonzero and alignment a nonzero power of two"));
        }
        let len = size.checked_add(alignment - 1)
            .filter(|&len| len <= isize::MAX as usize)
            .ok_or_else(|| crate::memory_error!("Kernel allocation size overflow"))?;
        let mut buffer = self.pool.allocate(len);
        let base = buffer.as_mut_ptr() as usize;
        let offset = base.wrapping_neg() & (alignment - 1);
        // SAFETY: padding guarantees offset + size fits within the owned Vec.
        let ptr = unsafe { buffer.as_mut_ptr().add(offset) };
        self.allocations.lock().unwrap().insert(ptr as usize, (buffer, size));
        Ok(ptr)
    }

    /// Release kernel memory without reconstructing its allocation layout.
    ///
    /// # Safety
    /// The caller must stop using the pointer and any references derived from
    /// it before release, and must not reuse it after this function returns.
    pub unsafe fn deallocate_kernel_memory(&self, ptr: *mut u8) -> Result<()> {
        let (buffer, _) = self.allocations.lock().unwrap().remove(&(ptr as usize))
            .ok_or_else(|| crate::memory_error!("Invalid pointer for deallocation"))?;
        self.pool.deallocate(buffer);
        Ok(())
    }

    /// Get total allocated kernel memory
    pub fn total_kernel_memory(&self) -> usize {
        let allocations = self.allocations.lock().unwrap();
        allocations.values().map(|(_, size)| size).sum()
    }
}

impl Default for KernelMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let pool = MemoryPool::new();
        
        // Test allocation
        let buffer1 = pool.allocate(1024);
        assert_eq!(buffer1.len(), 1024);
        
        // Test deallocation and reuse
        pool.deallocate(buffer1);
        let buffer2 = pool.allocate(1024);
        assert_eq!(buffer2.len(), 1024);
        
        // Should be a cache hit
        assert!(pool.hit_ratio() > 0.0);
    }

    #[test]
    fn test_power_of_2_rounding() {
        let pool = MemoryPool::new();
        assert_eq!(pool.round_to_power_of_2(1000), 1024);
        assert_eq!(pool.round_to_power_of_2(1024), 1024);
        assert_eq!(pool.round_to_power_of_2(1500), 2048);
    }

    #[test]
    fn test_global_pool() {
        let buffer = allocate(2048);
        assert_eq!(buffer.len(), 2048);
        
        deallocate(buffer);
        let stats = global_stats();
        assert!(stats.total_allocations > 0);
    }

    #[test]
    fn test_kernel_memory_manager() {
        let manager = KernelMemoryManager::new();
        
        unsafe {
            let ptr = manager.allocate_kernel_memory(4096, 16).unwrap();
            assert!(!ptr.is_null());
            
            manager.deallocate_kernel_memory(ptr).unwrap();
        }
    }
}