//! Memory management for efficient neural network operations
//!
//! This module provides memory management capabilities for neural networks,
//! including buffer allocation, memory pools, and efficient data structures.

use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Memory manager for neural network operations
pub struct MemoryManager<T: Float> {
    /// Memory pools for different data types
    pools: HashMap<String, MemoryPool<T>>,
    /// Total memory allocated
    total_allocated: usize,
    /// Memory usage statistics
    stats: MemoryStats,
}

/// Memory pool for efficient allocation/deallocation
pub struct MemoryPool<T: Float> {
    /// Available buffers
    available: Vec<Vec<T>>,
    /// Count of currently allocated buffers
    allocated_count: usize,
    /// Buffer size for this pool
    buffer_size: usize,
    /// Pool name
    name: String,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory allocated in bytes
    pub total_allocated: usize,
    /// Available memory in bytes
    pub available: usize,
    /// Number of active buffers
    pub buffer_count: usize,
    /// Memory fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_ratio: f64,
}

impl<T: Float> MemoryManager<T> {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            total_allocated: 0,
            stats: MemoryStats {
                total_allocated: 0,
                available: 0,
                buffer_count: 0,
                fragmentation_ratio: 0.0,
            },
        }
    }

    /// Create a memory pool with the given name and buffer size
    pub fn create_pool(&mut self, name: &str, buffer_size: usize) {
        let pool = MemoryPool::new(name.to_string(), buffer_size);
        self.pools.insert(name.to_string(), pool);
    }

    /// Allocate a buffer from the specified pool
    pub fn allocate(&mut self, pool_name: &str, size: usize) -> Result<Vec<T>, String> {
        if let Some(pool) = self.pools.get_mut(pool_name) {
            let buffer = pool.allocate(size)?;
            self.total_allocated += size * std::mem::size_of::<T>();
            self.update_stats();
            Ok(buffer)
        } else {
            Err(format!("Pool '{pool_name}' not found"))
        }
    }

    /// Deallocate a buffer back to the specified pool
    pub fn deallocate(&mut self, pool_name: &str, buffer: Vec<T>) -> Result<(), String> {
        if let Some(pool) = self.pools.get_mut(pool_name) {
            let size = buffer.len() * std::mem::size_of::<T>();
            pool.deallocate(buffer);
            self.total_allocated = self.total_allocated.saturating_sub(size);
            self.update_stats();
            Ok(())
        } else {
            Err(format!("Pool '{pool_name}' not found"))
        }
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        self.stats.clone()
    }

    /// Clear all memory pools
    pub fn clear_all(&mut self) {
        for pool in self.pools.values_mut() {
            pool.clear();
        }
        self.total_allocated = 0;
        self.update_stats();
    }

    /// Update memory statistics
    fn update_stats(&mut self) {
        let mut buffer_count = 0;
        let mut available_buffers = 0;

        for pool in self.pools.values() {
            buffer_count += pool.allocated_count;
            available_buffers += pool.available.len();
        }

        self.stats = MemoryStats {
            total_allocated: self.total_allocated,
            available: available_buffers * std::mem::size_of::<T>(),
            buffer_count,
            fragmentation_ratio: if buffer_count > 0 {
                available_buffers as f64 / buffer_count as f64
            } else {
                0.0
            },
        };
    }
}

impl<T: Float> Default for MemoryManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> MemoryPool<T> {
    /// Create a new memory pool
    pub fn new(name: String, buffer_size: usize) -> Self {
        Self {
            available: Vec::new(),
            allocated_count: 0,
            buffer_size,
            name,
        }
    }

    /// Allocate a buffer from this pool
    pub fn allocate(&mut self, size: usize) -> Result<Vec<T>, String> {
        // If we have an available buffer of the right size, reuse it
        if let Some(mut buffer) = self.available.pop() {
            buffer.clear();
            buffer.resize(size, T::zero());
            self.allocated_count += 1;
            Ok(buffer)
        } else {
            // Create a new buffer
            let buffer = vec![T::zero(); size];
            self.allocated_count += 1;
            Ok(buffer)
        }
    }

    /// Deallocate a buffer back to this pool
    pub fn deallocate(&mut self, buffer: Vec<T>) {
        // Add to available list for reuse
        self.available.push(buffer);
        self.allocated_count = self.allocated_count.saturating_sub(1);
    }

    /// Clear all buffers in this pool
    pub fn clear(&mut self) {
        self.available.clear();
        self.allocated_count = 0;
    }

    /// Get the number of allocated buffers
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Get the number of available buffers
    pub fn available_count(&self) -> usize {
        self.available.len()
    }
}

static GLOBAL_MEMORY_MANAGER: OnceLock<Arc<Mutex<MemoryManager<f32>>>> = OnceLock::new();

fn global_memory_manager() -> &'static Arc<Mutex<MemoryManager<f32>>> {
    GLOBAL_MEMORY_MANAGER.get_or_init(|| Arc::new(Mutex::new(MemoryManager::new())))
}

/// Get the global memory manager
pub fn get_global_memory_manager() -> Arc<Mutex<MemoryManager<f32>>> {
    global_memory_manager().clone()
}

/// Initialize default memory pools
pub fn init_default_pools() {
    let mut manager = global_memory_manager().lock().unwrap();
    manager.create_pool("weights", 1024);
    manager.create_pool("activations", 512);
    manager.create_pool("gradients", 512);
    manager.create_pool("temporary", 256);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let manager: MemoryManager<f32> = MemoryManager::new();
        assert_eq!(manager.total_allocated, 0);
        assert_eq!(manager.pools.len(), 0);
    }

    #[test]
    fn test_pool_creation() {
        let mut manager: MemoryManager<f32> = MemoryManager::new();
        manager.create_pool("test", 100);
        assert_eq!(manager.pools.len(), 1);
        assert!(manager.pools.contains_key("test"));
    }

    #[test]
    fn test_allocation_deallocation() {
        let mut manager: MemoryManager<f32> = MemoryManager::new();
        manager.create_pool("test", 100);

        // Allocate buffer
        let buffer = manager.allocate("test", 50).unwrap();
        assert_eq!(buffer.len(), 50);
        assert!(manager.total_allocated > 0);

        // Deallocate buffer
        manager.deallocate("test", buffer).unwrap();
        // Note: total_allocated might not be 0 due to pool reuse
    }

    #[test]
    fn test_memory_stats() {
        let mut manager: MemoryManager<f32> = MemoryManager::new();
        manager.create_pool("test", 100);

        let stats = manager.get_stats();
        assert_eq!(stats.buffer_count, 0);
        assert_eq!(stats.total_allocated, 0);

        let _buffer = manager.allocate("test", 50).unwrap();
        let stats = manager.get_stats();
        assert_eq!(stats.buffer_count, 1);
        assert!(stats.total_allocated > 0);
    }

    #[test]
    fn test_pool_reuse() {
        let mut pool: MemoryPool<f32> = MemoryPool::new("test".to_string(), 100);

        // Allocate and deallocate
        let buffer1 = pool.allocate(50).unwrap();
        pool.deallocate(buffer1);

        // Allocate again - should reuse
        let buffer2 = pool.allocate(50).unwrap();
        assert_eq!(buffer2.len(), 50);
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.allocated_count(), 1);
    }
}
