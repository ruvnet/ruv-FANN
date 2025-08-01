//! Memory pool for efficient agent allocation and deallocation
//!
//! This module provides a memory pooling system to reduce allocation overhead
//! when spawning and releasing agents, critical for meeting sub-100ms spawn times.

use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

/// Memory pool for efficient memory management
#[wasm_bindgen]
pub struct MemoryPool {
    free_blocks: VecDeque<Vec<u8>>,
    block_size: usize,
    max_blocks: usize,
    total_allocated: usize,
    reuse_count: usize,
}

#[wasm_bindgen]
impl MemoryPool {
    /// Create a new memory pool with specified block size and maximum blocks
    #[wasm_bindgen(constructor)]
    pub fn new(block_size: usize, max_blocks: usize) -> Self {
        Self {
            free_blocks: VecDeque::with_capacity(max_blocks),
            block_size,
            max_blocks,
            total_allocated: 0,
            reuse_count: 0,
        }
    }

    /// Allocate a memory block from the pool
    pub fn allocate(&mut self) -> Option<Vec<u8>> {
        if let Some(block) = self.free_blocks.pop_front() {
            self.reuse_count += 1;
            Some(block)
        } else if self.total_allocated < self.max_blocks {
            self.total_allocated += 1;
            Some(vec![0u8; self.block_size])
        } else {
            None
        }
    }

    /// Return a memory block to the pool for reuse
    pub fn deallocate(&mut self, mut block: Vec<u8>) {
        if self.free_blocks.len() < self.max_blocks && block.len() == self.block_size {
            // Clear the block before returning to pool
            block.fill(0);
            self.free_blocks.push_back(block);
        }
    }

    /// Get the number of available blocks in the pool
    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.total_allocated * self.block_size
    }

    /// Get pool efficiency metrics
    pub fn get_metrics(&self) -> PoolMetrics {
        PoolMetrics {
            total_blocks: self.total_allocated as u32,
            free_blocks: self.free_blocks.len() as u32,
            block_size: self.block_size as u32,
            reuse_count: self.reuse_count as u32,
            memory_usage_mb: (self.memory_usage() as f32) / (1024.0 * 1024.0),
        }
    }
}

/// Pool metrics for monitoring
#[wasm_bindgen]
pub struct PoolMetrics {
    pub total_blocks: u32,
    pub free_blocks: u32,
    pub block_size: u32,
    pub reuse_count: u32,
    pub memory_usage_mb: f32,
}

/// Agent memory pool specifically optimized for neural network agents
#[wasm_bindgen]
pub struct AgentMemoryPool {
    // Different pools for different agent sizes
    small_pool: MemoryPool,  // 64KB blocks for simple agents
    medium_pool: MemoryPool, // 256KB blocks for standard agents
    large_pool: MemoryPool,  // 1MB blocks for complex agents
}

#[wasm_bindgen]
impl AgentMemoryPool {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            small_pool: MemoryPool::new(64 * 1024, 50), // 50 x 64KB = 3.2MB max
            medium_pool: MemoryPool::new(256 * 1024, 30), // 30 x 256KB = 7.5MB max
            large_pool: MemoryPool::new(1024 * 1024, 10), // 10 x 1MB = 10MB max
        }
    }

    /// Allocate memory for an agent based on complexity
    pub fn allocate_for_agent(&mut self, complexity: &str) -> Option<Vec<u8>> {
        match complexity {
            "simple" => self.small_pool.allocate(),
            "standard" => self.medium_pool.allocate(),
            "complex" => self.large_pool.allocate(),
            _ => self.medium_pool.allocate(),
        }
    }

    /// Return agent memory to the appropriate pool
    pub fn deallocate_agent_memory(&mut self, memory: Vec<u8>) {
        match memory.len() {
            65536 => self.small_pool.deallocate(memory),   // 64KB
            262144 => self.medium_pool.deallocate(memory), // 256KB
            1048576 => self.large_pool.deallocate(memory), // 1MB
            _ => {}                                        // Ignore non-standard sizes
        }
    }

    /// Get total memory usage across all pools
    pub fn total_memory_usage_mb(&self) -> f32 {
        let total = self.small_pool.memory_usage()
            + self.medium_pool.memory_usage()
            + self.large_pool.memory_usage();
        (total as f32) / (1024.0 * 1024.0)
    }

    /// Check if memory usage is within target (< 50MB for 10 agents)
    pub fn is_within_memory_target(&self) -> bool {
        self.total_memory_usage_mb() < 50.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = MemoryPool::new(1024, 5);

        // Allocate blocks
        let block1 = pool.allocate().unwrap();
        assert_eq!(block1.len(), 1024);
        assert_eq!(pool.available_blocks(), 0);

        // Return block to pool
        pool.deallocate(block1);
        assert_eq!(pool.available_blocks(), 1);

        // Reuse should increment counter
        let _block2 = pool.allocate().unwrap();
        assert_eq!(pool.get_metrics().reuse_count, 1);
    }

    #[test]
    fn test_agent_memory_pool() {
        let mut agent_pool = AgentMemoryPool::new();

        // Test different complexity allocations
        let simple = agent_pool.allocate_for_agent("simple").unwrap();
        assert_eq!(simple.len(), 64 * 1024);

        let standard = agent_pool.allocate_for_agent("standard").unwrap();
        assert_eq!(standard.len(), 256 * 1024);

        let complex = agent_pool.allocate_for_agent("complex").unwrap();
        assert_eq!(complex.len(), 1024 * 1024);

        // Return and verify memory target
        agent_pool.deallocate_agent_memory(simple);
        agent_pool.deallocate_agent_memory(standard);
        agent_pool.deallocate_agent_memory(complex);

        assert!(agent_pool.is_within_memory_target());
    }
}
