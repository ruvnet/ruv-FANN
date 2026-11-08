//! Grid and block dimension types

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 3D dimension type (similar to CUDA's dim3)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    /// Create a new Dim3
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
    
    /// Create a 1D dimension
    pub fn one_d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }
    
    /// Create a 2D dimension
    pub fn two_d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }
    
    /// Get total number of elements
    pub fn size(&self) -> u32 {
        self.x.saturating_mul(self.y).saturating_mul(self.z)
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self::one_d(x)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Self::two_d(x, y)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::new(x, y, z)
    }
}

/// Grid configuration for kernel launch
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    pub dim: Dim3,
}

impl Grid {
    /// Create a new grid configuration
    pub fn new<D: Into<Dim3>>(dim: D) -> Self {
        Self { dim: dim.into() }
    }
    
    /// Get total number of blocks
    pub fn num_blocks(&self) -> u32 {
        self.dim.size()
    }
}

/// Block configuration for kernel launch
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub dim: Dim3,
}

impl Block {
    /// Create a new block configuration
    pub fn new<D: Into<Dim3>>(dim: D) -> Self {
        Self { dim: dim.into() }
    }
    
    /// Get total number of threads per block
    pub fn num_threads(&self) -> u32 {
        self.dim.size()
    }
    
    /// Validate block dimensions against hardware limits
    pub fn validate(&self) -> crate::Result<()> {
        // Typical CUDA limits
        const MAX_THREADS_PER_BLOCK: u32 = 1024;
        const MAX_BLOCK_DIM_X: u32 = 1024;
        const MAX_BLOCK_DIM_Y: u32 = 1024;
        const MAX_BLOCK_DIM_Z: u32 = 64;
        
        if self.dim.x == 0 || self.dim.y == 0 || self.dim.z == 0 {
            return Err(crate::runtime_error!("Block dimensions must be nonzero"));
        }
        if self.num_threads() > MAX_THREADS_PER_BLOCK {
            return Err(crate::runtime_error!(
                "Block size {} exceeds maximum threads per block {}",
                self.num_threads(),
                MAX_THREADS_PER_BLOCK
            ));
        }
        
        if self.dim.x > MAX_BLOCK_DIM_X {
            return Err(crate::runtime_error!(
                "Block x dimension {} exceeds maximum {}",
                self.dim.x,
                MAX_BLOCK_DIM_X
            ));
        }
        
        if self.dim.y > MAX_BLOCK_DIM_Y {
            return Err(crate::runtime_error!(
                "Block y dimension {} exceeds maximum {}",
                self.dim.y,
                MAX_BLOCK_DIM_Y
            ));
        }
        
        if self.dim.z > MAX_BLOCK_DIM_Z {
            return Err(crate::runtime_error!(
                "Block z dimension {} exceeds maximum {}",
                self.dim.z,
                MAX_BLOCK_DIM_Z
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dim3_creation() {
        let d1 = Dim3::one_d(256);
        assert_eq!(d1, Dim3 { x: 256, y: 1, z: 1 });
        assert_eq!(d1.size(), 256);
        
        let d2 = Dim3::two_d(16, 16);
        assert_eq!(d2, Dim3 { x: 16, y: 16, z: 1 });
        assert_eq!(d2.size(), 256);
        
        let d3 = Dim3::new(8, 8, 4);
        assert_eq!(d3.size(), 256);
    }
    
    #[test]
    fn test_block_validation() {
        let valid_block = Block::new(256);
        assert!(valid_block.validate().is_ok());
        
        let invalid_block = Block::new(2048);
        assert!(invalid_block.validate().is_err());
    }
}