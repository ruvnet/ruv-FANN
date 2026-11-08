//! Initialized, aligned host memory for CPU execution and staging transfers.
//!
//! These buffers use the system allocator; they are not CUDA page-locked memory.

use crate::{runtime_error, Result};

/// An owned host buffer whose elements are initialized before safe access.
pub struct HostBuffer<T> {
    data: Vec<T>,
}

impl<T: Copy> HostBuffer<T> {
    /// Allocate a buffer filled with `T::default()`.
    ///
    /// Returns an error for zero length, size overflow, or allocation failure.
    pub fn new(len: usize) -> Result<Self>
    where
        T: Default,
    {
        if len == 0 {
            return Err(runtime_error!("Cannot allocate zero-length buffer"));
        }
        std::alloc::Layout::array::<T>(len)
            .map_err(|e| runtime_error!("Invalid buffer layout: {}", e))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|e| runtime_error!("Host allocation failed: {}", e))?;
        data.resize(len, T::default());
        Ok(Self { data })
    }

    /// Get buffer length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow initialized elements.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow initialized elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Copy from a slice of the same length.
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len() {
            return Err(runtime_error!(
                "Source length {} doesn't match buffer length {}",
                src.len(),
                self.len()
            ));
        }
        self.data.copy_from_slice(src);
        Ok(())
    }

    /// Copy to a slice of the same length.
    pub fn copy_to_slice(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len() {
            return Err(runtime_error!(
                "Destination length {} doesn't match buffer length {}",
                dst.len(),
                self.len()
            ));
        }
        dst.copy_from_slice(&self.data);
        Ok(())
    }

    /// Fill the buffer without an intermediate allocation.
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T: Copy> std::ops::Index<usize> for HostBuffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: Copy> std::ops::IndexMut<usize> for HostBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_buffer_allocation() {
        let buffer = HostBuffer::<f32>::new(1024).unwrap();
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_host_buffer_copy() {
        let mut buffer = HostBuffer::<i32>::new(10).unwrap();
        let data: Vec<i32> = (0..10).collect();

        buffer.copy_from_slice(&data).unwrap();

        let mut result = vec![0; 10];
        buffer.copy_to_slice(&mut result).unwrap();

        assert_eq!(data, result);
    }

    #[test]
    fn test_host_buffer_fill() {
        let mut buffer = HostBuffer::<f64>::new(100).unwrap();
        buffer.fill(3.14);

        for i in 0..100 {
            assert_eq!(buffer[i], 3.14);
        }
    }
}
