//! Checked ownership for the CPU runtime's raw allocations.

use crate::Result;
use std::alloc::Layout;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn allocations() -> &'static Mutex<HashMap<usize, Layout>> {
    static ALLOCATIONS: OnceLock<Mutex<HashMap<usize, Layout>>> = OnceLock::new();
    ALLOCATIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Allocate zero-initialized CPU memory for kernel data, aligned for `f64`.
///
/// Release the returned allocation with `free`; a zero-byte request returns null.
pub fn allocate(size: usize) -> Result<*mut u8> {
    if size == 0 {
        return Ok(std::ptr::null_mut());
    }
    let layout = Layout::from_size_align(size, std::mem::align_of::<f64>())
        .map_err(|e| crate::runtime_error!("Invalid allocation layout: {}", e))?;
    let mut registry = allocations()
        .lock()
        .map_err(|_| crate::runtime_error!("Allocation registry lock poisoned"))?;
    // SAFETY: the checked layout has nonzero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(crate::runtime_error!(
            "Memory allocation failed for {} bytes",
            size
        ));
    }
    registry.insert(ptr as usize, layout);
    Ok(ptr)
}

/// Copy raw CPU memory between non-overlapping buffers.
///
/// # Safety
/// For nonzero size, `src` and `dst` must be valid for reading/writing `size`
/// bytes, non-overlapping and not concurrently accessed. The destination must
/// not expose invalid values through safe typed access after the copy.
pub unsafe fn copy(dst: *mut u8, src: *const u8, size: usize) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    if dst.is_null() || src.is_null() {
        return Err(crate::runtime_error!("Null pointer in memory copy"));
    }
    // SAFETY: guaranteed by the caller's contract.
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Release an allocation created by `allocate` exactly once.
///
/// Returns an error for unknown or already-freed pointers. Null is a no-op.
pub fn free(ptr: *mut u8) -> Result<()> {
    if ptr.is_null() {
        return Ok(());
    }
    let mut registry = allocations()
        .lock()
        .map_err(|_| crate::runtime_error!("Allocation registry lock poisoned"))?;
    let layout = registry
        .remove(&(ptr as usize))
        .ok_or_else(|| crate::runtime_error!("Unknown or already freed runtime allocation"))?;
    // SAFETY: this exact base pointer and layout came from allocate; removal
    // under the registry lock prevents double frees through this API.
    unsafe {
        std::alloc::dealloc(ptr, layout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_copy_and_checked_free() {
        let ptr = allocate(16).unwrap();
        let mut out = [1u8; 16];
        // SAFETY: both allocations are valid, disjoint, and live for this copy.
        unsafe {
            copy(out.as_mut_ptr(), ptr, out.len()).unwrap();
        }
        assert_eq!(out, [0u8; 16]);
        free(ptr).unwrap();
        assert!(free(ptr).is_err());
        assert!(free(std::ptr::NonNull::<u8>::dangling().as_ptr()).is_err());
        free(std::ptr::null_mut()).unwrap();
    }
}
