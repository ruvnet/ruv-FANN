//! WASM runtime backend implementation

use super::backend_trait::{BackendCapabilities, BackendTrait, MemcpyKind};
use crate::{runtime_error, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// A private, exclusively created module file, removed on every exit path.
struct ModuleFile(std::path::PathBuf);

impl ModuleFile {
    fn create(bytes: &[u8]) -> Result<Self> {
        use std::io::Write;
        for _ in 0..32 {
            let path = std::env::temp_dir().join(format!(
                "cuda_wasm_module_{}_{:016x}.wasm", std::process::id(), rand::random::<u64>()
            ));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = match options.open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(runtime_error!("Cannot create module file: {}", error)),
            };
            let owned = Self(path);
            file.write_all(bytes).map_err(|e| runtime_error!("Cannot write module file: {}", e))?;
            return Ok(owned);
        }
        Err(runtime_error!("Cannot create a unique module file"))
    }
}

impl Drop for ModuleFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

/// CPU-based runtime backend for WASM environments.
///
/// Supports:
/// - Heap-based memory allocation with tracking for proper deallocation
/// - WASM module handles (format prefixes checked; external runtime validates)
/// - Kernel execution via external `wasmtime` or `wasmer` runtimes
pub struct WasmRuntime {
    capabilities: BackendCapabilities,
    /// Tracks allocated memory: pointer address -> allocation size
    allocations: Mutex<HashMap<usize, usize>>,
    /// Stores compiled WASM modules (binary bytecode)
    compiled_modules: Mutex<Vec<Vec<u8>>>,
}

impl Default for WasmRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Alignment used for all heap allocations in this backend.
const ALLOC_ALIGN: usize = 8;

impl WasmRuntime {
    /// Create a new WASM runtime backend
    pub fn new() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        Self {
            capabilities: BackendCapabilities {
                name: "WASM Runtime".to_string(),
                supports_cuda: false,
                supports_opencl: false,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: num_cpus,
                max_threads_per_block: num_cpus,
                max_blocks_per_grid: 1024,
                max_shared_memory: 64 * 1024, // 64 KB shared memory
                supports_dynamic_parallelism: false,
                supports_unified_memory: true,
                max_grid_dim: [1024, 1024, 1],
                max_block_dim: [num_cpus, 1, 1],
                warp_size: 1,
            },
            allocations: Mutex::new(HashMap::new()),
            compiled_modules: Mutex::new(Vec::new()),
        }
    }

    /// Return the number of currently tracked allocations.
    #[cfg(test)]
    fn allocation_count(&self) -> usize {
        self.allocations.lock().len()
    }

    /// Return the number of compiled modules stored.
    #[cfg(test)]
    fn module_count(&self) -> usize {
        self.compiled_modules.lock().len()
    }

    /// Detect an available WASM runtime binary on the system.
    ///
    /// Checks for `wasmtime` first, then `wasmer`. Returns the binary name
    /// if found, or `None` if neither is available.
    fn detect_wasm_runtime() -> Option<&'static str> {
        if std::process::Command::new("wasmtime")
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some("wasmtime");
        }
        if std::process::Command::new("wasmer")
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some("wasmer");
        }
        None
    }
}

#[async_trait(?Send)]
impl BackendTrait for WasmRuntime {
    fn name(&self) -> &str {
        &self.capabilities.name
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    async fn initialize(&mut self) -> Result<()> {
        // No special initialization needed for WASM runtime
        Ok(())
    }

    /// Compile a WASM kernel from source.
    ///
    /// Accepts either:
    /// - Raw WASM binary (must start with `\0asm` magic bytes)
    /// - WAT text format (must start with `(module`)
    ///
    /// Returns a 4-byte little-endian module index that can be passed to
    /// `launch_kernel`.
    async fn compile_kernel(&self, source: &str) -> Result<Vec<u8>> {
        let bytes = source.as_bytes();

        if bytes.len() >= 4 && &bytes[0..4] == b"\0asm" {
            // Binary format candidate; the external runtime validates the module.
            let mut modules = self.compiled_modules.lock();
            let index = modules.len();
            modules.push(bytes.to_vec());
            Ok((index as u32).to_le_bytes().to_vec())
        } else if source.trim_start().starts_with("(module") {
            // WAT text format -- store the raw text bytes; a full implementation
            // would convert WAT to WASM here, but we store as-is for the runtime
            // to handle.
            let mut modules = self.compiled_modules.lock();
            let index = modules.len();
            modules.push(bytes.to_vec());
            Ok((index as u32).to_le_bytes().to_vec())
        } else {
            Err(runtime_error!(
                "Invalid WASM module: expected WASM binary (\\0asm magic) or WAT text ((module prefix)"
            ))
        }
    }

    /// Launch a compiled WASM kernel.
    ///
    /// The `kernel` parameter must be a 4-byte little-endian module index
    /// previously returned by `compile_kernel`. The module is written to a
    /// temporary file and executed via an external runtime (`wasmtime` or
    /// `wasmer`).
    async unsafe fn launch_kernel(
        &self,
        kernel: &[u8],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*const u8],
    ) -> Result<()> {
        if grid != (1, 1, 1) || block != (1, 1, 1) || !args.is_empty() {
            return Err(runtime_error!(
                "External WASM runtime supports only a single invocation without kernel arguments"
            ));
        }
        if kernel.len() != 4 {
            return Err(runtime_error!(
                "Invalid kernel handle: expected 4-byte module index"
            ));
        }

        let index = u32::from_le_bytes([kernel[0], kernel[1], kernel[2], kernel[3]]) as usize;

        let module_bytes = {
            let modules = self.compiled_modules.lock();
            modules
                .get(index)
                .cloned()
                .ok_or_else(|| runtime_error!("Module index {} not found", index))?
        };

        let runtime_name = Self::detect_wasm_runtime().ok_or_else(|| {
            runtime_error!("No WASM runtime found (install wasmtime or wasmer)")
        })?;

        // Exclusive creation prevents following a pre-existing symlink. The
        // owner also removes the file on write, process-spawn or runtime errors.
        let module_file = ModuleFile::create(&module_bytes)?;
        let output = std::process::Command::new(runtime_name)
            .arg(&module_file.0)
            .output()
            .map_err(|e| runtime_error!("Failed to execute {}: {}", runtime_name, e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(runtime_error!(
                "{} execution failed (exit {}): {}",
                runtime_name,
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ));
        }

        Ok(())
    }

    /// Allocate memory on the heap.
    ///
    /// The allocation is tracked internally so that `free_memory` can correctly
    /// deallocate it. Returns an error for zero-sized allocations.
    fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Err(runtime_error!("Cannot allocate 0 bytes"));
        }

        let layout = std::alloc::Layout::from_size_align(size, ALLOC_ALIGN)
            .map_err(|e| runtime_error!("Invalid layout: {}", e))?;

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(runtime_error!("Failed to allocate {} bytes", size));
        }

        self.allocations.lock().insert(ptr as usize, size);

        Ok(ptr)
    }

    /// Free previously allocated memory.
    ///
    /// Looks up the pointer in the allocation tracking map, deallocates with the
    /// correct layout, and removes the entry. Returns an error if the pointer
    /// was not allocated by this backend.
    fn free_memory(&self, ptr: *mut u8) -> Result<()> {
        if ptr.is_null() {
            return Err(runtime_error!("Cannot free null pointer"));
        }

        let size = self
            .allocations
            .lock()
            .remove(&(ptr as usize))
            .ok_or_else(|| {
                runtime_error!(
                    "Pointer {:p} was not allocated by this backend",
                    ptr
                )
            })?;

        let layout = std::alloc::Layout::from_size_align(size, ALLOC_ALIGN)
            .map_err(|e| runtime_error!("Invalid layout during free: {}", e))?;

        unsafe {
            std::alloc::dealloc(ptr, layout);
        }

        Ok(())
    }

    /// Copy memory between buffers.
    ///
    /// Validates that neither pointer is null and that the size is non-zero
    /// before performing the copy.
    unsafe fn copy_memory(
        &self,
        dst: *mut u8,
        src: *const u8,
        size: usize,
        _kind: MemcpyKind,
    ) -> Result<()> {
        if dst.is_null() {
            return Err(runtime_error!("Destination pointer is null"));
        }
        if src.is_null() {
            return Err(runtime_error!("Source pointer is null"));
        }
        if size == 0 {
            return Err(runtime_error!("Copy size must be greater than 0"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // No-op for CPU backend -- all operations are synchronous
        Ok(())
    }
}

// SAFETY: WasmRuntime only contains a BackendCapabilities (owned data) and
// parking_lot::Mutex-wrapped collections which are Send + Sync.
unsafe impl Send for WasmRuntime {}
unsafe impl Sync for WasmRuntime {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runtime() -> WasmRuntime {
        WasmRuntime::new()
    }

    #[test]
    fn module_files_are_private_unique_and_removed_on_drop() {
        let first = ModuleFile::create(b"module-one").unwrap();
        let second = ModuleFile::create(b"module-two").unwrap();
        assert_ne!(first.0, second.0);
        assert_eq!(std::fs::read(&first.0).unwrap(), b"module-one");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&first.0).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let path = first.0.clone();
        drop(first);
        assert!(!path.exists());
    }

    // ── Capabilities ──────────────────────────────────────────────

    #[test]
    fn test_capabilities_reflect_cpu_count() {
        let rt = make_runtime();
        let caps = rt.capabilities();
        let expected = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);
        assert_eq!(caps.max_threads, expected);
        assert!(!caps.supports_cuda);
        assert!(caps.supports_unified_memory);
    }

    // ── Memory allocation ─────────────────────────────────────────

    #[test]
    fn test_allocate_and_free() {
        let rt = make_runtime();
        let ptr = rt.allocate_memory(1024).expect("allocation should succeed");
        assert!(!ptr.is_null());
        assert_eq!(rt.allocation_count(), 1);

        rt.free_memory(ptr).expect("free should succeed");
        assert_eq!(rt.allocation_count(), 0);
    }

    #[test]
    fn test_allocate_zero_bytes_fails() {
        let rt = make_runtime();
        assert!(rt.allocate_memory(0).is_err());
    }

    #[test]
    fn test_free_null_pointer_fails() {
        let rt = make_runtime();
        assert!(rt.free_memory(std::ptr::null_mut()).is_err());
    }

    #[test]
    fn test_free_unknown_pointer_fails() {
        let rt = make_runtime();
        // Fabricate a non-null pointer that was never allocated
        let fake: *mut u8 = 0xDEAD_BEEF as *mut u8;
        assert!(rt.free_memory(fake).is_err());
    }

    #[test]
    fn test_double_free_fails() {
        let rt = make_runtime();
        let ptr = rt.allocate_memory(64).unwrap();
        rt.free_memory(ptr).unwrap();
        // Second free should fail because the entry was removed
        assert!(rt.free_memory(ptr).is_err());
    }

    // ── Memory copy ───────────────────────────────────────────────

    #[test]
    fn test_copy_memory_roundtrip() {
        let rt = make_runtime();
        let src = rt.allocate_memory(4).unwrap();
        let dst = rt.allocate_memory(4).unwrap();

        unsafe {
            std::ptr::write_bytes(src, 0xAB, 4);
        }

        unsafe { rt.copy_memory(dst, src, 4, MemcpyKind::HostToHost) }
            .expect("copy should succeed");

        unsafe {
            for i in 0..4 {
                assert_eq!(*dst.add(i), 0xAB);
            }
        }

        rt.free_memory(src).unwrap();
        rt.free_memory(dst).unwrap();
    }

    #[test]
    fn test_copy_memory_null_dst_fails() {
        let rt = make_runtime();
        let src = rt.allocate_memory(4).unwrap();
        assert!(unsafe { rt
            .copy_memory(std::ptr::null_mut(), src, 4, MemcpyKind::HostToHost) }
            .is_err());
        rt.free_memory(src).unwrap();
    }

    #[test]
    fn test_copy_memory_null_src_fails() {
        let rt = make_runtime();
        let dst = rt.allocate_memory(4).unwrap();
        assert!(unsafe { rt
            .copy_memory(dst, std::ptr::null(), 4, MemcpyKind::HostToHost) }
            .is_err());
        rt.free_memory(dst).unwrap();
    }

    #[test]
    fn test_copy_memory_zero_size_fails() {
        let rt = make_runtime();
        let src = rt.allocate_memory(4).unwrap();
        let dst = rt.allocate_memory(4).unwrap();
        assert!(unsafe { rt
            .copy_memory(dst, src, 0, MemcpyKind::HostToHost) }
            .is_err());
        rt.free_memory(src).unwrap();
        rt.free_memory(dst).unwrap();
    }

    // ── Compile kernel ────────────────────────────────────────────

    #[tokio::test]
    async fn test_compile_wasm_binary() {
        let rt = make_runtime();
        // Minimal valid WASM header
        let wasm_source = "\0asm\x01\x00\x00\x00";
        let handle = rt.compile_kernel(wasm_source).await.unwrap();
        assert_eq!(handle.len(), 4);
        assert_eq!(rt.module_count(), 1);

        let index = u32::from_le_bytes([handle[0], handle[1], handle[2], handle[3]]);
        assert_eq!(index, 0);
    }

    #[tokio::test]
    async fn test_compile_wat_text() {
        let rt = make_runtime();
        let wat = "(module)";
        let handle = rt.compile_kernel(wat).await.unwrap();
        assert_eq!(handle.len(), 4);
        assert_eq!(rt.module_count(), 1);
    }

    #[tokio::test]
    async fn test_compile_invalid_source_fails() {
        let rt = make_runtime();
        assert!(rt.compile_kernel("not wasm at all").await.is_err());
    }

    #[tokio::test]
    async fn test_compile_multiple_modules() {
        let rt = make_runtime();
        let h1 = rt.compile_kernel("(module)").await.unwrap();
        let h2 = rt.compile_kernel("(module (func))").await.unwrap();

        let i1 = u32::from_le_bytes([h1[0], h1[1], h1[2], h1[3]]);
        let i2 = u32::from_le_bytes([h2[0], h2[1], h2[2], h2[3]]);
        assert_eq!(i1, 0);
        assert_eq!(i2, 1);
        assert_eq!(rt.module_count(), 2);
    }

    // ── Launch kernel (runtime detection) ─────────────────────────

    #[tokio::test]
    async fn test_launch_rejects_unsupported_grid_and_arguments() {
        let rt = WasmRuntime::new();
        let handle = rt.compile_kernel("(module)").await.unwrap();
        let argument = 7u8;
        let arguments = [&argument as *const u8];
        for (grid, block, args) in [
            ((2, 1, 1), (1, 1, 1), &[][..]),
            ((1, 1, 1), (2, 1, 1), &[][..]),
            ((1, 1, 1), (1, 1, 1), &arguments[..]),
        ] {
            // SAFETY: handle is owned by rt, and argument storage is live.
            // These unsupported shapes must fail before any execution.
            let error = unsafe { rt.launch_kernel(&handle, grid, block, args) }
                .await.unwrap_err();
            assert!(error.to_string().contains("single invocation without kernel arguments"));
        }
    }

    #[tokio::test]
    async fn test_launch_kernel_invalid_handle() {
        let rt = make_runtime();
        // Too-short handle
        let result = unsafe { rt
            .launch_kernel(&[0u8, 1], (1, 1, 1), (1, 1, 1), &[]) }
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_launch_kernel_missing_module() {
        let rt = make_runtime();
        // Module index 99 was never compiled
        let handle = 99u32.to_le_bytes();
        let result = unsafe { rt
            .launch_kernel(&handle, (1, 1, 1), (1, 1, 1), &[]) }
            .await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("not found"));
    }

    // ── Synchronize ───────────────────────────────────────────────

    #[test]
    fn test_synchronize() {
        let rt = make_runtime();
        assert!(rt.synchronize().is_ok());
    }

    // ── Default trait ─────────────────────────────────────────────

    #[test]
    fn test_default() {
        let rt = WasmRuntime::default();
        assert_eq!(rt.name(), "WASM Runtime");
    }
}
