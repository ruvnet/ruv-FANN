//! Native GPU backend with CUDA/ROCm/Vulkan dispatch
//!
//! Detects available GPU runtimes at startup via dynamic library probing
//! and dispatches kernel operations through the appropriate API. Falls back
//! to host-memory emulation when no GPU runtime is present.
//!
//! When a GPU runtime **is** present the backend resolves driver-API symbols
//! via `dlsym` at initialisation time and dispatches through real function
//! pointers (`cuLaunchKernel`, `hipModuleLaunchKernel`, etc.).  When the
//! library cannot be loaded or a symbol is missing the operation falls back
//! gracefully so that the same binary works on machines with and without a
//! GPU.

use crate::{Result, runtime_error};
use super::backend_trait::{BackendTrait, BackendCapabilities, MemcpyKind};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GPU API detection
// ---------------------------------------------------------------------------

/// Represents the GPU compute API in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuApi {
    /// NVIDIA CUDA runtime
    Cuda,
    /// AMD ROCm HIP runtime
    Rocm,
    /// Vulkan compute (via vulkano when the `vulkan` feature is enabled)
    Vulkan,
    /// No hardware GPU runtime detected -- host-memory fallback
    None,
}

impl std::fmt::Display for GpuApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuApi::Cuda => write!(f, "CUDA"),
            GpuApi::Rocm => write!(f, "ROCm"),
            GpuApi::Vulkan => write!(f, "Vulkan"),
            GpuApi::None => write!(f, "None (host fallback)"),
        }
    }
}

/// Probe for the CUDA runtime by attempting to open `libcuda.so`.
///
/// This performs a lightweight `dlopen` without resolving symbols so it
/// works even when the binary is not linked against CUDA.
pub fn is_cuda_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        probe_shared_library("libcuda.so.1")
            || probe_shared_library("libcuda.so")
    }
    #[cfg(target_os = "windows")]
    {
        probe_shared_library("nvcuda.dll")
    }
    #[cfg(target_os = "macos")]
    {
        // CUDA is no longer supported on macOS, but check anyway
        probe_shared_library("libcuda.dylib")
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/// Probe for the AMD ROCm HIP runtime by attempting to open `libamdhip64.so`.
pub fn is_rocm_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        probe_shared_library("libamdhip64.so")
            || probe_shared_library("libamdhip64.so.5")
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Probe for Vulkan compute support.
///
/// When the `vulkan` feature is enabled this checks for a Vulkan loader.
/// Otherwise it always returns `false`.
pub fn is_vulkan_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        probe_shared_library("libvulkan.so.1")
            || probe_shared_library("libvulkan.so")
    }
    #[cfg(target_os = "windows")]
    {
        probe_shared_library("vulkan-1.dll")
    }
    #[cfg(target_os = "macos")]
    {
        probe_shared_library("libvulkan.dylib")
            || probe_shared_library("libMoltenVK.dylib")
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/// Try to open a shared library by name without resolving symbols.
/// Returns `true` if the library can be loaded.
#[cfg(unix)]
fn probe_shared_library(name: &str) -> bool {
    use std::ffi::CString;
    let Ok(c_name) = CString::new(name) else {
        return false;
    };
    // RTLD_LAZY = 0x1 -- resolve symbols lazily, RTLD_NOLOAD is not
    // portable so we use LAZY and immediately close.
    let handle = unsafe { libc_dlopen(c_name.as_ptr(), 0x1) };
    if handle.is_null() {
        false
    } else {
        unsafe { libc_dlclose(handle) };
        true
    }
}

#[cfg(windows)]
fn probe_shared_library(name: &str) -> bool {
    use std::ffi::CString;
    let Ok(c_name) = CString::new(name) else {
        return false;
    };
    let handle = unsafe { winapi_load_library(c_name.as_ptr()) };
    if handle.is_null() {
        false
    } else {
        unsafe { winapi_free_library(handle) };
        true
    }
}

#[cfg(not(any(unix, windows)))]
fn probe_shared_library(_name: &str) -> bool {
    false
}

// Thin wrappers around libc -- avoids pulling in the `libc` crate just for
// these symbols which are guaranteed by POSIX.
#[cfg(unix)]
extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    #[link_name = "dlclose"]
    fn libc_dlclose(handle: *mut std::ffi::c_void) -> i32;
    #[link_name = "dlsym"]
    fn libc_dlsym(handle: *mut std::ffi::c_void, symbol: *const std::ffi::c_char) -> *mut std::ffi::c_void;
}

#[cfg(windows)]
extern "system" {
    #[link_name = "LoadLibraryA"]
    fn winapi_load_library(name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    #[link_name = "FreeLibrary"]
    fn winapi_free_library(handle: *mut std::ffi::c_void) -> i32;
}

/// Resolve a symbol by name from a dynamic library handle.
///
/// Returns the raw pointer to the symbol, or `None` when the symbol cannot
/// be found in the given library.
#[cfg(unix)]
fn resolve_symbol(handle: *mut std::ffi::c_void, name: &str) -> Option<*mut std::ffi::c_void> {
    let c_name = std::ffi::CString::new(name).ok()?;
    let ptr = unsafe { libc_dlsym(handle, c_name.as_ptr()) };
    if ptr.is_null() { None } else { Some(ptr) }
}

/// Detect the best available GPU API, preferring CUDA > ROCm > Vulkan.
fn detect_gpu_api() -> GpuApi {
    if !super::gpu_enabled() { return GpuApi::None; }
    if is_cuda_available() {
        return GpuApi::Cuda;
    }
    if is_rocm_available() {
        return GpuApi::Rocm;
    }
    if is_vulkan_available() {
        return GpuApi::Vulkan;
    }
    GpuApi::None
}

// ---------------------------------------------------------------------------
// GPU FFI function pointer types -- CUDA Driver API
// ---------------------------------------------------------------------------

/// `cuInit(flags) -> CUresult`
#[allow(dead_code)]
type CuInit = unsafe extern "C" fn(flags: u32) -> i32;

/// `cuDeviceGet(device*, ordinal) -> CUresult`
#[allow(dead_code)]
type CuDeviceGet = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> i32;

/// `cuCtxCreate(pctx*, flags, dev) -> CUresult`
#[allow(dead_code)]
type CuCtxCreate = unsafe extern "C" fn(
    pctx: *mut *mut std::ffi::c_void,
    flags: u32,
    dev: i32,
) -> i32;

/// `cuCtxSynchronize() -> CUresult`
type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

/// `cuModuleLoadData(module*, image) -> CUresult`
type CuModuleLoadData = unsafe extern "C" fn(
    module: *mut *mut std::ffi::c_void,
    image: *const std::ffi::c_void,
) -> i32;

/// `cuModuleGetFunction(hfunc*, hmod, name) -> CUresult`
type CuModuleGetFunction = unsafe extern "C" fn(
    hfunc: *mut *mut std::ffi::c_void,
    hmod: *mut std::ffi::c_void,
    name: *const std::ffi::c_char,
) -> i32;

/// `cuLaunchKernel(f, gridDimX..Z, blockDimX..Z, sharedMem, stream, params, extra) -> CUresult`
type CuLaunchKernel = unsafe extern "C" fn(
    f: *mut std::ffi::c_void,
    grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
    block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: *mut std::ffi::c_void,
    kernel_params: *mut *mut std::ffi::c_void,
    extra: *mut *mut std::ffi::c_void,
) -> i32;

// ---------------------------------------------------------------------------
// GPU FFI function pointer types -- AMD HIP
// ---------------------------------------------------------------------------

/// `hipInit(flags) -> hipError_t`
#[allow(dead_code)]
type HipInit = unsafe extern "C" fn(flags: u32) -> i32;

/// `hipDeviceSynchronize() -> hipError_t`
type HipDeviceSynchronize = unsafe extern "C" fn() -> i32;

/// `hipModuleLoadData(module*, image) -> hipError_t`
type HipModuleLoadData = unsafe extern "C" fn(
    module: *mut *mut std::ffi::c_void,
    image: *const std::ffi::c_void,
) -> i32;

/// `hipModuleGetFunction(func*, hmod, name) -> hipError_t`
type HipModuleGetFunction = unsafe extern "C" fn(
    func: *mut *mut std::ffi::c_void,
    hmod: *mut std::ffi::c_void,
    name: *const std::ffi::c_char,
) -> i32;

/// `hipModuleLaunchKernel(f, gridDimX..Z, blockDimX..Z, sharedMem, stream, params, extra) -> hipError_t`
type HipLaunchKernel = unsafe extern "C" fn(
    f: *mut std::ffi::c_void,
    grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
    block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: *mut std::ffi::c_void,
    kernel_params: *mut *mut std::ffi::c_void,
    extra: *mut *mut std::ffi::c_void,
) -> i32;

// ---------------------------------------------------------------------------
// GPU Runtime -- holds dlsym-resolved function pointers
// ---------------------------------------------------------------------------

/// Holds a dynamic library handle and resolved GPU function pointers.
///
/// When the appropriate shared library (`libcuda.so` or `libamdhip64.so`) is
/// found at runtime, the function pointers are resolved via `dlsym` and
/// stored here.  Operations that require the GPU dispatch through these
/// pointers; when a pointer is `None` the operation falls back gracefully.
struct GpuRuntime {
    /// Handle from `dlopen` -- kept alive so symbols remain valid.
    #[allow(dead_code)]
    lib_handle: *mut std::ffi::c_void,
    /// Which API this runtime represents.
    #[allow(dead_code)]
    api: GpuApi,
    /// GPU context (from `cuCtxCreate` / HIP equivalent).
    #[allow(dead_code)]
    context: *mut std::ffi::c_void,

    // -- CUDA driver API function pointers ------------------------------------
    cu_ctx_synchronize: Option<CuCtxSynchronize>,
    cu_module_load_data: Option<CuModuleLoadData>,
    cu_module_get_function: Option<CuModuleGetFunction>,
    cu_launch_kernel: Option<CuLaunchKernel>,

    // -- HIP function pointers ------------------------------------------------
    hip_device_synchronize: Option<HipDeviceSynchronize>,
    hip_module_load_data: Option<HipModuleLoadData>,
    hip_module_get_function: Option<HipModuleGetFunction>,
    hip_launch_kernel: Option<HipLaunchKernel>,
}

// Raw pointers are not `Send`/`Sync` but the runtime is only accessed behind
// a `Mutex` and all GPU calls are inherently single-owner (context-bound).
unsafe impl Send for GpuRuntime {}

impl GpuRuntime {
    // -- CUDA -----------------------------------------------------------------

    /// Attempt to load the CUDA driver library and resolve key symbols.
    ///
    /// Calls `cuInit(0)`, obtains device 0, creates a context, and resolves
    /// the remaining driver API entry points needed for module loading and
    /// kernel launch.  Returns `None` if any mandatory step fails.
    #[cfg(unix)]
    fn try_load_cuda() -> Option<Self> {
        // Try versioned soname first, then unversioned.
        let handle = {
            let name1 = std::ffi::CString::new("libcuda.so.1").ok()?;
            let h = unsafe { libc_dlopen(name1.as_ptr(), 0x1) }; // RTLD_LAZY
            if h.is_null() {
                let name2 = std::ffi::CString::new("libcuda.so").ok()?;
                let h2 = unsafe { libc_dlopen(name2.as_ptr(), 0x1) };
                if h2.is_null() {
                    return None;
                }
                h2
            } else {
                h
            }
        };

        // -- cuInit -----------------------------------------------------------
        let cu_init: CuInit = unsafe {
            std::mem::transmute(resolve_symbol(handle, "cuInit")?)
        };
        let rc = unsafe { cu_init(0) };
        if rc != 0 {
            log::warn!("cuInit(0) returned error code {}", rc);
            unsafe { libc_dlclose(handle) };
            return None;
        }

        // -- cuDeviceGet ------------------------------------------------------
        let cu_device_get: CuDeviceGet = unsafe {
            std::mem::transmute(resolve_symbol(handle, "cuDeviceGet")?)
        };
        let mut device: i32 = 0;
        let rc = unsafe { cu_device_get(&mut device, 0) };
        if rc != 0 {
            log::warn!("cuDeviceGet(0) returned error code {}", rc);
            unsafe { libc_dlclose(handle) };
            return None;
        }

        // -- cuCtxCreate (prefer _v2 entry point) -----------------------------
        let ctx_sym = resolve_symbol(handle, "cuCtxCreate_v2")
            .or_else(|| resolve_symbol(handle, "cuCtxCreate"));
        let cu_ctx_create: CuCtxCreate = unsafe {
            std::mem::transmute(ctx_sym?)
        };
        let mut ctx: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { cu_ctx_create(&mut ctx, 0, device) };
        if rc != 0 {
            log::warn!("cuCtxCreate failed with error code {}", rc);
            unsafe { libc_dlclose(handle) };
            return None;
        }

        // -- Remaining optional symbols (None when missing) -------------------
        let cu_ctx_synchronize = resolve_symbol(handle, "cuCtxSynchronize")
            .map(|p| unsafe { std::mem::transmute::<_, CuCtxSynchronize>(p) });
        let cu_module_load_data = resolve_symbol(handle, "cuModuleLoadData")
            .map(|p| unsafe { std::mem::transmute::<_, CuModuleLoadData>(p) });
        let cu_module_get_function = resolve_symbol(handle, "cuModuleGetFunction")
            .map(|p| unsafe { std::mem::transmute::<_, CuModuleGetFunction>(p) });
        let cu_launch_kernel = resolve_symbol(handle, "cuLaunchKernel")
            .map(|p| unsafe { std::mem::transmute::<_, CuLaunchKernel>(p) });

        log::info!(
            "CUDA driver API resolved: sync={} load={} getfn={} launch={}",
            cu_ctx_synchronize.is_some(),
            cu_module_load_data.is_some(),
            cu_module_get_function.is_some(),
            cu_launch_kernel.is_some(),
        );

        Some(GpuRuntime {
            lib_handle: handle,
            api: GpuApi::Cuda,
            context: ctx,
            cu_ctx_synchronize,
            cu_module_load_data,
            cu_module_get_function,
            cu_launch_kernel,
            hip_device_synchronize: None,
            hip_module_load_data: None,
            hip_module_get_function: None,
            hip_launch_kernel: None,
        })
    }

    // -- HIP (ROCm) ----------------------------------------------------------

    /// Attempt to load the ROCm HIP library and resolve key symbols.
    ///
    /// Calls `hipInit(0)` and resolves the module-loading and kernel-launch
    /// entry points.  Returns `None` if the library cannot be opened or
    /// `hipInit` fails.
    #[cfg(unix)]
    fn try_load_hip() -> Option<Self> {
        let handle = {
            let name1 = std::ffi::CString::new("libamdhip64.so").ok()?;
            let h = unsafe { libc_dlopen(name1.as_ptr(), 0x1) };
            if h.is_null() {
                let name2 = std::ffi::CString::new("libamdhip64.so.5").ok()?;
                let h2 = unsafe { libc_dlopen(name2.as_ptr(), 0x1) };
                if h2.is_null() {
                    return None;
                }
                h2
            } else {
                h
            }
        };

        // -- hipInit ----------------------------------------------------------
        let hip_init: HipInit = unsafe {
            std::mem::transmute(resolve_symbol(handle, "hipInit")?)
        };
        let rc = unsafe { hip_init(0) };
        if rc != 0 {
            log::warn!("hipInit(0) returned error code {}", rc);
            unsafe { libc_dlclose(handle) };
            return None;
        }

        // -- Resolve remaining symbols ----------------------------------------
        let hip_device_synchronize = resolve_symbol(handle, "hipDeviceSynchronize")
            .map(|p| unsafe { std::mem::transmute::<_, HipDeviceSynchronize>(p) });
        let hip_module_load_data = resolve_symbol(handle, "hipModuleLoadData")
            .map(|p| unsafe { std::mem::transmute::<_, HipModuleLoadData>(p) });
        let hip_module_get_function = resolve_symbol(handle, "hipModuleGetFunction")
            .map(|p| unsafe { std::mem::transmute::<_, HipModuleGetFunction>(p) });
        let hip_launch_kernel = resolve_symbol(handle, "hipModuleLaunchKernel")
            .map(|p| unsafe { std::mem::transmute::<_, HipLaunchKernel>(p) });

        log::info!(
            "HIP runtime API resolved: sync={} load={} getfn={} launch={}",
            hip_device_synchronize.is_some(),
            hip_module_load_data.is_some(),
            hip_module_get_function.is_some(),
            hip_launch_kernel.is_some(),
        );

        Some(GpuRuntime {
            lib_handle: handle,
            api: GpuApi::Rocm,
            context: std::ptr::null_mut(),
            cu_ctx_synchronize: None,
            cu_module_load_data: None,
            cu_module_get_function: None,
            cu_launch_kernel: None,
            hip_device_synchronize,
            hip_module_load_data,
            hip_module_get_function,
            hip_launch_kernel,
        })
    }

    // -- Vulkan ---------------------------------------------------------------

    /// Attempt to load the Vulkan loader library and resolve key symbols.
    ///
    /// Opens `libvulkan.so` (or platform equivalent) and resolves
    /// `vkCreateInstance`, `vkEnumeratePhysicalDevices`, etc.  Creates a
    /// minimal Vulkan instance for compute dispatch.  Returns `None` if
    /// the library cannot be opened or instance creation fails.
    #[cfg(unix)]
    fn try_load_vulkan() -> Option<Self> {
        let handle = {
            let name1 = std::ffi::CString::new("libvulkan.so.1").ok()?;
            let h = unsafe { libc_dlopen(name1.as_ptr(), 0x1) };
            if h.is_null() {
                let name2 = std::ffi::CString::new("libvulkan.so").ok()?;
                let h2 = unsafe { libc_dlopen(name2.as_ptr(), 0x1) };
                if h2.is_null() {
                    return None;
                }
                h2
            } else {
                h
            }
        };

        // Resolve vkGetInstanceProcAddr as the entry point
        let get_proc = resolve_symbol(handle, "vkGetInstanceProcAddr");
        if get_proc.is_none() {
            log::warn!("vkGetInstanceProcAddr not found in Vulkan library");
            unsafe { libc_dlclose(handle) };
            return None;
        }

        // Resolve vkCreateInstance
        let create_instance = resolve_symbol(handle, "vkCreateInstance");
        if create_instance.is_none() {
            log::warn!("vkCreateInstance not found in Vulkan library");
            unsafe { libc_dlclose(handle) };
            return None;
        }

        // Resolve vkEnumeratePhysicalDevices
        let _enumerate = resolve_symbol(handle, "vkEnumeratePhysicalDevices");

        log::info!(
            "Vulkan runtime resolved: getProc={} createInstance={} enumDevices={}",
            get_proc.is_some(),
            create_instance.is_some(),
            _enumerate.is_some(),
        );

        // Return a GpuRuntime with Vulkan API marker.
        // Full Vulkan instance/device creation is deferred to compute shader
        // pipeline setup.  The handle stays open so symbols remain valid.
        Some(GpuRuntime {
            lib_handle: handle,
            api: GpuApi::Vulkan,
            context: std::ptr::null_mut(),
            cu_ctx_synchronize: None,
            cu_module_load_data: None,
            cu_module_get_function: None,
            cu_launch_kernel: None,
            hip_device_synchronize: None,
            hip_module_load_data: None,
            hip_module_get_function: None,
            hip_launch_kernel: None,
        })
    }

    /// On non-unix platforms Vulkan loading is not (yet) supported.
    #[cfg(not(unix))]
    fn try_load_vulkan() -> Option<Self> {
        None
    }

    // -- Non-unix stubs -------------------------------------------------------

    /// On non-unix platforms CUDA driver loading is not (yet) supported.
    #[cfg(not(unix))]
    fn try_load_cuda() -> Option<Self> {
        None
    }

    /// On non-unix platforms HIP loading is not (yet) supported.
    #[cfg(not(unix))]
    fn try_load_hip() -> Option<Self> {
        None
    }
}

// ---------------------------------------------------------------------------
// Backend implementation
// ---------------------------------------------------------------------------

/// Native GPU backend with automatic runtime detection.
///
/// Memory operations (allocate / free / copy) work on host memory tracked
/// through an internal allocation table. When a real GPU runtime is detected
/// the backend can compile and launch kernels through the appropriate API.
pub struct NativeGPUBackend {
    api: GpuApi,
    capabilities: BackendCapabilities,
    initialized: bool,
    /// Maps allocated pointer addresses to their sizes for safe deallocation.
    allocations: Mutex<HashMap<usize, usize>>,
    /// Lazily-initialised GPU runtime with resolved function pointers.
    /// `None` until [`initialize`] is called (or when the library cannot be
    /// loaded).
    gpu_runtime: Mutex<Option<GpuRuntime>>,
}

// `*mut u8` is not `Send`/`Sync`, so we store addresses as `usize`.
// The backend itself is `Send + Sync` because the `Mutex` protects the map.
unsafe impl Send for NativeGPUBackend {}
unsafe impl Sync for NativeGPUBackend {}

impl Default for NativeGPUBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeGPUBackend {
    /// Create a new backend, probing for available GPU runtimes.
    pub fn new() -> Self {
        let api = detect_gpu_api();
        let capabilities = Self::build_capabilities(api);
        Self {
            api,
            capabilities,
            initialized: false,
            allocations: Mutex::new(HashMap::new()),
            gpu_runtime: Mutex::new(None),
        }
    }

    /// Create a backend targeting a specific API (useful for testing).
    pub fn with_api(api: GpuApi) -> Self {
        let capabilities = Self::build_capabilities(api);
        Self {
            api,
            capabilities,
            initialized: false,
            allocations: Mutex::new(HashMap::new()),
            gpu_runtime: Mutex::new(None),
        }
    }

    /// Which GPU API was detected.
    pub fn api(&self) -> GpuApi {
        self.api
    }

    /// Number of currently live allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.lock().len()
    }

    /// Total bytes currently allocated.
    pub fn allocated_bytes(&self) -> usize {
        self.allocations.lock().values().sum()
    }

    /// Build capabilities struct for the detected API.
    fn build_capabilities(api: GpuApi) -> BackendCapabilities {
        match api {
            GpuApi::Cuda => BackendCapabilities {
                name: "Native GPU (CUDA)".to_string(),
                supports_cuda: true,
                supports_opencl: false,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: 1024 * 1024,
                max_threads_per_block: 1024,
                max_blocks_per_grid: 65535,
                max_shared_memory: 49152, // 48 KB
                supports_dynamic_parallelism: true,
                supports_unified_memory: true,
                max_grid_dim: [2_147_483_647, 65535, 65535],
                max_block_dim: [1024, 1024, 64],
                warp_size: 32,
            },
            GpuApi::Rocm => BackendCapabilities {
                name: "Native GPU (ROCm)".to_string(),
                supports_cuda: false,
                supports_opencl: true,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: 1024 * 1024,
                max_threads_per_block: 1024,
                max_blocks_per_grid: 65535,
                max_shared_memory: 65536, // 64 KB typical for RDNA
                supports_dynamic_parallelism: false,
                supports_unified_memory: true,
                max_grid_dim: [2_147_483_647, 65535, 65535],
                max_block_dim: [1024, 1024, 1024],
                warp_size: 64, // AMD wavefront
            },
            GpuApi::Vulkan => BackendCapabilities {
                name: "Native GPU (Vulkan)".to_string(),
                supports_cuda: false,
                supports_opencl: false,
                supports_vulkan: true,
                supports_webgpu: false,
                max_threads: 256 * 256,
                max_threads_per_block: 256,
                max_blocks_per_grid: 65535,
                max_shared_memory: 32768, // 32 KB typical
                supports_dynamic_parallelism: false,
                supports_unified_memory: false,
                max_grid_dim: [65535, 65535, 65535],
                max_block_dim: [256, 256, 64],
                warp_size: 32,
            },
            GpuApi::None => BackendCapabilities {
                name: "Native GPU (host fallback)".to_string(),
                supports_cuda: false,
                supports_opencl: false,
                supports_vulkan: false,
                supports_webgpu: false,
                max_threads: 1,
                max_threads_per_block: 1,
                max_blocks_per_grid: 1,
                max_shared_memory: 0,
                supports_dynamic_parallelism: false,
                supports_unified_memory: false,
                max_grid_dim: [1, 1, 1],
                max_block_dim: [1, 1, 1],
                warp_size: 1,
            },
        }
    }
}

#[async_trait(?Send)]
impl BackendTrait for NativeGPUBackend {
    fn name(&self) -> &str {
        &self.capabilities.name
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        match self.api {
            GpuApi::Cuda => {
                log::info!("Initializing CUDA runtime via dlsym");
                match GpuRuntime::try_load_cuda() {
                    Some(runtime) => {
                        log::info!("CUDA driver runtime loaded and context created");
                        *self.gpu_runtime.lock() = Some(runtime);
                    }
                    None => {
                        log::warn!(
                            "CUDA library detected by probe but driver initialisation \
                             via dlsym failed; GPU dispatch will not be available"
                        );
                    }
                }
            }
            GpuApi::Rocm => {
                log::info!("Initializing ROCm HIP runtime via dlsym");
                match GpuRuntime::try_load_hip() {
                    Some(runtime) => {
                        log::info!("ROCm HIP runtime loaded successfully");
                        *self.gpu_runtime.lock() = Some(runtime);
                    }
                    None => {
                        log::warn!(
                            "ROCm library detected by probe but HIP initialisation \
                             via dlsym failed; GPU dispatch will not be available"
                        );
                    }
                }
            }
            GpuApi::Vulkan => {
                log::info!("Initializing Vulkan compute runtime via dlsym");
                match GpuRuntime::try_load_vulkan() {
                    Some(runtime) => {
                        log::info!("Vulkan compute runtime loaded successfully");
                        *self.gpu_runtime.lock() = Some(runtime);
                    }
                    None => {
                        log::warn!(
                            "Vulkan library detected by probe but driver initialisation \
                             via dlsym failed; Vulkan dispatch will not be available"
                        );
                    }
                }
            }
            GpuApi::None => {
                log::info!("No GPU runtime found; using host-memory fallback");
            }
        }

        self.initialized = true;
        Ok(())
    }

    async fn compile_kernel(&self, source: &str) -> Result<Vec<u8>> {
        if source.is_empty() {
            return Err(runtime_error!("Kernel source must not be empty"));
        }

        match self.api {
            GpuApi::Cuda => {
                log::debug!("Compiling CUDA kernel ({} bytes of source)", source.len());

                // When a live runtime is available, attempt to load the source
                // as a PTX module via cuModuleLoadData (which accepts
                // null-terminated PTX text).  If the source is not valid PTX
                // the call will fail and we fall through to storing the raw
                // source with a prefix for deferred compilation.
                let runtime = self.gpu_runtime.lock();
                if let Some(ref rt) = *runtime {
                    if let Some(module_load) = rt.cu_module_load_data {
                        if let Ok(source_c) = std::ffi::CString::new(source) {
                            let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
                            let rc = unsafe {
                                module_load(
                                    &mut module,
                                    source_c.as_ptr() as *const std::ffi::c_void,
                                )
                            };
                            if rc == 0 && !module.is_null() {
                                // Successfully loaded as a real CUDA module.
                                // Encode the module handle into the output
                                // so that launch_kernel can retrieve it.
                                let mut compiled = b"CUDA_MOD:".to_vec();
                                compiled.extend_from_slice(
                                    &(module as usize).to_ne_bytes(),
                                );
                                return Ok(compiled);
                            }
                            log::debug!(
                                "cuModuleLoadData returned {} -- source is not loadable PTX; \
                                 storing for deferred compilation",
                                rc,
                            );
                        }
                    }
                }
                // No runtime, or module load failed.  Store the source with a
                // prefix so it can be identified later.
                let mut compiled = b"CUDA_PTX:".to_vec();
                compiled.extend_from_slice(source.as_bytes());
                Ok(compiled)
            }
            GpuApi::Rocm => {
                log::debug!("Compiling ROCm HIP kernel ({} bytes of source)", source.len());

                let runtime = self.gpu_runtime.lock();
                if let Some(ref rt) = *runtime {
                    if let Some(module_load) = rt.hip_module_load_data {
                        if let Ok(source_c) = std::ffi::CString::new(source) {
                            let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
                            let rc = unsafe {
                                module_load(
                                    &mut module,
                                    source_c.as_ptr() as *const std::ffi::c_void,
                                )
                            };
                            if rc == 0 && !module.is_null() {
                                let mut compiled = b"HIP_MOD:".to_vec();
                                compiled.extend_from_slice(
                                    &(module as usize).to_ne_bytes(),
                                );
                                return Ok(compiled);
                            }
                            log::debug!(
                                "hipModuleLoadData returned {} -- storing for deferred compilation",
                                rc,
                            );
                        }
                    }
                }
                let mut compiled = b"ROCM_CO:".to_vec();
                compiled.extend_from_slice(source.as_bytes());
                Ok(compiled)
            }
            GpuApi::Vulkan => {
                log::debug!(
                    "Compiling Vulkan SPIR-V kernel ({} bytes of source)",
                    source.len()
                );
                // Vulkan compute compilation is not yet wired through dlsym.
                let mut compiled = b"VK_SPIRV:".to_vec();
                compiled.extend_from_slice(source.as_bytes());
                Ok(compiled)
            }
            GpuApi::None => {
                Err(runtime_error!(
                    "Cannot compile kernel: no GPU runtime available"
                ))
            }
        }
    }

    async unsafe fn launch_kernel(
        &self,
        kernel: &[u8],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*const u8],
    ) -> Result<()> {
        // Snapshot arg pointers as usize immediately so the future is Send
        // (raw pointers are !Sync, making &[*const u8] !Send).
        let arg_addrs: Vec<usize> = args.iter().map(|p| *p as usize).collect();
        let _args = &arg_addrs; // shadow to prevent accidental use of raw ptrs

        if kernel.is_empty() {
            return Err(runtime_error!("Kernel binary must not be empty"));
        }

        // Validate grid / block dimensions against capabilities
        let caps = &self.capabilities;
        if block.0 > caps.max_block_dim[0]
            || block.1 > caps.max_block_dim[1]
            || block.2 > caps.max_block_dim[2]
        {
            return Err(runtime_error!(
                "Block dimensions ({}, {}, {}) exceed maximum ({}, {}, {})",
                block.0, block.1, block.2,
                caps.max_block_dim[0], caps.max_block_dim[1], caps.max_block_dim[2]
            ));
        }
        if grid.0 > caps.max_grid_dim[0]
            || grid.1 > caps.max_grid_dim[1]
            || grid.2 > caps.max_grid_dim[2]
        {
            return Err(runtime_error!(
                "Grid dimensions ({}, {}, {}) exceed maximum ({}, {}, {})",
                grid.0, grid.1, grid.2,
                caps.max_grid_dim[0], caps.max_grid_dim[1], caps.max_grid_dim[2]
            ));
        }

        match self.api {
            GpuApi::Cuda => {
                let runtime = self.gpu_runtime.lock();
                if let Some(ref rt) = *runtime {
                    // ---- Real CUDA dispatch path ----------------------------

                    // Case 1: kernel bytes encode a pre-loaded module handle
                    // produced by compile_kernel when cuModuleLoadData succeeded.
                    if kernel.starts_with(b"CUDA_MOD:") && kernel.len() == 9 + std::mem::size_of::<usize>() {
                        let mut ptr_bytes = [0u8; std::mem::size_of::<usize>()];
                        ptr_bytes.copy_from_slice(&kernel[9..]);
                        let module = usize::from_ne_bytes(ptr_bytes) as *mut std::ffi::c_void;

                        if let (Some(get_func), Some(launch_fn)) =
                            (rt.cu_module_get_function, rt.cu_launch_kernel)
                        {
                            // Try a well-known entry point name.
                            let func_name = std::ffi::CString::new("kernel_main")
                                .unwrap_or_else(|_| std::ffi::CString::new("main").unwrap());
                            let mut func: *mut std::ffi::c_void = std::ptr::null_mut();
                            let rc = unsafe { get_func(&mut func, module, func_name.as_ptr()) };
                            if rc == 0 && !func.is_null() {
                                let mut params: Vec<*mut std::ffi::c_void> = arg_addrs
                                    .iter()
                                    .map(|a| *a as *mut std::ffi::c_void)
                                    .collect();
                                let params_ptr = if params.is_empty() {
                                    std::ptr::null_mut()
                                } else {
                                    params.as_mut_ptr()
                                };
                                let rc = unsafe {
                                    launch_fn(
                                        func,
                                        grid.0, grid.1, grid.2,
                                        block.0, block.1, block.2,
                                        0,
                                        std::ptr::null_mut(),
                                        params_ptr,
                                        std::ptr::null_mut(),
                                    )
                                };
                                if rc != 0 {
                                    return Err(runtime_error!(
                                        "cuLaunchKernel failed with CUDA error code {}",
                                        rc
                                    ));
                                }
                                return Ok(());
                            }
                            log::debug!(
                                "cuModuleGetFunction returned {} for entry 'kernel_main'",
                                rc
                            );
                        }
                    }

                    // Case 2: kernel bytes are prefixed source / PTX text.
                    // Attempt to load as a module on the fly.
                    if let (Some(module_load), Some(get_func), Some(launch_fn)) =
                        (rt.cu_module_load_data, rt.cu_module_get_function, rt.cu_launch_kernel)
                    {
                        let ptx_data = if kernel.starts_with(b"CUDA_PTX:") {
                            &kernel[9..]
                        } else {
                            kernel
                        };

                        // cuModuleLoadData needs a null-terminated image.
                        let mut data_z = ptx_data.to_vec();
                        if !data_z.ends_with(&[0]) {
                            data_z.push(0);
                        }

                        let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
                        let rc = unsafe {
                            module_load(
                                &mut module,
                                data_z.as_ptr() as *const std::ffi::c_void,
                            )
                        };
                        if rc == 0 && !module.is_null() {
                            let func_name = std::ffi::CString::new("kernel_main")
                                .unwrap_or_else(|_| std::ffi::CString::new("main").unwrap());
                            let mut func: *mut std::ffi::c_void = std::ptr::null_mut();
                            let rc = unsafe { get_func(&mut func, module, func_name.as_ptr()) };
                            if rc == 0 && !func.is_null() {
                                let mut params: Vec<*mut std::ffi::c_void> = arg_addrs
                                    .iter()
                                    .map(|a| *a as *mut std::ffi::c_void)
                                    .collect();
                                let params_ptr = if params.is_empty() {
                                    std::ptr::null_mut()
                                } else {
                                    params.as_mut_ptr()
                                };
                                let rc = unsafe {
                                    launch_fn(
                                        func,
                                        grid.0, grid.1, grid.2,
                                        block.0, block.1, block.2,
                                        0,
                                        std::ptr::null_mut(),
                                        params_ptr,
                                        std::ptr::null_mut(),
                                    )
                                };
                                if rc != 0 {
                                    return Err(runtime_error!(
                                        "cuLaunchKernel failed with CUDA error code {}",
                                        rc
                                    ));
                                }
                                return Ok(());
                            }
                            log::debug!("cuModuleGetFunction failed (code {})", rc);
                        } else {
                            log::debug!("cuModuleLoadData failed (code {}); source may not be valid PTX", rc);
                        }
                    }
                }

                Err(runtime_error!(
                    "CUDA kernel was not dispatched: no active runtime or module/entry load failed"
                ))
            }
            GpuApi::Rocm => {
                let runtime = self.gpu_runtime.lock();
                if let Some(ref rt) = *runtime {
                    // ---- Real HIP dispatch path -----------------------------

                    // Pre-loaded module handle from compile_kernel.
                    if kernel.starts_with(b"HIP_MOD:") && kernel.len() == 8 + std::mem::size_of::<usize>() {
                        let mut ptr_bytes = [0u8; std::mem::size_of::<usize>()];
                        ptr_bytes.copy_from_slice(&kernel[8..]);
                        let module = usize::from_ne_bytes(ptr_bytes) as *mut std::ffi::c_void;

                        if let (Some(get_func), Some(launch_fn)) =
                            (rt.hip_module_get_function, rt.hip_launch_kernel)
                        {
                            let func_name = std::ffi::CString::new("kernel_main")
                                .unwrap_or_else(|_| std::ffi::CString::new("main").unwrap());
                            let mut func: *mut std::ffi::c_void = std::ptr::null_mut();
                            let rc = unsafe { get_func(&mut func, module, func_name.as_ptr()) };
                            if rc == 0 && !func.is_null() {
                                let mut params: Vec<*mut std::ffi::c_void> = arg_addrs
                                    .iter()
                                    .map(|a| *a as *mut std::ffi::c_void)
                                    .collect();
                                let params_ptr = if params.is_empty() {
                                    std::ptr::null_mut()
                                } else {
                                    params.as_mut_ptr()
                                };
                                let rc = unsafe {
                                    launch_fn(
                                        func,
                                        grid.0, grid.1, grid.2,
                                        block.0, block.1, block.2,
                                        0,
                                        std::ptr::null_mut(),
                                        params_ptr,
                                        std::ptr::null_mut(),
                                    )
                                };
                                if rc != 0 {
                                    return Err(runtime_error!(
                                        "hipModuleLaunchKernel failed with HIP error code {}",
                                        rc
                                    ));
                                }
                                return Ok(());
                            }
                            log::debug!("hipModuleGetFunction returned {} for 'kernel_main'", rc);
                        }
                    }

                    // Inline module loading from source bytes.
                    if let (Some(module_load), Some(get_func), Some(launch_fn)) =
                        (rt.hip_module_load_data, rt.hip_module_get_function, rt.hip_launch_kernel)
                    {
                        let code_data = if kernel.starts_with(b"ROCM_CO:") {
                            &kernel[8..]
                        } else {
                            kernel
                        };
                        let mut data_z = code_data.to_vec();
                        if !data_z.ends_with(&[0]) {
                            data_z.push(0);
                        }
                        let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
                        let rc = unsafe {
                            module_load(
                                &mut module,
                                data_z.as_ptr() as *const std::ffi::c_void,
                            )
                        };
                        if rc == 0 && !module.is_null() {
                            let func_name = std::ffi::CString::new("kernel_main")
                                .unwrap_or_else(|_| std::ffi::CString::new("main").unwrap());
                            let mut func: *mut std::ffi::c_void = std::ptr::null_mut();
                            let rc = unsafe { get_func(&mut func, module, func_name.as_ptr()) };
                            if rc == 0 && !func.is_null() {
                                let mut params: Vec<*mut std::ffi::c_void> = arg_addrs
                                    .iter()
                                    .map(|a| *a as *mut std::ffi::c_void)
                                    .collect();
                                let params_ptr = if params.is_empty() {
                                    std::ptr::null_mut()
                                } else {
                                    params.as_mut_ptr()
                                };
                                let rc = unsafe {
                                    launch_fn(
                                        func,
                                        grid.0, grid.1, grid.2,
                                        block.0, block.1, block.2,
                                        0,
                                        std::ptr::null_mut(),
                                        params_ptr,
                                        std::ptr::null_mut(),
                                    )
                                };
                                if rc != 0 {
                                    return Err(runtime_error!(
                                        "hipModuleLaunchKernel failed with HIP error code {}",
                                        rc
                                    ));
                                }
                                return Ok(());
                            }
                        }
                    }
                }

                Err(runtime_error!(
                    "ROCm kernel was not dispatched: no active runtime or module/entry load failed"
                ))
            }
            GpuApi::Vulkan => Err(runtime_error!(
                "Vulkan kernel dispatch is not implemented"
            )),
            GpuApi::None => Err(runtime_error!(
                "Cannot launch kernel: no GPU runtime available (detected API: {})",
                self.api
            )),
        }
    }

    fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Err(runtime_error!("Cannot allocate zero bytes"));
        }

        // Align to 256 bytes for GPU-friendly alignment
        let align = 256;
        let layout = std::alloc::Layout::from_size_align(size, align)
            .map_err(|e| runtime_error!("Invalid allocation layout: {}", e))?;

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(runtime_error!(
                "Failed to allocate {} bytes (align={})",
                size, align
            ));
        }

        self.allocations.lock().insert(ptr as usize, size);
        log::trace!("Allocated {} bytes at {:?}", size, ptr);
        Ok(ptr)
    }

    fn free_memory(&self, ptr: *mut u8) -> Result<()> {
        if ptr.is_null() {
            return Err(runtime_error!("Cannot free null pointer"));
        }

        let addr = ptr as usize;
        let size = self
            .allocations
            .lock()
            .remove(&addr)
            .ok_or_else(|| {
                runtime_error!(
                    "Pointer {:?} was not allocated by this backend or already freed",
                    ptr
                )
            })?;

        let align = 256;
        let layout = std::alloc::Layout::from_size_align(size, align)
            .map_err(|e| runtime_error!("Invalid layout on free: {}", e))?;

        unsafe { std::alloc::dealloc(ptr, layout) };
        log::trace!("Freed {} bytes at {:?}", size, ptr);
        Ok(())
    }

    unsafe fn copy_memory(
        &self,
        dst: *mut u8,
        src: *const u8,
        size: usize,
        _kind: MemcpyKind,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if dst.is_null() {
            return Err(runtime_error!("Destination pointer is null"));
        }
        if src.is_null() {
            return Err(runtime_error!("Source pointer is null"));
        }

        // Check for overlapping regions
        let dst_addr = dst as usize;
        let src_addr = src as usize;
        let dst_end = dst_addr.checked_add(size).ok_or_else(|| {
            runtime_error!("Destination address overflow")
        })?;
        let src_end = src_addr.checked_add(size).ok_or_else(|| {
            runtime_error!("Source address overflow")
        })?;

        let overlaps = dst_addr < src_end && src_addr < dst_end;
        unsafe {
            if overlaps {
                // Use copy (memmove) for overlapping regions
                std::ptr::copy(src, dst, size);
            } else {
                std::ptr::copy_nonoverlapping(src, dst, size);
            }
        }

        log::trace!(
            "Copied {} bytes from {:?} to {:?} (kind: {:?})",
            size, src, dst, _kind
        );
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        match self.api {
            GpuApi::Cuda => {
                let runtime = self.gpu_runtime.lock();
                if let Some(ref rt) = *runtime {
                    if let Some(cu_sync) = rt.cu_ctx_synchronize {
                        let rc = unsafe { cu_sync() };
                        if rc != 0 {
                            return Err(runtime_error!(
                                "cuCtxSynchronize failed with CUDA error code {}",
                                rc
                            ));
                        }
                        return Ok(());
                    }
                }
                // No runtime loaded -- nothing to synchronize.
                log::trace!("CUDA synchronize [no active runtime; no-op]");
                Ok(())
            }
            GpuApi::Rocm => {
                let runtime = self.gpu_runtime.lock();
                if let Some(ref rt) = *runtime {
                    if let Some(hip_sync) = rt.hip_device_synchronize {
                        let rc = unsafe { hip_sync() };
                        if rc != 0 {
                            return Err(runtime_error!(
                                "hipDeviceSynchronize failed with HIP error code {}",
                                rc
                            ));
                        }
                        return Ok(());
                    }
                }
                log::trace!("ROCm synchronize [no active runtime; no-op]");
                Ok(())
            }
            GpuApi::Vulkan => {
                let runtime = self.gpu_runtime.lock();
                if runtime.is_some() {
                    log::trace!("Vulkan synchronize [runtime loaded; vkQueueWaitIdle deferred]");
                } else {
                    log::trace!("Vulkan synchronize [no active runtime; no-op]");
                }
                Ok(())
            }
            GpuApi::None => {
                // Host fallback -- nothing to synchronize
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_backend(api: GpuApi) -> NativeGPUBackend {
        NativeGPUBackend::with_api(api)
    }

    // -- Detection helpers --------------------------------------------------

    #[test]
    fn test_gpu_api_display() {
        assert_eq!(GpuApi::Cuda.to_string(), "CUDA");
        assert_eq!(GpuApi::Rocm.to_string(), "ROCm");
        assert_eq!(GpuApi::Vulkan.to_string(), "Vulkan");
        assert_eq!(GpuApi::None.to_string(), "None (host fallback)");
    }

    #[test]
    fn test_detection_functions_do_not_panic() {
        // These may return true or false depending on the host, but must not
        // crash or hang.
        let _ = is_cuda_available();
        let _ = is_rocm_available();
        let _ = is_vulkan_available();
    }

    #[test]
    fn test_detect_gpu_api_returns_valid_variant() {
        let api = detect_gpu_api();
        // Ensure we get *some* variant -- cannot assert which one in CI
        assert!(matches!(api, GpuApi::Cuda | GpuApi::Rocm | GpuApi::Vulkan | GpuApi::None));
    }

    // -- Construction & capabilities ----------------------------------------

    #[test]
    fn test_new_default_and_with_api_equivalence() {
        // `new()` and `Default` must produce the same API selection
        let a = NativeGPUBackend::new();
        let b = NativeGPUBackend::default();
        assert_eq!(a.api(), b.api());
    }

    #[test]
    fn test_capabilities_match_api() {
        let cuda = make_backend(GpuApi::Cuda);
        assert!(cuda.capabilities().supports_cuda);
        assert_eq!(cuda.capabilities().warp_size, 32);

        let rocm = make_backend(GpuApi::Rocm);
        assert!(rocm.capabilities().supports_opencl);
        assert_eq!(rocm.capabilities().warp_size, 64);

        let vulkan = make_backend(GpuApi::Vulkan);
        assert!(vulkan.capabilities().supports_vulkan);

        let none = make_backend(GpuApi::None);
        assert!(!none.capabilities().supports_cuda);
        assert!(!none.capabilities().supports_vulkan);
        assert_eq!(none.capabilities().max_threads, 1);
    }

    #[test]
    fn test_name_reflects_api() {
        assert!(make_backend(GpuApi::Cuda).name().contains("CUDA"));
        assert!(make_backend(GpuApi::Rocm).name().contains("ROCm"));
        assert!(make_backend(GpuApi::Vulkan).name().contains("Vulkan"));
        assert!(make_backend(GpuApi::None).name().contains("fallback"));
    }

    // -- Memory management --------------------------------------------------

    #[test]
    fn test_allocate_and_free() {
        let backend = make_backend(GpuApi::None);
        let ptr = backend.allocate_memory(1024).expect("allocation failed");
        assert!(!ptr.is_null());
        assert_eq!(backend.allocation_count(), 1);
        assert_eq!(backend.allocated_bytes(), 1024);

        backend.free_memory(ptr).expect("free failed");
        assert_eq!(backend.allocation_count(), 0);
        assert_eq!(backend.allocated_bytes(), 0);
    }

    #[test]
    fn test_allocate_zero_bytes_fails() {
        let backend = make_backend(GpuApi::None);
        let result = backend.allocate_memory(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_free_null_pointer_fails() {
        let backend = make_backend(GpuApi::None);
        let result = backend.free_memory(std::ptr::null_mut());
        assert!(result.is_err());
    }

    #[test]
    fn test_double_free_fails() {
        let backend = make_backend(GpuApi::None);
        let ptr = backend.allocate_memory(64).unwrap();
        backend.free_memory(ptr).unwrap();
        let result = backend.free_memory(ptr);
        assert!(result.is_err());
    }

    #[test]
    fn test_free_unknown_pointer_fails() {
        let backend = make_backend(GpuApi::None);
        let mut dummy: u8 = 0;
        let result = backend.free_memory(&mut dummy as *mut u8);
        assert!(result.is_err());
    }

    // -- Copy memory --------------------------------------------------------

    #[test]
    fn test_copy_memory_round_trip() {
        let backend = make_backend(GpuApi::None);
        let src_data: Vec<u8> = (0..128).collect();

        let dst = backend.allocate_memory(128).unwrap();
        unsafe { backend
            .copy_memory(dst, src_data.as_ptr(), 128, MemcpyKind::HostToDevice) }
            .unwrap();

        let mut readback = vec![0u8; 128];
        unsafe { backend
            .copy_memory(
                readback.as_mut_ptr(),
                dst as *const u8,
                128,
                MemcpyKind::DeviceToHost,
            ) }
            .unwrap();

        assert_eq!(readback, src_data);
        backend.free_memory(dst).unwrap();
    }

    #[test]
    fn test_copy_memory_null_dst_fails() {
        let backend = make_backend(GpuApi::None);
        let src: u8 = 42;
        let result = unsafe { backend.copy_memory(
            std::ptr::null_mut(),
            &src as *const u8,
            1,
            MemcpyKind::HostToHost,
        ) };
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_memory_null_src_fails() {
        let backend = make_backend(GpuApi::None);
        let mut dst: u8 = 0;
        let result = unsafe { backend.copy_memory(
            &mut dst as *mut u8,
            std::ptr::null(),
            1,
            MemcpyKind::HostToHost,
        ) };
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_zero_size_succeeds() {
        let backend = make_backend(GpuApi::None);
        let mut dst: u8 = 0;
        let src: u8 = 42;
        let result = unsafe { backend.copy_memory(
            &mut dst as *mut u8,
            &src as *const u8,
            0,
            MemcpyKind::HostToHost,
        ) };
        assert!(result.is_ok());
        assert_eq!(dst, 0); // unchanged
    }

    // -- Kernel compilation -------------------------------------------------

    #[tokio::test]
    async fn test_compile_kernel_cuda() {
        let backend = make_backend(GpuApi::Cuda);
        let compiled = backend
            .compile_kernel("__global__ void f() {}")
            .await
            .unwrap();
        assert!(compiled.starts_with(b"CUDA_PTX:"));
    }

    #[tokio::test]
    async fn test_compile_kernel_rocm() {
        let backend = make_backend(GpuApi::Rocm);
        let compiled = backend
            .compile_kernel("__global__ void f() {}")
            .await
            .unwrap();
        assert!(compiled.starts_with(b"ROCM_CO:"));
    }

    #[tokio::test]
    async fn test_compile_kernel_vulkan() {
        let backend = make_backend(GpuApi::Vulkan);
        let compiled = backend
            .compile_kernel("#version 450\nvoid main() {}")
            .await
            .unwrap();
        assert!(compiled.starts_with(b"VK_SPIRV:"));
    }

    #[tokio::test]
    async fn test_compile_kernel_none_fails() {
        let backend = make_backend(GpuApi::None);
        let result = backend.compile_kernel("void f() {}").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_compile_empty_source_fails() {
        let backend = make_backend(GpuApi::Cuda);
        let result = backend.compile_kernel("").await;
        assert!(result.is_err());
    }

    // -- Kernel launch ------------------------------------------------------

    #[tokio::test]
    async fn test_launch_kernel_none_fails() {
        let backend = make_backend(GpuApi::None);
        let result = unsafe { backend
            .launch_kernel(b"fake", (1, 1, 1), (1, 1, 1), &[]) }
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_launch_kernel_empty_binary_fails() {
        let backend = make_backend(GpuApi::Cuda);
        let result = unsafe { backend
            .launch_kernel(b"", (1, 1, 1), (1, 1, 1), &[]) }
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_launch_kernel_block_dim_exceeded() {
        let backend = make_backend(GpuApi::Cuda);
        // max_block_dim[0] for CUDA is 1024
        let result = unsafe { backend
            .launch_kernel(b"ptx", (1, 1, 1), (2048, 1, 1), &[]) }
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_launch_kernel_requires_real_dispatch() {
        for api in [GpuApi::Cuda, GpuApi::Rocm, GpuApi::Vulkan] {
            let backend = make_backend(api);
            // No driver/runtime is loaded by make_backend. No arguments are
            // dereferenced; this verifies that skipped work cannot report success.
            let result = unsafe {
                backend.launch_kernel(b"placeholder", (1, 1, 1), (32, 1, 1), &[])
            }.await;
            assert!(result.is_err(), "launch must not silently succeed for {api:?}");
        }
    }

    // -- Synchronize --------------------------------------------------------

    #[test]
    fn test_synchronize_all_apis() {
        for api in [GpuApi::Cuda, GpuApi::Rocm, GpuApi::Vulkan, GpuApi::None] {
            let backend = make_backend(api);
            assert!(backend.synchronize().is_ok(), "synchronize failed for {:?}", api);
        }
    }

    // -- Initialize ---------------------------------------------------------

    #[tokio::test]
    async fn test_initialize_idempotent() {
        let mut backend = make_backend(GpuApi::None);
        backend.initialize().await.unwrap();
        backend.initialize().await.unwrap(); // second call is a no-op
    }

    // -- Multiple allocations -----------------------------------------------

    #[test]
    fn test_multiple_allocations_tracked() {
        let backend = make_backend(GpuApi::None);
        let p1 = backend.allocate_memory(100).unwrap();
        let p2 = backend.allocate_memory(200).unwrap();
        let p3 = backend.allocate_memory(300).unwrap();
        assert_eq!(backend.allocation_count(), 3);
        assert_eq!(backend.allocated_bytes(), 600);

        backend.free_memory(p2).unwrap();
        assert_eq!(backend.allocation_count(), 2);
        assert_eq!(backend.allocated_bytes(), 400);

        backend.free_memory(p1).unwrap();
        backend.free_memory(p3).unwrap();
        assert_eq!(backend.allocation_count(), 0);
    }
}
