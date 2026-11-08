use cuda_rust_wasm::runtime::{
    flash_attention::{FlashAttention, FlashAttentionConfig},
    Block,
};
use cuda_rust_wasm::{
    memory::{HostBuffer, UnifiedMemory},
    parser::CudaParser,
};

#[test]
fn host_buffers_initialize_valid_typed_values_and_align() {
    #[derive(Clone, Copy)]
    #[repr(align(64))]
    struct Aligned(u8);
    impl Default for Aligned {
        fn default() -> Self {
            Self(7)
        }
    }
    let buffer = HostBuffer::<Aligned>::new(4).unwrap();
    assert_eq!(buffer.as_slice().as_ptr() as usize % 64, 0);
    assert!(buffer.as_slice().iter().all(|x| x.0 == 7));
    assert!(HostBuffer::<u64>::new(usize::MAX).is_err());
    let booleans = HostBuffer::<bool>::new(4).unwrap();
    assert_eq!(booleans.as_slice(), &[false; 4]);
}

#[test]
fn unified_memory_initializes_unwritten_tail() {
    let mut memory = UnifiedMemory::new(32).unwrap();
    memory.copy_from_slice(&[1, 2, 3]).unwrap();
    let mut result = [255; 32];
    memory.copy_to_slice(&mut result).unwrap();
    assert_eq!(&result[..3], &[1, 2, 3]);
    assert_eq!(&result[3..], &[0; 29]);
}

#[test]
fn attention_rejects_zero_tiles_and_ignored_dropout() {
    for field in 0..4 {
        let mut config = FlashAttentionConfig::new(4, 1);
        match field {
            0 => config.head_dim = 0,
            1 => config.num_heads = 0,
            2 => config.block_size_q = 0,
            _ => config.block_size_kv = 0,
        }
        assert!(FlashAttention::new(config)
            .forward(&[0.; 4], &[0.; 4], &[0.; 4])
            .is_err());
    }
    let attention = FlashAttention::new(FlashAttentionConfig::new(4, 1).with_dropout(0.5));
    assert!(attention.forward(&[0.; 4], &[0.; 4], &[0.; 4]).is_err());
    let attention = FlashAttention::new(FlashAttentionConfig::new(4, 1));
    assert!(attention
        .forward_multi_head(&[], &[], &[], usize::MAX, 2, 2)
        .is_err());
}

#[test]
fn parser_handles_utf8_and_bounds_recursive_input() {
    let parser = CudaParser::new();
    // Previously sliced at byte 1 of a multi-byte code point and panicked.
    assert!(parser.parse("💥").is_ok());
    assert!(parser
        .parse(&format!(
            "__global__ void f() {{ int x = {}1; }}",
            "!".repeat(300)
        ))
        .is_err());
    assert!(parser
        .parse(&format!(
            "__global__ void f() {{ int x = {}1{}; }}",
            "(".repeat(100),
            ")".repeat(100)
        ))
        .is_err());
    assert!(parser.parse("/* not closed").is_err());
}

#[test]
fn block_limits_reject_overflow_and_zero_axes() {
    assert!(Block::new((u32::MAX, u32::MAX, u32::MAX))
        .validate()
        .is_err());
    assert!(Block::new((0, 1, 1)).validate().is_err());
}

#[test]
fn full_pool_release_does_not_invert_allocation_locks() {
    use cuda_rust_wasm::memory::{MemoryPool, PoolConfig};
    use std::sync::{mpsc, Arc};
    let pool = Arc::new(MemoryPool::with_config(PoolConfig {
        max_pool_size: 0,
        prealloc_count: 0,
        ..PoolConfig::default()
    }));
    let (tx, rx) = mpsc::channel();
    for _ in 0..4 {
        let pool = pool.clone();
        let tx = tx.clone();
        std::thread::spawn(move || {
            for _ in 0..2000 {
                let bytes = pool.allocate(1024);
                pool.deallocate(bytes);
            }
            tx.send(()).unwrap();
        });
    }
    for _ in 0..4 {
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("pool locks deadlocked");
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn config_debug_redacts_credentials() {
    use cuda_rust_wasm::nutanix::config::NutanixConfig;
    let config =
        NutanixConfig::with_basic_auth("https://example.invalid", "user", "password-canary");
    let debug = format!("{:?}", config);
    assert!(!debug.contains("password-canary"));
    let config = NutanixConfig::new("https://example.invalid", "api-key-canary");
    assert!(!format!("{:?}", config).contains("api-key-canary"));
}

#[test]
fn kernel_manager_tracks_aligned_owned_allocations() {
    use cuda_rust_wasm::memory::KernelMemoryManager;
    let manager = KernelMemoryManager::new();
    for alignment in [1, 8, 64, 4096] {
        let pointer = manager.allocate_kernel_memory(64, alignment).unwrap();
        assert_eq!(pointer as usize % alignment, 0);
        assert_eq!(manager.total_kernel_memory(), 64);
        // SAFETY: pointer is live and has at least 64 writable bytes; no aliases.
        unsafe {
            std::ptr::write_bytes(pointer, 0xAA, 64);
            manager.deallocate_kernel_memory(pointer).unwrap();
            assert!(manager.deallocate_kernel_memory(pointer).is_err());
        }
        assert_eq!(manager.total_kernel_memory(), 0);
    }
    assert!(manager.allocate_kernel_memory(64, 0).is_err());
    assert!(manager.allocate_kernel_memory(64, 3).is_err());
    assert!(manager.allocate_kernel_memory(usize::MAX, 64).is_err());
}

#[test]
fn shared_memory_initializes_and_drops_nonzero_valid_types() {
    use cuda_rust_wasm::kernel::shared_memory::SharedMemory;
    use std::num::NonZeroU8;
    #[derive(Default)]
    struct Initialized(Option<NonZeroU8>);
    let data = SharedMemory::<Initialized>::new(4);
    assert!(data.as_slice().iter().all(|item| item.0.is_none()));
    let strings = SharedMemory::<String>::new(2);
    assert!(strings.get(0).is_empty());
}
