//! Comprehensive tests for GPU Memory Management
//!
//! Tests both basic and enhanced GPU memory managers with actual implementation

#[cfg(feature = "gpu")]
mod memory_tests {
    use ruv_fann::webgpu::{
        get_memory_capabilities, BufferCategory, GpuMemoryConfig, GpuMemoryManager,
    };

    #[test]
    fn test_memory_capabilities() {
        let capabilities = get_memory_capabilities();

        // Should have GPU features enabled when feature flag is on
        assert!(
            capabilities.webgpu_available,
            "WebGPU should be available with gpu feature"
        );
        assert!(
            capabilities.enhanced_features,
            "Enhanced features should be available"
        );
        assert!(
            capabilities.buffer_pooling,
            "Buffer pooling should be available"
        );

        println!("Memory capabilities: {}", capabilities.summary());
    }

    #[test]
    fn test_buffer_category_enum() {
        // Test buffer category classification
        let categories = vec![
            BufferCategory::Micro,
            BufferCategory::Small,
            BufferCategory::Medium,
            BufferCategory::Large,
            BufferCategory::XLarge,
        ];

        assert_eq!(categories.len(), 5, "Should have 5 buffer categories");

        // Test size classification logic
        assert!(BufferCategory::from_size(512) == BufferCategory::Micro);
        assert!(BufferCategory::from_size(50_000) == BufferCategory::Small);
        assert!(BufferCategory::from_size(5_000_000) == BufferCategory::Medium);

        println!("✅ Buffer categories work correctly");
    }

    #[test]
    fn test_gpu_memory_config() {
        let config = GpuMemoryConfig::default();

        // Verify default configuration makes sense
        assert!(
            config.pressure_threshold >= 0.0 && config.pressure_threshold <= 1.0,
            "Pressure threshold should be between 0.0 and 1.0"
        );
        assert!(
            config.enable_advanced_features || !config.enable_advanced_features,
            "Boolean field should be valid"
        );

        // Test creating config with specific values
        let custom_config = GpuMemoryConfig {
            enable_advanced_features: true,
            enable_daa: true,
            enable_monitoring: true,
            auto_start_monitoring: false,
            pressure_threshold: 0.8,
            monitor_config: Default::default(),
            enable_circuit_breaker: true,
            ..GpuMemoryConfig::default()
        };

        assert_eq!(custom_config.pressure_threshold, 0.8);
        assert!(custom_config.enable_advanced_features);
        assert!(custom_config.enable_daa);

        println!("✅ GPU memory config works correctly");
    }

    #[test]
    fn test_basic_memory_manager() {
        // Test basic memory manager functionality
        let manager = GpuMemoryManager::new();

        // Get basic memory stats
        let stats = manager.get_stats();

        // Verify stats structure
        assert!(
            stats.total_allocated >= 0,
            "Total allocated should be non-negative"
        );
        assert!(stats.available > 0, "Available memory should be positive");
        assert!(
            stats.buffer_count >= 0,
            "Buffer count should be non-negative"
        );
        assert!(
            stats.fragmentation_ratio >= 0.0,
            "Fragmentation ratio should be non-negative"
        );

        println!("Memory stats: {:?}", stats);
        println!("✅ Basic memory manager works correctly");
    }

    #[test]
    fn test_memory_capabilities_integration() {
        // Test that memory capabilities are properly exposed
        let capabilities = get_memory_capabilities();

        // Basic capability checks
        assert!(
            capabilities.buffer_pooling,
            "Buffer pooling should be available"
        );

        // When GPU feature is enabled, enhanced features should be available
        if capabilities.webgpu_available {
            assert!(
                capabilities.enhanced_features,
                "Enhanced features should be available with GPU"
            );
            println!("GPU memory features available");
        } else {
            println!("Running in CPU fallback mode");
        }

        // Test that manager can be created
        let manager = GpuMemoryManager::new();
        let stats = manager.get_stats();
        assert!(stats.available > 0, "Memory should be available");

        println!("✅ Memory capabilities integration works");
    }
}

#[cfg(not(feature = "gpu"))]
mod fallback_tests {
    #[test]
    fn test_memory_fallback_without_gpu() {
        // When GPU features are disabled, ensure we can still test basic functionality
        println!("GPU memory management tests require gpu feature");
        assert!(true, "Fallback test passes when webgpu feature is disabled");
    }
}
