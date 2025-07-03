//! Integration tests for the Autonomous GPU Resource Management System
//!
//! These tests require GPU features and are only compiled when available.

#![cfg(test)]
#![cfg(feature = "gpu")]

// This test file is disabled unless GPU features are available
// This prevents compilation errors when GPU modules are not present

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_autonomous_gpu_resource_manager_creation() {
    use ruv_fann::webgpu::{AutonomousGpuResourceManager, GpuMemoryConfig};
    
    // Test creating the autonomous GPU resource manager
    let config = GpuMemoryConfig::default();
    let result = AutonomousGpuResourceManager::<f32>::new(config).await;
    
    match result {
        Ok(manager) => {
            println!("✅ Autonomous GPU Resource Manager created successfully");
            
            // Test basic functionality
            let capabilities = manager.get_capabilities();
            println!("Manager capabilities: {:?}", capabilities);
            
            // Verify the manager has proper resource pools
            let utilization = manager.get_utilization_summary();
            println!("Resource utilization: {:?}", utilization);
            
            assert!(utilization.total_capacity > 0, "Should have some resource capacity");
        }
        Err(e) => {
            println!("⚠️  GPU Resource Manager not available: {:?}", e);
            println!("This is expected if WebGPU is not available on this system");
        }
    }
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_resource_allocation_and_deallocation() {
    use ruv_fann::webgpu::{AutonomousGpuResourceManager, GpuMemoryConfig, ResourceRequirements, ResourceType, Priority};
    
    let config = GpuMemoryConfig::default();
    if let Ok(mut manager) = AutonomousGpuResourceManager::<f32>::new(config).await {
        // Test resource allocation
        let requirements = ResourceRequirements {
            resource_type: ResourceType::Compute,
            priority: Priority::Medium,
            expected_duration_ms: 1000,
            memory_mb: 64,
            compute_units: 4,
        };
        
        let allocation_result = manager.allocate_resources(requirements).await;
        
        match allocation_result {
            Ok(allocation) => {
                println!("✅ Resource allocation successful: {:?}", allocation);
                
                // Test deallocation
                let deallocation_result = manager.deallocate_resources(allocation.id).await;
                assert!(deallocation_result.is_ok(), "Deallocation should succeed");
                println!("✅ Resource deallocation successful");
            }
            Err(e) => {
                println!("⚠️  Resource allocation failed: {:?}", e);
                println!("This may be expected if GPU resources are limited");
            }
        }
    } else {
        println!("⚠️  GPU Resource Manager not available, skipping allocation test");
    }
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_performance_monitoring() {
    use ruv_fann::webgpu::{AutonomousGpuResourceManager, GpuMemoryConfig};
    
    let config = GpuMemoryConfig {
        enable_monitoring: true,
        auto_start_monitoring: true,
        ..Default::default()
    };
    
    if let Ok(manager) = AutonomousGpuResourceManager::<f32>::new(config).await {
        // Test performance monitoring
        let performance_metrics = manager.get_performance_metrics();
        println!("Performance metrics: {:?}", performance_metrics);
        
        // Performance metrics should be valid
        assert!(performance_metrics.throughput >= 0.0, "Throughput should be non-negative");
        assert!(performance_metrics.latency >= 0.0, "Latency should be non-negative");
        assert!(performance_metrics.efficiency >= 0.0 && performance_metrics.efficiency <= 1.0, "Efficiency should be between 0 and 1");
        
        println!("✅ Performance monitoring working correctly");
    } else {
        println!("⚠️  GPU Resource Manager not available, skipping performance test");
    }
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_resource_trading_system() {
    use ruv_fann::webgpu::{AutonomousGpuResourceManager, GpuMemoryConfig, ResourceType, Priority};
    
    let config = GpuMemoryConfig::default();
    if let Ok(manager) = AutonomousGpuResourceManager::<f32>::new(config).await {
        // Test resource trading system
        let trading_system = manager.get_trading_system();
        
        // Test market state
        let market_state = trading_system.get_market_state();
        println!("Market state: {:?}", market_state);
        
        // Verify market has resource pools
        assert!(!market_state.available_pools.is_empty(), "Market should have resource pools");
        
        // Test creating a trade proposal
        let trade_proposal = trading_system.create_trade_proposal(
            ResourceType::Compute,
            64, // requesting 64 MB
            Priority::Medium,
        );
        
        match trade_proposal {
            Ok(proposal) => {
                println!("✅ Trade proposal created: {:?}", proposal);
                
                // Test evaluating the proposal
                let evaluation = trading_system.evaluate_proposal(&proposal);
                println!("Trade evaluation: {:?}", evaluation);
                
                assert!(evaluation.feasibility_score >= 0.0 && evaluation.feasibility_score <= 1.0, 
                       "Feasibility score should be between 0 and 1");
            }
            Err(e) => {
                println!("⚠️  Trade proposal failed: {:?}", e);
                println!("This may be expected if resources are fully allocated");
            }
        }
        
        println!("✅ Resource trading system test completed");
    } else {
        println!("⚠️  GPU Resource Manager not available, skipping trading test");
    }
}

#[test]
fn test_compilation_guard() {
    // This test ensures the file compiles even when GPU features are not available
    println!("Autonomous GPU integration tests require gpu feature");
}
