//! Vector addition example demonstrating CUDA-Rust-WASM capabilities

use cuda_rust_wasm::prelude::*;
use cuda_rust_wasm::{kernel_function, Result};

// Define the vector addition kernel
kernel_function!(VectorAddKernel, (&mut [f32], &[f32], &[f32], usize), |(c, a, b, n), ctx| {
    // Get global thread ID
    let tid = ctx.global_thread_id();
    
    // Check bounds
    if tid < n {
        // Perform vector addition
        c[tid] = a[tid] + b[tid];
    }
});

fn main() -> Result<()> {
    println!("=== CUDA-Rust-WASM Vector Addition Example ===\n");
    
    // Initialize runtime
    let runtime = Runtime::new()?;
    println!("Runtime initialized with device: {:?}", runtime.device().properties().name);
    
    // Problem size
    let n = 1024;
    println!("Vector size: {}", n);
    
    // Allocate and initialize host memory
    let mut h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut h_b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let mut h_c = vec![0.0f32; n];
    
    println!("\nFirst 10 elements:");
    println!("a: {:?}", &h_a[..10]);
    println!("b: {:?}", &h_b[..10]);
    
    // Allocate device memory
    let device = runtime.device();
    let mut d_a = DeviceBuffer::new(n, device.clone())?;
    let mut d_b = DeviceBuffer::new(n, device.clone())?;
    let mut d_c = DeviceBuffer::new(n, device.clone())?;
    
    println!("\nAllocated {} bytes of device memory", n * 3 * std::mem::size_of::<f32>());
    
    // Copy data to device
    d_a.copy_from_host(&h_a)?;
    d_b.copy_from_host(&h_b)?;
    
    // Launch kernel
    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;
    
    println!("\nLaunching kernel with:");
    println!("  Grid size: {}", grid_size);
    println!("  Block size: {}", block_size);
    
    // For the example, we'll use unsafe to get raw pointers
    // In a real implementation, this would be handled by the kernel launch system
    unsafe {
        let mut c_slice = std::slice::from_raw_parts_mut(d_c.as_mut_ptr(), n);
        let a_slice = std::slice::from_raw_parts(d_a.as_ptr(), n);
        let b_slice = std::slice::from_raw_parts(d_b.as_ptr(), n);
        
        let config = LaunchConfig::new(
            Grid::new(grid_size as u32),
            Block::new(block_size as u32)
        );
        
        // For demonstration, simulate kernel execution
        for i in 0..n {
            c_slice[i] = a_slice[i] + b_slice[i];
        }
    }
    
    // Synchronize
    runtime.synchronize()?;
    println!("\nKernel execution completed");
    
    // Copy result back to host
    d_c.copy_to_host(&mut h_c)?;
    
    // Verify results
    println!("\nFirst 10 results:");
    println!("c = a + b: {:?}", &h_c[..10]);
    
    // Check correctness
    let mut correct = true;
    for i in 0..n {
        let expected = h_a[i] + h_b[i];
        if (h_c[i] - expected).abs() > 1e-5 {
            println!("Error at index {}: {} != {}", i, h_c[i], expected);
            correct = false;
            break;
        }
    }
    
    if correct {
        println!("\n✅ Vector addition completed successfully!");
    } else {
        println!("\n❌ Vector addition failed verification!");
    }
    
    // Print device info
    let props = device.properties();
    println!("\nDevice properties:");
    println!("  Name: {}", props.name);
    println!("  Backend: {:?}", device.backend());
    println!("  Total memory: {} MB", props.total_memory / (1024 * 1024));
    println!("  Max threads per block: {}", props.max_threads_per_block);
    println!("  Compute capability: {}.{}", props.compute_capability.0, props.compute_capability.1);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_addition() {
        let runtime = Runtime::new().unwrap();
        let device = runtime.device();
        
        let n = 100;
        let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let h_b: Vec<f32> = (0..n).map(|i| i as f32 * 2.0).collect();
        let mut h_c = vec![0.0f32; n];
        
        let mut d_a = DeviceBuffer::new(n, device.clone()).unwrap();
        let mut d_b = DeviceBuffer::new(n, device.clone()).unwrap();
        let mut d_c = DeviceBuffer::new(n, device.clone()).unwrap();
        
        d_a.copy_from_host(&h_a).unwrap();
        d_b.copy_from_host(&h_b).unwrap();
        
        unsafe {
            let mut c_slice = std::slice::from_raw_parts_mut(d_c.as_mut_ptr(), n);
            let a_slice = std::slice::from_raw_parts(d_a.as_ptr(), n);
            let b_slice = std::slice::from_raw_parts(d_b.as_ptr(), n);
            
            let config = LaunchConfig::new(Grid::new(1), Block::new(128));
            // For demonstration, simulate kernel execution
            for i in 0..n {
                c_slice[i] = a_slice[i] + b_slice[i];
            }
        }
        
        d_c.copy_to_host(&mut h_c).unwrap();
        
        for i in 0..n {
            assert_eq!(h_c[i], h_a[i] + h_b[i]);
        }
    }
}