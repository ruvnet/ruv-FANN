//! Simple GPU Training Verification Test
//!
//! Quick test to verify GPU neural network training works correctly.
//! Uses a small 10K parameter network for fast validation.
//!
//! Run with: cargo run --example gpu_training_test --features gpu

use ruv_fann::training::*;
use ruv_fann::*;
use std::time::Instant;
use rand::prelude::*;
use rand_distr::StandardNormal;

fn main() {
    println!("🧪 GPU Neural Network Training Test");
    println!("==================================");
    
    // Create small test network: 50 -> 100 -> 50 -> 10 (~10K parameters)
    let input_size = 50;
    let hidden1_size = 100;
    let hidden2_size = 50;
    let output_size = 10;
    
    println!("📊 Network Architecture: {} -> {} -> {} -> {}", 
        input_size, hidden1_size, hidden2_size, output_size);
    
    // Generate test data
    let samples = 500; // Small dataset for quick test
    let training_data = generate_test_data(input_size, output_size, samples);
    println!("📈 Training samples: {}", samples);
    
    // Test 1: CPU Adam baseline
    println!("\n1️⃣ Testing CPU Adam Training:");
    let cpu_result = test_cpu_training(&training_data, input_size, hidden1_size, hidden2_size, output_size);
    
    // Test 2: GPU Adam (if available)
    #[cfg(feature = "gpu")]
    {
        println!("\n2️⃣ Testing GPU Adam Training:");
        
        if !is_gpu_available() {
            println!("   ⚠️  GPU not available - skipping GPU test");
            println!("   ℹ️  CPU test completed successfully");
            return;
        }
        
        println!("   🔧 GPU Available: {}", get_gpu_capabilities());
        let gpu_result = test_gpu_training(&training_data, input_size, hidden1_size, hidden2_size, output_size);
        
        // Compare results
        if let (Some(cpu_time), Some(gpu_time)) = (cpu_result, gpu_result) {
            let speedup = cpu_time / gpu_time;
            
            println!("\n🎯 Performance Comparison:");
            println!("   • CPU Time: {:.2}s", cpu_time);
            println!("   • GPU Time: {:.2}s", gpu_time);
            println!("   • Speedup: {:.2}x", speedup);
            
            if speedup > 2.0 {
                println!("   ✅ Excellent GPU acceleration!");
            } else if speedup > 1.1 {
                println!("   ⚡ Good GPU acceleration");
            } else if speedup > 0.9 {
                println!("   📈 GPU performance comparable to CPU");
            } else {
                println!("   ℹ️  CPU faster for this small network size");
            }
            
            println!("\n🎉 GPU training verification: SUCCESS!");
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("\n⚠️  GPU support not compiled");
        println!("   Enable with: cargo run --example gpu_training_test --features gpu");
        if cpu_result.is_some() {
            println!("   ✅ CPU training verified successfully");
        }
    }
}

fn generate_test_data(input_size: usize, output_size: usize, samples: usize) -> TrainingData<f32> {
    let mut rng = SmallRng::from_os_rng();
    let mut inputs = Vec::with_capacity(samples);
    let mut outputs = Vec::with_capacity(samples);
    
    for _ in 0..samples {
        // Generate random input
        let input: Vec<f32> = (0..input_size)
            .map(|_| rng.sample::<f32, _>(StandardNormal) * 0.5)
            .collect();
        
        // Generate target output with learnable pattern
        let mut output = vec![0.0; output_size];
        for i in 0..output_size {
            let mut value = 0.0;
            // Create non-linear relationship
            for (j, &inp) in input.iter().enumerate() {
                let weight = ((i + j) as f32 * 0.1).sin();
                value += inp * weight;
            }
            output[i] = value.tanh(); // Normalize to [-1, 1]
        }
        
        inputs.push(input);
        outputs.push(output);
    }
    
    TrainingData { inputs, outputs }
}

fn test_cpu_training(
    data: &TrainingData<f32>,
    input_size: usize,
    hidden1_size: usize,
    hidden2_size: usize,
    output_size: usize,
) -> Option<f64> {
    // Build network
    let mut network = NetworkBuilder::new()
        .input_layer(input_size)
        .hidden_layer(hidden1_size)
        .hidden_layer(hidden2_size)
        .output_layer(output_size)
        .build();
    
    // Create Adam optimizer
    let mut trainer = Adam::new(0.001)
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_epsilon(1e-8);
    
    let epochs = 100; // Extended test
    let start_time = Instant::now();
    
    for epoch in 0..epochs {
        match trainer.train_epoch(&mut network, data) {
            Ok(error) => {
                if epoch % 20 == 0 || epoch == epochs - 1 {
                    println!("   Epoch {}: Error = {:.6}", epoch, error);
                }
            }
            Err(e) => {
                println!("   ❌ CPU training failed: {}", e);
                return None;
            }
        }
    }
    
    let elapsed = start_time.elapsed().as_secs_f64();
    println!("   ✅ CPU training completed in {:.2}s", elapsed);
    Some(elapsed)
}

#[cfg(feature = "gpu")]
fn test_gpu_training(
    data: &TrainingData<f32>,
    input_size: usize,
    hidden1_size: usize,
    hidden2_size: usize,
    output_size: usize,
) -> Option<f64> {
    // Build network
    let mut network = NetworkBuilder::new()
        .input_layer(input_size)
        .hidden_layer(hidden1_size)
        .hidden_layer(hidden2_size)
        .output_layer(output_size)
        .build();
    
    // Create GPU Adam optimizer
    let mut trainer = match GpuAdam::new(0.001) {
        Ok(trainer) => {
            println!("   ✅ GPU Adam optimizer initialized");
            trainer
                .with_beta1(0.9)
                .with_beta2(0.999)
                .with_epsilon(1e-8)
        }
        Err(e) => {
            println!("   ❌ GPU Adam initialization failed: {}", e);
            return None;
        }
    };
    
    let epochs = 100; // Extended test
    let start_time = Instant::now();
    
    for epoch in 0..epochs {
        match trainer.train_epoch(&mut network, data) {
            Ok(error) => {
                if epoch % 20 == 0 || epoch == epochs - 1 {
                    println!("   Epoch {}: Error = {:.6}", epoch, error);
                }
            }
            Err(e) => {
                println!("   ❌ GPU training failed: {}", e);
                return None;
            }
        }
    }
    
    let elapsed = start_time.elapsed().as_secs_f64();
    
    // Show GPU performance stats
    let gpu_stats = trainer.get_performance_stats();
    println!("   ✅ GPU training completed in {:.2}s", elapsed);
    println!("   📊 GPU Stats:");
    println!("      • Total GPU time: {:.2}ms", gpu_stats.total_gpu_time_ms);
    println!("      • Kernel launches: {}", gpu_stats.kernel_launches);
    println!("      • Avg batch time: {:.2}ms", gpu_stats.avg_batch_time_ms);
    
    Some(elapsed)
}