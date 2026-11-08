//! Simple test to verify Adam optimizer is working correctly
//! Tests on XOR problem which should converge quickly

use ruv_fann::training::TrainingAlgorithm;
use ruv_fann::training::*;
use ruv_fann::*;

fn main() {
    println!("🧪 Testing Adam Optimizer on XOR Problem");
    println!("=========================================");

    // Create a simple XOR network
    let network = NetworkBuilder::new()
        .input_layer(2)
        .hidden_layer(4)
        .output_layer(1)
        .build();

    // XOR training data
    let train_data = TrainingData {
        inputs: vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        outputs: vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    };

    // Test SGD first
    println!("\n🔄 Testing SGD baseline:");
    let mut network_sgd = network.clone();
    let mut sgd = IncrementalBackprop::new(0.5);

    for epoch in 0..50 {
        match sgd.train_epoch(&mut network_sgd, &train_data) {
            Ok(error) => {
                if epoch % 10 == 0 {
                    println!("  Epoch {}: Error = {:.6}", epoch, error);
                }
                if error < 0.01 {
                    println!(
                        "  ✅ SGD converged at epoch {} with error {:.6}",
                        epoch, error
                    );
                    break;
                }
            }
            Err(e) => {
                println!("  ❌ SGD failed: {}", e);
                break;
            }
        }
    }

    // Test Adam
    println!("\n🚀 Testing Adam optimizer:");
    let mut network_adam = network.clone();
    let mut adam = Adam::new(0.01);

    for epoch in 0..50 {
        match adam.train_epoch(&mut network_adam, &train_data) {
            Ok(error) => {
                if epoch % 10 == 0 {
                    println!("  Epoch {}: Error = {:.6}", epoch, error);
                }
                if error < 0.01 {
                    println!(
                        "  ✅ Adam converged at epoch {} with error {:.6}",
                        epoch, error
                    );
                    break;
                }
            }
            Err(e) => {
                println!("  ❌ Adam failed: {}", e);
                break;
            }
        }
    }

    // Test the final networks
    println!("\n📊 Final Results:");
    println!("SGD Network:");
    test_network(&mut network_sgd, &train_data);

    println!("\nAdam Network:");
    test_network(&mut network_adam, &train_data);
}

fn test_network(network: &mut Network<f32>, data: &TrainingData<f32>) {
    for (i, (input, expected)) in data.inputs.iter().zip(data.outputs.iter()).enumerate() {
        let output = network.run(input).unwrap();
        println!(
            "  Input: {:?} -> Output: {:.3}, Expected: {:.3}",
            input, output[0], expected[0]
        );
    }
}
