//! Simple test to verify basic compilation

use ruv_fann::*;

fn main() {
    println!("Testing ruv-fann import");
    
    // Create a simple neural network
    let network = NeuralNetwork::new(2, &[3], 1);
    match network {
        Ok(_) => println!("Neural network created successfully"),
        Err(e) => println!("Failed to create neural network: {}", e),
    }
}