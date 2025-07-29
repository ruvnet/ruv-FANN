//! Personal Tiny Star Model Training Example

use ruv_fann::{Network, ActivationFunction};
use ruv_fann::training::TrainingData;

fn main() {
    println!("🌟 Personal Tiny Model Training Demo");
    
    // Step 1: Create a simple tiny network
    let mut network = Network::new(&[2, 2, 1]); // Simplest: 2->2->1
    
    // Step 2: Set activation function
    network.set_activation_function(1, ActivationFunction::Sigmoid);
    network.set_activation_function(2, ActivationFunction::Sigmoid);
    
    // Step 3: Create training data (XOR problem - classic neural network test)
    let mut training_data = TrainingData {
        inputs: vec![
            vec![0.0, 0.0],  // Input pattern 1
            vec![0.0, 1.0],  // Input pattern 2  
            vec![1.0, 0.0],  // Input pattern 3
            vec![1.0, 1.0],  // Input pattern 4
        ],
        outputs: vec![
            vec![0.0],  // Expected output 1 (0 XOR 0 = 0)
            vec![1.0],  // Expected output 2 (0 XOR 1 = 1)
            vec![1.0],  // Expected output 3 (1 XOR 0 = 1)
            vec![0.0],  // Expected output 4 (1 XOR 1 = 0)
        ],
    };
    
    // Step 4: Train the model
    println!("🔥 Training XOR model...");
    match network.train(&training_data.inputs, &training_data.outputs, 0.5, 1000) {
        Ok(_) => println!("✅ Training successful!"),
        Err(e) => println!("❌ Training failed: {:?}", e),
    }
    
    // Step 5: Test model accuracy
    let accuracy = test_model(&mut network, &training_data);
    println!("🎯 Model accuracy: {:.1}%", accuracy * 100.0);
    
    // Step 6: Test individual predictions
    println!("\n💎 Testing XOR Predictions:");
    for i in 0..training_data.inputs.len() {
        let input = &training_data.inputs[i];
        let expected = training_data.outputs[i][0];
        let predicted = network.run(input)[0];
        
        let correct = if (predicted - expected).abs() < 0.5 { "✅" } else { "❌" };
        
        println!("   Input: {:?} → Expected: {:.1}, Got: {:.3} {}", 
                input, expected, predicted, correct);
    }
    
    // Step 7: Calculate model size
    let connections = network.total_connections();
    let neurons = network.total_neurons();
    let size_bytes = (connections + neurons) * 4; // 4 bytes per f32
    
    println!("\n📊 Model Statistics:");
    println!("   🧠 Total neurons: {}", neurons);
    println!("   🔗 Total connections: {}", connections);
    println!("   💎 Model size: {:.1}KB", size_bytes as f32 / 1024.0);
    
    println!("\n🎉 Model Training Complete!");
    println!("🌟 You've successfully trained a neural network from scratch!");
}

fn test_model(network: &mut Network<f32>, data: &TrainingData<f32>) -> f32 {
    let mut correct = 0;
    let total = data.inputs.len();
    
    for i in 0..total {
        let predicted = network.run(&data.inputs[i])[0];
        let expected = data.outputs[i][0];
        
        // Consider prediction correct if within 0.5 of expected
        if (predicted - expected).abs() < 0.5 {
            correct += 1;
        }
    }
    
    correct as f32 / total as f32
}