// WebGPU Compute Shaders: Gradient Computation for Backpropagation
// High-performance implementations for neural network training
// Supports error gradients, weight gradients, and optimizer updates
// Target: 20-40x speedup over CPU backpropagation

// ===== ERROR GRADIENT COMPUTATION =====

@group(0) @binding(0) var<storage, read> layer_outputs: array<f32>;
@group(0) @binding(1) var<storage, read> expected_outputs: array<f32>;
@group(0) @binding(2) var<storage, read> activation_derivatives: array<f32>;
@group(0) @binding(3) var<storage, read_write> error_gradients: array<f32>;

struct ErrorGradientUniforms {
    output_size: u32,
    loss_function: u32,  // 0: MSE, 1: Cross-entropy, 2: MAE
    padding: vec2<u32>,
}

@group(0) @binding(4) var<uniform> error_uniforms: ErrorGradientUniforms;

// Compute output layer error gradients based on loss function
@compute @workgroup_size(256)
fn compute_output_error_gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= error_uniforms.output_size) {
        return;
    }
    
    let output = layer_outputs[idx];
    let expected = expected_outputs[idx];
    let derivative = activation_derivatives[idx];
    
    var error_gradient: f32;
    
    // Compute error based on loss function
    switch (error_uniforms.loss_function) {
        case 0u: { // Mean Squared Error
            error_gradient = (output - expected) * derivative;
        }
        case 1u: { // Cross-entropy (for sigmoid/softmax outputs)
            error_gradient = output - expected; // Simplified for sigmoid
        }
        case 2u: { // Mean Absolute Error
            error_gradient = sign(output - expected) * derivative;
        }
        default: { // Default to MSE
            error_gradient = (output - expected) * derivative;
        }
    }
    
    error_gradients[idx] = error_gradient;
}

// ===== HIDDEN LAYER ERROR PROPAGATION =====

@group(1) @binding(0) var<storage, read> next_layer_errors: array<f32>;
@group(1) @binding(1) var<storage, read> weights: array<f32>;
@group(1) @binding(2) var<storage, read> current_derivatives: array<f32>;
@group(1) @binding(3) var<storage, read_write> current_errors: array<f32>;

struct HiddenErrorUniforms {
    current_size: u32,
    next_size: u32,
    padding: vec2<u32>,
}

@group(1) @binding(4) var<uniform> hidden_uniforms: HiddenErrorUniforms;

// Backpropagate errors through hidden layers
@compute @workgroup_size(256)
fn backpropagate_hidden_errors(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current_neuron = global_id.x;
    
    if (current_neuron >= hidden_uniforms.current_size) {
        return;
    }
    
    var error_sum = 0.0;
    
    // Sum weighted errors from next layer
    for (var next_neuron = 0u; next_neuron < hidden_uniforms.next_size; next_neuron++) {
        let weight_idx = next_neuron * hidden_uniforms.current_size + current_neuron;
        error_sum += next_layer_errors[next_neuron] * weights[weight_idx];
    }
    
    // Multiply by activation derivative
    current_errors[current_neuron] = error_sum * current_derivatives[current_neuron];
}

// ===== WEIGHT GRADIENT COMPUTATION =====

@group(2) @binding(0) var<storage, read> input_activations: array<f32>;
@group(2) @binding(1) var<storage, read> output_errors: array<f32>;
@group(2) @binding(2) var<storage, read_write> weight_gradients: array<f32>;

struct WeightGradientUniforms {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    learning_rate: f32,
}

@group(2) @binding(3) var<uniform> weight_uniforms: WeightGradientUniforms;

// Compute weight gradients using outer product of inputs and errors
@compute @workgroup_size(16, 16)
fn compute_weight_gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_idx = global_id.x;
    let output_idx = global_id.y;
    
    if (input_idx >= weight_uniforms.input_size || output_idx >= weight_uniforms.output_size) {
        return;
    }
    
    let weight_idx = output_idx * weight_uniforms.input_size + input_idx;
    var gradient_sum = 0.0;
    
    // Accumulate gradients across batch
    for (var batch = 0u; batch < weight_uniforms.batch_size; batch++) {
        let input_batch_idx = batch * weight_uniforms.input_size + input_idx;
        let error_batch_idx = batch * weight_uniforms.output_size + output_idx;
        
        gradient_sum += input_activations[input_batch_idx] * output_errors[error_batch_idx];
    }
    
    // Average over batch and apply learning rate
    weight_gradients[weight_idx] = gradient_sum / f32(weight_uniforms.batch_size);
}

// ===== BIAS GRADIENT COMPUTATION =====

@group(3) @binding(0) var<storage, read> bias_errors: array<f32>;
@group(3) @binding(1) var<storage, read_write> bias_gradients: array<f32>;

struct BiasGradientUniforms {
    layer_size: u32,
    batch_size: u32,
    padding: vec2<u32>,
}

@group(3) @binding(2) var<uniform> bias_uniforms: BiasGradientUniforms;

// Compute bias gradients (sum errors across batch)
@compute @workgroup_size(256)
fn compute_bias_gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let neuron_idx = global_id.x;
    
    if (neuron_idx >= bias_uniforms.layer_size) {
        return;
    }
    
    var gradient_sum = 0.0;
    
    // Sum errors across batch
    for (var batch = 0u; batch < bias_uniforms.batch_size; batch++) {
        let error_idx = batch * bias_uniforms.layer_size + neuron_idx;
        gradient_sum += bias_errors[error_idx];
    }
    
    // Average over batch
    bias_gradients[neuron_idx] = gradient_sum / f32(bias_uniforms.batch_size);
}

// ===== WEIGHT UPDATE KERNELS =====

@group(5) @binding(0) var<storage, read_write> weights: array<f32>;
@group(5) @binding(1) var<storage, read> weight_gradients: array<f32>;
@group(5) @binding(2) var<storage, read_write> momentum_buffer: array<f32>;
@group(5) @binding(3) var<storage, read_write> velocity_buffer: array<f32>; // For Adam optimizer

struct OptimizerUniforms {
    weight_count: u32,
    learning_rate: f32,
    momentum: f32,
    beta1: f32,        // Adam beta1
    beta2: f32,        // Adam beta2
    epsilon: f32,      // Adam epsilon
    weight_decay: f32, // L2 regularization
    optimizer_type: u32, // 0: SGD, 1: Momentum, 2: Adam
}

@group(5) @binding(4) var<uniform> opt_uniforms: OptimizerUniforms;

// SGD with momentum weight update
@compute @workgroup_size(256)
fn sgd_momentum_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= opt_uniforms.weight_count) {
        return;
    }
    
    let gradient = weight_gradients[idx];
    let current_weight = weights[idx];
    
    // Apply L2 regularization (weight decay)
    let regularized_gradient = gradient + opt_uniforms.weight_decay * current_weight;
    
    // Update momentum
    let momentum_update = opt_uniforms.momentum * momentum_buffer[idx] - opt_uniforms.learning_rate * regularized_gradient;
    momentum_buffer[idx] = momentum_update;
    
    // Update weight
    weights[idx] = current_weight + momentum_update;
}

// Simple weight update for basic SGD
@compute @workgroup_size(256)
fn simple_weight_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= opt_uniforms.weight_count) {
        return;
    }
    
    let gradient = weight_gradients[idx];
    weights[idx] = weights[idx] - opt_uniforms.learning_rate * gradient;
}