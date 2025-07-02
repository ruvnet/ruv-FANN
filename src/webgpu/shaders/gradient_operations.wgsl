// WebGPU Compute Shaders: Gradient Operations for Backpropagation
// High-performance GPU implementations of gradient computations
// Optimized for neural network training with vectorized operations

@group(0) @binding(0) var<storage, read> gradients_output: array<f32>;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradients_input: array<f32>;
@group(0) @binding(4) var<storage, read_write> weight_gradients: array<f32>;

struct GradientUniforms {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    learning_rate: f32,
    steepness: f32,       // For activation derivative calculation
    alpha: f32,           // For parameterized functions
    reserved: u32,
}

@group(0) @binding(5) var<uniform> uniforms: GradientUniforms;

// Sigmoid derivative: f'(x) = f(x) * (1 - f(x)) where f(x) is sigmoid output
@compute @workgroup_size(256)
fn sigmoid_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.output_size * uniforms.batch_size) {
        return;
    }
    
    let sigmoid_output = activations[index];
    let derivative = sigmoid_output * (1.0 - sigmoid_output) * uniforms.steepness;
    gradients_input[index] = gradients_output[index] * derivative;
}

// ReLU derivative: f'(x) = 1 if x > 0, 0 if x <= 0
@compute @workgroup_size(256)
fn relu_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.output_size * uniforms.batch_size) {
        return;
    }
    
    let derivative = select(0.0, 1.0, activations[index] > 0.0);
    gradients_input[index] = gradients_output[index] * derivative;
}

// Leaky ReLU derivative: f'(x) = 1 if x > 0, alpha if x <= 0
@compute @workgroup_size(256)
fn leaky_relu_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.output_size * uniforms.batch_size) {
        return;
    }
    
    let derivative = select(uniforms.alpha, 1.0, activations[index] > 0.0);
    gradients_input[index] = gradients_output[index] * derivative;
}

// Tanh derivative: f'(x) = steepness * (1 - f(x)^2) where f(x) is tanh output
@compute @workgroup_size(256)
fn tanh_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.output_size * uniforms.batch_size) {
        return;
    }
    
    let tanh_output = activations[index];
    let derivative = uniforms.steepness * (1.0 - tanh_output * tanh_output);
    gradients_input[index] = gradients_output[index] * derivative;
}

// Linear derivative: f'(x) = steepness (constant)
@compute @workgroup_size(256)
fn linear_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.output_size * uniforms.batch_size) {
        return;
    }
    
    gradients_input[index] = gradients_output[index] * uniforms.steepness;
}

// Weight gradient computation for fully connected layers
// Computes: dW = activation_input^T * gradient_output
@compute @workgroup_size(16, 16)
fn weight_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_idx = global_id.x;
    let output_idx = global_id.y;
    
    if (input_idx >= uniforms.input_size || output_idx >= uniforms.output_size) {
        return;
    }
    
    let weight_idx = output_idx * uniforms.input_size + input_idx;
    var gradient_sum: f32 = 0.0;
    
    // Sum gradients across all samples in the batch
    for (var batch_idx = 0u; batch_idx < uniforms.batch_size; batch_idx++) {
        let activation_idx = batch_idx * uniforms.input_size + input_idx;
        let grad_output_idx = batch_idx * uniforms.output_size + output_idx;
        
        gradient_sum += activations[activation_idx] * gradients_output[grad_output_idx];
    }
    
    // Average across batch and apply learning rate
    weight_gradients[weight_idx] = gradient_sum / f32(uniforms.batch_size) * uniforms.learning_rate;
}

// Input gradient computation for fully connected layers
// Computes: gradient_input = weights^T * gradient_output
@compute @workgroup_size(256)
fn input_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_input_idx = global_id.x;
    
    if (batch_input_idx >= uniforms.input_size * uniforms.batch_size) {
        return;
    }
    
    let batch_idx = batch_input_idx / uniforms.input_size;
    let input_idx = batch_input_idx % uniforms.input_size;
    
    var gradient_sum: f32 = 0.0;
    
    // Sum: weights[output][input] * gradient_output[batch][output]
    for (var output_idx = 0u; output_idx < uniforms.output_size; output_idx++) {
        let weight_idx = output_idx * uniforms.input_size + input_idx;
        let grad_output_idx = batch_idx * uniforms.output_size + output_idx;
        
        gradient_sum += weights[weight_idx] * gradients_output[grad_output_idx];
    }
    
    gradients_input[batch_input_idx] = gradient_sum;
}

// Efficient gradient clipping to prevent exploding gradients
@compute @workgroup_size(256)
fn gradient_clipping_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.input_size) {
        return;
    }
    
    let gradient = gradients_input[index];
    let clip_value = uniforms.alpha; // Reuse alpha parameter for clip threshold
    
    // Clip gradient to [-clip_value, clip_value]
    gradients_input[index] = clamp(gradient, -clip_value, clip_value);
}

// L2 regularization gradient addition
@compute @workgroup_size(256)
fn l2_regularization_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.input_size) {
        return;
    }
    
    let weight = weights[index];
    let l2_lambda = uniforms.alpha; // Reuse alpha for L2 regularization strength
    
    // Add L2 regularization term: gradient += lambda * weight
    weight_gradients[index] += l2_lambda * weight;
}

// Momentum update for weights (SGD with momentum)
@compute @workgroup_size(256)
fn momentum_update_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.input_size) {
        return;
    }
    
    // Momentum stored in gradients_input buffer for this kernel
    let momentum_decay = uniforms.alpha; // Typically 0.9
    let gradient = weight_gradients[index];
    let momentum = gradients_input[index];
    
    // Update momentum: momentum = momentum_decay * momentum + gradient
    let new_momentum = momentum_decay * momentum + gradient;
    gradients_input[index] = new_momentum;
    
    // Update weight gradients with momentum
    weight_gradients[index] = new_momentum;
}

// Adam optimizer state update (simplified version)
@compute @workgroup_size(256)
fn adam_update_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.input_size) {
        return;
    }
    
    let gradient = weight_gradients[index];
    
    // Note: In full implementation, we would need separate buffers for m and v
    // For now, this is a placeholder for the Adam update logic
    let beta1 = 0.9;  // First moment decay
    let beta2 = 0.999; // Second moment decay
    let epsilon = 1e-8;
    
    // m = beta1 * m + (1 - beta1) * gradient
    // v = beta2 * v + (1 - beta2) * gradient^2
    // gradient_adjusted = m / (sqrt(v) + epsilon)
    
    // Simplified update (would need proper state management in real implementation)
    let gradient_squared = gradient * gradient;
    let adjusted_gradient = gradient / (sqrt(gradient_squared) + epsilon);
    
    weight_gradients[index] = adjusted_gradient * uniforms.learning_rate;
}

// Batch normalization gradient computation
@compute @workgroup_size(256)
fn batch_norm_gradient_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let feature_idx = global_id.x;
    
    if (feature_idx >= uniforms.input_size) {
        return;
    }
    
    // Simplified batch norm gradient computation
    // In practice, this would require mean and variance computations
    var gradient_sum: f32 = 0.0;
    var mean: f32 = 0.0;
    
    // Compute mean activation for this feature
    for (var batch_idx = 0u; batch_idx < uniforms.batch_size; batch_idx++) {
        let activation_idx = batch_idx * uniforms.input_size + feature_idx;
        mean += activations[activation_idx];
    }
    mean /= f32(uniforms.batch_size);
    
    // Compute gradient (simplified)
    for (var batch_idx = 0u; batch_idx < uniforms.batch_size; batch_idx++) {
        let activation_idx = batch_idx * uniforms.input_size + feature_idx;
        let grad_idx = batch_idx * uniforms.input_size + feature_idx;
        
        let centered = activations[activation_idx] - mean;
        gradient_sum += gradients_output[grad_idx] * centered;
    }
    
    // Store normalized gradient
    gradients_input[feature_idx] = gradient_sum / f32(uniforms.batch_size);
}