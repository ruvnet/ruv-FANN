// WebGPU Compute Shaders: Neural Network Activation Functions
// High-performance GPU implementations of common activation functions
// Optimized for batch processing with vectorized operations

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Uniforms {
    length: u32,
    steepness: f32,
    alpha: f32,      // For parameterized functions (e.g., Leaky ReLU alpha)
    reserved: u32,
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

// Sigmoid activation: f(x) = 1 / (1 + exp(-steepness * x))
@compute @workgroup_size(256)
fn sigmoid_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    
    // Clamp to prevent overflow
    let clamped_x = clamp(x, -10.0, 10.0);
    
    // Use optimized sigmoid approximation for better performance
    // f(x) = 1 / (1 + exp(-x)) with fast exp approximation
    output[index] = 1.0 / (1.0 + exp(-clamped_x));
}

// ReLU activation: f(x) = max(0, x)
@compute @workgroup_size(256)
fn relu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    output[index] = max(0.0, input[index]);
}

// Leaky ReLU activation: f(x) = x if x > 0, alpha * x if x <= 0
@compute @workgroup_size(256)
fn leaky_relu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index];
    output[index] = select(uniforms.alpha * x, x, x > 0.0);
}

// Hyperbolic tangent: f(x) = tanh(steepness * x)
@compute @workgroup_size(256)
fn tanh_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    
    // Clamp to prevent overflow
    let clamped_x = clamp(x, -5.0, 5.0);
    
    output[index] = tanh(clamped_x);
}

// Linear activation: f(x) = steepness * x
@compute @workgroup_size(256)
fn linear_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    output[index] = input[index] * uniforms.steepness;
}

// Gaussian activation: f(x) = exp(-x^2 * steepness^2)
@compute @workgroup_size(256)
fn gaussian_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = exp(-x * x);
}

// Symmetric Gaussian: f(x) = exp(-x^2 * steepness^2) * 2 - 1
@compute @workgroup_size(256)
fn gaussian_symmetric_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = exp(-x * x) * 2.0 - 1.0;
}

// Elliott activation: f(x) = ((x * steepness) / 2) / (1 + |x * steepness|) + 0.5
@compute @workgroup_size(256)
fn elliott_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = (x / 2.0) / (1.0 + abs(x)) + 0.5;
}

// Symmetric Elliott: f(x) = (x * steepness) / (1 + |x * steepness|)
@compute @workgroup_size(256)
fn elliott_symmetric_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = x / (1.0 + abs(x));
}

// Sine activation: f(x) = sin(x * steepness) / 2 + 0.5
@compute @workgroup_size(256)
fn sin_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = sin(x) / 2.0 + 0.5;
}

// Cosine activation: f(x) = cos(x * steepness) / 2 + 0.5
@compute @workgroup_size(256)
fn cos_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = cos(x) / 2.0 + 0.5;
}

// Symmetric Sine: f(x) = sin(x * steepness)
@compute @workgroup_size(256)
fn sin_symmetric_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = sin(x);
}

// Symmetric Cosine: f(x) = cos(x * steepness)
@compute @workgroup_size(256)
fn cos_symmetric_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = cos(x);
}

// Bounded Linear (Linear Piece): f(x) = max(0, min(1, x * steepness))
@compute @workgroup_size(256)
fn linear_piece_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = clamp(x, 0.0, 1.0);
}

// Symmetric Bounded Linear: f(x) = max(-1, min(1, x * steepness))
@compute @workgroup_size(256)
fn linear_piece_symmetric_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    let x = input[index] * uniforms.steepness;
    output[index] = clamp(x, -1.0, 1.0);
}

// Threshold activation: f(x) = 0 if x < 0, 1 if x >= 0
@compute @workgroup_size(256)
fn threshold_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    output[index] = select(0.0, 1.0, input[index] >= 0.0);
}

// Symmetric Threshold: f(x) = -1 if x < 0, 1 if x >= 0
@compute @workgroup_size(256)
fn threshold_symmetric_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.length) {
        return;
    }
    
    output[index] = select(-1.0, 1.0, input[index] >= 0.0);
}