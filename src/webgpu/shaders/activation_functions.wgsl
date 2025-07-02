// WebGPU Compute Shaders: High-Performance Activation Functions
// Optimized neural network activation functions with numerical stability
// Supports all FANN activation functions with GPU-specific optimizations
// Target: 10-30x speedup over CPU implementations

// ===== SHARED UNIFORMS =====

struct ActivationUniforms {
    length: u32,
    steepness: f32,
    alpha: f32,        // For parameterized functions (Leaky ReLU, ELU)
    beta: f32,         // Additional parameter for advanced functions
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: ActivationUniforms;

// ===== OPTIMIZED SIGMOID IMPLEMENTATIONS =====

// Fast sigmoid approximation using rational function
// More numerically stable than exp-based implementation
@compute @workgroup_size(256)
fn sigmoid_fast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    // Process 4 elements vectorized
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        
        // Fast sigmoid approximation: 0.5 * (x / (1.0 + abs(x))) + 0.5
        // More stable than exp and faster to compute
        output[idx] = 0.5 * (x / (1.0 + abs(x))) + 0.5;
    }
}

// Precise sigmoid using optimized exp with clamping
@compute @workgroup_size(256)
fn sigmoid_precise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        
        // Numerically stable sigmoid implementation
        // Prevents overflow by using different formulations for positive/negative x
        if (x >= 0.0) {
            let exp_neg_x = exp(-min(x, 88.0)); // Clamp to prevent overflow
            output[idx] = 1.0 / (1.0 + exp_neg_x);
        } else {
            let exp_x = exp(max(x, -88.0)); // Clamp to prevent underflow
            output[idx] = exp_x / (1.0 + exp_x);
        }
    }
}

// ===== RELU FAMILY =====

// Standard ReLU with vectorized processing
@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    // Vectorized ReLU processing - 4 elements at once
    let remaining = uniforms.length - base_idx;
    let vec_count = min(remaining, 4u) / 4u;
    
    if (vec_count > 0u) {
        let input_vec = vec4<f32>(
            input[base_idx],
            input[base_idx + 1u],
            input[base_idx + 2u],
            input[base_idx + 3u]
        );
        
        let result_vec = max(vec4<f32>(0.0), input_vec);
        
        output[base_idx] = result_vec.x;
        output[base_idx + 1u] = result_vec.y;
        output[base_idx + 2u] = result_vec.z;
        output[base_idx + 3u] = result_vec.w;
    } else {
        // Handle remaining elements
        for (var i = 0u; i < remaining; i++) {
            let idx = base_idx + i;
            output[idx] = max(0.0, input[idx]);
        }
    }
}

// Leaky ReLU: f(x) = x if x > 0, alpha * x if x <= 0
@compute @workgroup_size(256)
fn leaky_relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx];
        // Branchless leaky ReLU using select
        output[idx] = select(uniforms.alpha * x, x, x > 0.0);
    }
}

// ELU (Exponential Linear Unit): f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
@compute @workgroup_size(256)
fn elu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx];
        
        if (x > 0.0) {
            output[idx] = x;
        } else {
            // Clamp input to prevent overflow
            let clamped_x = max(x, -88.0);
            output[idx] = uniforms.alpha * (exp(clamped_x) - 1.0);
        }
    }
}

// Swish/SiLU: f(x) = x * sigmoid(beta * x)
@compute @workgroup_size(256)
fn swish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx];
        let beta_x = x * uniforms.beta;
        
        // Fast sigmoid approximation for better performance
        let sigmoid_val = 0.5 * (beta_x / (1.0 + abs(beta_x))) + 0.5;
        output[idx] = x * sigmoid_val;
    }
}

// GELU (Gaussian Error Linear Unit) - fast approximation
@compute @workgroup_size(256)
fn gelu_fast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx];
        
        // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let x3 = x * x * x;
        let inner = 0.797885 * (x + 0.044715 * x3); // sqrt(2/π) ≈ 0.797885
        output[idx] = 0.5 * x * (1.0 + tanh(inner));
    }
}

// ===== HYPERBOLIC FUNCTIONS =====

// Optimized tanh with numerical stability
@compute @workgroup_size(256)
fn tanh_stable(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        
        // Clamp to prevent overflow
        let clamped_x = clamp(x, -10.0, 10.0);
        output[idx] = tanh(clamped_x);
    }
}

// Fast tanh approximation using rational function
@compute @workgroup_size(256)
fn tanh_fast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        
        // Fast tanh approximation: x / (1 + abs(x))
        // Less accurate but much faster than true tanh
        output[idx] = x / (1.0 + abs(x));
    }
}

// ===== GAUSSIAN FUNCTIONS =====

// Gaussian activation with numerical stability
@compute @workgroup_size(256)
fn gaussian(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        
        // Gaussian: exp(-x²) with overflow protection
        let x_squared = x * x;
        let clamped_x_sq = min(x_squared, 88.0); // Prevent overflow
        output[idx] = exp(-clamped_x_sq);
    }
}

// Symmetric Gaussian: 2 * exp(-x²) - 1
@compute @workgroup_size(256)
fn gaussian_symmetric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        
        let x_squared = x * x;
        let clamped_x_sq = min(x_squared, 88.0);
        output[idx] = 2.0 * exp(-clamped_x_sq) - 1.0;
    }
}

// ===== LINEAR AND PIECE-WISE FUNCTIONS =====

// Linear activation with steepness
@compute @workgroup_size(256)
fn linear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    // Vectorized linear transformation
    let remaining = uniforms.length - base_idx;
    let vec_count = remaining / 4u;
    
    if (vec_count > 0u && base_idx + 4u <= uniforms.length) {
        let input_vec = vec4<f32>(
            input[base_idx],
            input[base_idx + 1u],
            input[base_idx + 2u],
            input[base_idx + 3u]
        );
        
        let result_vec = input_vec * uniforms.steepness;
        
        output[base_idx] = result_vec.x;
        output[base_idx + 1u] = result_vec.y;
        output[base_idx + 2u] = result_vec.z;
        output[base_idx + 3u] = result_vec.w;
    } else {
        // Handle remaining elements
        let process_count = min(remaining, 4u);
        for (var i = 0u; i < process_count; i++) {
            let idx = base_idx + i;
            output[idx] = input[idx] * uniforms.steepness;
        }
    }
}

// Bounded linear (Linear Piece): clamp(x * steepness, 0, 1)
@compute @workgroup_size(256)
fn linear_piece(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let scaled_input = input[idx] * uniforms.steepness;
        output[idx] = clamp(scaled_input, 0.0, 1.0);
    }
}

// Symmetric bounded linear: clamp(x * steepness, -1, 1)
@compute @workgroup_size(256)
fn linear_piece_symmetric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let scaled_input = input[idx] * uniforms.steepness;
        output[idx] = clamp(scaled_input, -1.0, 1.0);
    }
}

// ===== THRESHOLD FUNCTIONS =====

// Binary threshold: 0 if x < 0, 1 if x >= 0
@compute @workgroup_size(256)
fn threshold(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        output[idx] = select(0.0, 1.0, input[idx] >= 0.0);
    }
}

// Symmetric threshold: -1 if x < 0, 1 if x >= 0
@compute @workgroup_size(256)
fn threshold_symmetric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        output[idx] = select(-1.0, 1.0, input[idx] >= 0.0);
    }
}

// ===== TRIGONOMETRIC FUNCTIONS =====

// Sine activation with period control
@compute @workgroup_size(256)
fn sin_activation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        output[idx] = sin(x) * 0.5 + 0.5; // Scale to [0, 1]
    }
}

// Cosine activation with period control  
@compute @workgroup_size(256)
fn cos_activation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        output[idx] = cos(x) * 0.5 + 0.5; // Scale to [0, 1]
    }
}

// ===== ELLIOTT FUNCTIONS =====

// Elliott activation: ((x * steepness) / 2) / (1 + |x * steepness|) + 0.5
@compute @workgroup_size(256)
fn elliott(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        output[idx] = (x * 0.5) / (1.0 + abs(x)) + 0.5;
    }
}

// Symmetric Elliott: (x * steepness) / (1 + |x * steepness|)
@compute @workgroup_size(256)
fn elliott_symmetric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= uniforms.length) {
        return;
    }
    
    let remaining = uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        let x = input[idx] * uniforms.steepness;
        output[idx] = x / (1.0 + abs(x));
    }
}