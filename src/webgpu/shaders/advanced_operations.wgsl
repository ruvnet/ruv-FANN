// WebGPU Compute Shaders: Advanced Neural Network Operations
// High-performance GPU implementations of specialized neural operations
// Includes convolution, pooling, attention, and optimization kernels

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> scratch_buffer: array<f32>;

struct AdvancedUniforms {
    input_height: u32,
    input_width: u32,
    input_channels: u32,
    output_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    batch_size: u32,
    sequence_length: u32,    // For attention mechanisms
    head_dimension: u32,     // For multi-head attention
    num_heads: u32,
    scale_factor: f32,       // For attention scaling
}

@group(0) @binding(5) var<uniform> uniforms: AdvancedUniforms;

// 2D Convolution operation optimized for GPU
@compute @workgroup_size(16, 16, 1)
fn conv2d_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let out_y = global_id.y;
    let out_x = global_id.z;
    
    if (batch_idx >= uniforms.batch_size || 
        out_y >= uniforms.input_height || 
        out_x >= uniforms.input_width) {
        return;
    }
    
    let output_height = (uniforms.input_height + 2u * uniforms.padding - uniforms.kernel_size) / uniforms.stride + 1u;
    let output_width = (uniforms.input_width + 2u * uniforms.padding - uniforms.kernel_size) / uniforms.stride + 1u;
    
    if (out_y >= output_height || out_x >= output_width) {
        return;
    }
    
    // Compute convolution for each output channel
    for (var out_c = 0u; out_c < uniforms.output_channels; out_c++) {
        var sum: f32 = 0.0;
        
        // Convolve with kernel
        for (var ky = 0u; ky < uniforms.kernel_size; ky++) {
            for (var kx = 0u; kx < uniforms.kernel_size; kx++) {
                for (var in_c = 0u; in_c < uniforms.input_channels; in_c++) {
                    let in_y = out_y * uniforms.stride + ky - uniforms.padding;
                    let in_x = out_x * uniforms.stride + kx - uniforms.padding;
                    
                    // Check bounds (padding)
                    if (in_y < uniforms.input_height && in_x < uniforms.input_width) {
                        let input_idx = batch_idx * (uniforms.input_channels * uniforms.input_height * uniforms.input_width) +
                                      in_c * (uniforms.input_height * uniforms.input_width) +
                                      in_y * uniforms.input_width + in_x;
                        
                        let weight_idx = out_c * (uniforms.input_channels * uniforms.kernel_size * uniforms.kernel_size) +
                                       in_c * (uniforms.kernel_size * uniforms.kernel_size) +
                                       ky * uniforms.kernel_size + kx;
                        
                        sum += input_data[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        
        // Add bias and store result
        let output_idx = batch_idx * (uniforms.output_channels * output_height * output_width) +
                        out_c * (output_height * output_width) +
                        out_y * output_width + out_x;
        
        output_data[output_idx] = sum + bias[out_c];
    }
}

// Max pooling operation
@compute @workgroup_size(16, 16, 1)
fn max_pool2d_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let out_y = global_id.y;
    let out_x = global_id.z;
    
    let pool_size = uniforms.kernel_size;
    let output_height = uniforms.input_height / pool_size;
    let output_width = uniforms.input_width / pool_size;
    
    if (batch_idx >= uniforms.batch_size || 
        out_y >= output_height || 
        out_x >= output_width) {
        return;
    }
    
    for (var c = 0u; c < uniforms.input_channels; c++) {
        var max_val: f32 = -3.4e38; // Negative infinity approximation
        
        // Find maximum in pool window
        for (var py = 0u; py < pool_size; py++) {
            for (var px = 0u; px < pool_size; px++) {
                let in_y = out_y * pool_size + py;
                let in_x = out_x * pool_size + px;
                
                let input_idx = batch_idx * (uniforms.input_channels * uniforms.input_height * uniforms.input_width) +
                              c * (uniforms.input_height * uniforms.input_width) +
                              in_y * uniforms.input_width + in_x;
                
                max_val = max(max_val, input_data[input_idx]);
            }
        }
        
        let output_idx = batch_idx * (uniforms.input_channels * output_height * output_width) +
                        c * (output_height * output_width) +
                        out_y * output_width + out_x;
        
        output_data[output_idx] = max_val;
    }
}

// Average pooling operation
@compute @workgroup_size(16, 16, 1)
fn avg_pool2d_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let out_y = global_id.y;
    let out_x = global_id.z;
    
    let pool_size = uniforms.kernel_size;
    let output_height = uniforms.input_height / pool_size;
    let output_width = uniforms.input_width / pool_size;
    
    if (batch_idx >= uniforms.batch_size || 
        out_y >= output_height || 
        out_x >= output_width) {
        return;
    }
    
    for (var c = 0u; c < uniforms.input_channels; c++) {
        var sum: f32 = 0.0;
        
        // Sum values in pool window
        for (var py = 0u; py < pool_size; py++) {
            for (var px = 0u; px < pool_size; px++) {
                let in_y = out_y * pool_size + py;
                let in_x = out_x * pool_size + px;
                
                let input_idx = batch_idx * (uniforms.input_channels * uniforms.input_height * uniforms.input_width) +
                              c * (uniforms.input_height * uniforms.input_width) +
                              in_y * uniforms.input_width + in_x;
                
                sum += input_data[input_idx];
            }
        }
        
        // Average and store
        let avg_val = sum / f32(pool_size * pool_size);
        let output_idx = batch_idx * (uniforms.input_channels * output_height * output_width) +
                        c * (output_height * output_width) +
                        out_y * output_width + out_x;
        
        output_data[output_idx] = avg_val;
    }
}

// Softmax operation with numerical stability
@compute @workgroup_size(256)
fn softmax_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= uniforms.batch_size) {
        return;
    }
    
    let vector_size = uniforms.input_channels;
    let base_idx = batch_idx * vector_size;
    
    // Find maximum for numerical stability
    var max_val: f32 = -3.4e38;
    for (var i = 0u; i < vector_size; i++) {
        max_val = max(max_val, input_data[base_idx + i]);
    }
    
    // Compute sum of exponentials
    var exp_sum: f32 = 0.0;
    for (var i = 0u; i < vector_size; i++) {
        let exp_val = exp(input_data[base_idx + i] - max_val);
        scratch_buffer[base_idx + i] = exp_val;
        exp_sum += exp_val;
    }
    
    // Normalize
    for (var i = 0u; i < vector_size; i++) {
        output_data[base_idx + i] = scratch_buffer[base_idx + i] / exp_sum;
    }
}

// Layer normalization
@compute @workgroup_size(256)
fn layer_norm_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= uniforms.batch_size) {
        return;
    }
    
    let feature_size = uniforms.input_channels;
    let base_idx = batch_idx * feature_size;
    let epsilon = 1e-5;
    
    // Compute mean
    var mean: f32 = 0.0;
    for (var i = 0u; i < feature_size; i++) {
        mean += input_data[base_idx + i];
    }
    mean /= f32(feature_size);
    
    // Compute variance
    var variance: f32 = 0.0;
    for (var i = 0u; i < feature_size; i++) {
        let diff = input_data[base_idx + i] - mean;
        variance += diff * diff;
    }
    variance /= f32(feature_size);
    
    // Normalize
    let inv_std = 1.0 / sqrt(variance + epsilon);
    for (var i = 0u; i < feature_size; i++) {
        let normalized = (input_data[base_idx + i] - mean) * inv_std;
        
        // Apply learned scale and bias if available
        let scale = select(1.0, weights[i], i < uniforms.input_channels);
        let bias_val = select(0.0, bias[i], i < uniforms.input_channels);
        
        output_data[base_idx + i] = normalized * scale + bias_val;
    }
}

// Scaled dot-product attention (simplified)
@compute @workgroup_size(16, 16)
fn scaled_dot_product_attention(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let head_idx = global_id.y;
    let seq_pos = global_id.z;
    
    if (batch_idx >= uniforms.batch_size || 
        head_idx >= uniforms.num_heads || 
        seq_pos >= uniforms.sequence_length) {
        return;
    }
    
    let head_dim = uniforms.head_dimension;
    let scale = uniforms.scale_factor;
    
    // This is a simplified version - full attention would require Q, K, V matrices
    let base_offset = batch_idx * uniforms.num_heads * uniforms.sequence_length * head_dim +
                     head_idx * uniforms.sequence_length * head_dim +
                     seq_pos * head_dim;
    
    // Compute attention scores (simplified)
    var attention_sum: f32 = 0.0;
    for (var i = 0u; i < uniforms.sequence_length; i++) {
        let query_offset = base_offset;
        let key_offset = batch_idx * uniforms.num_heads * uniforms.sequence_length * head_dim +
                        head_idx * uniforms.sequence_length * head_dim +
                        i * head_dim;
        
        // Compute dot product
        var dot_product: f32 = 0.0;
        for (var d = 0u; d < head_dim; d++) {
            dot_product += input_data[query_offset + d] * weights[key_offset + d];
        }
        
        let attention_score = exp(dot_product * scale);
        scratch_buffer[seq_pos * uniforms.sequence_length + i] = attention_score;
        attention_sum += attention_score;
    }
    
    // Normalize attention weights and compute output
    for (var d = 0u; d < head_dim; d++) {
        var weighted_sum: f32 = 0.0;
        
        for (var i = 0u; i < uniforms.sequence_length; i++) {
            let attention_weight = scratch_buffer[seq_pos * uniforms.sequence_length + i] / attention_sum;
            let value_offset = batch_idx * uniforms.num_heads * uniforms.sequence_length * head_dim +
                              head_idx * uniforms.sequence_length * head_dim +
                              i * head_dim + d;
            
            weighted_sum += attention_weight * bias[value_offset]; // Reuse bias buffer for values
        }
        
        output_data[base_offset + d] = weighted_sum;
    }
}

// Element-wise operations
@compute @workgroup_size(256)
fn element_wise_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.batch_size * uniforms.input_channels) {
        return;
    }
    
    output_data[index] = input_data[index] + weights[index];
}

@compute @workgroup_size(256)
fn element_wise_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.batch_size * uniforms.input_channels) {
        return;
    }
    
    output_data[index] = input_data[index] * weights[index];
}

// GELU activation: f(x) = x * Phi(x) where Phi is cumulative distribution function
@compute @workgroup_size(256)
fn gelu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.batch_size * uniforms.input_channels) {
        return;
    }
    
    let x = input_data[index];
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608;
    let a = 0.044715;
    
    let x_cubed = x * x * x;
    let inner = sqrt_2_over_pi * (x + a * x_cubed);
    let gelu_approx = 0.5 * x * (1.0 + tanh(inner));
    
    output_data[index] = gelu_approx;
}

// Swish/SiLU activation: f(x) = x * sigmoid(x)
@compute @workgroup_size(256)
fn swish_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= uniforms.batch_size * uniforms.input_channels) {
        return;
    }
    
    let x = input_data[index];
    let sigmoid_x = 1.0 / (1.0 + exp(-x));
    
    output_data[index] = x * sigmoid_x;
}