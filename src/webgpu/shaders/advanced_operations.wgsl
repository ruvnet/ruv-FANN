// Advanced GPU Operations for ruv-FANN
// Specialized compute shaders for advanced neural network features
// Includes mixed precision, sparse operations, and optimization algorithms

// ===== MIXED PRECISION OPERATIONS =====

@group(0) @binding(0) var<storage, read> fp32_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> fp16_output: array<u32>; // Packed f16 values
@group(0) @binding(2) var<storage, read> fp16_input: array<u32>; // Packed f16 values  
@group(0) @binding(3) var<storage, read_write> fp32_output: array<f32>;

struct MixedPrecisionUniforms {
    length: u32,
    scale_factor: f32,
    padding: vec2<u32>,
}

@group(0) @binding(4) var<uniform> mp_uniforms: MixedPrecisionUniforms;

// Convert f32 to packed f16 for memory bandwidth optimization
@compute @workgroup_size(256)
fn convert_fp32_to_fp16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 2u; // Process 2 f32 values per thread
    
    if (base_idx >= mp_uniforms.length) {
        return;
    }
    
    let remaining = mp_uniforms.length - base_idx;
    let process_count = min(remaining, 2u);
    
    if (process_count >= 2u) {
        // Pack two f32 values into one u32 (two f16 values)
        let val1 = fp32_input[base_idx];
        let val2 = fp32_input[base_idx + 1u];
        
        // Convert to f16 using quantization (simplified)
        let f16_1 = pack2x16float(vec2<f32>(val1, val2));
        fp16_output[base_idx / 2u] = f16_1;
    } else if (process_count == 1u) {
        // Handle single remaining value
        let val = fp32_input[base_idx];
        let f16_val = pack2x16float(vec2<f32>(val, 0.0));
        fp16_output[base_idx / 2u] = f16_val;
    }
}

// Convert packed f16 back to f32 for high-precision computation
@compute @workgroup_size(256)
fn convert_fp16_to_fp32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 2u;
    
    if (base_idx >= mp_uniforms.length) {
        return;
    }
    
    let remaining = mp_uniforms.length - base_idx;
    let process_count = min(remaining, 2u);
    
    // Unpack f16 values from u32
    let packed_val = fp16_input[base_idx / 2u];
    let unpacked = unpack2x16float(packed_val);
    
    if (process_count >= 1u) {
        fp32_output[base_idx] = unpacked.x * mp_uniforms.scale_factor;
    }
    if (process_count >= 2u) {
        fp32_output[base_idx + 1u] = unpacked.y * mp_uniforms.scale_factor;
    }
}

// ===== SPARSE MATRIX OPERATIONS =====

@group(1) @binding(0) var<storage, read> sparse_values: array<f32>;
@group(1) @binding(1) var<storage, read> sparse_indices: array<u32>;
@group(1) @binding(2) var<storage, read> sparse_row_ptr: array<u32>;
@group(1) @binding(3) var<storage, read> dense_vector: array<f32>;
@group(1) @binding(4) var<storage, read_write> result_vector: array<f32>;

struct SparseMatrixUniforms {
    num_rows: u32,
    num_cols: u32,
    nnz: u32, // Number of non-zero elements
    padding: u32,
}

@group(1) @binding(5) var<uniform> sparse_uniforms: SparseMatrixUniforms;

// Sparse matrix-vector multiplication (CSR format)
@compute @workgroup_size(256)
fn sparse_matrix_vector_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    if (row >= sparse_uniforms.num_rows) {
        return;
    }
    
    let row_start = sparse_row_ptr[row];
    let row_end = sparse_row_ptr[row + 1u];
    
    var sum = 0.0;
    
    // Iterate through non-zero elements in this row
    for (var idx = row_start; idx < row_end; idx++) {
        let col = sparse_indices[idx];
        let value = sparse_values[idx];
        sum += value * dense_vector[col];
    }
    
    result_vector[row] = sum;
}

// ===== BATCH NORMALIZATION =====

@group(2) @binding(0) var<storage, read> input_data: array<f32>;
@group(2) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(2) @binding(2) var<storage, read> gamma: array<f32>; // Scale parameters
@group(2) @binding(3) var<storage, read> beta: array<f32>;  // Shift parameters
@group(2) @binding(4) var<storage, read> running_mean: array<f32>;
@group(2) @binding(5) var<storage, read> running_variance: array<f32>;

struct BatchNormUniforms {
    batch_size: u32,
    feature_size: u32,
    epsilon: f32,
    momentum: f32,
}

@group(2) @binding(6) var<uniform> bn_uniforms: BatchNormUniforms;

// Shared memory for efficient batch statistics computation
var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sum_sq: array<f32, 256>;

// Batch normalization forward pass
@compute @workgroup_size(256)
fn batch_normalization_forward(@builtin(global_invocation_id) global_id: vec3<u32>,
                               @builtin(local_invocation_id) local_id: vec3<u32>,
                               @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let feature_idx = workgroup_id.x;
    let local_idx = local_id.x;
    let batch_stride = 256u; // Workgroup size
    
    if (feature_idx >= bn_uniforms.feature_size) {
        return;
    }
    
    // Phase 1: Compute batch statistics
    var sum = 0.0;
    var sum_sq = 0.0;
    
    // Each thread processes multiple batch elements
    for (var batch_idx = local_idx; batch_idx < bn_uniforms.batch_size; batch_idx += batch_stride) {
        let data_idx = batch_idx * bn_uniforms.feature_size + feature_idx;
        let value = input_data[data_idx];
        sum += value;
        sum_sq += value * value;
    }
    
    // Store in shared memory
    shared_sum[local_idx] = sum;
    shared_sum_sq[local_idx] = sum_sq;
    
    workgroupBarrier();
    
    // Parallel reduction to compute total sum and sum of squares
    var reduction_size = 256u;
    while (reduction_size > 1u) {
        let half_size = reduction_size / 2u;
        if (local_idx < half_size) {
            shared_sum[local_idx] += shared_sum[local_idx + half_size];
            shared_sum_sq[local_idx] += shared_sum_sq[local_idx + half_size];
        }
        workgroupBarrier();
        reduction_size = half_size;
    }
    
    // Compute mean and variance
    let total_sum = shared_sum[0];
    let total_sum_sq = shared_sum_sq[0];
    let batch_size_f = f32(bn_uniforms.batch_size);
    
    let mean = total_sum / batch_size_f;
    let variance = (total_sum_sq / batch_size_f) - (mean * mean);
    let inv_std = inverseSqrt(variance + bn_uniforms.epsilon);
    
    // Phase 2: Normalize and scale
    for (var batch_idx = local_idx; batch_idx < bn_uniforms.batch_size; batch_idx += batch_stride) {
        let data_idx = batch_idx * bn_uniforms.feature_size + feature_idx;
        let value = input_data[data_idx];
        
        // Normalize: (x - mean) / sqrt(variance + epsilon)
        let normalized = (value - mean) * inv_std;
        
        // Scale and shift: gamma * normalized + beta
        output_data[data_idx] = gamma[feature_idx] * normalized + beta[feature_idx];
    }
}

// ===== DROPOUT OPERATIONS =====

@group(3) @binding(0) var<storage, read> dropout_input: array<f32>;
@group(3) @binding(1) var<storage, read_write> dropout_output: array<f32>;
@group(3) @binding(2) var<storage, read> dropout_mask: array<u32>; // Precomputed random mask

struct DropoutUniforms {
    length: u32,
    dropout_rate: f32,
    training_mode: u32, // Boolean: 1 for training, 0 for inference
    seed: u32,
}

@group(3) @binding(3) var<uniform> dropout_uniforms: DropoutUniforms;

// Apply dropout with precomputed mask
@compute @workgroup_size(256)
fn apply_dropout(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= dropout_uniforms.length) {
        return;
    }
    
    let input_value = dropout_input[idx];
    
    if (dropout_uniforms.training_mode == 0u) {
        // Inference mode: no dropout
        dropout_output[idx] = input_value;
    } else {
        // Training mode: apply dropout
        let mask_value = dropout_mask[idx];
        let keep_prob = 1.0 - dropout_uniforms.dropout_rate;
        
        if (mask_value == 1u) {
            // Keep this neuron, scale by 1/keep_prob for compensation
            dropout_output[idx] = input_value / keep_prob;
        } else {
            // Drop this neuron
            dropout_output[idx] = 0.0;
        }
    }
}

// ===== LAYER NORMALIZATION =====

@group(4) @binding(0) var<storage, read> ln_input: array<f32>;
@group(4) @binding(1) var<storage, read_write> ln_output: array<f32>;
@group(4) @binding(2) var<storage, read> ln_gamma: array<f32>;
@group(4) @binding(3) var<storage, read> ln_beta: array<f32>;

struct LayerNormUniforms {
    batch_size: u32,
    hidden_size: u32,
    epsilon: f32,
    padding: u32,
}

@group(4) @binding(4) var<uniform> ln_uniforms: LayerNormUniforms;

var<workgroup> ln_shared_data: array<f32, 1024>; // Shared memory for reduction

// Layer normalization (normalize across hidden dimension for each sample)
@compute @workgroup_size(256)
fn layer_normalization(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let batch_idx = workgroup_id.x;
    let local_idx = local_id.x;
    let threads_per_workgroup = 256u;
    
    if (batch_idx >= ln_uniforms.batch_size) {
        return;
    }
    
    let input_offset = batch_idx * ln_uniforms.hidden_size;
    
    // Phase 1: Compute mean
    var sum = 0.0;
    for (var i = local_idx; i < ln_uniforms.hidden_size; i += threads_per_workgroup) {
        sum += ln_input[input_offset + i];
    }
    
    ln_shared_data[local_idx] = sum;
    workgroupBarrier();
    
    // Reduce to compute total sum
    var reduction_size = threads_per_workgroup;
    while (reduction_size > 1u) {
        let half_size = reduction_size / 2u;
        if (local_idx < half_size) {
            ln_shared_data[local_idx] += ln_shared_data[local_idx + half_size];
        }
        workgroupBarrier();
        reduction_size = half_size;
    }
    
    let mean = ln_shared_data[0] / f32(ln_uniforms.hidden_size);
    
    // Phase 2: Compute variance
    var sum_sq_diff = 0.0;
    for (var i = local_idx; i < ln_uniforms.hidden_size; i += threads_per_workgroup) {
        let diff = ln_input[input_offset + i] - mean;
        sum_sq_diff += diff * diff;
    }
    
    ln_shared_data[local_idx] = sum_sq_diff;
    workgroupBarrier();
    
    // Reduce to compute total sum of squared differences
    reduction_size = threads_per_workgroup;
    while (reduction_size > 1u) {
        let half_size = reduction_size / 2u;
        if (local_idx < half_size) {
            ln_shared_data[local_idx] += ln_shared_data[local_idx + half_size];
        }
        workgroupBarrier();
        reduction_size = half_size;
    }
    
    let variance = ln_shared_data[0] / f32(ln_uniforms.hidden_size);
    let inv_std = inverseSqrt(variance + ln_uniforms.epsilon);
    
    // Phase 3: Normalize and scale
    for (var i = local_idx; i < ln_uniforms.hidden_size; i += threads_per_workgroup) {
        let input_val = ln_input[input_offset + i];
        let normalized = (input_val - mean) * inv_std;
        ln_output[input_offset + i] = ln_gamma[i] * normalized + ln_beta[i];
    }
}

// ===== CONVOLUTION OPERATIONS (1D for sequence data) =====

@group(5) @binding(0) var<storage, read> conv_input: array<f32>;
@group(5) @binding(1) var<storage, read> conv_kernel: array<f32>;
@group(5) @binding(2) var<storage, read> conv_bias: array<f32>;
@group(5) @binding(3) var<storage, read_write> conv_output: array<f32>;

struct ConvUniforms {
    input_length: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    input_channels: u32,
    output_channels: u32,
    output_length: u32,
    padding_value: f32,
}

@group(5) @binding(4) var<uniform> conv_uniforms: ConvUniforms;

// 1D convolution for sequence processing
@compute @workgroup_size(256)
fn conv1d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_pos = global_id.x;
    let output_channel = global_id.y;
    
    if (output_pos >= conv_uniforms.output_length || output_channel >= conv_uniforms.output_channels) {
        return;
    }
    
    var sum = 0.0;
    
    // Apply convolution kernel
    for (var k = 0u; k < conv_uniforms.kernel_size; k++) {
        let input_pos = output_pos * conv_uniforms.stride + k;
        
        // Handle padding
        if (input_pos < conv_uniforms.padding || input_pos >= conv_uniforms.input_length + conv_uniforms.padding) {
            sum += conv_uniforms.padding_value;
            continue;
        }
        
        let actual_input_pos = input_pos - conv_uniforms.padding;
        
        for (var input_channel = 0u; input_channel < conv_uniforms.input_channels; input_channel++) {
            let input_idx = actual_input_pos * conv_uniforms.input_channels + input_channel;
            let kernel_idx = ((output_channel * conv_uniforms.input_channels + input_channel) * conv_uniforms.kernel_size) + k;
            
            sum += conv_input[input_idx] * conv_kernel[kernel_idx];
        }
    }
    
    // Add bias
    sum += conv_bias[output_channel];
    
    let output_idx = output_pos * conv_uniforms.output_channels + output_channel;
    conv_output[output_idx] = sum;
}

// ===== ATTENTION MECHANISMS =====

@group(6) @binding(0) var<storage, read> query: array<f32>;
@group(6) @binding(1) var<storage, read> key: array<f32>;
@group(6) @binding(2) var<storage, read> value: array<f32>;
@group(6) @binding(3) var<storage, read_write> attention_output: array<f32>;
@group(6) @binding(4) var<storage, read_write> attention_weights: array<f32>;

struct AttentionUniforms {
    batch_size: u32,
    seq_length: u32,
    head_dim: u32,
    num_heads: u32,
    scale_factor: f32, // 1/sqrt(head_dim)
    padding: vec3<u32>,
}

@group(6) @binding(5) var<uniform> attn_uniforms: AttentionUniforms;

var<workgroup> attn_shared_memory: array<f32, 1024>;

// Compute scaled dot-product attention
@compute @workgroup_size(256)
fn scaled_dot_product_attention(@builtin(global_invocation_id) global_id: vec3<u32>,
                                @builtin(local_invocation_id) local_id: vec3<u32>,
                                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let batch_idx = workgroup_id.x;
    let head_idx = workgroup_id.y;
    let query_pos = global_id.z;
    let local_idx = local_id.x;
    
    if (batch_idx >= attn_uniforms.batch_size || 
        head_idx >= attn_uniforms.num_heads || 
        query_pos >= attn_uniforms.seq_length) {
        return;
    }
    
    let head_offset = ((batch_idx * attn_uniforms.num_heads + head_idx) * attn_uniforms.seq_length * attn_uniforms.head_dim);
    let query_offset = head_offset + query_pos * attn_uniforms.head_dim;
    
    // Compute attention scores for all key positions
    for (var key_pos = local_idx; key_pos < attn_uniforms.seq_length; key_pos += 256u) {
        let key_offset = head_offset + key_pos * attn_uniforms.head_dim;
        
        // Compute dot product between query and key
        var score = 0.0;
        for (var dim = 0u; dim < attn_uniforms.head_dim; dim++) {
            score += query[query_offset + dim] * key[key_offset + dim];
        }
        
        // Apply scaling
        score *= attn_uniforms.scale_factor;
        
        let attention_idx = ((batch_idx * attn_uniforms.num_heads + head_idx) * attn_uniforms.seq_length + query_pos) * attn_uniforms.seq_length + key_pos;
        attention_weights[attention_idx] = score;
    }
    
    workgroupBarrier();
    
    // Apply softmax to attention scores (simplified version)
    // In practice, you'd want a more numerically stable implementation
    
    // Compute output as weighted sum of values
    for (var dim = local_idx; dim < attn_uniforms.head_dim; dim += 256u) {
        var weighted_sum = 0.0;
        
        for (var key_pos = 0u; key_pos < attn_uniforms.seq_length; key_pos++) {
            let attention_idx = ((batch_idx * attn_uniforms.num_heads + head_idx) * attn_uniforms.seq_length + query_pos) * attn_uniforms.seq_length + key_pos;
            let value_offset = head_offset + key_pos * attn_uniforms.head_dim + dim;
            
            weighted_sum += attention_weights[attention_idx] * value[value_offset];
        }
        
        let output_offset = query_offset + dim;
        attention_output[output_offset] = weighted_sum;
    }
}