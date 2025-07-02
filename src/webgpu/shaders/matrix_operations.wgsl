// WebGPU Compute Shaders: Optimized Matrix Operations
// High-performance implementations for neural network forward and backward passes
// Target: 5-50x speedup over CPU implementations
// Optimized for memory coalescing, vectorization, and workgroup efficiency

// ===== MATRIX-VECTOR MULTIPLICATION =====

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct MatrixVectorUniforms {
    rows: u32,
    cols: u32,
    padding: vec2<u32>,
}

@group(0) @binding(3) var<uniform> mv_uniforms: MatrixVectorUniforms;

// Optimized matrix-vector multiplication with vectorized loads
// Target workgroup size: 256 threads for optimal occupancy
@compute @workgroup_size(256)
fn matrix_vector_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    if (row >= mv_uniforms.rows) {
        return;
    }
    
    var sum = vec4<f32>(0.0);
    let cols = mv_uniforms.cols;
    let row_offset = row * cols;
    
    // Process 4 elements at a time for better memory bandwidth utilization
    let vec4_cols = cols / 4u;
    let remainder_start = vec4_cols * 4u;
    
    // Vectorized inner loop - processes 4 elements per iteration
    for (var i = 0u; i < vec4_cols; i++) {
        let col_base = i * 4u;
        let matrix_base = row_offset + col_base;
        
        // Load 4 consecutive matrix elements (coalesced access)
        let matrix_vec = vec4<f32>(
            matrix[matrix_base],
            matrix[matrix_base + 1u],
            matrix[matrix_base + 2u],
            matrix[matrix_base + 3u]
        );
        
        // Load 4 consecutive vector elements
        let vector_vec = vec4<f32>(
            vector[col_base],
            vector[col_base + 1u],
            vector[col_base + 2u],
            vector[col_base + 3u]
        );
        
        // Vectorized multiply-add
        sum += matrix_vec * vector_vec;
    }
    
    // Accumulate vectorized results
    var final_sum = sum.x + sum.y + sum.z + sum.w;
    
    // Handle remainder elements (if cols not divisible by 4)
    for (var col = remainder_start; col < cols; col++) {
        final_sum += matrix[row_offset + col] * vector[col];
    }
    
    result[row] = final_sum;
}

// ===== BATCH MATRIX-VECTOR MULTIPLICATION =====

@group(1) @binding(0) var<storage, read> batch_matrix: array<f32>;
@group(1) @binding(1) var<storage, read> batch_vectors: array<f32>;
@group(1) @binding(2) var<storage, read_write> batch_results: array<f32>;

struct BatchUniforms {
    rows: u32,
    cols: u32,
    batch_size: u32,
    padding: u32,
}

@group(1) @binding(3) var<uniform> batch_uniforms: BatchUniforms;

// Shared memory for matrix row caching (16KB shared memory on most GPUs)
var<workgroup> shared_matrix_row: array<f32, 1024>; // Cache up to 1024 elements

// 2D workgroup layout optimized for batch processing
// 16x16 = 256 threads, good occupancy on most hardware
@compute @workgroup_size(16, 16)
fn batch_matrix_vector_multiply(@builtin(global_invocation_id) global_id: vec3<u32>,
                                @builtin(local_invocation_id) local_id: vec3<u32>,
                                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let row = global_id.x;
    let batch_idx = global_id.y;
    let local_row = local_id.x;
    let local_batch = local_id.y;
    
    if (row >= batch_uniforms.rows || batch_idx >= batch_uniforms.batch_size) {
        return;
    }
    
    let cols = batch_uniforms.cols;
    let matrix_row_offset = row * cols;
    let vector_offset = batch_idx * cols;
    
    // Cooperative loading of matrix row into shared memory
    // Each thread in the workgroup loads part of the matrix row
    let threads_per_row = 16u; // workgroup_size.x
    let elements_per_thread = (cols + threads_per_row - 1u) / threads_per_row;
    
    // Load matrix row into shared memory cooperatively
    for (var i = 0u; i < elements_per_thread; i++) {
        let col_idx = local_row * elements_per_thread + i;
        if (col_idx < cols && col_idx < 1024u) {
            shared_matrix_row[col_idx] = batch_matrix[matrix_row_offset + col_idx];
        }
    }
    
    // Synchronize workgroup after loading shared data
    workgroupBarrier();
    
    // Compute dot product using shared matrix data
    var sum = vec4<f32>(0.0);
    let vec4_cols = min(cols, 1024u) / 4u;
    let remainder_start = vec4_cols * 4u;
    
    // Vectorized computation using shared memory
    for (var i = 0u; i < vec4_cols; i++) {
        let col_base = i * 4u;
        let vector_base = vector_offset + col_base;
        
        // Load from shared memory (very fast)
        let matrix_vec = vec4<f32>(
            shared_matrix_row[col_base],
            shared_matrix_row[col_base + 1u],
            shared_matrix_row[col_base + 2u],
            shared_matrix_row[col_base + 3u]
        );
        
        // Load vector elements from global memory
        let vector_vec = vec4<f32>(
            batch_vectors[vector_base],
            batch_vectors[vector_base + 1u],
            batch_vectors[vector_base + 2u],
            batch_vectors[vector_base + 3u]
        );
        
        sum += matrix_vec * vector_vec;
    }
    
    var final_sum = sum.x + sum.y + sum.z + sum.w;
    
    // Handle remainder and large matrices
    for (var col = remainder_start; col < cols; col++) {
        let matrix_val = select(batch_matrix[matrix_row_offset + col], 
                               shared_matrix_row[col], 
                               col < 1024u);
        final_sum += matrix_val * batch_vectors[vector_offset + col];
    }
    
    // Store result: results[batch_idx * rows + row]
    batch_results[batch_idx * batch_uniforms.rows + row] = final_sum;
}

// ===== MATRIX TRANSPOSE =====

@group(2) @binding(0) var<storage, read> input_matrix: array<f32>;
@group(2) @binding(1) var<storage, read_write> output_matrix: array<f32>;

struct TransposeUniforms {
    input_rows: u32,
    input_cols: u32,
    padding: vec2<u32>,
}

@group(2) @binding(2) var<uniform> transpose_uniforms: TransposeUniforms;

// Tiled matrix transpose for optimal memory access patterns
// Uses shared memory to minimize global memory transactions
var<workgroup> tile: array<array<f32, 16>, 16>; // 16x16 tile in shared memory

@compute @workgroup_size(16, 16)
fn matrix_transpose(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>) {
    let input_row = global_id.x;
    let input_col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;
    
    let input_rows = transpose_uniforms.input_rows;
    let input_cols = transpose_uniforms.input_cols;
    
    // Load tile into shared memory (coalesced reads)
    if (input_row < input_rows && input_col < input_cols) {
        tile[local_row][local_col] = input_matrix[input_row * input_cols + input_col];
    } else {
        tile[local_row][local_col] = 0.0;
    }
    
    workgroupBarrier();
    
    // Compute transposed coordinates
    let output_row = global_id.y; // input_col becomes output_row
    let output_col = global_id.x; // input_row becomes output_col
    
    // Write transposed tile (coalesced writes)
    if (output_row < input_cols && output_col < input_rows) {
        output_matrix[output_row * input_rows + output_col] = tile[local_col][local_row];
    }
}

// ===== ELEMENT-WISE OPERATIONS =====

@group(3) @binding(0) var<storage, read> input_a: array<f32>;
@group(3) @binding(1) var<storage, read> input_b: array<f32>;
@group(3) @binding(2) var<storage, read_write> output: array<f32>;

struct ElementwiseUniforms {
    length: u32,
    scale_a: f32,
    scale_b: f32,
    bias: f32,
}

@group(3) @binding(3) var<uniform> elem_uniforms: ElementwiseUniforms;

// Vectorized element-wise addition: output = scale_a * a + scale_b * b + bias
@compute @workgroup_size(256)
fn elementwise_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= elem_uniforms.length) {
        return;
    }
    
    // Process 4 elements at once
    let remaining = elem_uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        output[idx] = elem_uniforms.scale_a * input_a[idx] + 
                     elem_uniforms.scale_b * input_b[idx] + 
                     elem_uniforms.bias;
    }
}

// Vectorized element-wise multiplication
@compute @workgroup_size(256)
fn elementwise_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    
    if (base_idx >= elem_uniforms.length) {
        return;
    }
    
    let remaining = elem_uniforms.length - base_idx;
    let process_count = min(remaining, 4u);
    
    for (var i = 0u; i < process_count; i++) {
        let idx = base_idx + i;
        output[idx] = input_a[idx] * input_b[idx];
    }
}