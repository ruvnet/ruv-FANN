// WebGPU Compute Shader: Batch Matrix-Vector Multiplication
// Efficiently processes multiple vectors against the same matrix
// Optimized for neural network batch inference

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vectors: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

struct Uniforms {
    rows: u32,
    cols: u32,
    batch_size: u32,
    reserved: u32,
}

@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// 2D workgroup layout: 16x16 = 256 threads per workgroup
// X dimension: matrix rows, Y dimension: batch items
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let batch_idx = global_id.y;
    
    // Bounds check
    if (row >= uniforms.rows || batch_idx >= uniforms.batch_size) {
        return;
    }
    
    // Compute dot product for this row and batch item
    var sum: f32 = 0.0;
    
    let cols = uniforms.cols;
    let matrix_row_offset = row * cols;
    let vector_offset = batch_idx * cols;
    
    // Vectorized processing for better memory bandwidth utilization
    let vectorized_cols = cols & ~3u; // Round down to multiple of 4
    
    for (var col = 0u; col < vectorized_cols; col += 4u) {
        let matrix_base = matrix_row_offset + col;
        let vector_base = vector_offset + col;
        
        // Load 4 matrix elements (same for all batches)
        let m0 = matrix[matrix_base];
        let m1 = matrix[matrix_base + 1u];
        let m2 = matrix[matrix_base + 2u];
        let m3 = matrix[matrix_base + 3u];
        
        // Load 4 vector elements (batch-specific)
        let v0 = vectors[vector_base];
        let v1 = vectors[vector_base + 1u];
        let v2 = vectors[vector_base + 2u];
        let v3 = vectors[vector_base + 3u];
        
        // Fused multiply-add
        sum += m0 * v0 + m1 * v1 + m2 * v2 + m3 * v3;
    }
    
    // Handle remaining elements
    for (var col = vectorized_cols; col < cols; col++) {
        sum += matrix[matrix_row_offset + col] * vectors[vector_offset + col];
    }
    
    // Store result: results[batch_idx * rows + row]
    let result_index = batch_idx * uniforms.rows + row;
    results[result_index] = sum;
}