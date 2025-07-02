// WebGPU Compute Shader: Matrix-Vector Multiplication
// Optimized for neural network forward propagation
// Supports matrices up to GPU limits with efficient memory access patterns

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Uniforms {
    rows: u32,
    cols: u32,
    batch_id: u32,
    reserved: u32,
}

@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Workgroup size optimized for most GPUs (256 threads per workgroup)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    // Bounds check
    if (row >= uniforms.rows) {
        return;
    }
    
    // Compute dot product for this row
    var sum: f32 = 0.0;
    
    // Vectorized accumulation for better performance
    let cols = uniforms.cols;
    let row_offset = row * cols;
    
    // Process 4 elements at a time when possible
    let vectorized_cols = cols & ~3u; // Round down to multiple of 4
    
    for (var col = 0u; col < vectorized_cols; col += 4u) {
        let matrix_base = row_offset + col;
        
        // Load 4 matrix elements
        let m0 = matrix[matrix_base];
        let m1 = matrix[matrix_base + 1u];
        let m2 = matrix[matrix_base + 2u];
        let m3 = matrix[matrix_base + 3u];
        
        // Load 4 vector elements
        let v0 = vector[col];
        let v1 = vector[col + 1u];
        let v2 = vector[col + 2u];
        let v3 = vector[col + 3u];
        
        // Fused multiply-add
        sum += m0 * v0 + m1 * v1 + m2 * v2 + m3 * v3;
    }
    
    // Handle remaining elements
    for (var col = vectorized_cols; col < cols; col++) {
        sum += matrix[row_offset + col] * vector[col];
    }
    
    // Store result
    result[row] = sum;
}