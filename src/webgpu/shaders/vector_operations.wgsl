// Simple vector operations for ruv-FANN

// Vector addition: c = a + b
@group(0) @binding(0) var<storage, read> vec_a: array<f32>;
@group(0) @binding(1) var<storage, read> vec_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> vec_result: array<f32>;

struct VectorOpUniforms {
    length: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(3) var<uniform> uniforms: VectorOpUniforms;

@compute @workgroup_size(256)
fn vector_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= uniforms.length) {
        return;
    }
    
    vec_result[idx] = vec_a[idx] + vec_b[idx];
}

@compute @workgroup_size(256)
fn vector_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= uniforms.length) {
        return;
    }
    
    // For scale, we use vec_b[0] as the scalar value
    vec_result[idx] = vec_a[idx] * vec_b[0];
}