// Adam optimizer shader for WebGPU
// Fused gradient update kernel for efficient training

const WORKGROUP_SIZE: u32 = 64u;

struct AdamParams {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: u32,
}

@group(0) @binding(0)
var<storage, read_write> weights: array<f32>;

@group(0) @binding(1)
var<storage, read_write> gradients: array<f32>;

@group(0) @binding(2)
var<storage, read_write> m_moments: array<f32>;

@group(0) @binding(3)
var<storage, read_write> v_moments: array<f32>;

@group(0) @binding(4)
var<uniform> params: AdamParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn adam_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&weights) || idx >= arrayLength(&gradients)
        || idx >= arrayLength(&m_moments) || idx >= arrayLength(&v_moments)
        || params.step == 0u) { return; }
    
    // Read current values
    let grad = gradients[idx];
    let m = m_moments[idx];
    let v = v_moments[idx];
    
    // Update biased first moment estimate
    let new_m = params.beta1 * m + (1.0 - params.beta1) * grad;
    
    // Update biased second raw moment estimate
    let new_v = params.beta2 * v + (1.0 - params.beta2) * grad * grad;
    
    // Store updated moments
    m_moments[idx] = new_m;
    v_moments[idx] = new_v;
    
    // Compute bias-corrected learning rate
    let bias_correction1 = 1.0 - pow(params.beta1, f32(params.step));
    let bias_correction2 = 1.0 - pow(params.beta2, f32(params.step));
    
    // Compute bias-corrected moment estimates
    let m_hat = new_m / bias_correction1;
    let v_hat = new_v / bias_correction2;
    
    // Update weights
    weights[idx] -= params.learning_rate * m_hat / (sqrt(v_hat) + params.epsilon);
}

// AdamW variant with decoupled weight decay
struct AdamWParams {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: u32,
}

@group(0) @binding(4)
var<uniform> adamw_params: AdamWParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn adamw_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&weights) || idx >= arrayLength(&gradients)
        || idx >= arrayLength(&m_moments) || idx >= arrayLength(&v_moments)
        || adamw_params.step == 0u) { return; }
    
    // Read current values
    let weight = weights[idx];
    let grad = gradients[idx];
    let m = m_moments[idx];
    let v = v_moments[idx];
    
    // Update biased moment estimates
    let new_m = adamw_params.beta1 * m + (1.0 - adamw_params.beta1) * grad;
    let new_v = adamw_params.beta2 * v + (1.0 - adamw_params.beta2) * grad * grad;
    
    // Store updated moments
    m_moments[idx] = new_m;
    v_moments[idx] = new_v;
    
    // Compute bias-corrected learning rate
    let bias_correction1 = 1.0 - pow(adamw_params.beta1, f32(adamw_params.step));
    let bias_correction2 = 1.0 - pow(adamw_params.beta2, f32(adamw_params.step));
    
    // Compute bias-corrected moment estimates
    let m_hat = new_m / bias_correction1;
    let v_hat = new_v / bias_correction2;
    
    // Update weights with decoupled weight decay
    weights[idx] = weight - adamw_params.learning_rate * m_hat / (sqrt(v_hat) + adamw_params.epsilon)
                         - adamw_params.learning_rate * adamw_params.weight_decay * weight;
}

// Batch gradient accumulation
@compute @workgroup_size(WORKGROUP_SIZE)
fn accumulate_gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let width = arrayLength(&weights);
    if (idx >= width || width == 0u) { return; }
    let batch_size = arrayLength(&gradients) / width;
    if (batch_size == 0u || arrayLength(&gradients) % width != 0u) { return; }
    
    var sum = 0.0f;
    for (var i = 0u; i < batch_size; i += 1u) {
        sum += gradients[i * arrayLength(&weights) + idx];
    }
    
    // Average gradient across batch
    gradients[idx] = sum / f32(batch_size);
}
