//! Async browser-only f32 compute with real WebGPU buffers and readback.
//! Calendar, workers, DAA, canvas rendering, and persistent shared buffers are
//! outside this compute bridge. Native blocking ComputeBackend is separate.

use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect, Uint8Array};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::JsFuture;

const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_MULTIPLY_ADDS: u64 = 64 * 1024 * 1024;
const ACTIVATIONS: &str = include_str!("shaders/activation_functions.wgsl");
const MATRIX: &str = r#"
@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read> b: array<f32>;
@group(0) @binding(2) var<storage,read_write> output: array<f32>;
struct Dims { rows: u32, inner: u32, cols: u32, reserved: u32 }
@group(0) @binding(3) var<uniform> dims: Dims;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.y >= dims.rows || id.x >= dims.cols) { return; }
    var sum = 0.0;
    for (var k = 0u; k < dims.inner; k++) {
        sum += a[id.y * dims.inner + k] * b[k * dims.cols + id.x];
    }
    output[id.y * dims.cols + id.x] = sum;
}
"#;

type Result<T> = std::result::Result<T, JsValue>;
fn error(message: &str) -> JsValue {
    js_sys::Error::new(message).into()
}
fn get(target: &JsValue, name: &str) -> Result<JsValue> {
    Reflect::get(target, &name.into())
}
fn object(entries: &[(&str, JsValue)]) -> Result<JsValue> {
    let value = Object::new();
    for (key, field) in entries {
        Reflect::set(&value, &(*key).into(), field)?;
    }
    Ok(value.into())
}
fn call(target: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue> {
    let function = get(target, name)?
        .dyn_into::<Function>()
        .map_err(|_| error(&format!("WebGPU method '{name}' is unavailable")))?;
    function.apply(target, &args.iter().cloned().collect::<Array>())
}
async fn awaited(target: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue> {
    JsFuture::from(Promise::resolve(&call(target, name, args)?)).await
}
fn limit(target: &JsValue, name: &str) -> Result<usize> {
    let value = get(target, name)?
        .as_f64()
        .ok_or_else(|| error("Missing numeric WebGPU limit"))?;
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 || value > 9_007_199_254_740_991.0
    {
        return Err(error("Invalid WebGPU device limit"));
    }
    Ok(value.min(usize::MAX as f64) as usize)
}
fn navigator_gpu() -> Result<JsValue> {
    get(&get(&js_sys::global(), "navigator")?, "gpu")
}

struct Context {
    device: JsValue,
    queue: JsValue,
    max_bytes: usize,
    max_groups: usize,
}
impl Drop for Context {
    fn drop(&mut self) {
        let _ = call(&self.device, "destroy", &[]);
    }
}
struct Busy(Rc<Cell<bool>>);
impl Busy {
    fn acquire(flag: &Rc<Cell<bool>>) -> Result<Self> {
        if flag.replace(true) {
            return Err(error("A WebGPU operation is already in progress"));
        }
        Ok(Self(flag.clone()))
    }
}
impl Drop for Busy {
    fn drop(&mut self) {
        self.0.set(false);
    }
}
#[derive(Default)]
struct State {
    context: Option<Rc<Context>>,
    completed: u32,
    returned_bytes: u64,
}
struct Buffers(Vec<JsValue>);
impl Drop for Buffers {
    fn drop(&mut self) {
        for buffer in &self.0 {
            let _ = call(buffer, "destroy", &[]);
        }
    }
}

/// Browser compute object. Inputs/outputs are row-major Float32Arrays.
#[wasm_bindgen]
pub struct WasmGpuBridge {
    state: RefCell<State>,
    busy: Rc<Cell<bool>>,
}

#[wasm_bindgen]
impl WasmGpuBridge {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: RefCell::new(State::default()),
            busy: Rc::new(Cell::new(false)),
        }
    }

    /// Request a real adapter and device. Canvas rendering is not supported.
    pub async fn initialize_webgpu(&self, canvas_id: Option<JsValue>) -> Result<()> {
        let _busy = Busy::acquire(&self.busy)?;
        if canvas_id.is_some() {
            return Err(error(
                "Canvas rendering is unsupported by the compute bridge",
            ));
        }
        if self.state.borrow().context.is_some() {
            return Ok(());
        }
        let gpu = navigator_gpu()?;
        if gpu.is_null() || gpu.is_undefined() {
            return Err(error("navigator.gpu is unavailable"));
        }
        let adapter = awaited(
            &gpu,
            "requestAdapter",
            &[object(&[("powerPreference", "high-performance".into())])?],
        )
        .await?;
        if adapter.is_null() || adapter.is_undefined() {
            return Err(error("No browser GPU adapter"));
        }
        let device = awaited(&adapter, "requestDevice", &[Object::new().into()]).await?;
        let limits = get(&device, "limits")?;
        let max_bytes = MAX_BYTES
            .min(limit(&limits, "maxBufferSize")?)
            .min(limit(&limits, "maxStorageBufferBindingSize")?);
        let context = Context {
            queue: get(&device, "queue")?,
            device,
            max_bytes,
            max_groups: limit(&limits, "maxComputeWorkgroupsPerDimension")?,
        };
        self.state.borrow_mut().context = Some(Rc::new(context));
        Ok(())
    }

    /// Multiply A[rows, inner] by B[inner, cols] with a real 2D GPU dispatch.
    pub async fn gpu_matrix_multiply(
        &self,
        a: Float32Array,
        b: Float32Array,
        #[wasm_bindgen(unchecked_param_type = "number")] rows: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] inner: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] cols: JsValue,
    ) -> Result<Vec<f32>> {
        let _busy = Busy::acquire(&self.busy)?;
        let context = self.context()?;
        let (rows, inner, cols) = (dimension(&rows)?, dimension(&inner)?, dimension(&cols)?);
        let count = |x: u32, y: u32| {
            (x as usize)
                .checked_mul(y as usize)
                .filter(|v| *v > 0)
                .ok_or_else(|| error("Matrix dimensions must be positive and not overflow"))
        };
        let a = bounded_input(&a, context.max_bytes)?;
        let b = bounded_input(&b, context.max_bytes)?;
        let output = count(rows, cols)?;
        if a.length() as usize != count(rows, inner)? || b.length() as usize != count(inner, cols)?
        {
            return Err(error("Matrix data lengths do not match dimensions"));
        }
        let operations = (rows as u64)
            .checked_mul(inner as u64)
            .and_then(|n| n.checked_mul(cols as u64));
        if operations.map_or(true, |n| n > MAX_MULTIPLY_ADDS) {
            return Err(error("Matrix exceeds the 64M multiply-add dispatch budget"));
        }
        let groups = [cols.div_ceil(8), rows.div_ceil(8), 1];
        validate_buffers(
            &context,
            &[a.length() as usize, b.length() as usize],
            output,
            groups,
        )?;
        // Snapshot both fixed views into JS-owned storage before any Rust data
        // allocation can grow/detach this module's own input memory buffer.
        let (a, b) = (owned_input(&a)?, owned_input(&b)?);
        let (a, b) = (a.to_vec(), b.to_vec());
        let uniforms: Vec<u8> = [rows, inner, cols, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dispatch(
            &context,
            MATRIX,
            "main",
            &[&a, &b],
            &uniforms,
            output,
            groups,
        )
        .await?;
        self.record(result.len());
        Ok(result)
    }

    /// Apply a canonical ActivationFunction name to finite f32 inputs.
    pub async fn gpu_apply_activation(
        &self,
        inputs: Float32Array,
        #[wasm_bindgen(unchecked_param_type = "string")] activation_type: JsValue,
        steepness: f32,
    ) -> Result<Vec<f32>> {
        let _busy = Busy::acquire(&self.busy)?;
        let context = self.context()?;
        if !steepness.is_finite() {
            return Err(error("Activation steepness must be finite"));
        }
        let name = activation_type
            .dyn_into::<js_sys::JsString>()
            .map_err(|_| error("Activation must be a string"))?;
        if name.length() > 64 {
            return Err(error("Unsupported activation name"));
        }
        let entry = activation_entry(&String::from(name))?;
        let inputs = bounded_input(&inputs, context.max_bytes)?;
        let length = inputs.length();
        validate_buffers(
            &context,
            &[length as usize],
            length as usize,
            [length.div_ceil(256), 1, 1],
        )?;
        let inputs = owned_input(&inputs)?.to_vec();
        let uniforms: Vec<u8> = [length, steepness.to_bits(), 0.01_f32.to_bits(), 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dispatch(
            &context,
            ACTIVATIONS,
            entry,
            &[&inputs],
            &uniforms,
            inputs.len(),
            [length.div_ceil(256), 1, 1],
        )
        .await?;
        self.record(result.len());
        Ok(result)
    }

    /// Report actual API availability and initialized-device limits, not throughput estimates.
    pub fn check_feature_support(&self) -> Result<JsValue> {
        let gpu = navigator_gpu().unwrap_or(JsValue::UNDEFINED);
        let state = self.state.borrow();
        object(&[
            (
                "navigator_gpu",
                (!gpu.is_null() && !gpu.is_undefined()).into(),
            ),
            ("initialized", state.context.is_some().into()),
            ("supports_f64", false.into()),
            (
                "max_workgroups_per_dimension",
                state
                    .context
                    .as_ref()
                    .map_or(JsValue::NULL, |c| (c.max_groups as f64).into()),
            ),
            (
                "max_buffer_bytes",
                state
                    .context
                    .as_ref()
                    .map_or(JsValue::NULL, |c| (c.max_bytes as f64).into()),
            ),
        ])
    }

    /// Successful dispatches and bytes actually returned by asynchronous readback.
    pub fn get_performance_metrics(&self) -> Result<JsValue> {
        let state = self.state.borrow();
        object(&[
            ("completed_dispatches", state.completed.into()),
            ("returned_bytes", (state.returned_bytes as f64).into()),
            ("in_flight", self.busy.get().into()),
        ])
    }

    /// Destroy the owned device after current work completes. The object can be initialized again.
    pub fn destroy(&self) -> Result<()> {
        let _busy = Busy::acquire(&self.busy)?;
        self.state.borrow_mut().context = None;
        Ok(())
    }
}

impl Default for WasmGpuBridge {
    fn default() -> Self {
        Self::new()
    }
}
impl WasmGpuBridge {
    fn context(&self) -> Result<Rc<Context>> {
        self.state
            .borrow()
            .context
            .clone()
            .ok_or_else(|| error("WebGPU is not initialized"))
    }
    fn record(&self, length: usize) {
        let mut state = self.state.borrow_mut();
        state.completed = state.completed.saturating_add(1);
        state.returned_bytes = state.returned_bytes.saturating_add(length as u64 * 4);
    }
}

include!("wasm_gpu_dispatch.rs");
