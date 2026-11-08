// Private helpers included in wasm_gpu_bridge; no blocking native APIs.
fn activation_entry(name: &str) -> Result<&'static str> {
    if name.len() > 64 {
        return Err(error("Unsupported activation name"));
    }
    Ok(
        match name.replace(['_', '-'], "").to_ascii_lowercase().as_str() {
            "linear" => "linear_main",
            "threshold" => "threshold_main",
            "thresholdsymmetric" => "threshold_symmetric_main",
            "sigmoid" => "sigmoid_main",
            "tanh" | "sigmoidsymmetric" => "tanh_main",
            "relu" => "relu_main",
            "reluleaky" | "leakyrelu" => "leaky_relu_main",
            "gaussian" => "gaussian_main",
            "gaussiansymmetric" => "gaussian_symmetric_main",
            "elliot" | "elliott" => "elliott_main",
            "elliotsymmetric" | "elliottsymmetric" => "elliott_symmetric_main",
            "linearpiece" => "linear_piece_main",
            "linearpiecesymmetric" => "linear_piece_symmetric_main",
            "sin" => "sin_main",
            "cos" => "cos_main",
            "sinsymmetric" => "sin_symmetric_main",
            "cossymmetric" => "cos_symmetric_main",
            "gelu" => "gelu_main",
            "swish" => "swish_main",
            _ => return Err(error("Unsupported activation name")),
        },
    )
}

fn buffer(context: &Context, owned: &mut Buffers, bytes: usize, usage: u32) -> Result<JsValue> {
    let buffer = call(
        &context.device,
        "createBuffer",
        &[object(&[
            ("size", (bytes as f64).into()),
            ("usage", usage.into()),
        ])?],
    )?;
    owned.0.push(buffer.clone());
    Ok(buffer)
}
fn upload(context: &Context, owned: &mut Buffers, bytes: &[u8], usage: u32) -> Result<JsValue> {
    let buffer = buffer(context, owned, bytes.len(), usage | 8 /* COPY_DST */)?;
    call(
        &context.queue,
        "writeBuffer",
        &[buffer.clone(), 0.into(), Uint8Array::from(bytes).into()],
    )?;
    Ok(buffer)
}
async fn dispatch(
    context: &Context,
    source: &str,
    entry: &str,
    inputs: &[&[f32]],
    uniforms: &[u8],
    length: usize,
    groups: [u32; 3],
) -> Result<Vec<f32>> {
    validate_buffers(
        context,
        &inputs.iter().map(|v| v.len()).collect::<Vec<_>>(),
        length,
        groups,
    )?;
    if inputs.iter().flat_map(|v| v.iter()).any(|v| !v.is_finite()) {
        return Err(error("GPU inputs must be finite"));
    }
    call(&context.device, "pushErrorScope", &["out-of-memory".into()])?;
    call(&context.device, "pushErrorScope", &["validation".into()])?;
    let result = dispatch_inner(context, source, entry, inputs, uniforms, length, groups).await;
    let validation = awaited(&context.device, "popErrorScope", &[]).await;
    let allocation = awaited(&context.device, "popErrorScope", &[]).await;
    for captured in [validation?, allocation?] {
        if !captured.is_null() && !captured.is_undefined() {
            return Err(error(
                &get(&captured, "message")?
                    .as_string()
                    .unwrap_or_else(|| "WebGPU validation/allocation failed".into()),
            ));
        }
    }
    result
}
async fn dispatch_inner(
    context: &Context,
    source: &str,
    entry: &str,
    inputs: &[&[f32]],
    uniforms: &[u8],
    length: usize,
    groups: [u32; 3],
) -> Result<Vec<f32>> {
    let mut owned = Buffers(Vec::new());
    let module = call(
        &context.device,
        "createShaderModule",
        &[object(&[("code", source.into())])?],
    )?;
    let info = awaited(&module, "getCompilationInfo", &[]).await?;
    for message in Array::from(&get(&info, "messages")?).iter() {
        if get(&message, "type")?.as_string().as_deref() == Some("error") {
            return Err(error(
                &get(&message, "message")?.as_string().unwrap_or_default(),
            ));
        }
    }
    let compute = object(&[("module", module), ("entryPoint", entry.into())])?;
    let pipeline = awaited(
        &context.device,
        "createComputePipelineAsync",
        &[object(&[("layout", "auto".into()), ("compute", compute)])?],
    )
    .await?;
    let mut buffers = Vec::new();
    for input in inputs {
        buffers.push(upload(
            context,
            &mut owned,
            bytemuck::cast_slice(input),
            128, /* STORAGE */
        )?);
    }
    let output = buffer(
        context,
        &mut owned,
        length * 4,
        128 | 4, /* STORAGE | COPY_SRC */
    )?;
    buffers.push(output.clone());
    buffers.push(upload(
        context, &mut owned, uniforms, 64, /* UNIFORM */
    )?);
    let entries = Array::new();
    for (binding, buffer) in buffers.iter().enumerate() {
        entries.push(&object(&[
            ("binding", (binding as u32).into()),
            ("resource", object(&[("buffer", buffer.clone())])?),
        ])?);
    }
    let layout = call(&pipeline, "getBindGroupLayout", &[0.into()])?;
    let group = call(
        &context.device,
        "createBindGroup",
        &[object(&[("layout", layout), ("entries", entries.into())])?],
    )?;
    let staging = buffer(
        context,
        &mut owned,
        length * 4,
        1 | 8, /* MAP_READ | COPY_DST */
    )?;
    let encoder = call(&context.device, "createCommandEncoder", &[])?;
    let pass = call(&encoder, "beginComputePass", &[])?;
    call(&pass, "setPipeline", &[pipeline])?;
    call(&pass, "setBindGroup", &[0.into(), group])?;
    call(&pass, "dispatchWorkgroups", &groups.map(JsValue::from))?;
    call(&pass, "end", &[])?;
    call(
        &encoder,
        "copyBufferToBuffer",
        &[
            output,
            0.into(),
            staging.clone(),
            0.into(),
            (length as f64 * 4.0).into(),
        ],
    )?;
    let command = call(&encoder, "finish", &[])?;
    call(&context.queue, "submit", &[Array::of1(&command).into()])?;
    awaited(&staging, "mapAsync", &[1.into()]).await?;
    let mapped = call(&staging, "getMappedRange", &[])?;
    let values = Float32Array::new(&mapped).to_vec();
    call(&staging, "unmap", &[])?;
    if values.len() != length || values.iter().any(|v| !v.is_finite()) {
        return Err(error(
            "GPU result has an invalid length or non-finite values",
        ));
    }
    Ok(values)
}

// Validate the actual typed-array kind before invoking any element-copy API.
// An explicit-length view cannot grow with a Shared/Growable backing buffer.
fn bounded_input(input: &Float32Array, max_bytes: usize) -> Result<Float32Array> {
    if !js_sys::ArrayBuffer::is_view(input.as_ref())
        || typed_array_slot(input, &js_sys::Symbol::to_string_tag().into())?
            .as_string()
            .as_deref()
            != Some("Float32Array")
    {
        return Err(error("Expected a real Float32Array"));
    }
    let length = typed_array_slot(input, &"length".into())?
        .as_f64()
        .ok_or_else(|| error("Invalid typed-array length"))?;
    if !length.is_finite()
        || length < 1.0
        || length.fract() != 0.0
        || length > (max_bytes / 4) as f64
    {
        return Err(error("Input exceeds positive device/64MiB limits"));
    }
    let constructor = get(&js_sys::global(), "Float32Array")?.dyn_into::<Function>()?;
    let buffer = typed_array_slot(input, &"buffer".into())?;
    let offset = typed_array_slot(input, &"byteOffset".into())?;
    let view = Reflect::construct(&constructor, &Array::of3(&buffer, &offset, &length.into()))?
        .dyn_into::<Float32Array>()?;
    if view.length() as f64 != length {
        return Err(error("Typed-array view length changed"));
    }
    Ok(view)
}
fn validate_buffers(
    context: &Context,
    inputs: &[usize],
    length: usize,
    groups: [u32; 3],
) -> Result<()> {
    for &count in inputs.iter().chain(std::iter::once(&length)) {
        count
            .checked_mul(4)
            .filter(|b| *b > 0 && *b <= context.max_bytes)
            .ok_or_else(|| error("Buffer exceeds positive device/64MiB limits"))?;
    }
    if groups
        .iter()
        .any(|&n| n == 0 || n as usize > context.max_groups)
    {
        return Err(error("Dispatch exceeds device workgroup limits"));
    }
    Ok(())
}

fn dimension(value: &JsValue) -> Result<u32> {
    let value = value
        .as_f64()
        .ok_or_else(|| error("Matrix dimensions must be numbers"))?;
    if !value.is_finite() || value < 1.0 || value > u32::MAX as f64 || value.fract() != 0.0 {
        return Err(error(
            "Matrix dimensions must be positive integers in u32 range",
        ));
    }
    Ok(value as u32)
}

fn owned_input(input: &Float32Array) -> Result<Float32Array> {
    let constructor = get(&js_sys::global(), "Float32Array")?.dyn_into::<Function>()?;
    let owned = Reflect::construct(&constructor, &Array::of1(input.as_ref()))?
        .dyn_into::<Float32Array>()?;
    if owned.length() != input.length() {
        return Err(error("Input snapshot length changed"));
    }
    Ok(owned)
}

// Intrinsic getters use TypedArray internal slots; instanceof, own properties,
// and Object.prototype.toString can be forged by caller-controlled prototypes.
fn typed_array_slot(input: &Float32Array, key: &JsValue) -> Result<JsValue> {
    let prototype = get(&get(&js_sys::global(), "Float32Array")?, "prototype")?;
    let typed_array = Object::get_prototype_of(&prototype);
    let descriptor = Object::get_own_property_descriptor(&typed_array, key);
    get(&descriptor, "get")?
        .dyn_into::<Function>()?
        .call0(input.as_ref())
}
