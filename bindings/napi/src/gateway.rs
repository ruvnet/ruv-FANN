//! The only raw Node-API/pointer gateway. No Rust slice ever borrows JS memory.
//! Calls stay on the owning JS thread. Copies run immediately after intrinsic
//! checks, with no JS callbacks between pointer acquisition and the copy.
use napi::bindgen_prelude::{Function, FunctionRef, Unknown};
use napi::{sys, Env, Error, JsValue, Result, Status};
use std::{cell::RefCell, collections::HashMap, ptr, sync::Arc};

pub const MAX_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_ROWS: usize = 4096;
struct Intrinsics {
    resizable: FunctionRef<(), bool>,
}
thread_local! {
    static INTRINSICS: RefCell<HashMap<usize, Arc<Intrinsics>>> = RefCell::new(HashMap::new());
}

pub fn invalid(message: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, message.into())
}
fn status(value: sys::napi_status) -> Result<()> {
    if value == sys::Status::napi_ok {
        Ok(())
    } else {
        Err(Error::new(
            Status::from(value),
            "Node-API boundary check failed",
        ))
    }
}

/// Capture at addon registration, before any public model is constructed.
/// The getter is an intrinsic, not a caller-controlled `.resizable` property.
pub fn register(env: Env) -> Result<()> {
    let getter: Function<(), bool> =
        env.run_script("Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'resizable').get")?;
    let guard = Arc::new(Intrinsics {
        resizable: getter.create_ref()?,
    });
    let key = env.raw() as usize;
    INTRINSICS.with(|values| values.borrow_mut().insert(key, guard));
    env.add_env_cleanup_hook(key, |key| {
        INTRINSICS.with(|values| {
            values.borrow_mut().remove(&key);
        });
    })?;
    Ok(())
}

struct View {
    data: *mut f64,
    len: usize,
    arraybuffer: sys::napi_value,
}

/// Pure intrinsic queries: no property reads, coercion, user callbacks or allocation.
fn view(env: Env, value: Unknown<'_>, expected: usize) -> Result<View> {
    if expected
        .checked_mul(8)
        .filter(|n| *n <= MAX_BYTES)
        .is_none()
    {
        return Err(invalid("Float64Array exceeds 64 MiB"));
    }
    let raw = env.raw();
    let mut is_typed = false;
    // SAFETY: value is a live argument handle in this Node-API call scope.
    unsafe {
        status(sys::napi_is_typedarray(raw, value.raw(), &mut is_typed))?;
    }
    if !is_typed {
        return Err(invalid("Expected a genuine Float64Array"));
    }
    let (mut kind, mut len, mut offset) = (0, 0, 0);
    let (mut data, mut arraybuffer) = (ptr::null_mut(), ptr::null_mut());
    // SAFETY: Node established the typed-array brand; output slots are valid.
    unsafe {
        status(sys::napi_get_typedarray_info(
            raw,
            value.raw(),
            &mut kind,
            &mut len,
            &mut data,
            &mut arraybuffer,
            &mut offset,
        ))?;
    }
    if kind != sys::TypedarrayType::float64_array || len != expected {
        return Err(invalid("Float64Array type or exact length mismatch"));
    }
    let (mut ordinary, mut detached) = (false, false);
    // SAFETY: arraybuffer is the buffer returned by the intrinsic typed-array query.
    unsafe {
        status(sys::napi_is_arraybuffer(raw, arraybuffer, &mut ordinary))?;
    }
    if !ordinary {
        return Err(invalid("SharedArrayBuffer is not supported"));
    }
    unsafe {
        status(sys::napi_is_detached_arraybuffer(
            raw,
            arraybuffer,
            &mut detached,
        ))?;
    }
    if detached {
        return Err(invalid("Detached ArrayBuffer is not supported"));
    }
    let (mut base, mut bytes) = (ptr::null_mut(), 0);
    unsafe {
        status(sys::napi_get_arraybuffer_info(
            raw,
            arraybuffer,
            &mut base,
            &mut bytes,
        ))?;
    }
    let needed = expected
        .checked_mul(8)
        .ok_or_else(|| invalid("Array size overflow"))?;
    if offset
        .checked_add(needed)
        .filter(|end| *end <= bytes)
        .is_none()
        || (needed > 0 && (data.is_null() || (data as usize) % std::mem::align_of::<f64>() != 0))
    {
        return Err(invalid("Invalid Float64Array backing storage"));
    }
    Ok(View {
        data: data.cast(),
        len,
        arraybuffer,
    })
}

/// May invoke the cached intrinsic getter. Call before borrowing native state.
/// All subsequent pointer uses reacquire view() after callback-capable checks.
pub fn validate_array(env: Env, value: Unknown<'_>, expected: usize) -> Result<()> {
    let info = view(env, value, expected)?;
    let intrinsics = INTRINSICS
        .with(|values| values.borrow().get(&(env.raw() as usize)).cloned())
        .ok_or_else(|| invalid("Addon boundary intrinsics are unavailable"))?;
    // SAFETY: buffer handle belongs to this env/call scope and was intrinsically checked.
    let buffer = unsafe { Unknown::from_raw_unchecked(env.raw(), info.arraybuffer) };
    if intrinsics.resizable.borrow_back(&env)?.apply(buffer, ())? {
        return Err(invalid("Resizable ArrayBuffer is not supported"));
    }
    view(env, value, expected)?;
    Ok(())
}

/// Input snapshot only, never an immutable Rust reference to caller memory.
pub fn copy_input(env: Env, value: Unknown<'_>, owned: &mut [f64]) -> Result<()> {
    let info = view(env, value, owned.len())?;
    if info.len != 0 {
        // SAFETY: ordinary, live, aligned JS storage has exactly owned.len() f64s.
        // JS cannot run/resize it during this synchronous intrinsic-copy section.
        // Native staging is privately allocated and never exposed to JS.
        unsafe {
            ptr::copy_nonoverlapping(info.data, owned.as_mut_ptr(), info.len);
        }
    }
    if owned.iter().any(|value| !value.is_finite()) {
        return Err(invalid("Inputs must be finite"));
    }
    Ok(())
}

/// Called only after inference succeeded; therefore errors preserve caller output.
pub fn commit_output(env: Env, value: Unknown<'_>, owned: &[f64]) -> Result<()> {
    let info = view(env, value, owned.len())?;
    if info.len != 0 {
        // SAFETY: same checks as copy_input. Owned Rust output cannot overlap a
        // caller buffer. Input/output JS aliasing was removed by the input snapshot.
        unsafe {
            ptr::copy_nonoverlapping(owned.as_ptr(), info.data, info.len);
        }
    }
    Ok(())
}

/// Allocate the convenience result before borrowing native state, then recheck input.
pub fn result_array<'env>(env: Env, len: usize, _scope: Unknown<'env>) -> Result<Unknown<'env>> {
    let bytes = len
        .checked_mul(8)
        .filter(|n| *n <= MAX_BYTES)
        .ok_or_else(|| invalid("Output byte budget exceeded"))?;
    let (mut data, mut arraybuffer, mut result) =
        (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    // SAFETY: Node owns both allocations; no Rust pointer escapes in an external buffer.
    unsafe {
        status(sys::napi_create_arraybuffer(
            env.raw(),
            bytes,
            &mut data,
            &mut arraybuffer,
        ))?;
        status(sys::napi_create_typedarray(
            env.raw(),
            sys::TypedarrayType::float64_array,
            len,
            arraybuffer,
            0,
            &mut result,
        ))?;
        Ok(Unknown::from_raw_unchecked(env.raw(), result))
    }
}

/// Primitive string byte length is checked before allocating any Rust UTF-8 buffer.
pub fn json_string(env: Env, value: Unknown<'_>) -> Result<String> {
    let mut kind = 0;
    unsafe {
        status(sys::napi_typeof(env.raw(), value.raw(), &mut kind))?;
    }
    if kind != sys::ValueType::napi_string {
        return Err(invalid("Model JSON must be a primitive string"));
    }
    let mut len = 0;
    unsafe {
        status(sys::napi_get_value_string_utf8(
            env.raw(),
            value.raw(),
            ptr::null_mut(),
            0,
            &mut len,
        ))?;
    }
    if len > MAX_JSON_BYTES {
        return Err(invalid("Model JSON exceeds 4 MiB UTF-8"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len + 1)
        .map_err(|_| invalid("JSON allocation failed"))?;
    bytes.resize(len + 1, 0);
    let mut written = 0;
    unsafe {
        status(sys::napi_get_value_string_utf8(
            env.raw(),
            value.raw(),
            bytes.as_mut_ptr().cast(),
            bytes.len(),
            &mut written,
        ))?;
    }
    if written != len {
        return Err(invalid("Model JSON length changed"));
    }
    bytes.truncate(len);
    String::from_utf8(bytes).map_err(|_| invalid("Model JSON must be valid UTF-8"))
}

pub fn rows(env: Env, value: Unknown<'_>, minimum: usize, maximum: usize) -> Result<usize> {
    let number = number(env, value)?;
    if number.fract() != 0.0 || number < minimum as f64 || number > maximum as f64 {
        return Err(invalid("Rows must be a bounded integer"));
    }
    Ok(number as usize)
}

pub fn number(env: Env, value: Unknown<'_>) -> Result<f64> {
    let mut kind = 0;
    unsafe {
        status(sys::napi_typeof(env.raw(), value.raw(), &mut kind))?;
    }
    if kind != sys::ValueType::napi_number {
        return Err(invalid("Hyperparameter/rows must be a primitive number"));
    }
    let mut number = 0.0;
    unsafe {
        status(sys::napi_get_value_double(
            env.raw(),
            value.raw(),
            &mut number,
        ))?;
    }
    if !number.is_finite() {
        return Err(invalid("Hyperparameter/rows must be finite"));
    }
    Ok(number)
}
