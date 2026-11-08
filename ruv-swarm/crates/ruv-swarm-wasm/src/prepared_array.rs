//! Capture intrinsic functions once; read actual internal slots on every call.
use js_sys::Float64Array;
#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Function, Object, Reflect, Symbol};
use wasm_bindgen::prelude::*;

// Native numerical tests never access JS. Only WASM instances retain handles.
pub(super) struct ArrayBoundary {
    #[cfg(target_arch = "wasm32")]
    runtime: Runtime,
}

#[cfg(target_arch = "wasm32")]
struct Runtime {
    constructor: Function,
    tag: Function,
    length: Function,
    buffer: Function,
    offset: Function,
    array_buffer_length: Function,
    memory_buffer: Function,
    memory: JsValue,
    expected_tag: JsValue,
}

#[cfg(target_arch = "wasm32")]
impl ArrayBoundary {
    pub(super) fn new() -> Result<Self, JsValue> {
        let constructor = get(&js_sys::global(), "Float64Array")?.dyn_into::<Function>()?;
        let typed_array = Object::get_prototype_of(&get(constructor.as_ref(), "prototype")?);
        let buffer_prototype =
            get(&get(&js_sys::global(), "ArrayBuffer")?, "prototype")?.dyn_into::<Object>()?;
        let memory = wasm_bindgen::memory();
        let runtime = Runtime {
            tag: getter(&typed_array, &Symbol::to_string_tag().into())?,
            length: getter(&typed_array, &JsValue::from_str("length"))?,
            buffer: getter(&typed_array, &JsValue::from_str("buffer"))?,
            offset: getter(&typed_array, &JsValue::from_str("byteOffset"))?,
            array_buffer_length: getter(&buffer_prototype, &JsValue::from_str("byteLength"))?,
            memory_buffer: getter(
                &Object::get_prototype_of(&memory),
                &JsValue::from_str("buffer"),
            )?,
            memory,
            constructor,
            expected_tag: JsValue::from_str("Float64Array"),
        };
        Ok(Self { runtime })
    }

    pub(super) fn current_memory_buffer(&self) -> Result<JsValue, JsValue> {
        // Retain the Memory object, never its buffer: grow() replaces the buffer.
        self.runtime.memory_buffer.call0(&self.runtime.memory)
    }

    pub(super) fn bounded(
        &self,
        array: &Float64Array,
        expected: usize,
        memory: &JsValue,
    ) -> Result<Float64Array, JsValue> {
        let runtime = &self.runtime;
        // The intrinsic tag getter examines TypedArray internal slots. It also
        // rejects Object.create/prototype spoofs, DataViews and proxies without
        // needing instanceof or caller-controlled Symbol.toStringTag properties.
        if runtime.tag.call0(array.as_ref())? != runtime.expected_tag {
            return Err(JsValue::from_str("Expected a real Float64Array"));
        }
        let length = runtime.length.call0(array.as_ref())?.as_f64();
        if length != Some(expected as f64) {
            return Err(JsValue::from_str(
                "Float64Array length must match the prepared shape",
            ));
        }
        let buffer = runtime.buffer.call0(array.as_ref())?;
        // Calling ArrayBuffer's intrinsic getter rejects shared buffers even
        // when their prototypes have been changed to ArrayBuffer.prototype.
        runtime.array_buffer_length.call0(&buffer)?;
        if &buffer == memory {
            return Err(JsValue::from_str(
                "Use JS-owned arrays, not this module's WASM memory",
            ));
        }
        let offset = runtime.offset.call0(array.as_ref())?;
        // Explicit-length plain views ignore per-array getters/prototypes/set
        // methods and prevent a length-tracking view from changing copy bounds.
        Reflect::construct(
            &runtime.constructor,
            &Array::of3(&buffer, &offset, &JsValue::from_f64(expected as f64)),
        )?
        .dyn_into::<Float64Array>()
    }
}

#[cfg(target_arch = "wasm32")]
fn get(object: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(object, &JsValue::from_str(key))
}

#[cfg(target_arch = "wasm32")]
fn getter(prototype: &Object, key: &JsValue) -> Result<Function, JsValue> {
    get(&Object::get_own_property_descriptor(prototype, key), "get")?.dyn_into::<Function>()
}

#[cfg(not(target_arch = "wasm32"))]
impl ArrayBoundary {
    pub(super) fn new() -> Result<Self, JsValue> {
        Ok(Self {})
    }

    pub(super) fn current_memory_buffer(&self) -> Result<JsValue, JsValue> {
        Err(JsValue::NULL)
    }

    pub(super) fn bounded(
        &self,
        _: &Float64Array,
        _: usize,
        _: &JsValue,
    ) -> Result<Float64Array, JsValue> {
        Err(JsValue::NULL)
    }
}
