//! Prepared f64 bindings delegate all inference to the shared native engine.
use crate::network::{ActivationFunction as A, MAX_LAYERS, MAX_PARAMETERS};
use js_sys::Float64Array;
use ruv_fann::{
    ActivationFunction as NativeActivation, Connection, InferenceWorkspace, Layer, Network, Neuron,
    PreparationLimits, PreparedError, PreparedNetwork,
};
use std::mem::size_of;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

const MAX_BYTES: usize = 64 * 1024 * 1024;
type BuildResult<T> = Result<T, &'static str>;
#[path = "prepared_array.rs"]
mod array_boundary;
#[path = "prepared_batch.rs"]
mod batch;
#[path = "prepared_json.rs"]
mod json_import;
use array_boundary::ArrayBoundary;
pub use batch::WasmPreparedBatch;

/// Owned parameters and reusable scratch. No raw WASM memory views are exposed.
#[wasm_bindgen]
pub struct WasmPreparedNetwork {
    network: Rc<PreparedNetwork<f64>>,
    workspace: InferenceWorkspace<f64>,
    inputs: Vec<f64>,
    outputs: Vec<f64>,
    arrays: Rc<ArrayBoundary>,
    estimated: usize,
}

#[wasm_bindgen]
impl WasmPreparedNetwork {
    /// Import a native serde Network<f64> export (at most 4 MiB UTF-8 JSON).
    /// All native activations and sparse connection ordering are retained.
    pub fn from_network_json(
        #[wasm_bindgen(unchecked_param_type = "string")] json: JsValue,
    ) -> Result<WasmPreparedNetwork, JsValue> {
        if !json.is_string() {
            return Err(JsValue::from_str("Network JSON must be a string"));
        }
        let text = json.unchecked_into::<js_sys::JsString>();
        if text.length() as usize > json_import::MAX_JSON_BYTES {
            return Err(JsValue::from_str("Network JSON exceeds 4 MiB"));
        }
        let text = text
            .as_string()
            .ok_or_else(|| JsValue::from_str("Network JSON must be a string"))?;
        json_import::from_json(&text).map_err(JsValue::from_str)
    }

    #[wasm_bindgen(getter)]
    pub fn input_size(&self) -> usize {
        self.network.num_inputs()
    }

    #[wasm_bindgen(getter)]
    pub fn output_size(&self) -> usize {
        self.network.num_outputs()
    }

    /// Packed storage, native workspace and staging buffers; excludes the source
    /// model, temporary conversion graph and allocator bookkeeping/capacity.
    #[wasm_bindgen(getter)]
    pub fn estimated_bytes(&self) -> usize {
        self.estimated
    }

    /// Share this immutable snapshot with a bounded, reusable row batch.
    pub fn create_batch(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number")] max_rows: JsValue,
    ) -> Result<WasmPreparedBatch, JsValue> {
        let max_rows = batch::checked_rows(max_rows.as_f64(), 1, batch::MAX_ROWS)
            .map_err(JsValue::from_str)?;
        WasmPreparedBatch::new(Rc::clone(&self.network), Rc::clone(&self.arrays), max_rows)
            .map_err(JsValue::from_str)
    }

    /// Convenience result owns its buffer. Inputs must be JS-owned, non-shared
    /// Float64Arrays, validated before any Rust element allocation/copy.
    pub fn run(&mut self, inputs: &Float64Array) -> Result<Vec<f64>, JsValue> {
        let memory = self.arrays.current_memory_buffer()?;
        let inputs = self.arrays.bounded(inputs, self.input_size(), &memory)?;
        inputs.copy_to(&mut self.inputs);
        self.evaluate().map_err(inference_error)?;
        let mut result = reserve(self.output_size()).map_err(JsValue::from_str)?;
        result.extend_from_slice(&self.outputs);
        Ok(result)
    }

    /// Reuses all Rust buffers. Input/output may alias one another. Both must be
    /// JS-owned, non-shared Float64Arrays of the exact expected dimensions.
    /// Errors leave caller output unchanged; no Rust allocation occurs on this
    /// path, including errors. Temporary typed-array views are JS allocations.
    pub fn run_into(
        &mut self,
        inputs: &Float64Array,
        outputs: &Float64Array,
    ) -> Result<(), JsValue> {
        let memory = self.arrays.current_memory_buffer()?;
        let inputs = self.arrays.bounded(inputs, self.input_size(), &memory)?;
        let outputs = self.arrays.bounded(outputs, self.output_size(), &memory)?;
        inputs.copy_to(&mut self.inputs);
        self.evaluate().map_err(inference_error)?;
        outputs.copy_from(&self.outputs);
        Ok(())
    }
}

impl WasmPreparedNetwork {
    pub(crate) fn from_dense(
        shape: &[usize],
        weights: &[f64],
        biases: &[f64],
        activation: A,
    ) -> BuildResult<Self> {
        if !(2..=MAX_LAYERS).contains(&shape.len()) || shape.contains(&0) {
            return Err("A prepared network needs 2..=64 nonzero layers");
        }
        let expected_weights = shape
            .windows(2)
            .try_fold(0usize, |n, p| n.checked_add(p[0].checked_mul(p[1])?))
            .ok_or("Network dimensions overflow")?;
        let expected_biases = shape[1..]
            .iter()
            .try_fold(0usize, |n, &size| n.checked_add(size))
            .ok_or("Network dimensions overflow")?;
        if expected_weights
            .checked_add(expected_biases)
            .map_or(true, |n| n > MAX_PARAMETERS)
            || weights.len() != expected_weights
            || biases.len() != expected_biases
        {
            return Err("Parameters exceed the 1,000,000 limit or do not match the shape");
        }
        if !weights.iter().chain(biases).all(|v| v.is_finite()) {
            return Err("Prepared weights and biases must be finite");
        }
        let mut bias_layers = [false; MAX_LAYERS];
        let mut offset = 0;
        let mut connections = expected_weights;
        let mut neurons = 0usize;
        for (index, &size) in shape.iter().enumerate() {
            neurons = neurons
                .checked_add(size)
                .ok_or("Network dimensions overflow")?;
            if index > 0 {
                let has_bias = biases[offset..offset + size].iter().any(|&v| v != 0.0);
                bias_layers[index - 1] = has_bias;
                if has_bias {
                    neurons = neurons
                        .checked_add(1)
                        .ok_or("Network dimensions overflow")?;
                    connections = connections
                        .checked_add(size)
                        .ok_or("Network dimensions overflow")?;
                }
                offset += size;
            }
        }
        // A large input-only neuron vector can dwarf its dense parameters. Check
        // that conversion cost before constructing any native graph allocation.
        let graph_bytes = checked_bytes(neurons, size_of::<Neuron<f64>>())?
            .checked_add(checked_bytes(connections, size_of::<Connection<f64>>())?)
            .and_then(|n| n.checked_add(shape.len() * size_of::<Layer<f64>>()))
            .ok_or("Conversion storage overflow")?;
        if graph_bytes > MAX_BYTES {
            return Err("Temporary conversion graph exceeds 64 MiB");
        }
        let (activation, steepness) = native_activation(activation);
        let mut layers = reserve(shape.len())?;
        let mut weight_offset = 0;
        let mut bias_offset = 0;
        for (index, &size) in shape.iter().enumerate() {
            let mut layer = Layer {
                neurons: reserve(size + usize::from(bias_layers[index]))?,
            };
            for output in 0..size {
                let mut neuron = Neuron::new(activation, steepness);
                if index > 0 {
                    let previous = shape[index - 1];
                    let has_bias = bias_layers[index - 1];
                    neuron.connections = reserve(previous + usize::from(has_bias))?;
                    // Existing WASM starts the sum at bias, then visits inputs in
                    // order. Preserve that order for randomized nonzero biases.
                    if has_bias {
                        neuron.connections.push(Connection::new(
                            previous,
                            output,
                            biases[bias_offset + output],
                        ));
                    }
                    for input in 0..previous {
                        neuron.connections.push(Connection::new(
                            input,
                            output,
                            weights[weight_offset + input * size + output],
                        ));
                    }
                }
                layer.neurons.push(neuron);
            }
            if bias_layers[index] {
                layer.neurons.push(Neuron::new_bias());
            }
            layers.push(layer);
            if index > 0 {
                weight_offset += shape[index - 1] * size;
                bias_offset += size;
            }
        }
        let graph = Network {
            layers,
            connection_rate: 1.0,
        };
        Self::from_native(&graph)
    }

    fn from_native(graph: &Network<f64>) -> BuildResult<Self> {
        let staging = checked_bytes(
            graph
                .num_inputs()
                .checked_add(graph.num_outputs())
                .ok_or("Staging size overflow")?,
            size_of::<f64>(),
        )?
        // Native estimated_bytes already includes the ordinary workspace. Count
        // the remaining wrapper fields (including estimated), shared cache and
        // both Rc control blocks rather than hand-listing descriptor fields.
        .checked_add(size_of::<Self>() - size_of::<InferenceWorkspace<f64>>())
        .and_then(|n| n.checked_add(size_of::<ArrayBoundary>()))
        .and_then(|n| n.checked_add(4 * size_of::<usize>()))
        .ok_or("Staging size overflow")?;
        let native_limit = MAX_BYTES
            .checked_sub(staging)
            .ok_or("Prepared staging exceeds 64 MiB")?;
        let network = PreparedNetwork::from_network_with_limits(
            graph,
            PreparationLimits {
                max_bytes: native_limit,
                ..PreparationLimits::default()
            },
        )
        .map_err(|_| "Native prepared snapshot exceeds limits or has invalid parameters")?;
        let workspace = network
            .workspace()
            .map_err(|_| "Prepared workspace allocation failed")?;
        let estimated = network.estimated_bytes() + staging;
        let mut inputs = reserve(network.num_inputs())?;
        inputs.resize(network.num_inputs(), 0.0);
        let mut outputs = reserve(network.num_outputs())?;
        outputs.resize(network.num_outputs(), 0.0);
        let arrays = ArrayBoundary::new().map_err(|_| "Cannot capture typed-array intrinsics")?;
        Ok(Self {
            network: Rc::new(network),
            workspace,
            inputs,
            outputs,
            arrays: Rc::new(arrays),
            estimated,
        })
    }

    fn evaluate(&mut self) -> Result<(), PreparedError> {
        self.network
            .run_into(&self.inputs, &mut self.outputs, &mut self.workspace)
    }
}

fn native_activation(activation: A) -> (NativeActivation, f64) {
    use NativeActivation as N;
    let mapped = match activation {
        A::Linear => N::Linear,
        A::Sigmoid => N::Sigmoid,
        // 2 sigmoid(x)-1 = tanh(x/2), subject to tiny floating-point rounding.
        A::SymmetricSigmoid => return (N::Tanh, 0.5),
        A::Tanh => N::Tanh,
        A::ReLU => N::ReLU,
        A::LeakyReLU => N::LeakyRelu,
        A::Swish => N::Swish,
        A::Gaussian => N::Gaussian,
        A::Elliot => N::Elliot,
        A::SymmetricElliot => N::ElliotSymmetric,
        A::Sine => N::Sin,
        A::Cosine => N::Cos,
        A::SinSymmetric => N::SinSymmetric,
        A::CosSymmetric => N::CosSymmetric,
        A::ThresholdSymmetric | A::StepSymmetric => N::ThresholdSymmetric,
        A::Threshold | A::Step => N::Threshold,
    };
    (mapped, 1.0)
}

fn checked_bytes(count: usize, width: usize) -> BuildResult<usize> {
    count
        .checked_mul(width)
        .ok_or("Conversion storage overflow")
}
fn reserve<T>(length: usize) -> BuildResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| "Prepared allocation failed")?;
    Ok(values)
}
fn inference_error(error: PreparedError) -> JsValue {
    // Static messages avoid growing Rust memory and detaching a caller view on
    // an error. The public boundary also rejects this module's memory buffers.
    JsValue::from_str(match error {
        PreparedError::NonFiniteInput { .. } => "Prepared inputs must be finite",
        PreparedError::NumericOverflow { .. } => {
            "Prepared inference overflowed; normalize inputs and weights"
        }
        _ => "Prepared inference shape or workspace mismatch",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::WasmNeuralNetwork;

    const ACTIVATIONS: [A; 18] = [
        A::Linear,
        A::Sigmoid,
        A::SymmetricSigmoid,
        A::Tanh,
        A::ReLU,
        A::LeakyReLU,
        A::Swish,
        A::Gaussian,
        A::Elliot,
        A::SymmetricElliot,
        A::Sine,
        A::Cosine,
        A::SinSymmetric,
        A::CosSymmetric,
        A::ThresholdSymmetric,
        A::Threshold,
        A::StepSymmetric,
        A::Step,
    ];

    fn evaluate(prepared: &mut WasmPreparedNetwork, input: &[f64]) -> Vec<f64> {
        prepared.inputs.copy_from_slice(input);
        prepared.evaluate().unwrap();
        prepared.outputs.clone()
    }

    #[test]
    fn eighteen_activations_match_existing_dense_wasm_semantics() {
        for activation in ACTIVATIONS {
            for shape in [&[2, 2, 1][..], &[5, 7, 3][..]] {
                let mut source = WasmNeuralNetwork::new(shape, activation).unwrap();
                let weights = (0..source.get_weights().len())
                    .map(|i| ((i * 7 % 19) as f64 - 9.0) / 23.0)
                    .collect::<Vec<_>>();
                source.set_weights(&weights).unwrap();
                let mut prepared = source.prepare().unwrap();
                assert_eq!(prepared.input_size(), shape[0]);
                assert_eq!(prepared.output_size(), *shape.last().unwrap());
                assert!(prepared.estimated_bytes() <= MAX_BYTES);
                for sign in [-1.0, 1.0] {
                    let input = (0..shape[0])
                        .map(|i| sign * (i as f64 - 1.5) / 3.0)
                        .collect::<Vec<_>>();
                    let expected = source.run(&input).unwrap();
                    let actual = evaluate(&mut prepared, &input);
                    for (&left, &right) in actual.iter().zip(&expected) {
                        assert!(
                            (left - right).abs() < 1e-12,
                            "{activation:?}: {left} != {right}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn input_major_layout_precision_and_source_mutation_isolation() {
        let mut source = WasmNeuralNetwork::new(&[2, 2, 1], A::Linear).unwrap();
        source.set_weights(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut prepared = source.prepare().unwrap();
        assert_eq!(evaluate(&mut prepared, &[1.0, 2.0]), [95.0]);
        source.set_weights(&[0.0; 6]).unwrap();
        drop(source);
        assert_eq!(evaluate(&mut prepared, &[1.0, 2.0]), [95.0]);
        let mut precise =
            WasmPreparedNetwork::from_dense(&[1, 1], &[1.000000000001], &[0.0], A::Linear).unwrap();
        assert_eq!(evaluate(&mut precise, &[1.0]), [1.000000000001]);
    }

    #[test]
    fn nonzero_biases_preserve_bias_first_accumulation() {
        let mut cancellation =
            WasmPreparedNetwork::from_dense(&[2, 1], &[1e20, -1e20], &[1.0], A::Linear).unwrap();
        assert_eq!(evaluate(&mut cancellation, &[1.0, 1.0]), [0.0]);
        let mut biased = WasmPreparedNetwork::from_dense(
            &[2, 2, 1],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.25, -0.5, 1.0],
            A::Linear,
        )
        .unwrap();
        assert_eq!(evaluate(&mut biased, &[1.0, 2.0]), [94.25]);
    }

    #[test]
    fn checked_conversion_rejects_bad_shape_parameters_and_temporary_graph_budget() {
        for (shape, weights, biases) in [
            (&[1][..], &[1.0][..], &[0.0][..]),
            (&[1, 1], &[], &[0.0]),
            (&[1, 1], &[f64::NAN], &[0.0]),
            (&[1, 1], &[1.0], &[f64::INFINITY]),
            (&[usize::MAX, 2], &[], &[]),
        ] {
            assert!(WasmPreparedNetwork::from_dense(shape, weights, biases, A::Linear).is_err());
        }
        let weights = vec![0.0; 900_000];
        assert_eq!(
            WasmPreparedNetwork::from_dense(&[900_000, 1], &weights, &[0.0], A::Linear).err(),
            Some("Temporary conversion graph exceeds 64 MiB")
        );
    }

    #[test]
    fn numeric_errors_preserve_output_and_allow_subsequent_valid_inference() {
        let mut prepared =
            WasmPreparedNetwork::from_dense(&[1, 1], &[f64::MAX], &[0.0], A::Linear).unwrap();
        prepared.outputs[0] = 17.0;
        for input in [f64::NAN, f64::INFINITY, 2.0] {
            prepared.inputs[0] = input;
            assert!(prepared.evaluate().is_err());
            assert_eq!(prepared.outputs, [17.0]);
        }
        assert_eq!(evaluate(&mut prepared, &[0.0]), [0.0]);
        let mut output = [23.0];
        assert!(prepared
            .network
            .run_into(&[], &mut output, &mut prepared.workspace)
            .is_err());
        assert_eq!(output, [23.0]);
    }
}
