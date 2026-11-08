//! Bounded import of the existing native serde model format.
use super::{checked_bytes, BuildResult, WasmPreparedNetwork, MAX_BYTES};
use ruv_fann::{Connection, Layer, Network, Neuron};
use std::mem::size_of;

pub(super) const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;

pub(super) fn from_json(json: &str) -> BuildResult<WasmPreparedNetwork> {
    if json.len() > MAX_JSON_BYTES {
        return Err("Network JSON exceeds 4 MiB");
    }
    // serde_json retains its default nesting-depth limit. Floats/usize indices
    // are checked by serde and native preparation validates topology/parameters.
    let graph: Network<f64> =
        serde_json::from_str(json).map_err(|_| "Invalid native Network<f64> JSON")?;
    let neurons = graph
        .layers
        .iter()
        .try_fold(0usize, |n, layer| n.checked_add(layer.neurons.len()))
        .ok_or("Imported graph size overflow")?;
    let connections = graph
        .layers
        .iter()
        .flat_map(|l| &l.neurons)
        .try_fold(0usize, |n, neuron| n.checked_add(neuron.connections.len()))
        .ok_or("Imported graph size overflow")?;
    let graph_bytes = checked_bytes(neurons, size_of::<Neuron<f64>>())?
        .checked_add(checked_bytes(connections, size_of::<Connection<f64>>())?)
        .and_then(|n| n.checked_add(graph.layers.len().checked_mul(size_of::<Layer<f64>>())?))
        .ok_or("Imported graph size overflow")?;
    if graph_bytes > MAX_BYTES {
        return Err("Imported graph exceeds 64 MiB");
    }
    WasmPreparedNetwork::from_native(&graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_fann::ActivationFunction as A;

    const ACTIVATIONS: [A; 21] = [
        A::Linear,
        A::Threshold,
        A::ThresholdSymmetric,
        A::Sigmoid,
        A::SigmoidSymmetric,
        A::Tanh,
        A::Gaussian,
        A::GaussianSymmetric,
        A::Elliot,
        A::ElliotSymmetric,
        A::LinearPiece,
        A::LinearPieceSymmetric,
        A::ReLU,
        A::ReLULeaky,
        A::Sin,
        A::Cos,
        A::SinSymmetric,
        A::CosSymmetric,
        A::Gelu,
        A::Swish,
        A::LeakyRelu,
    ];

    fn model(activation: A, sparse: bool) -> Network<f64> {
        let mut graph = Network::<f64>::new(&[3, 5, 2]);
        let weights = (0..graph.total_connections())
            .map(|i| ((i * 7 % 19) as f64 - 9.0) / 23.0)
            .collect::<Vec<_>>();
        graph.set_weights(&weights).unwrap();
        graph.set_activation_function_hidden(activation);
        graph.set_activation_function_output(activation);
        if sparse {
            for layer in graph.layers.iter_mut().skip(1) {
                for neuron in layer.neurons.iter_mut().filter(|n| !n.is_bias) {
                    neuron.connections.reverse();
                    neuron.connections.remove(1);
                    neuron.connections.push(neuron.connections[0].clone());
                }
            }
        }
        graph
    }

    #[test]
    fn all_native_activations_and_sparse_order_roundtrip_through_json() {
        for activation in ACTIVATIONS {
            for sparse in [false, true] {
                let mut graph = model(activation, sparse);
                let input = [0.2, -0.7, 0.9];
                let expected = graph.run(&input).unwrap();
                let mut prepared = from_json(&serde_json::to_string(&graph).unwrap()).unwrap();
                prepared.inputs.copy_from_slice(&input);
                prepared.evaluate().unwrap();
                for (&left, &right) in prepared.outputs.iter().zip(&expected) {
                    assert!(
                        (left - right).abs() < 1e-12,
                        "{activation:?}, sparse={sparse}: {left} vs {right}"
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_nonfinite_future_activation_and_oversized_json_fail() {
        for text in ["", "{}", "[]", r#"{"layers":[],"connection_rate":1}"#] {
            assert!(from_json(text).is_err());
        }
        let valid = serde_json::to_string(&model(A::Linear, true)).unwrap();
        for replacement in ["FutureActivation", "NaN"] {
            assert!(from_json(&valid.replace("Linear", replacement)).is_err());
        }
        let invalid = valid.replace("\"weight\":", "\"weight\":1e999,\"ignored\":");
        assert!(from_json(&invalid).is_err());
        let mut graph = model(A::Linear, true);
        graph.layers[1].neurons[0].connections[0].from_neuron = usize::MAX;
        assert!(from_json(&serde_json::to_string(&graph).unwrap()).is_err());
        assert!(from_json(&" ".repeat(MAX_JSON_BYTES + 1)).is_err());
    }

    #[test]
    fn native_export_fixtures_are_reproducible() {
        for (name, sparse) in [("dense", false), ("sparse", true)] {
            let mut graph = model(if sparse { A::Swish } else { A::Sigmoid }, sparse);
            if !sparse {
                graph
                    .train(
                        &[vec![0.2, -0.7, 0.9], vec![-0.3, 0.5, 0.1]],
                        &[vec![0.8, 0.2], vec![0.1, 0.9]],
                        0.05,
                        40,
                    )
                    .unwrap();
            }
            let input = vec![0.2, -0.7, 0.9];
            let output = graph.run(&input).unwrap();
            let fixture = serde_json::json!({"network":graph,"input":input,"output":output,
                "description":if sparse {"native sparse shuffled+duplicate Swish"} else {"native dense Sigmoid trained40epochs from fixed weights"}});
            let text = serde_json::to_string_pretty(&fixture).unwrap();
            if let Ok(directory) = std::env::var("RUV_PREPARED_FIXTURE_DIR") {
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    std::path::Path::new(&directory).join(format!("prepared-{name}.json")),
                    &text,
                )
                .unwrap();
            }
            assert!(fixture["output"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v.as_f64().unwrap().is_finite()));
        }
    }
}
