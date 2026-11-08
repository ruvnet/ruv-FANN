use ruv_fann::{ActivationFunction, Layer, Network, Neuron};

#[test]
fn known_dense_and_input_only_work_estimates_include_all_components() {
    let prepared = Network::<f64>::new(&[5, 7, 3]).prepare().unwrap();
    assert_eq!(prepared.connection_count(), 66);
    // 66 edges,10 computed neurons,8 I/O values,2 biases,3 layers.
    assert_eq!(
        prepared.inference_work_per_row().unwrap(),
        2 * 66 + 3 * 10 + 2 * 8 + 2 + 3
    );
    let mut identity = Network::<f32>::new(&[3, 1]);
    identity.layers.truncate(1);
    let identity = identity.prepare().unwrap();
    assert_eq!(identity.connection_count(), 0);
    assert_eq!(
        identity.inference_work_per_row().unwrap(),
        2 * (3 + 3) + 1 + 1
    );
}

#[test]
fn disconnected_activation_graph_exceeds_budget_despite_zero_connections() {
    let layers = (0..30_000)
        .map(|_| Layer {
            neurons: vec![Neuron::<f64>::new(ActivationFunction::Sin, 1.0)],
        })
        .collect();
    let graph = Network {
        layers,
        connection_rate: 1.0,
    };
    #[cfg(feature = "serde")]
    assert!(serde_json::to_string(&graph).unwrap().len() <= 4 * 1024 * 1024);
    let prepared = graph.prepare().unwrap();
    assert_eq!(prepared.connection_count(), 0);
    let work = prepared.inference_work_per_row().unwrap();
    assert_eq!(work, 3 * 29_999 + 2 * (1 + 1) + 30_000);
    assert!(work.checked_mul(4096).unwrap() > 100_000_000);
    assert!(work.checked_mul(usize::MAX).is_none());
    // This test checks the preflight estimate; it deliberately does not execute
    // the half-billion-unit batch that the former edge-only budget admitted.
}
