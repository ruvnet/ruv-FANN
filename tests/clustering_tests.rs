#![cfg(feature = "clustering")]

use ruv_fann::{
    clustering::{fit, Config, InputClusteringError},
    TrainingData,
};

#[test]
fn training_inputs_use_the_shared_codebook_and_preserve_supervised_data() {
    let data = TrainingData {
        inputs: vec![
            vec![-10.0, -9.0],
            vec![-9.0, -10.0],
            vec![9.0, 10.0],
            vec![10.0, 9.0],
        ],
        outputs: vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]],
    };
    let original = data.clone();
    let config = Config {
        clusters: 2,
        epochs: 6,
        batch_size: Some(2),
        seed: 42,
        ..Config::default()
    };
    let model = data.cluster_inputs(&config).unwrap();
    let flat = data.inputs.iter().flatten().copied().collect::<Vec<_>>();
    let shared = fit(&flat, 2, &config).unwrap();
    assert_eq!(model.centroids, shared.centroids);
    assert_eq!(model.assignments, shared.assignments);
    assert_eq!(model.dimensions, 2);
    assert_eq!(model.assignments[0], model.assignments[1]);
    assert_eq!(model.assignments[2], model.assignments[3]);
    assert_ne!(model.assignments[0], model.assignments[2]);
    assert!(model.mse < 1.0);
    assert_eq!(data.inputs, original.inputs);
    assert_eq!(data.outputs, original.outputs);
}

#[test]
fn one_cluster_is_the_input_mean_independent_of_targets() {
    let data = TrainingData {
        inputs: vec![vec![0.0, 2.0], vec![4.0, 6.0], vec![8.0, 10.0]],
        outputs: vec![],
    };
    let model = data
        .cluster_inputs(&Config {
            clusters: 1,
            batch_size: Some(2),
            ..Config::default()
        })
        .unwrap();
    assert_eq!(model.centroids, vec![4.0, 6.0]);
    assert_eq!(model.assignments, vec![0, 0, 0]);
    assert_eq!(model.predict(&[4.0, 6.0]).unwrap(), vec![0]);
}

#[test]
fn input_shape_and_finiteness_fail_before_a_model_is_returned() {
    for inputs in [vec![], vec![vec![]]] {
        let data = TrainingData::<f32> {
            inputs,
            outputs: vec![],
        };
        assert!(matches!(
            data.cluster_inputs(&Config::default()),
            Err(InputClusteringError::EmptyInputs)
        ));
    }
    let data = TrainingData {
        inputs: vec![vec![1.0, 2.0], vec![3.0]],
        outputs: vec![],
    };
    assert!(matches!(
        data.cluster_inputs(&Config::default()),
        Err(InputClusteringError::RaggedInputs {
            row: 1,
            expected: 2,
            actual: 1
        })
    ));
    for coordinate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let data = TrainingData {
            inputs: vec![vec![coordinate]],
            outputs: vec![],
        };
        assert!(matches!(
            data.cluster_inputs(&Config {
                clusters: 1,
                ..Config::default()
            }),
            Err(InputClusteringError::Clustering(_))
        ));
    }
}
