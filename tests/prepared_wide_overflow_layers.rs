//! Exercise direct and non-first wide single-inference overflow priority.
use ruv_fann::{ActivationFunction as A, BatchFloat, Network, PreparedError};
use std::fmt::Debug;

fn overflow_priority<T: BatchFloat + Debug>() {
    for shape in [&[65, 4][..], &[65, 65, 4][..]] {
        let mut model = Network::<T>::new(shape);
        model
            .set_weights(&vec![T::zero(); model.total_connections()])
            .unwrap();
        model.set_activation_function_hidden(A::Linear);
        model.set_activation_steepness_hidden(T::one());
        model.set_activation_function_output(A::Linear);
        model.set_activation_steepness_output(T::one());
        let last = model.layers.len() - 1;
        if last == 2 {
            model.layers[1].neurons[0].connections[0].weight = T::one();
            model.layers[1].neurons[1].connections[1].weight = T::one();
        }
        model.layers[last].neurons[0].connections[0].weight = T::one();
        model.layers[last].neurons[0].activation_steepness = T::max_value();
        model.layers[last].neurons[1].connections[1].weight = T::max_value();
        let prepared = model.prepare().unwrap();
        let mut workspace = prepared.workspace().unwrap();
        let mut input = vec![T::zero(); 65];
        input[0] = T::from(2).unwrap();
        input[1] = T::from(2).unwrap();
        let marker = T::from(42).unwrap();
        let mut output = [marker; 4];
        assert_eq!(
            prepared.run_into(&input, &mut output, &mut workspace),
            Err(PreparedError::NumericOverflow {
                layer: last,
                neuron: 0
            })
        );
        assert_eq!(output, [marker; 4]);
        input.fill(T::zero());
        prepared
            .run_into(&input, &mut output, &mut workspace)
            .unwrap();
        assert_eq!(output, [T::zero(); 4]);
    }
}

#[test]
fn direct_and_general_wide_layers_keep_overflow_priority_f32() {
    overflow_priority::<f32>();
}

#[test]
fn direct_and_general_wide_layers_keep_overflow_priority_f64() {
    overflow_priority::<f64>();
}
