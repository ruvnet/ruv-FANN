//! Reproducible public Network::train timing for baseline/candidate binaries.
//! Build with --release --locked --no-default-features, then run on a quiet host.
//! Compile this identical file against both source revisions. Fixture creation,
//! network cloning and destruction are outside each timed block; raw CSV includes
//! 21 measurements after warm-up. This does not evaluate model quality.
use num_traits::Float;
use ruv_fann::{ActivationFunction, Network};
use std::{
    hint::black_box,
    ops::{AddAssign, MulAssign, SubAssign},
    time::Instant,
};

fn measure<T: Float + AddAssign + SubAssign + MulAssign>(
    precision: &str,
    shape: &[usize],
    reordered: bool,
) {
    let mut initial = Network::<T>::new(shape);
    let weights = (0..initial.total_connections())
        .map(|index| T::from(((index * 17 % 43) as f64 - 21.0) / 128.0).unwrap())
        .collect::<Vec<_>>();
    initial.set_weights(&weights).unwrap();
    initial.set_activation_function_hidden(ActivationFunction::Sigmoid);
    initial.set_activation_function_output(ActivationFunction::Sigmoid);
    if reordered {
        for layer in initial.layers.iter_mut().skip(1) {
            for neuron in &mut layer.neurons {
                neuron.connections.reverse();
            }
        }
    }
    let rows = 16;
    let inputs = (0..rows)
        .map(|sample| {
            (0..shape[0])
                .map(|feature| {
                    T::from(((sample * 7 + feature * 3) % 17) as f64 / 16.0 - 0.5).unwrap()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let targets = (0..rows)
        .map(|sample| {
            (0..*shape.last().unwrap())
                .map(|output| T::from(((sample * 3 + output) % 11) as f64 / 16.0).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let iterations = if shape[0] <= 2 { 64 } else { 4 };
    let shape_tag = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x");
    let topology = if reordered { "reordered" } else { "dense" };
    for sample in 0..28 {
        let mut network = initial.clone();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(&mut network)
                .train(black_box(&inputs), black_box(&targets), 0.03125, 1)
                .unwrap();
        }
        let nanoseconds = start.elapsed().as_nanos() as f64 / (iterations * rows) as f64;
        let checksum = network
            .layers
            .iter()
            .flat_map(|layer| &layer.neurons)
            .flat_map(|neuron| &neuron.connections)
            .map(|connection| connection.weight.to_f64().unwrap())
            .sum::<f64>();
        let weight_bits_hash = network
            .layers
            .iter()
            .flat_map(|layer| &layer.neurons)
            .flat_map(|neuron| &neuron.connections)
            .fold(0xcbf29ce484222325_u64, |hash, connection| {
                // Deterministic comparison aid, not a cryptographic digest.
                (hash ^ connection.weight.to_f64().unwrap().to_bits()).wrapping_mul(0x100000001b3)
            });
        black_box(&network);
        if sample >= 7 {
            println!("{precision},{shape_tag},{topology},{},{iterations},{rows},{nanoseconds:.6},{checksum:.17},{weight_bits_hash:016x}", sample - 7);
        }
    }
}

fn main() {
    println!("precision,shape,topology,measurement,iterations,rows,ns_per_pattern,weight_checksum,weight_bits_hash");
    for shape in [
        &[2, 3, 1][..],
        &[32, 64, 32, 8][..],
        &[128, 128, 64, 16][..],
    ] {
        for reordered in [false, true] {
            measure::<f32>("f32", shape, reordered);
            measure::<f64>("f64", shape, reordered);
        }
    }
}
