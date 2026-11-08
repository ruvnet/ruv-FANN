//! Native same-parameter batching comparison. Build once, then run the executable
//! without other builds: cargo build --release --locked --no-default-features
//! --example prepared_batch_benchmark; target/release/examples/prepared_batch_benchmark
//! Prints CSV. Timings include validation, transposition and canonical activation.
use ruv_fann::{prepared::BatchFloat, ActivationFunction as A, Network};
use std::{fmt::Debug, hint::black_box, time::Instant};

fn compare<T: BatchFloat + Debug>(precision: &str, shape: &[usize], activation: A, rows: usize) {
    let mut network = Network::<T>::new(shape);
    let weights = (0..network.total_connections())
        .map(|i| T::from(((i * 17 % 31) as f64 - 15.0) / 64.0).unwrap())
        .collect::<Vec<_>>();
    network.set_weights(&weights).unwrap();
    network.set_activation_function_hidden(activation);
    network.set_activation_function_output(activation);
    network.set_activation_steepness_hidden(T::one());
    network.set_activation_steepness_output(T::one());
    let prepared = network.prepare().unwrap();
    let mut single_workspace = prepared.workspace().unwrap();
    let bytes = prepared.batch_workspace_bytes(rows).unwrap();
    let started = Instant::now();
    let mut workspace = prepared.batch_workspace(rows, bytes).unwrap();
    let workspace_once_ns = started.elapsed().as_nanos();
    let input_size = prepared.num_inputs();
    let output_size = prepared.num_outputs();
    let inputs = (0..rows * input_size)
        .map(|i| T::from(((i * 7 % 13) as f64 - 6.0) / 8.0).unwrap())
        .collect::<Vec<_>>();
    let mut baseline = vec![T::zero(); rows * output_size];
    let mut output = baseline.clone();
    for (input, output) in inputs
        .chunks_exact(input_size)
        .zip(baseline.chunks_exact_mut(output_size))
    {
        prepared
            .run_into(input, output, &mut single_workspace)
            .unwrap();
    }
    prepared
        .run_batch_into(&inputs, &mut output, rows, &mut workspace)
        .unwrap();
    assert_eq!(
        output, baseline,
        "batch must retain every sample's addition order"
    );
    let iterations = (10_000_000 / (prepared.connection_count() * rows).max(1)).clamp(1, 10_000);
    let mut single_ns = [0.0; 9];
    let mut batch_ns = [0.0; 9];
    // Warm both complete public native APIs, then alternate timing order.
    for _ in 0..iterations {
        for (input, output) in inputs
            .chunks_exact(input_size)
            .zip(baseline.chunks_exact_mut(output_size))
        {
            prepared
                .run_into(input, output, &mut single_workspace)
                .unwrap();
        }
        prepared
            .run_batch_into(&inputs, &mut output, rows, &mut workspace)
            .unwrap();
    }
    for sample in 0..9 {
        for offset in 0..2 {
            let batch = (sample + offset) % 2 == 0;
            let started = Instant::now();
            for _ in 0..iterations {
                if batch {
                    prepared
                        .run_batch_into(
                            black_box(&inputs),
                            black_box(&mut output),
                            rows,
                            black_box(&mut workspace),
                        )
                        .unwrap();
                    black_box(&output);
                } else {
                    for (input, output) in black_box(&inputs)
                        .chunks_exact(input_size)
                        .zip(baseline.chunks_exact_mut(output_size))
                    {
                        prepared
                            .run_into(input, output, black_box(&mut single_workspace))
                            .unwrap();
                    }
                    black_box(&baseline);
                }
            }
            let elapsed = started.elapsed().as_nanos() as f64 / iterations as f64 / rows as f64;
            if batch {
                batch_ns[sample] = elapsed;
            } else {
                single_ns[sample] = elapsed;
            }
        }
    }
    single_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    batch_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let checksum = output
        .iter()
        .map(|value| value.to_f64().unwrap())
        .sum::<f64>();
    let shape_tag = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x");
    println!("{precision},{shape_tag},{activation:?},{},{},{},{rows},{},{bytes},{workspace_once_ns},{iterations},{:.3},{:.3},{:.6},0,{checksum:.12}",
        shape[0], shape[1], output_size, prepared.connection_count(), single_ns[4], batch_ns[4], single_ns[4] / batch_ns[4]);
}

fn main() {
    println!("precision,shape,activation,inputs,hidden,outputs,rows,connections,workspace_bytes,workspace_once_ns,iterations,median_single_ns_per_row,median_batch_ns_per_row,speed_ratio,max_absolute_error,checksum");
    for activation in [A::Linear, A::Sigmoid] {
        for shape in [
            &[5, 7, 3][..],
            &[100, 50, 25, 10][..],
            &[784, 128, 64, 10][..],
        ] {
            for rows in [1, 3, 8, 32] {
                compare::<f32>("f32", shape, activation, rows);
                compare::<f64>("f64", shape, activation, rows);
            }
        }
    }
}
