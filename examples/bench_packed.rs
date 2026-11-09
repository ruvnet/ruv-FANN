//! Fixed-workload native benchmark. Outputs JSON lines with raw timing blocks.
//! cargo +1.88.0 build --release --locked --no-default-features --example bench_packed
use ruv_fann::{prepared::BatchFloat, ActivationFunction as A, Network};
use std::{fmt::Debug, hint::black_box, time::Instant};

fn measure<T: BatchFloat + Debug>(
    id: usize,
    shape: &[usize],
    rows: usize,
    activation: A,
    samples: usize,
) {
    let mut model = Network::<T>::new(shape);
    let weights = (0..model.total_connections())
        .map(|i| T::from(((i * 17 % 31) as f64 - 15.0) / 64.0).unwrap())
        .collect::<Vec<_>>();
    model.set_weights(&weights).unwrap();
    model.set_activation_function_hidden(activation);
    model.set_activation_function_output(activation);
    model.set_activation_steepness_hidden(T::one());
    model.set_activation_steepness_output(T::one());
    let packed = model.prepare().unwrap();
    let mut scalar = packed.workspace().unwrap();
    let mut batch = packed
        .batch_workspace(rows, packed.batch_workspace_bytes(rows).unwrap())
        .unwrap();
    let inputs = (0..rows * shape[0])
        .map(|i| {
            T::from(((i % shape[0] * 7 % 13) as f64 - 6.0) / 8.0 + (i / shape[0] % 5) as f64 / 64.0)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected = vec![T::zero(); rows * shape.last().unwrap()];
    for (input, output) in inputs
        .chunks_exact(shape[0])
        .zip(expected.chunks_exact_mut(packed.num_outputs()))
    {
        packed.run_into(input, output, &mut scalar).unwrap();
    }
    let mut output = expected.clone();
    packed
        .run_batch_into(&inputs, &mut output, rows, &mut batch)
        .unwrap();
    assert_eq!(output, expected);
    let mut iterations = 1;
    for _ in 0..8 {
        let start = Instant::now();
        for _ in 0..iterations {
            packed
                .run_batch_into(
                    black_box(&inputs),
                    black_box(&mut output),
                    rows,
                    black_box(&mut batch),
                )
                .unwrap();
            black_box(&output);
        }
        if start.elapsed().as_nanos() >= 2_000_000 || iterations >= 4096 {
            break;
        }
        iterations = (iterations * 4).min(4096);
    }
    let mut samples_ns = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations {
            packed
                .run_batch_into(
                    black_box(&inputs),
                    black_box(&mut output),
                    rows,
                    black_box(&mut batch),
                )
                .unwrap();
            black_box(&output);
        }
        samples_ns.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    assert_eq!(output, expected);
    let checksum: f64 = output.iter().map(|v| v.to_f64().unwrap()).sum();
    assert!(checksum.is_finite());
    println!("{{\"schema\":\"ruv-fann.native-packed-bench/1\",\"case\":{id},\"precision\":\"{}\",\"shape\":{shape:?},\"activation\":\"{activation:?}\",\"rows\":{rows},\"iterations\":{iterations},\"samplesNs\":{samples_ns:?},\"maximumAbsoluteError\":0,\"checksum\":{checksum:?},\"snapshotBytes\":{},\"workspaceBytes\":{}}}",
        if std::mem::size_of::<T>() == 4 { "f32" } else { "f64" }, packed.snapshot_bytes(), batch.estimated_bytes());
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut selected = None;
    let mut samples = 20;
    for pair in args.chunks_exact(2) {
        match pair[0].as_str() {
            "--case" => selected = Some(pair[1].parse::<usize>().expect("case integer")),
            "--samples" => samples = pair[1].parse::<usize>().expect("samples integer"),
            _ => panic!("expected --case ID --samples COUNT"),
        }
    }
    assert!(args.len() % 2 == 0 && (1..=40).contains(&samples));
    let mut id = 0;
    for activation in [A::Linear, A::Sigmoid] {
        for shape in [
            &[5, 7, 3][..],
            &[100, 50, 25, 10][..],
            &[784, 128, 64, 10][..],
        ] {
            for rows in [1, 3, 8, 32, 128] {
                if selected.is_none() || selected == Some(id) {
                    measure::<f32>(id, shape, rows, activation, samples);
                }
                id += 1;
                if selected.is_none() || selected == Some(id) {
                    measure::<f64>(id, shape, rows, activation, samples);
                }
                id += 1;
            }
        }
    }
    assert!(selected.is_none() || selected.unwrap() < id);
}
