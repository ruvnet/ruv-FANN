//! Single-row scoring of small dense heads through the prepared engine,
//! against the same arithmetic written by hand. Small heads scored one row at
//! a time are dominated by fixed per-call cost, which batch benchmarks hide.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use ruv_fann::{ActivationFunction, DenseLayer, PreparedNetwork};

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn bench(c: &mut Criterion) {
    let w1: Vec<f32> = (0..10).map(|i| (i as f32 * 0.37).sin()).collect();
    let b1 = [0.25f32];
    let linear = PreparedNetwork::from_dense_layers(&[DenseLayer {
        inputs: 10,
        outputs: 1,
        weights: &w1,
        bias: Some(&b1),
        activation: ActivationFunction::Sigmoid,
        steepness: 1.0,
    }])
    .unwrap();

    let wh: Vec<f32> = (0..40).map(|i| (i as f32 * 0.11).cos()).collect();
    let bh = [0.1f32, -0.2, 0.3, -0.4];
    let wo = [0.5f32, -0.6, 0.7, -0.8];
    let bo = [0.05f32];
    let mlp = PreparedNetwork::from_dense_layers(&[
        DenseLayer {
            inputs: 10,
            outputs: 4,
            weights: &wh,
            bias: Some(&bh),
            activation: ActivationFunction::ReLU,
            steepness: 1.0,
        },
        DenseLayer {
            inputs: 4,
            outputs: 1,
            weights: &wo,
            bias: Some(&bo),
            activation: ActivationFunction::Sigmoid,
            steepness: 1.0,
        },
    ])
    .unwrap();

    let x: Vec<f32> = (0..10).map(|i| (i as f32 * 0.21).cos()).collect();
    let mut group = c.benchmark_group("prepared_single_row");

    group.bench_function("hand_10x1_sigmoid", |b| {
        b.iter(|| {
            let x = black_box(&x);
            sigmoid(
                x.iter()
                    .zip(black_box(&w1))
                    .map(|(a, w)| a * w)
                    .sum::<f32>()
                    + b1[0],
            )
        })
    });

    let mut ws = linear.workspace().unwrap();
    let mut out = [0.0f32];
    group.bench_function("prepared_10x1_sigmoid", |b| {
        b.iter(|| {
            linear.run_into(black_box(&x), &mut out, &mut ws).unwrap();
            out[0]
        })
    });

    let mut ws = mlp.workspace().unwrap();
    group.bench_function("prepared_10x4x1_relu_sigmoid", |b| {
        b.iter(|| {
            mlp.run_into(black_box(&x), &mut out, &mut ws).unwrap();
            out[0]
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
