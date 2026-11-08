//! Measured scalar and runtime-selected SIMD CPU work, with no invented CUDA times.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cuda_rust_wasm::simd::vector_ops::{vector_add_f32, vector_dot_f32};
fn cpu_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vector_operations");
    for len in [1024usize, 16384, 262144] {
        let left = vec![1.0f32; len];
        let right = vec![2.0f32; len];
        let mut output = vec![0.0f32; len];
        vector_add_f32(&left, &right, &mut output);
        assert!(output.iter().all(|&x| x == 3.0));
        assert_eq!(vector_dot_f32(&left, &right), 2.0 * len as f32);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("scalar_add", len), &len, |b, _| {
            b.iter(|| {
                for ((a, b), out) in black_box(&left).iter().zip(&right).zip(&mut output) {
                    *out = a + b;
                }
                black_box(&output);
            });
        });
        group.bench_with_input(BenchmarkId::new("selected_simd_add", len), &len, |b, _| {
            b.iter(|| {
                vector_add_f32(black_box(&left), &right, &mut output);
                black_box(&output);
            });
        });
        group.bench_with_input(BenchmarkId::new("selected_simd_dot", len), &len, |b, _| {
            b.iter(|| black_box(vector_dot_f32(black_box(&left), &right)));
        });
    }
    group.finish();
}
criterion_group!(benches, cpu_vectors);
criterion_main!(benches);
