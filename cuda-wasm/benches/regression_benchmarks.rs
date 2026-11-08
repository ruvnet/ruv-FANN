//! Criterion provides measured distributions and saved-baseline comparisons.
//! Example: cargo bench --features regression-tests --bench regression_benchmarks
//! -- --save-baseline reviewed; later use --baseline reviewed on the same machine.
//! No synthetic standard deviations, wall-clock ratios or fixed GPU targets.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cuda_rust_wasm::{memory::HostBuffer, parser::CudaParser};
fn regression_inputs(c: &mut Criterion) {
    let mut buffer = HostBuffer::<f32>::new(16384).unwrap();
    c.bench_function("cpu_fill_16384_f32", |b| {
        b.iter(|| buffer.fill(black_box(1.0)))
    });
    assert!(buffer.as_slice().iter().all(|&x| x == 1.0));
    let source = "__global__ void copy(float* a, float* b) { b[0] = a[0]; }";
    let parser = CudaParser::new();
    assert_eq!(parser.parse(source).unwrap().items.len(), 1);
    c.bench_function("parse_small_cuda_kernel", |b| {
        b.iter(|| black_box(parser.parse(black_box(source)).unwrap()))
    });
}
criterion_group!(benches, regression_inputs);
criterion_main!(benches);
