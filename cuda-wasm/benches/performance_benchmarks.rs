//! Actual profiler overhead measurements on the same CPU workload.
//! Criterion reports distributions; there are no machine-dependent pass/fail ratios.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cuda_rust_wasm::profiling::{GlobalProfiler, ScopedTimer};

fn work(values: &[f32]) -> f32 {
    values.iter().map(|x| x * x + 0.25).sum()
}
fn profiler_cost(c: &mut Criterion) {
    let values = vec![0.5f32; 256];
    assert_eq!(work(&values), 128.0);
    let profiler = GlobalProfiler::new();
    profiler.enable();
    let mut group = c.benchmark_group("cpu_profiler_same_workload");
    group.bench_function("without_scope", |b| {
        b.iter(|| black_box(work(black_box(&values))))
    });
    group.bench_function("with_scope_and_aggregation", |b| {
        b.iter(|| {
            let _scope = ScopedTimer::new(&profiler, "cpu_work".into());
            black_box(work(black_box(&values)));
        })
    });
    group.finish();
    assert!(profiler.get_profile("cpu_work").unwrap().count > 0);
}
criterion_group!(benches, profiler_cost);
criterion_main!(benches);
