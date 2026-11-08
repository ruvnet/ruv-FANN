//! Local measured baseline/RNN operations on the same input/horizon.
//! No synthetic external results or speedup/accuracy claims are embedded.
#[path = "../tests/support/mod.rs"]
mod support;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
fn local_comparison(c: &mut Criterion) {
    let history = support::frame(2, 32);
    let mut baseline = support::baseline(3);
    baseline.fit(history.clone()).unwrap();
    let mut group = c.benchmark_group("local_2_series_horizon_3");
    group.throughput(Throughput::Elements(6));
    group.bench_function("naive_forecast", |b| {
        b.iter(|| black_box(baseline.predict().unwrap()))
    });
    #[cfg(feature = "models")]
    {
        let mut recurrent = support::neural(3);
        recurrent.fit(history.clone()).unwrap();
        group.bench_function("rnn_forecast", |b| {
            b.iter(|| black_box(recurrent.predict().unwrap()))
        });
    }
    group.finish();
    c.bench_function("naive_cv_2_windows_horizon_3", |b| {
        b.iter_batched(
            || support::baseline(3),
            |mut model| {
                black_box(
                    model
                        .cross_validation(
                            history.clone(),
                            neuro_divergent::CrossValidationConfig::new(2, 3),
                        )
                        .unwrap(),
                )
            },
            criterion::BatchSize::SmallInput,
        )
    });
}
criterion_group!(benches, local_comparison);
criterion_main!(benches);
