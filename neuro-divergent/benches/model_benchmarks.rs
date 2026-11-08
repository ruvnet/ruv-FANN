//! Seeded construction only; no invented Python/model-family comparisons.
#[path = "../tests/support/mod.rs"]
mod support;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
fn construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn_seeded_construction");
    #[cfg(feature = "models")]
    for hidden in [4, 16, 64] {
        group.bench_with_input(BenchmarkId::new("hidden", hidden), &hidden, |b, &width| {
            b.iter(|| black_box(support::rnn(3, width, 20)))
        });
    }
    group.finish();
}
criterion_group!(benches, construction);
criterion_main!(benches);
