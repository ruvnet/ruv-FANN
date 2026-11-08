//! Actual joint recurrent/projection optimization, including 20 complete epochs.
#[path = "../tests/support/mod.rs"]
mod support;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
fn training(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn_bptt_20_epochs");
    #[cfg(feature = "models")]
    {
        use neuro_divergent::models::BaseModel;
        for count in [8, 32] {
            let data = support::model_dataset(count, 12, 3);
            group.throughput(Throughput::Elements((count * 20) as u64));
            group.bench_with_input(BenchmarkId::new("samples", count), &data, |b, data| {
                b.iter_batched(
                    || support::rnn(3, 8, 20),
                    |mut model| black_box(model.fit(black_box(data)).unwrap()),
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}
criterion_group!(benches, training);
criterion_main!(benches);
