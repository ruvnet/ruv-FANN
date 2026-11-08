//! Real fitted RNN inference. Input preparation and fitting are outside timing.
#[path = "../tests/support/mod.rs"]
mod support;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
fn inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn_forecast_batch");
    #[cfg(feature = "models")]
    {
        use neuro_divergent::models::{BaseModel, TimeSeriesInput};
        let mut model = support::rnn(3, 16, 20);
        model.fit(&support::model_dataset(16, 12, 3)).unwrap();
        for batch in [1, 16, 64] {
            let inputs: Vec<_> = (0..batch)
                .map(|_| TimeSeriesInput::new(vec![0.1; 12]))
                .collect();
            group.throughput(Throughput::Elements(batch as u64));
            group.bench_with_input(BenchmarkId::from_parameter(batch), &inputs, |b, inputs| {
                b.iter(|| black_box(model.predict_batch(black_box(inputs)).unwrap()))
            });
        }
    }
    group.finish();
}
criterion_group!(benches, inference);
criterion_main!(benches);
