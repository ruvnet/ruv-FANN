//! Implemented preprocessing paths at stated observation counts.
use chrono::{Duration, TimeZone, Utc};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neuro_divergent_data::{
    features::FeatureEngine,
    preprocessing::{Scaler, StandardScaler},
    TimeSeriesDatasetBuilder,
};
fn preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");
    for n in [128, 4096] {
        let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut scaler = StandardScaler::new();
        scaler.fit(&values).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("standard_transform", n),
            &values,
            |b, values| b.iter(|| black_box(scaler.transform(black_box(values)).unwrap())),
        );
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let series = TimeSeriesDatasetBuilder::new("bench".into())
            .with_frequency("D".into())
            .with_values(values)
            .with_timestamps((0..n).map(|i| start + Duration::days(i as i64)).collect())
            .build()
            .unwrap();
        let engine = FeatureEngine::new()
            .with_lag_features(vec![1, 3])
            .with_rolling_features(vec![7]);
        group.bench_with_input(
            BenchmarkId::new("causal_lag_rolling_features", n),
            &series,
            |b, series| b.iter(|| black_box(engine.generate_features(black_box(series)).unwrap())),
        );
    }
    group.finish();
}
criterion_group!(benches, preprocessing);
criterion_main!(benches);
