//! Benchmarks for the ML training pipeline

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use ruv_swarm_ml_training::{
    EventType, LSTMModel, NBEATSModel, NeuroDivergentModel, PerformanceMetrics, PromptData,
    StackType, StreamDataLoader, StreamEvent, TCNModel, TrainingConfig,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn generate_bench_events(count: usize) -> Vec<StreamEvent> {
    let base_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    (0..count)
        .map(|i| StreamEvent {
            timestamp: base_time + (i as u64 * 60),
            agent_id: format!("agent_{}", i % 10),
            event_type: EventType::TaskCompleted,
            performance_metrics: PerformanceMetrics {
                latency_ms: 50.0 + (i as f64 % 20.0),
                tokens_per_second: 100.0 - (i as f64 % 10.0),
                memory_usage_mb: 256.0,
                cpu_usage_percent: 40.0,
                success_rate: 0.95,
            },
            prompt_data: if i % 2 == 0 {
                Some(PromptData {
                    prompt_text: format!("Prompt {}", i),
                    prompt_tokens: 50,
                    response_tokens: 100,
                    quality_score: 0.85,
                })
            } else {
                None
            },
        })
        .collect()
}

fn benchmark_data_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_loading");

    for size in [1000, 5000, 10000].iter() {
        group.bench_function(format!("load_{}_events", size), |b| {
            b.iter_batched(
                || generate_bench_events(*size),
                |events| {
                    let runtime = tokio::runtime::Runtime::new().unwrap();
                    runtime.block_on(async {
                        let loader = StreamDataLoader::new(1000, 50);
                        loader.load_from_stream(events.into_iter()).await.unwrap()
                    })
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn benchmark_model_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_prediction");

    // Setup test data
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let events = generate_bench_events(1000);
    let dataset = runtime.block_on(async {
        let loader = StreamDataLoader::new(1000, 50);
        loader.load_from_stream(events.into_iter()).await.unwrap()
    });

    let test_sequence = &dataset.sequences[0];

    // Benchmark LSTM
    group.bench_function("lstm_predict", |b| {
        let model = LSTMModel::new(128, 2);
        b.iter(|| model.predict(test_sequence).unwrap())
    });

    // Benchmark TCN
    group.bench_function("tcn_predict", |b| {
        let model = TCNModel::new(vec![64, 64, 64], 3);
        b.iter(|| model.predict(test_sequence).unwrap())
    });

    // Benchmark N-BEATS
    group.bench_function("nbeats_predict", |b| {
        let model = NBEATSModel::new(vec![StackType::Trend, StackType::Seasonality], 4);
        b.iter(|| model.predict(test_sequence).unwrap())
    });

    group.finish();
}

fn benchmark_feature_extraction(c: &mut Criterion) {
    use ruv_swarm_ml_training::FeatureExtractor;

    struct BenchFeatureExtractor;
    impl FeatureExtractor for BenchFeatureExtractor {
        fn extract(&self, event: &StreamEvent) -> Vec<f64> {
            vec![
                event.performance_metrics.latency_ms,
                event.performance_metrics.tokens_per_second,
                event.performance_metrics.memory_usage_mb,
                event.performance_metrics.cpu_usage_percent,
                event.performance_metrics.success_rate,
            ]
        }

        fn feature_names(&self) -> Vec<String> {
            vec![
                "latency".to_string(),
                "tokens_per_sec".to_string(),
                "memory".to_string(),
                "cpu".to_string(),
                "success_rate".to_string(),
            ]
        }
    }

    let event = StreamEvent {
        timestamp: 1000,
        agent_id: "agent_1".to_string(),
        event_type: EventType::TaskCompleted,
        performance_metrics: PerformanceMetrics {
            latency_ms: 50.0,
            tokens_per_second: 100.0,
            memory_usage_mb: 256.0,
            cpu_usage_percent: 40.0,
            success_rate: 0.95,
        },
        prompt_data: Some(PromptData {
            prompt_text: "Test".to_string(),
            prompt_tokens: 50,
            response_tokens: 100,
            quality_score: 0.85,
        }),
    };

    c.bench_function("feature_extraction", |b| {
        let extractor = BenchFeatureExtractor;
        b.iter(|| extractor.extract(&event))
    });
}

fn benchmark_training_epoch(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_epoch");
    group.sample_size(10); // Reduce sample size for longer benchmarks

    // Setup small dataset
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let events = generate_bench_events(500);
    let dataset = runtime.block_on(async {
        let loader = StreamDataLoader::new(100, 20);
        loader.load_from_stream(events.into_iter()).await.unwrap()
    });

    let config = TrainingConfig {
        epochs: 1,
        batch_size: 32,
        validation_split: 0.2,
        early_stopping_patience: 5,
        save_checkpoints: false,
        checkpoint_dir: "/tmp".to_string(),
    };

    // Benchmark single epoch for each model
    group.bench_function("lstm_epoch", |b| {
        b.iter(|| {
            let mut model = LSTMModel::new(64, 1);
            model.train(&dataset, &config).unwrap()
        })
    });

    group.bench_function("tcn_epoch", |b| {
        b.iter(|| {
            let mut model = TCNModel::new(vec![32, 32], 3);
            model.train(&dataset, &config).unwrap()
        })
    });

    group.bench_function("nbeats_epoch", |b| {
        b.iter(|| {
            let mut model = NBEATSModel::new(vec![StackType::Generic], 2);
            model.train(&dataset, &config).unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_data_loading,
    benchmark_model_prediction,
    benchmark_feature_extraction,
    benchmark_training_epoch
);
criterion_main!(benches);
