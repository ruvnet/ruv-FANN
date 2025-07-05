//! Performance benchmarks for Cell Sleep Mode Forecaster

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use std::sync::Arc;
use chrono::{Utc, Duration};

use cell_sleep_forecaster::{
    CellSleepForecaster, PrbUtilization,
    config::ForecastingConfig,
    forecasting::ForecastingModel,
    optimization::SleepOptimizer,
    metrics::MetricsCalculator,
};

/// Benchmark forecasting performance with different data sizes
fn bench_forecasting_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("forecasting_performance");
    
    // Test different historical data sizes
    for data_size in [144, 288, 576, 1440].iter() { // 1 day, 2 days, 4 days, 10 days
        group.bench_with_input(
            BenchmarkId::new("forecast_generation", data_size),
            data_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = Arc::new(ForecastingConfig::default());
                    let forecaster = CellSleepForecaster::new(config).await.unwrap();
                    let test_data = generate_benchmark_data("bench_cell", size);
                    
                    forecaster.forecast_prb_utilization("bench_cell", &test_data).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark sleep window optimization
fn bench_sleep_optimization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("sleep_optimization");
    
    // Test different forecast sizes
    for forecast_size in [6, 12, 24, 48].iter() { // 1h, 2h, 4h, 8h forecasts
        group.bench_with_input(
            BenchmarkId::new("sleep_window_detection", forecast_size),
            forecast_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = Arc::new(ForecastingConfig::default());
                    let optimizer = SleepOptimizer::new(config);
                    let forecast_data = generate_forecast_data("bench_cell", size);
                    
                    optimizer.identify_sleep_windows("bench_cell", &forecast_data).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark metrics calculation
fn bench_metrics_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_calculation");
    
    // Test different data sizes for MAPE calculation
    for data_size in [60, 144, 288, 576].iter() {
        let actual_data = generate_benchmark_data("bench_cell", *data_size);
        let predicted_data = generate_forecast_data("bench_cell", *data_size);
        
        group.bench_with_input(
            BenchmarkId::new("forecast_evaluation", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    MetricsCalculator::evaluate_forecast(&actual_data, &predicted_data, 20.0).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark model training performance
fn bench_model_training(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("model_training");
    
    // Test training with different data sizes
    for data_size in [144, 288, 576, 1440].iter() {
        group.bench_with_input(
            BenchmarkId::new("model_training", data_size),
            data_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = Arc::new(ForecastingConfig::default());
                    let mut model = ForecastingModel::new(config, "bench_cell").await.unwrap();
                    let training_data = generate_benchmark_data("bench_cell", size);
                    
                    model.train(&training_data).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark energy savings calculation
fn bench_energy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_calculation");
    
    // Test with different numbers of sleep windows
    for window_count in [1, 5, 10, 20, 50].iter() {
        let sleep_windows = generate_sleep_windows(*window_count);
        
        group.bench_with_input(
            BenchmarkId::new("energy_savings_analysis", window_count),
            window_count,
            |b, _| {
                b.iter(|| {
                    MetricsCalculator::analyze_energy_savings(&sleep_windows, 0.12, 0.5).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent forecasting (multiple cells)
fn bench_concurrent_forecasting(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_forecasting");
    
    // Test concurrent processing of multiple cells
    for cell_count in [1, 5, 10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_cells", cell_count),
            cell_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let config = Arc::new(ForecastingConfig::default());
                    let forecaster = Arc::new(CellSleepForecaster::new(config).await.unwrap());
                    
                    let mut tasks = Vec::new();
                    
                    for i in 0..count {
                        let cell_id = format!("bench_cell_{}", i);
                        let test_data = generate_benchmark_data(&cell_id, 144);
                        let forecaster_clone = forecaster.clone();
                        
                        let task = tokio::spawn(async move {
                            forecaster_clone.forecast_prb_utilization(&cell_id, &test_data).await.unwrap()
                        });
                        
                        tasks.push(task);
                    }
                    
                    // Wait for all tasks to complete
                    for task in tasks {
                        task.await.unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark throughput (requests per second)
fn bench_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("throughput_1000_requests", |b| {
        b.to_async(&rt).iter(|| async {
            let config = Arc::new(ForecastingConfig::default());
            let forecaster = CellSleepForecaster::new(config).await.unwrap();
            let test_data = generate_benchmark_data("throughput_cell", 144);
            
            // Process 1000 requests
            for _ in 0..1000 {
                let _ = forecaster.forecast_prb_utilization("throughput_cell", &test_data).await.unwrap();
            }
        });
    });
}

/// Generate test data for benchmarking
fn generate_benchmark_data(cell_id: &str, points: usize) -> Vec<PrbUtilization> {
    let mut data = Vec::new();
    let start_time = Utc::now() - Duration::minutes((points * 10) as i64);
    
    for i in 0..points {
        let timestamp = start_time + Duration::minutes((i * 10) as i64);
        
        // Generate deterministic but realistic pattern for consistent benchmarking
        let hour = (i % 144) as f64 / 6.0; // 24-hour cycle
        let base_utilization = 30.0 + 40.0 * (hour * std::f64::consts::PI / 12.0).sin();
        let noise = ((i * 7) % 100) as f64 / 100.0 * 10.0 - 5.0; // Deterministic noise
        let utilization = (base_utilization + noise).max(0.0).min(100.0);
        
        let prb_used = (utilization * 100.0 / 100.0) as u32;
        let throughput = utilization * 2.0;
        let users = (utilization * 0.5) as u32;
        let signal_quality = 0.8 + ((i * 3) % 100) as f64 / 100.0 * 0.2;
        
        let mut prb = PrbUtilization::new(
            cell_id.to_string(),
            100,
            prb_used,
            throughput,
            users,
            signal_quality,
        );
        prb.timestamp = timestamp;
        
        data.push(prb);
    }
    
    data
}

/// Generate forecast data for benchmarking
fn generate_forecast_data(cell_id: &str, points: usize) -> Vec<PrbUtilization> {
    let mut data = Vec::new();
    let start_time = Utc::now();
    
    for i in 0..points {
        let timestamp = start_time + Duration::minutes((i * 10) as i64);
        
        // Generate forecast pattern (slightly different from historical for realistic error)
        let hour = (i % 144) as f64 / 6.0;
        let base_utilization = 28.0 + 38.0 * (hour * std::f64::consts::PI / 12.0).sin(); // Slightly different
        let noise = ((i * 5) % 100) as f64 / 100.0 * 8.0 - 4.0; // Different noise pattern
        let utilization = (base_utilization + noise).max(0.0).min(100.0);
        
        let prb_used = (utilization * 100.0 / 100.0) as u32;
        let throughput = utilization * 2.1; // Slightly different relationship
        let users = (utilization * 0.55) as u32;
        let signal_quality = 0.82 + ((i * 2) % 100) as f64 / 100.0 * 0.18;
        
        let mut prb = PrbUtilization::new(
            cell_id.to_string(),
            100,
            prb_used,
            throughput,
            users,
            signal_quality,
        );
        prb.timestamp = timestamp;
        
        data.push(prb);
    }
    
    data
}

/// Generate sleep windows for benchmarking
fn generate_sleep_windows(count: usize) -> Vec<cell_sleep_forecaster::SleepWindow> {
    let mut windows = Vec::new();
    let start_time = Utc::now();
    
    for i in 0..count {
        let window_start = start_time + Duration::hours(i as i64 * 2);
        let duration = 60 + (i % 3) * 30; // 60, 90, or 120 minutes
        let window_end = window_start + Duration::minutes(duration as i64);
        
        windows.push(cell_sleep_forecaster::SleepWindow {
            cell_id: format!("bench_cell_{}", i),
            start_time: window_start,
            end_time: window_end,
            duration_minutes: duration as u32,
            confidence_score: 0.8 + (i % 20) as f64 / 100.0,
            predicted_utilization: 5.0 + (i % 15) as f64,
            energy_savings_kwh: 1.5 + (i % 10) as f64 * 0.5,
            risk_score: 0.1 + (i % 10) as f64 / 100.0,
        });
    }
    
    windows
}

criterion_group!(
    benches,
    bench_forecasting_performance,
    bench_sleep_optimization,
    bench_metrics_calculation,
    bench_model_training,
    bench_energy_calculation,
    bench_concurrent_forecasting,
    bench_throughput
);

criterion_main!(benches);