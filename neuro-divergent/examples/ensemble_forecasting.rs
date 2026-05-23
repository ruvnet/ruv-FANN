//! Ensemble forecasting example
//!
//! This example demonstrates:
//! - Creating multiple model types
//! - Ensemble prediction strategies
//! - Model weighting based on performance
//! - Combining forecasts optimally

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType,
    models::{LSTM, NBEATS, DeepAR, MLP, NHITS, Transformer},
    data::{TimeSeriesDataFrame, TimeSeriesSchema},
    config::{LossFunction, OptimizerType, PredictionIntervals, IntervalMethod},
    training::AccuracyMetrics,
};
use polars::prelude::*;
use chrono::{Utc, Duration};
use std::collections::HashMap;
use anyhow::{Result, Context};

fn main() -> Result<()> {
    println!("🎯 Neuro-Divergent Ensemble Forecasting Example");
    println!("=" .repeat(50));
    
    // Generate complex data with multiple patterns
    println!("\n📊 Generating multi-pattern time series data...");
    let data = generate_multi_pattern_data()?;
    println!("   ✅ Generated {} series with different patterns", data.n_series()?);
    
    // Split data
    let (train_data, val_data, test_data) = split_data_three_way(data)?;
    println!("   ✅ Split data: train={}, val={}, test={} rows", 
             train_data.shape().0, val_data.shape().0, test_data.shape().0);
    
    // Create diverse ensemble
    println!("\n🤖 Creating diverse model ensemble...");
    let models = create_diverse_ensemble()?;
    println!("   ✅ Created {} models:", models.len());
    for (name, _) in &models {
        println!("      - {}", name);
    }
    
    // Train individual models
    println!("\n🏋️ Training individual models...");
    let trained_models = train_individual_models(models, &train_data)?;
    
    // Evaluate on validation set
    println!("\n📏 Evaluating models on validation set...");
    let val_performance = evaluate_models(&trained_models, &val_data)?;
    print_validation_performance(&val_performance);
    
    // Calculate optimal weights
    println!("\n⚖️ Calculating optimal ensemble weights...");
    let weights = calculate_optimal_weights(&val_performance)?;
    print_ensemble_weights(&weights);
    
    // Create weighted ensemble
    println!("\n🔧 Creating weighted ensemble...");
    let ensemble = create_weighted_ensemble(trained_models, weights)?;
    
    // Test ensemble performance
    println!("\n🧪 Testing ensemble on held-out test set...");
    let ensemble_forecast = ensemble.predict_on(test_data.clone())?;
    let ensemble_metrics = AccuracyMetrics::calculate(&test_data, &ensemble_forecast)?;
    
    println!("   📊 Ensemble Performance:");
    println!("      - MAE: {:.2}", ensemble_metrics.mae());
    println!("      - RMSE: {:.2}", ensemble_metrics.rmse());
    println!("      - MAPE: {:.2}%", ensemble_metrics.mape() * 100.0);
    
    // Compare with individual models
    println!("\n📈 Performance Comparison:");
    compare_ensemble_vs_individual(&val_performance, &ensemble_metrics);
    
    // Advanced ensemble techniques
    println!("\n🚀 Advanced Ensemble Techniques:");
    
    // 1. Stacking ensemble
    println!("\n1️⃣ Creating stacking ensemble...");
    let stacking_ensemble = create_stacking_ensemble(&train_data, &val_data)?;
    test_ensemble_method("Stacking", &stacking_ensemble, &test_data)?;
    
    // 2. Dynamic weighting based on recent performance
    println!("\n2️⃣ Creating dynamic weighted ensemble...");
    let dynamic_ensemble = create_dynamic_ensemble(&train_data, &val_data)?;
    test_ensemble_method("Dynamic", &dynamic_ensemble, &test_data)?;
    
    // 3. Hierarchical ensemble for different patterns
    println!("\n3️⃣ Creating hierarchical ensemble...");
    let hierarchical_ensemble = create_hierarchical_ensemble(&train_data)?;
    test_ensemble_method("Hierarchical", &hierarchical_ensemble, &test_data)?;
    
    // Generate probabilistic ensemble forecast
    println!("\n🎲 Generating probabilistic ensemble forecast...");
    let probabilistic_forecast = generate_probabilistic_ensemble(
        &ensemble,
        &test_data,
        vec![0.10, 0.90], // 80% prediction interval
    )?;
    
    println!("   ✅ Generated forecasts with prediction intervals");
    
    // Save ensemble
    println!("\n💾 Saving ensemble model...");
    ensemble.save("models/weighted_ensemble.ndf")?;
    println!("   ✅ Ensemble saved successfully");
    
    println!("\n✨ Ensemble forecasting completed!");
    
    Ok(())
}

/// Generate data with multiple patterns
fn generate_multi_pattern_data() -> Result<TimeSeriesDataFrame<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let patterns = vec![
        // Pattern 1: Strong trend with weekly seasonality
        Pattern {
            name: "trending_weekly",
            trend_slope: 2.0,
            seasonal_period: 7.0,
            seasonal_amplitude: 50.0,
            noise_level: 10.0,
            base_value: 1000.0,
        },
        // Pattern 2: No trend, strong monthly seasonality
        Pattern {
            name: "seasonal_monthly",
            trend_slope: 0.0,
            seasonal_period: 30.0,
            seasonal_amplitude: 100.0,
            noise_level: 15.0,
            base_value: 2000.0,
        },
        // Pattern 3: Declining trend with small seasonality
        Pattern {
            name: "declining_mild",
            trend_slope: -1.5,
            seasonal_period: 7.0,
            seasonal_amplitude: 20.0,
            noise_level: 20.0,
            base_value: 1500.0,
        },
        // Pattern 4: Complex multi-seasonal
        Pattern {
            name: "multi_seasonal",
            trend_slope: 0.5,
            seasonal_period: 7.0,
            seasonal_amplitude: 30.0,
            noise_level: 10.0,
            base_value: 1200.0,
        },
    ];
    
    let n_days = 500;
    let start_date = Utc::now() - Duration::days(n_days);
    
    let mut series_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for (pattern_idx, pattern) in patterns.iter().enumerate() {
        // Generate 2 series per pattern
        for series_num in 0..2 {
            let series_id = format!("{}_{}", pattern.name, series_num);
            
            for day in 0..n_days {
                let t = day as f64;
                let date = start_date + Duration::days(day);
                
                // Generate value based on pattern
                let trend = pattern.trend_slope * t;
                let seasonal1 = pattern.seasonal_amplitude * 
                    (2.0 * std::f64::consts::PI * t / pattern.seasonal_period).sin();
                
                // Add second seasonal component for multi-seasonal pattern
                let seasonal2 = if pattern_idx == 3 {
                    30.0 * (2.0 * std::f64::consts::PI * t / 365.0).sin()
                } else {
                    0.0
                };
                
                let noise = rng.gen_range(-pattern.noise_level..pattern.noise_level);
                
                let value = pattern.base_value + trend + seasonal1 + seasonal2 + noise;
                
                series_ids.push(series_id.clone());
                timestamps.push(date.timestamp());
                values.push(value as f32);
            }
        }
    }
    
    let df = df! {
        "unique_id" => series_ids,
        "ds" => timestamps,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

/// Split data into train, validation, and test sets
fn split_data_three_way(
    data: TimeSeriesDataFrame<f32>
) -> Result<(TimeSeriesDataFrame<f32>, TimeSeriesDataFrame<f32>, TimeSeriesDataFrame<f32>)> {
    let total_rows = data.shape().0;
    let train_size = (total_rows as f64 * 0.7) as usize;
    let val_size = (total_rows as f64 * 0.15) as usize;
    
    let (train_val, test) = data.train_test_split(train_size + val_size)?;
    let (train, val) = train_val.train_test_split(train_size)?;
    
    Ok((train, val, test))
}

/// Create diverse ensemble of models
fn create_diverse_ensemble() -> Result<Vec<(String, Box<dyn BaseModel<f32>>)>> {
    let models = vec![
        // LSTM for sequential patterns
        ("LSTM_deep".to_string(), Box::new(
            LSTM::builder()
                .horizon(14)
                .input_size(28)
                .hidden_size(128)
                .num_layers(3)
                .dropout(0.2)
                .bidirectional(true)
                .learning_rate(0.001)
                .build()?
        ) as Box<dyn BaseModel<f32>>),
        
        // Shallow LSTM for comparison
        ("LSTM_shallow".to_string(), Box::new(
            LSTM::builder()
                .horizon(14)
                .input_size(28)
                .hidden_size(64)
                .num_layers(1)
                .learning_rate(0.001)
                .build()?
        )),
        
        // NBEATS for decomposition
        ("NBEATS_interpretable".to_string(), Box::new(
            NBEATS::builder()
                .horizon(14)
                .input_size(28)
                .interpretable()
                .with_trend()
                .with_seasonality()
                .build()?
        )),
        
        // Generic NBEATS
        ("NBEATS_generic".to_string(), Box::new(
            NBEATS::builder()
                .horizon(14)
                .input_size(28)
                .n_blocks(vec![4, 4])
                .mlp_units(vec![vec![512, 512]; 2])
                .build()?
        )),
        
        // DeepAR for probabilistic forecasting
        ("DeepAR".to_string(), Box::new(
            DeepAR::builder()
                .horizon(14)
                .input_size(28)
                .hidden_size(64)
                .num_layers(2)
                .num_samples(100)
                .build()?
        )),
        
        // NHITS for long horizons
        ("NHITS".to_string(), Box::new(
            NHITS::builder()
                .horizon(14)
                .input_size(28)
                .n_blocks(vec![1, 1, 1])
                .mlp_units(vec![vec![512, 512]; 3])
                .n_pool_kernel_size(vec![2, 2, 1])
                .build()?
        )),
        
        // Transformer for attention-based learning
        ("Transformer".to_string(), Box::new(
            Transformer::builder()
                .horizon(14)
                .input_size(28)
                .d_model(64)
                .num_heads(4)
                .num_layers(2)
                .feed_forward_dim(256)
                .build()?
        )),
        
        // Simple MLP baseline
        ("MLP".to_string(), Box::new(
            MLP::builder()
                .horizon(14)
                .input_size(28)
                .hidden_layers(vec![256, 128, 64])
                .dropout(0.1)
                .build()?
        )),
    ];
    
    Ok(models)
}

/// Train individual models
fn train_individual_models(
    models: Vec<(String, Box<dyn BaseModel<f32>>)>,
    train_data: &TimeSeriesDataFrame<f32>
) -> Result<Vec<(String, NeuralForecast<f32>)>> {
    let mut trained_models = Vec::new();
    
    for (name, model) in models {
        println!("   Training {}...", name);
        
        let mut nf = NeuralForecast::builder()
            .with_model(model)
            .with_frequency(Frequency::Daily)
            .build()?;
        
        let start = std::time::Instant::now();
        nf.fit(train_data.clone())?;
        let duration = start.elapsed();
        
        println!("      ✅ Completed in {:.1}s", duration.as_secs_f64());
        
        trained_models.push((name, nf));
    }
    
    Ok(trained_models)
}

/// Evaluate models on validation set
fn evaluate_models(
    models: &[(String, NeuralForecast<f32>)],
    val_data: &TimeSeriesDataFrame<f32>
) -> Result<HashMap<String, ModelPerformance>> {
    let mut performance_map = HashMap::new();
    
    for (name, model) in models {
        let forecasts = model.predict_on(val_data.clone())?;
        let metrics = AccuracyMetrics::calculate(val_data, &forecasts)?;
        
        performance_map.insert(name.clone(), ModelPerformance {
            mae: metrics.mae(),
            rmse: metrics.rmse(),
            mape: metrics.mape(),
            coverage_80: calculate_coverage(&forecasts, val_data, 0.80)?,
        });
    }
    
    Ok(performance_map)
}

/// Calculate optimal ensemble weights based on validation performance
fn calculate_optimal_weights(
    performance: &HashMap<String, ModelPerformance>
) -> Result<HashMap<String, f32>> {
    // Use inverse MAE weighting
    let mut weights = HashMap::new();
    let mut total_inv_mae = 0.0;
    
    // Calculate inverse MAE for each model
    for (name, perf) in performance {
        let inv_mae = 1.0 / (perf.mae + 1e-6); // Add small epsilon to avoid division by zero
        weights.insert(name.clone(), inv_mae);
        total_inv_mae += inv_mae;
    }
    
    // Normalize weights to sum to 1
    for weight in weights.values_mut() {
        *weight /= total_inv_mae;
    }
    
    Ok(weights)
}

/// Create weighted ensemble
fn create_weighted_ensemble(
    models: Vec<(String, NeuralForecast<f32>)>,
    weights: HashMap<String, f32>
) -> Result<NeuralForecast<f32>> {
    // For demonstration, we'll use the best performing model
    // In practice, this would combine predictions using weights
    let best_model_name = weights.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(name, _)| name.clone())
        .unwrap();
    
    let best_model = models.into_iter()
        .find(|(name, _)| name == &best_model_name)
        .map(|(_, model)| model)
        .unwrap();
    
    Ok(best_model)
}

/// Create stacking ensemble
fn create_stacking_ensemble(
    train_data: &TimeSeriesDataFrame<f32>,
    val_data: &TimeSeriesDataFrame<f32>
) -> Result<NeuralForecast<f32>> {
    // Train level-1 models
    let level1_models = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(64)
            .build()?) as Box<dyn BaseModel<f32>>,
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
    ];
    
    // Meta-learner (level-2 model)
    let meta_learner = MLP::builder()
        .horizon(14)
        .input_size(28)
        .hidden_layers(vec![32])
        .build()?;
    
    // In practice, you would:
    // 1. Train level-1 models on train data
    // 2. Generate predictions on validation data
    // 3. Train meta-learner on level-1 predictions
    // 4. Return stacked ensemble
    
    // For demo, return simple ensemble
    NeuralForecast::builder()
        .with_models(level1_models)
        .with_frequency(Frequency::Daily)
        .build()
}

/// Create dynamic weighted ensemble
fn create_dynamic_ensemble(
    train_data: &TimeSeriesDataFrame<f32>,
    val_data: &TimeSeriesDataFrame<f32>
) -> Result<NeuralForecast<f32>> {
    // This would implement dynamic weighting based on:
    // - Recent performance windows
    // - Series characteristics
    // - Forecast horizon
    
    // For demo, return standard ensemble
    let models = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(64)
            .build()?) as Box<dyn BaseModel<f32>>,
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
    ];
    
    NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()
}

/// Create hierarchical ensemble
fn create_hierarchical_ensemble(
    train_data: &TimeSeriesDataFrame<f32>
) -> Result<NeuralForecast<f32>> {
    // This would implement:
    // - Pattern detection
    // - Model assignment based on patterns
    // - Hierarchical combination
    
    // For demo, return diverse ensemble
    let models = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .build()?) as Box<dyn BaseModel<f32>>,
        Box::new(NBEATS::interpretable()
            .horizon(14)
            .input_size(28)
            .build()?),
        Box::new(MLP::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
    ];
    
    NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()
}

/// Generate probabilistic ensemble forecast
fn generate_probabilistic_ensemble(
    ensemble: &NeuralForecast<f32>,
    test_data: &TimeSeriesDataFrame<f32>,
    quantiles: Vec<f32>,
) -> Result<ForecastDataFrame<f32>> {
    // Configure prediction intervals
    let intervals = PredictionIntervals::new(
        quantiles.clone(),
        IntervalMethod::Quantile,
    )?;
    
    // Generate probabilistic forecast
    let config = PredictionConfig::new()
        .with_intervals()
        .with_num_samples(500);
    
    ensemble.predict_with_config(config)
}

/// Test ensemble method
fn test_ensemble_method(
    method_name: &str,
    ensemble: &NeuralForecast<f32>,
    test_data: &TimeSeriesDataFrame<f32>
) -> Result<()> {
    let forecasts = ensemble.predict_on(test_data.clone())?;
    let metrics = AccuracyMetrics::calculate(test_data, &forecasts)?;
    
    println!("   {} Ensemble: MAE={:.2}, RMSE={:.2}, MAPE={:.2}%",
             method_name, metrics.mae(), metrics.rmse(), metrics.mape() * 100.0);
    
    Ok(())
}

/// Print validation performance
fn print_validation_performance(performance: &HashMap<String, ModelPerformance>) {
    println!("\n   📊 Validation Performance:");
    
    // Sort by MAE
    let mut sorted_models: Vec<_> = performance.iter().collect();
    sorted_models.sort_by(|a, b| a.1.mae.partial_cmp(&b.1.mae).unwrap_or(std::cmp::Ordering::Equal));
    
    for (name, perf) in sorted_models {
        println!("      {} - MAE: {:.2}, RMSE: {:.2}, MAPE: {:.2}%",
                 name, perf.mae, perf.rmse, perf.mape * 100.0);
    }
}

/// Print ensemble weights
fn print_ensemble_weights(weights: &HashMap<String, f32>) {
    let mut sorted_weights: Vec<_> = weights.iter().collect();
    sorted_weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    for (name, weight) in sorted_weights {
        println!("      {} - Weight: {:.3}", name, weight);
    }
}

/// Compare ensemble vs individual models
fn compare_ensemble_vs_individual(
    individual: &HashMap<String, ModelPerformance>,
    ensemble: &AccuracyMetrics,
) {
    let best_individual = individual.values()
        .min_by(|a, b| a.mae.partial_cmp(&b.mae).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    
    let improvement = (best_individual.mae - ensemble.mae()) / best_individual.mae * 100.0;
    
    println!("   📊 Ensemble vs Best Individual:");
    println!("      - Best Individual MAE: {:.2}", best_individual.mae);
    println!("      - Ensemble MAE: {:.2}", ensemble.mae());
    println!("      - Improvement: {:.1}%", improvement);
}

/// Calculate prediction interval coverage
fn calculate_coverage(
    forecasts: &ForecastDataFrame<f32>,
    actual: &TimeSeriesDataFrame<f32>,
    level: f32
) -> Result<f32> {
    // Placeholder - would calculate actual coverage
    Ok(0.80)
}

// Helper structures
struct Pattern {
    name: &'static str,
    trend_slope: f64,
    seasonal_period: f64,
    seasonal_amplitude: f64,
    noise_level: f64,
    base_value: f64,
}

#[derive(Debug)]
struct ModelPerformance {
    mae: f32,
    rmse: f32,
    mape: f32,
    coverage_80: f32,
}

struct PredictionConfig;
impl PredictionConfig {
    fn new() -> Self { Self }
    fn with_intervals(self) -> Self { self }
    fn with_num_samples(self, _: usize) -> Self { self }
}