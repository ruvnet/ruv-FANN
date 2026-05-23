//! Integration tests for cross-validation workflows
//!
//! Tests cross-validation scenarios including:
//! - Time series cross-validation
//! - Model selection via CV
//! - Hyperparameter tuning
//! - CV with different window strategies

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType,
    models::{LSTM, NBEATS, MLP, RNN},
    data::{TimeSeriesDataFrame, TimeSeriesSchema},
    config::{CrossValidationConfig, LossFunction, OptimizerType},
    training::AccuracyMetrics,
};
use std::collections::HashMap;
use polars::prelude::*;
use chrono::{DateTime, Utc, Duration};
use num_traits::Float;
use rand::Rng;

/// Generate synthetic data for CV testing
fn generate_cv_test_data<T: Float>(
    n_series: usize,
    n_points: usize,
    trend: bool,
    seasonal_period: Option<usize>,
) -> Result<TimeSeriesDataFrame<T>, Box<dyn std::error::Error>>
where
    T: From<f64> + Into<f64>,
{
    let mut rng = rand::thread_rng();
    let start_date = Utc::now() - Duration::days(n_points as i64);
    
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for series_id in 0..n_series {
        let series_name = format!("series_{}", series_id);
        let base_value = rng.gen_range(50.0..150.0);
        let trend_slope = if trend { rng.gen_range(-0.1..0.3) } else { 0.0 };
        
        for i in 0..n_points {
            let date = start_date + Duration::days(i as i64);
            unique_ids.push(series_name.clone());
            timestamps.push(date.timestamp());
            
            // Generate value
            let mut value = base_value + trend_slope * i as f64;
            
            // Add seasonality
            if let Some(period) = seasonal_period {
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                value += seasonal;
            }
            
            // Add noise
            value += rng.gen_range(-5.0..5.0);
            
            values.push(value);
        }
    }
    
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

/// Test basic cross-validation setup
#[test]
fn test_basic_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        3,    // 3 series
        365,  // 1 year of daily data
        true, // with trend
        Some(7), // weekly seasonality
    )?;
    
    // Create model
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .max_steps(10) // Small for testing
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Configure cross-validation
    let cv_config = CrossValidationConfig::new(5, 7) // 5 windows, 7-day horizon
        .with_step_size(7) // Non-overlapping windows
        .with_refit(true); // Refit on each window
    
    // Run cross-validation
    let cv_results = nf.cross_validation(data, cv_config)?;
    
    // Validate results
    assert_eq!(cv_results.n_windows(), 5);
    assert_eq!(cv_results.model_names(), vec!["LSTM"]);
    
    // Calculate CV metrics
    let metrics = cv_results.calculate_metrics()?;
    let lstm_metrics = &metrics["LSTM"];
    
    println!("Cross-validation metrics:");
    println!("MAE: {:.3}", lstm_metrics.mae());
    println!("RMSE: {:.3}", lstm_metrics.rmse());
    println!("MAPE: {:.3}%", lstm_metrics.mape() * 100.0);
    
    Ok(())
}

/// Test expanding window cross-validation
#[test]
fn test_expanding_window_cv() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        2,
        200,
        true,
        Some(30), // Monthly seasonality
    )?;
    
    // Create models
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(32)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Expanding window CV - each window uses all previous data
    let initial_train_size = 100;
    let n_windows = 5;
    let step_size = 14;
    
    let mut cv_results = Vec::new();
    
    for window in 0..n_windows {
        let train_end = initial_train_size + window * step_size;
        let test_end = train_end + 14; // 14-day test window
        
        if test_end > data.shape().0 {
            break;
        }
        
        // Split data
        let train_data = data.slice(0, train_end)?;
        let test_data = data.slice(train_end, test_end)?;
        
        // Fit models on expanding training set
        nf.fit(train_data)?;
        
        // Predict on test window
        let forecasts = nf.predict_on(test_data.clone())?;
        
        // Calculate metrics
        let window_metrics = AccuracyMetrics::calculate(&test_data, &forecasts)?;
        cv_results.push((window, train_end, window_metrics));
        
        println!("Window {}: train_size={}, MAE={:.3}", 
                 window, train_end, window_metrics.mae());
    }
    
    // Analyze if performance improves with more data
    let maes: Vec<f32> = cv_results.iter().map(|(_, _, m)| m.mae()).collect();
    let first_mae = maes[0];
    let last_mae = maes[maes.len() - 1];
    
    println!("MAE improvement: {:.1}%", 
             (first_mae - last_mae) / first_mae * 100.0);
    
    Ok(())
}

/// Test cross-validation with seasonal splits
#[test]
fn test_seasonal_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with strong seasonality
    let data = generate_cv_test_data::<f32>(
        5,
        730, // 2 years
        false, // no trend
        Some(365), // yearly seasonality
    )?;
    
    // Create seasonal-aware model
    let nbeats = NBEATS::builder()
        .horizon(30)
        .input_size(60)
        .interpretable()
        .with_seasonality()
        .seasonality_period(365)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(nbeats))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // CV with seasonal awareness
    let cv_config = CrossValidationConfig::new(4, 30)
        .with_season_length(365)
        .with_step_size(90) // Quarterly steps
        .with_refit(true);
    
    let cv_results = nf.cross_validation(data, cv_config)?;
    
    // Analyze seasonal performance
    let metrics_by_season = cv_results.calculate_seasonal_metrics()?;
    
    println!("Seasonal CV performance:");
    for (season, metrics) in metrics_by_season {
        println!("Season {}: MAE={:.3}", season, metrics.mae());
    }
    
    Ok(())
}

/// Test model selection via cross-validation
#[test]
fn test_model_selection_cv() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        4,
        300,
        true,
        Some(7),
    )?;
    
    // Define candidate models with different configurations
    let candidate_models = vec![
        ("LSTM_small", Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(16)
            .num_layers(1)
            .build()?) as Box<dyn BaseModel<f32>>),
        ("LSTM_medium", Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .num_layers(2)
            .build()?)),
        ("LSTM_large", Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(64)
            .num_layers(3)
            .build()?)),
        ("NBEATS_generic", Box::new(NBEATS::builder()
            .horizon(7)
            .input_size(14)
            .build()?)),
        ("NBEATS_interpretable", Box::new(NBEATS::builder()
            .horizon(7)
            .input_size(14)
            .interpretable()
            .build()?)),
        ("MLP_shallow", Box::new(MLP::builder()
            .horizon(7)
            .input_size(14)
            .hidden_layers(vec![32])
            .build()?)),
        ("MLP_deep", Box::new(MLP::builder()
            .horizon(7)
            .input_size(14)
            .hidden_layers(vec![64, 32, 16])
            .build()?)),
    ];
    
    // CV configuration for model selection
    let cv_config = CrossValidationConfig::new(3, 7)
        .with_step_size(7)
        .with_refit(true);
    
    // Evaluate each model
    let mut model_scores = HashMap::new();
    
    for (name, model) in candidate_models {
        let mut nf = NeuralForecast::builder()
            .with_model(model)
            .with_frequency(Frequency::Daily)
            .build()?;
        
        let cv_results = nf.cross_validation(data.clone(), cv_config.clone())?;
        let metrics = cv_results.calculate_metrics()?;
        
        // Use average MAE as selection criterion
        let avg_mae = metrics.values()
            .next()
            .map(|m| m.mae())
            .unwrap_or(f32::INFINITY);
        
        model_scores.insert(name.to_string(), avg_mae);
        println!("{}: CV MAE = {:.3}", name, avg_mae);
    }
    
    // Select best model
    let best_model = model_scores.iter()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(name, _)| name.clone())
        .unwrap();
    
    println!("\nBest model: {}", best_model);
    
    Ok(())
}

/// Test hyperparameter tuning via cross-validation
#[test]
fn test_hyperparameter_tuning_cv() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        3,
        250,
        true,
        Some(7),
    )?;
    
    // Define hyperparameter grid for LSTM
    let hidden_sizes = vec![16, 32, 64];
    let num_layers = vec![1, 2];
    let dropout_rates = vec![0.0, 0.1, 0.2];
    let learning_rates = vec![0.0001, 0.001, 0.01];
    
    // CV configuration
    let cv_config = CrossValidationConfig::new(3, 7)
        .with_step_size(7)
        .with_refit(false); // Don't refit to speed up
    
    let mut best_params = HashMap::new();
    let mut best_score = f32::INFINITY;
    
    // Grid search
    for &hidden_size in &hidden_sizes {
        for &n_layers in &num_layers {
            for &dropout in &dropout_rates {
                for &lr in &learning_rates {
                    // Create model with current hyperparameters
                    let lstm = LSTM::builder()
                        .horizon(7)
                        .input_size(14)
                        .hidden_size(hidden_size)
                        .num_layers(n_layers)
                        .dropout(dropout)
                        .learning_rate(lr)
                        .max_steps(20)
                        .build()?;
                    
                    let mut nf = NeuralForecast::builder()
                        .with_model(Box::new(lstm))
                        .with_frequency(Frequency::Daily)
                        .build()?;
                    
                    // Run CV
                    let cv_results = nf.cross_validation(data.clone(), cv_config.clone())?;
                    let metrics = cv_results.calculate_metrics()?;
                    let score = metrics["LSTM"].mae();
                    
                    // Update best parameters
                    if score < best_score {
                        best_score = score;
                        best_params.clear();
                        best_params.insert("hidden_size", hidden_size as f32);
                        best_params.insert("num_layers", n_layers as f32);
                        best_params.insert("dropout", dropout);
                        best_params.insert("learning_rate", lr);
                    }
                }
            }
        }
    }
    
    println!("Best hyperparameters found:");
    for (param, value) in &best_params {
        println!("{}: {}", param, value);
    }
    println!("Best CV score: {:.3}", best_score);
    
    Ok(())
}

/// Test cross-validation with different loss functions
#[test]
fn test_cv_loss_function_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with outliers
    let mut data = generate_cv_test_data::<f32>(
        2,
        200,
        true,
        Some(7),
    )?;
    
    // Add some outliers
    let mut df = data.to_polars()?;
    let mut y_values: Vec<f64> = df.column("y")?.f64()?.to_vec()
        .into_iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();
    
    let mut rng = rand::thread_rng();
    for i in 0..y_values.len() {
        if rng.gen_bool(0.05) { // 5% outliers
            y_values[i] *= rng.gen_range(2.0..4.0);
        }
    }
    
    df = df.with_column(Series::new("y", y_values))?;
    data = TimeSeriesDataFrame::from_polars(df, data.schema.clone(), data.frequency)?;
    
    // Test different loss functions
    let loss_functions = vec![
        LossFunction::MSE,
        LossFunction::MAE,
        LossFunction::Huber,
        LossFunction::MAPE,
    ];
    
    let cv_config = CrossValidationConfig::new(3, 7)
        .with_step_size(7)
        .with_refit(true);
    
    let mut loss_performance = HashMap::new();
    
    for loss_fn in loss_functions {
        let lstm = LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .loss_function(loss_fn)
            .max_steps(30)
            .build()?;
        
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(lstm))
            .with_frequency(Frequency::Daily)
            .build()?;
        
        let cv_results = nf.cross_validation(data.clone(), cv_config.clone())?;
        let metrics = cv_results.calculate_metrics()?;
        
        loss_performance.insert(loss_fn, metrics["LSTM"].clone());
        
        println!("{:?} - MAE: {:.3}, RMSE: {:.3}", 
                 loss_fn, metrics["LSTM"].mae(), metrics["LSTM"].rmse());
    }
    
    // Huber loss should perform better with outliers
    let huber_mae = loss_performance[&LossFunction::Huber].mae();
    let mse_mae = loss_performance[&LossFunction::MSE].mae();
    
    println!("\nHuber vs MSE improvement: {:.1}%", 
             (mse_mae - huber_mae) / mse_mae * 100.0);
    
    Ok(())
}

/// Test cross-validation with ensemble models
#[test]
fn test_ensemble_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        5,
        400,
        true,
        Some(7),
    )?;
    
    // Create ensemble
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(32)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
        Box::new(MLP::builder()
            .horizon(14)
            .input_size(28)
            .hidden_layers(vec![32, 16])
            .build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Run CV on ensemble
    let cv_config = CrossValidationConfig::new(4, 14)
        .with_step_size(14)
        .with_refit(true);
    
    let cv_results = nf.cross_validation(data, cv_config)?;
    
    // Compare individual model performance
    let model_metrics = cv_results.calculate_metrics()?;
    
    println!("Ensemble CV results:");
    for (model_name, metrics) in &model_metrics {
        println!("{}: MAE={:.3}, RMSE={:.3}", 
                 model_name, metrics.mae(), metrics.rmse());
    }
    
    // Calculate ensemble average performance
    let ensemble_mae: f32 = model_metrics.values()
        .map(|m| m.mae())
        .sum::<f32>() / model_metrics.len() as f32;
    
    println!("\nEnsemble average MAE: {:.3}", ensemble_mae);
    
    Ok(())
}

/// Test cross-validation stability across different random seeds
#[test]
fn test_cv_stability() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        3,
        200,
        true,
        Some(7),
    )?;
    
    let cv_config = CrossValidationConfig::new(3, 7)
        .with_step_size(7)
        .with_refit(true);
    
    let n_runs = 3;
    let mut run_results = Vec::new();
    
    for run in 0..n_runs {
        // Create model with different random seed
        let lstm = LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .random_seed(run as u64)
            .max_steps(20)
            .build()?;
        
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(lstm))
            .with_frequency(Frequency::Daily)
            .build()?;
        
        let cv_results = nf.cross_validation(data.clone(), cv_config.clone())?;
        let metrics = cv_results.calculate_metrics()?;
        let mae = metrics["LSTM"].mae();
        
        run_results.push(mae);
        println!("Run {}: MAE = {:.3}", run + 1, mae);
    }
    
    // Calculate stability metrics
    let mean_mae: f32 = run_results.iter().sum::<f32>() / n_runs as f32;
    let std_mae: f32 = {
        let variance: f32 = run_results.iter()
            .map(|&x| (x - mean_mae).powi(2))
            .sum::<f32>() / n_runs as f32;
        variance.sqrt()
    };
    
    println!("\nCV stability:");
    println!("Mean MAE: {:.3}", mean_mae);
    println!("Std MAE: {:.3}", std_mae);
    println!("CV coefficient: {:.3}", std_mae / mean_mae);
    
    // CV coefficient should be reasonably small
    assert!(std_mae / mean_mae < 0.1, "CV results are unstable");
    
    Ok(())
}

/// Test cross-validation with early stopping
#[test]
fn test_cv_with_early_stopping() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        3,
        300,
        true,
        Some(7),
    )?;
    
    // Create models with and without early stopping
    let lstm_with_es = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .max_steps(100)
        .early_stopping_patience(5)
        .build()?;
    
    let lstm_without_es = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .max_steps(100)
        .build()?;
    
    let cv_config = CrossValidationConfig::new(3, 7)
        .with_step_size(7)
        .with_refit(true);
    
    // Test with early stopping
    let start = std::time::Instant::now();
    let mut nf_es = NeuralForecast::builder()
        .with_model(Box::new(lstm_with_es))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    let cv_results_es = nf_es.cross_validation(data.clone(), cv_config.clone())?;
    let time_es = start.elapsed();
    
    // Test without early stopping
    let start = std::time::Instant::now();
    let mut nf_no_es = NeuralForecast::builder()
        .with_model(Box::new(lstm_without_es))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    let cv_results_no_es = nf_no_es.cross_validation(data, cv_config)?;
    let time_no_es = start.elapsed();
    
    // Compare results
    let metrics_es = cv_results_es.calculate_metrics()?;
    let metrics_no_es = cv_results_no_es.calculate_metrics()?;
    
    println!("With early stopping:");
    println!("  Time: {:?}", time_es);
    println!("  MAE: {:.3}", metrics_es["LSTM"].mae());
    
    println!("\nWithout early stopping:");
    println!("  Time: {:?}", time_no_es);
    println!("  MAE: {:.3}", metrics_no_es["LSTM"].mae());
    
    // Early stopping should be faster
    assert!(time_es < time_no_es);
    
    Ok(())
}

/// Test cross-validation with custom metrics
#[test]
fn test_cv_custom_metrics() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_cv_test_data::<f32>(
        2,
        200,
        true,
        Some(7),
    )?;
    
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    let cv_config = CrossValidationConfig::new(3, 7)
        .with_step_size(7)
        .with_refit(true);
    
    let cv_results = nf.cross_validation(data, cv_config)?;
    
    // Calculate various metrics
    let metrics = cv_results.calculate_metrics()?;
    let lstm_metrics = &metrics["LSTM"];
    
    // Standard metrics
    println!("Standard metrics:");
    println!("  MAE: {:.3}", lstm_metrics.mae());
    println!("  MSE: {:.3}", lstm_metrics.mse());
    println!("  RMSE: {:.3}", lstm_metrics.rmse());
    println!("  MAPE: {:.3}%", lstm_metrics.mape() * 100.0);
    println!("  SMAPE: {:.3}%", lstm_metrics.smape() * 100.0);
    
    // Custom metrics (e.g., directional accuracy)
    let directional_accuracy = cv_results.calculate_directional_accuracy()?;
    println!("\nDirectional accuracy: {:.3}%", directional_accuracy * 100.0);
    
    // Coverage for prediction intervals (if available)
    if cv_results.has_intervals() {
        let coverage = cv_results.calculate_interval_coverage()?;
        println!("Prediction interval coverage: {:.3}%", coverage * 100.0);
    }
    
    Ok(())
}