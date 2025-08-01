//! Ensemble forecasting methods
//!
//! This module provides ensemble methods for combining multiple forecasting
//! models to improve prediction accuracy and robustness.

use alloc::{
    boxed::Box,
    collections::BTreeMap as HashMap,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cmp::Ordering;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::models::{ModelType, ForecastModel, ModelFactory, TimeSeriesData};

/// Ensemble forecaster for combining multiple models
#[derive(Debug)]
pub struct EnsembleForecaster {
    models: Vec<EnsembleModel>,
    ensemble_strategy: EnsembleStrategy,
    weights: Option<Vec<f32>>,
    meta_learner: Option<MetaLearner>,
    config: EnsembleConfig,
}

/// Individual model in the ensemble
#[derive(Clone, Debug)]
pub struct EnsembleModel {
    pub name: String,
    pub model_type: ModelType,
    pub weight: f32,
    pub performance_metrics: ModelPerformanceMetrics,
    pub out_of_sample_predictions: Vec<Vec<f32>>,
    pub training_predictions: Vec<Vec<f32>>,
}

/// Model performance metrics
#[derive(Clone, Debug)]
pub struct ModelPerformanceMetrics {
    pub mae: f32,
    pub mse: f32,
    pub mape: f32,
    pub smape: f32,
    pub coverage: f32, // For prediction intervals
}

/// Ensemble strategy
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnsembleStrategy {
    SimpleAverage,
    WeightedAverage,
    Median,
    TrimmedMean(f32), // Trim percentage
    Voting,
    Stacking,
    BayesianModelAveraging,
}

/// Ensemble configuration
#[derive(Clone, Debug)]
pub struct EnsembleConfig {
    pub strategy: EnsembleStrategy,
    pub models: Vec<String>,
    pub weights: Option<Vec<f32>>,
    pub meta_learner: Option<String>,
    pub optimization_metric: OptimizationMetric,
    pub stacking_cv_folds: usize,
    pub bootstrap_samples: usize,
    pub quantile_levels: Vec<f32>,
}

/// Optimization metric for ensemble weights
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizationMetric {
    MAE,
    MSE,
    MAPE,
    SMAPE,
    CombinedScore,
    LogLikelihood,
    Quantile,
    Sharpe,
}

/// Meta-learner for stacking ensemble
#[derive(Clone, Debug)]
pub struct MetaLearner {
    pub learner_type: MetaLearnerType,
    pub parameters: Vec<f32>,
    pub is_trained: bool,
}

/// Types of meta-learners for stacking
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetaLearnerType {
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    RandomForest,
    XGBoost,
}

impl EnsembleForecaster {
    /// Create a new ensemble forecaster
    pub fn new(config: EnsembleConfig) -> Result<Self, String> {
        if let Some(ref weights) = config.weights {
            if weights.len() != config.models.len() {
                return Err("Number of weights must match number of models".to_string());
            }

            // Validate weights sum to 1.0 for weighted average
            if config.strategy == EnsembleStrategy::WeightedAverage {
                let weight_sum: f32 = weights.iter().sum();
                if (weight_sum - 1.0).abs() > 1e-6 {
                    return Err("Weights must sum to 1.0 for weighted average".to_string());
                }
            }
        }

        // Initialize meta-learner for stacking
        let meta_learner = if config.strategy == EnsembleStrategy::Stacking {
            Some(MetaLearner {
                learner_type: MetaLearnerType::LinearRegression,
                parameters: Vec::new(),
                is_trained: false,
            })
        } else {
            None
        };

        Ok(Self {
            models: Vec::new(),
            ensemble_strategy: config.strategy,
            weights: config.weights.clone(),
            meta_learner,
            config,
        })
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, model: EnsembleModel) {
        self.models.push(model);
    }
    
    /// Create ensemble from neural network models
    pub fn from_neural_models(
        config: EnsembleConfig,
        model_types: Vec<ModelType>,
        input_size: usize,
        output_size: usize,
    ) -> Result<(Self, Vec<Box<dyn ForecastModel>>), String> {
        let mut forecaster = Self::new(config)?;
        let mut neural_models = Vec::new();
        
        for model_type in model_types {
            // Create the neural network model
            let model = ModelFactory::create_model(model_type, input_size, output_size)?;
            
            // Create ensemble model metadata
            let ensemble_model = EnsembleModel {
                name: model_type.to_string(),
                model_type,
                weight: 1.0 / forecaster.config.models.len() as f32,
                performance_metrics: ModelPerformanceMetrics {
                    mae: 0.0,
                    mse: 0.0,
                    mape: 0.0,
                    smape: 0.0,
                    coverage: 0.0,
                },
                out_of_sample_predictions: Vec::new(),
                training_predictions: Vec::new(),
            };
            
            forecaster.add_model(ensemble_model);
            neural_models.push(model);
        }
        
        Ok((forecaster, neural_models))
    }
    
    /// Train ensemble of neural network models
    pub fn train_ensemble(
        &mut self,
        models: &mut [Box<dyn ForecastModel>],
        training_data: &TimeSeriesData,
        validation_data: Option<&TimeSeriesData>,
    ) -> Result<(), String> {
        if models.len() != self.models.len() {
            return Err("Number of neural models must match ensemble models".to_string());
        }
        
        // Train each model
        for (i, model) in models.iter_mut().enumerate() {
            match model.fit(training_data) {
                Ok(_) => {
                    // Update model performance if validation data is available
                    if let Some(val_data) = validation_data {
                        // Use a reasonable horizon that matches model output size
                        let horizon = val_data.values.len().min(3); // Limit to 3 for this example
                        let predictions = model.predict(horizon)?;
                        let metrics = calculate_model_metrics(&predictions, &val_data.values);
                        self.models[i].performance_metrics = metrics;
                        self.models[i].training_predictions.push(predictions);
                    }
                }
                Err(e) => {
                    return Err(format!("Failed to train model {}: {}", self.models[i].name, e));
                }
            }
        }
        
        // Optimize ensemble weights if validation data is available
        if let Some(val_data) = validation_data {
            let horizon = val_data.values.len().min(3); // Limit to 3 for this example
            let predictions: Vec<Vec<f32>> = models
                .iter_mut()
                .map(|model| model.predict(horizon).unwrap_or_default())
                .collect();
            
            self.optimize_weights(&predictions, &val_data.values, self.config.optimization_metric)?;
        }
        
        Ok(())
    }
    
    /// Generate ensemble predictions from neural network models
    pub fn predict_with_models(
        &self,
        models: &mut [Box<dyn ForecastModel>],
        horizon: usize,
    ) -> Result<EnsembleForecast, String> {
        if models.len() != self.models.len() {
            return Err("Number of neural models must match ensemble models".to_string());
        }
        
        // Generate predictions from each model
        let predictions: Result<Vec<Vec<f32>>, String> = models
            .iter_mut()
            .map(|model| model.predict(horizon))
            .collect();
        
        let predictions = predictions?;
        self.ensemble_predict(&predictions)
    }

    /// Generate ensemble forecast
    pub fn ensemble_predict(&self, predictions: &[Vec<f32>]) -> Result<EnsembleForecast, String> {
        if predictions.is_empty() {
            return Err("No predictions provided".to_string());
        }

        // Validate all predictions have the same length
        let horizon = predictions[0].len();
        if !predictions.iter().all(|p| p.len() == horizon) {
            return Err("All predictions must have the same horizon".to_string());
        }

        let point_forecast = match self.ensemble_strategy {
            EnsembleStrategy::SimpleAverage => self.simple_average(predictions)?,
            EnsembleStrategy::WeightedAverage => self.weighted_average(predictions)?,
            EnsembleStrategy::Median => self.median_ensemble(predictions)?,
            EnsembleStrategy::TrimmedMean(trim_pct) => self.trimmed_mean(predictions, trim_pct)?,
            EnsembleStrategy::Voting => self.voting_ensemble(predictions)?,
            EnsembleStrategy::Stacking => {
                if self.meta_learner.is_none() || !self.meta_learner.as_ref().unwrap().is_trained {
                    return Err("Stacking requires trained meta-learner".to_string());
                }
                self.stacking_ensemble(predictions)?
            }
            EnsembleStrategy::BayesianModelAveraging => {
                self.bayesian_model_averaging(predictions)?
            }
        };

        // Calculate prediction intervals
        let intervals = self.calculate_prediction_intervals(predictions, &point_forecast);

        // Calculate ensemble metrics
        let metrics = self.calculate_ensemble_metrics(predictions, &point_forecast);

        Ok(EnsembleForecast {
            point_forecast,
            prediction_intervals: intervals,
            ensemble_metrics: metrics,
            models_used: predictions.len(),
            strategy: self.ensemble_strategy,
        })
    }

    /// Simple average of all predictions
    fn simple_average(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for pred in predictions {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value;
            }
        }

        for value in &mut result {
            *value /= predictions.len() as f32;
        }

        Ok(result)
    }

    /// Weighted average of predictions
    fn weighted_average(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| "Weights not provided for weighted average".to_string())?;

        if weights.len() != predictions.len() {
            return Err("Number of weights must match number of predictions".to_string());
        }

        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for (pred, &weight) in predictions.iter().zip(weights.iter()) {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value * weight;
            }
        }

        Ok(result)
    }

    /// Median ensemble
    fn median_ensemble(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for i in 0..horizon {
            let mut values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            result[i] = if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };
        }

        Ok(result)
    }

    /// Trimmed mean ensemble
    fn trimmed_mean(
        &self,
        predictions: &[Vec<f32>],
        trim_percent: f32,
    ) -> Result<Vec<f32>, String> {
        if !(0.0..0.5).contains(&trim_percent) {
            return Err("Trim percentage must be between 0 and 0.5".to_string());
        }

        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];
        let trim_count = ((predictions.len() as f32) * trim_percent).floor() as usize;

        for i in 0..horizon {
            let mut values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Remove extreme values
            if trim_count > 0 && values.len() > 2 * trim_count {
                values = values[trim_count..values.len() - trim_count].to_vec();
            }

            result[i] = values.iter().sum::<f32>() / values.len() as f32;
        }

        Ok(result)
    }

    /// Voting ensemble (for classification-like problems)
    fn voting_ensemble(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        // For regression, we can use a threshold-based voting
        // This is a simplified implementation
        self.median_ensemble(predictions)
    }

    /// Bayesian model averaging
    fn bayesian_model_averaging(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        // Calculate model weights based on historical performance
        let model_weights = self.calculate_bayesian_weights();

        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for (pred, &weight) in predictions.iter().zip(model_weights.iter()) {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value * weight;
            }
        }

        Ok(result)
    }

    /// Stacking ensemble using trained meta-learner
    fn stacking_ensemble(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let meta_learner = self.meta_learner.as_ref().unwrap();
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        // For each time step, use meta-learner to combine predictions
        for i in 0..horizon {
            let features: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();
            let prediction = self.apply_meta_learner(&features, meta_learner)?;
            result[i] = prediction;
        }

        Ok(result)
    }

    /// Apply meta-learner to combine predictions
    fn apply_meta_learner(&self, features: &[f32], meta_learner: &MetaLearner) -> Result<f32, String> {
        match meta_learner.learner_type {
            MetaLearnerType::LinearRegression => {
                if meta_learner.parameters.len() != features.len() + 1 {
                    return Err("Meta-learner parameters don't match feature size".to_string());
                }
                
                let mut result = meta_learner.parameters[0]; // bias term
                for (i, &feature) in features.iter().enumerate() {
                    result += feature * meta_learner.parameters[i + 1];
                }
                Ok(result)
            }
            MetaLearnerType::Ridge => {
                // Ridge regression is similar to linear regression but with L2 regularization
                // For inference, it's the same as linear regression
                self.apply_linear_regression(features, &meta_learner.parameters)
            }
            MetaLearnerType::Lasso => {
                // Lasso regression for inference
                self.apply_linear_regression(features, &meta_learner.parameters)
            }
            MetaLearnerType::ElasticNet => {
                // Elastic net for inference
                self.apply_linear_regression(features, &meta_learner.parameters)
            }
            MetaLearnerType::RandomForest => {
                // Simplified random forest - use weighted average
                let avg: f32 = features.iter().sum::<f32>() / features.len() as f32;
                Ok(avg)
            }
            MetaLearnerType::XGBoost => {
                // Simplified XGBoost - use weighted average with non-linear transformation
                let weighted_sum: f32 = features.iter().zip(meta_learner.parameters.iter())
                    .map(|(&f, &w)| f * w)
                    .sum();
                Ok(weighted_sum.tanh()) // Simple non-linear transformation
            }
        }
    }

    /// Apply linear regression with given parameters
    fn apply_linear_regression(&self, features: &[f32], parameters: &[f32]) -> Result<f32, String> {
        if parameters.len() != features.len() + 1 {
            return Err("Parameter count mismatch".to_string());
        }
        
        let mut result = parameters[0]; // bias term
        for (i, &feature) in features.iter().enumerate() {
            result += feature * parameters[i + 1];
        }
        Ok(result)
    }

    /// Calculate Bayesian weights based on model performance
    fn calculate_bayesian_weights(&self) -> Vec<f32> {
        if self.models.is_empty() {
            return vec![1.0];
        }

        // Use inverse MSE as weight basis
        let mse_values: Vec<f32> = self
            .models
            .iter()
            .map(|m| m.performance_metrics.mse)
            .collect();

        let min_mse = mse_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Calculate weights proportional to inverse MSE
        let raw_weights: Vec<f32> = mse_values
            .iter()
            .map(|&mse| if mse > 0.0 { min_mse / mse } else { 1.0 })
            .collect();

        // Normalize weights
        let weight_sum: f32 = raw_weights.iter().sum();
        raw_weights.iter().map(|&w| w / weight_sum).collect()
    }

    /// Calculate prediction intervals using multiple methods
    fn calculate_prediction_intervals(
        &self,
        predictions: &[Vec<f32>],
        point_forecast: &[f32],
    ) -> PredictionIntervals {
        // Use quantile-based method for more robust intervals
        self.calculate_quantile_intervals(predictions)
    }

    /// Calculate prediction intervals using quantile method
    fn calculate_quantile_intervals(&self, predictions: &[Vec<f32>]) -> PredictionIntervals {
        let horizon = predictions[0].len();
        let mut lower_50 = vec![0.0; horizon];
        let mut upper_50 = vec![0.0; horizon];
        let mut lower_80 = vec![0.0; horizon];
        let mut upper_80 = vec![0.0; horizon];
        let mut lower_95 = vec![0.0; horizon];
        let mut upper_95 = vec![0.0; horizon];

        for i in 0..horizon {
            let mut values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

            // Calculate quantiles
            lower_50[i] = self.calculate_quantile(&values, 0.25);
            upper_50[i] = self.calculate_quantile(&values, 0.75);
            
            lower_80[i] = self.calculate_quantile(&values, 0.10);
            upper_80[i] = self.calculate_quantile(&values, 0.90);
            
            lower_95[i] = self.calculate_quantile(&values, 0.025);
            upper_95[i] = self.calculate_quantile(&values, 0.975);
        }

        PredictionIntervals {
            level_50: (lower_50, upper_50),
            level_80: (lower_80, upper_80),
            level_95: (lower_95, upper_95),
        }
    }

    /// Calculate quantile from sorted values
    fn calculate_quantile(&self, sorted_values: &[f32], quantile: f32) -> f32 {
        if sorted_values.is_empty() {
            return 0.0;
        }
        
        let index = (quantile * (sorted_values.len() - 1) as f32).round() as usize;
        let clamped_index = index.min(sorted_values.len() - 1);
        sorted_values[clamped_index]
    }

    /// Calculate ensemble performance metrics
    fn calculate_ensemble_metrics(
        &self,
        predictions: &[Vec<f32>],
        ensemble_forecast: &[f32],
    ) -> EnsembleMetrics {
        let horizon = ensemble_forecast.len();

        // Calculate diversity metrics
        let diversity_metrics = self.calculate_diversity_metrics(predictions);
        
        // Calculate prediction variance
        let mut prediction_variance = 0.0;
        for i in 0..horizon {
            let values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = ensemble_forecast[i];
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            prediction_variance += variance;
        }
        prediction_variance /= horizon as f32;

        let average_weight = if predictions.len() > 0 {
            1.0 / predictions.len() as f32
        } else {
            0.0
        };

        EnsembleMetrics {
            diversity_score: diversity_metrics.overall_diversity,
            average_model_weight: average_weight,
            prediction_variance,
            effective_models: self.calculate_effective_models(),
            correlation_matrix: diversity_metrics.correlation_matrix,
            disagreement_measure: diversity_metrics.disagreement,
            entropy_measure: diversity_metrics.entropy,
        }
    }

    /// Calculate comprehensive diversity metrics
    pub fn calculate_diversity_metrics(&self, predictions: &[Vec<f32>]) -> DiversityMetrics {
        let n_models = predictions.len();
        let horizon = predictions[0].len();
        
        // Calculate correlation matrix
        let mut correlation_matrix = vec![vec![0.0; n_models]; n_models];
        for i in 0..n_models {
            for j in 0..n_models {
                if i == j {
                    correlation_matrix[i][j] = 1.0;
                } else {
                    // Calculate correlation between prediction arrays
                    let corr = self.calculate_correlation(&predictions[i], &predictions[j]);
                    correlation_matrix[i][j] = corr;
                }
            }
        }
        
        // Calculate average pairwise correlation
        let mut pairwise_correlations = Vec::new();
        for i in 0..n_models {
            for j in (i + 1)..n_models {
                pairwise_correlations.push(correlation_matrix[i][j]);
            }
        }
        
        let avg_correlation = if pairwise_correlations.is_empty() {
            0.0
        } else {
            let sum_corr: f32 = pairwise_correlations.iter().sum();
            let avg = sum_corr / pairwise_correlations.len() as f32;
            // Clamp correlation to prevent numerical issues
            avg.max(-1.0).min(1.0)
        };
        
        // Calculate disagreement measure (average pairwise squared difference)
        let mut disagreement = 0.0;
        for i in 0..horizon {
            let mut step_disagreement = 0.0;
            let mut pair_count = 0;
            for m1 in 0..n_models {
                for m2 in (m1 + 1)..n_models {
                    step_disagreement += (predictions[m1][i] - predictions[m2][i]).powi(2);
                    pair_count += 1;
                }
            }
            if pair_count > 0 {
                disagreement += step_disagreement / pair_count as f32;
            }
        }
        disagreement /= horizon as f32;
        
        // Calculate entropy-based diversity
        let weights = match &self.weights {
            Some(w) => w.clone(),
            None => vec![1.0 / n_models as f32; n_models],
        };
        
        let entropy = -weights.iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| w * w.ln())
            .sum::<f32>();
        
        // Calculate pairwise correlations for compatibility
        let mut pairwise_correlations = Vec::new();
        for i in 0..n_models {
            for j in (i + 1)..n_models {
                pairwise_correlations.push(correlation_matrix[i][j]);
            }
        }

        let diversity_score = 1.0 - avg_correlation.abs();
        let effective_model_count = if entropy.is_finite() && entropy > 0.0 {
            entropy.exp()
        } else {
            n_models as f32
        };

        DiversityMetrics {
            pairwise_correlations,
            effective_model_count,
            diversity_score,
            overall_diversity: diversity_score,
            correlation_matrix,
            disagreement,
            entropy,
        }
    }

    /// Calculate correlation between two prediction series
    fn calculate_correlation(&self, pred1: &[f32], pred2: &[f32]) -> f32 {
        let n = pred1.len() as f32;
        let mean1 = pred1.iter().sum::<f32>() / n;
        let mean2 = pred2.iter().sum::<f32>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..pred1.len() {
            let diff1 = pred1[i] - mean1;
            let diff2 = pred2[i] - mean2;
            cov += diff1 * diff2;
            var1 += diff1.powi(2);
            var2 += diff2.powi(2);
        }

        if var1 == 0.0 || var2 == 0.0 {
            return 0.0;
        }

        cov / (var1.sqrt() * var2.sqrt())
    }

    /// Calculate effective number of models (based on weight entropy)
    fn calculate_effective_models(&self) -> f32 {
        let weights = match &self.weights {
            Some(w) => w.clone(),
            None => vec![1.0 / self.models.len() as f32; self.models.len()],
        };

        if weights.is_empty() {
            return 1.0;
        }

        // Calculate entropy-based effective number
        let entropy: f32 = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum();

        if entropy.is_finite() && entropy > 0.0 {
            entropy.exp()
        } else {
            weights.len() as f32
        }
    }

    /// Optimize ensemble weights using validation data
    pub fn optimize_weights(
        &mut self,
        validation_predictions: &[Vec<f32>],
        validation_actuals: &[f32],
        metric: OptimizationMetric,
    ) -> Result<Vec<f32>, String> {
        if validation_predictions.len() != self.models.len() {
            return Err(format!("Number of predictions ({}) must match number of models ({})", 
                validation_predictions.len(), self.models.len()));
        }

        // Simple grid search for weights (can be replaced with more sophisticated optimization)
        let mut best_weights = vec![1.0 / self.models.len() as f32; self.models.len()];
        let mut best_score = f32::INFINITY;

        // Generate weight combinations
        let weight_options = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        // This is a simplified version - real implementation would use optimization algorithms
        for i in 0..self.models.len() {
            let mut test_weights = best_weights.clone();
            for &w in &weight_options {
                test_weights[i] = w;

                // Normalize weights
                let sum: f32 = test_weights.iter().sum();
                if sum > 0.0 {
                    for weight in &mut test_weights {
                        *weight /= sum;
                    }

                    // Calculate ensemble forecast with test weights
                    self.weights = Some(test_weights.clone());
                    if let Ok(forecast) = self.weighted_average(validation_predictions) {
                        let score = self.calculate_optimization_score(
                            &forecast,
                            validation_actuals,
                            metric,
                        );

                        if score < best_score {
                            best_score = score;
                            best_weights = test_weights.clone();
                        }
                    }
                }
            }
        }

        self.weights = Some(best_weights.clone());
        Ok(best_weights)
    }

    /// Calculate optimization score
    fn calculate_optimization_score(
        &self,
        forecast: &[f32],
        actuals: &[f32],
        metric: OptimizationMetric,
    ) -> f32 {
        match metric {
            OptimizationMetric::MAE => calculate_mae(forecast, actuals),
            OptimizationMetric::MSE => calculate_mse(forecast, actuals),
            OptimizationMetric::MAPE => calculate_mape(forecast, actuals),
            OptimizationMetric::SMAPE => calculate_smape(forecast, actuals),
            OptimizationMetric::CombinedScore => {
                // Weighted combination of metrics
                let mae = calculate_mae(forecast, actuals);
                let mape = calculate_mape(forecast, actuals);
                0.5 * mae + 0.5 * mape
            }
            OptimizationMetric::LogLikelihood => {
                // Placeholder for log-likelihood calculation
                calculate_mse(forecast, actuals)
            }
            OptimizationMetric::Quantile => {
                // Placeholder for quantile loss calculation
                calculate_mae(forecast, actuals)
            }
            OptimizationMetric::Sharpe => {
                // Placeholder for Sharpe ratio calculation
                calculate_mae(forecast, actuals)
            }
        }
    }

    /// Get current ensemble weights
    pub fn get_current_weights(&self) -> Vec<f32> {
        match &self.weights {
            Some(weights) => weights.clone(),
            None => vec![1.0 / self.models.len() as f32; self.models.len()],
        }
    }

    /// Get current ensemble weights  
    pub fn get_weights(&self) -> Vec<f32> {
        match &self.weights {
            Some(weights) => weights.clone(),
            None => vec![1.0 / self.models.len() as f32; self.models.len()],
        }
    }
}

/// Ensemble forecast result
#[derive(Clone, Debug)]
pub struct EnsembleForecast {
    pub point_forecast: Vec<f32>,
    pub prediction_intervals: PredictionIntervals,
    pub ensemble_metrics: EnsembleMetrics,
    pub models_used: usize,
    pub strategy: EnsembleStrategy,
}

/// Prediction intervals at different confidence levels
#[derive(Clone, Debug)]
pub struct PredictionIntervals {
    pub level_50: (Vec<f32>, Vec<f32>), // (lower, upper)
    pub level_80: (Vec<f32>, Vec<f32>),
    pub level_95: (Vec<f32>, Vec<f32>),
}


/// Ensemble performance metrics
#[derive(Clone, Debug)]
pub struct EnsembleMetrics {
    pub diversity_score: f32, // 0-1, higher is more diverse
    pub average_model_weight: f32,
    pub prediction_variance: f32,
    pub effective_models: f32, // Effective number of models based on weights
    pub correlation_matrix: Vec<Vec<f32>>,
    pub disagreement_measure: f32,
    pub entropy_measure: f32,
}

/// Diversity metrics for ensemble analysis
#[derive(Clone, Debug)]
pub struct DiversityMetrics {
    pub pairwise_correlations: Vec<f32>,
    pub effective_model_count: f32,
    pub diversity_score: f32,
    pub overall_diversity: f32,
    pub correlation_matrix: Vec<Vec<f32>>,
    pub disagreement: f32,
    pub entropy: f32,
}

/// Calculate Mean Absolute Error
fn calculate_mae(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .map(|(&f, &a)| (f - a).abs())
        .sum::<f32>()
        / forecast.len() as f32
}

/// Calculate Mean Squared Error
fn calculate_mse(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .map(|(&f, &a)| (f - a).powi(2))
        .sum::<f32>()
        / forecast.len() as f32
}

/// Calculate Mean Absolute Percentage Error
fn calculate_mape(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .filter(|(_, &a)| a != 0.0)
        .map(|(&f, &a)| ((f - a) / a).abs())
        .sum::<f32>()
        / forecast.len() as f32
        * 100.0
}

/// Calculate Symmetric Mean Absolute Percentage Error
fn calculate_smape(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .map(|(&f, &a)| {
            let denominator = (f.abs() + a.abs()) / 2.0;
            if denominator == 0.0 {
                0.0
            } else {
                (f - a).abs() / denominator
            }
        })
        .sum::<f32>()
        / forecast.len() as f32
        * 100.0
}

/// Calculate comprehensive model performance metrics
fn calculate_model_metrics(predictions: &[f32], actuals: &[f32]) -> ModelPerformanceMetrics {
    let mae = calculate_mae(predictions, actuals);
    let mse = calculate_mse(predictions, actuals);
    let mape = calculate_mape(predictions, actuals);
    let smape = calculate_smape(predictions, actuals);
    
    // Calculate coverage (simplified - in practice would use prediction intervals)
    let coverage = 0.95; // Placeholder
    
    ModelPerformanceMetrics {
        mae,
        mse,
        mape,
        smape,
        coverage,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_average() {
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::SimpleAverage,
            models: vec!["model1".to_string(), "model2".to_string()],
            weights: None,
            meta_learner: None,
            optimization_metric: OptimizationMetric::MAE,
            stacking_cv_folds: 5,
            bootstrap_samples: 100,
            quantile_levels: vec![0.1, 0.5, 0.9],
        };

        let forecaster = EnsembleForecaster::new(config).unwrap();

        let predictions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let result = forecaster.ensemble_predict(&predictions).unwrap();

        assert_eq!(result.point_forecast, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_weighted_average() {
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::WeightedAverage,
            models: vec!["model1".to_string(), "model2".to_string()],
            weights: Some(vec![0.3, 0.7]),
            meta_learner: None,
            optimization_metric: OptimizationMetric::MAE,
            stacking_cv_folds: 5,
            bootstrap_samples: 100,
            quantile_levels: vec![0.1, 0.5, 0.9],
        };

        let forecaster = EnsembleForecaster::new(config).unwrap();

        let predictions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let result = forecaster.ensemble_predict(&predictions).unwrap();

        assert_eq!(result.point_forecast[0], 1.0 * 0.3 + 2.0 * 0.7);
        assert_eq!(result.point_forecast[1], 2.0 * 0.3 + 3.0 * 0.7);
    }
}
