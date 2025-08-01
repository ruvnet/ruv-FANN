//! WASM bindings for neural forecasting
//!
//! This module provides WebAssembly bindings for all forecasting functionality,
//! making it accessible from JavaScript.

#![cfg(target_arch = "wasm32")]

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use wasm_bindgen::prelude::*;

use crate::{
    agent_forecasting::{AgentForecastingManager, ForecastDomain, ForecastRequirements},
    ensemble::{EnsembleConfig, EnsembleForecaster, EnsembleStrategy},
    models::{ModelFactory, ModelType},
    time_series::{TimeSeriesData, TimeSeriesProcessor, TransformationType},
};

/// WASM-compatible neural forecast interface
#[wasm_bindgen]
pub struct WasmNeuralForecast {
    agent_manager: AgentForecastingManager,
    processor: TimeSeriesProcessor,
}

#[wasm_bindgen]
impl WasmNeuralForecast {
    /// Create a new neural forecast instance
    #[wasm_bindgen(constructor)]
    pub fn new(memory_limit_mb: f32) -> Self {
        Self {
            agent_manager: AgentForecastingManager::new(memory_limit_mb),
            processor: TimeSeriesProcessor::new(),
        }
    }

    /// Assign a forecasting model to an agent
    #[wasm_bindgen]
    pub fn assign_agent_model(
        &mut self,
        agent_id: String,
        agent_type: String,
        horizon: usize,
        accuracy_target: f32,
        latency_ms: f32,
    ) -> Result<String, JsValue> {
        let requirements = ForecastRequirements {
            horizon,
            frequency: "H".to_string(),
            accuracy_target,
            latency_requirement_ms: latency_ms,
            interpretability_needed: false,
            online_learning: true,
        };

        self.agent_manager
            .assign_model(agent_id, agent_type, requirements)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Get available models
    #[wasm_bindgen]
    pub fn get_available_models() -> JsValue {
        let models = ModelFactory::get_available_models();
        let model_info: Vec<_> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "type": format!("{:?}", m.model_type),
                    "category": format!("{:?}", m.category),
                    "minSamples": m.min_samples,
                    "supportsMultivariate": m.supports_multivariate,
                    "typicalMemoryMB": m.typical_memory_mb,
                    "interpretabilityScore": m.interpretability_score,
                })
            })
            .collect();

        JsValue::from_str(&serde_json::to_string(&model_info).unwrap())
    }

    /// Process time series data
    #[wasm_bindgen]
    pub fn process_time_series(
        &mut self,
        values: Vec<f32>,
        timestamps: Vec<f64>,
        transformations: Vec<String>,
    ) -> Result<JsValue, JsValue> {
        let data = TimeSeriesData {
            values,
            timestamps,
            frequency: "H".to_string(),
            unique_id: "processed".to_string(),
        };

        let transforms: Result<Vec<_>, _> = transformations
            .iter()
            .map(|t| match t.as_str() {
                "normalize" => Ok(TransformationType::Normalize),
                "standardize" => Ok(TransformationType::Standardize),
                "log" => Ok(TransformationType::Log),
                "difference" => Ok(TransformationType::Difference),
                _ => Err(format!("Unknown transformation: {}", t)),
            })
            .collect();

        let transforms = transforms.map_err(|e| JsValue::from_str(&e))?;

        let processed = self
            .processor
            .fit_transform(data, transforms)
            .map_err(|e| JsValue::from_str(&e))?;

        Ok(JsValue::from_str(
            &serde_json::to_string(&serde_json::json!({
                "values": processed.values,
                "timestamps": processed.timestamps,
                "transformations": transformations,
            }))
            .unwrap(),
        ))
    }

    /// Update agent performance
    #[wasm_bindgen]
    pub fn update_agent_performance(
        &mut self,
        agent_id: &str,
        latency_ms: f32,
        accuracy: f32,
        confidence: f32,
    ) -> Result<(), JsValue> {
        self.agent_manager
            .update_performance(agent_id, latency_ms, accuracy, confidence)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Get agent forecast state
    #[wasm_bindgen]
    pub fn get_agent_state(&self, agent_id: &str) -> Result<JsValue, JsValue> {
        let state = self
            .agent_manager
            .get_agent_state(agent_id)
            .ok_or_else(|| JsValue::from_str("Agent not found"))?;

        Ok(JsValue::from_str(
            &serde_json::to_string(&serde_json::json!({
                "agentId": state.agent_id,
                "agentType": state.agent_type,
                "primaryModel": format!("{:?}", state.primary_model),
                "ensembleSize": state.ensemble_models.len(),
                "forecastDomain": format!("{:?}", state.model_specialization.forecast_domain),
                "onlineLearning": state.adaptive_config.online_learning_enabled,
                "performanceHistory": {
                    "totalForecasts": state.performance_history.total_forecasts,
                    "avgConfidence": state.performance_history.average_confidence,
                    "avgLatency": state.performance_history.average_latency_ms,
                },
            }))
            .unwrap(),
        ))
    }
}

/// WASM-compatible ensemble forecaster
#[wasm_bindgen]
pub struct WasmEnsembleForecaster {
    forecaster: EnsembleForecaster,
}

#[wasm_bindgen]
impl WasmEnsembleForecaster {
    /// Create a new ensemble forecaster
    #[wasm_bindgen(constructor)]
    pub fn new(
        strategy: String,
        model_names: Vec<String>,
    ) -> Result<WasmEnsembleForecaster, JsValue> {
        let ensemble_strategy = match strategy.as_str() {
            "simple_average" => EnsembleStrategy::SimpleAverage,
            "weighted_average" => EnsembleStrategy::WeightedAverage,
            "median" => EnsembleStrategy::Median,
            "trimmed_mean" => EnsembleStrategy::TrimmedMean(0.2),
            _ => return Err(JsValue::from_str("Unknown ensemble strategy")),
        };

        let config = EnsembleConfig {
            strategy: ensemble_strategy,
            models: model_names,
            weights: None,
            meta_learner: None,
            optimization_metric: crate::ensemble::OptimizationMetric::MAE,
        };

        let forecaster = EnsembleForecaster::new(config).map_err(|e| JsValue::from_str(&e))?;

        Ok(Self { forecaster })
    }

    /// Generate ensemble forecast
    #[wasm_bindgen]
    pub fn predict(&self, predictions: JsValue) -> Result<JsValue, JsValue> {
        let predictions: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(predictions)
            .map_err(|e| JsValue::from_str(&format!("Invalid predictions: {}", e)))?;

        let result = self
            .forecaster
            .ensemble_predict(&predictions)
            .map_err(|e| JsValue::from_str(&e))?;

        Ok(JsValue::from_str(
            &serde_json::to_string(&serde_json::json!({
                "pointForecast": result.point_forecast,
                "intervals": {
                    "50": {
                        "lower": result.prediction_intervals.level_50.0,
                        "upper": result.prediction_intervals.level_50.1,
                    },
                    "80": {
                        "lower": result.prediction_intervals.level_80.0,
                        "upper": result.prediction_intervals.level_80.1,
                    },
                    "95": {
                        "lower": result.prediction_intervals.level_95.0,
                        "upper": result.prediction_intervals.level_95.1,
                    },
                },
                "metrics": {
                    "diversityScore": result.ensemble_metrics.diversity_score,
                    "predictionVariance": result.ensemble_metrics.prediction_variance,
                    "effectiveModels": result.ensemble_metrics.effective_models,
                },
                "strategy": format!("{:?}", result.strategy),
            }))
            .unwrap(),
        ))
    }
}

/// WASM utilities for model information
#[wasm_bindgen]
pub struct WasmModelFactory;

#[wasm_bindgen]
impl WasmModelFactory {
    /// Get model requirements
    #[wasm_bindgen]
    pub fn get_model_requirements(model_name: &str) -> Result<JsValue, JsValue> {
        let model_type = match model_name.to_uppercase().as_str() {
            "LSTM" => ModelType::LSTM,
            "GRU" => ModelType::GRU,
            "NBEATS" => ModelType::NBEATS,
            "NHITS" => ModelType::NHITS,
            "TFT" => ModelType::TFT,
            "DEEPAR" => ModelType::DeepAR,
            "TCN" => ModelType::TCN,
            _ => return Err(JsValue::from_str("Unknown model type")),
        };

        let requirements = ModelFactory::get_model_requirements(model_type);

        Ok(JsValue::from_str(
            &serde_json::to_string(&serde_json::json!({
                "requiredParams": requirements.required_params,
                "optionalParams": requirements.optional_params,
                "minSamples": requirements.min_samples,
                "maxHorizon": requirements.max_horizon,
                "supportsMissingValues": requirements.supports_missing_values,
            }))
            .unwrap(),
        ))
    }

    /// Get all supported model types
    #[wasm_bindgen]
    pub fn get_all_models() -> JsValue {
        let models = vec![
            // Basic Models
            ("MLP", "Multi-Layer Perceptron", "basic"),
            ("DLinear", "Decomposition Linear", "basic"),
            ("NLinear", "Normalization Linear", "basic"),
            ("MLPMultivariate", "Multivariate MLP", "basic"),
            // Recurrent Models
            ("RNN", "Recurrent Neural Network", "recurrent"),
            ("LSTM", "Long Short-Term Memory", "recurrent"),
            ("GRU", "Gated Recurrent Unit", "recurrent"),
            // Advanced Models
            ("NBEATS", "Neural Basis Expansion Analysis", "advanced"),
            ("NBEATSx", "Extended NBEATS", "advanced"),
            ("NHITS", "Neural Hierarchical Interpolation", "advanced"),
            ("TiDE", "Time-series Dense Encoder", "advanced"),
            // Transformer Models
            ("TFT", "Temporal Fusion Transformer", "transformer"),
            ("Informer", "Informer Transformer", "transformer"),
            ("AutoFormer", "AutoFormer", "transformer"),
            ("FedFormer", "FedFormer", "transformer"),
            ("PatchTST", "Patch Time Series Transformer", "transformer"),
            ("iTransformer", "Inverted Transformer", "transformer"),
            // Specialized Models
            ("DeepAR", "Deep AutoRegressive", "specialized"),
            ("DeepNPTS", "Deep Non-Parametric Time Series", "specialized"),
            ("TCN", "Temporal Convolutional Network", "specialized"),
            ("BiTCN", "Bidirectional TCN", "specialized"),
            ("TimesNet", "TimesNet", "specialized"),
            (
                "StemGNN",
                "Spectral Temporal Graph Neural Network",
                "specialized",
            ),
            ("TSMixer", "Time Series Mixer", "specialized"),
            ("TSMixerx", "Extended TSMixer", "specialized"),
            ("PatchMixer", "Patch Mixer", "specialized"),
            ("SegRNN", "Segment Recurrent Neural Network", "specialized"),
            ("DishTS", "Dish Time Series", "specialized"),
        ];

        let model_list: Vec<_> = models
            .iter()
            .map(|(code, name, category)| {
                serde_json::json!({
                    "code": code,
                    "name": name,
                    "category": category,
                })
            })
            .collect();

        JsValue::from_str(&serde_json::to_string(&model_list).unwrap())
    }
}
