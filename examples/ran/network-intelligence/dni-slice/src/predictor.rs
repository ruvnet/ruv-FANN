use crate::{
    slice_service::*,
    config::{PredictionConfig, NeuralNetworkConfig, FeatureEngineeringConfig},
    error::{SliceError, SliceResult},
    models::{SliceModel, ModelRegistry},
    storage::SliceStorage,
    metrics::SliceMetricsCollector,
};
use ruv_fann::{Network, ActivationFunction, TrainingAlgorithm, ErrorFunction, StopFunction};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex, mpsc};
use tracing::{info, warn, error, debug};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2};

/// Core SLA breach prediction engine using ruv-FANN
pub struct SlaBreachPredictor {
    config: PredictionConfig,
    nn_config: NeuralNetworkConfig,
    fe_config: FeatureEngineeringConfig,
    models: Arc<RwLock<ModelRegistry>>,
    storage: Arc<SliceStorage>,
    metrics: Arc<SliceMetricsCollector>,
    prediction_cache: Arc<RwLock<HashMap<String, CachedPrediction>>>,
    active_predictions: Arc<Mutex<HashMap<String, PredictionTask>>>,
    task_sender: Option<mpsc::UnboundedSender<PredictionRequest>>,
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPrediction {
    pub slice_id: String,
    pub predictions: Vec<SlaBreachPrediction>,
    pub timestamp: DateTime<Utc>,
    pub ttl: Duration,
}

#[derive(Debug, Clone)]
pub struct PredictionTask {
    pub slice_id: String,
    pub request_id: String,
    pub started_at: DateTime<Utc>,
    pub horizon_minutes: i32,
}

#[derive(Debug, Clone)]
pub struct PredictionRequest {
    pub slice_id: String,
    pub request_id: String,
    pub current_metrics: SliceMetrics,
    pub historical_metrics: Vec<SliceMetrics>,
    pub sla_definition: SlaDefinition,
    pub config: Option<PredictionConfig>,
    pub response_sender: tokio::sync::oneshot::Sender<SliceResult<PredictSlaBreachResponse>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub model_id: String,
    pub predictions: Vec<f64>,
    pub confidence: f64,
    pub feature_importance: Option<Vec<f64>>,
    pub uncertainty: Option<Vec<f64>>,
}

/// Prediction engine that orchestrates multiple models for ensemble prediction
pub struct PredictionEngine {
    models: HashMap<String, Arc<RwLock<Network>>>,
    feature_extractors: HashMap<String, FeatureExtractor>,
    ensemble_weights: HashMap<String, f64>,
}

impl SlaBreachPredictor {
    pub async fn new(
        config: PredictionConfig,
        storage: Arc<SliceStorage>,
        metrics: Arc<SliceMetricsCollector>,
    ) -> SliceResult<Self> {
        info!("Initializing SLA Breach Predictor");
        
        let nn_config = NeuralNetworkConfig::default();
        let fe_config = FeatureEngineeringConfig::default();
        
        let models = Arc::new(RwLock::new(ModelRegistry::new()));
        let prediction_cache = Arc::new(RwLock::new(HashMap::new()));
        let active_predictions = Arc::new(Mutex::new(HashMap::new()));
        
        Ok(Self {
            config,
            nn_config,
            fe_config,
            models,
            storage,
            metrics,
            prediction_cache,
            active_predictions,
            task_sender: None,
            task_handle: None,
        })
    }
    
    pub async fn start_prediction_loop(&mut self) -> SliceResult<()> {
        info!("Starting prediction loop");
        
        let (sender, mut receiver) = mpsc::unbounded_channel::<PredictionRequest>();
        self.task_sender = Some(sender);
        
        let models = Arc::clone(&self.models);
        let storage = Arc::clone(&self.storage);
        let metrics = Arc::clone(&self.metrics);
        let active_predictions = Arc::clone(&self.active_predictions);
        let prediction_cache = Arc::clone(&self.prediction_cache);
        let config = self.config.clone();
        let nn_config = self.nn_config.clone();
        let fe_config = self.fe_config.clone();
        
        let handle = tokio::spawn(async move {
            info!("Prediction loop started");
            
            while let Some(request) = receiver.recv().await {
                let start_time = std::time::Instant::now();
                
                // Register active prediction
                {
                    let mut active = active_predictions.lock().await;
                    active.insert(request.request_id.clone(), PredictionTask {
                        slice_id: request.slice_id.clone(),
                        request_id: request.request_id.clone(),
                        started_at: Utc::now(),
                        horizon_minutes: config.default_horizon_minutes,
                    });
                }
                
                // Process prediction request
                let result = Self::process_prediction_request(
                    request.clone(),
                    &models,
                    &storage,
                    &metrics,
                    &prediction_cache,
                    &config,
                    &nn_config,
                    &fe_config,
                ).await;
                
                // Remove from active predictions
                {
                    let mut active = active_predictions.lock().await;
                    active.remove(&request.request_id);
                }
                
                // Record metrics
                metrics.record_prediction_latency(start_time.elapsed()).await;
                if result.is_ok() {
                    metrics.increment_predictions_success().await;
                } else {
                    metrics.increment_predictions_error().await;
                }
                
                // Send response
                if let Err(_) = request.response_sender.send(result) {
                    warn!("Failed to send prediction response for request {}", request.request_id);
                }
            }
            
            info!("Prediction loop ended");
        });
        
        self.task_handle = Some(handle);
        Ok(())
    }
    
    pub async fn stop_prediction_loop(&mut self) -> SliceResult<()> {
        info!("Stopping prediction loop");
        
        if let Some(sender) = self.task_sender.take() {
            drop(sender);
        }
        
        if let Some(handle) = self.task_handle.take() {
            handle.await.map_err(|e| SliceError::TokioJoin(e))?;
        }
        
        info!("Prediction loop stopped");
        Ok(())
    }
    
    pub async fn predict_sla_breach(
        &self,
        slice_id: String,
        current_metrics: SliceMetrics,
        historical_metrics: Vec<SliceMetrics>,
        sla_definition: SlaDefinition,
        config: Option<PredictionConfig>,
    ) -> SliceResult<PredictSlaBreachResponse> {
        let request_id = Uuid::new_v4().to_string();
        
        // Check cache first
        if self.config.cache_predictions {
            if let Some(cached) = self.get_cached_prediction(&slice_id).await? {
                debug!("Returning cached prediction for slice {}", slice_id);
                return Ok(PredictSlaBreachResponse {
                    slice_id: slice_id.clone(),
                    breach_predictions: cached.predictions,
                    risk_analysis: self.analyze_risk(&slice_id, &current_metrics, &sla_definition).await?,
                    recommendations: self.generate_recommendations(&slice_id, &current_metrics, &sla_definition).await?,
                    model_confidence: 0.9, // TODO: Calculate from cached data
                    model_version: "v1.0.0".to_string(),
                    prediction_timestamp: Some(prost_types::Timestamp {
                        seconds: cached.timestamp.timestamp(),
                        nanos: cached.timestamp.timestamp_subsec_nanos() as i32,
                    }),
                    metadata: None,
                });
            }
        }
        
        // Create prediction request
        let (response_sender, response_receiver) = tokio::sync::oneshot::channel();
        
        let request = PredictionRequest {
            slice_id: slice_id.clone(),
            request_id: request_id.clone(),
            current_metrics,
            historical_metrics,
            sla_definition,
            config: config.clone(),
            response_sender,
        };
        
        // Send request to background task
        if let Some(sender) = &self.task_sender {
            sender.send(request).map_err(|_| SliceError::internal("Prediction service unavailable"))?;
        } else {
            return Err(SliceError::service_unavailable("Prediction loop not started"));
        }
        
        // Wait for response
        let timeout = tokio::time::Duration::from_secs(self.config.request_timeout_seconds);
        let result = tokio::time::timeout(timeout, response_receiver).await
            .map_err(|_| SliceError::timeout(format!("Prediction timeout for slice {}", slice_id)))?
            .map_err(|_| SliceError::internal("Failed to receive prediction response"))?;
        
        result
    }
    
    async fn process_prediction_request(
        request: PredictionRequest,
        models: &Arc<RwLock<ModelRegistry>>,
        storage: &Arc<SliceStorage>,
        metrics: &Arc<SliceMetricsCollector>,
        prediction_cache: &Arc<RwLock<HashMap<String, CachedPrediction>>>,
        config: &PredictionConfig,
        nn_config: &NeuralNetworkConfig,
        fe_config: &FeatureEngineeringConfig,
    ) -> SliceResult<PredictSlaBreachResponse> {
        debug!("Processing prediction request for slice {}", request.slice_id);
        
        // Extract features from metrics
        let feature_vector = Self::extract_features(
            &request.current_metrics,
            &request.historical_metrics,
            fe_config,
        )?;
        
        // Get or create models for the slice
        let slice_models = Self::get_or_create_models(
            &request.slice_id,
            &request.sla_definition,
            models,
            storage,
            nn_config,
        ).await?;
        
        // Run ensemble predictions
        let model_predictions = Self::run_ensemble_predictions(
            &feature_vector,
            &slice_models,
            config,
        ).await?;
        
        // Aggregate predictions
        let breach_predictions = Self::aggregate_predictions(
            &model_predictions,
            &request.sla_definition,
            config,
        )?;
        
        // Generate risk analysis
        let risk_analysis = Self::analyze_risk_detailed(
            &request.slice_id,
            &request.current_metrics,
            &request.sla_definition,
            &breach_predictions,
        ).await?;
        
        // Generate recommendations
        let recommendations = Self::generate_recommendations_detailed(
            &request.slice_id,
            &request.current_metrics,
            &request.sla_definition,
            &breach_predictions,
            &risk_analysis,
        ).await?;
        
        // Calculate overall model confidence
        let model_confidence = model_predictions.iter()
            .map(|p| p.confidence)
            .sum::<f64>() / model_predictions.len() as f64;
        
        let response = PredictSlaBreachResponse {
            slice_id: request.slice_id.clone(),
            breach_predictions,
            risk_analysis: Some(risk_analysis),
            recommendations,
            model_confidence,
            model_version: "v1.0.0".to_string(),
            prediction_timestamp: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(),
                nanos: Utc::now().timestamp_subsec_nanos() as i32,
            }),
            metadata: None,
        };
        
        // Cache the prediction if enabled
        if config.cache_predictions {
            let cached = CachedPrediction {
                slice_id: request.slice_id.clone(),
                predictions: response.breach_predictions.clone(),
                timestamp: Utc::now(),
                ttl: Duration::seconds(config.cache_ttl_seconds as i64),
            };
            
            let mut cache = prediction_cache.write().await;
            cache.insert(request.slice_id, cached);
        }
        
        debug!("Prediction completed for slice {}", request.slice_id);
        Ok(response)
    }
    
    fn extract_features(
        current_metrics: &SliceMetrics,
        historical_metrics: &[SliceMetrics],
        config: &FeatureEngineeringConfig,
    ) -> SliceResult<FeatureVector> {
        let mut features = Vec::new();
        let mut feature_names = Vec::new();
        
        // Current metrics features
        features.extend_from_slice(&[
            current_metrics.prb_usage_percent,
            current_metrics.throughput_mbps,
            current_metrics.pdu_session_count as f64,
            current_metrics.latency_ms,
            current_metrics.jitter_ms,
            current_metrics.packet_loss_percent,
            current_metrics.availability_percent,
            current_metrics.resource_utilization_percent,
            current_metrics.active_users as f64,
            current_metrics.cpu_usage_percent,
            current_metrics.memory_usage_percent,
        ]);
        
        feature_names.extend_from_slice(&[
            "prb_usage_current",
            "throughput_current",
            "pdu_sessions_current",
            "latency_current",
            "jitter_current",
            "packet_loss_current",
            "availability_current",
            "resource_util_current",
            "active_users_current",
            "cpu_usage_current",
            "memory_usage_current",
        ].map(|s| s.to_string()));
        
        // Historical features (lag features, rolling statistics)
        if !historical_metrics.is_empty() {
            let throughput_series: Vec<f64> = historical_metrics.iter()
                .map(|m| m.throughput_mbps)
                .collect();
            
            let latency_series: Vec<f64> = historical_metrics.iter()
                .map(|m| m.latency_ms)
                .collect();
            
            let prb_series: Vec<f64> = historical_metrics.iter()
                .map(|m| m.prb_usage_percent)
                .collect();
            
            // Lag features
            for &lag in &config.lag_features {
                let lag_index = lag as usize;
                if lag_index < throughput_series.len() {
                    features.push(throughput_series[throughput_series.len() - 1 - lag_index]);
                    feature_names.push(format!("throughput_lag_{}", lag));
                    
                    features.push(latency_series[latency_series.len() - 1 - lag_index]);
                    feature_names.push(format!("latency_lag_{}", lag));
                    
                    features.push(prb_series[prb_series.len() - 1 - lag_index]);
                    feature_names.push(format!("prb_usage_lag_{}", lag));
                }
            }
            
            // Rolling window statistics
            for &window_size in &config.rolling_window_sizes {
                let window_size = window_size as usize;
                if window_size <= throughput_series.len() {
                    let window_start = throughput_series.len() - window_size;
                    
                    // Throughput statistics
                    let throughput_window = &throughput_series[window_start..];
                    let throughput_mean = throughput_window.iter().sum::<f64>() / window_size as f64;
                    let throughput_std = {
                        let variance = throughput_window.iter()
                            .map(|x| (x - throughput_mean).powi(2))
                            .sum::<f64>() / window_size as f64;
                        variance.sqrt()
                    };
                    let throughput_min = throughput_window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let throughput_max = throughput_window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    
                    features.extend_from_slice(&[throughput_mean, throughput_std, throughput_min, throughput_max]);
                    feature_names.extend_from_slice(&[
                        format!("throughput_mean_{}", window_size),
                        format!("throughput_std_{}", window_size),
                        format!("throughput_min_{}", window_size),
                        format!("throughput_max_{}", window_size),
                    ]);
                    
                    // Latency statistics
                    let latency_window = &latency_series[window_start..];
                    let latency_mean = latency_window.iter().sum::<f64>() / window_size as f64;
                    let latency_std = {
                        let variance = latency_window.iter()
                            .map(|x| (x - latency_mean).powi(2))
                            .sum::<f64>() / window_size as f64;
                        variance.sqrt()
                    };
                    
                    features.extend_from_slice(&[latency_mean, latency_std]);
                    feature_names.extend_from_slice(&[
                        format!("latency_mean_{}", window_size),
                        format!("latency_std_{}", window_size),
                    ]);
                }
            }
        }
        
        // Time-based features
        if let Some(timestamp) = current_metrics.timestamp.as_ref() {
            let dt = chrono::DateTime::<Utc>::from_timestamp(
                timestamp.seconds,
                timestamp.nanos as u32
            ).unwrap_or_else(|| Utc::now());
            
            let hour_of_day = dt.hour() as f64;
            let day_of_week = dt.weekday().number_from_monday() as f64;
            let is_weekend = if dt.weekday().number_from_monday() >= 6 { 1.0 } else { 0.0 };
            
            features.extend_from_slice(&[hour_of_day, day_of_week, is_weekend]);
            feature_names.extend_from_slice(&[
                "hour_of_day".to_string(),
                "day_of_week".to_string(),
                "is_weekend".to_string(),
            ]);
        }
        
        Ok(FeatureVector {
            features,
            feature_names,
            timestamp: Utc::now(),
        })
    }
    
    async fn get_or_create_models(
        slice_id: &str,
        sla_definition: &SlaDefinition,
        models: &Arc<RwLock<ModelRegistry>>,
        storage: &Arc<SliceStorage>,
        config: &NeuralNetworkConfig,
    ) -> SliceResult<Vec<SliceModel>> {
        let model_registry = models.read().await;
        
        // Try to get existing models for this slice
        if let Some(slice_models) = model_registry.get_models_for_slice(slice_id) {
            return Ok(slice_models);
        }
        
        drop(model_registry);
        
        // Need to create new models
        let mut model_registry = models.write().await;
        
        // Double-check after acquiring write lock
        if let Some(slice_models) = model_registry.get_models_for_slice(slice_id) {
            return Ok(slice_models);
        }
        
        info!("Creating new models for slice {}", slice_id);
        
        let slice_models = Self::create_models_for_slice(slice_id, sla_definition, config).await?;
        model_registry.register_models(slice_id, slice_models.clone());
        
        Ok(slice_models)
    }
    
    async fn create_models_for_slice(
        slice_id: &str,
        sla_definition: &SlaDefinition,
        config: &NeuralNetworkConfig,
    ) -> SliceResult<Vec<SliceModel>> {
        let mut models = Vec::new();
        
        // Create models for different SLA metrics
        let metrics_to_predict = vec![
            "THROUGHPUT",
            "LATENCY", 
            "JITTER",
            "PACKET_LOSS",
            "AVAILABILITY"
        ];
        
        for metric in metrics_to_predict {
            let model = Self::create_model_for_metric(slice_id, metric, config).await?;
            models.push(model);
        }
        
        Ok(models)
    }
    
    async fn create_model_for_metric(
        slice_id: &str,
        metric: &str,
        config: &NeuralNetworkConfig,
    ) -> SliceResult<SliceModel> {
        info!("Creating model for slice {} metric {}", slice_id, metric);
        
        // Calculate input and output sizes
        let input_size = Self::calculate_input_size(config);
        let output_size = 1; // Predicting breach probability
        
        // Create network architecture
        let mut layers = vec![input_size];
        layers.extend(&config.hidden_layers);
        layers.push(output_size);
        
        // Create FANN network
        let mut network = Network::new(&layers)
            .map_err(|e| SliceError::neural_network(format!("Failed to create network: {:?}", e)))?;
        
        // Configure activation functions
        let activation_fn = match config.activation_function.as_str() {
            "SIGMOID" => ActivationFunction::Sigmoid,
            "TANH" => ActivationFunction::Tanh,
            "RELU" => ActivationFunction::Linear, // FANN doesn't have ReLU, use linear
            _ => ActivationFunction::Sigmoid,
        };
        
        network.set_activation_function_hidden(activation_fn);
        network.set_activation_function_output(activation_fn);
        
        // Configure training parameters
        network.set_learning_rate(config.learning_rate as f32);
        
        let training_algorithm = match config.optimizer.as_str() {
            "RPROP" => TrainingAlgorithm::Rprop,
            "QUICKPROP" => TrainingAlgorithm::Quickprop,
            _ => TrainingAlgorithm::Rprop,
        };
        network.set_training_algorithm(training_algorithm);
        
        // Create slice model
        let model = SliceModel::new(
            format!("{}_{}", slice_id, metric),
            slice_id.to_string(),
            metric.to_string(),
            network,
            config.clone(),
        );
        
        Ok(model)
    }
    
    fn calculate_input_size(config: &NeuralNetworkConfig) -> usize {
        // Base features: 11 current metrics
        let mut size = 11;
        
        // Lag features: 3 metrics * number of lags
        size += 3 * 5; // Default lag features: [1, 2, 3, 5, 10]
        
        // Rolling window features: 2 metrics * 4 windows * 4 statistics
        size += 2 * 4 * 4; // throughput and latency, 4 window sizes, 4 stats each
        
        // Time features: 3 (hour, day_of_week, is_weekend)
        size += 3;
        
        size
    }
    
    async fn run_ensemble_predictions(
        feature_vector: &FeatureVector,
        models: &[SliceModel],
        config: &PredictionConfig,
    ) -> SliceResult<Vec<ModelPrediction>> {
        let mut predictions = Vec::new();
        
        for model in models {
            let prediction = Self::run_single_model_prediction(feature_vector, model, config).await?;
            predictions.push(prediction);
        }
        
        Ok(predictions)
    }
    
    async fn run_single_model_prediction(
        feature_vector: &FeatureVector,
        model: &SliceModel,
        config: &PredictionConfig,
    ) -> SliceResult<ModelPrediction> {
        let network = model.get_network().await;
        
        // Prepare input
        let input: Vec<f32> = feature_vector.features.iter()
            .map(|&x| x as f32)
            .collect();
        
        // Run prediction
        let output = network.run(&input)
            .map_err(|e| SliceError::neural_network(format!("Prediction failed: {:?}", e)))?;
        
        // Convert output to f64
        let predictions: Vec<f64> = output.iter().map(|&x| x as f64).collect();
        
        // Calculate confidence (simplified)
        let confidence = predictions[0].max(0.0).min(1.0);
        
        Ok(ModelPrediction {
            model_id: model.id().clone(),
            predictions,
            confidence,
            feature_importance: None, // TODO: Implement feature importance
            uncertainty: None, // TODO: Implement uncertainty quantification
        })
    }
    
    fn aggregate_predictions(
        model_predictions: &[ModelPrediction],
        sla_definition: &SlaDefinition,
        config: &PredictionConfig,
    ) -> SliceResult<Vec<SlaBreachPrediction>> {
        let mut breach_predictions = Vec::new();
        
        // Group predictions by metric type
        let mut metric_predictions: HashMap<String, Vec<&ModelPrediction>> = HashMap::new();
        
        for prediction in model_predictions {
            let metric_type = Self::extract_metric_type_from_model_id(&prediction.model_id);
            metric_predictions.entry(metric_type).or_insert_with(Vec::new).push(prediction);
        }
        
        // Aggregate predictions for each metric
        for (metric_type, predictions) in metric_predictions {
            if predictions.is_empty() {
                continue;
            }
            
            // Ensemble voting - average the predictions
            let avg_prediction = predictions.iter()
                .map(|p| p.predictions[0])
                .sum::<f64>() / predictions.len() as f64;
            
            let avg_confidence = predictions.iter()
                .map(|p| p.confidence)
                .sum::<f64>() / predictions.len() as f64;
            
            // Get SLA threshold for this metric
            let sla_threshold = Self::get_sla_threshold_for_metric(&metric_type, sla_definition);
            
            // Calculate breach time (simplified)
            let predicted_breach_time = Utc::now() + Duration::minutes(config.default_horizon_minutes as i64);
            
            // Calculate severity
            let severity_score = avg_prediction * avg_confidence;
            
            // Determine contributing factors (simplified)
            let contributing_factors = vec![
                format!("High {} utilization", metric_type.to_lowercase()),
                "Resource contention".to_string(),
            ];
            
            let breach_prediction = SlaBreachPrediction {
                metric_type: metric_type.clone(),
                breach_probability: avg_prediction,
                predicted_breach_time: Some(prost_types::Timestamp {
                    seconds: predicted_breach_time.timestamp(),
                    nanos: predicted_breach_time.timestamp_subsec_nanos() as i32,
                }),
                predicted_value: avg_prediction * sla_threshold,
                sla_threshold,
                confidence_lower: avg_prediction - 0.1,
                confidence_upper: avg_prediction + 0.1,
                severity_score,
                breach_type: if avg_prediction > 0.8 { "SUDDEN".to_string() } else { "GRADUAL".to_string() },
                contributing_factors,
            };
            
            breach_predictions.push(breach_prediction);
        }
        
        Ok(breach_predictions)
    }
    
    fn extract_metric_type_from_model_id(model_id: &str) -> String {
        if let Some(last_underscore) = model_id.rfind('_') {
            model_id[last_underscore + 1..].to_string()
        } else {
            "UNKNOWN".to_string()
        }
    }
    
    fn get_sla_threshold_for_metric(metric_type: &str, sla_definition: &SlaDefinition) -> f64 {
        match metric_type {
            "THROUGHPUT" => sla_definition.guaranteed_throughput_mbps,
            "LATENCY" => sla_definition.max_latency_ms,
            "JITTER" => sla_definition.max_jitter_ms,
            "PACKET_LOSS" => sla_definition.max_packet_loss_percent,
            "AVAILABILITY" => sla_definition.min_availability_percent,
            _ => 1.0,
        }
    }
    
    async fn analyze_risk_detailed(
        slice_id: &str,
        current_metrics: &SliceMetrics,
        sla_definition: &SlaDefinition,
        breach_predictions: &[SlaBreachPrediction],
    ) -> SliceResult<SliceRiskAnalysis> {
        // Calculate overall risk score
        let overall_risk_score = breach_predictions.iter()
            .map(|p| p.breach_probability * p.severity_score)
            .sum::<f64>() / breach_predictions.len().max(1) as f64;
        
        // Determine risk level
        let risk_level = if overall_risk_score >= 0.8 {
            "CRITICAL"
        } else if overall_risk_score >= 0.6 {
            "HIGH"
        } else if overall_risk_score >= 0.4 {
            "MEDIUM"
        } else {
            "LOW"
        };
        
        // Calculate component scores
        let resource_contention_score = (current_metrics.cpu_usage_percent + 
                                       current_metrics.memory_usage_percent + 
                                       current_metrics.resource_utilization_percent) / 300.0;
        
        let demand_volatility_score = 0.5; // TODO: Calculate from historical data
        let historical_stability_score = 0.8; // TODO: Calculate from historical data
        let network_congestion_score = current_metrics.prb_usage_percent / 100.0;
        
        // Generate risk factors
        let mut risk_factors = Vec::new();
        
        if resource_contention_score > 0.7 {
            risk_factors.push(RiskFactor {
                factor_type: "RESOURCE_CONTENTION".to_string(),
                description: "High resource utilization detected".to_string(),
                impact_score: resource_contention_score,
                probability: 0.8,
                mitigation_strategy: "Scale resources or load balance".to_string(),
            });
        }
        
        if network_congestion_score > 0.8 {
            risk_factors.push(RiskFactor {
                factor_type: "NETWORK_CONGESTION".to_string(),
                description: "High PRB utilization".to_string(),
                impact_score: network_congestion_score,
                probability: 0.9,
                mitigation_strategy: "Optimize resource allocation".to_string(),
            });
        }
        
        Ok(SliceRiskAnalysis {
            slice_id: slice_id.to_string(),
            overall_risk_score,
            risk_level: risk_level.to_string(),
            risk_factors,
            resource_contention_score,
            demand_volatility_score,
            historical_stability_score,
            network_congestion_score,
            trend_analysis: "STABLE".to_string(), // TODO: Implement trend analysis
            analysis_timestamp: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(),
                nanos: Utc::now().timestamp_subsec_nanos() as i32,
            }),
        })
    }
    
    async fn generate_recommendations_detailed(
        slice_id: &str,
        current_metrics: &SliceMetrics,
        sla_definition: &SlaDefinition,
        breach_predictions: &[SlaBreachPrediction],
        risk_analysis: &SliceRiskAnalysis,
    ) -> SliceResult<Vec<SliceRecommendation>> {
        let mut recommendations = Vec::new();
        
        // High-risk throughput breach
        if breach_predictions.iter().any(|p| p.metric_type == "THROUGHPUT" && p.breach_probability > 0.7) {
            recommendations.push(SliceRecommendation {
                recommendation_id: Uuid::new_v4().to_string(),
                recommendation_type: "RESOURCE_ALLOCATION".to_string(),
                title: "Increase Bandwidth Allocation".to_string(),
                description: "Allocate additional bandwidth to prevent throughput SLA breach".to_string(),
                effectiveness_score: 0.85,
                implementation_cost: 1500.0,
                roi_estimate: 2.3,
                priority: "HIGH".to_string(),
                complexity: "MODERATE".to_string(),
                prerequisites: vec!["Available bandwidth".to_string()],
                implementation_guide: "Contact resource management to increase allocation".to_string(),
                recommendation_timestamp: Some(prost_types::Timestamp {
                    seconds: Utc::now().timestamp(),
                    nanos: Utc::now().timestamp_subsec_nanos() as i32,
                }),
                estimated_implementation_time_minutes: 30,
            });
        }
        
        // High resource contention
        if risk_analysis.resource_contention_score > 0.7 {
            recommendations.push(SliceRecommendation {
                recommendation_id: Uuid::new_v4().to_string(),
                recommendation_type: "LOAD_BALANCING".to_string(),
                title: "Implement Load Balancing".to_string(),
                description: "Distribute load across multiple resources to reduce contention".to_string(),
                effectiveness_score: 0.75,
                implementation_cost: 2000.0,
                roi_estimate: 1.8,
                priority: "MEDIUM".to_string(),
                complexity: "COMPLEX".to_string(),
                prerequisites: vec!["Load balancer availability".to_string()],
                implementation_guide: "Configure load balancing policies".to_string(),
                recommendation_timestamp: Some(prost_types::Timestamp {
                    seconds: Utc::now().timestamp(),
                    nanos: Utc::now().timestamp_subsec_nanos() as i32,
                }),
                estimated_implementation_time_minutes: 120,
            });
        }
        
        Ok(recommendations)
    }
    
    async fn get_cached_prediction(&self, slice_id: &str) -> SliceResult<Option<CachedPrediction>> {
        let cache = self.prediction_cache.read().await;
        
        if let Some(cached) = cache.get(slice_id) {
            // Check if cache is still valid
            if Utc::now() - cached.timestamp < cached.ttl {
                return Ok(Some(cached.clone()));
            }
        }
        
        Ok(None)
    }
    
    async fn analyze_risk(&self, slice_id: &str, current_metrics: &SliceMetrics, sla_definition: &SlaDefinition) -> SliceResult<Option<SliceRiskAnalysis>> {
        // Simplified risk analysis for cached responses
        let overall_risk_score = (current_metrics.resource_utilization_percent + 
                                current_metrics.prb_usage_percent) / 200.0;
        
        let risk_level = if overall_risk_score >= 0.8 { "HIGH" } else { "MEDIUM" };
        
        Ok(Some(SliceRiskAnalysis {
            slice_id: slice_id.to_string(),
            overall_risk_score,
            risk_level: risk_level.to_string(),
            risk_factors: vec![],
            resource_contention_score: current_metrics.resource_utilization_percent / 100.0,
            demand_volatility_score: 0.5,
            historical_stability_score: 0.8,
            network_congestion_score: current_metrics.prb_usage_percent / 100.0,
            trend_analysis: "STABLE".to_string(),
            analysis_timestamp: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(),
                nanos: Utc::now().timestamp_subsec_nanos() as i32,
            }),
        }))
    }
    
    async fn generate_recommendations(&self, slice_id: &str, current_metrics: &SliceMetrics, sla_definition: &SlaDefinition) -> SliceResult<Vec<SliceRecommendation>> {
        // Simplified recommendations for cached responses
        Ok(vec![
            SliceRecommendation {
                recommendation_id: Uuid::new_v4().to_string(),
                recommendation_type: "MONITORING".to_string(),
                title: "Continue Monitoring".to_string(),
                description: "Monitor slice performance for changes".to_string(),
                effectiveness_score: 0.6,
                implementation_cost: 0.0,
                roi_estimate: 1.0,
                priority: "LOW".to_string(),
                complexity: "SIMPLE".to_string(),
                prerequisites: vec![],
                implementation_guide: "No action required".to_string(),
                recommendation_timestamp: Some(prost_types::Timestamp {
                    seconds: Utc::now().timestamp(),
                    nanos: Utc::now().timestamp_subsec_nanos() as i32,
                }),
                estimated_implementation_time_minutes: 0,
            }
        ])
    }
}

/// Feature extractor for different types of features
pub struct FeatureExtractor {
    pub extractor_type: String,
    pub config: FeatureEngineeringConfig,
}

impl FeatureExtractor {
    pub fn new(extractor_type: String, config: FeatureEngineeringConfig) -> Self {
        Self {
            extractor_type,
            config,
        }
    }
    
    pub fn extract(&self, metrics: &[SliceMetrics]) -> SliceResult<FeatureVector> {
        match self.extractor_type.as_str() {
            "TIME_SERIES" => self.extract_time_series_features(metrics),
            "STATISTICAL" => self.extract_statistical_features(metrics),
            "TEMPORAL" => self.extract_temporal_features(metrics),
            _ => Err(SliceError::feature_engineering(format!("Unknown extractor type: {}", self.extractor_type))),
        }
    }
    
    fn extract_time_series_features(&self, metrics: &[SliceMetrics]) -> SliceResult<FeatureVector> {
        // Implementation for time series feature extraction
        let mut features = Vec::new();
        let mut feature_names = Vec::new();
        
        if let Some(latest) = metrics.last() {
            features.push(latest.throughput_mbps);
            feature_names.push("throughput".to_string());
        }
        
        Ok(FeatureVector {
            features,
            feature_names,
            timestamp: Utc::now(),
        })
    }
    
    fn extract_statistical_features(&self, metrics: &[SliceMetrics]) -> SliceResult<FeatureVector> {
        // Implementation for statistical feature extraction
        let mut features = Vec::new();
        let mut feature_names = Vec::new();
        
        if !metrics.is_empty() {
            let throughput_values: Vec<f64> = metrics.iter().map(|m| m.throughput_mbps).collect();
            let mean = throughput_values.iter().sum::<f64>() / throughput_values.len() as f64;
            let variance = throughput_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / throughput_values.len() as f64;
            
            features.extend_from_slice(&[mean, variance.sqrt()]);
            feature_names.extend_from_slice(&["throughput_mean".to_string(), "throughput_std".to_string()]);
        }
        
        Ok(FeatureVector {
            features,
            feature_names,
            timestamp: Utc::now(),
        })
    }
    
    fn extract_temporal_features(&self, metrics: &[SliceMetrics]) -> SliceResult<FeatureVector> {
        // Implementation for temporal feature extraction
        let mut features = Vec::new();
        let mut feature_names = Vec::new();
        
        if let Some(latest) = metrics.last() {
            if let Some(timestamp) = &latest.timestamp {
                let dt = chrono::DateTime::<Utc>::from_timestamp(
                    timestamp.seconds,
                    timestamp.nanos as u32
                ).unwrap_or_else(|| Utc::now());
                
                features.push(dt.hour() as f64);
                feature_names.push("hour_of_day".to_string());
            }
        }
        
        Ok(FeatureVector {
            features,
            feature_names,
            timestamp: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_predictor_creation() {
        let config = PredictionConfig::default();
        let storage = Arc::new(SliceStorage::new(&StorageConfig::default()).await.unwrap());
        let metrics = Arc::new(SliceMetricsCollector::new());
        
        let predictor = SlaBreachPredictor::new(config, storage, metrics).await;
        assert!(predictor.is_ok());
    }
    
    #[test]
    fn test_feature_extraction() {
        let config = FeatureEngineeringConfig::default();
        let current_metrics = SliceMetrics {
            slice_id: "test".to_string(),
            prb_usage_percent: 70.0,
            throughput_mbps: 100.0,
            pdu_session_count: 50,
            latency_ms: 20.0,
            jitter_ms: 5.0,
            packet_loss_percent: 0.1,
            availability_percent: 99.9,
            resource_utilization_percent: 60.0,
            active_users: 100,
            cpu_usage_percent: 50.0,
            memory_usage_percent: 40.0,
            custom_metrics: HashMap::new(),
            timestamp: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(),
                nanos: 0,
            }),
        };
        
        let historical_metrics = vec![current_metrics.clone()];
        
        let result = SlaBreachPredictor::extract_features(
            &current_metrics,
            &historical_metrics,
            &config,
        );
        
        assert!(result.is_ok());
        let features = result.unwrap();
        assert!(!features.features.is_empty());
        assert_eq!(features.features.len(), features.feature_names.len());
    }
}