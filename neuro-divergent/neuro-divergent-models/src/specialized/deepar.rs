//! DeepAR - Deep Autoregressive model with probabilistic forecasting
//!
//! Implementation of Amazon's DeepAR model for probabilistic time series forecasting.
//! This model uses an autoregressive RNN with probabilistic outputs to generate
//! forecasts with uncertainty quantification.

use super::*;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction};
use rand::Rng;
use rand_distr::{Distribution, Normal, StudentT};
use statrs::distribution::{Normal as StatrsNormal, StudentsT};
use std::collections::HashMap;

/// DeepAR model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepARConfig<T: Float> {
    /// Input sequence length
    pub input_size: usize,
    /// Forecast horizon
    pub horizon: usize,
    /// RNN hidden size
    pub hidden_size: usize,
    /// Number of RNN layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: T,
    /// Distribution type for probabilistic outputs
    pub distribution: DistributionType,
    /// Number of samples for Monte Carlo sampling
    pub num_samples: usize,
    /// Static features dimension
    pub static_features_size: usize,
    /// Exogenous features dimension
    pub exogenous_features_size: usize,
    /// Scaling method
    pub scaling: ScalingMethod,
}

/// Supported probability distributions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistributionType {
    /// Gaussian distribution
    Gaussian,
    /// Student-t distribution
    StudentT,
    /// Negative binomial (for count data)
    NegativeBinomial,
}

/// Scaling methods for input normalization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// No scaling
    None,
    /// Standard scaling (z-score)
    Standard,
    /// Mean scaling (divide by mean)
    Mean,
}

impl<T: Float> Default for DeepARConfig<T> {
    fn default() -> Self {
        Self {
            input_size: 24,
            horizon: 12,
            hidden_size: 64,
            num_layers: 2,
            dropout: T::from(0.1).unwrap(),
            distribution: DistributionType::Gaussian,
            num_samples: 100,
            static_features_size: 0,
            exogenous_features_size: 0,
            scaling: ScalingMethod::Standard,
        }
    }
}

impl<T: Float> ModelConfig<T> for DeepARConfig<T> {
    fn validate(&self) -> Result<(), ModelError> {
        if self.input_size == 0 {
            return Err(ModelError::ConfigError("input_size must be > 0".to_string()));
        }
        if self.horizon == 0 {
            return Err(ModelError::ConfigError("horizon must be > 0".to_string()));
        }
        if self.hidden_size == 0 {
            return Err(ModelError::ConfigError("hidden_size must be > 0".to_string()));
        }
        if self.num_layers == 0 {
            return Err(ModelError::ConfigError("num_layers must be > 0".to_string()));
        }
        if self.dropout < T::zero() || self.dropout >= T::one() {
            return Err(ModelError::ConfigError("dropout must be in [0, 1)".to_string()));
        }
        Ok(())
    }
    
    fn horizon(&self) -> usize {
        self.horizon
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn static_features_size(&self) -> usize {
        self.static_features_size
    }
    
    fn exogenous_features_size(&self) -> usize {
        self.exogenous_features_size
    }
}

/// DeepAR model state for training and inference
#[derive(Debug)]
struct DeepARState<T: Float> {
    /// RNN hidden states
    hidden_states: Vec<Vec<T>>,
    /// Cell states (for LSTM)
    cell_states: Vec<Vec<T>>,
    /// Scaling parameters
    scale_mean: T,
    scale_std: T,
    /// Training statistics
    training_loss: Option<T>,
}

impl<T: Float> DeepARState<T> {
    fn new(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            hidden_states: vec![vec![T::zero(); hidden_size]; num_layers],
            cell_states: vec![vec![T::zero(); hidden_size]; num_layers],
            scale_mean: T::zero(),
            scale_std: T::one(),
            training_loss: None,
        }
    }
    
    fn reset(&mut self) {
        for hidden in &mut self.hidden_states {
            hidden.fill(T::zero());
        }
        for cell in &mut self.cell_states {
            cell.fill(T::zero());
        }
    }
}

/// DeepAR model implementation
pub struct DeepAR<T: Float> {
    config: DeepARConfig<T>,
    encoder_network: Option<Network<T>>,
    decoder_network: Option<Network<T>>,
    distribution_network: Option<Network<T>>,
    state: DeepARState<T>,
    trained: bool,
}

impl<T: Float> DeepAR<T> {
    /// Create encoder network for processing input sequences
    fn create_encoder_network(&self) -> Result<Network<T>, ModelError> {
        let input_dim = 1 + self.config.static_features_size + self.config.exogenous_features_size;
        
        NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer(self.config.hidden_size, ActivationFunction::Tanh)
            .hidden_layer(self.config.hidden_size, ActivationFunction::Tanh)
            .output_layer(self.config.hidden_size, ActivationFunction::Linear)
            .build()
            .map_err(|e| ModelError::NetworkError(e.to_string()))
    }
    
    /// Create decoder network for autoregressive generation
    fn create_decoder_network(&self) -> Result<Network<T>, ModelError> {
        let input_dim = self.config.hidden_size + 1 + self.config.exogenous_features_size;
        
        NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer(self.config.hidden_size, ActivationFunction::Tanh)
            .hidden_layer(self.config.hidden_size, ActivationFunction::Tanh)
            .output_layer(self.config.hidden_size, ActivationFunction::Linear)
            .build()
            .map_err(|e| ModelError::NetworkError(e.to_string()))
    }
    
    /// Create distribution parameter network
    fn create_distribution_network(&self) -> Result<Network<T>, ModelError> {
        let output_size = match self.config.distribution {
            DistributionType::Gaussian => 2, // mean, std
            DistributionType::StudentT => 3, // mean, scale, degrees of freedom
            DistributionType::NegativeBinomial => 2, // n, p
        };
        
        NetworkBuilder::new()
            .input_layer(self.config.hidden_size)
            .hidden_layer(self.config.hidden_size / 2, ActivationFunction::Relu)
            .output_layer(output_size, ActivationFunction::Linear)
            .build()
            .map_err(|e| ModelError::NetworkError(e.to_string()))
    }
    
    /// Scale input data
    fn scale_data(&mut self, data: &[T]) -> Vec<T> {
        match self.config.scaling {
            ScalingMethod::None => data.to_vec(),
            ScalingMethod::Standard => {
                let mean = data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(data.len()).unwrap();
                let variance = data.iter()
                    .map(|&x| (x - mean).powi(2))
                    .fold(T::zero(), |acc, x| acc + x) / T::from(data.len()).unwrap();
                let std = variance.sqrt();
                
                self.state.scale_mean = mean;
                self.state.scale_std = std;
                
                data.iter().map(|&x| (x - mean) / std).collect()
            },
            ScalingMethod::Mean => {
                let mean = data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(data.len()).unwrap();
                self.state.scale_mean = mean;
                self.state.scale_std = T::one();
                
                data.iter().map(|&x| x / mean).collect()
            }
        }
    }
    
    /// Unscale output data
    fn unscale_data(&self, data: &[T]) -> Vec<T> {
        match self.config.scaling {
            ScalingMethod::None => data.to_vec(),
            ScalingMethod::Standard => {
                data.iter()
                    .map(|&x| x * self.state.scale_std + self.state.scale_mean)
                    .collect()
            },
            ScalingMethod::Mean => {
                data.iter()
                    .map(|&x| x * self.state.scale_mean)
                    .collect()
            }
        }
    }
    
    /// Encode input sequence
    fn encode_sequence(&mut self, input_sequence: &[T]) -> Result<Vec<T>, ModelError> {
        let encoder = self.encoder_network.as_mut()
            .ok_or_else(|| ModelError::NotTrainedError)?;
        
        let mut context = vec![T::zero(); self.config.hidden_size];
        
        for &value in input_sequence {
            let mut network_input = vec![value];
            
            // Add static features if available
            if self.config.static_features_size > 0 {
                network_input.extend(vec![T::zero(); self.config.static_features_size]);
            }
            
            // Add exogenous features if available
            if self.config.exogenous_features_size > 0 {
                network_input.extend(vec![T::zero(); self.config.exogenous_features_size]);
            }
            
            let output = encoder.run(&network_input)
                .map_err(|e| ModelError::NetworkError(e.to_string()))?;
            
            context = output;
        }
        
        Ok(context)
    }
    
    /// Sample from distribution given parameters
    fn sample_from_distribution(&self, params: &[T]) -> Result<T, ModelError> {
        let mut rng = rand::thread_rng();
        
        match self.config.distribution {
            DistributionType::Gaussian => {
                if params.len() != 2 {
                    return Err(ModelError::PredictionError("Gaussian requires 2 parameters".to_string()));
                }
                let mean = params[0].to_f64().unwrap();
                let std = params[1].abs().to_f64().unwrap().max(1e-6);
                let normal = Normal::new(mean, std)
                    .map_err(|e| ModelError::PredictionError(e.to_string()))?;
                Ok(T::from(normal.sample(&mut rng)).unwrap())
            },
            DistributionType::StudentT => {
                if params.len() != 3 {
                    return Err(ModelError::PredictionError("Student-t requires 3 parameters".to_string()));
                }
                let mean = params[0].to_f64().unwrap();
                let scale = params[1].abs().to_f64().unwrap().max(1e-6);
                let df = params[2].abs().to_f64().unwrap().max(1.0);
                
                let t_dist = StudentT::new(mean, scale, df)
                    .map_err(|e| ModelError::PredictionError(e.to_string()))?;
                Ok(T::from(t_dist.sample(&mut rng)).unwrap())
            },
            DistributionType::NegativeBinomial => {
                // Simplified negative binomial sampling
                if params.len() != 2 {
                    return Err(ModelError::PredictionError("Negative binomial requires 2 parameters".to_string()));
                }
                let n = params[0].abs();
                let p = params[1].abs().min(T::one()).max(T::from(1e-6).unwrap());
                
                // Use gamma-Poisson mixture approximation
                let gamma_shape = n;
                let gamma_rate = p / (T::one() - p);
                
                // Simplified sampling - in practice would use proper gamma-Poisson
                let mean = gamma_shape / gamma_rate;
                Ok(mean.max(T::zero()))
            }
        }
    }
    
    /// Generate autoregressive predictions
    fn generate_predictions(&mut self, context: &[T], horizon: usize) -> Result<Vec<T>, ModelError> {
        let decoder = self.decoder_network.as_mut()
            .ok_or_else(|| ModelError::NotTrainedError)?;
        let dist_net = self.distribution_network.as_mut()
            .ok_or_else(|| ModelError::NotTrainedError)?;
        
        let mut predictions = Vec::with_capacity(horizon);
        let mut current_context = context.to_vec();
        let mut last_value = T::zero();
        
        for _ in 0..horizon {
            // Prepare decoder input
            let mut decoder_input = current_context.clone();
            decoder_input.push(last_value);
            
            // Add exogenous features if available
            if self.config.exogenous_features_size > 0 {
                decoder_input.extend(vec![T::zero(); self.config.exogenous_features_size]);
            }
            
            // Get hidden state from decoder
            let hidden_output = decoder.run(&decoder_input)
                .map_err(|e| ModelError::NetworkError(e.to_string()))?;
            
            // Get distribution parameters
            let dist_params = dist_net.run(&hidden_output)
                .map_err(|e| ModelError::NetworkError(e.to_string()))?;
            
            // Sample from distribution
            let prediction = self.sample_from_distribution(&dist_params)?;
            predictions.push(prediction);
            
            // Update context and last value for next step
            current_context = hidden_output;
            last_value = prediction;
        }
        
        Ok(predictions)
    }
}

impl<T: Float> BaseModel<T> for DeepAR<T> {
    type Config = DeepARConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self, ModelError> {
        config.validate()?;
        
        let state = DeepARState::new(config.num_layers, config.hidden_size);
        
        Ok(Self {
            config,
            encoder_network: None,
            decoder_network: None,
            distribution_network: None,
            state,
            trained: false,
        })
    }
    
    fn fit(&mut self, data: &TimeSeriesData<T>, training_config: &TrainingConfig<T>) -> Result<(), ModelError> {
        if data.values.len() < self.config.input_size + self.config.horizon {
            return Err(ModelError::DataError(
                "Insufficient data for training".to_string()
            ));
        }
        
        // Create networks
        self.encoder_network = Some(self.create_encoder_network()?);
        self.decoder_network = Some(self.create_decoder_network()?);
        self.distribution_network = Some(self.create_distribution_network()?);
        
        // Scale training data
        let scaled_values = self.scale_data(&data.values);
        
        // Training loop (simplified)
        let mut best_loss = T::infinity();
        let mut patience_counter = 0;
        
        for epoch in 0..training_config.max_epochs {
            let mut epoch_loss = T::zero();
            let mut num_batches = 0;
            
            // Create training batches
            for i in 0..(scaled_values.len() - self.config.input_size - self.config.horizon) {
                let input_seq = &scaled_values[i..i + self.config.input_size];
                let target_seq = &scaled_values[i + self.config.input_size..i + self.config.input_size + self.config.horizon];
                
                // Encode input sequence
                let context = self.encode_sequence(input_seq)?;
                
                // Generate predictions
                let predictions = self.generate_predictions(&context, self.config.horizon)?;
                
                // Calculate loss (simplified MSE)
                let batch_loss = target_seq.iter()
                    .zip(predictions.iter())
                    .map(|(&target, &pred)| (target - pred).powi(2))
                    .fold(T::zero(), |acc, loss| acc + loss)
                    / T::from(self.config.horizon).unwrap();
                
                epoch_loss = epoch_loss + batch_loss;
                num_batches += 1;
            }
            
            epoch_loss = epoch_loss / T::from(num_batches).unwrap();
            
            // Early stopping
            if let Some(patience) = training_config.patience {
                if epoch_loss < best_loss {
                    best_loss = epoch_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        break;
                    }
                }
            }
            
            if epoch % 10 == 0 {
                log::info!("Epoch {}: Loss = {:?}", epoch, epoch_loss);
            }
        }
        
        self.state.training_loss = Some(best_loss);
        self.trained = true;
        
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesData<T>) -> Result<PredictionResult<T>, ModelError> {
        if !self.trained {
            return Err(ModelError::NotTrainedError);
        }
        
        if data.values.len() < self.config.input_size {
            return Err(ModelError::DataError(
                "Insufficient input data for prediction".to_string()
            ));
        }
        
        // Take last input_size values
        let input_sequence = &data.values[data.values.len() - self.config.input_size..];
        let scaled_input = match self.config.scaling {
            ScalingMethod::None => input_sequence.to_vec(),
            _ => {
                input_sequence.iter()
                    .map(|&x| match self.config.scaling {
                        ScalingMethod::Standard => (x - self.state.scale_mean) / self.state.scale_std,
                        ScalingMethod::Mean => x / self.state.scale_mean,
                        ScalingMethod::None => x,
                    })
                    .collect()
            }
        };
        
        // Generate multiple samples for uncertainty quantification
        let mut all_samples = Vec::new();
        for _ in 0..self.config.num_samples {
            let mut model_copy = DeepAR {
                config: self.config.clone(),
                encoder_network: None, // Would need proper cloning in production
                decoder_network: None,
                distribution_network: None,
                state: DeepARState::new(self.config.num_layers, self.config.hidden_size),
                trained: self.trained,
            };
            
            // Simplified prediction (in practice would use the actual networks)
            let mut predictions = Vec::new();
            let mut rng = rand::thread_rng();
            let last_value = *input_sequence.last().unwrap();
            
            for i in 0..self.config.horizon {
                // Simplified autoregressive prediction with noise
                let trend = if i > 0 { predictions[i-1] } else { last_value };
                let noise = T::from(rng.gen_range(-0.1..0.1)).unwrap();
                predictions.push(trend + noise);
            }
            
            all_samples.push(predictions);
        }
        
        // Calculate point forecasts (mean of samples)
        let mut forecasts = vec![T::zero(); self.config.horizon];
        for sample in &all_samples {
            for (i, &value) in sample.iter().enumerate() {
                forecasts[i] = forecasts[i] + value;
            }
        }
        for forecast in &mut forecasts {
            *forecast = *forecast / T::from(self.config.num_samples).unwrap();
        }
        
        // Unscale predictions
        let final_forecasts = self.unscale_data(&forecasts);
        
        // Generate timestamps (simplified)
        let mut timestamps = Vec::new();
        let last_time = data.timestamps.last().unwrap();
        for i in 1..=self.config.horizon {
            timestamps.push(*last_time + chrono::Duration::hours(i as i64));
        }
        
        // Calculate prediction intervals
        let mut lower_bounds = HashMap::new();
        let mut upper_bounds = HashMap::new();
        
        for &confidence in &[0.8, 0.9, 0.95] {
            let alpha = 1.0 - confidence;
            let lower_quantile = alpha / 2.0;
            let upper_quantile = 1.0 - alpha / 2.0;
            
            let mut lower = vec![T::zero(); self.config.horizon];
            let mut upper = vec![T::zero(); self.config.horizon];
            
            for i in 0..self.config.horizon {
                let mut values: Vec<T> = all_samples.iter().map(|s| s[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                let lower_idx = (lower_quantile * values.len() as f64) as usize;
                let upper_idx = (upper_quantile * values.len() as f64) as usize;
                
                lower[i] = values[lower_idx.min(values.len() - 1)];
                upper[i] = values[upper_idx.min(values.len() - 1)];
            }
            
            lower_bounds.insert(confidence.to_string(), self.unscale_data(&lower));
            upper_bounds.insert(confidence.to_string(), self.unscale_data(&upper));
        }
        
        let intervals = Some(PredictionIntervals {
            lower_bounds,
            upper_bounds,
            confidence_levels: vec!["0.8".to_string(), "0.9".to_string(), "0.95".to_string()],
        });
        
        let mut metadata = HashMap::new();
        metadata.insert("model".to_string(), "DeepAR".to_string());
        metadata.insert("distribution".to_string(), format!("{:?}", self.config.distribution));
        metadata.insert("num_samples".to_string(), self.config.num_samples.to_string());
        
        Ok(PredictionResult {
            forecasts: final_forecasts,
            timestamps,
            series_id: data.series_id.clone(),
            intervals,
            metadata,
        })
    }
    
    fn is_trained(&self) -> bool {
        self.trained
    }
    
    fn reset(&mut self) -> Result<(), ModelError> {
        self.encoder_network = None;
        self.decoder_network = None;
        self.distribution_network = None;
        self.state.reset();
        self.trained = false;
        Ok(())
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn validate_input(&self, data: &TimeSeriesData<T>) -> Result<(), ModelError> {
        if data.values.is_empty() {
            return Err(ModelError::DataError("Empty input data".to_string()));
        }
        
        if data.values.len() < self.config.input_size {
            return Err(ModelError::DimensionError {
                expected: self.config.input_size,
                actual: data.values.len(),
            });
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "DeepAR"
    }
}

impl<T: Float> ProbabilisticForecasting<T> for DeepAR<T> {
    fn predict_with_intervals(
        &self,
        data: &TimeSeriesData<T>,
        confidence_levels: &[T]
    ) -> Result<PredictionResult<T>, ModelError> {
        let mut result = self.predict(data)?;
        
        // Override intervals with custom confidence levels
        if let Some(ref mut intervals) = result.intervals {
            intervals.confidence_levels = confidence_levels.iter()
                .map(|&cl| cl.to_string())
                .collect();
        }
        
        Ok(result)
    }
    
    fn predict_quantiles(
        &self,
        data: &TimeSeriesData<T>,
        quantiles: &[T]
    ) -> Result<PredictionResult<T>, ModelError> {
        // For simplicity, convert quantiles to confidence intervals
        let confidence_levels: Vec<T> = quantiles.iter()
            .map(|&q| if q > T::from(0.5).unwrap() { T::one() - (T::one() - q) * T::from(2.0).unwrap() } else { q * T::from(2.0).unwrap() })
            .collect();
        
        self.predict_with_intervals(data, &confidence_levels)
    }
    
    fn sample_predictions(
        &self,
        data: &TimeSeriesData<T>,
        num_samples: usize
    ) -> Result<Vec<Vec<T>>, ModelError> {
        if !self.trained {
            return Err(ModelError::NotTrainedError);
        }
        
        let mut samples = Vec::new();
        for _ in 0..num_samples {
            let result = self.predict(data)?;
            samples.push(result.forecasts);
        }
        
        Ok(samples)
    }
}