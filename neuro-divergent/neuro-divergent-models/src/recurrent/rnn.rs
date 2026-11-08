//! Trainable stacked vanilla RNN with a learned, multi-step FANN projection.

use super::bptt::{validate_projection, RecurrentNetwork};
use super::layers::{BidirectionalMergeMode, RecurrentLayer};
use crate::config::RNNConfig;
use crate::errors::{NeuroDivergentError as Error, NeuroDivergentResult as Result};
use crate::foundation::{
    BaseModel, ForecastOutput, ModelConfig, NetworkAdapter, RecurrentState, TimeSeriesDataset,
    TimeSeriesInput, TrainingHistory, TrainingMetrics, ValidationMetrics,
};
use crate::utils::math;
use neuro_divergent_training::{Adam, Optimizer};
use num_traits::Float;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruv_fann::{ActivationFunction, Network, NetworkBuilder, TrainingData};
use std::collections::HashMap;

/// A stacked vanilla RNN trained jointly with a linear hidden-to-horizon projection.
///
/// `input_size` counts features per time step, not lookback length. Each step is
/// `[target, static_features..., historical_features...]`. Windows are independent
/// and start from the model's configured initial states (zero by default).
/// Full BPTT is the default; [`Self::set_bptt_truncation`] can detach an earlier
/// prefix. Adam optimizes weighted mean squared error, with optional global norm
/// clipping and coupled L2 regularization. No automatic scaling is performed.
#[derive(Clone)]
pub struct RNN<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    config: RNNConfig<T>,
    recurrent: RecurrentNetwork<T>,
    output_layer: Network<T>,
    is_trained: bool,
    training_history: Option<TrainingHistory<T>>,
    model_metrics: HashMap<String, f64>,
    seed: u64,
    bptt_truncation: Option<usize>,
}

struct PreparedSample<T> {
    sequence: Vec<Vec<T>>,
    target: Vec<T>,
    weight: T,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RNN<T> {
    /// Initialize an independently randomized model with validated dimensions.
    pub fn new(config: RNNConfig<T>) -> Result<Self> {
        Self::new_seeded(config, rand::random())
    }

    /// Initialize reproducibly, including recurrent weights and the output projection.
    pub fn new_seeded(config: RNNConfig<T>, seed: u64) -> Result<Self> {
        config.validate()?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let recurrent = RecurrentNetwork::new(&config, &mut rng);
        let mut output_layer = NetworkBuilder::new()
            .input_layer(config.hidden_size)
            .output_layer_with_activation(config.horizon, ActivationFunction::Linear, T::one())
            .build();
        let bound = (6.0 / (config.hidden_size as f64 + config.horizon as f64)).sqrt();
        let weights: Vec<T> = (0..config.horizon)
            .flat_map(|_| {
                (0..config.hidden_size)
                    .map(|_| T::from(rng.gen_range(-bound..bound)).unwrap())
                    .chain(std::iter::once(T::zero()))
                    .collect::<Vec<_>>()
            })
            .collect();
        output_layer.set_weights(&weights)?;
        Ok(Self {
            config,
            recurrent,
            output_layer,
            is_trained: false,
            training_history: None,
            model_metrics: HashMap::new(),
            seed,
            bptt_truncation: None,
        })
    }

    /// An owned configuration for adapters and checkpoint descriptors.
    pub fn config_owned(&self) -> RNNConfig<T> {
        self.config.clone()
    }

    /// None differentiates the entire lookback; Some(K) differentiates its final K
    /// steps and treats the states at the prefix boundary as constants.
    pub fn set_bptt_truncation(&mut self, steps: Option<usize>) -> Result<()> {
        if steps == Some(0) {
            return Err(Error::config("BPTT truncation must be positive"));
        }
        self.bptt_truncation = steps;
        Ok(())
    }

    pub fn bptt_truncation(&self) -> Option<usize> {
        self.bptt_truncation
    }

    fn prepare_sequence(&self, input: &TimeSeriesInput<T>) -> Result<Vec<Vec<T>>> {
        input.validate()?;
        if input
            .future_features
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        {
            return Err(Error::data(
                "Vanilla RNN does not consume future exogenous features",
            ));
        }
        let mut sequence = Vec::with_capacity(input.historical_targets.len());
        for (step, &target) in input.historical_targets.iter().enumerate() {
            let mut features = vec![target];
            if let Some(values) = &input.static_features {
                features.extend_from_slice(values);
            }
            if let Some(values) = &input.historical_features {
                features.extend_from_slice(&values[step]);
            }
            if features.len() != self.config.input_size {
                return Err(Error::dimension_mismatch(
                    self.config.input_size,
                    features.len(),
                ));
            }
            if features.iter().any(|v| !v.is_finite()) {
                return Err(Error::data("RNN inputs must be finite"));
            }
            sequence.push(features);
        }
        Ok(sequence)
    }

    fn prepare_dataset(&self, dataset: &TimeSeriesDataset<T>) -> Result<Vec<PreparedSample<T>>> {
        if dataset.samples.is_empty() {
            return Err(Error::data("RNN dataset cannot be empty"));
        }
        let mut total_weight = T::zero();
        let mut prepared = Vec::with_capacity(dataset.samples.len());
        for sample in &dataset.samples {
            if sample.target.len() != self.config.horizon {
                return Err(Error::dimension_mismatch(
                    self.config.horizon,
                    sample.target.len(),
                ));
            }
            let weight = sample.weight.unwrap_or_else(T::one);
            if !weight.is_finite()
                || weight < T::zero()
                || sample.target.iter().any(|v| !v.is_finite())
            {
                return Err(Error::data(
                    "RNN targets and nonnegative sample weights must be finite",
                ));
            }
            total_weight = total_weight + weight;
            prepared.push(PreparedSample {
                sequence: self.prepare_sequence(&sample.input)?,
                target: sample.target.clone(),
                weight,
            });
        }
        if !total_weight.is_finite() || total_weight <= T::zero() {
            return Err(Error::data(
                "RNN requires a positive, finite total sample weight",
            ));
        }
        for sample in &mut prepared {
            sample.weight = sample.weight / total_weight;
        }
        Ok(prepared)
    }

    fn encode(&self, sequence: &[Vec<T>]) -> Result<Vec<T>> {
        let tape = self.recurrent.forward(sequence, None::<&mut ChaCha8Rng>)?;
        Ok(self.recurrent.final_hidden(&tape).to_vec())
    }

    fn dataset_loss(&self, samples: &[PreparedSample<T>]) -> Result<T> {
        let mut projection = self.output_layer.clone();
        let mut loss = T::zero();
        for sample in samples.iter().filter(|sample| sample.weight > T::zero()) {
            let hidden = self.encode(&sample.sequence)?;
            let prediction = projection.run(&hidden)?;
            loss = loss + sample.weight * math::mse(&prediction, &sample.target)?;
        }
        if !loss.is_finite() {
            return Err(Error::training("RNN loss overflowed; scale the data"));
        }
        Ok(loss)
    }

    fn fit_internal(&mut self, dataset: &TimeSeriesDataset<T>) -> Result<TrainingMetrics<T>> {
        self.config.validate()?;
        validate_projection(
            &self.output_layer,
            self.config.hidden_size,
            self.config.horizon,
        )?;
        let samples = self.prepare_dataset(dataset)?;
        let start = std::time::Instant::now();
        let mut parameters = self.recurrent.parameters(&self.output_layer);
        let mut best_parameters = parameters.clone();
        let initial_loss = self.dataset_loss(&samples)?;
        let mut best_loss = initial_loss;
        let mut history = TrainingHistory::new();
        let mut optimizer = Adam::new(
            self.config.learning_rate,
            T::from(0.9).unwrap(),
            T::from(0.999).unwrap(),
        );
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        for _ in 0..self.config.max_steps {
            let epoch_start = std::time::Instant::now();
            let mut gradients: Vec<Vec<T>> = parameters
                .iter()
                .map(|row| vec![T::zero(); row.len()])
                .collect();
            let mut projection = self.output_layer.clone();
            for sample in samples.iter().filter(|sample| sample.weight > T::zero()) {
                let tape = self.recurrent.forward(&sample.sequence, Some(&mut rng))?;
                let prediction = projection.run(self.recurrent.final_hidden(&tape))?;
                let (_, sample_gradients) = self.recurrent.gradients(
                    &tape,
                    &projection,
                    &prediction,
                    &sample.target,
                    self.bptt_truncation,
                )?;
                for (gradient, sample_gradient) in gradients
                    .iter_mut()
                    .flatten()
                    .zip(sample_gradients.iter().flatten())
                {
                    *gradient = *gradient + sample.weight * *sample_gradient;
                }
            }
            for (gradient, &parameter) in gradients
                .iter_mut()
                .flatten()
                .zip(parameters.iter().flatten())
            {
                *gradient = *gradient + self.config.weight_decay * parameter;
            }
            if let Some(clip) = self.config.gradient_clip_val {
                neuro_divergent_training::utils::clip_gradients_by_norm(&mut gradients, clip);
            }
            optimizer
                .step(&mut parameters, &gradients)
                .map_err(|error| Error::training(error.to_string()))?;
            self.recurrent
                .set_parameters(&mut self.output_layer, &parameters)?;
            let loss = self.dataset_loss(&samples)?;
            history.add_epoch(
                loss,
                None,
                self.config.learning_rate,
                epoch_start.elapsed().as_secs_f64(),
            );
            if loss < best_loss {
                best_loss = loss;
                best_parameters = parameters.clone();
            }
        }
        self.recurrent
            .set_parameters(&mut self.output_layer, &best_parameters)?;
        self.is_trained = true;
        self.model_metrics
            .insert("initial_loss".into(), initial_loss.to_f64().unwrap());
        self.model_metrics
            .insert("training_loss".into(), best_loss.to_f64().unwrap());
        self.model_metrics.insert(
            "num_parameters".into(),
            parameters.iter().map(Vec::len).sum::<usize>() as f64,
        );
        self.model_metrics
            .insert("hidden_size".into(), self.config.hidden_size as f64);
        self.model_metrics
            .insert("num_layers".into(), self.config.num_layers as f64);
        let epochs_completed = history.num_epochs();
        self.training_history = Some(history);
        Ok(TrainingMetrics {
            final_loss: best_loss,
            epochs_completed,
            training_time_seconds: start.elapsed().as_secs_f64(),
            best_validation_loss: None,
            early_stopped: false,
            convergence_achieved: best_loss < T::from(1e-6).unwrap(),
        })
    }

    /// Bidirectional training is not implemented; this never silently changes the model.
    pub fn enable_bidirectional(&mut self, _merge_mode: BidirectionalMergeMode) -> Result<()> {
        Err(Error::config("Bidirectional RNN training is unsupported"))
    }

    /// The next fit validates the new dropout rate. Predictions always disable dropout.
    pub fn set_dropout(&mut self, rate: T) {
        self.config.dropout = rate;
        self.recurrent.dropout = rate;
    }
    pub fn get_dropout(&self) -> T {
        self.config.dropout
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> BaseModel<T> for RNN<T> {
    fn name(&self) -> &str {
        "RNN"
    }
    fn config(&self) -> &dyn ModelConfig<T> {
        &self.config
    }
    fn is_trained(&self) -> bool {
        self.is_trained
    }
    fn fit(&mut self, dataset: &TimeSeriesDataset<T>) -> Result<TrainingMetrics<T>> {
        // Failed validation or numerical updates preserve the original fitted model.
        let mut candidate = self.clone();
        let metrics = candidate.fit_internal(dataset)?;
        *self = candidate;
        Ok(metrics)
    }
    fn predict(&self, input: &TimeSeriesInput<T>) -> Result<ForecastOutput<T>> {
        if !self.is_trained {
            return Err(Error::prediction("Model has not been trained"));
        }
        validate_projection(
            &self.output_layer,
            self.config.hidden_size,
            self.config.horizon,
        )?;
        let hidden = self.encode(&self.prepare_sequence(input)?)?;
        let values = self.output_layer.clone().run(&hidden)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::prediction("RNN forecast overflowed"));
        }
        let mut output = ForecastOutput::new(values);
        output.unique_id = input.unique_id.clone();
        Ok(output)
    }
    fn predict_batch(&self, inputs: &[TimeSeriesInput<T>]) -> Result<Vec<ForecastOutput<T>>> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }
    fn validate(&self, dataset: &TimeSeriesDataset<T>) -> Result<ValidationMetrics<T>> {
        if !self.is_trained {
            return Err(Error::prediction("Model has not been trained"));
        }
        let samples = self.prepare_dataset(dataset)?;
        let mse = self.dataset_loss(&samples)?;
        let mut mae = T::zero();
        for sample in &samples {
            let values = self
                .output_layer
                .clone()
                .run(&self.encode(&sample.sequence)?)?;
            mae = mae + sample.weight * math::mae(&values, &sample.target)?;
        }
        Ok(ValidationMetrics {
            mse,
            mae,
            mape: None,
            smape: None,
            r2_score: None,
            directional_accuracy: None,
            custom_metrics: HashMap::new(),
        })
    }
    fn save_state(&self) -> Result<Vec<u8>> {
        self.encode_state()
    }
    fn load_state(&mut self, bytes: &[u8]) -> Result<()> {
        *self = Self::from_state(bytes)?;
        Ok(())
    }
    fn input_size(&self) -> usize {
        self.config.input_size
    }
    fn horizon(&self) -> usize {
        self.config.horizon
    }
    fn reset(&mut self) {
        // Constructor already validated this stored configuration; retain allocation
        // if a caller temporarily supplied an invalid dropout through the legacy setter.
        if let Ok(mut fresh) = Self::new_seeded(self.config.clone(), self.seed) {
            fresh.bptt_truncation = self.bptt_truncation;
            *self = fresh;
        } else {
            self.is_trained = false;
            self.training_history = None;
            self.model_metrics.clear();
        }
    }
    fn training_history(&self) -> Option<&TrainingHistory<T>> {
        self.training_history.as_ref()
    }
    fn model_metrics(&self) -> HashMap<String, f64> {
        self.model_metrics.clone()
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> NetworkAdapter<T>
    for RNN<T>
{
    fn prepare_input(&self, input: &TimeSeriesInput<T>) -> Result<Vec<T>> {
        self.encode(&self.prepare_sequence(input)?)
    }
    fn process_output(&self, output: Vec<T>) -> Result<ForecastOutput<T>> {
        if output.len() != self.config.horizon || output.iter().any(|v| !v.is_finite()) {
            return Err(Error::prediction(
                "Invalid RNN output horizon or non-finite forecast",
            ));
        }
        Ok(ForecastOutput::new(output))
    }
    fn create_training_data(&self, dataset: &TimeSeriesDataset<T>) -> Result<TrainingData<T>> {
        let samples = self.prepare_dataset(dataset)?;
        let inputs = samples
            .iter()
            .map(|sample| self.encode(&sample.sequence))
            .collect::<Result<_>>()?;
        Ok(TrainingData {
            inputs,
            outputs: samples.into_iter().map(|sample| sample.target).collect(),
        })
    }
    fn network(&self) -> &Network<T> {
        &self.output_layer
    }
    fn network_mut(&mut self) -> &mut Network<T> {
        &mut self.output_layer
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RecurrentState<T>
    for RNN<T>
{
    fn reset(&mut self) {
        for layer in &mut self.recurrent.layers {
            layer.reset_state();
        }
    }
    fn get_state(&self) -> Vec<T> {
        self.recurrent
            .layers
            .iter()
            .flat_map(|layer| layer.get_state())
            .collect()
    }
    fn set_state(&mut self, state: Vec<T>) -> Result<()> {
        if state.len() != self.state_dimension() || state.iter().any(|v| !v.is_finite()) {
            return Err(Error::state(
                "Invalid RNN initial-state dimensions or non-finite values",
            ));
        }
        for (layer, state) in self
            .recurrent
            .layers
            .iter_mut()
            .zip(state.chunks(self.config.hidden_size))
        {
            layer.set_state(state.to_vec())?;
        }
        Ok(())
    }
    fn state_dimension(&self) -> usize {
        self.config.num_layers * self.config.hidden_size
    }
    fn clone_state(&self) -> Box<dyn RecurrentState<T>> {
        Box::new(self.clone())
    }
}

#[path = "checkpoint.rs"]
mod checkpoint;
#[cfg(test)]
#[path = "gradient_tests.rs"]
mod gradient_tests;
#[cfg(test)]
#[path = "rnn_tests.rs"]
mod tests;
