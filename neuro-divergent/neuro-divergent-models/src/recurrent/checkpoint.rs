//! Versioned, bounded inference checkpoints without generic serde bounds.

use super::*;

const MAGIC: &[u8; 8] = b"NDRNN\0\x01\0";
const MAX_BYTES: usize = 256 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 1024 * 1024;

fn invalid() -> Error {
    Error::state("Invalid, truncated, or unsupported RNN checkpoint")
}

struct Writer(Vec<u8>);
impl Writer {
    fn integer(&mut self, value: usize) {
        self.0.extend_from_slice(&(value as u64).to_le_bytes());
    }
    fn scalar<T: Float>(&mut self, value: T) -> Result<()> {
        let value = value
            .to_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(invalid)?;
        self.0.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }
    fn values<T: Float>(&mut self, values: &[T]) -> Result<()> {
        self.integer(values.len());
        for &value in values {
            self.scalar(value)?;
        }
        if self.0.len() > MAX_BYTES {
            return Err(invalid());
        }
        Ok(())
    }
    fn string(&mut self, value: &str) -> Result<()> {
        if value.len() > MAX_STRING_BYTES {
            return Err(invalid());
        }
        self.integer(value.len());
        self.0.extend_from_slice(value.as_bytes());
        Ok(())
    }
    fn strings(&mut self, values: &Option<Vec<String>>) -> Result<()> {
        self.0.push(u8::from(values.is_some()));
        if let Some(values) = values {
            if values.len() > 65536 {
                return Err(invalid());
            }
            self.integer(values.len());
            for value in values {
                self.string(value)?;
            }
        }
        Ok(())
    }
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.0.len() {
            return Err(invalid());
        }
        let (value, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(value)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn flag(&mut self) -> Result<bool> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid()),
        }
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn integer(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid())
    }
    fn scalar<T: Float>(&mut self) -> Result<T> {
        let value = f64::from_le_bytes(self.take(8)?.try_into().unwrap());
        if !value.is_finite() {
            return Err(invalid());
        }
        T::from(value).filter(|v| v.is_finite()).ok_or_else(invalid)
    }
    fn values<T: Float>(&mut self, expected: Option<usize>) -> Result<Vec<T>> {
        let count = self.integer()?;
        if expected.is_some_and(|expected| count != expected) || count > self.0.len() / 8 {
            return Err(invalid());
        }
        (0..count).map(|_| self.scalar()).collect()
    }
    fn string(&mut self) -> Result<String> {
        let len = self.integer()?;
        if len > MAX_STRING_BYTES {
            return Err(invalid());
        }
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| invalid())
    }
    fn strings(&mut self) -> Result<Option<Vec<String>>> {
        if !self.flag()? {
            return Ok(None);
        }
        let count = self.integer()?;
        if count > 65536 || count > self.0.len() / 8 {
            return Err(invalid());
        }
        (0..count)
            .map(|_| self.string())
            .collect::<Result<_>>()
            .map(Some)
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RNN<T> {
    /// Restore configuration, all learned weights, initial states, training history,
    /// and metrics. Scalars use little-endian f64 with checked conversion to T.
    /// This is an inference checkpoint; a subsequent fit starts a fresh Adam state.
    pub fn from_state(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid());
        }
        let mut reader = Reader(bytes);
        if reader.take(8)? != MAGIC {
            return Err(invalid());
        }
        let mut config = RNNConfig::default_with_horizon(reader.integer()?);
        config.input_size = reader.integer()?;
        config.hidden_size = reader.integer()?;
        config.num_layers = reader.integer()?;
        config.max_steps = reader.integer()?;
        config.activation = match reader.byte()? {
            0 => ActivationFunction::Linear,
            1 => ActivationFunction::Tanh,
            2 => ActivationFunction::Sigmoid,
            3 => ActivationFunction::SigmoidSymmetric,
            4 => ActivationFunction::ReLU,
            5 => ActivationFunction::ReLULeaky,
            6 => ActivationFunction::LeakyRelu,
            _ => return Err(invalid()),
        };
        config.dropout = reader.scalar()?;
        config.learning_rate = reader.scalar()?;
        config.weight_decay = reader.scalar()?;
        config.gradient_clip_val = if reader.flag()? {
            Some(reader.scalar()?)
        } else {
            None
        };
        config.static_features = reader.strings()?;
        config.hist_exog_features = reader.strings()?;
        config.futr_exog_features = reader.strings()?;
        config.validate()?;
        let seed = reader.u64()?;
        let truncation = match reader.integer()? {
            0 => None,
            steps => Some(steps),
        };
        let trained = reader.flag()?;
        if reader.integer()? != config.num_layers + 1 {
            return Err(invalid());
        }
        // Verify and decode tensor lengths before allocating the model architecture.
        let mut parameters = Vec::new();
        for layer in 0..config.num_layers {
            let width = if layer == 0 {
                config.input_size
            } else {
                config.hidden_size
            };
            let count = config.hidden_size * (width + config.hidden_size + 1);
            parameters.push(reader.values(Some(count))?);
        }
        parameters.push(reader.values(Some(config.horizon * (config.hidden_size + 1)))?);
        let states = reader.values(Some(config.hidden_size * config.num_layers))?;
        let history = if reader.flag()? {
            let losses = reader.values(None)?;
            let validation_losses = reader.values(None)?;
            let learning_rates = reader.values(Some(losses.len()))?;
            let training_times: Vec<f64> = reader.values(Some(losses.len()))?;
            if validation_losses.len() > losses.len()
                || training_times.iter().any(|v| *v < 0.0)
                || losses
                    .iter()
                    .chain(&validation_losses)
                    .any(|&v: &T| v < T::zero())
            {
                return Err(invalid());
            }
            Some(TrainingHistory {
                epoch_losses: losses,
                validation_losses,
                learning_rates,
                training_times,
                custom_metrics: HashMap::new(),
            })
        } else {
            None
        };
        let metric_count = reader.integer()?;
        if metric_count > reader.0.len() / 16 {
            return Err(invalid());
        }
        let mut metrics = HashMap::new();
        for _ in 0..metric_count {
            let key = reader.string()?;
            if metrics.insert(key, reader.scalar()?).is_some() {
                return Err(invalid());
            }
        }
        if !reader.0.is_empty() {
            return Err(invalid());
        }
        let mut model = Self::new_seeded(config, seed)?;
        model
            .recurrent
            .set_parameters(&mut model.output_layer, &parameters)?;
        RecurrentState::set_state(&mut model, states)?;
        model.bptt_truncation = truncation;
        model.is_trained = trained;
        model.training_history = history;
        model.model_metrics = metrics;
        Ok(model)
    }

    pub(super) fn encode_state(&self) -> Result<Vec<u8>> {
        self.config.validate()?;
        validate_projection(
            &self.output_layer,
            self.config.hidden_size,
            self.config.horizon,
        )?;
        let mut writer = Writer(MAGIC.to_vec());
        for value in [
            self.config.horizon,
            self.config.input_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.max_steps,
        ] {
            writer.integer(value);
        }
        writer.0.push(match self.config.activation {
            ActivationFunction::Linear => 0,
            ActivationFunction::Tanh => 1,
            ActivationFunction::Sigmoid => 2,
            ActivationFunction::SigmoidSymmetric => 3,
            ActivationFunction::ReLU => 4,
            ActivationFunction::ReLULeaky => 5,
            ActivationFunction::LeakyRelu => 6,
            _ => return Err(invalid()),
        });
        writer.scalar(self.config.dropout)?;
        writer.scalar(self.config.learning_rate)?;
        writer.scalar(self.config.weight_decay)?;
        writer
            .0
            .push(u8::from(self.config.gradient_clip_val.is_some()));
        if let Some(value) = self.config.gradient_clip_val {
            writer.scalar(value)?;
        }
        writer.strings(&self.config.static_features)?;
        writer.strings(&self.config.hist_exog_features)?;
        writer.strings(&self.config.futr_exog_features)?;
        writer.0.extend_from_slice(&self.seed.to_le_bytes());
        writer.integer(self.bptt_truncation.unwrap_or(0));
        writer.0.push(u8::from(self.is_trained));
        let parameters = self.recurrent.parameters(&self.output_layer);
        writer.integer(parameters.len());
        for values in &parameters {
            writer.values(values)?;
        }
        writer.values(&RecurrentState::get_state(self))?;
        writer.0.push(u8::from(self.training_history.is_some()));
        if let Some(history) = &self.training_history {
            writer.values(&history.epoch_losses)?;
            writer.values(&history.validation_losses)?;
            writer.values(&history.learning_rates)?;
            writer.values(&history.training_times)?;
        }
        writer.integer(self.model_metrics.len());
        let mut metrics: Vec<_> = self.model_metrics.iter().collect();
        metrics.sort_by_key(|&(key, _)| key);
        for (key, &value) in metrics {
            writer.string(key)?;
            writer.scalar(value)?;
        }
        if writer.0.len() > MAX_BYTES {
            return Err(invalid());
        }
        Ok(writer.0)
    }
}
