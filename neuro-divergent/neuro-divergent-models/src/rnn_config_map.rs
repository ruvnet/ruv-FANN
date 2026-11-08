use super::config::RNNConfig;
use super::foundation::{ConfigValue, ModelConfig};
use super::{NeuroDivergentError as Error, NeuroDivergentResult as Result};
use num_traits::Float;
use ruv_fann::ActivationFunction;
use std::collections::HashMap;

pub(super) fn to_map<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
    config: &RNNConfig<T>,
) -> HashMap<String, ConfigValue<T>> {
    let mut map = HashMap::new();
    for (name, value) in [
        ("horizon", config.horizon),
        ("input_size", config.input_size),
        ("hidden_size", config.hidden_size),
        ("num_layers", config.num_layers),
        ("max_steps", config.max_steps),
    ] {
        map.insert(name.into(), ConfigValue::UInt(value));
    }
    for (name, value) in [
        ("dropout", config.dropout),
        ("learning_rate", config.learning_rate),
        ("weight_decay", config.weight_decay),
    ] {
        map.insert(name.into(), ConfigValue::Float(value));
    }
    map.insert(
        "activation".into(),
        ConfigValue::String(config.activation.name().into()),
    );
    if let Some(clip) = config.gradient_clip_val {
        map.insert("gradient_clip_val".into(), ConfigValue::Float(clip));
    }
    for (name, values) in [
        ("static_features", &config.static_features),
        ("hist_exog_features", &config.hist_exog_features),
        ("futr_exog_features", &config.futr_exog_features),
    ] {
        if let Some(values) = values {
            map.insert(name.into(), ConfigValue::StringVec(values.clone()));
        }
    }
    map
}

pub(super) fn from_map<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
    map: HashMap<String, ConfigValue<T>>,
) -> Result<RNNConfig<T>> {
    let mut config = RNNConfig::default_with_horizon(1);
    for (name, value) in map {
        match (name.as_str(), value) {
            ("horizon", ConfigValue::UInt(v)) => config.horizon = v,
            ("input_size", ConfigValue::UInt(v)) => config.input_size = v,
            ("hidden_size", ConfigValue::UInt(v)) => config.hidden_size = v,
            ("num_layers", ConfigValue::UInt(v)) => config.num_layers = v,
            ("max_steps", ConfigValue::UInt(v)) => config.max_steps = v,
            ("dropout", ConfigValue::Float(v)) => config.dropout = v,
            ("learning_rate", ConfigValue::Float(v)) => config.learning_rate = v,
            ("weight_decay", ConfigValue::Float(v)) => config.weight_decay = v,
            ("gradient_clip_val", ConfigValue::Float(v)) => config.gradient_clip_val = Some(v),
            ("static_features", ConfigValue::StringVec(v)) => config.static_features = Some(v),
            ("hist_exog_features", ConfigValue::StringVec(v)) => {
                config.hist_exog_features = Some(v)
            }
            ("futr_exog_features", ConfigValue::StringVec(v)) => {
                config.futr_exog_features = Some(v)
            }
            ("activation", ConfigValue::String(v)) => {
                config.activation = match v.as_str() {
                    "Tanh" => ActivationFunction::Tanh,
                    "Sigmoid" => ActivationFunction::Sigmoid,
                    "SigmoidSymmetric" => ActivationFunction::SigmoidSymmetric,
                    "Linear" => ActivationFunction::Linear,
                    "ReLU" => ActivationFunction::ReLU,
                    "ReLULeaky" => ActivationFunction::ReLULeaky,
                    "LeakyRelu" => ActivationFunction::LeakyRelu,
                    _ => return Err(Error::config("Unsupported RNN activation name")),
                }
            }
            _ => {
                return Err(Error::config(format!(
                    "Unknown RNN setting or incorrect value type: {name}"
                )))
            }
        }
    }
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rnn_config_map_preserves_training_and_architecture_settings() {
        let mut config = RNNConfig::<f64>::default_with_horizon(3);
        config.input_size = 2;
        config.hidden_size = 5;
        config.num_layers = 2;
        config.max_steps = 17;
        config.learning_rate = 0.03;
        config.weight_decay = 0.02;
        config.gradient_clip_val = Some(1.2);
        config.activation = ActivationFunction::LeakyRelu;
        config.hist_exog_features = Some(vec!["covariate".into()]);
        let restored = from_map(to_map(&config)).unwrap();
        assert_eq!(restored.horizon, 3);
        assert_eq!(restored.input_size, 2);
        assert_eq!(restored.hidden_size, 5);
        assert_eq!(restored.num_layers, 2);
        assert_eq!(restored.max_steps, 17);
        assert_eq!(restored.learning_rate, 0.03);
        assert_eq!(restored.weight_decay, 0.02);
        assert_eq!(restored.gradient_clip_val, Some(1.2));
        assert_eq!(restored.activation, ActivationFunction::LeakyRelu);
        assert_eq!(restored.hist_exog_features, config.hist_exog_features);
        let bad = HashMap::from([("horizon".into(), ConfigValue::<f64>::String("3".into()))]);
        assert!(from_map(bad).is_err());
    }
}
