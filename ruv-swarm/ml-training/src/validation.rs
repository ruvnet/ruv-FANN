use crate::{Result, TrainingConfig, TrainingDataset, TrainingError};

pub(crate) fn validate_dataset(data: &TrainingDataset) -> Result<()> {
    if data.sequences.is_empty() || data.sequences.len() != data.labels.len() {
        return Err(TrainingError::DataLoadError(
            "Require nonempty, matching sequences and labels".into(),
        ));
    }
    let width = data.labels[0].len();
    if width == 0
        || data
            .labels
            .iter()
            .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
    {
        return Err(TrainingError::DataLoadError(
            "Labels must have consistent nonzero width and finite values".into(),
        ));
    }
    for sequence in &data.sequences {
        if sequence.features.is_empty()
            || sequence.features.len() != sequence.timestamps.len()
            || sequence
                .features
                .iter()
                .any(|row| row.is_empty() || row.iter().any(|value| !value.is_finite()))
        {
            return Err(TrainingError::DataLoadError(
                "Sequences need matching feature/timestamp rows and finite values".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_prediction(prediction: &[f64], label: &[f64]) -> Result<()> {
    if prediction.len() != label.len() || prediction.iter().any(|value| !value.is_finite()) {
        return Err(TrainingError::EvaluationError(
            "Model prediction must match label dimensions and be finite".into(),
        ));
    }
    Ok(())
}

pub(crate) fn split_dataset(
    data: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<(TrainingDataset, TrainingDataset)> {
    validate_dataset(data)?;
    if config.epochs == 0
        || config.batch_size == 0
        || !config.validation_split.is_finite()
        || !(0.0..1.0).contains(&config.validation_split)
        || config.validation_split == 0.0
        || data.sequences.len() < 2
    {
        return Err(TrainingError::ConfigurationError(
            "Require positive epochs/batch size, at least two samples and 0 < validation_split < 1"
                .into(),
        ));
    }
    let validation_count = ((data.sequences.len() as f64 * config.validation_split).ceil()
        as usize)
        .clamp(1, data.sequences.len() - 1);
    let split = data.sequences.len() - validation_count;
    let partition = |range: std::ops::Range<usize>| {
        let mut metadata = data.metadata.clone();
        metadata.total_samples = range.len();
        TrainingDataset {
            sequences: data.sequences[range.clone()].to_vec(),
            labels: data.labels[range].to_vec(),
            metadata,
        }
    };
    Ok((partition(0..split), partition(split..data.sequences.len())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DatasetMetadata, HyperparameterOptimizer, ModelEvaluator, NeuroDivergentModel,
        OptimizationMethod, ParameterRange, SearchSpace, TimeSeriesSequence, TrainingMetrics,
    };
    use std::collections::HashMap;

    fn dataset() -> TrainingDataset {
        TrainingDataset {
            sequences: (0..10)
                .map(|index| TimeSeriesSequence {
                    timestamps: vec![index],
                    features: vec![vec![index as f64 / 10.0]],
                    agent_id: "sample".into(),
                })
                .collect(),
            labels: (0..10).map(|index| vec![index as f64 / 10.0]).collect(),
            metadata: DatasetMetadata {
                total_samples: 10,
                feature_count: 1,
                sequence_length: 1,
                agents: vec!["sample".into()],
            },
        }
    }

    // A real closed-form mean baseline used to verify application backend boundaries.
    struct MeanBaseline {
        mean: f64,
    }
    impl NeuroDivergentModel for MeanBaseline {
        fn name(&self) -> &str {
            "mean baseline"
        }
        fn train(&mut self, data: &TrainingDataset, _: &TrainingConfig) -> Result<TrainingMetrics> {
            assert!(data
                .sequences
                .iter()
                .all(|sequence| sequence.timestamps[0] < 8));
            self.mean = data
                .labels
                .iter()
                .map(|row| row[0] / data.labels.len() as f64)
                .sum();
            let loss = data
                .labels
                .iter()
                .map(|row| (row[0] - self.mean).powi(2))
                .sum::<f64>()
                / data.labels.len() as f64;
            Ok(TrainingMetrics {
                epoch_losses: vec![(0, loss)],
                validation_losses: vec![],
                best_epoch: 0,
                best_loss: loss,
            })
        }
        fn predict(&self, sequence: &TimeSeriesSequence) -> Result<Vec<f64>> {
            assert!(
                sequence.timestamps[0] >= 8,
                "Validation data must not overlap training rows"
            );
            Ok(vec![self.mean])
        }
        fn save(&self, _: &str) -> Result<()> {
            Err(TrainingError::ModelError(
                "Persistence is outside this fixture".into(),
            ))
        }
        fn load(&mut self, _: &str) -> Result<()> {
            Err(TrainingError::ModelError(
                "Persistence is outside this fixture".into(),
            ))
        }
        fn get_hyperparameters(&self) -> HashMap<String, f64> {
            HashMap::new()
        }
        fn set_hyperparameters(&mut self, _: HashMap<String, f64>) {}
    }

    #[tokio::test]
    async fn optimization_uses_distinct_training_and_validation_rows() {
        let optimizer = HyperparameterOptimizer::new(
            SearchSpace {
                parameters: HashMap::new(),
            },
            OptimizationMethod::RandomSearch,
            3,
        );
        let result = optimizer
            .optimize(
                || Box::new(MeanBaseline { mean: 0.0 }),
                &dataset(),
                &TrainingConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(result.trial_results.len(), 3);
        // Learned mean of0..0.7 is0.35; held-out0.8/0.9 MSE is0.2525.
        assert!((result.best_score - 0.2525).abs() < 1e-12);
    }

    #[test]
    fn evaluation_rejects_empty_or_misaligned_data() {
        let mut data = dataset();
        data.labels.pop();
        assert!(validate_dataset(&data).is_err());
        assert!(ModelEvaluator::new()
            .evaluate_and_select(vec![], &dataset())
            .is_err());
        assert!(validate_prediction(&[f64::NAN], &[1.0]).is_err());
        assert!(validate_prediction(&[], &[1.0]).is_err());
    }

    #[tokio::test]
    async fn unsupported_searches_and_invalid_ranges_fail_before_training() {
        for method in [
            OptimizationMethod::GridSearch,
            OptimizationMethod::BayesianOptimization,
            OptimizationMethod::Hyperband,
        ] {
            let optimizer = HyperparameterOptimizer::new(
                SearchSpace {
                    parameters: HashMap::new(),
                },
                method,
                1,
            );
            assert!(optimizer
                .optimize(
                    || panic!("Unsupported search must not create a model"),
                    &dataset(),
                    &TrainingConfig::default()
                )
                .await
                .is_err());
        }
        let optimizer = HyperparameterOptimizer::new(
            SearchSpace {
                parameters: HashMap::from([(
                    "hidden_size".into(),
                    ParameterRange::Discrete { values: vec![] },
                )]),
            },
            OptimizationMethod::RandomSearch,
            1,
        );
        assert!(optimizer
            .optimize(
                || panic!("Invalid ranges must not create a model"),
                &dataset(),
                &TrainingConfig::default()
            )
            .await
            .is_err());
    }

    #[test]
    fn constant_target_r_squared_is_finite() {
        use crate::{EvaluationMetric, R2Metric};
        assert_eq!(R2Metric.calculate(&[vec![1.0]], &[vec![1.0]]), 1.0);
        assert_eq!(R2Metric.calculate(&[vec![2.0]], &[vec![1.0]]), 0.0);
    }
}
