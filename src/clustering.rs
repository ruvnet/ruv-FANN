//! Optional e-means input clustering, shared with RuVector.
//!
//! Enable the `clustering` feature to build codebooks or group training inputs.
//! This module does not replace supervised neural-network training algorithms.

use crate::TrainingData;
use thiserror::Error as ThisError;

pub use ruvector_e_means::{fit, Config, Error, Finalization, Model};

/// Errors at the boundary between nested training inputs and flat e-means data.
#[derive(Debug, ThisError)]
pub enum InputClusteringError {
    #[error("input clustering requires at least one nonempty input row")]
    EmptyInputs,
    #[error("input row {row} has {actual} dimensions; expected {expected}")]
    RaggedInputs {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error(transparent)]
    Clustering(#[from] Error),
}

impl TrainingData<f32> {
    /// Cluster the input vectors without changing inputs, outputs or network weights.
    ///
    /// Outputs do not participate in unsupervised clustering. The shared e-means
    /// implementation validates the configuration and rejects non-finite coordinates.
    pub fn cluster_inputs(&self, config: &Config) -> Result<Model, InputClusteringError> {
        let dimensions = self
            .inputs
            .first()
            .map(Vec::len)
            .filter(|&dimensions| dimensions > 0)
            .ok_or(InputClusteringError::EmptyInputs)?;
        for (row, input) in self.inputs.iter().enumerate() {
            if input.len() != dimensions {
                return Err(InputClusteringError::RaggedInputs {
                    row,
                    expected: dimensions,
                    actual: input.len(),
                });
            }
        }
        let inputs: Vec<f32> = self.inputs.iter().flatten().copied().collect();
        Ok(fit(&inputs, dimensions, config)?)
    }
}
