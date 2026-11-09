use super::{
    add, budget, mul, numerical, payload, zeros, AutogradError, AutogradNetwork, AutogradWorkspace,
};
use crate::BatchFloat;
use std::mem::size_of;

/// AdamW hyperparameters. Decoupled weight decay applies to weights and biases.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig<T: BatchFloat> {
    pub learning_rate: T,
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub weight_decay: T,
}

impl<T: BatchFloat> Default for AdamWConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.001).expect("f32/f64 constant"),
            beta1: T::from(0.9).expect("f32/f64 constant"),
            beta2: T::from(0.999).expect("f32/f64 constant"),
            epsilon: T::from(1e-8).expect("f32/f64 constant"),
            weight_decay: T::from(0.01).expect("f32/f64 constant"),
        }
    }
}

/// Preallocated AdamW moments and proposed moments, bound to one model revision.
/// Replacing parameters or using SGD invalidates this state; construct a fresh
/// optimizer to reset moments. Failed updates leave moments and time unchanged.
#[derive(Debug)]
pub struct AdamW<T: BatchFloat> {
    first: Vec<T>,
    second: Vec<T>,
    proposed_first: Vec<T>,
    proposed_second: Vec<T>,
    config: AdamWConfig<T>,
    model_id: usize,
    revision: u64,
    steps: u64,
    beta1_power: T,
    beta2_power: T,
    estimated_bytes: usize,
}

impl<T: BatchFloat> AdamW<T> {
    /// Preflight four moment buffers. Model/workspace bytes are excluded.
    pub fn required_bytes(network: &AutogradNetwork<T>) -> Result<usize, AutogradError> {
        add(
            size_of::<Self>(),
            payload::<T>(mul(4, network.parameter_count())?)?,
        )
    }
    pub fn new(
        network: &AutogradNetwork<T>,
        config: AdamWConfig<T>,
        max_bytes: usize,
    ) -> Result<Self, AutogradError> {
        validate_config(config)?;
        let estimated_bytes = Self::required_bytes(network)?;
        budget("AdamW bytes", estimated_bytes, max_bytes)?;
        Ok(Self {
            first: zeros(network.parameter_count())?,
            second: zeros(network.parameter_count())?,
            proposed_first: zeros(network.parameter_count())?,
            proposed_second: zeros(network.parameter_count())?,
            config,
            model_id: network.id,
            revision: network.revision,
            steps: 0,
            beta1_power: T::one(),
            beta2_power: T::one(),
            estimated_bytes,
        })
    }
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }
    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn config(&self) -> AdamWConfig<T> {
        self.config
    }
    pub fn first_moments(&self) -> &[T] {
        &self.first
    }
    pub fn second_moments(&self) -> &[T] {
        &self.second
    }
}

impl<T: BatchFloat> AutogradNetwork<T> {
    /// Stage and validate every SGD proposal before changing any parameter.
    pub fn sgd_step(
        &mut self,
        workspace: &mut AutogradWorkspace<T>,
        learning_rate: T,
    ) -> Result<(), AutogradError> {
        positive(learning_rate, "learning_rate")?;
        self.check_gradients(workspace)?;
        let revision = self.next_revision()?;
        for (index, &parameter) in self.parameters.iter().enumerate() {
            workspace.proposed[index] = numerical(
                parameter - learning_rate * workspace.gradients[index],
                "SGD parameter",
                0,
                index,
            )?;
        }
        self.parameters.copy_from_slice(&workspace.proposed);
        self.revision = revision;
        workspace.gradient_revision = None;
        Ok(())
    }

    /// Standard bias-corrected AdamW with decoupled decay, including biases.
    /// Model parameters, both moments, bias-correction powers, time and revision
    /// commit together only after all proposed values have passed finite checks.
    pub fn adamw_step(
        &mut self,
        workspace: &mut AutogradWorkspace<T>,
        optimizer: &mut AdamW<T>,
    ) -> Result<(), AutogradError> {
        self.check_gradients(workspace)?;
        if optimizer.model_id != self.id {
            return Err(AutogradError::OptimizerMismatch);
        }
        if optimizer.revision != self.revision {
            return Err(AutogradError::StaleOptimizer);
        }
        let revision = self.next_revision()?;
        let steps = optimizer
            .steps
            .checked_add(1)
            .ok_or(AutogradError::RevisionExhausted)?;
        let config = optimizer.config;
        let one = T::one();
        let beta1_power = numerical(
            optimizer.beta1_power * config.beta1,
            "AdamW beta1 power",
            0,
            0,
        )?;
        let beta2_power = numerical(
            optimizer.beta2_power * config.beta2,
            "AdamW beta2 power",
            0,
            0,
        )?;
        let first_correction = one - beta1_power;
        let second_correction = one - beta2_power;
        let decay = numerical(
            one - config.learning_rate * config.weight_decay,
            "AdamW decay",
            0,
            0,
        )?;
        for (index, &parameter) in self.parameters.iter().enumerate() {
            let gradient = workspace.gradients[index];
            let square = numerical(gradient * gradient, "AdamW gradient square", 0, index)?;
            let first = numerical(
                config.beta1 * optimizer.first[index] + (one - config.beta1) * gradient,
                "AdamW first moment",
                0,
                index,
            )?;
            let second = numerical(
                config.beta2 * optimizer.second[index] + (one - config.beta2) * square,
                "AdamW second moment",
                0,
                index,
            )?;
            let first_hat =
                numerical(first / first_correction, "AdamW first correction", 0, index)?;
            let second_hat = numerical(
                second / second_correction,
                "AdamW second correction",
                0,
                index,
            )?;
            let denominator = numerical(
                second_hat.sqrt() + config.epsilon,
                "AdamW denominator",
                0,
                index,
            )?;
            workspace.proposed[index] = numerical(
                parameter * decay - config.learning_rate * (first_hat / denominator),
                "AdamW parameter",
                0,
                index,
            )?;
            optimizer.proposed_first[index] = first;
            optimizer.proposed_second[index] = second;
        }
        self.parameters.copy_from_slice(&workspace.proposed);
        optimizer.first.copy_from_slice(&optimizer.proposed_first);
        optimizer.second.copy_from_slice(&optimizer.proposed_second);
        optimizer.beta1_power = beta1_power;
        optimizer.beta2_power = beta2_power;
        optimizer.steps = steps;
        optimizer.revision = revision;
        self.revision = revision;
        workspace.gradient_revision = None;
        Ok(())
    }
}

fn positive<T: BatchFloat>(value: T, field: &'static str) -> Result<(), AutogradError> {
    if !value.is_finite() || value <= T::zero() {
        Err(AutogradError::InvalidOptimizerConfig { field })
    } else {
        Ok(())
    }
}
fn validate_config<T: BatchFloat>(config: AdamWConfig<T>) -> Result<(), AutogradError> {
    positive(config.learning_rate, "learning_rate")?;
    positive(config.epsilon, "epsilon")?;
    for (value, field) in [(config.beta1, "beta1"), (config.beta2, "beta2")] {
        if !value.is_finite() || value < T::zero() || value >= T::one() {
            return Err(AutogradError::InvalidOptimizerConfig { field });
        }
    }
    if !config.weight_decay.is_finite() || config.weight_decay < T::zero() {
        return Err(AutogradError::InvalidOptimizerConfig {
            field: "weight_decay",
        });
    }
    Ok(())
}
