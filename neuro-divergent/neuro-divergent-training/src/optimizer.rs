//! # Optimizers for Neural Forecasting
//!
//! Modern optimization algorithms with forecasting-specific enhancements.
//! All optimizers implement adaptive learning rates and are designed to handle
//! the unique challenges of time series forecasting.
//!
//! ## Available Optimizers
//!
//! - **Adam**: Adaptive Moment Estimation with bias correction
//! - **AdamW**: Adam with weight decay (decoupled regularization)  
//! - **SGD**: Stochastic Gradient Descent with momentum and Nesterov acceleration
//! - **RMSprop**: RMSprop with forecasting-specific adaptations
//! - **ForecastingAdam**: Custom Adam variant optimized for temporal patterns
//!
//! ## Features
//!
//! - Gradient clipping support
//! - Mixed precision training compatibility
//! - Memory-efficient parameter updates
//! - Temporal pattern awareness for forecasting models

use crate::{TrainingError, TrainingResult};
use num_traits::Float;
use std::collections::HashMap;
use std::marker::PhantomData;
#[path = "optimizer_validation.rs"]
mod validation;
use validation::*;

/// Core trait for optimization algorithms
pub trait Optimizer<T: Float + Send + Sync>: Send + Sync {
    /// Apply parameter updates given gradients
    fn step(&mut self, parameters: &mut [Vec<T>], gradients: &[Vec<T>]) -> TrainingResult<()>;

    /// Get current learning rate
    fn learning_rate(&self) -> T;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: T);

    /// Reset optimizer state (e.g., momentum terms)
    fn reset(&mut self);

    /// Get optimizer name
    fn name(&self) -> &'static str;

    /// Get optimizer state for checkpointing
    fn state(&self) -> OptimizerState<T>;

    /// Restore optimizer from state
    fn restore_state(&mut self, state: OptimizerState<T>) -> TrainingResult<()>;

    /// Zero gradients (optional, some optimizers may accumulate)
    fn zero_grad(&mut self) {}

    /// Clip gradients if supported
    fn clip_gradients(&mut self, gradients: &mut [Vec<T>], max_norm: T) -> T {
        crate::utils::clip_gradients_by_norm(gradients, max_norm)
    }
}

/// Optimizer state for serialization and checkpointing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizerState<T: Float + Send + Sync> {
    pub step_count: usize,
    pub learning_rate: T,
    pub momentum_buffers: Vec<Vec<T>>,
    pub variance_buffers: Vec<Vec<T>>,
    pub additional_state: HashMap<String, Vec<T>>,
}

/// Optimizer wrapper enum for dynamic dispatch
#[derive(Clone)]
pub enum OptimizerType<T: Float + Send + Sync> {
    Adam(Adam<T>),
    AdamW(AdamW<T>),
    SGD(SGD<T>),
    RMSprop(RMSprop<T>),
    ForecastingAdam(ForecastingAdam<T>),
}

impl<T: Float + Send + Sync> Optimizer<T> for OptimizerType<T> {
    fn step(&mut self, parameters: &mut [Vec<T>], gradients: &[Vec<T>]) -> TrainingResult<()> {
        match self {
            OptimizerType::Adam(opt) => opt.step(parameters, gradients),
            OptimizerType::AdamW(opt) => opt.step(parameters, gradients),
            OptimizerType::SGD(opt) => opt.step(parameters, gradients),
            OptimizerType::RMSprop(opt) => opt.step(parameters, gradients),
            OptimizerType::ForecastingAdam(opt) => opt.step(parameters, gradients),
        }
    }

    fn learning_rate(&self) -> T {
        match self {
            OptimizerType::Adam(opt) => opt.learning_rate(),
            OptimizerType::AdamW(opt) => opt.learning_rate(),
            OptimizerType::SGD(opt) => opt.learning_rate(),
            OptimizerType::RMSprop(opt) => opt.learning_rate(),
            OptimizerType::ForecastingAdam(opt) => opt.learning_rate(),
        }
    }

    fn set_learning_rate(&mut self, lr: T) {
        match self {
            OptimizerType::Adam(opt) => opt.set_learning_rate(lr),
            OptimizerType::AdamW(opt) => opt.set_learning_rate(lr),
            OptimizerType::SGD(opt) => opt.set_learning_rate(lr),
            OptimizerType::RMSprop(opt) => opt.set_learning_rate(lr),
            OptimizerType::ForecastingAdam(opt) => opt.set_learning_rate(lr),
        }
    }

    fn reset(&mut self) {
        match self {
            OptimizerType::Adam(opt) => opt.reset(),
            OptimizerType::AdamW(opt) => opt.reset(),
            OptimizerType::SGD(opt) => opt.reset(),
            OptimizerType::RMSprop(opt) => opt.reset(),
            OptimizerType::ForecastingAdam(opt) => opt.reset(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            OptimizerType::Adam(opt) => opt.name(),
            OptimizerType::AdamW(opt) => opt.name(),
            OptimizerType::SGD(opt) => opt.name(),
            OptimizerType::RMSprop(opt) => opt.name(),
            OptimizerType::ForecastingAdam(opt) => opt.name(),
        }
    }

    fn state(&self) -> OptimizerState<T> {
        match self {
            OptimizerType::Adam(opt) => opt.state(),
            OptimizerType::AdamW(opt) => opt.state(),
            OptimizerType::SGD(opt) => opt.state(),
            OptimizerType::RMSprop(opt) => opt.state(),
            OptimizerType::ForecastingAdam(opt) => opt.state(),
        }
    }

    fn restore_state(&mut self, state: OptimizerState<T>) -> TrainingResult<()> {
        match self {
            OptimizerType::Adam(opt) => opt.restore_state(state),
            OptimizerType::AdamW(opt) => opt.restore_state(state),
            OptimizerType::SGD(opt) => opt.restore_state(state),
            OptimizerType::RMSprop(opt) => opt.restore_state(state),
            OptimizerType::ForecastingAdam(opt) => opt.restore_state(state),
        }
    }
}

// =============================================================================
// Adam Optimizer
// =============================================================================

/// Adam: Adaptive Moment Estimation
#[derive(Clone)]
pub struct Adam<T: Float + Send + Sync> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    step_count: usize,
    momentum: Vec<Vec<T>>,
    variance: Vec<Vec<T>>,
    amsgrad: bool,
    max_variance: Vec<Vec<T>>,
}

impl<T: Float + Send + Sync> Adam<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: T::from(1e-8).unwrap(),
            step_count: 0,
            momentum: Vec::new(),
            variance: Vec::new(),
            amsgrad: false,
            max_variance: Vec::new(),
        }
    }

    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        if amsgrad && self.max_variance.is_empty() {
            self.max_variance = self.variance.clone();
        }
        if !amsgrad {
            self.max_variance.clear();
        }
        self
    }

    fn initialize_buffers(&mut self, parameters: &[Vec<T>]) {
        if self.momentum.is_empty() {
            self.momentum = parameters
                .iter()
                .map(|layer| vec![T::zero(); layer.len()])
                .collect();
            self.variance = parameters
                .iter()
                .map(|layer| vec![T::zero(); layer.len()])
                .collect();

            if self.amsgrad {
                self.max_variance = parameters
                    .iter()
                    .map(|layer| vec![T::zero(); layer.len()])
                    .collect();
            }
        }
    }
}

impl<T: Float + Send + Sync> Optimizer<T> for Adam<T> {
    fn step(&mut self, parameters: &mut [Vec<T>], gradients: &[Vec<T>]) -> TrainingResult<()> {
        if !self.learning_rate.is_finite()
            || self.learning_rate < T::zero()
            || !self.beta1.is_finite()
            || self.beta1 < T::zero()
            || self.beta1 >= T::one()
            || !self.beta2.is_finite()
            || self.beta2 < T::zero()
            || self.beta2 >= T::one()
            || !self.epsilon.is_finite()
            || self.epsilon <= T::zero()
        {
            return Err(TrainingError::OptimizerError(
                "Invalid Adam hyperparameters".into(),
            ));
        }
        if parameters.len() != gradients.len() {
            return Err(TrainingError::OptimizerError(
                "Parameter-gradient length mismatch".into(),
            ));
        }
        for (parameters, gradients) in parameters.iter().zip(gradients) {
            if parameters.len() != gradients.len()
                || parameters.iter().chain(gradients).any(|v| !v.is_finite())
            {
                return Err(TrainingError::OptimizerError(
                    "Invalid parameter/gradient shape or non-finite value".into(),
                ));
            }
        }
        if !self.momentum.is_empty()
            && (self.momentum.len() != parameters.len()
                || self
                    .momentum
                    .iter()
                    .zip(parameters.iter())
                    .any(|(buffer, params)| buffer.len() != params.len()))
        {
            return Err(TrainingError::OptimizerError(
                "Parameter shape changed; reset the optimizer".into(),
            ));
        }
        // Stage both parameters and moments: an overflowing update must not leave
        // a partially changed model or advance the optimizer's step counter.
        let mut next = self.clone();
        let mut next_parameters = parameters.to_vec();
        next.initialize_buffers(parameters);
        next.step_count = next.step_count.checked_add(1).ok_or_else(|| {
            TrainingError::OptimizerError("Optimizer step counter overflowed".into())
        })?;
        let step = T::from(next.step_count).ok_or_else(|| {
            TrainingError::OptimizerError("Optimizer step counter is not representable".into())
        })?;
        let correction1 = T::one() - self.beta1.powf(step);
        let correction2 = T::one() - self.beta2.powf(step);
        for (layer, (params, grads)) in next_parameters.iter_mut().zip(gradients).enumerate() {
            for (index, (param, &gradient)) in params.iter_mut().zip(grads).enumerate() {
                let momentum =
                    self.beta1 * next.momentum[layer][index] + (T::one() - self.beta1) * gradient;
                let variance = self.beta2 * next.variance[layer][index]
                    + ((T::one() - self.beta2) * gradient) * gradient;
                next.momentum[layer][index] = momentum;
                next.variance[layer][index] = variance;
                let variance = if self.amsgrad {
                    next.max_variance[layer][index] = next.max_variance[layer][index].max(variance);
                    next.max_variance[layer][index]
                } else {
                    variance
                };
                // Epsilon belongs outside sqrt(v_hat), after bias correction.
                let update = self.learning_rate * (momentum / correction1)
                    / ((variance / correction2).sqrt() + self.epsilon);
                let value = *param - update;
                if !momentum.is_finite() || !variance.is_finite() || !value.is_finite() {
                    return Err(TrainingError::OptimizerError(
                        "Adam update overflowed; clip or scale gradients".into(),
                    ));
                }
                *param = value;
            }
        }
        for (parameters, next_parameters) in parameters.iter_mut().zip(next_parameters) {
            *parameters = next_parameters;
        }
        *self = next;
        Ok(())
    }

    fn learning_rate(&self) -> T {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.step_count = 0;
        self.momentum.clear();
        self.variance.clear();
        self.max_variance.clear();
    }

    fn name(&self) -> &'static str {
        if self.amsgrad {
            "AMSGrad"
        } else {
            "Adam"
        }
    }

    fn state(&self) -> OptimizerState<T> {
        let mut additional_state = HashMap::new();
        if self.amsgrad {
            additional_state.insert(
                "max_variance".to_string(),
                self.max_variance.iter().flatten().copied().collect(),
            );
        }

        OptimizerState {
            step_count: self.step_count,
            learning_rate: self.learning_rate,
            momentum_buffers: self.momentum.clone(),
            variance_buffers: self.variance.clone(),
            additional_state,
        }
    }

    fn restore_state(&mut self, state: OptimizerState<T>) -> TrainingResult<()> {
        if !state.learning_rate.is_finite()
            || state.learning_rate < T::zero()
            || state.momentum_buffers.len() != state.variance_buffers.len()
            || state
                .momentum_buffers
                .iter()
                .zip(state.variance_buffers.iter())
                .any(|(momentum, variance)| {
                    momentum.len() != variance.len()
                        || momentum.iter().any(|value| !value.is_finite())
                        || variance
                            .iter()
                            .any(|value| !value.is_finite() || *value < T::zero())
                })
        {
            return Err(TrainingError::OptimizerError(
                "Invalid Adam checkpoint buffers".to_string(),
            ));
        }
        let max_variance = if self.amsgrad {
            let flat = state.additional_state.get("max_variance").ok_or_else(|| {
                TrainingError::OptimizerError(
                    "AMSGrad checkpoint is missing maximum variances".to_string(),
                )
            })?;
            let expected: usize = state.variance_buffers.iter().map(Vec::len).sum();
            if flat.len() != expected
                || flat
                    .iter()
                    .any(|value| !value.is_finite() || *value < T::zero())
            {
                return Err(TrainingError::OptimizerError(
                    "Invalid AMSGrad checkpoint buffers".to_string(),
                ));
            }
            let mut offset = 0;
            state
                .variance_buffers
                .iter()
                .map(|row| {
                    let buffer = flat[offset..offset + row.len()].to_vec();
                    offset += row.len();
                    buffer
                })
                .collect()
        } else {
            Vec::new()
        };
        self.step_count = state.step_count;
        self.learning_rate = state.learning_rate;
        self.momentum = state.momentum_buffers;
        self.variance = state.variance_buffers;
        self.max_variance = max_variance;
        Ok(())
    }
}

// =============================================================================
// AdamW Optimizer
// =============================================================================

/// AdamW: Adam with decoupled weight decay
#[derive(Clone)]
pub struct AdamW<T: Float + Send + Sync> {
    adam: Adam<T>,
    weight_decay: T,
}

impl<T: Float + Send + Sync> AdamW<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T, weight_decay: T) -> Self {
        Self {
            adam: Adam::new(learning_rate, beta1, beta2),
            weight_decay,
        }
    }

    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.adam = self.adam.with_epsilon(epsilon);
        self
    }
}

impl<T: Float + Send + Sync> Optimizer<T> for AdamW<T> {
    fn step(&mut self, parameters: &mut [Vec<T>], gradients: &[Vec<T>]) -> TrainingResult<()> {
        if !self.weight_decay.is_finite() || self.weight_decay < T::zero() {
            return Err(TrainingError::OptimizerError(
                "AdamW weight decay must be finite and nonnegative".into(),
            ));
        }
        let mut candidate = parameters.to_vec();
        let mut optimizer = self.adam.clone();
        let factor = T::one() - self.adam.learning_rate * self.weight_decay;
        for parameter in candidate.iter_mut().flatten() {
            *parameter = *parameter * factor;
        }
        optimizer.step(&mut candidate, gradients)?;
        for (parameters, candidate) in parameters.iter_mut().zip(candidate) {
            *parameters = candidate;
        }
        self.adam = optimizer;
        Ok(())
    }

    fn learning_rate(&self) -> T {
        self.adam.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.adam.set_learning_rate(lr);
    }

    fn reset(&mut self) {
        self.adam.reset();
    }

    fn name(&self) -> &'static str {
        "AdamW"
    }

    fn state(&self) -> OptimizerState<T> {
        let mut state = self.adam.state();
        state
            .additional_state
            .insert("weight_decay".to_string(), vec![self.weight_decay]);
        state
    }

    fn restore_state(&mut self, mut state: OptimizerState<T>) -> TrainingResult<()> {
        if let Some(wd) = state.additional_state.remove("weight_decay") {
            if !wd.is_empty() {
                self.weight_decay = wd[0];
            }
        }
        self.adam.restore_state(state)
    }
}

// =============================================================================
// SGD Optimizer
// =============================================================================

/// SGD: Stochastic Gradient Descent with momentum and Nesterov acceleration
#[derive(Clone)]
pub struct SGD<T: Float + Send + Sync> {
    learning_rate: T,
    momentum: T,
    dampening: T,
    weight_decay: T,
    nesterov: bool,
    momentum_buffers: Vec<Vec<T>>,
}

impl<T: Float + Send + Sync> SGD<T> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            momentum: T::zero(),
            dampening: T::zero(),
            weight_decay: T::zero(),
            nesterov: false,
            momentum_buffers: Vec::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_dampening(mut self, dampening: T) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    fn initialize_buffers(&mut self, parameters: &[Vec<T>]) {
        if self.momentum_buffers.is_empty() && !self.momentum.is_zero() {
            self.momentum_buffers = parameters
                .iter()
                .map(|layer| vec![T::zero(); layer.len()])
                .collect();
        }
    }
}

impl<T: Float + Send + Sync> Optimizer<T> for SGD<T> {
    fn step(&mut self, parameters: &mut [Vec<T>], gradients: &[Vec<T>]) -> TrainingResult<()> {
        validate_step(parameters, gradients, self.learning_rate)?;
        if !unit_interval(self.momentum, false)
            || !unit_interval(self.dampening, true)
            || !nonnegative(self.weight_decay)
            || (self.nesterov && (self.momentum == T::zero() || self.dampening != T::zero()))
        {
            return Err(TrainingError::OptimizerError(
                "Invalid SGD hyperparameters".into(),
            ));
        }
        validate_buffers(&self.momentum_buffers, parameters, false)?;
        let mut candidate = self.clone();
        let mut next_parameters = parameters.to_vec();
        candidate.initialize_buffers(&next_parameters);

        for (layer_idx, (params, grads)) in
            next_parameters.iter_mut().zip(gradients.iter()).enumerate()
        {
            for (param_idx, (&grad, param)) in grads.iter().zip(params.iter_mut()).enumerate() {
                let mut d_param = grad;

                // Add weight decay
                if !candidate.weight_decay.is_zero() {
                    d_param = d_param + candidate.weight_decay * *param;
                }

                // Apply momentum
                if !candidate.momentum.is_zero() {
                    let momentum_buffer = &mut candidate.momentum_buffers[layer_idx][param_idx];
                    *momentum_buffer = candidate.momentum * *momentum_buffer
                        + (T::one() - candidate.dampening) * d_param;

                    if candidate.nesterov {
                        d_param = d_param + candidate.momentum * *momentum_buffer;
                    } else {
                        d_param = *momentum_buffer;
                    }
                }

                // Update parameter
                *param = *param - candidate.learning_rate * d_param;
            }
        }

        validate_step(&next_parameters, gradients, self.learning_rate)?;
        validate_buffers(&candidate.momentum_buffers, &next_parameters, false)?;
        for (parameters, values) in parameters.iter_mut().zip(next_parameters) {
            *parameters = values;
        }
        *self = candidate;
        Ok(())
    }

    fn learning_rate(&self) -> T {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.momentum_buffers.clear();
    }

    fn name(&self) -> &'static str {
        if self.nesterov {
            "SGD-Nesterov"
        } else {
            "SGD"
        }
    }

    fn state(&self) -> OptimizerState<T> {
        OptimizerState {
            step_count: 0,
            learning_rate: self.learning_rate,
            momentum_buffers: self.momentum_buffers.clone(),
            variance_buffers: Vec::new(),
            additional_state: HashMap::from([
                ("momentum".to_string(), vec![self.momentum]),
                ("dampening".to_string(), vec![self.dampening]),
                ("weight_decay".to_string(), vec![self.weight_decay]),
            ]),
        }
    }

    fn restore_state(&mut self, state: OptimizerState<T>) -> TrainingResult<()> {
        self.learning_rate = state.learning_rate;
        self.momentum_buffers = state.momentum_buffers;

        if let Some(momentum) = state.additional_state.get("momentum") {
            if !momentum.is_empty() {
                self.momentum = momentum[0];
            }
        }
        if let Some(dampening) = state.additional_state.get("dampening") {
            if !dampening.is_empty() {
                self.dampening = dampening[0];
            }
        }
        if let Some(weight_decay) = state.additional_state.get("weight_decay") {
            if !weight_decay.is_empty() {
                self.weight_decay = weight_decay[0];
            }
        }

        Ok(())
    }
}

// =============================================================================
// RMSprop Optimizer
// =============================================================================

/// RMSprop: Root Mean Square Propagation with forecasting adaptations
#[derive(Clone)]
pub struct RMSprop<T: Float + Send + Sync> {
    learning_rate: T,
    alpha: T,
    epsilon: T,
    weight_decay: T,
    momentum: T,
    centered: bool,
    square_avg: Vec<Vec<T>>,
    momentum_buffer: Vec<Vec<T>>,
    grad_avg: Vec<Vec<T>>,
}

impl<T: Float + Send + Sync> RMSprop<T> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            alpha: T::from(0.99).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            weight_decay: T::zero(),
            momentum: T::zero(),
            centered: false,
            square_avg: Vec::new(),
            momentum_buffer: Vec::new(),
            grad_avg: Vec::new(),
        }
    }

    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    fn initialize_buffers(&mut self, parameters: &[Vec<T>]) {
        let zero_buffers = || {
            parameters
                .iter()
                .map(|row| vec![T::zero(); row.len()])
                .collect()
        };
        if self.square_avg.is_empty() {
            self.square_avg = zero_buffers();
        }
        if !self.momentum.is_zero() && self.momentum_buffer.is_empty() {
            self.momentum_buffer = zero_buffers();
        }
        if self.centered && self.grad_avg.is_empty() {
            self.grad_avg = zero_buffers();
        }
    }
}

impl<T: Float + Send + Sync> Optimizer<T> for RMSprop<T> {
    fn step(&mut self, parameters: &mut [Vec<T>], gradients: &[Vec<T>]) -> TrainingResult<()> {
        validate_step(parameters, gradients, self.learning_rate)?;
        if !unit_interval(self.alpha, false)
            || !unit_interval(self.momentum, false)
            || !nonnegative(self.weight_decay)
            || !self.epsilon.is_finite()
            || self.epsilon <= T::zero()
        {
            return Err(TrainingError::OptimizerError(
                "Invalid RMSprop hyperparameters".into(),
            ));
        }
        validate_buffers(&self.square_avg, parameters, true)?;
        validate_buffers(&self.momentum_buffer, parameters, false)?;
        validate_buffers(&self.grad_avg, parameters, false)?;
        let mut candidate = self.clone();
        let mut next_parameters = parameters.to_vec();
        candidate.initialize_buffers(&next_parameters);

        for (layer_idx, (params, grads)) in
            next_parameters.iter_mut().zip(gradients.iter()).enumerate()
        {
            for (param_idx, (&grad, param)) in grads.iter().zip(params.iter_mut()).enumerate() {
                let mut d_param = grad;

                // Add weight decay
                if !candidate.weight_decay.is_zero() {
                    d_param = d_param + candidate.weight_decay * *param;
                }

                // Update square average
                candidate.square_avg[layer_idx][param_idx] = candidate.alpha
                    * candidate.square_avg[layer_idx][param_idx]
                    + (T::one() - candidate.alpha) * d_param * d_param;

                let mut avg = candidate.square_avg[layer_idx][param_idx];

                if candidate.centered {
                    // Update gradient average
                    candidate.grad_avg[layer_idx][param_idx] = candidate.alpha
                        * candidate.grad_avg[layer_idx][param_idx]
                        + (T::one() - candidate.alpha) * d_param;

                    let grad_avg = candidate.grad_avg[layer_idx][param_idx];
                    avg = (avg - grad_avg * grad_avg).max(T::zero());
                }

                let update = d_param / (avg.sqrt() + candidate.epsilon);

                if !candidate.momentum.is_zero() {
                    candidate.momentum_buffer[layer_idx][param_idx] = candidate.momentum
                        * candidate.momentum_buffer[layer_idx][param_idx]
                        + update;
                    *param = *param
                        - candidate.learning_rate * candidate.momentum_buffer[layer_idx][param_idx];
                } else {
                    *param = *param - candidate.learning_rate * update;
                }
            }
        }

        validate_step(&next_parameters, gradients, self.learning_rate)?;
        validate_buffers(&candidate.square_avg, &next_parameters, true)?;
        validate_buffers(&candidate.momentum_buffer, &next_parameters, false)?;
        validate_buffers(&candidate.grad_avg, &next_parameters, false)?;
        for (parameters, values) in parameters.iter_mut().zip(next_parameters) {
            *parameters = values;
        }
        *self = candidate;
        Ok(())
    }

    fn learning_rate(&self) -> T {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.square_avg.clear();
        self.momentum_buffer.clear();
        self.grad_avg.clear();
    }

    fn name(&self) -> &'static str {
        if self.centered {
            "RMSprop-Centered"
        } else {
            "RMSprop"
        }
    }

    fn state(&self) -> OptimizerState<T> {
        let mut additional_state = HashMap::new();
        additional_state.insert("alpha".to_string(), vec![self.alpha]);
        additional_state.insert("weight_decay".to_string(), vec![self.weight_decay]);
        additional_state.insert("momentum".to_string(), vec![self.momentum]);

        if self.centered {
            additional_state.insert(
                "grad_avg".to_string(),
                self.grad_avg.iter().flatten().copied().collect(),
            );
        }

        OptimizerState {
            step_count: 0,
            learning_rate: self.learning_rate,
            momentum_buffers: self.momentum_buffer.clone(),
            variance_buffers: self.square_avg.clone(),
            additional_state,
        }
    }

    fn restore_state(&mut self, state: OptimizerState<T>) -> TrainingResult<()> {
        self.learning_rate = state.learning_rate;
        self.momentum_buffer = state.momentum_buffers;
        self.square_avg = state.variance_buffers;

        if let Some(alpha) = state.additional_state.get("alpha") {
            if !alpha.is_empty() {
                self.alpha = alpha[0];
            }
        }
        if let Some(weight_decay) = state.additional_state.get("weight_decay") {
            if !weight_decay.is_empty() {
                self.weight_decay = weight_decay[0];
            }
        }
        if let Some(momentum) = state.additional_state.get("momentum") {
            if !momentum.is_empty() {
                self.momentum = momentum[0];
            }
        }

        if self.centered {
            if let Some(grad_avg_flat) = state.additional_state.get("grad_avg") {
                // Reconstruct grad_avg from flattened data
                let mut idx = 0;
                for layer_grad_avg in &mut self.grad_avg {
                    for grad_avg in layer_grad_avg {
                        if idx < grad_avg_flat.len() {
                            *grad_avg = grad_avg_flat[idx];
                            idx += 1;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// Forecasting Adam Optimizer
// =============================================================================

/// ForecastingAdam: Custom Adam variant optimized for temporal patterns
#[derive(Clone)]
pub struct ForecastingAdam<T: Float + Send + Sync> {
    adam: Adam<T>,
    temporal_momentum: T,
    seasonal_correction: bool,
    lookback_window: usize,
    gradient_history: Vec<Vec<Vec<T>>>,
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> ForecastingAdam<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T) -> Self {
        Self {
            adam: Adam::new(learning_rate, beta1, beta2),
            temporal_momentum: T::from(0.1).unwrap(),
            seasonal_correction: true,
            lookback_window: 10,
            gradient_history: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn with_temporal_momentum(mut self, temporal_momentum: T) -> Self {
        self.temporal_momentum = temporal_momentum;
        self
    }

    pub fn with_seasonal_correction(mut self, seasonal_correction: bool) -> Self {
        self.seasonal_correction = seasonal_correction;
        self
    }

    pub fn with_lookback_window(mut self, lookback_window: usize) -> Self {
        self.lookback_window = lookback_window;
        self
    }

    fn update_gradient_history(&mut self, gradients: &[Vec<T>]) {
        // Keep history of gradients for temporal pattern detection
        self.gradient_history.push(gradients.to_vec());

        if self.gradient_history.len() > self.lookback_window {
            self.gradient_history.remove(0);
        }
    }

    fn apply_temporal_correction(&self, gradients: &mut [Vec<T>]) {
        if self.gradient_history.len() < 2 {
            return;
        }

        // Apply temporal momentum to smooth out gradient updates
        let prev_gradients = &self.gradient_history[self.gradient_history.len() - 2];

        for (layer_idx, (current_grads, prev_grads)) in
            gradients.iter_mut().zip(prev_gradients.iter()).enumerate()
        {
            for (grad, &prev_grad) in current_grads.iter_mut().zip(prev_grads.iter()) {
                *grad = (T::one() - self.temporal_momentum) * *grad
                    + self.temporal_momentum * prev_grad;
            }
        }
    }

    fn apply_seasonal_correction(&self, gradients: &mut [Vec<T>]) {
        if !self.seasonal_correction || self.gradient_history.len() < self.lookback_window {
            return;
        }

        // Detect seasonal patterns in gradients and apply correction
        for layer_idx in 0..gradients.len() {
            for param_idx in 0..gradients[layer_idx].len() {
                let mut seasonal_avg = T::zero();
                let mut count = 0;

                // Look for seasonal patterns (e.g., every 7 steps for weekly seasonality)
                for step in (0..self.gradient_history.len()).step_by(7) {
                    if step < self.gradient_history.len()
                        && param_idx < self.gradient_history[step][layer_idx].len()
                    {
                        seasonal_avg =
                            seasonal_avg + self.gradient_history[step][layer_idx][param_idx];
                        count += 1;
                    }
                }

                if count > 0 {
                    seasonal_avg = seasonal_avg / T::from(count).unwrap();
                    let correction = T::from(0.1).unwrap();
                    gradients[layer_idx][param_idx] = (T::one() - correction)
                        * gradients[layer_idx][param_idx]
                        + correction * seasonal_avg;
                }
            }
        }
    }
}

impl<T: Float + Send + Sync> Optimizer<T> for ForecastingAdam<T> {
    fn step(&mut self, parameters: &mut [Vec<T>], gradients: &[Vec<T>]) -> TrainingResult<()> {
        validate_step(parameters, gradients, self.adam.learning_rate)?;
        if !unit_interval(self.temporal_momentum, true) || self.lookback_window == 0 {
            return Err(TrainingError::OptimizerError(
                "Invalid temporal momentum or zero lookback window".into(),
            ));
        }
        for history in &self.gradient_history {
            validate_step(parameters, history, self.adam.learning_rate)?;
        }
        let mut candidate = self.clone();
        let mut corrected_gradients = gradients.to_vec();
        candidate.apply_temporal_correction(&mut corrected_gradients);
        candidate.apply_seasonal_correction(&mut corrected_gradients);
        candidate.update_gradient_history(&corrected_gradients);
        candidate.adam.step(parameters, &corrected_gradients)?;
        *self = candidate;
        Ok(())
    }

    fn learning_rate(&self) -> T {
        self.adam.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.adam.set_learning_rate(lr);
    }

    fn reset(&mut self) {
        self.adam.reset();
        self.gradient_history.clear();
    }

    fn name(&self) -> &'static str {
        "ForecastingAdam"
    }

    fn state(&self) -> OptimizerState<T> {
        let mut state = self.adam.state();
        state.additional_state.insert(
            "temporal_momentum".to_string(),
            vec![self.temporal_momentum],
        );
        state.additional_state.insert(
            "lookback_window".to_string(),
            vec![T::from(self.lookback_window).unwrap()],
        );
        state
    }

    fn restore_state(&mut self, mut state: OptimizerState<T>) -> TrainingResult<()> {
        if let Some(temporal_momentum) = state.additional_state.remove("temporal_momentum") {
            if !temporal_momentum.is_empty() {
                self.temporal_momentum = temporal_momentum[0];
            }
        }
        if let Some(lookback) = state.additional_state.remove("lookback_window") {
            if !lookback.is_empty() {
                self.lookback_window = lookback[0].to_usize().unwrap_or(10);
            }
        }

        self.adam.restore_state(state)
    }
}

// =============================================================================
// Builder Pattern for Optimizers
// =============================================================================

/// Builder for creating optimizers with fluent interface
pub struct OptimizerBuilder<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> OptimizerBuilder<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    pub fn adam(learning_rate: T) -> Adam<T> {
        Adam::new(
            learning_rate,
            T::from(0.9).unwrap(),
            T::from(0.999).unwrap(),
        )
    }

    pub fn adamw(learning_rate: T, weight_decay: T) -> AdamW<T> {
        AdamW::new(
            learning_rate,
            T::from(0.9).unwrap(),
            T::from(0.999).unwrap(),
            weight_decay,
        )
    }

    pub fn sgd(learning_rate: T) -> SGD<T> {
        SGD::new(learning_rate)
    }

    pub fn rmsprop(learning_rate: T) -> RMSprop<T> {
        RMSprop::new(learning_rate)
    }

    pub fn forecasting_adam(learning_rate: T) -> ForecastingAdam<T> {
        ForecastingAdam::new(
            learning_rate,
            T::from(0.9).unwrap(),
            T::from(0.999).unwrap(),
        )
    }
}

impl<T: Float + Send + Sync> Default for OptimizerBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Adam::new(0.01, 0.9, 0.999);
        let mut parameters = vec![vec![1.0, 2.0], vec![3.0]];
        let gradients = vec![vec![0.1, -0.2], vec![0.05]];

        // First step
        optimizer.step(&mut parameters, &gradients).unwrap();

        // Parameters should have moved
        assert!(parameters[0][0] < 1.0); // Moved in negative gradient direction
        assert!(parameters[0][1] > 2.0); // Moved in positive gradient direction
        assert!(parameters[1][0] < 3.0); // Moved in negative gradient direction

        // Second step with same gradients
        let initial_param = parameters[0][0];
        optimizer.step(&mut parameters, &gradients).unwrap();

        // Should continue moving in same direction with momentum
        assert!(parameters[0][0] < initial_param);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut optimizer = SGD::new(0.01).with_momentum(0.9);
        let mut parameters = vec![vec![1.0]];
        let gradients = vec![vec![1.0]];

        // First step
        optimizer.step(&mut parameters, &gradients).unwrap();
        let first_step = 1.0 - parameters[0][0];

        // Second step with same gradients
        optimizer.step(&mut parameters, &gradients).unwrap();
        let second_step = 1.0 - first_step - parameters[0][0];

        // With momentum, second step should be larger
        assert!(second_step > first_step);
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut optimizer = AdamW::new(0.01, 0.9, 0.999, 0.01);
        let mut parameters = vec![vec![1.0]];
        let gradients = vec![vec![0.0]]; // No gradient

        let initial_param = parameters[0][0];
        optimizer.step(&mut parameters, &gradients).unwrap();

        // Even with zero gradient, parameter should decrease due to weight decay
        assert!(parameters[0][0] < initial_param);
    }

    #[test]
    fn test_optimizer_state_save_restore() {
        let mut optimizer = Adam::new(0.01, 0.9, 0.999);
        let mut parameters = vec![vec![1.0, 2.0]];
        let gradients = vec![vec![0.1, -0.2]];

        // Take a few steps
        optimizer.step(&mut parameters, &gradients).unwrap();
        optimizer.step(&mut parameters, &gradients).unwrap();

        // Save state
        let state = optimizer.state();
        let checkpoint_params = parameters.clone();

        // Continue training
        optimizer.step(&mut parameters, &gradients).unwrap();
        let expected_parameters = parameters.clone();
        optimizer.step(&mut parameters, &gradients).unwrap();

        // Restore state
        optimizer.restore_state(state).unwrap();
        parameters = checkpoint_params;

        // Take the same step again - should produce same result
        let mut test_params = parameters.clone();
        optimizer.step(&mut test_params, &gradients).unwrap();

        assert_eq!(optimizer.step_count, 3);
        assert_eq!(test_params, expected_parameters);
    }

    #[test]
    fn test_forecasting_adam() {
        let mut optimizer = ForecastingAdam::new(0.01, 0.9, 0.999);
        let mut parameters = vec![vec![1.0]];
        let gradients = vec![vec![0.1]];

        // Take several steps to build up gradient history
        for _ in 0..5 {
            optimizer.step(&mut parameters, &gradients).unwrap();
        }

        // Check that gradient history is being maintained
        assert!(optimizer.gradient_history.len() > 0);
        assert!(optimizer.gradient_history.len() <= optimizer.lookback_window);
    }
}

#[cfg(test)]
#[path = "optimizer_regression_tests.rs"]
mod regression_tests;
