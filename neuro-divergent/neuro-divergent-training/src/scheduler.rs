//! # Learning Rate Schedulers for Neural Forecasting
//!
//! Advanced learning rate scheduling strategies designed for time series forecasting models.
//! These schedulers adapt the learning rate during training to improve convergence and
//! handle the unique challenges of temporal data.
//!
//! ## Available Schedulers
//!
//! - **ExponentialScheduler**: Exponential decay with optional warmup
//! - **StepScheduler**: Step-wise decay at specified intervals
//! - **CosineScheduler**: Cosine annealing with restarts
//! - **PlateauScheduler**: Reduce on loss plateau with patience
//! - **WarmupScheduler**: Linear warmup followed by decay
//! - **CyclicScheduler**: Cyclic learning rates for better exploration
//! - **OneCycleScheduler**: One cycle policy for fast convergence
//! - **SeasonalScheduler**: Seasonal-aware scheduling for time series
//!
//! ## Features
//!
//! - Temporal pattern awareness
//! - Warmup and cooldown phases
//! - Adaptive scheduling based on validation metrics
//! - Support for cyclic and one-cycle policies
//! - Integration with early stopping mechanisms

use crate::{TrainingError, TrainingResult};
use num_traits::Float;
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Core trait for learning rate schedulers
pub trait LearningRateScheduler<T: Float + Send + Sync>: Send + Sync {
    /// Get the learning rate for the current step/epoch
    fn get_learning_rate(&mut self, step: usize) -> T;

    /// Update scheduler state with current metrics
    fn step(&mut self, step: usize, metrics: Option<T>) -> TrainingResult<T>;

    /// Reset scheduler to initial state
    fn reset(&mut self);

    /// Get scheduler name
    fn name(&self) -> &'static str;

    /// Check if scheduler is finished (for one-cycle policies)
    fn is_finished(&self) -> bool {
        false
    }

    /// Get current learning rate without updating internal state
    fn current_lr(&self) -> T;

    /// Get scheduler state for checkpointing
    fn state(&self) -> SchedulerState<T>;

    /// Restore scheduler from state
    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()>;
}

/// Scheduler state for serialization and checkpointing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SchedulerState<T: Float + Send + Sync> {
    pub current_step: usize,
    pub current_lr: T,
    pub best_metric: Option<T>,
    pub patience_counter: usize,
    pub cycle_count: usize,
    pub warmup_steps: usize,
    pub additional_state: std::collections::HashMap<String, Vec<T>>,
}

/// Scheduler wrapper enum for dynamic dispatch
#[derive(Clone)]
pub enum SchedulerType<T: Float + Send + Sync> {
    Exponential(ExponentialScheduler<T>),
    Step(StepScheduler<T>),
    Cosine(CosineScheduler<T>),
    Plateau(PlateauScheduler<T>),
    Warmup(WarmupScheduler<T>),
    Cyclic(CyclicScheduler<T>),
    OneCycle(OneCycleScheduler<T>),
    Seasonal(SeasonalScheduler<T>),
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for SchedulerType<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        match self {
            SchedulerType::Exponential(s) => s.get_learning_rate(step),
            SchedulerType::Step(s) => s.get_learning_rate(step),
            SchedulerType::Cosine(s) => s.get_learning_rate(step),
            SchedulerType::Plateau(s) => s.get_learning_rate(step),
            SchedulerType::Warmup(s) => s.get_learning_rate(step),
            SchedulerType::Cyclic(s) => s.get_learning_rate(step),
            SchedulerType::OneCycle(s) => s.get_learning_rate(step),
            SchedulerType::Seasonal(s) => s.get_learning_rate(step),
        }
    }

    fn step(&mut self, step: usize, metrics: Option<T>) -> TrainingResult<T> {
        match self {
            SchedulerType::Exponential(s) => s.step(step, metrics),
            SchedulerType::Step(s) => s.step(step, metrics),
            SchedulerType::Cosine(s) => s.step(step, metrics),
            SchedulerType::Plateau(s) => s.step(step, metrics),
            SchedulerType::Warmup(s) => s.step(step, metrics),
            SchedulerType::Cyclic(s) => s.step(step, metrics),
            SchedulerType::OneCycle(s) => s.step(step, metrics),
            SchedulerType::Seasonal(s) => s.step(step, metrics),
        }
    }

    fn reset(&mut self) {
        match self {
            SchedulerType::Exponential(s) => s.reset(),
            SchedulerType::Step(s) => s.reset(),
            SchedulerType::Cosine(s) => s.reset(),
            SchedulerType::Plateau(s) => s.reset(),
            SchedulerType::Warmup(s) => s.reset(),
            SchedulerType::Cyclic(s) => s.reset(),
            SchedulerType::OneCycle(s) => s.reset(),
            SchedulerType::Seasonal(s) => s.reset(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            SchedulerType::Exponential(s) => s.name(),
            SchedulerType::Step(s) => s.name(),
            SchedulerType::Cosine(s) => s.name(),
            SchedulerType::Plateau(s) => s.name(),
            SchedulerType::Warmup(s) => s.name(),
            SchedulerType::Cyclic(s) => s.name(),
            SchedulerType::OneCycle(s) => s.name(),
            SchedulerType::Seasonal(s) => s.name(),
        }
    }

    fn is_finished(&self) -> bool {
        match self {
            SchedulerType::Exponential(s) => s.is_finished(),
            SchedulerType::Step(s) => s.is_finished(),
            SchedulerType::Cosine(s) => s.is_finished(),
            SchedulerType::Plateau(s) => s.is_finished(),
            SchedulerType::Warmup(s) => s.is_finished(),
            SchedulerType::Cyclic(s) => s.is_finished(),
            SchedulerType::OneCycle(s) => s.is_finished(),
            SchedulerType::Seasonal(s) => s.is_finished(),
        }
    }

    fn current_lr(&self) -> T {
        match self {
            SchedulerType::Exponential(s) => s.current_lr(),
            SchedulerType::Step(s) => s.current_lr(),
            SchedulerType::Cosine(s) => s.current_lr(),
            SchedulerType::Plateau(s) => s.current_lr(),
            SchedulerType::Warmup(s) => s.current_lr(),
            SchedulerType::Cyclic(s) => s.current_lr(),
            SchedulerType::OneCycle(s) => s.current_lr(),
            SchedulerType::Seasonal(s) => s.current_lr(),
        }
    }

    fn state(&self) -> SchedulerState<T> {
        match self {
            SchedulerType::Exponential(s) => s.state(),
            SchedulerType::Step(s) => s.state(),
            SchedulerType::Cosine(s) => s.state(),
            SchedulerType::Plateau(s) => s.state(),
            SchedulerType::Warmup(s) => s.state(),
            SchedulerType::Cyclic(s) => s.state(),
            SchedulerType::OneCycle(s) => s.state(),
            SchedulerType::Seasonal(s) => s.state(),
        }
    }

    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()> {
        match self {
            SchedulerType::Exponential(s) => s.restore_state(state),
            SchedulerType::Step(s) => s.restore_state(state),
            SchedulerType::Cosine(s) => s.restore_state(state),
            SchedulerType::Plateau(s) => s.restore_state(state),
            SchedulerType::Warmup(s) => s.restore_state(state),
            SchedulerType::Cyclic(s) => s.restore_state(state),
            SchedulerType::OneCycle(s) => s.restore_state(state),
            SchedulerType::Seasonal(s) => s.restore_state(state),
        }
    }
}

// =============================================================================
// Exponential Scheduler
// =============================================================================

/// Exponential learning rate decay with optional warmup
#[derive(Clone)]
pub struct ExponentialScheduler<T: Float + Send + Sync> {
    initial_lr: T,
    decay_rate: T,
    current_lr: T,
    current_step: usize,
    warmup_steps: usize,
    min_lr: T,
}

impl<T: Float + Send + Sync> ExponentialScheduler<T> {
    pub fn new(initial_lr: T, decay_rate: T) -> Self {
        Self {
            initial_lr,
            decay_rate,
            current_lr: initial_lr,
            current_step: 0,
            warmup_steps: 0,
            min_lr: T::zero(),
        }
    }

    pub fn with_warmup(mut self, warmup_steps: usize) -> Self {
        self.warmup_steps = warmup_steps;
        self
    }

    pub fn with_min_lr(mut self, min_lr: T) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for ExponentialScheduler<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        self.current_step = step;

        if step < self.warmup_steps {
            // Linear warmup
            let warmup_factor = T::from(step).unwrap() / T::from(self.warmup_steps).unwrap();
            self.current_lr = self.initial_lr * warmup_factor;
        } else {
            // Exponential decay
            let decay_steps = step - self.warmup_steps;
            self.current_lr = self.initial_lr * self.decay_rate.powi(decay_steps as i32);
        }

        self.current_lr = self.current_lr.max(self.min_lr);
        self.current_lr
    }

    fn step(&mut self, step: usize, _metrics: Option<T>) -> TrainingResult<T> {
        Ok(self.get_learning_rate(step))
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_step = 0;
    }

    fn name(&self) -> &'static str {
        "ExponentialScheduler"
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        SchedulerState {
            current_step: self.current_step,
            current_lr: self.current_lr,
            best_metric: None,
            patience_counter: 0,
            cycle_count: 0,
            warmup_steps: self.warmup_steps,
            additional_state: std::collections::HashMap::from([
                ("initial_lr".to_string(), vec![self.initial_lr]),
                ("decay_rate".to_string(), vec![self.decay_rate]),
                ("min_lr".to_string(), vec![self.min_lr]),
            ]),
        }
    }

    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_step = state.current_step;
        self.current_lr = state.current_lr;
        self.warmup_steps = state.warmup_steps;

        if let Some(initial_lr) = state.additional_state.get("initial_lr") {
            if !initial_lr.is_empty() {
                self.initial_lr = initial_lr[0];
            }
        }
        if let Some(decay_rate) = state.additional_state.get("decay_rate") {
            if !decay_rate.is_empty() {
                self.decay_rate = decay_rate[0];
            }
        }
        if let Some(min_lr) = state.additional_state.get("min_lr") {
            if !min_lr.is_empty() {
                self.min_lr = min_lr[0];
            }
        }

        Ok(())
    }
}

// =============================================================================
// Step Scheduler
// =============================================================================

/// Step-wise learning rate decay
#[derive(Clone)]
pub struct StepScheduler<T: Float + Send + Sync> {
    initial_lr: T,
    step_size: usize,
    gamma: T,
    current_lr: T,
    current_step: usize,
    milestones: Vec<usize>,
}

impl<T: Float + Send + Sync> StepScheduler<T> {
    pub fn new(initial_lr: T, step_size: usize, gamma: T) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_lr: initial_lr,
            current_step: 0,
            milestones: Vec::new(),
        }
    }

    pub fn with_milestones(mut self, milestones: Vec<usize>) -> Self {
        self.milestones = milestones;
        self.milestones.sort();
        self
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for StepScheduler<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        self.current_step = step;

        if !self.milestones.is_empty() {
            // Use milestones
            let num_reductions = self
                .milestones
                .iter()
                .take_while(|&&milestone| milestone <= step)
                .count();
            self.current_lr = self.initial_lr * self.gamma.powi(num_reductions as i32);
        } else {
            // Use step_size
            let num_reductions = step / self.step_size;
            self.current_lr = self.initial_lr * self.gamma.powi(num_reductions as i32);
        }

        self.current_lr
    }

    fn step(&mut self, step: usize, _metrics: Option<T>) -> TrainingResult<T> {
        Ok(self.get_learning_rate(step))
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_step = 0;
    }

    fn name(&self) -> &'static str {
        "StepScheduler"
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        SchedulerState {
            current_step: self.current_step,
            current_lr: self.current_lr,
            best_metric: None,
            patience_counter: 0,
            cycle_count: 0,
            warmup_steps: 0,
            additional_state: std::collections::HashMap::from([
                ("initial_lr".to_string(), vec![self.initial_lr]),
                (
                    "step_size".to_string(),
                    vec![T::from(self.step_size).unwrap()],
                ),
                ("gamma".to_string(), vec![self.gamma]),
            ]),
        }
    }

    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_step = state.current_step;
        self.current_lr = state.current_lr;

        if let Some(initial_lr) = state.additional_state.get("initial_lr") {
            if !initial_lr.is_empty() {
                self.initial_lr = initial_lr[0];
            }
        }
        if let Some(step_size) = state.additional_state.get("step_size") {
            if !step_size.is_empty() {
                self.step_size = step_size[0].to_usize().unwrap_or(1);
            }
        }
        if let Some(gamma) = state.additional_state.get("gamma") {
            if !gamma.is_empty() {
                self.gamma = gamma[0];
            }
        }

        Ok(())
    }
}

// =============================================================================
// Cosine Scheduler
// =============================================================================

/// Cosine annealing learning rate scheduler with restarts
#[derive(Clone)]
pub struct CosineScheduler<T: Float + Send + Sync> {
    restarts_enabled: bool,
    initial_lr: T,
    min_lr: T,
    t_max: usize,
    current_lr: T,
    current_step: usize,
    restart_factor: T,
    restart_count: usize,
    current_t_max: usize,
}

impl<T: Float + Send + Sync> CosineScheduler<T> {
    pub fn new(initial_lr: T, t_max: usize) -> Self {
        let t_max = t_max.max(1);
        Self {
            restarts_enabled: false,
            initial_lr,
            min_lr: T::zero(),
            t_max,
            current_lr: initial_lr,
            current_step: 0,
            restart_factor: T::one(),
            restart_count: 0,
            current_t_max: t_max,
        }
    }

    pub fn with_min_lr(mut self, min_lr: T) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Enable warm restarts. The multiplier must be finite and at least one.
    ///
    /// # Panics
    /// Panics if the multiplier would produce a nonpositive or shrinking cycle.
    pub fn with_restarts(mut self, restart_factor: T) -> Self {
        assert!(
            restart_factor.is_finite() && restart_factor >= T::one(),
            "Restart factor must be finite and at least one"
        );
        self.restarts_enabled = true;
        self.restart_factor = restart_factor;
        self
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for CosineScheduler<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        self.current_step = step;

        let mut t_cur = step.min(self.t_max);
        let mut t_max = self.t_max;
        self.restart_count = 0;
        if self.restarts_enabled {
            t_cur = step;
            // Derive each cycle from the absolute step, so repeated queries or
            // querying earlier checkpoints cannot advance the scheduler twice.
            while t_cur >= t_max {
                t_cur -= t_max;
                self.restart_count += 1;
                let next = (T::from(t_max).unwrap() * self.restart_factor)
                    .to_usize()
                    .unwrap_or(usize::MAX)
                    .max(1);
                if next == t_max {
                    self.restart_count += t_cur / t_max;
                    t_cur %= t_max;
                    break;
                }
                t_max = next;
            }
        }
        self.current_t_max = t_max;

        // Cosine annealing formula
        let pi = T::from(std::f64::consts::PI).unwrap();
        let cos_factor = (T::one()
            + (pi * T::from(t_cur).unwrap() / T::from(t_max).unwrap()).cos())
            / T::from(2.0).unwrap();
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_factor;

        self.current_lr
    }

    fn step(&mut self, step: usize, _metrics: Option<T>) -> TrainingResult<T> {
        Ok(self.get_learning_rate(step))
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_step = 0;
        self.restart_count = 0;
        self.current_t_max = self.t_max;
    }

    fn name(&self) -> &'static str {
        "CosineScheduler"
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        SchedulerState {
            current_step: self.current_step,
            current_lr: self.current_lr,
            best_metric: None,
            patience_counter: 0,
            cycle_count: self.restart_count,
            warmup_steps: 0,
            additional_state: std::collections::HashMap::from([
                (
                    "restarts_enabled".to_string(),
                    vec![if self.restarts_enabled {
                        T::one()
                    } else {
                        T::zero()
                    }],
                ),
                ("initial_lr".to_string(), vec![self.initial_lr]),
                ("min_lr".to_string(), vec![self.min_lr]),
                ("t_max".to_string(), vec![T::from(self.t_max).unwrap()]),
                ("restart_factor".to_string(), vec![self.restart_factor]),
                (
                    "current_t_max".to_string(),
                    vec![T::from(self.current_t_max).unwrap()],
                ),
            ]),
        }
    }

    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_step = state.current_step;
        self.current_lr = state.current_lr;
        self.restart_count = state.cycle_count;

        if let Some(initial_lr) = state.additional_state.get("initial_lr") {
            if !initial_lr.is_empty() {
                self.initial_lr = initial_lr[0];
            }
        }
        if let Some(min_lr) = state.additional_state.get("min_lr") {
            if !min_lr.is_empty() {
                self.min_lr = min_lr[0];
            }
        }
        if let Some(t_max) = state.additional_state.get("t_max") {
            if !t_max.is_empty() {
                self.t_max = t_max[0].to_usize().unwrap_or(100);
            }
        }
        if let Some(restart_factor) = state.additional_state.get("restart_factor") {
            if !restart_factor.is_empty() {
                self.restart_factor = restart_factor[0];
            }
        }
        if let Some(current_t_max) = state.additional_state.get("current_t_max") {
            if !current_t_max.is_empty() {
                self.current_t_max = current_t_max[0].to_usize().unwrap_or(100);
            }
        }

        self.t_max = self.t_max.max(1);
        self.current_t_max = self.current_t_max.max(1);
        if !self.restart_factor.is_finite() || self.restart_factor < T::one() {
            return Err(TrainingError::SchedulerError(
                "Invalid restart factor in checkpoint".to_string(),
            ));
        }
        self.restarts_enabled = state
            .additional_state
            .get("restarts_enabled")
            .and_then(|value| value.first())
            .map_or(false, |value| *value == T::one());
        Ok(())
    }
}

// =============================================================================
// Plateau Scheduler
// =============================================================================

/// Reduce learning rate on plateau
#[derive(Clone)]
pub struct PlateauScheduler<T: Float + Send + Sync> {
    initial_lr: T,
    factor: T,
    patience: usize,
    threshold: T,
    min_lr: T,
    mode: PlateauMode,
    current_lr: T,
    best_metric: Option<T>,
    patience_counter: usize,
    cooldown_counter: usize,
    cooldown: usize,
}

#[derive(Clone, Debug)]
pub enum PlateauMode {
    Min,
    Max,
}

impl<T: Float + Send + Sync> PlateauScheduler<T> {
    pub fn new(initial_lr: T, mode: PlateauMode, factor: T, patience: usize) -> Self {
        Self {
            initial_lr,
            factor,
            patience,
            threshold: T::from(1e-4).unwrap(),
            min_lr: T::zero(),
            mode,
            current_lr: initial_lr,
            best_metric: None,
            patience_counter: 0,
            cooldown_counter: 0,
            cooldown: 0,
        }
    }

    pub fn with_threshold(mut self, threshold: T) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_min_lr(mut self, min_lr: T) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn with_cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    fn is_better(&self, current: T, best: T) -> bool {
        match self.mode {
            PlateauMode::Min => current < best - self.threshold,
            PlateauMode::Max => current > best + self.threshold,
        }
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for PlateauScheduler<T> {
    fn get_learning_rate(&mut self, _step: usize) -> T {
        self.current_lr
    }

    fn step(&mut self, _step: usize, metrics: Option<T>) -> TrainingResult<T> {
        if let Some(current_metric) = metrics {
            if self.cooldown_counter > 0 {
                self.cooldown_counter -= 1;
                return Ok(self.current_lr);
            }

            if let Some(best_metric) = self.best_metric {
                if self.is_better(current_metric, best_metric) {
                    self.best_metric = Some(current_metric);
                    self.patience_counter = 0;
                } else {
                    self.patience_counter += 1;

                    if self.patience_counter >= self.patience {
                        self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                        self.patience_counter = 0;
                        self.cooldown_counter = self.cooldown;
                    }
                }
            } else {
                self.best_metric = Some(current_metric);
                self.patience_counter = 0;
            }
        }

        Ok(self.current_lr)
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.best_metric = None;
        self.patience_counter = 0;
        self.cooldown_counter = 0;
    }

    fn name(&self) -> &'static str {
        "PlateauScheduler"
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        SchedulerState {
            current_step: 0,
            current_lr: self.current_lr,
            best_metric: self.best_metric,
            patience_counter: self.patience_counter,
            cycle_count: self.cooldown_counter,
            warmup_steps: 0,
            additional_state: std::collections::HashMap::from([
                ("initial_lr".to_string(), vec![self.initial_lr]),
                ("factor".to_string(), vec![self.factor]),
                (
                    "patience".to_string(),
                    vec![T::from(self.patience).unwrap()],
                ),
                ("threshold".to_string(), vec![self.threshold]),
                ("min_lr".to_string(), vec![self.min_lr]),
                (
                    "cooldown".to_string(),
                    vec![T::from(self.cooldown).unwrap()],
                ),
            ]),
        }
    }

    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_lr = state.current_lr;
        self.best_metric = state.best_metric;
        self.patience_counter = state.patience_counter;
        self.cooldown_counter = state.cycle_count;

        if let Some(initial_lr) = state.additional_state.get("initial_lr") {
            if !initial_lr.is_empty() {
                self.initial_lr = initial_lr[0];
            }
        }
        if let Some(factor) = state.additional_state.get("factor") {
            if !factor.is_empty() {
                self.factor = factor[0];
            }
        }
        if let Some(patience) = state.additional_state.get("patience") {
            if !patience.is_empty() {
                self.patience = patience[0].to_usize().unwrap_or(10);
            }
        }
        if let Some(threshold) = state.additional_state.get("threshold") {
            if !threshold.is_empty() {
                self.threshold = threshold[0];
            }
        }
        if let Some(min_lr) = state.additional_state.get("min_lr") {
            if !min_lr.is_empty() {
                self.min_lr = min_lr[0];
            }
        }
        if let Some(cooldown) = state.additional_state.get("cooldown") {
            if !cooldown.is_empty() {
                self.cooldown = cooldown[0].to_usize().unwrap_or(0);
            }
        }

        Ok(())
    }
}

// =============================================================================
// Warmup Scheduler
// =============================================================================

/// Linear warmup followed by decay
#[derive(Clone)]
pub struct WarmupScheduler<T: Float + Send + Sync> {
    warmup_steps: usize,
    initial_lr: T,
    target_lr: T,
    decay_scheduler: Box<SchedulerType<T>>,
    current_lr: T,
    current_step: usize,
}

impl<T: Float + Send + Sync> WarmupScheduler<T> {
    pub fn new(warmup_steps: usize, target_lr: T, decay_scheduler: SchedulerType<T>) -> Self {
        Self {
            warmup_steps,
            initial_lr: T::zero(),
            target_lr,
            decay_scheduler: Box::new(decay_scheduler),
            current_lr: T::zero(),
            current_step: 0,
        }
    }

    pub fn with_initial_lr(mut self, initial_lr: T) -> Self {
        self.initial_lr = initial_lr;
        self.current_lr = initial_lr;
        self
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for WarmupScheduler<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        self.current_step = step;

        if step < self.warmup_steps {
            // Linear warmup
            let warmup_factor = T::from(step).unwrap() / T::from(self.warmup_steps).unwrap();
            self.current_lr = self.initial_lr + (self.target_lr - self.initial_lr) * warmup_factor;
        } else {
            // Use decay scheduler
            let decay_step = step - self.warmup_steps;
            self.current_lr = self.decay_scheduler.get_learning_rate(decay_step);
        }

        self.current_lr
    }

    fn step(&mut self, step: usize, metrics: Option<T>) -> TrainingResult<T> {
        if step >= self.warmup_steps {
            let decay_step = step - self.warmup_steps;
            self.decay_scheduler.step(decay_step, metrics)?;
        }
        Ok(self.get_learning_rate(step))
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_step = 0;
        self.decay_scheduler.reset();
    }

    fn name(&self) -> &'static str {
        "WarmupScheduler"
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        let mut state = self.decay_scheduler.state();
        state.current_step = self.current_step;
        state.current_lr = self.current_lr;
        state.warmup_steps = self.warmup_steps;
        state
            .additional_state
            .insert("initial_lr".to_string(), vec![self.initial_lr]);
        state
            .additional_state
            .insert("target_lr".to_string(), vec![self.target_lr]);
        state
    }

    fn restore_state(&mut self, mut state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_step = state.current_step;
        self.current_lr = state.current_lr;
        self.warmup_steps = state.warmup_steps;

        if let Some(initial_lr) = state.additional_state.remove("initial_lr") {
            if !initial_lr.is_empty() {
                self.initial_lr = initial_lr[0];
            }
        }
        if let Some(target_lr) = state.additional_state.remove("target_lr") {
            if !target_lr.is_empty() {
                self.target_lr = target_lr[0];
            }
        }

        self.decay_scheduler.restore_state(state)
    }
}

// =============================================================================
// Cyclic Scheduler
// =============================================================================

/// Cyclic learning rate scheduler
#[derive(Clone)]
pub struct CyclicScheduler<T: Float + Send + Sync> {
    base_lr: T,
    max_lr: T,
    step_size_up: usize,
    step_size_down: usize,
    mode: CyclicMode,
    gamma: T,
    current_lr: T,
    current_step: usize,
    cycle_count: usize,
}

#[derive(Clone, Debug)]
pub enum CyclicMode {
    Triangular,
    Triangular2,
    ExpRange,
}

impl<T: Float + Send + Sync> CyclicScheduler<T> {
    pub fn new(base_lr: T, max_lr: T, step_size_up: usize) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down: step_size_up,
            mode: CyclicMode::Triangular,
            gamma: T::from(0.99994).unwrap(),
            current_lr: base_lr,
            current_step: 0,
            cycle_count: 0,
        }
    }

    pub fn with_step_size_down(mut self, step_size_down: usize) -> Self {
        self.step_size_down = step_size_down;
        self
    }

    pub fn with_mode(mut self, mode: CyclicMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_gamma(mut self, gamma: T) -> Self {
        self.gamma = gamma;
        self
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for CyclicScheduler<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        self.current_step = step;

        let cycle_length = self.step_size_up + self.step_size_down;
        let cycle = step / cycle_length;
        let x = step % cycle_length;

        if cycle != self.cycle_count {
            self.cycle_count = cycle;
        }

        let scale_fn = match self.mode {
            CyclicMode::Triangular => T::one(),
            CyclicMode::Triangular2 => T::one() / T::from(2_u32.pow(cycle as u32)).unwrap(),
            CyclicMode::ExpRange => self.gamma.powi(step as i32),
        };

        if x <= self.step_size_up {
            // Ascending phase
            let factor = T::from(x).unwrap() / T::from(self.step_size_up).unwrap();
            self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * factor * scale_fn;
        } else {
            // Descending phase
            let factor = T::one()
                - T::from(x - self.step_size_up).unwrap() / T::from(self.step_size_down).unwrap();
            self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * factor * scale_fn;
        }

        self.current_lr
    }

    fn step(&mut self, step: usize, _metrics: Option<T>) -> TrainingResult<T> {
        Ok(self.get_learning_rate(step))
    }

    fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.current_step = 0;
        self.cycle_count = 0;
    }

    fn name(&self) -> &'static str {
        "CyclicScheduler"
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        SchedulerState {
            current_step: self.current_step,
            current_lr: self.current_lr,
            best_metric: None,
            patience_counter: 0,
            cycle_count: self.cycle_count,
            warmup_steps: 0,
            additional_state: std::collections::HashMap::from([
                ("base_lr".to_string(), vec![self.base_lr]),
                ("max_lr".to_string(), vec![self.max_lr]),
                (
                    "step_size_up".to_string(),
                    vec![T::from(self.step_size_up).unwrap()],
                ),
                (
                    "step_size_down".to_string(),
                    vec![T::from(self.step_size_down).unwrap()],
                ),
                ("gamma".to_string(), vec![self.gamma]),
            ]),
        }
    }

    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_step = state.current_step;
        self.current_lr = state.current_lr;
        self.cycle_count = state.cycle_count;

        if let Some(base_lr) = state.additional_state.get("base_lr") {
            if !base_lr.is_empty() {
                self.base_lr = base_lr[0];
            }
        }
        if let Some(max_lr) = state.additional_state.get("max_lr") {
            if !max_lr.is_empty() {
                self.max_lr = max_lr[0];
            }
        }
        if let Some(step_size_up) = state.additional_state.get("step_size_up") {
            if !step_size_up.is_empty() {
                self.step_size_up = step_size_up[0].to_usize().unwrap_or(100);
            }
        }
        if let Some(step_size_down) = state.additional_state.get("step_size_down") {
            if !step_size_down.is_empty() {
                self.step_size_down = step_size_down[0].to_usize().unwrap_or(100);
            }
        }
        if let Some(gamma) = state.additional_state.get("gamma") {
            if !gamma.is_empty() {
                self.gamma = gamma[0];
            }
        }

        Ok(())
    }
}

// =============================================================================
// One Cycle Scheduler
// =============================================================================

/// One cycle learning rate policy
#[derive(Clone)]
pub struct OneCycleScheduler<T: Float + Send + Sync> {
    max_lr: T,
    total_steps: usize,
    pct_start: T,
    anneal_strategy: AnnealStrategy,
    current_lr: T,
    current_step: usize,
    div_factor: T,
    final_div_factor: T,
    finished: bool,
}

#[derive(Clone, Debug)]
pub enum AnnealStrategy {
    Cos,
    Linear,
}

impl<T: Float + Send + Sync> OneCycleScheduler<T> {
    pub fn new(max_lr: T, total_steps: usize) -> Self {
        Self {
            max_lr,
            total_steps,
            pct_start: T::from(0.3).unwrap(),
            anneal_strategy: AnnealStrategy::Cos,
            current_lr: max_lr / T::from(25.0).unwrap(),
            current_step: 0,
            div_factor: T::from(25.0).unwrap(),
            final_div_factor: T::from(10000.0).unwrap(),
            finished: false,
        }
    }

    pub fn with_pct_start(mut self, pct_start: T) -> Self {
        self.pct_start = pct_start;
        self
    }

    pub fn with_anneal_strategy(mut self, anneal_strategy: AnnealStrategy) -> Self {
        self.anneal_strategy = anneal_strategy;
        self
    }

    pub fn with_div_factor(mut self, div_factor: T) -> Self {
        self.div_factor = div_factor;
        self.current_lr = self.max_lr / div_factor;
        self
    }

    pub fn with_final_div_factor(mut self, final_div_factor: T) -> Self {
        self.final_div_factor = final_div_factor;
        self
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for OneCycleScheduler<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        if step >= self.total_steps {
            self.finished = true;
            return self.max_lr / self.final_div_factor;
        }

        self.current_step = step;
        let step_ratio = T::from(step).unwrap() / T::from(self.total_steps).unwrap();

        if step_ratio <= self.pct_start {
            // Ascending phase
            let phase_ratio = step_ratio / self.pct_start;
            self.current_lr = self.max_lr / self.div_factor
                + (self.max_lr - self.max_lr / self.div_factor) * phase_ratio;
        } else {
            // Descending phase
            let phase_ratio = (step_ratio - self.pct_start) / (T::one() - self.pct_start);

            match self.anneal_strategy {
                AnnealStrategy::Cos => {
                    let pi = T::from(std::f64::consts::PI).unwrap();
                    let cos_factor = (T::one() + (pi * phase_ratio).cos()) / T::from(2.0).unwrap();
                    self.current_lr = self.max_lr / self.final_div_factor
                        + (self.max_lr - self.max_lr / self.final_div_factor) * cos_factor;
                }
                AnnealStrategy::Linear => {
                    self.current_lr = self.max_lr * (T::one() - phase_ratio)
                        + (self.max_lr / self.final_div_factor) * phase_ratio;
                }
            }
        }

        self.current_lr
    }

    fn step(&mut self, step: usize, _metrics: Option<T>) -> TrainingResult<T> {
        Ok(self.get_learning_rate(step))
    }

    fn reset(&mut self) {
        self.current_lr = self.max_lr / self.div_factor;
        self.current_step = 0;
        self.finished = false;
    }

    fn name(&self) -> &'static str {
        "OneCycleScheduler"
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        SchedulerState {
            current_step: self.current_step,
            current_lr: self.current_lr,
            best_metric: None,
            patience_counter: 0,
            cycle_count: 0,
            warmup_steps: 0,
            additional_state: std::collections::HashMap::from([
                ("max_lr".to_string(), vec![self.max_lr]),
                (
                    "total_steps".to_string(),
                    vec![T::from(self.total_steps).unwrap()],
                ),
                ("pct_start".to_string(), vec![self.pct_start]),
                ("div_factor".to_string(), vec![self.div_factor]),
                ("final_div_factor".to_string(), vec![self.final_div_factor]),
                (
                    "finished".to_string(),
                    vec![if self.finished { T::one() } else { T::zero() }],
                ),
            ]),
        }
    }

    fn restore_state(&mut self, state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_step = state.current_step;
        self.current_lr = state.current_lr;

        if let Some(max_lr) = state.additional_state.get("max_lr") {
            if !max_lr.is_empty() {
                self.max_lr = max_lr[0];
            }
        }
        if let Some(total_steps) = state.additional_state.get("total_steps") {
            if !total_steps.is_empty() {
                self.total_steps = total_steps[0].to_usize().unwrap_or(1000);
            }
        }
        if let Some(pct_start) = state.additional_state.get("pct_start") {
            if !pct_start.is_empty() {
                self.pct_start = pct_start[0];
            }
        }
        if let Some(div_factor) = state.additional_state.get("div_factor") {
            if !div_factor.is_empty() {
                self.div_factor = div_factor[0];
            }
        }
        if let Some(final_div_factor) = state.additional_state.get("final_div_factor") {
            if !final_div_factor.is_empty() {
                self.final_div_factor = final_div_factor[0];
            }
        }
        if let Some(finished) = state.additional_state.get("finished") {
            if !finished.is_empty() {
                self.finished = !finished[0].is_zero();
            }
        }

        Ok(())
    }
}

// =============================================================================
// Seasonal Scheduler
// =============================================================================

/// Seasonal-aware learning rate scheduler for time series
#[derive(Clone)]
pub struct SeasonalScheduler<T: Float + Send + Sync> {
    base_scheduler: Box<SchedulerType<T>>,
    seasonal_factors: Vec<T>,
    current_lr: T,
    current_step: usize,
    season_length: usize,
}

impl<T: Float + Send + Sync> SeasonalScheduler<T> {
    pub fn new(
        base_scheduler: SchedulerType<T>,
        seasonal_factors: Vec<T>,
        season_length: usize,
    ) -> Self {
        let current_lr = base_scheduler.current_lr();
        Self {
            base_scheduler: Box::new(base_scheduler),
            seasonal_factors,
            current_lr,
            current_step: 0,
            season_length,
        }
    }
}

impl<T: Float + Send + Sync> LearningRateScheduler<T> for SeasonalScheduler<T> {
    fn get_learning_rate(&mut self, step: usize) -> T {
        self.current_step = step;

        // Get base learning rate
        let base_lr = self.base_scheduler.get_learning_rate(step);

        // Apply seasonal adjustment
        if !self.seasonal_factors.is_empty() {
            let season_idx =
                (step % self.season_length) * self.seasonal_factors.len() / self.season_length;
            let seasonal_factor =
                self.seasonal_factors[season_idx.min(self.seasonal_factors.len() - 1)];
            self.current_lr = base_lr * seasonal_factor;
        } else {
            self.current_lr = base_lr;
        }

        self.current_lr
    }

    fn step(&mut self, step: usize, metrics: Option<T>) -> TrainingResult<T> {
        self.base_scheduler.step(step, metrics)?;
        Ok(self.get_learning_rate(step))
    }

    fn reset(&mut self) {
        self.base_scheduler.reset();
        self.current_lr = self.base_scheduler.current_lr();
        self.current_step = 0;
    }

    fn name(&self) -> &'static str {
        "SeasonalScheduler"
    }

    fn current_lr(&self) -> T {
        self.current_lr
    }

    fn state(&self) -> SchedulerState<T> {
        let mut state = self.base_scheduler.state();
        state.current_step = self.current_step;
        state.current_lr = self.current_lr;
        state.additional_state.insert(
            "season_length".to_string(),
            vec![T::from(self.season_length).unwrap()],
        );
        state.additional_state.insert(
            "seasonal_factors".to_string(),
            self.seasonal_factors.clone(),
        );
        state
    }

    fn restore_state(&mut self, mut state: SchedulerState<T>) -> TrainingResult<()> {
        self.current_step = state.current_step;
        self.current_lr = state.current_lr;

        if let Some(season_length) = state.additional_state.remove("season_length") {
            if !season_length.is_empty() {
                self.season_length = season_length[0].to_usize().unwrap_or(1);
            }
        }
        if let Some(seasonal_factors) = state.additional_state.remove("seasonal_factors") {
            self.seasonal_factors = seasonal_factors;
        }

        self.base_scheduler.restore_state(state)
    }
}

// =============================================================================
// Builder Pattern for Schedulers
// =============================================================================

/// Builder for creating schedulers with fluent interface
pub struct SchedulerBuilder<T: Float + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync> SchedulerBuilder<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    pub fn exponential(initial_lr: T, decay_rate: T) -> ExponentialScheduler<T> {
        ExponentialScheduler::new(initial_lr, decay_rate)
    }

    pub fn step(initial_lr: T, step_size: usize, gamma: T) -> StepScheduler<T> {
        StepScheduler::new(initial_lr, step_size, gamma)
    }

    pub fn cosine(initial_lr: T, t_max: usize) -> CosineScheduler<T> {
        CosineScheduler::new(initial_lr, t_max)
    }

    pub fn plateau(
        initial_lr: T,
        mode: PlateauMode,
        factor: T,
        patience: usize,
    ) -> PlateauScheduler<T> {
        PlateauScheduler::new(initial_lr, mode, factor, patience)
    }

    pub fn cyclic(base_lr: T, max_lr: T, step_size_up: usize) -> CyclicScheduler<T> {
        CyclicScheduler::new(base_lr, max_lr, step_size_up)
    }

    pub fn one_cycle(max_lr: T, total_steps: usize) -> OneCycleScheduler<T> {
        OneCycleScheduler::new(max_lr, total_steps)
    }

    pub fn seasonal(
        base_scheduler: SchedulerType<T>,
        seasonal_factors: Vec<T>,
        season_length: usize,
    ) -> SeasonalScheduler<T> {
        SeasonalScheduler::new(base_scheduler, seasonal_factors, season_length)
    }
}

impl<T: Float + Send + Sync> Default for SchedulerBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exponential_scheduler() {
        let mut scheduler = ExponentialScheduler::new(0.1, 0.9);

        assert_relative_eq!(scheduler.get_learning_rate(0), 0.1, epsilon = 1e-6);
        assert_relative_eq!(scheduler.get_learning_rate(1), 0.09, epsilon = 1e-6);
        assert_relative_eq!(scheduler.get_learning_rate(2), 0.081, epsilon = 1e-6);
    }

    #[test]
    fn test_step_scheduler() {
        let mut scheduler = StepScheduler::new(0.1, 2, 0.5);

        assert_relative_eq!(scheduler.get_learning_rate(0), 0.1, epsilon = 1e-6);
        assert_relative_eq!(scheduler.get_learning_rate(1), 0.1, epsilon = 1e-6);
        assert_relative_eq!(scheduler.get_learning_rate(2), 0.05, epsilon = 1e-6);
        assert_relative_eq!(scheduler.get_learning_rate(3), 0.05, epsilon = 1e-6);
        assert_relative_eq!(scheduler.get_learning_rate(4), 0.025, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_scheduler() {
        let mut scheduler = CosineScheduler::new(0.1, 10);

        let lr_start = scheduler.get_learning_rate(0);
        let lr_mid = scheduler.get_learning_rate(5);
        let lr_end = scheduler.get_learning_rate(10);

        assert_relative_eq!(lr_start, 0.1, epsilon = 1e-6);
        assert!(lr_mid < lr_start);
        assert!(lr_end < lr_mid);
    }

    #[test]
    fn test_plateau_scheduler() {
        let mut scheduler = PlateauScheduler::new(0.1, PlateauMode::Min, 0.5, 2);

        // Initially should maintain learning rate
        scheduler.step(0, Some(1.0)).unwrap();
        assert_relative_eq!(scheduler.current_lr(), 0.1, epsilon = 1e-6);

        // After improvement, should maintain
        scheduler.step(1, Some(0.9)).unwrap();
        assert_relative_eq!(scheduler.current_lr(), 0.1, epsilon = 1e-6);

        // After plateau, should reduce
        scheduler.step(2, Some(0.91)).unwrap();
        scheduler.step(3, Some(0.92)).unwrap();
        assert_relative_eq!(scheduler.current_lr(), 0.05, epsilon = 1e-6);
    }

    #[test]
    fn test_one_cycle_scheduler() {
        let mut scheduler = OneCycleScheduler::new(0.1, 100);

        let lr_start = scheduler.get_learning_rate(0);
        let lr_peak = scheduler.get_learning_rate(30); // 30% of 100
        let lr_end = scheduler.get_learning_rate(99);

        assert!(lr_start < lr_peak);
        assert!(lr_end < lr_peak);
        assert!(lr_end < lr_start);

        // Should be finished after total steps
        scheduler.get_learning_rate(100);
        assert!(scheduler.is_finished());
    }

    #[test]
    fn test_seasonal_scheduler() {
        let base_scheduler = SchedulerType::Exponential(ExponentialScheduler::new(0.1, 0.9));
        let seasonal_factors = vec![1.0, 1.2, 0.8, 1.1];
        let mut scheduler = SeasonalScheduler::new(base_scheduler, seasonal_factors, 4);

        // Test that seasonal factors are applied
        let lr_0 = scheduler.get_learning_rate(0); // Factor 1.0
        let lr_1 = scheduler.get_learning_rate(1); // Factor 1.2
        let lr_2 = scheduler.get_learning_rate(2); // Factor 0.8

        // Seasonal factors should modify the base rate
        assert!(lr_1 > lr_0); // 1.2 > 1.0
        assert!(lr_2 < lr_0); // 0.8 < 1.0
    }

    #[test]
    fn test_scheduler_state_save_restore() {
        let mut scheduler = ExponentialScheduler::new(0.1, 0.9);

        // Take some steps
        scheduler.get_learning_rate(5);

        // Save state
        let state = scheduler.state();
        let checkpoint_lr = scheduler.current_lr();

        // Continue and modify
        scheduler.get_learning_rate(10);

        // Restore state
        scheduler.restore_state(state).unwrap();

        // Should be back to checkpoint
        assert_relative_eq!(scheduler.current_lr(), checkpoint_lr, epsilon = 1e-6);
    }
}

#[cfg(test)]
#[path = "scheduler_regression_tests.rs"]
mod regression_tests;
