# neuro-divergent-training

Loss functions, forecasting metrics, optimizers, learning-rate schedules, and gradient helpers. Use these pieces in an explicit training loop. This crate does not currently provide the previously advertised `TrainerBuilder`, automatic mixed-precision loop, or end-to-end cross-validation runner.

![A training cutoff separates optimization from future evaluation](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/neuro-divergent/docs/assets/forecasting-flow.svg)

## Apply an optimizer step

```rust
use neuro_divergent_training::{Adam, Optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = vec![vec![1.0_f64, -1.0]];
    let gradients = vec![vec![0.2, -0.3]];
    let mut optimizer = Adam::new(0.001, 0.9, 0.999).with_amsgrad(true);
    optimizer.step(&mut parameters, &gradients)?;
    let state = optimizer.state();
    let mut resumed = Adam::new(0.001, 0.9, 0.999).with_amsgrad(true);
    resumed.restore_state(state)?;
    Ok(())
}
```

Every public optimizer checks exact nested parameter/gradient shapes and finite values before an update, and stages changes so a rejected step preserves parameters and moments. Adam also validates its learning rate, beta, and epsilon settings. Its bias correction follows the reference equation, including epsilon after the corrected variance square root. AdamW validates before applying weight decay; SGD and RMSprop validate momentum controls and permit a new parameter shape after `reset`.

Restoring an AMSGrad state into a fresh optimizer restores its variance maxima, so the next step matches uninterrupted training. Construct the resumed optimizer with the same beta, epsilon, and AMSGrad settings. Optimizer state alone is not a complete training checkpoint: save model parameters, the scheduler, and data position separately.

The `serde` feature derives serialization for the actual `OptimizerState` and `SchedulerState` types. `checkpointing` enables that capability; save and restore those states with your chosen storage format. JSON round-trip tests check continued AMSGrad updates and scheduler rates. The feature does not run automatic checkpoints or persist the model for you.

Gradient norm and clipping helpers avoid overflow from squaring large finite gradients. Plain cosine annealing reaches the minimum rate and stays there. Call `with_restarts` explicitly for warm restarts; repeated and out-of-order step queries are deterministic.

Public losses reject empty or non-finite inputs, mismatched shapes, invalid scale/quantile settings, and overflowing results. Gaussian CRPS validates positive standard deviations and uses derivatives checked against finite differences. Loss validation does not establish the accuracy of every experimental distribution approximation.

Use `LossAdapter::calculate_loss` and `calculate_gradient` for checked errors. FANN's scalar `ErrorFunction` interface cannot return an error; an incompatible or invalid loss returns NaN through that interface instead of reporting a false zero loss.

For reproducible randomness, `utils::seeded_rng(seed)` returns an RNG to pass explicitly to your code. The legacy `set_seed` name is deprecated and also returns an RNG; it cannot reseed `rand::thread_rng`. Root `QuantileLoss` and `PinballLoss` name the training losses; their evaluation counterparts are `QuantileMetric` and `PinballMetric` (also available under `metrics`).

## Validate

From the repository root:

```sh
cargo test --manifest-path neuro-divergent/Cargo.toml -p neuro-divergent-training --lib --locked
```

Regression tests cover reference Adam updates with small gradients, rejected updates without partial changes, large finite gradient clipping, exact AMSGrad continuation, malformed optimizer state, seeded RNG replay, cosine endpoints, restarts, and scheduler restoration. The models crate uses this Adam implementation in its actual recurrent BPTT loop. Other optimizers and losses retain experimental behavior and need workload-specific validation. `mixed_precision` is a reserved marker; optional SIMD and checkpoint dependencies do not by themselves implement a complete accelerated trainer.

The 0.2.0 candidate uses ruv-FANN 0.2.2 and removes an unused dependency on the forecasting core. See the [family review](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-forecasting.md) for exact results and remaining release concerns.
