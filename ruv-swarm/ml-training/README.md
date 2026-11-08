# ruv-swarm-ml-training

Experimental training interfaces and event feature extraction. This crate's built-in LSTM, TCN and N-BEATS types are configuration scaffolds; their training, prediction, save and load methods now return explicit unsupported-backend errors instead of mock predictions or successful no-op saves.

For implemented dense-network forecasting, use `ruv-swarm-ml`. This package does not train the architectures named above and does not provide a GPU or distributed training engine.

## Use the data preparation layer

`StreamDataLoader::new(buffer_size, sequence_length)` groups typed `StreamEvent` values by agent and produces sliding feature windows with next-event labels. Call `load_from_stream(events.into_iter()).await` and inspect the returned `TrainingDataset` before handing it to an application-owned `NeuroDivergentModel` implementation.

The loader requires `0 < sequence_length <= buffer_size <= 100,000`. It rejects input beyond the event-buffer count and a generated feature matrix above 10,000,000 scalar elements. These limits bound those structures, not every caller-owned event string or model allocation. Validate event values and prompt sizes according to the application's data source.

`TrainingPipeline` still selects unavailable built-in backends and therefore returns their errors. Numeric `RandomSearch` is implemented for an application-owned model: each trial trains on its training partition and scores a separate validation partition. Grid, Bayesian, Hyperband and categorical search return explicit unsupported-method errors. Trial counts, numeric ranges, sample shapes and finite values are validated. Do not treat generated configuration as trained weights or an AutoML result.

## Check it

```sh
cargo +1.88.0 test --locked -p ruv-swarm-ml-training --all-targets --all-features
```

Rust 1.88 coverage includes nine library tests, seven integration tests and ten Criterion smoke cases. A held-out-data regression trains a real closed-form mean baseline in the test and verifies that validation values do not leak into training. Examples handle unavailable built-in backends explicitly. The extra math dependencies under `full` do not enable a neural backend.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
