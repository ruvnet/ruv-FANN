# Forecasting evidence and migration

Read these checkout paths as needed:

- `neuro-divergent/tests/README.md`: migrated test categories and supported behavior.
- `neuro-divergent/src/neural_forecast.rs`: fitted context, prediction and checkpoint contract.
- `neuro-divergent/neuro-divergent-models/src/recurrent/rnn.rs`: actual recurrent model.
- `neuro-divergent/neuro-divergent-models/src/forecasting.rs`: forecast adapter and supported model integration.
- `neuro-divergent/neuro-divergent-data/src/preprocessing.rs`: scaler/difference behavior.
- `docs/review-forecasting.md`: numerical findings and chronological/runtime checks.
- `docs/research-modernization.md`: primary forecasting research and adoption criteria.

## Evaluation record

Record dataset source/revision/license, exact series IDs, frequency, horizon, cutoffs, lookback, refit behavior, seed, missing-value policy, fitted transforms, metric definition and aggregation. Preserve the predictions/targets needed to reproduce the score without retaining private raw data unnecessarily. Include train, preprocessing and prediction costs separately and together when useful.

For scaled metrics, define behavior when the scale denominator is zero. For an ensemble, ensure complete key intersections before averaging; missing model rows are failures rather than zeros. Hold back evaluation data until the algorithm and preprocessing choices are frozen.

## Regression cases

1. A naive forecast repeats the last past observation for every future step.
2. A seasonal baseline uses the declared period and rejects insufficient context.
3. A learned RNN changes predictions through trained recurrent and projection parameters, with hidden size different from horizon.
4. A cross-validation window never sees its future targets in model training or scaler fitting, even when reusing a previously fitted object.
5. Unequal series have independent actual cutoffs and correctly joined forecast keys.
6. Checkpoint reload preserves predictions, schema, fitted cutoffs, date precision, frequency and timezone metadata.
7. Invalid, oversize, truncated or special-file checkpoints fail before destructive writes or blocking reads.

## Research boundary

Chronos-2, TimesFM, PatchTST and other papers are external evaluation candidates. A current paper citation does not implement its architecture, weights, training recipe or benchmark. Before adopting one, verify the model/checkpoint license, runtime, supported input mapping, cost limits and held-out evaluation. Public Polars types and underlying FANN versions are compatibility boundaries; read the candidate manifests before adapting older 0.1 callers.
