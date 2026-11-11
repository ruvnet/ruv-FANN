# ruv-fann-wsp-search — bounded neural weight search

Search a small, explicit set of neural parameters using real WASM predictions. Reuse the baseline measurement, screen bounded candidates and independently validate the winner before asking your application to activate it.

**Topics:** weight-space programming · coordinate search · Boolean truth tables · Hebbian proposals.

```sh
npm install ruv-fann-wsp-search@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruv-fann-wsp-search) · [Core 0.2.0 entrypoint: `ruv-fann/search`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/search` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Bounded discrete coordinate search with real WASM inference, plus explicit experimental truth-table compilation and Hebbian patch proposals. ESM, Node 20.9+ and secure-context browsers. Depends on `ruv-fann-wsp@0.2.0` and `ruv-fann@0.1.2`.

Integration sketch: supply `model`, `developmentCases` and your trusted acceptance policy. Keep independent holdout cases away from search.

```js
import { searchWeightPatch } from 'ruv-fann-wsp-search';

const result = await searchWeightPatch(model, developmentCases, {
  maxMSE: 0, maxRegressionMSE: 0
}, {
  locations: [
    { layer: 1, parameter: 'weights', index: 0, values: [0, 1] },
    { layer: 1, parameter: 'weights', index: 1, values: [0, 1] }
  ],
  maxEvaluations: 16
});
// A result is a proposal. Explicitly submit result.patch to a WeightWorkspace
// with your acceptance dataset/policy and promote its accepted proposal separately.
console.log(result.evaluation?.accepted, result.evaluations);
```

Search minimizes MSE by default. A trusted synchronous `objective({predictions, metrics, model})` callback may instead return a finite discontinuous score, for example the count of misclassifications. Callbacks receive recursively frozen snapshots; they execute as ordinary application code and their runtime is not bounded by the neural work budget. Custom objectives never override the final MSE/regression policy. Candidate screens share one baseline measurement. The winner receives an independent authoritative `evaluateWeightPatch` against the original baseline. There is no automatic promotion, backpropagation, provider call or global optimum guarantee. Ties preserve the earlier model; no strict improvement returns `patch: null, evaluation: null`.

Limits: 16 unique locations, 32 unique absolute values per location, 256 candidate evaluations (default 64), 16 passes (default 4), and at most 10 million aggregate parameter-row operations reserved before execution. `maxWork` may tighten that ceiling; the reservation includes the baseline and both authoritative final inference passes. WSP patch bounds and dataset limits still apply. Inputs, choices, bounds and policies are snapshotted before the first await. Browser use needs the core runtime's WASM asset layout.

## Experimental direct compilation

```js
import { compileBooleanTable } from 'ruv-fann-wsp-search';
const xor = compileBooleanTable({ inputs: 2, outputs: 1, targets: [0, 1, 1, 0] });
```

Rows enumerate binary inputs from `00..0` to `11..1`, with the most significant bit first. The compiler builds one Threshold pattern detector per row and a Threshold output layer. It accepts complete numeric 0/1 tables for 1–6 inputs and 1–16 outputs. It reproduces the complete binary truth table by construction; behavior on nonbinary inputs is unspecified. This is direct model construction, not a claim that business rules can be compiled into arbitrary pretrained networks.

## Experimental localized plasticity

```js
import { createHebbianPatch } from 'ruv-fann-wsp-search';
const patch = await createHebbianPatch(model, { inputs: [1, 2], rows: 2 }, {
  layer: 0, indices: [0], learningRate: 0.1,
  limits: { maxAbsDelta: 0.25 }
});
```

For each explicit weight, propose `rate * mean(pre * post)` from actual WASM activations. Clip to patch bounds, preserve topology, and independently evaluate before promotion. This synthetic correlation rule is an experiment; it does not establish useful biological learning, drift correction, safety, or generalization. Zero/underflow updates are rejected. No arbitrary code or remote objective is loaded.

Run `npm test` in the package or `npm test` from the toolkit workspace after installation. Tests use exhaustive Boolean oracles, independent analytic MSE/correlation calculations, custom classification scores, and malformed/budget/ownership checks.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
