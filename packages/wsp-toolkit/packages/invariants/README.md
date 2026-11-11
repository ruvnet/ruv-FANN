# ruv-fann-wsp-invariants — finite-domain neural output checks

Check expected outputs and inclusive bounds over an explicit, finite set of inputs. Exhaustive Boolean cases can cover a small binary domain; they do not prove behavior over arbitrary real inputs or deployment conditions.

**Topics:** neural output constraints · exhaustive Boolean domains · bounded WASM evaluation · invariant suites.

```sh
npm install ruv-fann-wsp-invariants@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruv-fann-wsp-invariants) · [Core 0.2.0 entrypoint: `ruv-fann/invariants`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/invariants` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Bounded output constraints evaluated with real `ruv-fann@0.1.2` scalar/SIMD WebAssembly. ESM, Node 20.9+ and secure-context browsers; no gradients, providers or automatic promotion.

```js
import { checkModelInvariants, createBooleanInvariant } from 'ruv-fann-wsp-invariants';

const candidateModel = {schema: 'ruv-fann.dense-layers/1', layers: [
  {inputs: 2, outputs: 2, weights: [1,-1,-1,1], bias: [0,0], activation: 'ReLU', steepness: 1},
  {inputs: 2, outputs: 1, weights: [1,1], bias: [0], activation: 'Linear', steepness: 1}
]};
const suite = createBooleanInvariant({inputs: 2, outputs: 1, targets: [0, 1, 1, 0]});
const result = await checkModelInvariants(candidateModel, suite, {simd: false});
if (!result.accepted) console.log(result.violations);
```

`checkModelInvariants(model, suite, options?)` accepts a dense model object or JSON string and a suite:

```js
{
  schema: 'ruv-fann.invariants/1',
  cases: [{name: 'output-envelope', inputs: [0, 1, 2], rows: 3,
    minimum: [0], maximum: [4], expected: [0, 2, 4], tolerance: 1e-12}]
}
```

Each case needs at least one of `minimum`, `maximum` or `expected`. Inclusive bounds have one value per model output; expected values are flattened row-major. Absolute tolerance applies only to expected values and defaults to zero. Names must be unique bounded ASCII identifiers. Caller-owned objects/arrays are copied before the first await. Unknown properties, accessors, sparse arrays, nonfinite numbers, mismatched dimensions and relaxed options are rejected.

`estimateInvariantWork(model, suite)` synchronously validates the complete model and suite using the same structural, shape, name, tolerance and aggregate budget checks. It returns `parameterCount * sum(case.rows)` without inference, hashing or runtime initialization. A coordinator can reject malformed trusted criteria and reserve work before dispatching proposal workers. Numerical overflow is detected during the later real-WASM check.

The receipt binds the canonical model hash and a suite hash including input/output dimensions. `checks` counts scalar comparisons; `violationCount` counts every failing comparison. At most 128 `{case,row,output,constraint,actual,bound}` entries are retained, with `truncatedViolations` signaling additional failures. Overflow/nonfinite inference fails closed with an exception. This evidence is caller-owned, unsigned and never authorizes promotion.

There are at most 16 cases, 1,024 total rows, 131,072 scalar suite elements, 2 MiB canonical suite JSON and 10 million aggregate `parameterCount * rows` operations. Model limits come from WSP. Each case uses one full-model inference pass in bounded batches of 32, and every prepared resource is disposed on success and failure. No layer-by-layer trace storage is retained.

`createBooleanInvariant({inputs, outputs, targets, tolerance?})` enumerates every input in a complete 1..6-bit domain, in most-significant-bit-first order. It returns one `boolean-domain` case. Outputs must be a complete numeric 0/1 truth table, up to 16 outputs. A zero-tolerance passing check exhausts that finite binary domain. It provides no guarantee for nonbinary inputs, physical deployment conditions, unseen distributions or alignment. Choose protected independent cases and a promotion authority outside the proposal workers; cryptographic integrity does not establish behavioral safety.

<!-- constellation-links:start -->

Explore the [ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/), [constellation onramp](https://www.npmjs.com/package/ruvnet-constellation) and [shared contracts](https://www.npmjs.com/package/ruvnet-constellation-contracts). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; package.json defines install-time dependencies. The [versioned catalog](https://github.com/ruvnet/ruv-FANN/blob/main/packages/wsp-toolkit/packages/constellation/catalog.json) declares release profiles and package versions; registry availability and publication provenance require independent verification.

<!-- constellation-links:end -->
