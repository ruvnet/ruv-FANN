# ruv-fann-wsp-invariants

Bounded output constraints evaluated with real `ruv-fann@0.1.2` scalar/SIMD WebAssembly. ESM, Node 20.9+ and secure-context browsers; no gradients, providers or automatic promotion.

```js
import { checkModelInvariants, createBooleanInvariant } from 'ruv-fann-wsp-invariants';

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

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
