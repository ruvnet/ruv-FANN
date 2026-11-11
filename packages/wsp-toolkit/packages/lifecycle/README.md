# ruv-fann-wsp-lifecycle — regression gates and checkpoint branches

Evaluate several named datasets against the same bounded weight patch, or compare isolated candidate snapshots. A passing result is evidence; your application still controls promotion.

**Topics:** neural model regression testing · independent holdouts · checkpoint branches · CI gates.

```sh
npm install ruv-fann-wsp-lifecycle@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruv-fann-wsp-lifecycle) · [Core 0.2.0 entrypoint: `ruv-fann/lifecycle`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/lifecycle` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Multi-suite regression gates and in-memory candidate branch evaluation around `ruv-fann-wsp@0.2.0`. ESM; Node 20.9+; browser-compatible library, Node-only CLI.

`runRegressionGate({model, patch, suites:[{name,dataset,policy}]}, options?)` returns hash-bound, real WASM evidence and accepts only if every suite accepts. A maximum of 16 suites, 2 MiB input and 10 million aggregate baseline/candidate parameter-row operations apply. `evaluatePatchBranches(model, patches, dataset, policy, options?)` evaluates at most 16 branches against the same snapshotted base, with the same aggregate budget. Sequential allocation bounds memory. Branches are data snapshots, not security sandboxes or separate runtime instances.

`wsp-gate bundle.json` writes compact JSON evidence; exit 0 accepts, 1 rejects, 2 reports malformed input/runtime failure. This is suitable for a pull-request checkpoint gate. Evidence is not signed and cannot authorize promotion. Use the WSP workspace's internal proposal ID and expected revision for explicit promotion and monotonic rollback. Choose independent validation cases appropriate to the application; passing a small suite does not prove generalization or safety.

<!-- constellation-links:start -->

Explore the [ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/), [constellation onramp](https://www.npmjs.com/package/ruvnet-constellation) and [shared contracts](https://www.npmjs.com/package/ruvnet-constellation-contracts). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; package.json defines install-time dependencies. The [versioned catalog](https://github.com/ruvnet/ruv-FANN/blob/main/packages/wsp-toolkit/packages/constellation/catalog.json) declares release profiles and package versions; registry availability and publication provenance require independent verification.

<!-- constellation-links:end -->

## Discoverability topics

neural model regression testing · named evaluation suites · independent holdouts · checkpoint branches · candidate comparisons · bounded weight patches · scalar and SIMD inference · application-controlled promotion.
