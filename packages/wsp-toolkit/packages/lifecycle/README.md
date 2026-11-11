# ruv-fann-wsp-lifecycle

Multi-suite regression gates and in-memory candidate branch evaluation around `ruv-fann-wsp@0.2.0`. ESM; Node 20.9+; browser-compatible library, Node-only CLI.

`runRegressionGate({model, patch, suites:[{name,dataset,policy}]}, options?)` returns hash-bound, real WASM evidence and accepts only if every suite accepts. A maximum of 16 suites, 2 MiB input and 10 million aggregate baseline/candidate parameter-row operations apply. `evaluatePatchBranches(model, patches, dataset, policy, options?)` evaluates at most 16 branches against the same snapshotted base, with the same aggregate budget. Sequential allocation bounds memory. Branches are data snapshots, not security sandboxes or separate runtime instances.

`wsp-gate bundle.json` writes compact JSON evidence; exit 0 accepts, 1 rejects, 2 reports malformed input/runtime failure. This is suitable for a pull-request checkpoint gate. Evidence is not signed and cannot authorize promotion. Use the WSP workspace's internal proposal ID and expected revision for explicit promotion and monotonic rollback. Choose independent validation cases appropriate to the application; passing a small suite does not prove generalization or safety.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
