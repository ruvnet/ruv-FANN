# ruv-fann-wsp-loop

A bounded, local autonomous weight-evolution coordinator. Fixed-operation worker threads propose patches; the coordinator independently evaluates trusted regression suites and output constraints; AgentDB's internal proposal IDs and expected revisions authorize local checkpoint promotion. The loop runs no generated code, provider requests, shell commands, public publishing or financial transactions.

```js
import { runEvolutionLoop } from 'ruv-fann-wsp-loop';
const report = await runEvolutionLoop({
  registry, // Caller-owned durable registry, e.g. ruv-fann-wsp-registry over AgentDB.
  modules: [{
    id: 'router',
    development: {dataset: developmentCases, policy},
    holdouts: [{name: 'heldout', dataset: heldoutCases, policy}],
    invariants: outputConstraints,
    strategies: [{name: 'coordinate-search', kind: 'search', options: {
      locations: [{layer: 0, parameter: 'weights', index: 0, values: [0, 1, 2]}],
      maxEvaluations: 8, maxPasses: 2
    }}]
  }],
  concurrency: 2, maxRounds: 4, maxProposals: 16, maxWork: 1_000_000,
  jobTimeoutMs: 5000, maxDurationMs: 30000
});
```

The registry must already contain each module. The full module configuration is snapshotted before awaiting. Proposers receive the active model and development cases; held-out data and invariant definitions stay with the coordinator. A worker's score, acceptance flag or model never authorizes a commit. Each candidate is re-applied against the source hash, re-evaluated using real WASM, checked for strict development-loss improvement, and checked against the independent invariant suite. Duplicate candidate hashes share validation. Candidates are ranked within the same task by development MSE and stable hash/strategy tie-breaks; unrelated tasks are not compared by raw loss.

Only then does the coordinator ask AgentDB to create a fresh stored proposal and promote it against the original revision. Stale proposals cannot replace newer state. No retry silently broadens the acceptance policy. The run stops at goals, lack of progress, round/proposal/work limits, cancellation or deadline. Registry or durability failures reject the run instead of becoming successful evolution reports.

`reservedWork` is a conservative parameter-row upper bound: each scheduled proposal reserves its complete planned search, two-pass regression suites, one-pass output constraints, and a possible two-pass AgentDB evaluation. It counts unused reservations and is not a wall-clock measurement. Limits apply to one invocation; applications need a shared external budget authority for several coordinators. CPU time includes hashing, memory allocation, persistence and worker startup outside that arithmetic estimate.

Abort/deadline stops new steps and terminates pending proposal workers. A registry operation already entered is awaited; an atomic commit already started can finish and is reported. Arbitrary injected registry I/O must honor its own operational timeouts. Worker threads run trusted shipped code with separate WASM instances; they are not hardened sandboxes.

Output invariants provide finite-domain evidence. This package claims no universal alignment proof, physical actuation safety, arbitrary-model improvement, semantic routing, self-funding or decentralized consensus. The runnable toolkit demo shows actual AgentDB persistence, worker execution and rollback on small synthetic models.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
