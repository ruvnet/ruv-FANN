# ruv-fann-wsp-swarm — bounded Node.js proposal workers

Schedule fixed-operation proposal jobs in actual Node.js workers with separate WASM instances. The coordinator receives candidate data and diagnostic metrics; workers never receive promotion authority.

**Topics:** Node.js worker threads · parallel weight proposals · work budgets · cancellation.

```sh
npm install ruv-fann-wsp-swarm@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruv-fann-wsp-swarm) · [Core 0.2.0 entrypoint: `ruv-fann/swarm`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/swarm` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

A bounded scheduler for actual Node `worker_threads`. Each worker executes a shipped, fixed operation with its own JavaScript/WASM runtime. No provider call, generated JavaScript, shell command, model publication or checkpoint promotion is exposed. Worker threads are **not a security sandbox**.

Integration sketch: supply a dense `model`, development `dataset`, acceptance `policy` and explicit parameter `locations`.

```js
import { runProposalSwarm, estimateSwarmWork } from 'ruv-fann-wsp-swarm';

const jobs = [false, true].map((simd, index) => ({
  id: `candidate-${index}`, kind: 'search', model, dataset, policy,
  searchOptions: { locations, maxEvaluations: 8, maxPasses: 2, simd }
}));
const reservedWork = estimateSwarmWork(jobs);
const results = await runProposalSwarm(jobs, { concurrency: 2, timeoutMs: 5000, maxTotalWork: reservedWork });
```

`runProposalSwarm(jobs, options?)` returns one result per input job in the same order. Search jobs run `ruv-fann-wsp-search@0.1.0` with real scalar/SIMD WASM. `{id,kind:'patch',patch}` jobs transport a bounded candidate without evaluating it. A successful result is `{id,status:'ok',patch,workerThreadId,search?}`; search summaries contain `evaluations`, `inferencePasses`, `reservedWork` and `backend`. A search with no strictly improving patch returns `patch: null`. Runtime failure, timeout and cancellation return statuses `error`, `timeout` and `cancelled`; errors contain at most 512 characters. Parent-process monotonic `startedAtMs`/`finishedAtMs` timestamps describe started workers and include awaited cleanup.

Every caller-owned data value is snapshotted before the first asynchronous yield. Unknown operations/options, objective callbacks, functions, accessors, sparse arrays, cycles and executable fields are rejected. Hard limits: 64 jobs, 4 concurrent workers, 10 MiB aggregate JSON input, 60-second per-worker deadline, 10 million parameter-row work per search and 100 million across the swarm. `maxTotalWork` may tighten the total bound. Search work uses the same conservative baseline/screens/final-evaluation reservation as the search library; patch transport reserves zero inference work, so the coordinator must separately budget its authoritative evaluation. `estimateSwarmWork(jobs)` validates the same input contract and returns this reservation without starting workers. Data/model limits from the base WSP SDK also apply.

Abort signals prevent queued jobs from starting and terminate active workers. Deadline measurement includes worker startup. The scheduler awaits termination after successful messages, errors, exits, timeouts and aborts, and ignores late messages. Operations within one job are synchronous WASM, so cancellation terminates its worker rather than depending on cooperative polling.

All results remain untrusted proposals. An independent coordinator must apply patches to the expected base, re-evaluate trusted selection/held-out data and invariants, then use its own durable proposal authority and expected revision. Worker metadata and receipts never grant promotion authority. Runtime resource limits are supplemental V8 limits; they are not hard limits on all native/WASM memory or a boundary for hostile library code.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
