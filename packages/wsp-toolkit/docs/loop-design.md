# Bounded autonomous loop and swarm

## Outcome and executable acceptance

Extend the existing toolkit with a runnable local evolution loop: multiple worker threads propose surgical weight changes, an independent coordinator re-evaluates selection and held-out suites plus output invariants, and AgentDB owns durable proposal IDs, compare-and-swap revisions and rollback. The acceptance demo repairs XOR, rejects a tempting local-only candidate using an independent gate, preserves rejected/stale revisions, stops within explicit budgets, reopens durable state and rolls back exactly. This is a present-day bounded implementation; no physical neuromorphic hardware, universal alignment proof, public registry publication, provider call or financial action is in scope.

## Ownership

- Registry worker owns `packages/registry/**` and `tooling/agentdb/**`: AgentDB capability adapter, real fixture installer/lockfile and integration tests. Package name `ruv-fann-wsp-registry@0.1.0`.
- Swarm worker owns `packages/swarm/**`: fixed-operation worker threads and bounded scheduler. Package name `ruv-fann-wsp-swarm@0.1.0`.
- Invariants worker owns `packages/invariants/**`: real WASM finite-domain output constraints. Package name `ruv-fann-wsp-invariants@0.1.0`.
- Coordinator owns `packages/loop/**`, root manifests/lockfiles, scripts, E2E, examples, docs and integration. Package name `ruv-fann-wsp-loop@0.1.0`.

No overlapping edits. All libraries ESM with public src/index.js and src/index.d.ts, README, MIT LICENSE and test/*.test.mjs. Runtime dependency WSP exact 0.2.0; sibling packages 0.1.0. No worker edits toolkit root lockfile. Existing uncommitted toolkit work is retained.

## Shared interfaces

### Registry

`createAgentDBRegistry(programs, {flush?, close?}?)` receives a real AgentDB WeightPrograms capability (AgentDB 3.10.0); wrapper returns promise methods `create(id,model)`, `get(id)`, `list(limit?)`, `propose(id,patch,dataset,policy)`, `promote(id,proposalId,expectedRevision)`, `rollback(id,targetRevision,expectedRevision)`, `close()`. Serialize mutations across awaits. Snapshot inputs before queueing; do not accept caller-authored receipts as promotion authority. Flush durable mutations before acknowledging. Failed flush poisons mutations until reopen; report failure. Accessors/getters/unknown arguments/bounds must fail safely. AgentDB is an optional peer, not auto-installed by the library.

`tooling/agentdb/package.json` pins agentdb 3.10.0, private, with its own lockfile and `.npmrc` omitting optional dependencies to avoid unrelated native/model packages. `tooling/agentdb/runtime.mjs` exports `openDemoRegistry(filename)` using the real package's WeightPrograms and db-fallback, with explicit flush/close. Root setup command will run npm ci in this fixture; tests must fail with actionable error if absent, never silently skip. Node 20.9+ runtime; do not mutate process-wide environment.

### Swarm

`runProposalSwarm(jobs, {concurrency?, timeoutMs?, signal?}?)` runs fixed trusted library code in actual Node worker_threads. Each job has `{id, kind:'search', model, dataset, policy, searchOptions}` or `{id, kind:'patch', patch}`. Search uses ruv-fann-wsp-search with real WASM; patch simply transports an untrusted candidate. Result shape per input order: `{id,status:'ok'|'error'|'timeout'|'cancelled', patch?:WeightPatch|null, workerThreadId?:number, search?:{evaluations,inferencePasses,reservedWork,backend}, error?:string}`. SearchOptions are data-only: objective callbacks disallowed in workers. Hard caps <=4 concurrent workers, <=64 jobs, <=10MiB aggregate JSON, <=60000ms per job, aggregate candidate-work cap. Snapshot all inputs before awaits, validate IDs and options, bound error text, terminate and await every worker on all paths. Worker threads are not a security sandbox. Unknown operations/code/shell/functions refused. Coordinator distrusts all worker results and recomputes authoritative evidence.

### Invariants

`checkModelInvariants(model, suite, {simd?}?)` uses real WASM and returns `{schema:'ruv-fann.invariant-check/1', modelHash, suiteHash, accepted, checks, violations, backend, work}`. Suite shape `{schema:'ruv-fann.invariants/1', cases:[{name,inputs,rows,minimum?:number[],maximum?:number[],expected?:number[],tolerance?:number}]}`; bounds are per-output; expected is flattened row-major. At least one constraint per case. Bounds/exact shapes, finite numbers, source ownership, cumulative work <=1e7 and all rows <=1024. `createBooleanInvariant({inputs,outputs,targets,tolerance?})` emits a suite covering every binary input (1..6 input bits); returned case name `boolean-domain`. Finite-domain evidence is not a proof over all real inputs or cryptographic alignment. No arbitrary callbacks.

### Loop coordinator

Expose a bounded `runEvolutionLoop` driven by trusted module configuration and an injected registry. Each generation snapshots a module's active revision; dispatches immutable worker proposals; independently applies/checks each patch against trusted suites and invariants; ranks only comparable accepted candidates; re-proposes the selected candidate inside AgentDB; promotes using its internal ID and original expected revision. Worker receipts never authorize promotion. Stale state is rejected without overwriting newer state. Loss must improve by a configured threshold. Max rounds, proposals, aggregate work, workers, deadlines and AbortSignal bound the run; failures are visible and leave live checkpoints unchanged. No provider requests or automatic public publication. Durable models/history can resume from AgentDB; counters describe each invocation, not a global metering service.

## Success and failure walks

Success: broken XOR r1 → two search workers → two-weight candidate → held-out/exhaustive constraints → AgentDB proposal → CAS promote r2 → reopen r2 → rollback to r1 content as r3.

Failure: a candidate fitting the development slice fails held-out/output constraints → record rejection in loop report → no promote. A revision changed by another coordinator before promotion causes a stale result, never an overwrite. Worker timeout/abort terminates the worker and prevents later results from committing. A durability failure cannot be reported as a successful promotion.

## Architecture decisions and limitations

Use fixed-operation Node workers for real concurrent execution and separate WASM instances; use platform subagents only to build/review this implementation. Keep AgentDB's large optional dependency tree in a separate pinned integration fixture; the registry library accepts a caller-owned real controller. No mocked database success is called persistence evidence. Hosted runtime connectors are currently unavailable; no remote execution is inferred from coordination records. Universal verification, learned semantic routing, code-generation providers, npm publication and physical actuation remain explicit extensions requiring their own contracts.
