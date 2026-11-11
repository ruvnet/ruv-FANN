# ruv-fann-wsp-registry — AgentDB revisions and durability acknowledgement

Adapt an application-owned AgentDB WeightPrograms controller into a bounded serialized API. Explicit persistence callbacks distinguish an acknowledged durable mutation from a memory-only update.

**Topics:** AgentDB weight programs · revision CAS · durable acknowledgements · rollback history.

```sh
npm install ruv-fann-wsp-registry@0.1.0
```

[Package on npm](https://www.npmjs.com/package/ruv-fann-wsp-registry) · [Core 0.2.0 entrypoint: `ruv-fann/registry`](https://github.com/ruvnet/ruv-FANN/tree/main/packages/core#capability-entrypoints) · [Interactive ruv-fann field guide](https://ruvnet.github.io/ruv-FANN/) · [Toolkit API and release guide](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp-toolkit#readme)

Choose this standalone package for its declared dependency footprint, or use `ruv-fann/registry` from the complete `ruv-fann@0.2.0` facade. Package version: **0.1.0**. [Umbrella release and evidence](https://github.com/ruvnet/ruv-FANN/releases/tag/ruv-fann-toolkit-v0.2.0).

Adapt a caller-owned real AgentDB `WeightPrograms` controller into a serialized promise API with explicit persistence acknowledgement. ESM, Node 20.9+. AgentDB 3.10.0 is an optional peer; the library never installs it or opens a database automatically.

```js
import { createDatabase } from 'agentdb/db-fallback';
import { WeightPrograms } from 'agentdb/controllers/WeightPrograms';
import { createAgentDBRegistry } from 'ruv-fann-wsp-registry';
const db = await createDatabase('weights.db');
const programs = new WeightPrograms(db);
const registry = createAgentDBRegistry(programs, {
  flush: () => db.save?.(),
  close: () => {
    programs.dispose();
    // Every acknowledged write was already flushed; avoid close() saving again.
    if (typeof db.save === 'function') db.save = () => {};
    db.close();
  }
});
```

Promise methods: `create(id,model)`, `get(id)`, `list(limit?)`, `propose(id,patch,dataset,policy)`, `promote(id,proposalId,expectedRevision)`, `rollback(id,targetRevision,expectedRevision)`, and `close()`. `status()` reports queue/close/poison state and whether a flush hook is configured. AgentDB revision conflicts are reported as `RegistryStaleError` (`WSP_REGISTRY_STALE`). Promotion only accepts an internal proposal ID; a detached evaluation object is never authority. AgentDB supplies real WASM evaluation, revision CAS, retained history and monotonic rollback.

All operations share one queue, including reads, so acknowledged reads follow preceding flushes. Each input is copied synchronously before queueing without invoking getters, iterators or toJSON. The hard queue limit is 16 operations and can only be tightened with `maxPending`. Each copied JSON argument/result is bounded to 2 MiB, 262,144 tree nodes and depth 16. Controller methods and callback capabilities are trusted local code, not a sandbox. Do not mutate the same controller outside its adapter.

Writes resolve only after the optional flush hook completes. Without `flush`, status explicitly reports `memory-only`; this does not establish durability. Failed flushes reject with `RegistryPersistenceError` (`WSP_REGISTRY_PERSISTENCE`), poison the adapter and block further reads/mutations until close/reopen. The `options.close(state)` cleanup hook receives `{poisoned}` and should discard unflushed memory rather than silently retry persistence. A failed acknowledgement can have reached disk; inspect reopened state before deciding whether to retry. Cleanup is awaited once after all queued work settles.

The toolkit's separate `tooling/agentdb` fixture installs the real controller and exercises reopen, stale CAS, rejection, snapshots, persistence failure and cleanup. File-backed sql.js usage assumes one registry owner per file; this adapter does not add distributed locking. Hashes and accepted evaluations are neither signatures nor generalization guarantees. Application owners still provide independent suites, invariants and promotion policy.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
