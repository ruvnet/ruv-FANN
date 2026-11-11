# ruv-fann-wsp-registry

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
