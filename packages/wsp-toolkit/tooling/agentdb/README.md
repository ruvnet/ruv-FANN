# AgentDB integration runtime

This private fixture pins real `agentdb@3.10.0` and omits its optional native, model and telemetry dependencies. It uses AgentDB's published `WeightPrograms` and `db-fallback` modules, including the actual `sql.js` persistence path. No code is vendored or substituted. AgentDB retains its declared `ruv-fann-wsp@0.1.0` internally; the toolkit coordinator independently evaluates with its exact `ruv-fann-wsp@0.2.0` API.

```sh
npm ci --prefix tooling/agentdb --ignore-scripts --omit=optional --no-audit --no-fund
npm audit --prefix tooling/agentdb
```

`openDemoRegistry(filename, registryFactory?)` opens a serialized registry and flushes each acknowledged mutation. The optional factory is a trusted adapter constructor for installed-package consumer tests; it defaults to this workspace registry. It never installs dependencies or changes process-wide environment. Tests fail with a setup instruction when the fixture is absent. The demo path uses one registry owner for a file; independent `sql.js` file writers are detected by AgentDB's optimistic persistence check, not coordinated as a distributed database.

A failed flush marks the adapter poisoned and blocks further state reads or mutations. Cleanup never rewrites `sql.js` files because all acknowledged mutations were already flushed; after poison it discards unflushed memory instead of retrying save automatically. A failed filesystem acknowledgement can still have reached disk; reopen and inspect rather than assuming the mutation was absent or retrying it automatically. This fixture supplies no server or remote connection.
