# Protocol, runtime and packaging reference

Read these checkout sources:

| Concern | Source |
| --- | --- |
| Actual 17 Node tool definitions | `ruv-swarm/npm/src/mcp-tools-enhanced.js` |
| MCP framing, EOF and backpressure | `ruv-swarm/npm/src/stdio-rpc.js`, `ruv-swarm/npm/src/bounded-lines.js` |
| Deterministic inference and forecasts | `ruv-swarm/npm/src/neural-runtime.js` |
| Public types and exports | `ruv-swarm/npm/src/index.d.ts`, `ruv-swarm/npm/package.json` |
| SQLite persistence and safe updates | `ruv-swarm/npm/src/persistence.js`, `ruv-swarm/npm/src/persistence-updates.js` |
| Actual WASM provenance validation | `ruv-swarm/npm/scripts/wasm-provenance.js` |
| Native Rust handler example | `ruv-swarm/crates/ruv-swarm-agents/README.md` |
| Separate Rust MCP scope | `ruv-swarm/crates/ruv-swarm-mcp/README.md` |

## Supported Node tools

Metadata: `swarm_init`, `swarm_status`, `swarm_monitor`, `agent_spawn`, `agent_list`, `agent_metrics`, `agent_update_status`, `task_orchestrate`, `task_status`, `task_results`.

Compute/observation: `neural_status`, `neural_infer`, `forecast_predict`, `benchmark_run`, `features_detect`, `memory_usage`, `persistence_stats`.

`task_orchestrate` is a supported tool name in this distribution. It does not imply that the separate Ruflo CLI has a `task orchestrate` command. Discover the actual host CLI help before suggesting coordination commands. If optional Ruflo CLI use is necessary for the authorized task, the previously validated line is `@claude-flow/cli@3.25.6`; do not silently upgrade it.

## Ownership

Release every generated WASM network/model/swarm with its `free()` method when the application no longer uses it. A freed object is not a reusable handle. An agent assignment and SQLite record must be updated consistently; test rollback if persistence fails. Do not reopen and overwrite a user's DB during a smoke test.

Native shared-memory transport is an in-process ring prototype, not cross-process shared storage. Real IndexedDB belongs to the separate persistence crate. The unified WASM facade does not automatically add that backend.

## Installation and publication

The plugin needs no `.mcp.json` changes. When the user separately requests server configuration, merge the explicit server registration and preserve custom servers, hooks and permissions; reject malformed JSON without replacing it. Configure an already installed, version-pinned executable rather than using an unbounded latest download.

The package's WASM source receipt covers actual source, manifests, locks, generated bindings/binaries and build options. Verify the same packed bytes in a clean consumer. A successful local API test does not establish Windows/macOS behavior or provider execution.
