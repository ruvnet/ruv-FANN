---
name: swarm-mcp
description: "Integrate or test ruv-swarm Rust scheduling and Node stdio MCP, SQLite records, dense WASM inference, statistical forecasts, and capability assignment. Use for agent-tool integration, MCP framing, persistence, or distinguishing task assignment from actual execution."
---

# Integrate the implemented swarm and MCP surfaces

First choose Rust libraries or the Node distribution. Read `ruv-swarm/README.md` and `ruv-swarm/npm/README.md` in the selected checkout. Rust scheduling/transport and Node stdio MCP are separate paths. This plugin does not automatically register a server or change permissions.

## Execution contract

`ruv-swarm` records and assigns work. In the Node facade, `spawn()` records an agent and `orchestrate()` selects idle agents with matching capabilities. Replies have real IDs, `assigned`/`unassigned` and assigned agents. They do not establish external model execution. `agent.execute()` rejects until an executor exists. A Rust `FunctionAgent` runs the actual async handler supplied by the application.

The Ruflo CLI, private Ruflo Ai Team records and ruOS execution are other products. Keep their evidence distinct: a ledger task is complete only after checking its actual output and tests. Do not dispatch private task content to public federation channels.

## Test a local API consumer

Use Node 22.13+ and the real installed `ruv-swarm` package or selected checkout. An installed archive consumer can use:

```js
import assert from 'node:assert/strict';
import { RuvSwarm } from 'ruv-swarm';

const runtime = await RuvSwarm.initialize();
const network = runtime.createNeuralNetwork([2, 1], 'Linear');
try {
  network.set_weights(Float64Array.of(2, -1));
  assert.equal(network.run(Float64Array.of(3, 4))[0], 2);
} finally {
  network.free();
}
const forecast = runtime.createForecastingModel('linear');
try {
  assert.equal(forecast.predict(Float64Array.of(1, 3, 5))[0], 7);
} finally {
  forecast.free();
}
```

This dense WASM format has no bias weight in that two-input example; do not copy the root Rust network's bias layout into it. Keep exact dimension/weight lengths and finite numeric inputs. Forecast models here are one-step `linear`, `mean` or `naive`, distinct from the neuro-divergent trained RNN.

## Exercise the real stdio server

From the selected checkout, with its Node dependencies available:

```sh
node ruv-swarm/npm/bin/ruv-swarm-secure.js mcp tools
node ruv-swarm/npm/bin/ruv-swarm-secure.js neural infer --layers 2,1 --weights 2,-1 --input 3,4
node ruv-swarm/npm/bin/ruv-swarm-secure.js neural forecast --model linear --input 1,3,5
```

Start MCP with the same executable's `mcp start`. Use a task-owned working directory and an explicit `RUV_SWARM_DB_PATH` when the test should not alter project state. Preserve existing databases and client configuration. The default DB is `.ruv-swarm/state.sqlite` relative to the working project.

For protocol testing, use newline-delimited JSON-RPC, complete the MCP initialize handshake and inspect `tools/list` from the running candidate. `neural_infer` with layers `[2,1]`, weights `[2,-1]`, input `[3,4]`, activation `Linear` should return output `[2]`. `forecast_predict` with model `linear` and `[1,3,5]` should return `[7]`. Read actual returned IDs for metadata follow-up calls.

Test a rejected unknown method/tool/key, missing task, malformed or oversized frame followed by a valid request, EOF draining, and state across reopening. Assert error and recovery behavior, not just process startup. The supported surface initially has 17 tools; read live discovery rather than assuming a stale count after future changes.

## Safety and resource boundaries

The Node database path requires the package-owned patched SQLite source build.
Use `npm run build:sqlite` and `node scripts/sqlite-build.mjs --verify` from the
swarm npm package. Normal installation builds the SHA-pinned SQLite 3.54.1 with
Python/C++20; the runtime validates its source ID, binary hash and Node ABI before
opening user data. Missing/stale binaries fail; a default wrapper prebuilt is
not an accepted replacement. Native Rust persistence uses Rust 1.89 and the same
patched SQLite, while the minimal neural WASM engine retains Rust 1.88.

- Input/output frames are capped at 1 MiB, with 32 pending requests and a 4 MiB reply queue. Metadata caps are 64 swarms, 1,000 agents and 10,000 tasks per database.
- SQL values must use parameters; update column names need a separate allowlist. Unknown keys cannot become SQL identifiers.
- Process invocations must preserve literal arguments with `execFile`/spawn arrays. Prompts beginning with flags need the supported `--` separator. Preserve normal Claude permission checks.
- Hash and consume the same WASM bytes. A missing or corrupt module is an error, never permission to substitute another package's binary or fake outputs.
- MCP neural-training tools, autonomous DAA, external model execution and invented telemetry are unsupported. The separate JavaScript `createTrainableNetwork` API runs dense MSE/autograd, SGD and AdamW; follow the core-inference skill for that contract. WebSocket serving is not implemented in the Node distribution.

## Verify changes

Run Node checks from `ruv-swarm/npm/`:

```sh
npm test
npm run test:security
npm run test:tooling
npm audit --audit-level=low
npm run validate:package
```

`npm run test:all` additionally requires the real Rust WebSocket server. `npm run test:installation` packs, clean-installs and tests the current candidate; prepack requires fresh genuine WASM provenance. Do not rewrite a receipt to cover changed source.

For Rust changes, plan the helper's `swarm` profile, then run appropriate crate tests or the workspace's all-feature/all-target gate. Browser transport and real IndexedDB require their separate WASM/runtime tests; native compilation is not browser execution.

## Completion and diagnosis

Record actual tool discovery, deterministic inference/forecast results, state persistence and protocol rejection/recovery. For addon errors, inspect Node ABI and the SQLite build prerequisite. For assignment without results, inspect the application executor rather than recording fabricated completion. For stale artifact errors, rebuild from frozen source with the documented pinned tools.

Read [protocol and ownership details](references/protocol.md) before changing transport, initialization or package exports.
