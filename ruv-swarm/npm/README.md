# ruv-swarm for Node.js

A JavaScript API and command-line tool for coordinating agents, storing task state in SQLite, and calling the repository's WebAssembly modules. It includes a stdio MCP server for clients such as Claude Code.

[![npm](https://img.shields.io/npm/v/ruv-swarm)](https://www.npmjs.com/package/ruv-swarm)

![The repository's neural-network, forecasting, and swarm packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)

## Install

Use Node.js **22.13 or newer**. SQLite uses a native addon; npm may need a C++ build toolchain when a matching prebuilt addon is unavailable. The maintained development tools also require this minimum.

```sh
npm install ruv-swarm
npx ruv-swarm --help
npx ruv-swarm --version
```

Pin the version you have tested in applications and MCP client configurations. The npm package and Rust crates have separate version numbers.

## Start the MCP server

```sh
npx ruv-swarm mcp start
```

Configure your client to launch that command using stdio. For example:

```json
{
  "mcpServers": {
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start"]
    }
  }
}
```

Use `ruv-swarm mcp tools` to list tools. The WebSocket server is not implemented in this distribution.

The server exposes 17 supported tools for persisted local metadata, dense inference, statistical forecasts, memory observations, and measured local benchmarks. Missing tasks return errors. Training, autonomous DAA execution, learned-pattern analysis, and external model execution are unavailable and return explicit errors.

The server accepts newline-delimited JSON-RPC. Each frame is limited to 1 MiB of UTF-8 data, and each process accepts up to 32 pending requests. Outgoing frames have the same limit; slow clients have a bounded 4 MiB reply queue. Closing stdin drains pending replies before shutdown. Oversized frames are discarded through the next newline so subsequent requests can still be handled. Local metadata is limited to 64 swarms, 1,000 agents, and 10,000 tasks per database; exceeding a limit fails explicitly. Monitoring records at most 100 aggregate snapshots per call. Keep access to the process and its SQLite files limited to the users who should control those agents.

## Run commands

```sh
npx ruv-swarm init mesh 5
npx ruv-swarm spawn researcher "Research agent"
npx ruv-swarm status
```

`ruv-swarm init --claude` writes supported tool documentation and a project `.mcp.json` registration. It preserves existing MCP servers, permissions and hooks. Review those files before adding hooks to your development environment.

The coordination layer tracks work and invokes available modules. Creating an agent does not, by itself, establish that an external model has performed a task. Features depend on the compiled WASM artifact and the APIs it exports.

## Invoke Claude Code

```sh
npx ruv-swarm claude-invoke "Summarize this project's architecture"
```

Claude's normal permission checks remain enabled by default. Prompts are passed as literal process arguments, so quotes and shell syntax are ordinary text. To explicitly disable permission checks for an invocation, use the `--dangerously-skip-permissions` flag. The internal integration helper preserves its `secure: false` opt-in; the supported package entry point does not export that helper.

Use `claude-invoke -- "prompt"` when a prompt starts with a flag. Generated bash helpers also separate options from prompt text. The former Windows batch prompt helper fails explicitly; use the Node CLI to preserve literal arguments.

## Use the JavaScript API

The package entry point exports `RuvSwarm`. It loads the bundled WASM module and wraps its local assignment APIs:

```js
import { RuvSwarm } from 'ruv-swarm';

const runtime = await RuvSwarm.initialize();
const swarm = await runtime.createSwarm({
  name: 'example', topology: 'mesh', maxAgents: 3
});
console.log(swarm.getStatus());
swarm.free();
```

`swarm.spawn({type, name, capabilities})` records an agent. `swarm.orchestrate({description, requiredCapabilities})` assigns matching idle agents and returns `assigned` or `unassigned`. It does not execute an external model. `agent.execute()` rejects until an external executor exists; `agent.reset()` returns an assigned agent to idle. Release the swarm with `free()` when finished.

The runtime also exposes dense neural inference and three one-step statistical forecasts:

```js
const network = runtime.createNeuralNetwork([2, 1], 'Linear');
network.set_weights(Float64Array.of(2, -1));
console.log(network.run(Float64Array.of(3, 4))); // Float64Array [2]
network.free();

const forecast = runtime.createForecastingModel('linear'); // also 'mean' or 'naive'
console.log(forecast.predict(Float64Array.of(1, 3, 5))); // Float64Array [7]
forecast.free();
```

The same operations are available through the CLI:

```sh
ruv-swarm neural infer --layers 2,1 --weights 2,-1 --input 3,4
ruv-swarm neural forecast --model linear --input 1,3,5
ruv-swarm benchmark run --iterations 1000
```

Neural network dimensions and inputs must have the exact expected shape and finite numbers. Forecast names such as LSTM and Transformer are unsupported. See [`src/index.d.ts`](https://github.com/ruvnet/ruv-FANN/blob/main/ruv-swarm/npm/src/index.d.ts) for the facade's types. Historical prototype files are outside the public export map; applications should use this facade.

## Reuse prepared inference

After setting weights, prepare an independent snapshot. It keeps working after
the source network is edited or freed:

```js
const network = runtime.createNeuralNetwork([2, 1], 'Linear');
network.set_weights(Float64Array.of(2, -1));
const prepared = network.prepare();
network.free();
try {
  const input = Float64Array.of(3, 4);
  const output = new Float64Array(1);
  prepared.run_into(input, output);
  console.log(output); // Float64Array [2]
  console.log(prepared.input_size, prepared.output_size, prepared.estimated_bytes);
} finally {
  prepared.free();
}
```

`run_into(input, output)` reuses Rust numeric storage and writes to the supplied
output array. `run(input)` returns a new output array. Both require real
`Float64Array` inputs with the exact dimensions and finite values. Shared buffers
and views into this module's WASM memory are rejected. Invalid inputs or numeric
overflow leave the caller's output unchanged. The binding still copies values
and creates temporary JavaScript views; buffer reuse does not eliminate the cost
of crossing into WASM.

To run a trained Rust model, export `Network<f64>` with
`serde_json::to_string(&network)` and pass that string to
`runtime.createPreparedNetwork(networkJson)`. Native sparse and reordered
connections and all 21 native activation variants are supported. Browser callers
use `WasmPreparedNetwork.from_network_json(networkJson)` on the generated class.
Inference evaluates in Float64. Import JSON is limited to 4 MiB of UTF-8.
Preparation permits an estimated 64 MiB for packed parameters, workspace and
input/output staging. This estimate excludes the original model, temporary
conversion graph and allocator bookkeeping; it is not a peak-memory guarantee.

## Calibrate historical forecast intervals

Supply genuine past predictions and observed outcomes. This small artificial
example has nine past errors, one model, and one forecast step. All timestamps
are safe-integer Unix milliseconds:

```js
const calibration = runtime.createConformalCalibrator({
  as_of: 100,
  horizon: 1,
  coverages: [0.9],
  records: Array.from({length: 9}, (_, i) => ({
    series: 'example', model: 'baseline',
    cutoff: i * 10, timestamp: i * 10 + 1,
    actual: i + 1, prediction: 0,
  })),
});
try {
  const intervals = calibration.apply({records: [{
    series: 'example', model: 'baseline',
    origin: 100, timestamp: 101, prediction: 20,
  }]});
  console.log(intervals[0].lower, intervals[0].upper); // 11, 29
  console.log(calibration.groups()); // Sample counts and fitted half-widths
} finally {
  calibration.free();
}
```

Errors are calibrated separately for each series, model and ordinal horizon.
Every series/cutoff window must contain the same model set and exactly `horizon`
distinct target timestamps per model, with aligned observations. Calibration
requires `cutoff < timestamp <= as_of`; application requires
`as_of <= origin < timestamp` and one origin per series. Missing groups,
insufficient calibration samples, duplicate targets, non-finite values, lossy
integer literals and inconsistent windows are errors. The native DataFrame API
and this WASM API use the same Rust calibration engine.

The corrected rank is `ceil((n + 1) * coverage)`. Coverage uses fractions between
zero and one. Requested coverage does not guarantee coverage for dependent time
series or after a distribution shift; it is not simultaneous trajectory
coverage. Refit calibration when the model, preprocessing or error regime changes.

| WASM limit | Maximum |
| --- | ---: |
| Input JSON per fit/application | 4 MiB UTF-8 |
| Output JSON | 8 MiB, with a conservative size check before interval allocation |
| Input records per call | 10,000 |
| Series/model/horizon groups | 1,000 |
| Models / coverage levels | 128 / 16 |
| Series or model name | 256 UTF-8 bytes |
| Interval output rows | 100,000 |

The conservative output check can reject a request whose final JSON would be
smaller than 8 MiB. Dates and BigInts are not accepted as timestamps; convert
dates with `date.getTime()`. Numeric predictions use finite IEEE-754 Float64.

## Use the generated bindings in a browser

Serve the package over HTTP and import `wasm/ruv_swarm_wasm.js` as an ES module.
Its default initializer fetches the ordinary WASM binary. The generated classes
include `WasmNeuralNetwork`, `WasmPreparedNetwork` and `WasmConformalCalibrator`:

```js
import init, {WasmConformalCalibrator} from './wasm/ruv_swarm_wasm.js';
await init();
// const calibration = new WasmConformalCalibrator(JSON.stringify(config));
// const intervals = JSON.parse(calibration.apply(JSON.stringify({records})));
// calibration.free();
```

The [complete browser example](https://github.com/ruvnet/ruv-FANN/blob/main/ruv-swarm/npm/examples/wasm-prepared-conformal.html)
runs inference and intervals without a Node.js server API or a Rust installation.
From a repository checkout, run `python3 -m http.server --directory ruv-swarm/npm 8000`
and open `http://localhost:8000/examples/wasm-prepared-conformal.html`.

The Node facade can select the SIMD binary with
`RuvSwarm.initialize({useSIMD: true})`. Both variants expose the same API.
The existing native CPU benchmark measures a different execution path; measure
your WASM workload with its JavaScript checks, copies and allocation costs.

## Develop and validate

From `ruv-swarm/npm`:

```sh
npm ci
npm test
npm run test:security
npm run test:all
npm audit --audit-level=low
npm pack --dry-run  # Requires fresh WASM build evidence
npm run validate:package  # Also works in the installed package
npm run test:installation  # Packs, locks, clean-installs and tests the current candidate
```

The repository's `scripts/verify-sota-wasm-browser.cjs` additionally loads both
shipped binaries in real Chromium, checks native dense/sparse model imports,
repeated buffer reuse, rejection behavior and calibrated interval results.
It requires Playwright and Chromium; set `CHROMIUM_BIN` when using a system
Chromium executable. The browser CI gate runs this script.

The default test command runs real process, SQL, integrity, framing, inference, forecasting, and assignment regression tests, then the older mock-based API examples. The mock examples demonstrate API shape; they do not verify inference or forecasting accuracy.

Tooling tests exercise the actual Jest preset merge and Rollup resolver, including the prototype-poisoning payload fixed by the maintained `@stackline/deepmerge` replacement. Full dependency audits currently pass with zero findings. `quality:check` runs maintained-code lint, both test groups, the full audit, and source-artifact validation. Historical lint commands cover additional prototype files and are separate from that supported release gate.

`npm run test:all` additionally starts the real Rust WebSocket MCP server, so it requires Rust and working Rust workspace dependencies. Its other suites include an independent SQLite fixture, a JavaScript neural simulation, and mock API examples. See the [npm review](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-npm.md) for exact results and remaining limits.

Publishing requires a fresh WASM build receipt matching the current source files, lockfiles, build settings, and every shipped binary and generated binding. A valid legacy binary alone is insufficient. The release includes core and SIMD. Optional historical forecasting and FANN binaries are excluded because the loader does not support them. `validate:package` checks the packaged artifacts without requiring a Rust checkout.

Rebuilding WASM also requires Rust, the `wasm32-unknown-unknown` target, `wasm-pack`, and working Rust workspace dependencies:

```sh
cargo install wasm-pack --version 0.15.1 --locked
rustup target add wasm32-unknown-unknown
npm run build
```

npm installation does not download a separate wasm-pack executable. This keeps installation independent of GitHub access and Node's optional environment-proxy support. Prebuilt package consumers do not need Rust or wasm-pack.

The build script invokes tools with argument arrays, including paths with spaces. It does not evaluate those paths in a shell.

## Performance and current limits

Measure the workload and artifact you intend to deploy. This README does not claim a SWE-bench score, a token reduction, or a speedup against an LLM provider: the repository does not supply reproducible evidence for those comparisons.

The current security work adds bounded input framing, shell-free CLI invocations, allowlisted SQL update fields, and verification of the same WASM bytes returned to callers. Dependency audits remain a separate check; they do not prove application code is secure.

## Migrating from 1.x

Version 2.0 narrows the public API to working capabilities. Use `RuvSwarm.initialize()` and `createSwarm()` for local assignment, `createNeuralNetwork()` for inference, and `createForecastingModel()` for supported forecasts. Untyped prototype exports such as `NeuralAgentFactory`, `NeuralNetwork`, `Swarm`, `Agent`, and `Task` are no longer exported. Internal source paths are not public package entry points.

MCP clients should refresh `tools/list`: unsupported autonomous, training and invented telemetry tools have been removed. Assignment replies return actual task IDs, `assigned` or `unassigned`, and `assignedAgents`; they contain no estimated completion or fabricated model accuracy. SQLite records survive process restarts. State defaults to `.ruv-swarm/state.sqlite` in the working project; `RUV_SWARM_DB_PATH` selects an explicit location. To use an older database, back it up and point that variable at the existing file.

The npm package is 2.0.0; its bundled core Rust WASM crate has its own version, 1.0.6. See the [changelog](./CHANGELOG.md) for the compatibility changes.

## License

MIT OR Apache-2.0. The package includes the swarm workspace's `LICENSE-MIT` and `LICENSE-APACHE` texts.
